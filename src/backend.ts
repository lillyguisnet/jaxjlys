// Shared interfaces and code for the low-level backend API.
//
// Think of each backend as a _connector_ to a specific hardware or software
// implementation of the array API.
//
// Backends do not share any of the built-in operational semantics of the
// library. This is a private API. You must allocate and free buffers manually,
// and dispatch happens on the level of each shader. Buffers are untyped.
//
// The "cpu" backend is very slow and used for debugging. Prefer "wasm".

import { AluOp, DType, Kernel } from "./alu";
import { CpuBackend } from "./backend/cpu";
import { WasmBackend } from "./backend/wasm";
import type { WebGPUBackend } from "./backend/webgpu";
import { Routine, Routines } from "./routine";

export type Device = "cpu" | "wasm" | "webgpu" | "webgl";
export const devices: Device[] = ["cpu", "wasm", "webgpu", "webgl"];

const initializedBackends = new Map<Device, Backend>();

// Default backends, initialized at startup.
initializedBackends.set("cpu", new CpuBackend());
if (typeof WebAssembly !== "undefined") {
  initializedBackends.set("wasm", new WasmBackend());
}

let defaultBackend: Device = initializedBackends.has("wasm") ? "wasm" : "cpu";

/** Configure the default device for arrays. */
export function defaultDevice(device?: Device): Device {
  if (device !== undefined) {
    if (initializedBackends.has(device)) {
      defaultBackend = device;
    } else {
      throw new Error(`Backend not initialized: ${device}`);
    }
  }
  return defaultBackend;
}

/**
 * Initialize `jax-js` library backends.
 *
 * By default, this will initialize all available backends. If one or more
 * backends is provided, only attempt to initialize those. Returns a list of
 * available backends.
 */
export async function init(...devicesToInit: Device[]): Promise<Device[]> {
  if (devicesToInit.length === 0) {
    devicesToInit = devices;
  }
  const promises: Promise<void>[] = [];
  for (const device of new Set(devicesToInit)) {
    if (!initializedBackends.has(device)) {
      promises.push(
        (async () => {
          const backend = await createBackend(device);
          if (backend) {
            initializedBackends.set(device, backend);
          }
        })(),
      );
    }
  }
  await Promise.all(promises);
  return Array.from(initializedBackends.keys());
}

// FORK PATCH (jaxjlys): public backend-reset API.
//
// Motivation: long-running workloads on the WASM backend accumulate state in
// the `WasmAllocator`'s bump pointer. Even after the signed-shift bug was
// fixed and the ceiling moved from ~2 GiB back to the spec-defined 4 GiB,
// workloads like SAM 2.1 Hiera-L that repeatedly allocate many-hundreds-of-MiB
// attention matrices can still climb toward that ceiling over the course of
// inference. Before this API existed, consumers had to reach into the module-
// private `initializedBackends` Map via `Map.prototype.has` monkey-patches to
// force `init()` to re-create a backend. That was fragile and non-portable.
//
// `resetBackend(device)` is the supported path: it tears down the current
// backend instance for `device` (releasing workers, GPU device, textures,
// etc.) and then re-runs `init(device)` to build a fresh one. The module-
// private state stays private; callers interact only with the public API.
//
// See FORK_NOTES.md §"Known issues we intend to patch" for context.
/**
 * Release and re-create the backend for a device.
 *
 * All external resources held by the existing backend instance are released
 * (WASM worker threads are terminated, GPU devices destroyed, textures and
 * compiled pipelines deleted, etc). A fresh backend of the same type is then
 * constructed via the normal `init()` path.
 *
 * **Warning: all `Array` objects, `Slot`s, `Executable`s, compiled kernels,
 * and JIT-cached programs that reference the old backend become invalid.**
 * The caller must copy any live data back to JS (via `array.data()`, or
 * equivalently `blockUntilReady()` + `read()` on the backend) _before_
 * calling `resetBackend()`, and re-upload it afterwards if needed.
 *
 * The `defaultDevice` setting is preserved across reset (it's stored as a
 * string, not a backend reference). If `device` was the default and the
 * re-creation fails (e.g. WebGPU adapter request fails after reset),
 * `defaultDevice()` will subsequently throw on next use.
 *
 * Returns the same shape as `init()`: the list of currently available
 * backends.
 */
export async function resetBackend(device: Device): Promise<Device[]> {
  const existing = initializedBackends.get(device);
  if (existing) {
    initializedBackends.delete(device);
    // FORK PATCH (jaxjlys): purge JIT cache entries that reference this backend
    // BEFORE destroying it, so no stale compiled program can be reused with
    // dangling backend slots. See purgeJitCacheForBackend() in frontend/jit.ts.
    const { purgeJitCacheForBackend } = await import("./frontend/jit");
    purgeJitCacheForBackend(existing);
    await existing.destroy();
  }
  return init(device);
}

/** Create a backend, if available. Internal function called by `init()`. */
async function createBackend(device: Device): Promise<Backend | null> {
  if (device === "cpu") {
    return new CpuBackend();
  } else if (device === "wasm") {
    if (typeof WebAssembly === "undefined") return null; // WebAssembly is not available.
    return new WasmBackend();
  } else if (device === "webgpu") {
    if (!navigator.gpu) return null; // WebGPU is not available.
    const adapter = await navigator.gpu.requestAdapter({
      powerPreference: "high-performance",
    });
    if (!adapter) return null;

    const { WebGPUBackend } = await import("./backend/webgpu");

    const importantLimits: Exclude<keyof GPUSupportedLimits, "__brand">[] = [
      "maxBufferSize",
      "maxComputeInvocationsPerWorkgroup",
      "maxComputeWorkgroupSizeX", // All of our workgroups use X or Y.
      "maxComputeWorkgroupSizeY",
      "maxComputeWorkgroupSizeZ",
      "maxComputeWorkgroupStorageSize",
      "maxComputeWorkgroupsPerDimension", // Grid size limited to 65535 due to AMD storage in u16.
      "maxStorageBufferBindingSize",
      "maxStorageBuffersPerShaderStage",
      "maxStorageTexturesPerShaderStage",
    ];

    const requestedFeatures: GPUFeatureName[] = [
      "shader-f16", // "enable f16;" feature support for f16 data type
      "timestamp-query", // Performance timing queries.
    ];

    try {
      const device = await adapter.requestDevice({
        requiredLimits: Object.fromEntries(
          importantLimits.map((limit) => [limit, adapter.limits[limit]]),
        ),
        requiredFeatures: requestedFeatures.filter((feature) =>
          adapter.features.has(feature),
        ),
      });
      return new WebGPUBackend(device);
    } catch (error) {
      // Browsers can throw a TypeError if features are not supported by the
      // adapter, or limits have not been set properly.
      console.error("Unexpected error requesting WebGPU device:", error);
      return null;
    }
  } else if (device === "webgl") {
    if (typeof WebGL2RenderingContext === "undefined") return null; // WebGL2 is not available.
    const canvas = new OffscreenCanvas(0, 0);
    const gl = canvas.getContext("webgl2", {
      alpha: false,
      antialias: false,
      premultipliedAlpha: false,
      preserveDrawingBuffer: false,
      depth: false,
      stencil: false,
      failIfMajorPerformanceCaveat: true,
    });
    if (!gl) return null;
    // Required extension for rendering to float textures.
    if (!gl.getExtension("EXT_color_buffer_float")) return null;
    const { WebGLBackend } = await import("./backend/webgl");
    return new WebGLBackend(gl);
  } else {
    device satisfies never;
    throw new Error(`Backend not found: ${device}`);
  }
}

/** Retrieve a backend that has been initialized. */
export function getBackend(device?: Device): Backend {
  device = device ?? defaultBackend;
  const backend = initializedBackends.get(device);
  if (!backend) {
    throw new Error(`${device} backend not ready, call init() first`);
  }
  return backend;
}

/** Unique identifier for an allocated, on-device buffer. */
export type Slot = number;

/** A device backend. */
export interface Backend {
  /** The name of the backend as a string. */
  readonly type: Device;

  /** Maximum number of arguments per dispatched kernel. */
  readonly maxArgs: number;

  /** Allocate a new slot with reference count 1. */
  malloc(size: number, initialData?: Uint8Array): Slot;

  /** Increment the reference count of the slot. */
  incRef(slot: Slot): void;

  /**
   * Decrement the reference count of the slot. If the reference count reaches
   * zero, it is freed. This should throw if the slot was already freed.
   */
  decRef(slot: Slot): void;

  /** Read a range of bytes from a buffer. */
  read(
    slot: Slot,
    start?: number,
    count?: number,
  ): Promise<Uint8Array<ArrayBuffer>>;

  /** Read a range of bytes from a buffer, blocking variant. */
  readSync(slot: Slot, start?: number, count?: number): Uint8Array<ArrayBuffer>;

  /** Prepare an expression to be executed later. */
  prepareKernel(kernel: Kernel): Promise<Executable>;

  /** Prepare an expression to be executed later, blocking variant. */
  prepareKernelSync(kernel: Kernel): Executable;

  /** Prepare an advanced routine to be executed later. */
  prepareRoutine(routine: Routine): Promise<Executable>;

  /** Prepare an advanced routine to be executed later, blocking variant. */
  prepareRoutineSync(routine: Routine): Executable;

  /**
   * Run a backend operation that was previously prepared.
   *
   * The operation may not run immediately, but operations are guaranteed to run
   * in the dispatch order. Also, `read()` will wait for all pending operations
   * on that slot to finish.
   */
  dispatch(exe: Executable, inputs: Slot[], outputs: Slot[]): void;

  // FORK PATCH (jaxjlys): mandatory teardown hook.
  //
  // Every backend holds external resources that the garbage collector cannot
  // reclaim on its own (web workers, GPU devices, textures, compiled pipelines).
  // `destroy()` releases those resources synchronously from the host's point
  // of view; the returned promise resolves once any async teardown has finished.
  //
  // Used by `resetBackend()`. Implementations should be idempotent: calling
  // `destroy()` twice must not throw. After `destroy()` returns, invoking any
  // other method on the backend is undefined behaviour.
  /**
   * Release external resources held by this backend.
   *
   * See `resetBackend()` for typical usage.
   */
  destroy(): Promise<void>;
}

export class Executable<T = any> {
  constructor(
    /** The `Kernel` or `Routine` that was prepared. */
    readonly source: Kernel | Routine,
    /** Extra data specific to the backend running this executable. */
    readonly data: T,
  ) {}
}

export class SlotError extends Error {
  constructor(slot: Slot) {
    super(`Used a buffer that is invalid or already freed: ${slot}`);
  }
}

export class UnsupportedOpError extends Error {
  constructor(op: AluOp | null, dtype: DType, device: Device, arg?: any) {
    let msg = `${op || ""}<${dtype}> not supported in ${device} backend`;
    if (arg !== undefined) msg += ` with arg ${JSON.stringify(arg)}`;
    super(msg);
  }
}

export class UnsupportedRoutineError extends Error {
  constructor(name: Routines, device: Device) {
    super(`routine '${name}' is not supported in ${device} backend`);
  }
}

// Backend-specific functions are below this line.

/**
 * If the WebGPU backend has been initialized, return the `GPUDevice` that this
 * backend runs on. This is useful for sharing buffers.
 */
export function getWebGPUDevice(): GPUDevice {
  const backend = initializedBackends.get("webgpu") as
    | WebGPUBackend
    | undefined;
  if (!backend) {
    throw new Error(
      "WebGPU backend not initialized, call init('webgpu') first",
    );
  }
  return backend.device;
}
