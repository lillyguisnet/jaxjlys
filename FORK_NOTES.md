# Fork notes — `lillyguisnet/jaxjlys`

> **Read this before making changes.** This file exists to make the purpose and
> scope of the fork explicit for anyone (human or LLM) who touches this repo
> later.

## What this repo is

This is a **private fork** of [`ekzhang/jax-js`](https://github.com/ekzhang/jax-js)
maintained by `lillyguisnet` at [`lillyguisnet/jaxjlys`](https://github.com/lillyguisnet/jaxjlys).

The published npm package is `@jax-js/jax`. We keep the **same package name**
(`@jax-js/jax`) in `package.json` so that consuming projects do not need to
rewrite their imports — they simply point their dependency at this local
checkout (or a git URL) instead of the npm registry.

## Why it exists

The parent project ([`webtwardis`](https://github.com/lillyguisnet/webtwardis))
uses jax-js for browser-native ML inference (ViT classifier and SAM 2.1
segmentation, streaming safetensors weights from OPFS). During the SAM 2.1
proof-of-concept (`jaxjs-sam2-poc.html`) we hit concrete bugs and limitations
in upstream jax-js that we need to patch locally:

1. A **signed right-shift bug** in `WasmAllocator#bumpAlloc` that effectively
   caps practical WASM memory at ~2 GiB instead of the advertised 4 GiB.
   See _Known issues we intend to patch_ below.
2. Possible future patches to improve numerical kernels, expose allocator
   stats, add chunked softmax for large attention matrices, etc.

Rather than monkey-patch at runtime (which is awkward because jax-js uses
true JS private fields `#foo`), we maintain our own build so we can edit the
real source.

## Relationship to upstream

| Remote     | URL                                           | Purpose                           |
| ---------- | --------------------------------------------- | --------------------------------- |
| `origin`   | `https://github.com/lillyguisnet/jaxjlys.git` | Our fork. Push changes here.      |
| `upstream` | `https://github.com/ekzhang/jax-js.git`       | Track upstream for rebases/merges |

The fork was created via `gh repo fork ekzhang/jax-js --clone --fork-name=jaxjlys`,
which configures both remotes automatically.

To pull in upstream changes later:

```bash
git fetch upstream
git merge upstream/main          # or: git rebase upstream/main
pnpm install
pnpm build
```

Expect merge conflicts in any file we've patched (most likely
`src/backend/wasm/allocator.ts`). Resolve by keeping our patched version and
re-applying the fix on top of any upstream refactor.

## Known issues we intend to patch

**None applied yet** — this document currently describes only the
_planned_ changes. When a patch actually lands, move its entry from the
"planned" list to the "applied" list and link to the commit.

### Planned

#### 1. `WasmAllocator#bumpAlloc` signed-shift memory-grow bug

**File:** `src/backend/wasm/allocator.ts`, lines 52–56 (bundled to `dist/backend-*.js` line 2696).

**Current code:**

```ts
this.#memory.grow(
  ((ptr + size + 65535) >> 16) - (this.#memory.buffer.byteLength >> 16),
);
```

**Bug.** `>>` is a _signed_ right shift that converts its operand to an
`int32` first. When the bump pointer plus a new allocation exceeds
`2^31 - 1` ≈ **2.15 GiB**, `ptr + size + 65535` overflows into the negative
half of `int32`, and the page count passed to `WebAssembly.Memory.grow()`
becomes negative. Chromium then throws:

```
TypeError: WebAssembly.Memory.grow(): Argument 0 must be non-negative
```

**Why it matters for us.** SAM 2.1's Hiera-Large encoder at stage 2 performs
global self-attention over 4096 tokens, allocating a ~512 MiB attention matrix
plus similarly sized softmax intermediates. Even with aggressive inter-block
cleanup, the bump pointer climbs past 2 GiB well before the theoretical 4 GiB
ceiling, crashing inference mid-encoder.

**Planned fix.** Replace both `>> 16` with `>>> 16` (unsigned right shift):

```ts
this.#memory.grow(
  (((ptr + size + 65535) >>> 16) - (this.#memory.buffer.byteLength >>> 16)),
);
```

`>>>` keeps the operand in the unsigned 32-bit domain, so values up to
`2^32 - 1` ≈ 4 GiB are represented correctly. This fixes the crash without
changing behavior for allocations below 2 GiB.

**Status.** Not applied yet (parent project wants to work on its own
SAM2-side fixes first). See `webtwardis/jaxjs-sam2-poc.html` for the
consumer that will benefit.

### Applied

_(empty — no patches merged yet)_

## Building

This is a pnpm workspace. Use the pinned version via corepack:

```bash
corepack enable
corepack prepare pnpm@10.32.1 --activate
pnpm install
pnpm build
```

`pnpm build` invokes `tsdown` for each workspace package and emits:

- `dist/index.js` (ESM entry, also exports)
- `dist/index.cjs` (CJS entry)
- `dist/backend-<hash>.js` (WASM backend — **this is the file most of our patches touch**)
- `dist/webgpu-<hash>.js`, `dist/webgl-<hash>.js` (GPU backends)
- `dist/index.d.ts` (+ `.cts` / `.d.cts` variants)

The hashed filenames (`backend-DZvR7mZV.js`, etc.) are stable across builds
because `tsdown` hashes based on content. If a patch changes the backend
source, expect the hash suffix to change — consumers must re-import the new
file name. Upstream currently uses `backend-DZvR7mZV.js`; our identical build
produces the same hash.

Subpackages under `packages/` (`loaders`, `onnx`, `optax`) build as part of
the same `pnpm build` invocation but are not currently consumed by
webtwardis.

## Consuming this fork from another project

### Local development (same machine)

In the consumer's `package.json`:

```json
{
  "dependencies": {
    "@jax-js/jax": "file:../jaxjlys"
  }
}
```

Then `npm install` in the consumer. npm will read this directory's
`package.json`, verify the `"files": ["/dist/*..."]` whitelist, and copy the
built `dist/` into the consumer's `node_modules/@jax-js/jax/`.

**Important:** the consumer copies only files matching the `files` whitelist
(`dist/*.{js,cjs,d.ts,d.cts}`). You must run `pnpm build` in this repo
before the consumer's `npm install` — otherwise `dist/` is empty and the
consumer gets a broken package.

### From git (another machine or CI)

```json
{
  "dependencies": {
    "@jax-js/jax": "github:lillyguisnet/jaxjlys#main"
  }
}
```

This requires committing `dist/` to the branch (currently gitignored), **or**
configuring a `prepare` script so `npm install` runs the build on the
consumer side. Neither is set up yet — revisit when needed.

## Directory layout (summary)

```
jaxjlys/
├── FORK_NOTES.md          ← you are here
├── README.md              ← upstream's README (unchanged, with a pointer to this file)
├── package.json           ← @jax-js/jax — the main package lives at repo root
├── src/
│   ├── backend/
│   │   ├── wasm/
│   │   │   ├── allocator.ts   ← signed-shift bug lives here (line 52)
│   │   │   ├── wasmblr.ts     ← WASM codegen
│   │   │   ├── parallel.ts    ← worker pool
│   │   │   └── builtins.ts    ← math primitives (erf, softmax, etc.)
│   │   ├── webgl/
│   │   └── webgpu/
│   ├── frontend/           ← jit, primitives, tracing
│   └── library/numpy/      ← numpy/nn API surface
├── packages/               ← subpackages (loaders, onnx, optax)
├── dist/                   ← build output (gitignored)
└── tsdown.config.ts        ← build config
```

## Conventions for patches

When you patch something:

1. **Edit the `.ts` source, not the bundled `dist/`.** Rebuild with `pnpm build`.
2. **Move the entry in this file from "Planned" to "Applied"** and include:
   - the commit SHA,
   - a 1–2 line summary of what changed,
   - a link to the consumer bug/issue that motivated it.
3. **Keep a comment at the patch site** starting with `// FORK PATCH:` so
   future merges with upstream surface the change:
   ```ts
   // FORK PATCH: unsigned shift — see FORK_NOTES.md §1
   this.#memory.grow(
     ((ptr + size + 65535) >>> 16) - (this.#memory.buffer.byteLength >>> 16),
   );
   ```
4. **Bump the version only if publishing.** While we consume via `file:` or
   `github:`, the `0.1.11` version in `package.json` can stay — npm identifies
   the dep by path, not semver.
