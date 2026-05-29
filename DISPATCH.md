# DISPATCH — FlashAttention-2 via CUDA 13.3 Tile (C++)

Progress log for the autonomous Tile-FA2 mandate (see `GOAL.md`). A fresh run
continues from "Next step" at the bottom. Newest entry last.

## Mission (short)
Build a production Tile-C++ FA2 attention path (prefill + paged-decode) that meets
or beats the native hand-tuned path, gated behind `--attn-backend=tile` (default
`native`) until proven faster at equal correctness. Anti-degeneration contract (§4)
is absolute: no metric regresses, native path is never deleted.

## Environment (verified 2026-05-29)
- Toolkit **CUDA 13.3** (nvcc V13.3.33, PTX ISA 9.3); host driver **610.47** (≫ R590). §11 preconditions OK.
- Build: Docker `imp:builder-133` (cuda-toolkit-13-3). Dev container `impdev` (host `/home/kekz/models` mounted).
- imp already switched to 13.3 (Dockerfile builder+runtime); full GPU suite green, PR #477 (hand-written FA2) open.

## Phase 0 — Investigation (§5) — DONE, decision = GO
CUDA Tile C++ structural viability for FA2 on sm_120a, evidence-based:
- API: header `cuda_tile.h` → `crt/cuda_tile.h`, namespace `cuda::tiles` (alias `ct`). Kernel attr
  `__tile_global__`. Ops: `ct::tensor_span{ptr, ct::extents{...}}`, `ct::partition_view{span, ct::shape{...}}`,
  `.load_masked(blockIdx...)` / `.store_masked(tile, blockIdx...)`, `ct::mma(A,B,acc)`, `ct::matmul(A,B)`,
  `ct::full<tile<E,shape<...>>>(v)`, `ct::bid()`, `ct::irange()`, literals `_ic`. Compile `nvcc -std=c++20 --enable-tile -arch=sm_120a`.
- **Arch:** NVIDIA blog confirms cuTile supports "Blackwell (compute capabilities 10.x and 12.x)" → **CC 12.x = sm_120 = RTX 5090 officially supported.**
- **SASS proof (`tools/analysis/tile_probe_mma.cu`, fp16 64×64×64, AOT via `--tilecubin`):
  lowers to `HMMA.16816.F16` (= mma.sync m16n8k16). NO tcgen05/UTCMMA.** sm_120-runnable. §11 NOT triggered.
- **dtypes (matmul):** int8, fp8 e4m3/e5m2, fp16, bf16, tf32, float, double. **No e2m1/NVFP4.** Irrelevant for
  attention (KV is FP16/FP8/INT8); NVFP4 is GEMM-weight-only. Documented gap (§5.6), not a blocker.
- **⚠ Runtime JIT unavailable in WSL2 container** ("PTX JIT compiler library not found" — driver dlopen fails even
  with caps=all + WSL ptxjitcompiler on path). Tile IR is JIT'd at launch by default. **→ imp MUST AOT-compile tile
  code** (tileiras → SASS/cubin embedded), which is required anyway (§10, no runtime deps). AOT confirmed working.
- Probes: `tools/analysis/tile_probe.cu` (fp32 tiny → scalar FMA, ran but WSL-JIT no-op), `tile_probe_mma.cu`
  (fp16 → HMMA, AOT SASS inspected). Both committed as investigation artifacts.

### Open integration questions (resolve in Phase 1)
1. **AOT build integration:** how to embed tileiras-assembled SASS into a linkable object the host can launch
   (the default `--enable-tile` embeds tile IR → JIT). `--tilecubin`/`--tilefatbin` produce device-only artifacts;
   need the host-linkable form (separate compile + `cuModuleLoad`, or an nvcc flag that embeds AOT tile SASS in the
   fatbin). MUST be solved before any perf claim — and CMake must do it for `sm_120a` (+ `sm_120f` fallback).
2. **Tile execution model:** `<<<grid,1>>>` launch; tile maps across a warp/block via SHFL.BFLY (seen in SASS).
   Confirm block/warp mapping + smem usage for an attention tile (Bq×Bkv scores + online softmax in the tile model).
3. **Online softmax in Tile:** FA2 needs running max/sum + rescale across KV tiles. Determine the cuTile idiom
   (reductions over a tile axis, `ct::` reduce/exp ops) — see the official "Tuning Flash Attention ... CUDA Tile" blog + TileGym.

## Next step
Phase 1 (§4.1): establish baselines BEFORE writing the Tile FA2 —
`git tag baseline/fa2-tile-pre`, freeze `bench/baseline.json` (decode/prefill/long-ctx for the reference
models incl. Qwen3.6-35B-A3B-NVFP4 decode, Qwen3-Coder MoE prefill, Qwen3-14B-NVFP4 pp4096), capture golden
logits + a perplexity reference. Then resolve AOT-build-integration (open Q1) with a hello-world tile object
linked + launched from a host harness, run on GPU, verify correct. Only then start the Tile FA2 kernel skeleton.
