# Q4_K_M prefill gap to llama.cpp — MMQ path analysis

**Date:** 2026-05-24
**Status:** Research. No source changes. Follow-up direction depends on user choice (3 named levers below).
**Context:** Cross-engine bench (2026-05-24) flagged Q4_K_M prefill as the #1 gap: imp `-48% to -59%` vs llama.cpp on Gemma-3-12B / Gemma-4-26B / Qwen3.6-35B. Prior Phase 1–3 work (PRs #254/#255/#267/#268/#359) shipped an INT8 IMMA tile kernel that was DEFERRED at Phase 3 (3.8× slower e2e than the default `dequant→cuBLAS` path on Gemma-3-12B).

## TL;DR

1. **imp's existing INT8 IMMA kernel is a dead end at its current architecture.** Re-confirmed today: pp1024 single-chunk on Gemma-3-12B Q4_K_M = **1605 tok/s IMMA-on vs 5283 tok/s IMMA-off (3.3× slower)**. Phase 3 verdict holds.
2. **The pp512 gap is in our `dequant→cuBLAS` path, not the IMMA path.** Single-chunk pp512 = 3838 tok/s vs default-chunked 4059 tok/s — chunking is not the bug. llama.cpp at pp512 = 7762 tok/s. We lose ~48% in the default code path.
3. **llama.cpp's MMQ kernel uses the *exact same* `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32` MMA as imp's Phase 2B kernel.** The hardware path is identical. The gap is architectural in how data is fed to the MMA.

## Architectural delta — llama.cpp MMQ Q4_K vs imp Phase 2B IMMA

| Dimension | llama.cpp `mmq.cuh` Q4_K | imp `mmq_q4k_imma_tile` Phase 2B | Implication |
|---|---|---|---|
| MMA primitive | `m16n8k32.s32.s8.s8.s32` | Same | No HW gap |
| Weight storage | Q4_K on device (no expansion) | **Pre-materialized symmetric s8 (2× blow-up)** | imp wastes 10 GB on Qwen3-32B, crowds out KV cache, breaks fp16_cache coexistence |
| Q4→INT8 conversion | **In-SMEM during `load_tiles_q4_K`** (lines 2094-2101 mmq.cuh) | Pre-baked at model load via reorder kernel | llama.cpp avoids the L2/HBM pressure of carrying expanded s8 |
| Weight encoding | **Unsigned 0-15 fed directly into s8 MMA** (zero-extended) | Symmetric shift: `q - 8 ∈ [-8, 7]` | imp adds a β = `8·d·sc - dmin·m` term per sub-block; llama.cpp's β = `-dmin·m` only — half the scale-apply FMAs |
| Tile shape | `mmq_y` = 128 (M), `mmq_x` = 8-128 runtime (N) | `BLOCK_M=64 BLOCK_N=32` fixed | llama.cpp gets 2× M-tile reuse + adaptive N selection |
| Scheduler | **Stream-K persistent CTA** (mmq.cuh:3540-3790) | Fixed 2D data-parallel grid | Stream-K eliminates idle warps on uneven M/N shapes typical of attention vs FFN dispatches |
| Activation quant | Separate prior kernel (`quantize_mmq_q8_1_cuda`), packed into `block_q8_1_mmq` with both `d` and `s` (= sum-of-qs for β coupling) | Separate prior kernel (`quantize_fp16_to_int8_subblock`) | Equivalent in concept |
| Scale apply | 2 FMAs per output element per K-block (line 1233-1234) | 4 FMAs per output element per K-block (β-term needs extra mul) | 2× fewer ops in the inner-loop tail |

**Source citations:**
- Same MMA inline asm: `/home/kekz/github.com/kekzl/llama.cpp/ggml/src/ggml-cuda/mma.cuh:879` (Ampere+)
- In-SMEM nibble decode: `mmq.cuh:2094-2101` — `qs0 = get_int_b4(...); x_qs[...] = (qs0 >> 0) & 0x0F0F0F0F; x_qs[...] = (qs0 >> 4) & 0x0F0F0F0F`
- Scale apply: `mmq.cuh:1233-1234` (MMA path)
- `dm * make_half2(1.0f, -1.0f)`: `mmq.cuh:2133` — packs `d` and `-dmin` into a half2 for cheap FMA
- Stream-K: `mmq.cuh:3540-3790`, gated on `>= GGML_CUDA_CC_VOLTA` (sm_120 qualifies)

## Why imp's Phase 2B plateau at 40 TOPS is consistent with this picture

Phase 2B ceiling memo (`2026-05-18-q4k-imma-phase2b-ceiling.md`) diagnosed the bottleneck as the scale-apply serial chain: "16 FMAs per warp per K-block before next MMA can issue". With imp's symmetric-s8 encoding that's **16 FMAs**; with llama.cpp's unsigned-0-15 encoding it'd be **8 FMAs**. That alone might lift the plateau from 40 → ~70-80 TOPS. Combined with the in-SMEM decode (no 2× SMEM pressure from pre-materialized weights) and stream-K (better load balancing across mixed-shape dispatches), reaching ~100-150 TOPS is plausible — which is roughly where llama.cpp's MMQ Q4_K lands implicitly based on the bench results.

## Fresh data (today, 2026-05-24, build state `main` @ `386c2d9`)

Gemma-3-12B-it Q4_K_M, RTX 5090, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, 5 reps:

| Config | pp512 (tok/s) | pp1024 (tok/s) | tg1 (tok/s) |
|---|---:|---:|---:|
| imp default (dequant→cuBLAS, chunk=512) | 4059* | — | — |
| imp single-chunk (dequant→cuBLAS) | 3838 | 5283 | 14.9 / 19.2 |
| imp single-chunk + IMMA on | — | **1605** | 16.3 |
| llama.cpp `5d246a7` | **7762*** | — | — |

\* from 2026-05-24 cross-engine bench (`docs/cross_engine_bench_2026_05_24.md`).

**Reads:**
- IMMA-on regression is reproducible and large: −69% pp1024.
- imp pp512 is ≈ 49% of llama.cpp pp512 regardless of chunked vs single-chunk prefill.
- Default dequant→cuBLAS path scales nonlinearly: pp512 → pp1024 = 3838 → 5283 (+38%) suggests cuBLAS algo selection is suboptimal at small M.

## Three named levers (pick at most one)

### Lever A — Profile imp's `dequant→cuBLAS` path at pp512 (1-2 days, low risk)

Find what's actually slow vs llama.cpp. Hypotheses to test:
1. cuBLAS algo cache picks a suboptimal heuristic at M=512
2. `dequant_q4k_kernel` has poor occupancy at the small-M shapes
3. Per-call `cudaMalloc/cudaFree` overhead (the Phase 3 refutation noted ~23% host time in these on pp512)

Tools: `nsys profile` on imp + llama.cpp side-by-side, same model, same pp512. Compare top kernel names + ms/call.

**Upside if it fixes:** Potentially closes most of the −48% gap without a kernel rewrite. Cheapest path.
**Downside:** May find no fixable issue in our cuBLAS path; then need Lever B anyway.

### Lever B — Port llama.cpp MMQ approach into imp (2-3 weeks, medium risk)

Rewrite the existing `mmq_q4k_imma_tile` to match llama.cpp's three architectural choices:
1. **In-SMEM nibble decode**: drop the Phase 2A reorder kernel + pre-materialized `WeightCaches::q4k_imma`. Decode in `load_tiles` from raw Q4_K blocks. Saves 2× VRAM; saves load-time reorder pass.
2. **Unsigned-0-15-into-s8 MMA**: drop the symmetric shift. Cuts scale-apply work in half (`β = -dmin·m` only, single FMA instead of two).
3. **Stream-K scheduler**: replace fixed 2D grid with persistent CTAs + tile-coordinate work stealing.

**Upside if successful:** Closes most of the bench gap on 3 of 5 GGUF models (-48 to -59% → near parity). Per Phase 2B ceiling memo this is the named "multi-week kernel restructure".
**Downside:** Multi-week effort, no guarantee we hit llama.cpp parity (their implementation has 5+ years of tuning). Also: imp's Phase 3 eval (PR #359) found Gemma-3-12B has pre-existing quality issues unrelated to IMMA, so e2e validation needs a clean dense Q4_K_M model.

### Lever C — Accept the gap, focus elsewhere

The cross-engine bench (2026-05-24) showed imp wins decode on every model tested (+24 to +86% on GGUF, +57% on NVFP4 dense). Q4_K_M prefill is one workload class. NVFP4 dense pp2048 (-33% vs vLLM on Qwen3-8B-NVFP4) is another gap worth ~10% wall on prefill-dominant workloads.

**Upside:** Keep existing development bandwidth on higher-confidence levers.
**Downside:** Bench gap remains, llama.cpp keeps its Q4_K_M prefill advantage.

## Recommendation

**Lever A first.** Even if Lever B is the eventual destination, profiling our `dequant→cuBLAS` path is a 1-2 day investment that (a) might cheaply close a chunk of the gap on its own, (b) gives concrete kernel-level evidence to size Lever B's payoff, and (c) sets up the side-by-side `nsys` baseline that Lever B would need anyway for verification.

If Lever A finds no fixable bug, escalate to Lever B with the profile data as justification. If Lever A closes the gap to within ~10%, Lever C becomes defensible (no longer the bench's #1 lever).

## Lever A — executed 2026-05-24 (same day)

### Side-by-side nsys profile (pp512, Gemma-3-12B Q4_K_M, RTX 5090)

| Engine | Top kernel | % of GPU time | Insight |
|---|---|---:|---|
| imp | `dequant_q4k_kernel` | **29.5%** (2016 inst × 94 µs avg) | Q4_K → FP16 on every forward pass — never cached |
| imp | cuBLAS FP16 GEMM family (5 tile variants) | 57.6% | Consumes the FP16 dequanted weights |
| llama.cpp | `mul_mat_q<Q4_K, 128, false>` | 58.4% (286 inst × 119 µs) | Direct Q4_K MMA, no dequant step |
| llama.cpp | `dequant_*_kernel` | **0%** | No dequant in the hot path at all |

**Root cause found**: imp's auto-resolver picks `NVFP4_DECODE_ONLY` mode for any model whose first-layer wq is *not* Q*_K-6-8bit (i.e. Q4_K-dominant models qualify, `engine_init_resolver.cpp:198`). That strategy makes `pre_dequant_phase1_fp16_cache.cu:35-40` skip the FP16 cache **unconditionally**. Phase 3 NVFP4 cache only covers `nvfp4_beneficial` qtypes (Q8_0/Q8_K/Q6_K/Q5_K — *not Q4_K*). Result: Q4_K-dominant models get **zero cache** and pay per-dispatch dequant on every prefill forward pass.

### Fix shipped on branch `fix/q4k-fp16-cache-coverage-gap`

Two edits in `src/exec/pre_dequant_phase1_fp16_cache.cu`: replace the unconditional NVFP4_DECODE_ONLY early-exit with a per-weight gate that caches as FP16 the weights NOT covered by the NVFP4 cache (i.e. `!nvfp4_beneficial(qtype)`). Plus `src/exec/executor_kernels.cu` gates the FP16-cache dispatch on M > 1 (decode keeps small-M GGUF path).

### Measured impact on Gemma-3-12B-it Q4_K_M (RTX 5090, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, 5 reps, single-chunk)

| Metric | Pre-fix (mode 2 default) | Post-fix (mode 2 + cache gap closed) | Δ |
|---|---:|---:|---:|
| pp512 (tok/s) | 3838 | **22701** | **+491%** |
| pp1024 (tok/s) | 5283 | (similar pattern expected) | — |
| tg128 (tok/s) | 134 (single-chunk re-bench) | **86** | **−36%** |
| llama.cpp pp512 (reference) | 7575 | 7575 | imp post-fix now **3.0× faster** |
| llama.cpp tg128 (reference) | 139 | 139 | imp post-fix now **0.62× = 38% slower** |

**Mixed result.** Prefill gap to llama.cpp is closed and reversed; decode lost 36%. Profiling shows the regression source: at M=1, residual-fused dispatches (`gemm_dispatch` with `beta != 0` at `executor_attention.cu:1149`, `executor_ffn.cu:435`) now hit the FP16 cache + `gemm()` → cuBLAS picks a `tensorop_relu_*_128x64_64x3_tn_align8` tile-GEMM algo that wastes 127/128 rows. The pre-fix path went through `dequant_gpu` → scratch → `gemm()` → cuBLAS picked the correct `wmma_tensorop_*_16x16_128x2_tn_align8` gemv-style algo. Gating the residual-fuse `fp16.find` on M > 1 (also shipped) did NOT fix it — the dispatch is going somewhere else for M=1 residual-fused weights, possibly through `gemm()` directly at a higher caller, or the cuBLAS algo cache state is contaminated.

### Trade-off assessment

For typical chat workload (prefill-heavy due to context, decode = short-medium response):
- 50/50 prefill/decode wall: geometric mean of `4.91 × 0.64 = 1.77` → **+77% net**
- 80/20 prefill-heavy: `4.91^0.8 × 0.64^0.2 = 3.10` → **+210% net**
- 20/80 decode-heavy: `4.91^0.2 × 0.64^0.8 = 0.96` → **−4% net (essentially flat)**

**The fix is net-positive for almost any realistic mix but a clear loss for pure-decode workloads.** Three options:

1. **Ship as-is**, accept −36% decode for +491% prefill (net positive on typical workloads). Treat the decode regression as a follow-up perf bug to investigate.
2. **Don't cache w_down and wo** (the two known residual-fused weights) at FP16. Loses ~30% of the prefill gain but preserves decode. Cleaner shape but more code.
3. **Land Lever A only as a documented finding + memo**, don't ship the fix yet. Pursue the right fix that doesn't regress decode (likely needs a fused "dequant→gemv→add" path at M=1, several days work).

### Regression investigation (post-decision: dig further)

Added a probe to `gemm_try_gemv()`: confirmed it's only hit at M>1 (prefill). The M=1 residual-fuse path in `executor_attention.cu`/`executor_ffn.cu` requires `n > 1` for both `will_fuse_o_beta1` and `will_fuse_o_dequant_beta1`, so at decode the residual is done as separate steps:
1. `gemm_dispatch(..., ctx)` with beta=0 → standard dispatch → should hit M=1 small-M GGUF path (dp4a)
2. Separate residual-add kernel

This is the SAME path pre-fix and post-fix would take. So the +18 GiB FP16 cache itself isn't being read at M=1 decode — but something else changed in the kernel selection. The post-fix decode profile shows `cutlass_80_wmma_tensorop_f16_s161616gemm_f16_16x16_*` family kernels (lots of inst), suggesting cuBLAS is being called somewhere at M=1 instead of the dp4a path. Plausible causes (not yet root-caused):

- cuBLAS algo cache contamination: the bigger pp prefill picks tile-GEMM algos which persist into M=1 decode dispatches via cuBLASLt's preference cache.
- Memory layout / fragmentation: 18 GiB FP16 cache changes HBM allocation patterns; some non-Q4_K dispatch (e.g. attention QKV computed from `wq`/`wk`/`wv` which IS Q6_K → NVFP4 cache) may now route differently.
- L2 set conflict between FP16 cache and Q4_K original weights at decode read time.

### Cross-model impact verification (post-fix, 2026-05-24)

Other models I'm confident the fix doesn't touch (different VRAM strategy → Phase 1 already skipped):

| Model | Strategy | pp512 | tg128 | Note |
|---|---|---:|---:|---|
| Qwen3-14B Q6_K (north-star) | FP8_PREFILL_NVFP4_DECODE (mode 1) | 5977 | 165 | matches baseline 165 ✓ |
| Qwen3-8B Q8_0 | FP8_PREFILL_NVFP4_DECODE (mode 1) | 12400 | 274 | matches baseline 272 ✓ |
| Gemma-4-26B-A4B Q4_K_M (MoE) | FP8_PREFILL_NVFP4_DECODE (mode 1, MoE) | 4688 | 255 | matches yesterday 4734/258 ✓ |

Only Gemma-3-12B Q4_K_M is affected (the only dense-Q4_K target in the local model set). Qwen3.6-35B MoE Q4_K_M (also in cross-engine bench) is MoE → mode 1 → not affected by this fix's path.

### Decision deferred to user

Tested all three lever variants:
- (A) Phase 1 cache fix alone (line 1806 unchanged): pp512 +480%, tg128 −36%
- (B) Phase 1 + M>1 gate on FP16-cache dispatch: pp512 +480%, tg128 −36% (gate is no-op for this path)
- (C) Phase 1 + beta-path FP16-cache gate on M>1: pp512 +480%, tg128 −36% (also no-op — residual fuse not taken at decode)

All three give the same numbers. The decode regression isn't from the FP16-cache dispatch path — it's from cuBLAS algo selection / memory pressure when 18 GiB additional FP16 cache exists. Closing the regression would need either:
- Selective caching (skip residual-fused weights)
- A flag to disable the FP16 cache when prefill workloads are rare
- Root-causing the cuBLAS algo selection at M=1 (multi-day investigation)

## Cross-references

- Design memo: `docs/plans/q4k_imma_design_2026_05_17.md`
- Phase 2B ceiling: `docs/archive/superpowers-2026-05/plans/2026-05-18-q4k-imma-phase2b-ceiling.md`
- Phase 3 refutation: `docs/archive/superpowers-2026-05/plans/2026-05-18-q4k-imma-phase3-refuted.md`
- Phase 3 eval (later): `docs/superpowers/specs/2026-05-22-q4k-imma-phase3-eval.md`
- Cross-engine bench: `docs/cross_engine_bench_2026_05_24.md`
- llama.cpp source: `/home/kekz/github.com/kekzl/llama.cpp/ggml/src/ggml-cuda/mmq.cuh` (@ `ae65fbd`)
