# Track E — final closing notes

**Date:** 2026-05-22
**Status:** CLOSED — code removed, kept as historical reference in this doc
**TL;DR:** We cannot beat cuBLAS for FP16 prefill attention on sm_120a with
hand-written FA2-style kernels. Six variants tried; none won.

## Original goal

Replace cuBLAS materialised-S-matrix prefill (`attention_cublas_prefill`)
with a tiled streaming FA2-style kernel to (a) eliminate the 1 GiB S-matrix
workspace cap and (b) deliver 3-5× prefill speedup per the Säule-3 ceiling
microbench projection.

## The six versions tried

| Version | Architecture | pp8192 vs cuBLAS | Reason eliminated |
|---|---|---:|---|
| v1 | Raw PTX + 4+4 warp-spec + cp.async | "+17%" timing only | Layout bugs in ldmatrix→mma handoff → garbage output on real model weights. The +17% was an artefact of the broken kernel doing less work, not a real speedup. |
| v2 | WMMA QKᵀ + scalar FP32 PV | −62% | Threw away tensor cores for PV. Correct but unusably slow. |
| v3 | WMMA QKᵀ + WMMA PV (synchronous) | −36% | No cp.async pipelining. |
| v4 | v3 + cp.async K double-buffer + V overlap | −29% | Best correct FP16 design. Still loses to cuBLAS. |
| v5 | v4 + mxf4nvf4 QKᵀ (in-kernel K quantisation) | −29% | mxf4nvf4 mma is 2.6× HMMA throughput, but per-tile K quantisation eats the entire win. |
| v6 | Pre-quantised NVFP4 K/V + mxf4nvf4 QKᵀ + WMMA PV (with V dequant per tile) | −23% | Eliminated in-kernel K quant, but V dequant + un-cached K gmem load dominate. |

All variants verified correct on the 6 production models (Qwen3-8B Q8_0/NVFP4,
Gemma-4-26B Q8_0/Q4_K_M/NVFP4, Qwen3.6-35B-A3B-NVFP4) except v1 which had the
layout bugs.

## Why cuBLAS wins

cuBLAS achieves higher tensor-core utilisation than we can in a single
hand-written kernel:

1. **Larger internal tiles.** cuBLAS picks tile sizes that don't fit in our
   100 KiB SMEM cap on sm_120 (we're stuck at Br ≤ 128, Bkv ≤ 64 for hd=128;
   cuBLAS uses larger tiles by splitting across multiple SMs differently).
2. **TMA + cluster launch.** sm_120a supports TMA-WS grouped-GEMM tactics;
   cuBLAS uses them, we don't have a working integration.
3. **Shape-specific heuristics.** cuBLAS picks per-shape algorithm; our
   one-size-fits-all template wastes throughput on edge shapes.
4. **Pipelining maturity.** cuBLAS's internal pipelining is years of
   tuning; our 4+4 warp-spec attempt actually regressed (v5 vs v4)
   because at m16n16k16 throughput per warp is already saturating, and
   splitting warps into producers/consumers loses more compute than it
   gains in load overlap.

## What would actually work (if anyone returns to this)

1. **CUTLASS-based FA3 port** with TMA-WS + persistent kernels. Multi-week
   project. CUTLASS v4.5 has the primitives. Worth ~+30% if executed
   correctly, but a lot of integration work.
2. **Wait for Blackwell-datacenter** with tcgen05/wgmma + TMEM. Not
   available on consumer 5090.
3. **Spend cycles on the bigger lever** — per nsys profile of Qwen3-8B
   Q8_0 pp512: `dequant_q8_0_kernel` is 21.5% of total prefill time,
   FFN GEMMs total 29%. Attention is only 2.3%. Even a perfect Track E
   gives at most +5-8% e2e on Q8_0 workloads.

## What stays in the codebase

- `docs/superpowers/specs/2026-05-2[01]-track-e-*.md` — five historical
  spec/report docs (this one is the sixth). Kept as the audit trail.

## What was removed

- `src/compute/attention_tiled_streaming.{h,cu}` — the kernel
- `src/exec/executor_attention.cu` — Track E dispatch path (3 call sites)
- `tests/test_attention_tiled_streaming.cu` — correctness tests
- `tests/test_mma_layout_probe.cu` + bench scaffolding — probe tooling
- `tests/bench/attention_prefill_paths_bench.{cu,h}` + test wrapper
- `tests/bench/tiled_attention_ceiling_bench.{cu,h}` + test wrapper
- `scripts/analyze_attention_workspace_savings.py`
- All CMakeLists entries for the above

## Lessons embedded

1. **A unit test that passes against cuBLAS within FP16 tolerance does NOT
   imply the kernel is correct on real attention distributions.** Uniform
   synthetic fill (LCG, magnitude 0.125 or even 1.0) masks layout bugs
   that cascade catastrophically on real weights. Future kernel work must
   bench with Gaussian or sampled-from-checkpoint distributions.
2. **Real-model smoke tests must run in CI.** verify-fast skips models
   when not present in the container, which let multiple broken Track E
   PRs reach main during this work.
3. **Layout assumptions must be empirically verified via probes**, not
   inferred from PTX documentation alone. We misread the m16n8k16
   D-fragment layout multiple times.
4. **Auto-merge on bundled disable+fix branches is dangerous.** PR #352
   auto-merged on its disable commit before the fix commit landed.
5. **+N% perf claims must include a correctness gate**, not just a unit
   test. v1's +17% bench was meaningless because the kernel was producing
   garbage.

## PRs in the Track E saga

| PR | Status | Summary |
|---|---|---|
| #350 | merged | Base kernel (v1) — buggy |
| #351 | merged | 4+4 warp-spec, perf docs |
| #352 | merged | First disable hotfix |
| #353 | merged | PV repack fix (partial) |
| #354 | merged | Re-disable after multi-model break |
| #355 | merged | Layout probe + bug analysis docs |
| #356 | merged | v2 WMMA correct but slow |
| #357 | merged | Disable v2 |
| this | open | Close Track E — remove code |

Eight hours of work over two days. Net production impact: zero (cuBLAS
remained the right answer the whole time). Net knowledge: we now know
exactly why an FA2 kernel can't beat cuBLAS on sm_120a, and the layout
quirks of m16n16k16 ldmatrix + mma are documented for future kernel work.
