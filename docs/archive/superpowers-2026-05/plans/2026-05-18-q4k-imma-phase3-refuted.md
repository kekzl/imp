# Q4_K_M INT8 IMMA — Phase 3 E2E A/B: REFUTED on real model

**Date**: 2026-05-18
**Status**: Phase 3 end-to-end A/B on **Gemma-3-12B-it Q4_K_M** refutes the
Phase 2C production deployment. IMMA path is **3.8× slower** than the
default dequant→cuBLAS path at pp2048. Decode unchanged (knob doesn't fire
at M=1). **Recommendation: keep the knob default-OFF, retain the kernel as
research artifact, mark the Q4_K_M IMMA project DEFERRED.**
**Wrap-up of Phase 2B (architecture ceiling)**: `2026-05-18-q4k-imma-phase2b-ceiling.md`.

## Setup

- Model: `gemma-3-12b-it-Q4_K_M.gguf` (6.8 GiB)
- Hardware: RTX 5090 (sm_120a, GB202)
- Build: PR #269 stack (Phase 1 + 2A + 2B + 2C entry + 2C dispatch wire-up)
- Bench: `imp-cli --bench --bench-pp 2048 --bench-reps 10 --prefill-chunk-size 0
  --max-tokens 64 --temperature 0`, 3 trials, median reported
- Env: `CUBLAS_WORKSPACE_CONFIG=:4096:8` for cuBLAS determinism
- IMMA opt-in: `--set gemm.q4k_imma_enabled=true` (default off otherwise)

`--prefill-chunk-size 0` is critical — the default chunked prefill at
`chunk_size = 512` would never let M reach the IMMA dispatch gate (M ≥ 1024).
For the A/B we force the entire 2048-token prompt into a single forward pass
so the IMMA path actually fires.

## Results

| Config | pp2048 (tok/s) | tg64 (tok/s) |
| ---    | ---:           | ---:         |
| OFF (dequant→cuBLAS, default) | **6418** | 116 |
| ON (IMMA, opt-in)              | **1697** | 119 |
| Δ                              | **−74 %** | ±2 % (noise) |

Trial-by-trial:

```
OFF: 6319, 6418, 6414  (median 6418 tok/s, σ ≈ 1.5 %)
ON:  1698, 1691, 1696  (median 1697 tok/s, σ ≈ 0.2 %)
```

Decode (tg64) is unaffected because the dispatch site only emits the IMMA
strategy for M ≥ 1024; decode dispatches with M = 1 and go through the
existing GGUF small-M handlers regardless of the knob state.

## Why is IMMA empirically slower?

The Phase 1 microbench (PR #254) measured raw MMA throughput at 931 TOPS —
the *hardware ceiling*. The Phase 2B chain (PRs #256→#267) measured the
hand-rolled tile kernel at **40 TOPS** in isolation on synthetic shapes
(M=N=4096 K=2048). The Phase 2B wrap-up (`2026-05-18-q4k-imma-phase2b-ceiling.md`)
explicitly flagged this as ~4.3 % of the raw ceiling and noted that closing
the gap requires fundamentally different kernel architecture.

What Phase 3 surfaces empirically:

1. **cuBLAS FP16-TC is 6× faster than IMMA at this scale.** For one Q4_K_M
   GEMM at (M=2048, N=3840, K=3840) — typical Gemma-3-12B FFN tile shape —
   the cost breakdown is:
     - dequant→FP16: ~14 µs (Q4_K 7 MB read + FP16 28 MB write, ≈ 1.7 TB/s HBM)
     - cuBLAS FP16-TC GEMM: ~245 µs (60 GFLOPs at 244 TFLOPS measured)
     - **Total dequant→cuBLAS: ~0.26 ms per dispatch**
     - IMMA at 40 TOPS: 60 GOPS / 40 TOPS = **1.5 ms per dispatch**

   The 6× per-dispatch ratio compounds across the ~48 layers × 7 Q4_K
   weights per layer = ~336 dispatches per prefill, producing the observed
   3.8× e2e slowdown.

2. **Mixed shapes hurt more than the synthetic bench.** Gemma-3-12B's
   layers include attention Q/K/V/O at N=K=3840 (favourable for IMMA — the
   ~40 TOPS plateau region) but FFN gate/up at N=24576 K=3840 (much bigger
   N, more CTAs but also more activation-quant work) and FFN down at
   N=3840 K=24576 (LONG K, lots of inner-loop iterations). The mixed
   profile averages out below the 40 TOPS plateau.

3. **Per-call activation quantization is unamortised.** The IMMA path
   pays `quantize_fp16_to_int8_subblock` once per `mmq_q4k_imma_gemm` call.
   That's a separate kernel launch per dispatch — adds launch overhead
   the dequant→cuBLAS path doesn't pay (it just calls cuBLAS with the
   raw FP16 input).

4. **Weight-cache cold start.** The first prefill pass per (process-load)
   pays the Phase 2A reorder for ~336 weights serially. At ~10-20 µs per
   reorder kernel that's 3-7 ms of one-shot warm-up — visible as a small
   tax in the bench mean (10 reps amortise it).

## Re-eval triggers — when to revisit

The kernel stays in the codebase (PRs #267 #268 #269 retained). Re-eval
becomes worthwhile only when one of these conditions holds:

- A **dense Q4_K_M model with much larger FFN shapes** appears — N ≥ 8192
  with K ≥ 4096 — where the 40 TOPS plateau region is wider and the
  per-call overhead amortises across bigger work.
- A **major kernel restructure** ships:
    - persistent-CTA + stream-K scheduling
    - CUTLASS template instantiation (defer past the v4.5 inspection done
      in `cutlass_4_5_sm120_research_2026_05_10.md`)
    - tcgen05.* / warp-group MMA (SM100-only, n/a for sm_120)
- **A workload appears where cuBLAS dequant→FP16 isn't reachable**:
    - fp16_cache disabled AND dequant_scratch unavailable (rare on imp's
      current configuration; only happens under tight VRAM pressure)
- **Activation-quant cost can be fused** into the prior layer's epilogue
  (e.g. fuse silu·up_gate quantization into the FFN gate kernel that
  produces the activation).

## Decision

| Action | Status |
| --- | --- |
| Keep `mmq_q4k_imma_tile`, `mmq_q4k_imma_reorder`, `mmq_q4k_imma_gemm` | ✅ in `src/compute/` |
| Keep `WeightCaches::q4k_imma` + `gemm.q4k_imma_enabled` knob | ✅ default off |
| Keep `gemm_kernel_q4k_imma.cu` dispatch handler | ✅ on registry, gated by knob |
| Default-OFF documented in roadmap as **refuted at deployment** | ☑ this memo + roadmap update |
| Phase 2C slice 3 (cleanup: WeightCaches populate at load time) | ❌ unnecessary — knob stays off |
| Multi-week kernel restructure to close the cuBLAS gap | ❌ defer indefinitely |

The Q4_K_M IMMA project is **DEFERRED**. The architecture-ceiling memo
(`2026-05-18-q4k-imma-phase2b-ceiling.md`) predicted the kernel might
"already be competitive with the dequant→cuBLAS path the FFN cache-miss
workload would hit." Phase 3 refutes that prediction on the only available
test model. The cuBLAS FP16-TC path is materially faster end-to-end.

## Memos in this thread

- Phase 1 finding: `2026-05-18-q4k-imma-phase1-findings.md` (931 TOPS gate)
- Phase 2B ceiling wrap-up: `2026-05-18-q4k-imma-phase2b-ceiling.md` (40 TOPS plateau)
- **Phase 3 refutation (this memo)**: dense Q4_K_M e2e refuted at 3.8× slowdown

## PRs in the chain

| PR | Status | Subject |
| ---:| --- | --- |
| #254 | merged | Phase 1 microbench (931 TOPS) |
| #255 | merged | Phase 2A reorder kernel |
| #263 | merged | Phase 2C infrastructure (cache struct + knob) |
| #267 | merged | Phase 2B production tile kernel |
| #268 | merged | Phase 2C dispatcher entry (`mmq_q4k_imma_gemm`) |
| #269 | open   | Phase 2C slice 2 (registry handler + dispatch site gate) |
| (this) | n/a | Phase 3 findings doc + roadmap update |

Note: #269 remains useful to merge — the dispatch handler is the only way
the knob actually does anything, even if production users won't flip it on.
Future researchers re-evaluating the kernel will need the handler wired up
to bench the path quickly.
