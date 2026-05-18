# Q4_K_M INT8 IMMA — Phase 2B chain wrap-up & architecture ceiling

**Date**: 2026-05-18
**Status**: Phase 2B chain (PRs #256 → #261) explored 5 incremental optimisations
on top of the hand-rolled INT8 IMMA tile kernel. The kernel **plateaus at
~40 TOPS** on production FFN shapes — well short of cuBLAS FP16-TC's
~244 TFLOPS but already competitive with the dequant→cuBLAS path that the
target workload (Q4_K_M weights bypassing `fp16_cache`) would hit.
**Design memo**: `docs/plans/q4k_imma_design_2026_05_17.md`.
**Phase 1 findings**: `2026-05-18-q4k-imma-phase1-findings.md`.

## Summary table — Phase 2B sweep (M=4096 N=4096 K=2048, mean of 20 reps)

| Phase | Kernel change                                    | TOPS | Δ vs prev |
| ---   | ---                                              | ---: | ---:     |
| 2B    | 1 warp / CTA, BLOCK_M=16 N=8, sync loads         | n/a  | —        |
| 2B.1  | + 2-stage cp.async pipeline                      | 15.9*| n/a       |
| 2B.2  | + 4 warps / CTA, BLOCK_M=32 N=16                  | 19.3*| +21 %    |
| 2B.3  | + WRM·WRN=2·2 per warp, BLOCK_M=64 N=32           | 40.1 | **+108 %** |
| 2B.4  | + 3-stage cp.async                                | 38.5 | −4 %      |
| 2B.5  | + ldmatrix.x4 / x2                                 | 37.5 | −3 %      |

\* Phase 2B.1 / 2B.2 measured at smaller shapes (M=512 N=256 K=2048) where the
  CTA count was favourable. At full saturation (M=N=4096) Phase 2B.3 is the
  high-water mark.

## What worked

1. **Big tile per CTA (Phase 2B.3).** Going from `BLOCK_M=32 N=16` (4 MMAs
   per CTA per K-block) to `BLOCK_M=64 N=32` with WRM·WRN=2·2 per warp
   (16 MMAs per CTA per K-block) was the *one* change that meaningfully moved
   the needle — 2× throughput at production scale. Bigger tile = more weight
   reuse per CTA = less SMEM bandwidth pressure per MMA.

2. **The hardware ceiling exists.** Phase 1's 931 TOPS raw MMA ceiling
   (PR #254) was not throttled or sm_120-specific — the IMMA pipe really is
   willing to issue at that rate. We just can't *feed* it that fast.

## What didn't work (refuted hypotheses)

1. **3-stage cp.async pipeline (Phase 2B.4).** Mixed result: +35 % at small
   shapes where per-CTA latency dominates, but −7 % at full saturation. The
   2-stage pipeline is already enough to hide HBM latency at K=2048; the
   third stage's SMEM cost (+50 %) and longer prologue don't amortise.

2. **ldmatrix.x4 / x2 SMEM→reg coalescing (Phase 2B.5).** Bit-identical
   output, neutral-to-negative throughput. SMEM→register bandwidth was
   **not** the binding constraint at the plateau. The remaining bottleneck
   sits elsewhere — most likely scale-apply FP16→FP32 conversions + FMA
   chain (16 FMAs per warp per K-block) or per-K-block `__syncthreads`.

## Architectural ceiling diagnosis

At 40 TOPS / 931 TOPS = 4.3 % of raw MMA peak, the kernel is severely
**under-issued**. The scale-apply phase between MMA calls forces a
serial dependency: each K-block ends with a `__syncthreads()`, then the
warp serially computes 16 × (4 FP16→FP32 + 4 FMAs) = 128 FP32 ops per
K-block per warp before the next MMA can issue. At 64 K-blocks per K=2048
shape, that's 8192 FP32 ops per warp serially executed in the inner loop —
likely the SOL bound for this architecture.

## What it would take to do better

To close the gap to cuBLAS FP16-TC (~244 TFLOPS), **fundamentally different
kernel shape** is required:

| Approach                              | sm_120 viability | Effort |
| ---                                   | ---              | ---:   |
| Warp-group MMA (wgmma)                | not supported    | n/a    |
| `tcgen05.*` family                    | SM100 only       | n/a    |
| Persistent CTAs + stream-K scheduling  | yes              | 2-3 wk |
| CUTLASS template instantiation        | yes (clunky)     | 1-2 wk |
| Fuse scale-apply into MMA accumulator | yes (research)   | 1 wk   |

The first two paths are hardware-blocked. The remaining three are separate
multi-week kernel projects, not incremental tweaks on the current
hand-rolled tile. They're **not** worth pursuing for the Q4_K_M IMMA
project unless a workload appears where ~+10 % e2e on dense Q4_K_M models
materially matters.

## Production wire-up assessment (Phase 2C)

The Phase 2B.3 kernel at **40 TOPS plateau** is what would land in production
via Phase 2C. The relevant comparison is *not* to cuBLAS FP16-TC (which the
fp16_cache path already uses) but to the **dequant→cuBLAS** fallback that
fires when fp16_cache doesn't hit. Per `docs/plans/q4k_imma_design_2026_05_17.md`
§5.4 — §5.5, that fallback is roughly 50-80 TOPS effective on dense Q4_K_M
when dequant cost is included.

40 TOPS direct-IMMA is competitive with that *only at the M ≥ 1024 dense
shapes*; below that the dequant→cuBLAS path wins on raw kernel cost. The
shipped 6-phase chain therefore makes sense **only** if Phase 2C's
dispatcher gates IMMA dispatch to the M ≥ 1024 dense Q4_K_M region — a
narrow workload (Qwen3-32B Q4_K_M dense prefill, Gemma-3-12B Q4_K_M dense
prefill).

**Recommendation for Phase 2C**:

- Land the **Phase 2B.3 kernel only** (drop the 3-stage / ldmatrix variants —
  both were refuted or neutral). PR #259 is the production target.
- Gate dispatch behind a per-shape heuristic: `M >= 1024 && dense && Q4_K
  && !fp16_cache_hit`.
- Default off (`gemm.q4k_imma_enabled = false`); opt-in via `imp.conf`
  until the M ≥ 1024 perf data lands on real models.
- E2E A/B against Qwen3-32B-Instruct Q4_K_M and Gemma-3-12B-it Q4_K_M.

## Decision

**Phase 2B exploration COMPLETE**. The hand-rolled tile-kernel approach has
reached its honest ceiling at ~40 TOPS on sm_120a. Further perf requires
multi-week kernel restructure that the design memo's §6 plan does not
contemplate. Phase 2C should ship the Phase 2B.3 kernel as-is, with a
defensive dispatcher, and let real-world workloads drive whether the
multi-week restructures are worth funding.

## Code shipped this chain (sequential PRs)

| PR    | Title                                                       | Files |
| ---   | ---                                                         | ---   |
| #254  | Phase 1 microbench — INT8 IMMA confirmed at 931 TOPS         | new   |
| #255  | Phase 2A — Q4_K → symmetric-s8 reorder kernel                | new   |
| #256  | Phase 2B-minimum — INT8 IMMA tile kernel (correctness)        | new   |
| #257  | Phase 2B.1 — cp.async 2-stage pipeline                        | edit  |
| #258  | Phase 2B.2 — 4 warps / CTA, BLOCK_M=32 N=16                    | edit  |
| #259  | Phase 2B.3 — BLOCK_M=64 N=32, WRM·WRN=2·2  **← Phase 2C target** | edit  |
| #260  | Phase 2B.4 — 3-stage cp.async (mixed)                         | edit  |
| #261  | Phase 2B.5 — ldmatrix.x4/x2 (neutral)                          | edit  |

## Cross-references

- Design memo: `docs/plans/q4k_imma_design_2026_05_17.md` (812 lines)
- Phase 1 findings: `docs/superpowers/plans/2026-05-18-q4k-imma-phase1-findings.md`
- Roadmap: `docs/roadmap.md` §`pp=512 on large dense models`
- v2 HMMA retirement context: `mmq_q4k_v2_phase2_shipped_2026_05_16.md`
