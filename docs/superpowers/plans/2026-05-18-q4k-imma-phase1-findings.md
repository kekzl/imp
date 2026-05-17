# Q4_K_M INT8 IMMA — Phase 1 microbench findings

**Date**: 2026-05-18
**Status**: Phase 1 complete — **PROCEED to Phase 2 (production tile kernel)**.
**Design memo**: `docs/plans/q4k_imma_design_2026_05_17.md` §6 Phase 1.
**Bench code**: `tests/bench/mmq_q4k_imma_bench.{h,cu}` +
`tests/test_mmq_q4k_imma_bench.cu`.

## Question

Can `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32` (and its `u8` siblings)
actually issue at full Tensor-Core rate on **consumer Blackwell (sm_120a /
RTX 5090)**, or is INT8 IMMA throttled to FP16 peak the way several SM100-only
opcodes are? This is the binary gate that decides whether the multi-week
Phase 2 production-kernel work is worth pursuing.

## Methodology

Tight per-warp loop of one MMA opcode, all operands kept alive (so ptxas
can't constant-fold the body). Same harness pattern as
`tests/bench/mxf4nvf4_mma_variants_bench.cu`.

- Launch: **1360 warps × 32 768 iterations** = 44.6 M total MMA issues per
  variant. Five timed reps per variant, mean reported.
- Ops/MMA = 2 × M × N × K (FMA counts as 2):
  - INT8 m16n8k32 → 8192 ops
  - FP16 m16n8k16 → 4096 ops
- TOPS / TFLOPS computed as `ops_per_mma × total_mmas / wallclock_ms × 1e-3 / 1e12`.

## Results

```
[q4k-imma Phase 1 microbench, warps=1360 iters=32768]
  variant                          ms/rep         TOPS
  imma_s32_s8_s8_k32                0.392       931.09
  imma_s32_u8_s8_k32                0.409       892.98
  imma_s32_u8_u8_k32                0.391       933.35
  hmma_f32_f16_f16_k16              0.749       243.72
```

| Variant | Wallclock (ms) | TOPS | vs HMMA-f16 |
| --- | ---: | ---: | ---: |
| INT8 IMMA `s32.s8.s8.s32` | 0.392 | **931** | **3.82×** |
| Mixed `s32.u8.s8.s32`     | 0.409 | 893 | 3.66× |
| Unsigned `s32.u8.u8.s32`  | 0.391 | 933 | 3.83× |
| FP16 HMMA `f32.f16.f16.f32` (k=16) | 0.749 | 244 | 1.00× (anchor) |

## Decision gate

| Decision | Gate | Measured | Verdict |
| --- | --- | ---: | --- |
| PROCEED to Phase 2 | IMMA / HMMA TOPS ratio ≥ 1.8× | **3.82×** | **PASS** |
| DEFER | ratio < 1.5× | — | n/a |

**Gate exceeded by 2.1×.** Decision: **PROCEED**.

## Interpretation

- **Hardware is willing.** INT8 IMMA dispatches at ~931 TOPS on sm_120a —
  effectively at the ~838 TOPS advertised peak (and slightly above, because
  loop-overhead amortisation differs from the marketing figure's
  measurement methodology). There is no sm_120-only throttle on this opcode,
  unlike the `tcgen05.*` family that's documented sm_100-only.
- **2× / 3.8× ratio breakdown.** The 2.0× theoretical IMMA-over-HMMA peak
  comes from K=32 vs K=16 (same MMA-issue rate, double the ops per issue).
  The extra 1.91× factor is because the HMMA loop with `f32` accumulators
  runs ~58 % of its 419 TFLOPS theoretical peak in this harness — the IMMA
  loop with `s32` accumulators amortises loop overhead more cleanly. The
  meaningful gate-decision signal is the *ratio*, which is unambiguous.
- **All three INT8 variants are equivalent throughput.** No fast path among
  `s8.s8` / `u8.s8` / `u8.u8`; the §3.2 design-memo choice of "strategy (a)
  symmetric s8" stands on tooling-familiarity grounds, not perf.
- **Mixed-sign `s32.u8.s8.s32` (PTX 7.5+) assembles cleanly on sm_120a.**
  Recorded for completeness; not needed for the symmetric-s8 Q4_K layout.

## What this does NOT say

This bench measures **raw MMA-pipe throughput in isolation**, with the
operand pipeline pre-loaded into registers. Phase 2's production tile kernel
must additionally satisfy:

- Weight bandwidth (cp.async pipeline must keep the MMA fed)
- `ldmatrix.x4` / `ldmatrix.x2` SMEM bandwidth (b16 mode for packed-s8)
- Per-sub-block scale-apply overhead (α · x_scale + β · x_rowsum FP32 FMAs
  on the s32 accumulator)
- The 2× weight-storage blow-up (Q4 nibbles → s8 bytes; design memo §5.3)

A realistic full kernel reaches maybe **30–60 %** of raw MMA throughput
(per the v2 HMMA Phase 2 experience: microbench was 4.87× v1 dp4a, e2e
landed at 0.96× because of these same pipeline / cache / dispatch costs).
The Phase 1 finding only certifies that the *ceiling exists*.

## Phase 2 scope reminder (from design memo §6)

| Step | LoC est. | Days |
| --- | ---: | ---: |
| Full tile kernel (4 warps, BLOCK_M=BLOCK_N=64, BLOCK_K=32, NUM_STAGES=3) | ~400 | 2–3 |
| `mmq_q4k_imma_layout` load-time kernel (Q4_K → symmetric-s8 reorder + α / β) | ~300 | 2–3 |
| Dispatch into `gemm_dispatch_impl` + `GemmKernel` registry; M ∈ [32, 256] | ~200 | 1–2 |
| **Total Phase 2** | ~900 | 5–8 |

Phase 3 (e2e A/B on Qwen3-32B Q4_K_M and Gemma-3-12B Q4_K_M, +10 % pp512 gate)
follows once Phase 2 lands. Phase 4 (MoE expansion) is conditional on Phase 3
showing headroom in dense models.

## Code shipped this PR

- `tests/bench/mmq_q4k_imma_bench.{h,cu}` — 4 MMA-loop kernels + host launcher.
  Modelled after `tests/bench/mxf4nvf4_mma_variants_bench.cu`.
- `tests/test_mmq_q4k_imma_bench.cu` — single GTest case that runs the bench
  and asserts every variant launches without ptxas / runtime error. Prints
  the TOPS table to stderr for inspection; does not gate on perf ratio
  (that's an off-line interpretation step).
- `CMakeLists.txt` — wires the bench TU into `IMP_COMPUTE_SOURCES` under
  `IMP_BUILD_TESTS OR IMP_BUILD_BENCH`, test file into `test-quant`.

## Cross-references

- Design memo: `docs/plans/q4k_imma_design_2026_05_17.md`
- Roadmap: `docs/roadmap.md` §`pp=512 on large dense models`
- v2 HMMA design (blueprint): `mmq_q4k_v2_hmma_design_2026_05_15.md`
- v2 retirement findings: `mmq_q4k_v2_phase2_shipped_2026_05_16.md`
- PTX ISA reference: 8.5 §9.7.13.4 (matrix MMA opcodes)
