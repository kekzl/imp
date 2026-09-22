<!--
layer: L1
audience: records
verified: 2026-09-22
commit: 9cbb8004
-->

# Kernel design moves refuted, 2026

Moved verbatim from `docs/internals/KERNELS.md` during the docs prose cleanup (2026-09-22). Current kernel reference and lever-status table: [`docs/internals/KERNELS.md`](../internals/KERNELS.md).

---

### The occupancy follow-up - REFUTED (2026-06-17)

Hypothesis from the "48 % vs prod ~41 %": CUTLASS's datacenter-tuned **TMA + warp-specialization** pays an *occupancy tax* on consumer sm_120 that a simpler cp.async + good-layout path avoids. Isolated GEMM microbench (`imp-bench nvfp4`, dense `gemm_nvfp4_cutlass_sm120`, square and small-M shapes) + apples-to-apples ncu against the standalone, same box, same session, **refutes it**:

| @ 4096³ | Prod (`KernelTmaWarpSpecializedCooperativeBlockScaledSm120`) | Standalone (cp.async) |
|---|---|---|
| Time (ncu) | **163 µs** | 194 µs (+19 %) |
| Achieved occupancy | 20.9 % | 31.5 % |
| SM-pipe throughput | **71 %** | 54 % |
| DRAM throughput | 11.7 % | 22.3 % |
| Regs/thread, blocks/SM | 151, 1 | 122, 2 |

- **Prod wins at the peak shape** (903 → 1284 TOP/s @ 4k → 8k = 44.7 → 63.6 % of the 2019-TOPS measured peak; standalone 813 / 972). The earlier "prod ~41 %" was a roofline-pipeline number over real **small-M** model shapes: apples-to-oranges with the standalone's square-cubed measurement, which created the illusion the from-scratch kernel was ahead.
- **The occupancy "tax" is real but not a cost.** Prod runs at 21 % occupancy (1 block/SM, 151 regs), exactly the predicted tax, yet converts it into higher SM-pipe utilisation (71 % vs 54 %). Neither kernel is L2/DRAM-bound at square shapes (prod DRAM 11.7 %), so the standalone's CTA-tile-major + column-interleave L2 tricks buy nothing here.
- **Small-M prefill (M ≤ 512) is inefficient for both** (12 → 46 %): pure grid underfill, a 128×128 M-tile leaves most of the 170 SMs idle. The fix is smaller M-tiles / split-K (imp's `gemm_grouped_nvfp4_smallM` path), **not** occupancy; the cp.async standalone uses the same 128×128 tile and underfills identically.

**Verdict:** keep CUTLASS TMA + warp-spec as the production dense NVFP4 GEMM; do not port cp.async + layout into prod. The standalone remains a reference and `imp-bench nvfp4` a clean per-kernel ncu target.
