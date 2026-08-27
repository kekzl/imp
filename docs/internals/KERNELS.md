<!--
layer: L2
audience: kernel-devs
verified: 2026-08-28
commit: be825e4a
-->

# The Optimal sm_120a Attention Kernel

Canonical design reference for the imp hot-path attention kernel on RTX 5090 (GB202, **sm_120a**, consumer Blackwell). The spec future FA2 levers must measure themselves against; grounded in the profiling ground-truth and empirical refutations through 2026-06, not in datacenter (B200 / FA4) assumptions.

Companion docs: [`sm120.md`](SM120.md) (kernel notes), [`performance.md`](../PERF.md) (baselines + methodology). Hot-path source, both in `src/compute/attention_fmha_sm120.cu`: `fmha_sm120_fa2_kernel` (primary register-resident FA2, dispatched by `fmha_sm120_fa2_prefill`) and `fmha_sm120_kernel` (slow tiled-FMHA **fallback**, not an optimization target). The FA2 kernel is templated on `<Bq, HD, FP16QK, F16ACC, BKV, TWOSLOT, PVF16, …>`; the dispatcher bands the tile config by grid-fill (`blocks_128` vs `sm_count`).

---

## 1. Design thesis: what are we optimizing against?

Profiling ground-truth (issue #597, post-#609): FA2 is **tensor-pipe busiest at 52.8 %, occupancy smem-capped at 16.7 %, 0.75 waves → wait-latency-limited**, flat SOL < 37 % across all units. That is an **instruction-mix + dependency-chain** signature, NOT a bandwidth gap and NOT an occupancy gap. Four attack points, in priority order:

1. **f32-accumulate in QK^T** runs at **¼ TC rate** → the single largest compute loss.
2. **Softmax (exp / max / rescale) on the critical path** between the QK store and the PV load serializes the tensor pipe.
3. **Synchronous K/V loads** stall the MMA. With smem-capped 1-block/SM, occupancy *cannot* hide that latency; software pipelining must.
4. **`O_acc` in shared memory** eats the budget needed for larger tiles / deeper async rings.

## 2. Full spec (hd=128, NVFP4 model, long context)

### Tiling & occupancy
- **Bq = 128, Bkv = 64**, 8 warps / 256 threads, **1 block/SM** (smem-capped).
- `__launch_bounds__(256, 1)`: correct for an SMEM-limited kernel, allows maximum register allocation (the documented FMHA exception to the no-`__launch_bounds__` rule). Do **not** write `,2`: at hd=128 the smem budget never admits 2 blocks/SM, so the hint only costs register headroom.

### Shared-memory budget (target ≤ 99 KB optin; query `sharedMemPerBlockOptin`)
```
Q_tile      half[128 × 128]      = 32 KB   (loaded once, kernel-resident)
K/V ring    half[2 × 64 × 128]   = 32 KB   (2-3-stage cp.async double-buffer)
S/P overlay float[128 × 64]      = 32 KB   (f32 scores; half-P aliases the bytes)
row_m,row_l float[2 × 128]       ≈  1 KB
                                  ─────────
                                   ~97 KB → fits, exactly 1 block/SM
O_acc       → REGISTERS, not smem (0 KB)
```
**The lever that finances Bq=128:** `O_acc` leaves shared memory and lives as **MMA accumulator fragments in registers**, held by each warp across the *entire* KV loop (true FA2 register-resident). The tiled fallback keeps `O_acc` as a `float[Bq×HD]` smem block, which is why it cannot fit Bq=128 in 99 KB.

**What the fallback actually picks** (#1679). Its first three branches compare against `max_smem / 2` (two blocks per SM beat a bigger tile at one), so the selection is against 50688 bytes, not 101376. Measured on this device (`cudaDevAttrMaxSharedMemoryPerBlockOptin` = 101376) and computed from `compute_smem_sm120`:

| HD | Bkv | Bq | smem | branch |
|---|---|---|---|---|
| 64 | 64 | **64** | 48.5 KB | fits `occ2_cap` (Bq=128 would be 89.0 KB) |
| 96 | 64 | **32** | 38.2 KB | fits `occ2_cap` (Bq=64 would be 64.5 KB) |
| 128 | 64 | **32** | 48.2 KB | fits `occ2_cap` (Bq=64 would be 81.0 KB) |
| 256 | 64 | 32 | 88.2 KB | `max_smem` only, occupancy 1 |
| 512 | 32 | 16 | 82.1 KB | `max_smem` only, occupancy 1 |

The three bold rows are the ones an earlier version of this section and the selector's own comment both got wrong: they named the Bq that fits the **full** limit, not the one the code takes.

### MMA: dual-precision, both f16-accumulate
- **QK^T: `mma.sync.m16n8k16.f16.f16.f16.f16`** (f16 accumulator). Online-softmax subtracts the row max, so f16 dynamic range on the scores is safe. Cost +0.37 % PPL (the `attention.fa2_f16acc` knob). This is the **¼-rate → full-rate** jump.
- **PV: also f16-accumulate** (`mma.sync.m16n8k16`, default-on since PR #674; the "O sum needs f32 range" objection was refuted).

### Async pipeline (the missing overlap)
- Synchronous `float4` copies → a **3-stage `cp.async.cg.shared.global` 16-byte ring**. Producer lanes prefetch K tile *j+1* and V tile *j* while consumers run QK/PV on tile *j*. `commit_group` / `wait_group(N-1)` + `__syncthreads()` before any smem read.
- **Rationale:** at 1 block/SM (0.75 waves) more occupancy cannot hide GDDR7 latency; deep software pipelining (more in-flight async per wave) is the only correct response to smem-capped occupancy.

### Take softmax off the tensor-pipe critical chain
- `exp` via **`ex2.approx.f32` on the MUFU/SFU pipe**, parallel to the tensor pipe. While warp A runs PV-MMA for tile *j*, warp B computes the softmax for tile *j+1*. **No forced producer/consumer specialization**: the cross-tile pipeline (both combined and split-K/V-prefetch variants) **regressed +9 % / +15 %**; warps deliver phase diversity themselves.
- Keep running max/sum in register lanes; the **O rescale (`O *= α`) is a register op on the accumulator fragments**, not a smem read-modify-write.

### The NVFP4 precision boundary (the honest line)
QK and PV stay **f16**. The `mxf4nvf4.block_scale` MMA (k=64, 2.6× raw) requires Q/K in NVFP4 → the **format-intrinsic quality cliff** (e4m3-QK PPL 5722 vs 6.12, #511 - 3 mantissa bits × 36-layer compounding). **FP4 MMA is the weapon for the projection GEMMs** (q/k/v/o_proj, FFN), **not** for QK^T/PV *inside* attention. Attention math wants f16 mantissa; only the linear layers may go FP4.

## 3. Steady-state pipeline (per KV tile)

```
Tensor pipe:   [QK mma j ][ PV mma j-1      ][QK mma j+1]   ← never idle
SFU pipe:               [exp/max softmax j  ]               ← parallel, hidden
Async copy:    [cp.async K_{j+1}, V_j  ......]              ← hidden behind mma
Barriers:      1× __syncthreads / KV tile (cp.async.wait)
```

## 4. Lever status - the honest punchline

| Lever | Status | Evidence |
|-------|--------|----------|
| f16-acc QK^T + PV | **shipped** | `fa2_f16acc` / #674, +3-4 % pp, +0.37 % PPL |
| Register-resident O | **shipped** | primary FA2 (`mxf4nvf4_sm120.h`) |
| cp.async K/V double-buffer | **shipped** | −11.6 % kernel, long ctx |
| Smem row-stride padding | **shipped** | 1.54× kernel, PR #484 |
| Sawtooth L2 locality | **shipped** | in the fallback source today |
| Deeper async ring / cross-tile pipe | **REFUTED** | both variants **+9 % / +15 % regression** - phase-chain hypothesis false |
| Bq=128 / 2-CTA / occupancy push | **REFUTED** | reg-squeeze succeeded (16.5→30.6 %) but dense **+11 % regression**, SOL stayed flat |
| FP4-QK inside attention | **REFUTED** | #511 PPL 5722, format-intrinsic |

**Punchline:** the optimal sm_120 attention kernel and imp's primary FA2 have **converged**. Every un-refuted lever is in; every remaining design move (deeper pipeline, more occupancy, FP4-QK) is empirically refuted. The 52 %→100 % roofline gap is **architecture, not implementation debt.**

> No open re-litigation. The register-resident-O + Q-in-registers + **Bq=128** config is already the *shipped* large-seq path in `fmha_sm120_fa2_prefill` (selected when `blocks_128 >= sm_count`), with f16-acc QK^T and PV-f16-acc as config-gated variants on top. The Bq=128 path explicitly forgoes 2-CTA residency in favor of deeper cp.async overlap at Bkv=64 (comment at the top of `fmha_sm120_fa2_prefill`); the underfill band (`sm_count/2 <= blocks_128 < sm_count`) deliberately drops to **Bq=64 + TWOSLOT** to put 2 CTAs/SM resident where the grid would underfill the 170 SMs. The refuted levers were measured against this exact kernel family (#597 / #648 / #653 / #674); closed, not pending.

## 5. The wall (silicon, not code)

What would take the kernel from ~52 % to ~100 % roofline, we **cannot build**:

- **No `tcgen05` / async MMA** → the MMA always blocks the issuing warp. `cp.async` + warp diversity can *emulate* an FA4-style pipeline but never hide the MMA itself behind async.
- **No TMEM / TMA-WS** → no producer-warpgroup-TMA-into-mbarrier-ring + consumer-warpgroup-MMA-into-TMEM + softmax-warpgroup-on-the-side (FA4 on B200).
- **FP4 mma.sync = ½ datasheet, f32-acc = ¼** → the TC peak is half the marketing number.

On a B200 the optimal kernel is FA4 with three *hardware* async pipelines. On sm_120a it is **register-resident FA2 with f16-QK + cp.async double-buffer + SFU-overlapped softmax**, which *is* imp's primary FA2. We lose to datacenter Blackwell on architecture, not implementation; we win decode (uncontested NVFP4) and MoE-pp2048.

## 6. Decode counterpart (one paragraph - HBM-bound)

The optimal decode "kernel" is **traffic elimination**, not a compute design: NVFP4 GEMVs at the GDDR7 ceiling (~1.5 GB/ms = 86 % datasheet) + a conditional CUDA graph + PDL. Built, at the limit; the only remaining wall-breaker is algorithmic (speculation), not kernel-technical. See the decode levers in `MEMORY.md` and `docs/performance.md`.

## 7. Runnable companions + the GEMM addendum

Two self-contained, bit-exact reference kernels (one file each, no imp/CUTLASS deps) in `tools/standalone/`, built profiling-driven from scratch:

- **`fa2_sm120a_optimal.cu`**: the attention kernel specced above, runnable. 50 → 114 / 158 / 187 TFLOP/s (S=4k/8k/16k) ≈ 90 % of a dedicated speed-of-light FA effort; the residual gap is silicon (no tcgen05/TMEM), per §5.
- **`gemm_nvfp4_sm120a.cu`**: NVFP4 GEMM on the peak `mxf4nvf4` block-scaled mma (m16n8k64) with real per-16 UE4M3 scales. 1.8 → 807 / 972 TFLOP/s (4k/8k) = **48 % of the measured FP4 peak, beating the production CUTLASS path (~41 %)**. (`gemm_nvfp4_sm120a_tma.cu` is the documented TMA+warp-spec negative result.)

**The transferable lesson: re-diagnose before you optimize.** The GEMM read "L2-bound" at 82 %; every textbook lever (threadblock swizzle, 3-stage pipeline, TMA, warp-specialization) FAILED because the diagnosis was wrong. ncu showed L2 *requests* at 82 % but *sectors* at only 44 %: **request-rate-bound, not bandwidth-bound**. The fix was two layout changes:
1. **CTA-tile-major packing**: a tile's rows were 2 KB apart, each 32 B row in its own 128 B L2 line at 25 % fill; storing each CTA tile contiguous gives full lines → L2 requests 82 → 41 % (+26-31 %).
2. **Column-interleave** so a fragment pair {col T0, col T0+4} is adjacent and reads as one `uint2` → mio_throttle 5.77 → 2.74 (+1.5-7.6 %).

Questioning the metric beat porting CUTLASS's machinery by ~40 %. The headers of both kernels carry the full ncu trail.

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
