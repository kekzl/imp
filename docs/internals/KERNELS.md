<!--
layer: L2
audience: kernel-devs
verified: 2026-09-22
commit: 9cbb8004
-->

# The optimal sm_120a attention kernel

Canonical design reference for imp's hot-path attention kernel on RTX 5090 (GB202, **sm_120a**, consumer Blackwell). Grounded in profiling ground-truth and empirical refutations through 2026-06; not datacenter (B200/FA4) assumptions. Companion docs: [`SM120.md`](SM120.md) (kernel notes), [`../PERF.md`](../PERF.md) (baselines/methodology). Refuted design moves and the full occupancy-tax investigation: [`docs/archive/kernels_refuted_2026.md`](../archive/kernels_refuted_2026.md).

## Kernel table

| Kernel | File | Role | Status |
|---|---|---|---|
| `fmha_sm120_fa2_kernel` | `src/compute/attention_fmha_sm120.cu` | primary register-resident FA2 attention prefill, dispatched by `fmha_sm120_fa2_prefill` | shipped, optimization target |
| `fmha_sm120_kernel` | `src/compute/attention_fmha_sm120.cu` | tiled-FMHA fallback | shipped, not an optimization target |
| grouped-NVFP4 CUTLASS GEMM (`KernelTmaWarpSpecializedCooperativeBlockScaledSm120`) | `src/compute/gemm_cutlass_sm120.cu` | production dense NVFP4 GEMM (projections, FFN) | shipped |
| `gemm_grouped_nvfp4_smallM` | `src/compute/gemm_grouped_nvfp4_smallM.cu` | small-M / split-K grouped NVFP4 GEMM (MoE prefill) | shipped |
| decode FFN GEMV | `src/exec/executor_ffn.cu` | dp4a / mma.sync / NVFP4 GEMV, HBM-bound at the GDDR7 ceiling; decode levers: [`MEMORY.md`](MEMORY.md) | shipped |
| `fa2_sm120a_optimal.cu` | `tools/standalone/fa2_sm120a_optimal.cu` | bit-exact standalone FA2 reference, no imp/CUTLASS deps | reference only, not wired into imp |
| `gemm_nvfp4_sm120a.cu` | `tools/standalone/gemm_nvfp4_sm120a.cu` | standalone NVFP4 GEMM, cp.async + L2-layout tricks | reference only; production keeps CUTLASS TMA+warp-spec instead (archive) |
| `gemm_nvfp4_sm120a_tma.cu` | `tools/standalone/gemm_nvfp4_sm120a_tma.cu` | TMA + warp-specialization on the standalone kernel | documented negative result |

Paged attention decode kernels (per KV dtype): [`ATTENTION_DISPATCH.md`](ATTENTION_DISPATCH.md).

## 1. Design thesis

Profiling ground-truth (#597, post-#609): FA2 is tensor-pipe busiest at 52.8 %, occupancy smem-capped at 16.7 %, 0.75 waves -> wait-latency-limited, flat SOL < 37 % across all units. An **instruction-mix + dependency-chain** signature, not a bandwidth or occupancy gap. Attack points, priority order:

1. f32-accumulate in QK^T runs at 1/4 TC rate - the single largest compute loss.
2. Softmax (exp/max/rescale) on the critical path between the QK store and the PV load serializes the tensor pipe.
3. Synchronous K/V loads stall the MMA; at smem-capped 1 block/SM, only software pipelining hides that latency.
4. `O_acc` in shared memory eats the budget needed for larger tiles / deeper async rings.

## 2. Full spec (hd=128, NVFP4 model, long context)

**Tiling:** Bq=128, Bkv=64, 8 warps / 256 threads, 1 block/SM (smem-capped). `__launch_bounds__(256, 1)` - correct for an SMEM-limited kernel (documented exception to the no-`__launch_bounds__` rule); `,2` only costs register headroom since the smem budget never admits 2 blocks/SM at hd=128.

**Shared-memory budget** (target <= 99 KB optin, `sharedMemPerBlockOptin`):

```
Q_tile      half[128 x 128]      = 32 KB   (loaded once, kernel-resident)
K/V ring    half[2 x 64 x 128]   = 32 KB   (2-3-stage cp.async double-buffer)
S/P overlay float[128 x 64]      = 32 KB   (f32 scores; half-P aliases the bytes)
row_m,row_l float[2 x 128]       ~  1 KB
                                  -----
                                   ~97 KB -> fits, exactly 1 block/SM
O_acc       -> REGISTERS, not smem (0 KB)
```

`O_acc` lives as MMA accumulator fragments in registers, held by each warp across the entire KV loop (true FA2 register-resident) - this is what finances Bq=128; the tiled fallback keeps `O_acc` as a `float[Bq×HD]` smem block and cannot fit Bq=128 in 99 KB.

**Fallback tile selection** (#1679, `compute_smem_sm120`, device `cudaDevAttrMaxSharedMemoryPerBlockOptin` = 101376): the first three branches compare against `max_smem / 2` (two blocks/SM beat a bigger tile at one), i.e. against 50688 bytes, not 101376.

| HD | Bkv | Bq taken | smem | branch |
|---|---|---|---|---|
| 64 | 64 | 64 | 48.5 KB | fits `occ2_cap` (Bq=128 would be 89.0 KB) |
| 96 | 64 | 32 | 38.2 KB | fits `occ2_cap` (Bq=64 would be 64.5 KB) |
| 128 | 64 | 32 | 48.2 KB | fits `occ2_cap` (Bq=64 would be 81.0 KB) |
| 256 | 64 | 32 | 88.2 KB | `max_smem` only, occupancy 1 |
| 512 | 32 | 16 | 82.1 KB | `max_smem` only, occupancy 1 |

**MMA, dual-precision, both f16-accumulate:** QK^T uses `mma.sync.m16n8k16.f16.f16.f16.f16` (f16 accumulator; online-softmax subtracts the row max so f16 score range is safe; +0.37 % PPL via `attention.fa2_f16acc`) - the 1/4-rate to full-rate jump. PV also f16-accumulate (default-on since #674; the "O sum needs f32 range" objection was refuted).

**Async pipeline:** synchronous `float4` copies replaced by a 3-stage `cp.async.cg.shared.global` 16-byte ring; producer lanes prefetch K tile j+1 and V tile j while consumers run QK/PV on tile j (`commit_group`/`wait_group(N-1)` + `__syncthreads()` before any smem read). At 1 block/SM (0.75 waves) more occupancy cannot hide GDDR7 latency, so deeper software pipelining is the only correct response.

**Softmax off the critical chain:** `exp` via `ex2.approx.f32` on the MUFU/SFU pipe, parallel to the tensor pipe - warp A runs PV-MMA for tile j while warp B computes softmax for tile j+1. No forced producer/consumer specialization: both cross-tile pipeline variants regressed (+9 % / +15 %, archive); warps deliver phase diversity on their own. Running max/sum stay in register lanes; `O *= α` is a register op on the accumulator fragments.

**NVFP4 precision boundary:** QK and PV stay f16. `mxf4nvf4.block_scale` (k=64, 2.6x raw) needs Q/K in NVFP4, which hits a format-intrinsic quality cliff (e4m3-QK PPL 5722 vs 6.12, #511 - 3 mantissa bits x 36-layer compounding). FP4 MMA is the lever for the projection GEMMs (q/k/v/o_proj, FFN), not for QK^T/PV inside attention.

## 3. Steady-state pipeline (per KV tile)

```
Tensor pipe:   [QK mma j ][ PV mma j-1      ][QK mma j+1]   <- never idle
SFU pipe:               [exp/max softmax j  ]               <- parallel, hidden
Async copy:    [cp.async K_{j+1}, V_j  ......]              <- hidden behind mma
Barriers:      1x __syncthreads / KV tile (cp.async.wait)
```

## 4. Lever status

| Lever | Status | Evidence |
|---|---|---|
| f16-acc QK^T + PV | shipped | `fa2_f16acc` / #674, +3-4 % pp, +0.37 % PPL |
| Register-resident O | shipped | primary FA2 (`mxf4nvf4_sm120.h`) |
| cp.async K/V double-buffer | shipped | -11.6 % kernel, long ctx |
| Smem row-stride padding | shipped | 1.54x kernel, #484 |
| Sawtooth L2 locality | shipped | in the fallback source today |
| Deeper async ring / cross-tile pipe | REFUTED | both variants +9 % / +15 % regression |
| Bq=128 / 2-CTA / occupancy push | REFUTED | reg-squeeze succeeded (16.5->30.6 %) but +11 % regression, SOL flat |
| FP4-QK inside attention | REFUTED | #511 PPL 5722, format-intrinsic |
| CUTLASS TMA+warp-spec occupancy tax on prod GEMM | REFUTED | prod wins at peak shapes despite lower occupancy; full detail in archive |

The register-resident-O + Q-in-registers + Bq=128 config is the shipped large-seq path in `fmha_sm120_fa2_prefill` (selected when `blocks_128 >= sm_count`); the underfill band (`sm_count/2 <= blocks_128 < sm_count`) drops to Bq=64 + TWOSLOT to put 2 CTAs/SM resident where the grid would underfill the 170 SMs. Refuted levers were measured against this exact kernel family (#597/#648/#653/#674).

## 5. The wall (silicon, not code)

What `sm_120a` has and lacks against datacenter Blackwell, and why: [`ARCHITECTURE.md#target-architecture`](ARCHITECTURE.md#target-architecture).

| Consequence for this kernel | Detail |
|---|---|
| The MMA always blocks the issuing warp | `cp.async` + warp diversity can emulate an FA4-style pipeline but never hide the MMA itself behind async |
| No async-accumulator producer/consumer split | no producer-warpgroup-load-into-ring + consumer-warpgroup-MMA-into-accumulator-memory + softmax-warpgroup-on-the-side (FA4 on B200) |
| FP4 `mma.sync` = 1/2 datasheet, f32-acc = 1/4 | the TC peak is half the marketing number |

On a B200 the optimal kernel is FA4 with three hardware async pipelines. On `sm_120a` it is register-resident FA2 with f16-QK + cp.async double-buffer + SFU-overlapped softmax, which is imp's primary FA2 - architecture-limited relative to datacenter Blackwell, not an implementation gap.

## 6. Runnable companions

Two self-contained, bit-exact reference kernels (`tools/standalone/`, one file each, no imp/CUTLASS deps):

| Kernel | Result | Lesson |
|---|---|---|
| `fa2_sm120a_optimal.cu` | 50 -> 114/158/187 TFLOP/s (S=4k/8k/16k), ~90 % of a dedicated speed-of-light FA effort; residual gap is silicon, section 5 | - |
| `gemm_nvfp4_sm120a.cu` | 1.8 -> 807/972 TFLOP/s (4k/8k), 48 % of measured FP4 peak, beat production CUTLASS's roofline-pipeline number (~41 %) at the time | apples-to-oranges: prod's ~41 % was measured on small-M model shapes, not the standalone's square-cubed benchmark; re-measured head-to-head in the archive |

The GEMM's initial "L2-bound" diagnosis (82 % L2 *requests*, only 44 % *sectors*: request-rate-bound, not bandwidth-bound) drove two fixes: CTA-tile-major packing (full 128 B L2 lines, requests 82 -> 41 %) and column-interleave (`uint2` fragment reads, mio_throttle 5.77 -> 2.74). Full ncu trail in both kernels' headers; the occupancy-tax hypothesis this result suggested was tested head-to-head against production and refuted - [`docs/archive/kernels_refuted_2026.md`](../archive/kernels_refuted_2026.md).
