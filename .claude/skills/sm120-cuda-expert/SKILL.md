---
name: sm120-cuda-expert
description: Use when writing, reviewing, or optimizing CUDA kernels targeting sm_120a (RTX 5090 / Consumer Blackwell, GB202) in the imp inference engine. Triggers on CUDA/PTX kernel code, shared-memory layout, bank conflicts, tensor-core MMA (mxf4nvf4, tf32, 3xTF32/3xFP16), GEMV/GEMM/attention/quantization/GDN-scan kernels, decode tok/s under expected, kernel emitting HMMA instead of mxf4nvf4, occupancy, register pressure, __launch_bounds__, spills, PDL. Pair with `benchmark-cuda` for measurement and `check-degeneration` after hot-path changes.
---

# sm_120 CUDA Kernel Expert - imp

PTX templates: `references/ptx-patterns.md`. Dead ends, version gotchas, load-bearing fixes: `references/known-issues.md` (read BEFORE proposing a lever on a mature path).

## Architecture

| Spec | Value |
|---|---|
| SMs / CUDA cores / TC | 170 / 21,760 / 680 (4 per SM, 5th gen) |
| L1+SMEM per SM | 128 KB; opt-in shared ~99 KB per block (`cudaDeviceProp::sharedMemPerBlockOptin`), not H100's 228 KB |
| L2 | 96 MB (every single Qwen3.8 weight fits: isolated benches read L2) |
| VRAM | 32 GB GDDR7, 1792 GB/s datasheet; 1628 GB/s measured resident, ~237 spilled |
| MMA shapes | FP16 `m16n8k16`, TF32 `m16n8k8`, FP8 `m16n8k32`, FP4 block-scaled `m16n8k64` |
| Measured TC rates | FP4 `mma.sync` 2019 TOPS (~1/2 datasheet); f32-accumulate 1/4 rate; GeForce TF32 = 1/2 the FP16-fp32acc rate (253 TFLOPS) |
| Registers | 255 per thread is the DEVICE attribute; a kernel "at 255" must be read from ptxas (`make kernel-resources`), the dense FA2 instance allocates 144 |

No `tcgen05` / TMEM / `wgmma` / 2-CTA cluster MMA (SM100 only). Peak path = register `mma.sync` with block scaling, FA2-style. Toolkit bumps add nothing: CUDA 13.3 flipped 0 of 247 probed instructions.

## The three laws

1. **Batch=1 decode is at its ceiling; batched decode (M<=32) is its own regime.** Order at batch=1: fast per-launch GEMM (native `mxf4nvf4` smallm v2 for M<=32, `gemm.nvfp4_smallm`, #1766; CUTLASS at large M; never the `gemm_nvfp4` dequant->cuBLAS fallback), then CUDA-graph capture (`CudaGraphConditionalRunner`, `src/runtime/cuda_graph.h`), then bytes. Qwen3.8 M=1 spec-off: 95.8% busy, no single lever >=1% left (#1789); the decode graph is strictly serial there (union == sum). A per-token host round-trip drops a path out of the conditional loop (-27..-45%). At 32 streams grid-shape and launch levers PAID: row-block RMSNorm +6.8% (#1769), shared act-quantize +4.6% (#1771), producer quantize +2.6% (#1773), BF16 GDN state +12.5% (#1776/#1778), smallm sibling pairs +1.7% (#1788), row-batched sampling +2.2% (#1790), PDL device half +0.5-1.3% (#1833); residual-accumulate LOST -0.9% (#1793: the adds overlap with the next layer's GEMMs). Measure the regime with a 32-stream two-image A/B (benchmark-cuda).
2. **Occupancy is for reaching the roofline, not passing it.** <=48 regs for 100% occupancy on GEMV paths. `__launch_bounds__(T, N)` forces N blocks by SPILLING: `(128,12)` on the MR GEMV went 14.5 -> 26.8 us at MR=4; `(256,1)` on the shipped dense FA2 instance moved ptxas 137 -> 180 regs. What shipped: `(256,2)` on a SEPARATE wrapper kernel (TWOSLOT 35 KB, 137 -> 128 regs, 24 B spill, FA2 -10% at pp4096, #1843) with the production instance byte-identical in SASS. Read `--ptxas-options=-v` per template instance; the CI `kernels` gate (`tools/kernel_resource_baseline.txt`, REG >= 240 or any local frame) ratchets it. Two CTAs/SM on a tensor-pipe-bound kernel is flat (chunkpar K2: 333 vs 331 ms) or worse under latency binding (+8%: barriers double, half the work per MMA phase).
3. **Quantization type determines kernel strategy.** Q8_0: bandwidth-bound, row-parallel + smem activations. Q6_K: K-parallel + warp division. NVFP4 large M: `StorageTier::CUTLASS_NVFP4` (SfAtom layout); M<=32: smallm v2 on the PLAIN packed bytes (`h.source_data`/`h.source_scales`). Plain `StorageTier::NVFP4` at large M = slow `gemm_nvfp4`.

## Numerics rules for tensor-core rewrites

- A recurrent state path needs error compensation on EVERY link: plain tf32 on the GDN state GEMMs = state diff 3.4e-4, PPL +0.13%; on P@W it cancels (Qeff = D q - P W). 3xTF32 (a_hi + a_lo) costs +25 us; 3xFP16 `m16n8k16` (a_hi + a_lo in fp16) runs 2x the TF32 rate and shipped on all state-feeding GEMMs (#1852; K1 -22%, K2 -18%). Output-only terms (Y_A, y) stay plain tf32/fp16.
- Judge: unit-test state diff vs the fused kernel (~1e-6 = fp32-equivalent; 1e-4 fails), deterministic PPL on Qwen3.8-27B-NVFP4-vllm (fused 4.6283), `tools/analysis/layer_ab_diff.py` added divergence of the changed blocks ~0. Qwen3.6-35B PPL moves +-0.2..0.5% between fp32-equivalent kernels: no verdict there.
- Online-softmax tile levers cost precision per extra rescale: HD=256 FA2 at Bkv=32 = kernel -11%, PPL +0.53%, opt-in only (`attention.fa2_hd256_bkv=32`, #1840). Scale inside the exp FMA is free (#1844, PPL 4.6283 both arms); `exp2f` forms and scale-in-Q are +0.35% PPL for the same speed.
- FP16 intermediates on `out / row_scale` overflow to inf on small-absmax rows (FP8 SSM_IN, 2026-09-01); keep the rescale in FP32 or fuse it. Uniform-random unit-test rows cannot reach this.

## GDN chunk-parallel scan (#1847-#1852, `gdn.chunkpar_scan` default on)

`src/compute/gdn_scan_chunkpar.cu` (K1 factor kernel + launcher), `gdn_scan_chunkpar_pass.cu` (K2 state pass), `.cuh` shared. Grid (chunks x heads) for factors, (heads, 2) for the state pass; 42-126 MiB workspace (null degrades to the fused route); single-sequence prefill n >= 128, HD=SS=128. Qwen3.6-35B pp4096 e2e 12.9k -> 31.0k tok/s vs the 32-CTA fused scan (42% of the hybrid pp512 wall before).

| Lever | Measured |
|---|---|
| Solve histories in GLOBAL memory | K1 379 us; in smem (aliased over dead K/Q staging) + 256 threads: 196 us |
| K2 scalar FMA from smem | 628 us (= the sequential scan); float4 staging + column split: 242 us |
| Accumulator splits / 2-t unroll in K2 | REGRESS 242 -> 295/309 us (registers cost the 2-CTA residency); in K1 four partial accumulators pay |
| Blockwise triangular solve (16-row diagonal blocks in registers, off-diagonal 3xTF32) | 128 -> 8 barriers per chunk, K1 75 -> 49.5 us/CTA (#1850) |
| K2 8 warps + register-pipelined staging | K2 -30% (#1851) |
| Strip per head count under an L2 cap (`gdn.chunkpar_strip=0` auto: factor set <= 2/3 L2) | strip 14-16 wins K1 -21% but K2 +11..13% (126 MB factor set > 96 MB L2 reads DRAM); net 0 |
| XOR swizzle on stride-128 float4 tiles `chunk ^ ((row & 7) << 1)` | K1 bank conflicts 64% -> 25%; k16 fragment rows (2tg/2tg+1) need swz128 staging + `SH = COLS + 4` |
| K2 half staging (RS=32, 2 CTAs/SM) | flat under TF32 binding, +8% under FP16 binding: refuted twice |

Traps: `gdn_scan_chunkpar.cu` sat at exactly 600 code LOC (kernel hard threshold) after #1850, hence the split; nvcc 13.3 segfaults on a generic lambda with a `std::true_type` tag inside a kernel (use a free template function); dumps for `layer_ab_diff.py` are 3.6 GB per arm at 300 tokens.

## Paths that must stay active (check before optimizing)

| Path | Evidence at init / config |
|---|---|
| CUTLASS NVFP4 weight cache | `cutlass_nvfp4 weight cache` log line; without it decode runs dequant->cuBLAS |
| CUDA-graph decode loop, graph prewarm | `runtime.cuda_graphs`, `runtime.graph_prewarm` (32/32 captures in 2.3 s) |
| PDL both halves | `runtime.no_pdl=false`; registered kernels call `griddepcontrol.wait` before the first global access (`src/compute/pdl_device.cuh`); registered = waits, blanket registrations without a wait RACED (`GreedyDeterminism`) |
| FA2 prefill family | `attention.fmha_fa2`, `fa2_hd256`, `fa2_fp16qk`, `fa2_dense_2cta` (all on); `fa2_hd256_bkv` 64 |
| Stream-K on the CUTLASS prefill GEMM | `gemm.nvfp4_cutlass_streamk=1`: N > 2048, >= 1 wave, last wave <= half full; workspace sized as the max over every 128-tile grid (~22 MB), a 0 B workspace refuses EVERY launch (22k -> 4.2k tok/s); the SK-typed kernel in data-parallel mode is slower (109 vs 100 us) so shapes outside the rule keep the plain kernel |
| Batched-decode sidecars | `gemm.nvfp4_smallm` + `_impl=2` + `_pair`, `gdn.state_bf16` (+0.21% PPL by design), `gemm.fp8_ssm_proj`, `gemm.nvfp4_lm_head*`, `gemm.fp8_attn_proj` |
| Chunk-parallel GDN prefill | `gdn.chunkpar_scan`, `gdn.chunkpar_strip` |
| INT8-IMMA prefill for GGUF MoE | dense Q4_K IMMA loses to cuBLAS and is off by design |

FP8 prefill stays DISABLED on sm_120 (cuBLAS `NOT_SUPPORTED` at non-aligned M, `src/runtime/engine_init_resolver.cpp`); FP8 GDN-projection prefill re-measured 2026-09-01: class is 12-13% of the steady-state sum, ceiling +5%, inside the cuBLAS band, REFUTED (record `docs/plans/2026-08-31-fp8-ssm-prefill.md`). Prefill levers here: FA2 family, ragged prefill batching (`runtime.prefill_batch`, #1780), chunk-parallel scan.

## Closed classes (do not reopen without a new mechanism)

| Class | Verdict |
|---|---|
| M=1 NVFP4 decode GEMV | structural ceiling; occupancy raise, KPAR->MR, v2 pipeline at M=1 (#1789) all refuted |
| MoE grouped GEMM (NVFP4 prefill) | ~60% of weight floor; CUTLASS tile sweep, 32-row v2 grouped (-12.5% isolated, +3.5..+15% in situ), multi-tile CTA mt32/64/128 all refuted on real routing; the sm120 block-scaled builder rejects M=64 tiles (SF atom 128 rows) |
| FA2 occupancy at hd=128 | 16 warps/SM shipped (#1843); split-D warp pairing refuted; remaining path = in-CTA pipelining, and the kernel spent 5.5 scalar ALU per MMA in the softmax loop before #1844 |
| NVFP4 decode attention traffic sharing | GQA tile -9% (#1785), smem double buffer -3%; the lever was LOAD WIDTH (#1817: 20 `LDG.E.U8` per iteration -> word loads, +15.7%) |
| FP4-precision attention | closed 2026-07-04 (three refutations) |
| Batch=1 launch-class fusion | 0% e2e; split-K cap -21..-35%; raising split-K -2.9..-4.7% |

## Shared memory budget

| head_dim | Bq | SMEM |
|---|---|---|
| 64 | 128 | ~89 KB |
| 128 | 64 / 128 (TWOSLOT 2-CTA) | ~81 KB / 35 KB |
| 256 | 32 (Bkv=64) / Bkv=32 | ~68 KB / 35 KB |

`cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes)`. Padding two access patterns at once may not fit (chunkpar histories at stride 132 next to T/P padding did not fit 99 KB); swizzle instead.

## Compile flags

`--generate-code=arch=compute_120a,code=sm_120a` (`IMP_SM120_FLAGS`), `--expt-relaxed-constexpr --extended-lambda`, release `-O3 --use_fast_math`. `compute_120a` unlocks `mma.sync.aligned.kind::mxf4nvf4.block_scale`, TMA-WS grouped GEMM, cluster launch, extended mbarrier, FP4 saturation. Never `sm_120`/`sm_120f`/generic `compute_120` (no block-scale, no FP8 MMA). Guard with `#if __CUDA_ARCH__ >= 1200`.

## Common mistakes

| Mistake | Fix |
|---|---|
| Targeting `sm_120` / `sm_120f` | `compute_120a/sm_120a` |
| Decode routed through `gemm_nvfp4` | promote to `StorageTier::CUTLASS_NVFP4`; check the init line |
| No graphs-ON re-bench after a kernel win | the tok/s win is graph-mediated |
| H100 SMEM assumptions | 99 KB opt-in |
| `__launch_bounds__` on regular GEMV/attention paths | -4.5..-20%; min-blocks spills to DRAM; only wrapper-kernel bounds with a SASS-identical production instance |
| Byte-pointer inner loops (`const uint8_t*` walked element-wise) | ptxas cannot merge; count `LDG.*` forms in `cuobjdump -sass` first |
| `reinterpret_cast` on Q8_0 blocks (34 B, unaligned) | `memcpy()` |
| `__noinline__` device helpers | spill to local; `__forceinline__` |
| Missing `__syncthreads()` after `cp.async wait` | race |
| Pointer advance without `sizeof(T)` | known-issues (FP8 FMHA S_tile) |
| PDL registration without `griddepcontrol.wait` | race on the producer's output; `pdl::launch` has no default args, register at the launch site (`pdl::enable_kernel`) |
| Isolated bench without an L2-defeating ring | reads L2; rotate >= 4 x 100 MB slabs, warm per shape |
| "perf-neutral" without a SASS diff | `cuobjdump -sass`; byte-identical = proof |
| tf32 on a recurrent state path checked only by MoE PPL | unit-test state diff + Qwen3.8 deterministic PPL |

## Where to look

- `src/compute/`: attention (`attention_fmha_*`, `attention_paged_nvfp4.cu`), GDN (`gdn_scan_chunkpar*.cu`, `gdn_scan_tc.cu`), `pdl_device.cuh`, sparse attention kernels.
- `src/quant/`: dequant, `nvfp4_gemm_smallm_v2.cu`, `nvfp4_pack.cuh`.
- `src/exec/`: `executor_gemm_dispatch.cu`, `executor_gemm_smallm.cu`, `executor_sampling.cu`, `executor_attention_decode.cu`, producer-fused norm/swiglu+quantize, `sparse_attn_geometry.h`.
- `src/runtime/`: `cuda_graph.h`, `pdl.h`, `engine_init_resolver.cpp`.
- Gates: `make kernel-resources` (`tools/kernel_resources.py`), `tools/check_launch_guards.py`, `tools/check_filesize.py` (kernel `.cu` hard 600 code LOC; split kernel / launcher / instantiations).
- Docs: `docs/internals/SM120.md`, `docs/internals/KERNELS.md`, `docs/internals/PROFILING.md`, `docs/PERF.md`, `docs/roadmap.md` (verdict ledgers), `tools/roofline/history/BASELINE`.
