---
name: sm120-cuda-expert
description: Use when writing, reviewing, or optimizing CUDA kernels targeting sm_120a (RTX 5090 / Consumer Blackwell, GB202) in the imp inference engine. Triggers on CUDA/PTX kernel code, shared-memory layout, tensor-core MMA, GEMV/GEMM/attention/quantization kernels, decode tok/s under expected, kernel emitting HMMA instead of mxf4nvf4, occupancy or register-pressure questions. Pair with `benchmark-cuda` for measurement and `check-degeneration` after hot-path changes.
---

# sm_120 CUDA Kernel Expert — imp Inference Engine

Optimal-kernel reference for RTX 5090 (GB202, **sm_120a** — consumer Blackwell). For PTX inline-assembly templates see `references/ptx-patterns.md`. For dead ends, version-dependent gotchas, and root-cause references see `references/known-issues.md`.

## Architecture quick reference

| Spec | Value |
|------|-------|
| SMs · CUDA cores · TC | 170 · 21,760 · 680 (4/SM, 5th gen) |
| L1/SMEM per SM | 128 KB configurable (~99 KB opt-in shared, query `cudaDeviceProp::sharedMemPerBlockOptin`) |
| L2 cache | 96 MB unified |
| VRAM | 32 GB GDDR7, 1,792 GB/s |
| Boost clock | 2,407 MHz |
| FP4 / FP8 / FP16 TC | 3,354 TOPS / 1,677 TFLOPS / 838 TFLOPS |
| Native MMA shapes | FP16 `m16n8k16`, FP8 `m16n8k32`, FP4 block-scaled `m16n8k64` |

**sm_120 has NO `tcgen05` / TMEM / `wgmma` / 2-CTA cluster MMA** — those are SM100 (B200) only. Peak path is register-based `mma.sync` with block-scaling, FA2-style.

**TC-rate calibration (2026-06-07):** datasheet TOPS are NOT what `mma.sync` reaches. Measured FP4 `mma.sync` peak ≈ **2,019 TOPS (~½ datasheet)**, and **f32-accumulate runs at ¼ rate** — accumulate in f16 wherever PPL allows (`gemm.cublas_fp16_acc=auto` since PR #611, per-arch deny for Gemma-3/4 + gpt-oss; FA2 `attention.fa2_f16acc` opt-in). Use measured rates for roofline math, not the datasheet.

## The three laws

1. **Decode at batch=1 is launch-overhead-bound first, memory-bound second.** With ~80–120 launches per layer, per-launch µs costs dominate once GEMM is fast. Order of operations: (a) make per-launch GEMM fast (CUTLASS sm_120 NVFP4 fast-path, not slow `gemm_nvfp4` dequant→cuBLAS fallback); (b) capture decode in CUDA Graph (the async graph decode loop — `CudaGraphConditionalRunner` in `src/runtime/cuda_graph.h`, launched by `engine_graph_decode.cpp`); (c) only then chase memory traffic. A faster kernel alone shows little tok/s gain — but it *enables* graph capture to win because launches no longer hide behind GEMM. Always re-bench graphs ON after a hot-path patch. **Corollary: any per-token host round-trip silently drops a path out of the conditional loop and costs −27–45% decode** — NVFP4 think models bypassed it until PR #649 (+45%), constrained json/schema decode until the ConstrainedPipeline (PR #651: enqueue forward N+1 before the host FSM advances, json_schema 102→235 tok/s); logprobs still does a 600 KB D2H per token (open lever: device-side top-k).

2. **Occupancy is king — for getting kernels to the roofline, not past it.** Keep registers ≤48/thread for 100% occupancy. **Don't add `__launch_bounds__`** on regular GEMV/attention paths — overrides cost -4.5% to -20%. Two known correct exceptions: HD=128 GDN kernel needs `(HD,1)` to avoid a miscompile; FMHA `(256,1)` is correct on SMEM-limited kernels (~69 KB → 1 block/SM anyway, allows max register allocation). Caveat: the mature NVFP4 decode path already sits at its ceiling — full nsys+ncu sweep (2026-05-30, Qwen3-14B NVFP4) showed decode = 87% NVFP4 GEMVs at 66–70% HBM with the plateau co-limited by 4-bit dequant (L1TEX 91%); **raising occupancy further and KPAR→MR rerouting are refuted levers there** — don't re-pursue.

3. **Quantization type determines kernel strategy.** Q8_0 (simple dequant) → bandwidth-bound → row-parallel + smem-cached activations. Q6_K (complex dequant) → compute-influenced → K-parallel + warp-level work division. NVFP4 prequant → must hit `StorageTier::CUTLASS_NVFP4` (SfAtom layout) for the sm_120a fast path; plain `StorageTier::NVFP4` falls through to slow `gemm_nvfp4`. No universal "best" GEMV.

## What works (top hits)

| Technique | Gain | Detail |
|-----------|------|--------|
| **CUDA Graph decode (conditional graph + PDL)** | **+95–376% decode** | `CudaGraphConditionalRunner`; `max_steps` sized per request. Biggest gain on small-GEMM models. Requires CUTLASS NVFP4 fast-path active first. |
| CUTLASS sm_120a NVFP4 weight cache | enables graph win | `StorageTier::CUTLASS_NVFP4` (`src/core/storage_tier.h`); plain `StorageTier::NVFP4` = slowpath. |
| `mma.sync.kind::mxf4nvf4.block_scale` | 2.6× raw MMA over f8f6f4 | k=64 vs k=32; HW applies UE4M3 scale inside MMA. Needs `compute_120a` (see Compile flags). |
| FA2 register-resident prefill (`attention.fmha_fa2`, default on) | +13–19% long-ctx NVFP4 prefill | S/P/O in registers, 1 barrier/KV tile; declines safely for hd≠128. |
| FP16-QK FA2 for short prefill (`attention.fa2_fp16qk`, default on) | +25–35% pp512 NVFP4 | QK^T in f16 mma — avoids the short-seq e4m3 quality cliff (#511/#512); declined configs fall back to cuBLAS. |
| FA2 smem row-stride padding | 1.54× FA2 kernel, +27% pp16384 | head_dim=128 stride aliased all 32 banks (PR #484). ("Post-fix LSU-bound" was later corrected — see next row.) |
| FA2 cp.async K/V double-buffer | −11.6% FA2 kernel long-ctx | prefetch tile j+1 while j computes |
| FA2 f16-acc QK^T (`attention.fa2_f16acc`, opt-in) | +3–4% pp2048/4096, +0.37% PPL | PR #643. Profiling correction (#597): post-#609 FA2 is NOT LSU-bound — tensor pipe is busiest (52.8%), occupancy smem-capped at 16.7%, wait-latency-limited at 0.75 waves. **Don't re-pursue Bq=64 / occupancy without smem surgery.** |
| FA2 Bq=64/Bkv=32 variant in the grid-underfill band | kernel −6.7% pp512 / −2.9% pp4096 | PR #648: 2 CTAs/SM where the grid underfills. |
| INT8-IMMA prefill GEMM family (default on) | 30B-MoE GGUF prefill gap 2.4×→1.05× and ABOVE llama.cpp; gemma-4 +111%; Q8 pp512 ≈12.1k | PRs #612–#617: fused dequant on int8 tensor cores. Q8_0 (`gemm.q8_imma_enabled`), Q4_K raw-read, Q6_K half-MMA split, Q5_1 raw-read, N-tail support, `gemm.moe_imma_prefill`. Note `gemm.q4k_imma_prefill` default OFF for dense — **dense-IMMA loses to cuBLAS**; the win is MoE prefill. |
| cuBLAS f16-accumulate prefill (`gemm.cublas_fp16_acc=auto`) | +17% Q8 pp512 | PR #611 default-on via per-arch auto (denied: Gemma-3/4, gpt-oss — PPL). |
| Warp-per-row FP16 RMSNorm for batch prefill | shipped | PR #620 (#602). |
| NVFP4 lm_head (`gemm.nvfp4_lm_head`, `…_gdn` default ON) | +8–16% dense, +11.4% Qwen3.6 decode | BF16 lm_head quantized to NVFP4 at load; costs +2.2% PPL (owner-accepted). |
| GGUF `gemm.nvfp4_ssm_proj` (opt-in) | +53% GGUF-hybrid decode | reverses the −31% GGUF Qwen3.6 loss vs llama.cpp; `nvfp4_attn_proj` +3.8% Nemotron. **NVFP4 on GDN in/out projections REGRESSES −9 to −20% — dead end.** |
| NVFP4 prmt register LUT | +4.7–16% | `prmt.b32` replaces SMEM LUT |
| Inline Q8_1 in O-projection | +5–10% | Eliminates 1 launch + 1 DRAM round-trip |
| `__ldcs` on KV cache reads | +2–4% decode | Bypass L1, evict-first L2. **KV only, NOT weights.** |
| 8-warp Blackwell attention | better SM util | 256 threads vs Hopper's 128 |

**FP8 prefill is DISABLED on sm_120** (`attention.fp8_prefill` auto → off; cuBLAS FP8 returns `NOT_SUPPORTED` at non-aligned M on consumer Blackwell — `src/runtime/engine_init_resolver.cpp:156`). The historical "+40–60% prefill" FP8×FP8 cuBLAS path does not apply here; prefill levers are the FA2 family above.

PTX templates for all of these → `references/ptx-patterns.md`.

## Shared-memory budget

Max opt-in: **~99 KB per block** (NOT H100's 228 KB — query `cudaDeviceProp::sharedMemPerBlockOptin`).

| head_dim | Bq | SMEM | Notes |
|----|----|----|-------|
| 64 | 128 | ~89 KB | Double-buffer KV +16 KB |
| 128 | 64 | ~81 KB | Standard |
| 256 | 32 | ~88 KB | Qwen3.5 GDN partial-RoPE |

```cuda
cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
```

## Compile flags

**Target `sm_120a`** (the `a` arch suffix — superset of `120f` that adds `mma.sync.kind::mxf4nvf4.block_scale` and TMA-WS-Grouped-GEMM, both required for the CUTLASS NVFP4 fast path on Mamba2 shapes). Do NOT target `sm_120` or `sm_120f`. Do NOT add a generic `compute_120` PTX fallback (lacks FP8 MMA + block-scale).

```cmake
set(IMP_SM120_FLAGS "--generate-code=arch=compute_120a,code=sm_120a")
# also:
--expt-relaxed-constexpr --extended-lambda
# Release: -O3 --use_fast_math
```

`compute_120a` unlocks: `mma.sync.aligned.kind::mxf4nvf4.block_scale`, extended `cp.async.bulk.tensor` (TMA Multicast), Cluster-Launch + CLC, extended mbarrier phases, hardware FP4 saturation `F2FP.SATFINITE.E2M1`. Does NOT unlock `tcgen05.*`, TMEM, `wgmma` — SM100 only.

Guard sm_120 code: `#if __CUDA_ARCH__ >= 1200`.

## Common mistakes

| Mistake | Fix |
|---------|-----|
| Targeting `sm_120` / `sm_120f` instead of `sm_120a` | `compute_120a/sm_120a`. `120f` blocks `mxf4nvf4.block_scale` + TMA-WS-grouped-GEMM → forces GDN/Mamba2 onto slow `gemm_nvfp4`. |
| Routing decode through slow `gemm_nvfp4` (dequant→cuBLAS) | Promote weight to `StorageTier::CUTLASS_NVFP4`. Verify `cutlass_nvfp4 weight cache` log line at init. |
| Forgetting CUDA Graph re-bench after a kernel speedup | Compute speedup alone doesn't show in tok/s — pair every hot-path patch with graphs-ON A/B. |
| Assuming H100 SMEM (228 KB) | RTX 5090 max = 99 KB opt-in. Query `cudaDeviceProp::sharedMemPerBlockOptin`. |
| `__launch_bounds__` on regular paths | -4.5% to -20%. Exceptions: HD=128 GDN miscompile, FMHA SMEM-limited (`256,1`). |
| `reinterpret_cast` on Q8_0 blocks | 34-byte blocks NOT 4-aligned. Use `memcpy()`. |
| `__noinline__` on device helpers | Spills to Local Memory (DRAM). Use `__forceinline__`. |
| Missing `__syncthreads()` after `cp.async wait` | Race on SMEM reads. |
| Pointer advance without `sizeof(T)` | See FP8 FMHA S_tile bug — `references/known-issues.md`. |
| Register pressure > 48/thread | `--ptxas-options=-v` and refactor. |

## Where to look next

- **PTX templates** (mxf4nvf4, f8f6f4, cp.async, prmt LUT, FP16→FP8 cvt, warp shuffle, `__ldcs`) → `references/ptx-patterns.md`
- **Dead ends, version-dependent retries, resolved issues** → `references/known-issues.md`
- **Repo-internal docs**: `docs/sm120.md` (kernel notes), `docs/performance.md` (baselines + methodology). Historical sm_120 perf plan summarized in `docs/archive/README.md` (all five levers shipped or superseded by 2026-05-10; full text in git history).
- **Hot-path source**: `src/compute/` (attention, gemm, NVFP4), `src/quant/` (dequant), `tools/imp-bench/`
