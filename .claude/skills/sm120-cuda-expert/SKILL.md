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

## The three laws

1. **Decode at batch=1 is launch-overhead-bound first, memory-bound second.** With ~80–120 launches per layer, per-launch µs costs dominate once GEMM is fast. Order of operations: (a) make per-launch GEMM fast (CUTLASS sm_120 NVFP4 fast-path, not slow `gemm_nvfp4` dequant→cuBLAS fallback); (b) capture decode in CUDA Graph (`AsyncGraphLoop`); (c) only then chase memory traffic. A faster kernel alone shows little tok/s gain — but it *enables* graph capture to win because launches no longer hide behind GEMM. Always re-bench graphs ON after a hot-path patch.

2. **Occupancy is king.** Keep registers ≤48/thread for 100% occupancy. **Don't add `__launch_bounds__`** on regular GEMV/attention paths — overrides cost -4.5% to -20%. Two known correct exceptions: HD=128 GDN kernel needs `(HD,1)` to avoid a miscompile; FMHA `(256,1)` is correct on SMEM-limited kernels (~69 KB → 1 block/SM anyway, allows max register allocation).

3. **Quantization type determines kernel strategy.** Q8_0 (simple dequant) → bandwidth-bound → row-parallel + smem-cached activations. Q6_K (complex dequant) → compute-influenced → K-parallel + warp-level work division. NVFP4 prequant → must hit `StorageTier::CUTLASS_NVFP4` (SfAtom layout) for the sm_120a fast path; plain `StorageTier::NVFP4` falls through to slow `gemm_nvfp4`. No universal "best" GEMV.

## What works (top hits)

| Technique | Gain | Detail |
|-----------|------|--------|
| **CUDA Graph decode + AsyncGraphLoop** | **+95–376% decode** | Conditional graph w/ PDL, `max_steps=255`. Biggest gain on small-GEMM models. Requires CUTLASS NVFP4 fast-path active first. |
| CUTLASS sm_120a NVFP4 weight cache | enables graph win | `StorageTier::CUTLASS_NVFP4`. Verify via init log; `StorageTier::NVFP4` = slowpath. |
| `mma.sync.kind::mxf4nvf4.block_scale` | 2.6× raw MMA over f8f6f4 | k=64 vs k=32; HW applies UE4M3 scale inside MMA. Needs `compute_120a` (see Compile flags). |
| FP8 prefill cache | +40–60% prefill | FP8×FP8 cuBLAS = 2× TC throughput |
| NVFP4 prmt register LUT | +4.7–16% | `prmt.b32` replaces SMEM LUT |
| Inline Q8_1 in O-projection | +5–10% | Eliminates 1 launch + 1 DRAM round-trip |
| `__ldcs` on KV cache reads | +2–4% decode | Bypass L1, evict-first L2. **KV only, NOT weights.** |
| 8-warp Blackwell attention | better SM util | 256 threads vs Hopper's 128 |

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
- **Repo-internal docs**: `docs/sm120.md` (kernel notes), `docs/sm120-real-perf-plan.md` (active perf plan), `docs/performance.md` (baselines + methodology)
- **Hot-path source**: `src/compute/` (attention, gemm, NVFP4), `src/quant/` (dequant), `tools/imp-bench/`
