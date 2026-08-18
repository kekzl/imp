---
name: sm120-cuda-expert
description: Use when writing, reviewing, or optimizing CUDA kernels targeting sm_120a (RTX 5090 / Consumer Blackwell, GB202) in the imp inference engine. Triggers on CUDA/PTX kernel code, shared-memory layout, tensor-core MMA, GEMV/GEMM/attention/quantization kernels, decode tok/s under expected, kernel emitting HMMA instead of mxf4nvf4, occupancy or register-pressure questions. Pair with `benchmark-cuda` for measurement and `check-degeneration` after hot-path changes.
---

# sm_120 CUDA Kernel Expert — imp Inference Engine

Optimal-kernel reference for RTX 5090 (GB202, **sm_120a** — consumer Blackwell).
PTX inline-assembly templates → `references/ptx-patterns.md`. Dead ends,
version-dependent gotchas, root-cause fixes → `references/known-issues.md`.

## Architecture quick reference

| Spec | Value |
|------|-------|
| SMs · CUDA cores · TC | 170 · 21,760 · 680 (4/SM, 5th gen) |
| L1/SMEM per SM | 128 KB configurable (~99 KB opt-in shared, query `cudaDeviceProp::sharedMemPerBlockOptin`) |
| L2 cache | 96 MB unified |
| VRAM | 32 GB GDDR7, 1,792 GB/s |
| Boost clock | 2,407 MHz |
| Native MMA shapes | FP16 `m16n8k16`, FP8 `m16n8k32`, FP4 block-scaled `m16n8k64` |

**sm_120 has NO `tcgen05` / TMEM / `wgmma` / 2-CTA cluster MMA** — those are
SM100 (B200) only. The peak path is register-based `mma.sync` with
block-scaling, FA2-style. Ignore B200 kernel designs unless porting.

**Use measured TC rates for roofline math, not datasheet TOPS.** FP4 `mma.sync`
reaches ≈**2,019 TOPS (~½ datasheet)**, and **f32-accumulate runs at ¼ rate** —
accumulate in f16 wherever PPL allows. Against datasheet numbers every FP4
kernel looks falsely bad.

**Calibrate %-of-roofline expectations against the measured baseline, not
theory** — `tools/roofline/history/BASELINE` plus the newest report in
`docs/audit/`. Two readings from it that change decisions: the dense NVFP4
decode GEMV class sits at its structural ceiling (multiple levers refuted —
see known-issues), while paged decode attention shows single-digit %-roofline
that is *latency-bound at M=1*, so the mechanical "target 70%" there is not
real headroom.

## The three laws

1. **Decode at batch=1 is launch-overhead-bound first, memory-bound second.**
   With ~80–120 launches per layer, per-launch µs dominate once GEMM is fast.
   Order: (a) make per-launch GEMM fast — CUTLASS sm_120 NVFP4 fast-path, not
   the `gemm_nvfp4` dequant→cuBLAS fallback; (b) capture decode in a CUDA Graph
   (`CudaGraphConditionalRunner`, `src/runtime/cuda_graph.h`); (c) only then
   chase memory traffic. A faster kernel alone shows little tok/s gain — it
   *enables* the graph win. Always re-bench graphs-ON after a hot-path patch.
   **Corollary: any per-token host round-trip drops a path out of the
   conditional loop and costs −27–45% decode.** When adding a feature to the
   decode path, check it does not force a host sync (logprobs still does one —
   an open lever).

2. **Occupancy is king — for reaching the roofline, not passing it.** Keep
   registers ≤48/thread for 100% occupancy. **Don't add `__launch_bounds__`**
   on regular GEMV/attention paths (costs −4.5% to −20%); two correct
   exceptions exist (HD=128 GDN miscompile workaround, and SMEM-limited FMHA
   at `(256,1)`). On a path already at its measured ceiling, occupancy work is
   refuted — check the roofline baseline before spending time there.

3. **Quantization type determines kernel strategy.** Q8_0 (simple dequant) →
   bandwidth-bound → row-parallel + smem-cached activations. Q6_K (complex
   dequant) → compute-influenced → K-parallel + warp-level division. NVFP4
   prequant → must reach `StorageTier::CUTLASS_NVFP4` (SfAtom layout) for the
   fast path; plain `StorageTier::NVFP4` falls through to slow `gemm_nvfp4`.
   There is no universal best GEMV.

## Paths that must stay active (verify before optimizing anything else)

A "slow kernel" is usually a path that fell back, not a kernel that needs work:

- **CUTLASS NVFP4 weight cache** — check the `cutlass_nvfp4 weight cache` line
  at init. Without it decode runs the dequant→cuBLAS slowpath.
- **CUDA-graph decode loop** — the single biggest decode multiplier; requires
  the CUTLASS fast path first.
- **FA2 prefill family** (`attention.fmha_fa2`, `fa2_hd256`, `fa2_fp16qk`, all
  default-on) — the prefill attention path on this chip. Other head dims decline
  to cuBLAS safely.
- **Decode sidecars**: `gemm.fp8_ssm_proj` (GDN in/out projections),
  `gemm.nvfp4_lm_head*`, `gemm.fp8_attn_proj` — each default-on or "auto" with a
  measured quality trade recorded in `docs/GOAL.md`. Don't silently flip one.
- **INT8-IMMA prefill GEMM family** for GGUF. Note dense Q4_K IMMA loses to
  cuBLAS and is off by design; the win is MoE prefill.

**FP8 prefill is DISABLED on sm_120** — cuBLAS FP8 returns `NOT_SUPPORTED` at
non-aligned M (`src/runtime/engine_init_resolver.cpp`). Any plan quoting
"+40–60% from FP8×FP8 prefill" is for another chip; prefill levers here are the
FA2 family.

## Shared-memory budget

Max opt-in: **~99 KB per block** (NOT H100's 228 KB — query
`cudaDeviceProp::sharedMemPerBlockOptin`).

| head_dim | Bq | SMEM | Notes |
|----|----|----|-------|
| 64 | 128 | ~89 KB | Double-buffer KV +16 KB |
| 128 | 64 | ~81 KB | Standard |
| 256 | 32 | ~88 KB | Qwen3.5 GDN partial-RoPE |

```cuda
cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
```

## Compile flags

**Target `sm_120a`** — the `a` suffix is a superset of `120f` adding
`mma.sync.kind::mxf4nvf4.block_scale` and TMA-WS-Grouped-GEMM, both required
for the CUTLASS NVFP4 fast path on Mamba2 shapes. Do NOT target `sm_120` /
`sm_120f`, and do NOT add a generic `compute_120` PTX fallback (lacks FP8 MMA +
block-scale).

```cmake
set(IMP_SM120_FLAGS "--generate-code=arch=compute_120a,code=sm_120a")
--expt-relaxed-constexpr --extended-lambda      # also
# Release: -O3 --use_fast_math
```

`compute_120a` unlocks `mma.sync.aligned.kind::mxf4nvf4.block_scale`, extended
`cp.async.bulk.tensor` (TMA Multicast), Cluster-Launch + CLC, extended mbarrier
phases, hardware FP4 saturation. It does NOT unlock `tcgen05.*` / TMEM /
`wgmma`. Guard device code with `#if __CUDA_ARCH__ >= 1200`.

## Common mistakes

| Mistake | Fix |
|---------|-----|
| Targeting `sm_120` / `sm_120f` instead of `sm_120a` | `compute_120a/sm_120a`. `120f` blocks `mxf4nvf4.block_scale` → forces GDN/Mamba2 onto slow `gemm_nvfp4`. |
| Routing decode through slow `gemm_nvfp4` | Promote the weight to `StorageTier::CUTLASS_NVFP4`; verify the init log line. |
| Forgetting the graphs-ON re-bench after a kernel speedup | Compute speedup alone doesn't show in tok/s. |
| Assuming H100 SMEM (228 KB) | 99 KB opt-in here. Query the device property. |
| `__launch_bounds__` on regular paths | −4.5% to −20%. Only the two documented exceptions. |
| `reinterpret_cast` on Q8_0 blocks | 34-byte blocks are not 4-aligned — `memcpy()`. |
| `__noinline__` on device helpers | Spills to local memory (DRAM). Use `__forceinline__`. |
| Missing `__syncthreads()` after `cp.async wait` | Race on SMEM reads. |
| Pointer advance without `sizeof(T)` | Cost a real bug — see known-issues. |
| Registers > 48/thread | `--ptxas-options=-v`, then refactor. |
| Claiming a source tweak is "perf-neutral" without a SASS diff | `cuobjdump -sass` — byte-identical SASS is proof, a bench is not. |

## Where to look next

- **PTX templates** (mxf4nvf4, f8f6f4, cp.async, prmt LUT, FP16→FP8 cvt, warp
  shuffle, `__ldcs`) → `references/ptx-patterns.md`
- **Dead ends, version-dependent retries, load-bearing fixes** →
  `references/known-issues.md` — read this BEFORE proposing a lever; most
  obvious ideas on the mature paths have been measured and refuted.
- **Repo docs**: `docs/internals/SM120.md` (hardware), `docs/internals/KERNELS.md` (kernel notes), `docs/PERF.md`
  (baselines + methodology), `docs/audit/` (roofline reports).
- **Hot-path source**: `src/compute/` (attention, gemm, NVFP4), `src/quant/`
  (dequant), `tools/imp-bench/`
