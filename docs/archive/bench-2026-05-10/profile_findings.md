# Prefill profile findings — 2026-05-10

## Workload

`pp=512, max_tokens=1, reps=1` on Qwen3-Coder-30B-A3B-Instruct-FP4. Single prefill pass after warmup.

## Top kernels (nsys, all GPU time across pp512 + 1 decode step + warmup)

| Time % | Total ns | Instances | Avg ns | Kernel |
|---:|---:|---:|---:|---|
| **88.2%** | 1,068,033,219 | 432 | 2,472,299 | `imp::dequantize_nvfp4_moe_kernel` |
| 8.6%   | 104,512,580 | 432 | 241,927 | `cutlass_80_tensorop_f16_s16816gemm_f16_grouped_128x64_64x3_align8` (cuBLAS FP16 grouped) |
| 0.5%   | 6,390,874   | 576 | 11,095  | `cutlass::device_kernel<MainloopSm120TmaWarpSpecializedBlockScaled..mxf4nvf4..>` |
| 0.5%   | 6,231,970   | 144 | 43,278  | `imp::causal_softmax_fp32_to_fp16_kernel` |
| 0.4%   | 5,028,136   | 579 | 8,684   | `imp::rmsnorm_fp16_kernel` |
| 0.3%   | 3,984,364   | 149 | 26,741  | cuBLAS WMMA FP16 (attention) |

432 instances of dequant = **3 reps × 48 layers × 3 projections** (gate+up+down).
Per prefill: 144 calls × 2.47 ms = **356 ms of dequant alone**, vs total prefill = 412 ms.

## Reading the data

- The CUTLASS Sm120 NVFP4 block-scaled kernel **is present and working** (576 instances, 11 µs each — clean fast path) but only fires for **non-MoE attention NVFP4 weights** (the 192 tensors at `executor_pre_dequant.cu:516`).
- For MoE expert weights (144 tensors, 13.8 GiB), prefill dispatches through the slow path: dequant 96 MiB packed → 384 MiB FP16 per projection, then FP16 cuBLAS grouped batched GEMM. **Together 96.8% of prefill time**.
- This is exactly the scenario the comment at `executor_forward_moe.cu:1310` claims "Prefill n=120: ~2750 tok/s (vs legacy ~77) — 35× win" — meaning the fast path's perf was measured at some point. The bug is that the slow `NVFP4→FP16 batch dequant` branch sits **above** the fast `CUTLASS 3.x NVFP4 grouped` branch in the if/else chain at `executor_forward_moe.cu:1246` vs `:1340`, AND the fast path's `covers_ids` predicate fails because MoE expert tensors aren't tier'd as `CUTLASS_NVFP4`.

## Why the fast path's `covers_ids` fails on MoE

`cache_moe_native_nvfp4` (`executor_pre_dequant.cu:1776`) does two things:
1. Copies all per-expert NVFP4 packed bytes + native row-major UE4M3 micro-scales into one contiguous `nvfp4_moe_packed_native` / `nvfp4_moe_ms_native` buffer per layer-projection.
2. **Frees the per-expert source allocations** (`:1907-1922`) and sets each `expert_w_*[e].data = nullptr`.

After step 2, `register_tensor` (`:2125`) bails at `if (!t.data)` → `expert_*_ids[e] = kInvalidTensorID`.
Even with valid pointers, the fast path also requires:
- The per-expert pointer to be present in `wcache_.cutlass_nvfp4` (only attention NVFP4 was registered there at `:472-490`)
- The SF buffer to be in CUTLASS SfAtom layout (MoE has it in native row-major)

## Theoretical upside of moving MoE prefill onto CUTLASS sm_120 NVFP4 grouped

| | current | projected |
|---|---:|---:|
| dequant_nvfp4_moe_kernel | 360 ms | 0 ms |
| FP16 cuBLAS grouped GEMM | 35 ms | n/a |
| NVFP4 grouped (FP4 TC ≈ 4× FP16 TC peak) | n/a | 9–18 ms |
| other (RMSNorm, attn, routing, scatter…) | 17 ms | 17 ms |
| **prefill total** | **412 ms** | **26–35 ms** |
| **pp512 tok/s** | **1241** | **14600–19700** |

Best case puts us at **~80% of vLLM (25513)**, well above the 2× target (12750). Even at 50% kernel efficiency we hit 11k+ tok/s.

## Risk to decode

The change targets **MoE prefill only** (n>1 dispatch in `executor_forward_moe.cu`). The decode-side path is `gemv_nvfp4_moe_decode` / `gemv_nvfp4_moe_swiglu_decode` (`src/quant/nvfp4_gemm.h:63-73`), which still consumes the native-layout `NvFP4MoEQuantResult` directly and is untouched. CUDA Graph capture for decode (currently +234% on this model) is unaffected.

## Plan

Make the existing CUTLASS 3.x NVFP4 grouped path fire on MoE prefill:

1. **Build per-projection SfAtom SF buffers** at load time, alongside the native row-major `nvfp4_moe_ms_native` (cost: same ~1.7 GiB; native kept for the decode GEMV path which depends on it).
2. **Re-stamp expert tensor `Tensor::data`** to point into the contiguous packed buffer instead of nullptr (so `register_tensor` doesn't bail).
3. **Add per-expert entries to `wcache_.cutlass_nvfp4`** so `infer_tier_from_wcache` returns `CUTLASS_NVFP4` and the fast path's `covers_ids` passes.
4. **Re-order branches** in `executor_forward_moe.cu`: fast path before the `NVFP4→FP16 batch dequant` path.

No changes to the decode path, no changes to the public C API, no new third-party deps.
