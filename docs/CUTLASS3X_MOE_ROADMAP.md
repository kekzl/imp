# CUTLASS 3.x NVFP4 Grouped GEMM — MoE Roadmap

## Context

CUDA 13.2 Update 1 (installed 2026-04-16) improves Blackwell Grouped GEMM throughput by ~20% on large problem sizes and extends `cublasLtMatmulGrouped` to NVFP4 inputs with bias epilogues. Currently our MoE prefill path dequants NVFP4 weights → FP16 → standard grouped GEMM, losing the NVFP4 tensor-core throughput advantage and incurring 4+ `cudaStreamSynchronize` per MoE forward (one D2H per expert-offset copy).

The target is to replace this with CUTLASS 3.x `GroupProblemShape` + `PtrArray` grouped GEMM that takes device-side problem shapes — **zero D2H sync** — and computes NVFP4 × NVFP4 → FP16 directly.

Primary beneficiary: **Qwen3-Coder-30B-A3B-FP4** (128 experts × 48 layers × 3 projections). Secondary: any future MoE model loaded from NVFP4 SafeTensors.

## Current State (2026-04-20)

**Kernel scaffold** in `src/compute/gemm_cutlass_grouped_3x.{cu,h}`:
- `GrpGemm` type fully instantiates on SM120 (verified via `static_assert`)
- Uses `OpClassBlockScaledTensorOp` + `KernelScheduleAuto` → auto-selects SM120 PtrArray cooperative schedule
- NVFP4 input / NVFP4 weight / FP16 output / FP32 accumulator
- Dispatch function `gemm_grouped_cutlass_3x_nvfp4()` body is a **stub** (logs warn, returns false)

**2.x path** still primary: `gemm_grouped_cutlass_sm120.cu` (237 lines, 4 `cudaMemcpyAsync` + 1 `cudaStreamSynchronize`).

**Dead ends confirmed:**
- FP16 Grouped GEMM on SM120: not supported (`SM120 TmaWarpSpecialized builder currently only supports F8F6F4 MMA`)
- FP16 PtrArray on SM90: CUTLASS 4.4.2 API bug in `sm90_gemm_array_tma_warpspecialized_cooperative.hpp:291`
- `cvt .e2m1x2`: blocked in CUDA 13.2 ptxas, retry with 13.3+

## Remaining Work

### 1. Dispatch function body (`gemm_cutlass_grouped_3x.cu`)
- Build device-side arrays (single small kernel or via `cudaMemcpyAsync`):
  - `d_shapes[ne]`: per-expert `(M_i, N, K)` GroupProblemShape entries
  - `d_A_ptrs[ne]`, `d_SFA_ptrs[ne]`: per-expert packed FP4 activation + SfAtom scale pointers
  - `d_B_ptrs[ne]`, `d_SFB_ptrs[ne]`: per-expert weight + scale pointers
  - `d_D_ptrs[ne]`: per-expert output pointers (FP16)
  - `d_strides_A[ne]`, `d_strides_B[ne]`, `d_strides_D[ne]`: per-expert strides via `cutlass::make_cute_packed_stride`
- Populate `GrpGemm::Arguments` with device pointer arrays
- Persistent workspace + scratch buffer reuse (like 2.x path)

### 2. SafeTensors loader extension
Expert weights for Qwen3-Coder-30B-A3B-FP4: 73,728 tensors = 48 layers × 128 experts × 3 projections × 4 files (weight, weight_scale, weight_scale_2, input_scale).

- Extend the NVFP4-prequant linkage in `src/model/safetensors_loader.cpp` to recognize `mlp.experts.{e}.{gate,up,down}_proj.weight` patterns
- Build per-layer arrays of `NvFP4PreQuantWeight` (one per expert per projection)
- Store in `WeightCache::nvfp4_moe_prequant` keyed by layer + projection

### 3. Weight upload
- NVFP4 packed expert data + UE4M3 SfAtom scales → GPU
- Pre-build per-expert `CutlassNvFP4Weight` structs on host, copy struct array to GPU
- Allocate the `d_B_ptrs[ne]` / `d_SFB_ptrs[ne]` arrays once per layer at upload time (fixed)

### 4. MoE dispatch wiring (`executor_forward_moe.cu`)
- New branch in the `can_fp16_batch_nosync` check: if NVFP4 MoE prequant cache hit, use `gemm_grouped_cutlass_3x_nvfp4` instead of FP16 batch path
- Per MoE forward:
  - Quantize FP16 activations → NVFP4 (`quantize_fp16_to_nvfp4_cutlass`), indexed by expert offset
  - Launch dispatch for up/gate/down projections
- Gate behind env flag `IMP_CUTLASS3X_MOE=1` during rollout

### 5. E2E test
- Target: Qwen3-Coder-30B-A3B-FP4 via SafeTensors
- Correctness: `tests/test_e2e_models.cpp` pattern — generate a few tokens greedily, compare against the current 2.x-via-dequant baseline (allow small FP epsilon — NVFP4×NVFP4 vs FP16×FP16 won't bit-match)
- Perf: `imp-cli --bench --bench-pp 512` before/after — target +20% prefill (CUDA 13.2 Update 1 headline number)

## Out of scope
- Hopper BF16/FP16/FP8 Grouped GEMM: only sm_100, not sm_120 (per 13.2 Update 1 release notes)
- MXFP4 grouped GEMM: not prioritized — needs MR-GPTQ calibration first to break the 1-mantissa-bit quality ceiling (see `docs/MXFP4_QUANTIZATION.md`)
- sm_120 FP4 cuBLASLt: NVIDIA hasn't compiled FP4 kernels for consumer Blackwell (probe returns status=7). Re-test after each libcublas patch bump.

## References
- Memory: `cutlass_3x_grouped_gemm.md`, `cuda_13_2_update_1.md`, `nvfp4_prequant_status.md`
- CUTLASS Example 79d: Blackwell GeForce NVFP4 Grouped GEMM
- Existing single-expert NVFP4 GEMM: `src/compute/gemm_cutlass_sm120.cu`
- 2.x grouped reference: `src/compute/gemm_cutlass_grouped_sm120.cu`
