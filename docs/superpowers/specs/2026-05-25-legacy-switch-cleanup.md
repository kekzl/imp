# Legacy gemm_dispatch Switch Cleanup

**Status**: done
**Estimated effort**: 2-3h (actual: ~1h)
**Prereqs**: PR #399 (dispatch migration), PR #400 (M=1 gemv_dispatch) — both merged

## Goal

Trim the 290-LOC `gemm_dispatch(Tensor, Tensor, Tensor, GemmContext)` in
`src/exec/executor_kernels.cu` to a ~66 LOC static fallback that only handles:

1. `kInvalidTensorID` — unregistered weights (budget-exhausted, no overlay)
2. `beta != 0` residual-add for M=1 on registered weights
3. Generic dequant catch-all for M>1 prefill on uncached weights

## Plan correction

The original plan assumed the legacy switch was only reached for
`kInvalidTensorID`, but `StorageTier::FP16` handles at M=1 also fell through
the `default: break` in `gemm_via_handle_`'s switch. Deleting the FP16 cache,
FP16 raw, and dp4a branches without migrating FP16 tier would have broken
GGUF decode for models without NVFP4 decode cache (Q8_0, Q4_K_M, etc.).

Fix: added `case StorageTier::FP16:` to `gemm_via_handle_`'s M=1 switch that
routes GGUF block-quant sources through the dp4a/mmvq registry handler (same
5-10x decode advantage over cuBLAS on the FP16 overlay) and native FP16 weights
through `gemv_dispatch`. This made all legacy branches truly dead.

## What was deleted

All 10 tier-specific branches (MXFP4, Q4K_IMMA, FP8, NVFP4 M=1, CUTLASS M>1,
NVFP4 M>1, FP16 cache, FP16 raw, dp4a small-M, coverage-gap guard) plus
5 cache pointer variables (fp16, fp8, nv4, ct4, mx4).

## What stays

- Beta != 0 section: FP16 cache lookup or dequant->cuBLAS for residual-add
- Generic dequant catch-all (M>1 prefill for uncached weights)
- Dropped-weight coverage-gap warning
- Raw FP16/BF16 cuBLAS final fallback
- Function renamed to `gemm_dispatch_uncached_fallback` (static, declaration
  removed from executor_kernels.h)

## Changes

- `src/exec/executor_kernels.cu`: `gemm_dispatch` 290 LOC -> 66 LOC static
  `gemm_dispatch_uncached_fallback`. `gemm_via_handle_` gained FP16 tier
  routing with dp4a/mmvq for GGUF sources + gemv_dispatch for native FP16.
- `src/exec/executor_kernels.h`: removed `gemm_dispatch` declaration (now
  file-local)
