# Legacy gemm_dispatch Switch Cleanup

**Status**: planned (next session)
**Estimated effort**: 2-3h
**Prereqs**: PR #399 (dispatch migration), PR #400 (M=1 gemv_dispatch) — both merged

## Goal

Trim the 290-LOC `gemm_dispatch(Tensor, Tensor, Tensor, GemmContext)` in
`src/exec/executor_kernels.cu` to a ~50 LOC fallback that only handles:

1. `kInvalidTensorID` — unregistered weights (budget-exhausted, no overlay)
2. Generic dequant catch-all for M>1 prefill on uncached weights

## What's dead

After PR #399 + #400, `gemm_via_handle_()` routes all 29 call sites:

- **M>1**: WeightHandle dispatch (`weight_dispatch.cu`) for all tiers
- **M=1 NVFP4/FP8/MXFP4**: `gemv_dispatch` (WeightHandle)
- **M=1 CUTLASS_NVFP4**: secondary `wcache_.nvfp4` probe → kpar GEMV
- **M=1 FP16 + dp4a source**: `GemmKernelRegistry` dp4a handler
- **M=1 FP16 native**: `gemv_dispatch` → cuBLAS GEMV

The legacy switch is only reached from `gemm_via_handle_`'s fallback for
`kInvalidTensorID`. The following code paths in the switch are dead:

- MXFP4 native GGUF (line ~1642) — handled by `gemm_via_handle_` M>1
- Q4_K IMMA (line ~1667) — handled by M>1 WeightHandle
- FP8 prefill (line ~1681) — handled by M>1 WeightHandle
- NVFP4 decode GEMV M=1 (line ~1714) — handled by `gemv_dispatch`
- CUTLASS NVFP4 prefill M>1 (line ~1749) — handled by M>1 WeightHandle
- NVFP4 prefill M>1 (line ~1779) — handled by M>1 WeightHandle
- FP16 cache M>1 (line ~1806) — handled by M>1 WeightHandle
- FP16 raw (line ~1829) — handled by M>1 WeightHandle
- GGUF small-M dp4a (line ~1839) — handled by `gemm_via_handle_` FP16 M=1
- Coverage-gap guard (line ~1887) — `gemm_via_handle_` routes dropped weights

## What stays

- Generic dequant catch-all (line ~1869): `dequant_gpu` + cuBLAS for
  uncached weights at M>1. This is the safety net for budget-exhausted models.
- Final raw FP16 fallback (line ~1898): `gemm(input, weight, ...)` for
  natively FP16/BF16 uncached weights.

## Steps

1. Delete the dead branches (MXFP4, Q4K_IMMA, FP8, NVFP4 M=1, CUTLASS,
   FP16 cache, dp4a small-M, coverage-gap guard)
2. Keep generic-dequant + raw-FP16 fallback (~50 LOC)
3. Rename function to `gemm_dispatch_uncached_fallback_` or similar
4. Update `gemm_via_handle_` to call the renamed function
5. Build + 401 tests + smoke bench
