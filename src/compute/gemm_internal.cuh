#ifndef IMP_COMPUTE_GEMM_INTERNAL_CUH
#define IMP_COMPUTE_GEMM_INTERNAL_CUH

// Shared device helpers + GEMV launch constants used across the gemm*.cu
// translation units split out of gemm.cu. Symbols are kept verbatim from the
// original gemm.cu so the hot-path numerics are byte-identical.

#include "core/tensor.h"

#include <cublas_v2.h>
#include <cublasLt.h>
#include <cstddef>

namespace imp {

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------

// Warp-level sum reduction via __shfl_down_sync. Result valid in lane 0 only.
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

// GEMV launch constants: 256 threads = 8 warps per block.
static constexpr int kGemvThreads = 256;
static constexpr int kGemvWarps = kGemvThreads / 32;

// Compute the number of blocks needed to cover M rows at kGemvWarps rows/block.
static inline int gemv_blocks(int M) { return (M + kGemvWarps - 1) / kGemvWarps; }

// ---------------------------------------------------------------------------
// cuBLAS internals shared with the batched-GEMM TU (gemm_batched.cu). The
// definitions + the lazily-initialized handles/workspace live in gemm.cu.
// ---------------------------------------------------------------------------
cublasHandle_t gemm_internal_cublas_handle();
cublasLtHandle_t gemm_internal_cublaslt_handle();
cudaDataType_t gemm_internal_dtype_to_cuda(QType dt);
void* gemm_internal_workspace();
size_t gemm_internal_workspace_size();

// gemm() fast paths defined in gemm_gemv_dtype.cu (co-located with gemv). Both
// return true if they handled the call. Called from gemm() in gemm.cu.
bool gemm_try_gemv(const Tensor& A, const Tensor& B, Tensor& C, float alpha, float beta,
                   cudaStream_t stream);
bool gemm_try_sgemm(const Tensor& A, const Tensor& B, Tensor& C, float alpha, float beta,
                    cudaStream_t stream);

}  // namespace imp

#endif  // IMP_COMPUTE_GEMM_INTERNAL_CUH
