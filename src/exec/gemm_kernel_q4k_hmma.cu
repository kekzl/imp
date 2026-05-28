// =============================================================================
// gemm_kernel_q4k_hmma.cu -- Q4_K HMMA GEMM dispatch (Phase 0 scaffold)
// =============================================================================
//
// Config-gated dispatch for the Q4_K x FP16 HMMA GEMM kernel. Called from
// gemm_via_handle_ in executor_kernels.cu at the prefill (M>1) path,
// gated on `gemm.q4k_hmma_enabled` (default false).
//
// Separate file so the kernel linkage + dispatch logic stays modular
// (same pattern as gemm_kernel_q4k_imma.cu, gemm_kernel_fp8.cu, etc.).

#include "compute/mmq_q4k_hmma.h"
#include "core/logging.h"

#include <cuda_fp16.h>
#include <cstdint>

namespace imp {

bool try_q4k_hmma_dispatch(const void* activations_fp16, const void* weight_q4k,
                           void* output_fp16, int M, int N, int K, cudaStream_t stream) {
    // Shape constraints: M,N >= 16, K % 256 == 0.
    if (M < 16 || N < 16) return false;
    if (K % 256 != 0) return false;

    return mmq_q4k_hmma_gemm(activations_fp16, weight_q4k, output_fp16, M, N, K, stream);
}

}  // namespace imp
