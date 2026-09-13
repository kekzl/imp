// Q4_K HMMA GEMM dispatch (Phase 0 scaffold), config-gated on gemm.q4k_hmma_enabled
// (default false). Called from gemm_via_handle_ at the prefill (M>1) path; a direct call,
// not a GemmKernelRegistry entry.

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
