// src/compute/gemm_grouped_nvfp4_smallM.cu
#include "compute/gemm_grouped_nvfp4_smallM.h"
#include "core/logging.h"
#include <cuda_runtime.h>

namespace imp {

static int s_smallM_available = -1;

bool gemm_grouped_nvfp4_smallM_available() {
    if (s_smallM_available >= 0) return s_smallM_available;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    s_smallM_available = (prop.major * 10 + prop.minor >= 120) ? 1 : 0;
    return s_smallM_available;
}

void gemm_grouped_nvfp4_smallM_cleanup() {}

bool gemm_grouped_nvfp4_smallM(
    int /*n_experts*/, const int* /*host_M*/, int /*N*/, int /*K*/,
    const void* const* /*host_ptr_A*/, const void* const* /*host_ptr_SFA*/,
    const void* const* /*host_ptr_B*/, const void* const* /*host_ptr_SFB*/,
    void* const* /*host_ptr_D*/, const float* /*host_alpha*/,
    cudaStream_t /*stream*/) {
    return false;  // skeleton: caller falls back to CUTLASS path
}

}  // namespace imp
