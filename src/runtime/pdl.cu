#include "runtime/pdl.h"
#include "core/logging.h"
#include <cuda_runtime.h>
#include <unordered_set>

namespace imp {
namespace pdl {

// Registry of kernel functions with PDL enabled.
static std::unordered_set<const void*>& enabled_kernels() {
    static std::unordered_set<const void*> s;
    return s;
}

static bool s_pdl_available = false;
static bool s_pdl_checked = false;

void enable(const void* kernel_func) {
#if IMP_CUDA_13_1
    enabled_kernels().insert(kernel_func);
    IMP_LOG_DEBUG("PDL: enabled for kernel %p (registry size: %zu)",
                  kernel_func, enabled_kernels().size());
#else
    (void)kernel_func;
#endif
}

void disable(const void* kernel_func) {
#if IMP_CUDA_13_1
    enabled_kernels().erase(kernel_func);
    IMP_LOG_DEBUG("PDL: disabled for kernel %p", kernel_func);
#else
    (void)kernel_func;
#endif
}

bool is_enabled(const void* kernel_func) {
#if IMP_CUDA_13_1
    return enabled_kernels().count(kernel_func) > 0;
#else
    (void)kernel_func;
    return false;
#endif
}

bool is_available() {
    if (s_pdl_checked) return s_pdl_available;
    s_pdl_checked = true;

#if IMP_CUDA_13_1
    int device;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        s_pdl_available = false;
        return false;
    }

    // PDL requires compute capability >= 9.0 (Hopper+)
    int major = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
    s_pdl_available = (major >= 9);

    if (s_pdl_available) {
        IMP_LOG_INFO("PDL: available (sm_%d0+)", major);
    }
#else
    s_pdl_available = false;
#endif
    return s_pdl_available;
}

} // namespace pdl
} // namespace imp
