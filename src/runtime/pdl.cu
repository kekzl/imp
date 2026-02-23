#include "runtime/pdl.h"
#include "core/logging.h"
#include <cuda_runtime.h>

namespace imp {
namespace pdl {

// Track which kernel functions have PDL enabled, so we can set the
// launch attribute at launch time via cudaLaunchKernelEx.
// For now, we use a simple flag-based approach: the enable/disable
// functions just log the intent. The actual PDL is applied via
// cudaLaunchAttribute at launch time (handled by the CUDA runtime
// when using cudaLaunchKernelEx with programmatic stream serialization).
//
// In CUDA 13.1, PDL is enabled per-launch via cudaLaunchConfig_t,
// not via persistent function attributes. The enable() call here
// records that PDL should be applied, and the actual attribute is
// set in the launch wrappers that use cudaLaunchKernelEx.

void enable(const void* kernel_func) {
#if IMP_CUDA_13_1
    IMP_LOG_DEBUG("PDL: enabled for kernel %p", kernel_func);
    // PDL attribute is applied at launch time via cudaLaunchKernelEx
    // with cudaLaunchAttributeProgrammaticStreamSerialization.
    // This function records the intent; the actual mechanism uses
    // launch attributes per-call.
    (void)kernel_func;
#else
    (void)kernel_func;
#endif
}

void disable(const void* kernel_func) {
#if IMP_CUDA_13_1
    IMP_LOG_DEBUG("PDL: disabled for kernel %p", kernel_func);
    (void)kernel_func;
#else
    (void)kernel_func;
#endif
}

bool is_available() {
#if IMP_CUDA_13_1
    int device;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) return false;

    // PDL requires compute capability >= 9.0 (Hopper+)
    int major = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
    return major >= 9;
#else
    return false;
#endif
}

} // namespace pdl
} // namespace imp
