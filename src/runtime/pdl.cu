#include "runtime/pdl.h"
#include "core/logging.h"
#include <cuda_runtime.h>

namespace imp {
namespace pdl {

void enable(const void* kernel_func) {
#if IMP_CUDA_13_1
    // Set the Programmatic Stream Serialization attribute on this kernel.
    // When enabled, the GPU scheduler can overlap the tail of this kernel
    // with the head of the next kernel on the same stream, reducing
    // inter-kernel gaps by 5-15%.
    cudaFuncAttributes attrs;
    cudaError_t err = cudaFuncGetAttributes(&attrs, kernel_func);
    if (err != cudaSuccess) {
        IMP_LOG_WARN("PDL: cudaFuncGetAttributes failed: %s",
                     cudaGetErrorString(err));
        return;
    }

    cudaLaunchAttribute attr;
    attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr.val.programmaticStreamSerializationAllowed = 1;

    cudaLaunchConfig_t launch_config = {};
    launch_config.attrs = &attr;
    launch_config.numAttrs = 1;

    // Store the PDL attribute for this kernel via function-level attribute.
    // CUDA 13.1 supports setting this as a persistent function attribute.
    err = cudaFuncSetAttribute(kernel_func,
                               cudaFuncAttributeProgrammaticStreamSerialization,
                               1);
    if (err != cudaSuccess) {
        // Fallback: attribute not supported on this kernel/device, non-fatal
        IMP_LOG_DEBUG("PDL: cudaFuncSetAttribute not supported for kernel %p: %s",
                      kernel_func, cudaGetErrorString(err));
    }
#else
    (void)kernel_func;
#endif
}

void disable(const void* kernel_func) {
#if IMP_CUDA_13_1
    cudaError_t err = cudaFuncSetAttribute(
        kernel_func,
        cudaFuncAttributeProgrammaticStreamSerialization,
        0);
    if (err != cudaSuccess) {
        IMP_LOG_DEBUG("PDL: disable failed for kernel %p: %s",
                      kernel_func, cudaGetErrorString(err));
    }
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
