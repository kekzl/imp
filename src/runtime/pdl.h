#pragma once

#include <cuda_runtime.h>

namespace imp {

namespace pdl {

// Enable PDL on a kernel function. Must be called before the kernel is launched.
void enable(const void* kernel_func);

// Disable PDL on a kernel function (restore default behavior).
void disable(const void* kernel_func);

// Check if PDL is available on the current device/CUDA version.
bool is_available();

// Check if a specific kernel has PDL enabled.
bool is_enabled(const void* kernel_func);

// Convenience: enable PDL for a __global__ function template.
// Usage: pdl::enable_kernel(my_kernel<float>);
template <typename KernelFunc>
void enable_kernel(KernelFunc func) {
    enable(reinterpret_cast<const void*>(func));
}

// ---------------------------------------------------------------------------
// PDL-aware kernel launch.  Uses cudaLaunchKernelEx with the
// ProgrammaticStreamSerialization attribute when PDL is enabled for the
// kernel.  Falls back to standard <<<>>> launch when PDL is not
// enabled/available.
//
// Device half (2026-08-31, compute/pdl_device.cuh): a registered kernel calls
// pdl_wait() before its first global access and pdl_trigger() after its last
// input read, so a programmatic edge really lets the consumer's blocks land
// on the SMs during the producer's tail. Registration is the promise that
// the kernel waits (cuda_graph.cu converts an edge only when the CONSUMER is
// registered); a kernel without pdl_wait() must never be registered. The
// launch sites register themselves (pdl::enable_kernel right before
// pdl::launch) so every template instantiation that runs is covered.
// Before the device half, this attribute was inert (#1655 measured
// runtime.no_pdl true vs false inside noise); runtime.no_pdl=true remains
// the control arm and turns both halves off.
//
// Usage:
//   pdl::launch(my_kernel, grid, block, smem, stream, arg1, arg2, ...);
// ---------------------------------------------------------------------------
template <typename KernelFunc, typename... Args>
void launch(KernelFunc func, dim3 grid, dim3 block, size_t smem, cudaStream_t stream, Args... args) {
    const void* func_ptr = reinterpret_cast<const void*>(func);
    if (is_enabled(func_ptr)) {
        cudaLaunchConfig_t config = {};
        config.gridDim = grid;
        config.blockDim = block;
        config.dynamicSmemBytes = smem;
        config.stream = stream;

        cudaLaunchAttribute attr = {};
        attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attr.val.programmaticStreamSerializationAllowed = 1;

        config.attrs = &attr;
        config.numAttrs = 1;

        cudaLaunchKernelEx(&config, func, args...);
    } else {
        func<<<grid, block, smem, stream>>>(args...);
    }
}

// RAII guard: enables PDL on construction, can disable on destruction.
class ScopedPDL {
public:
    explicit ScopedPDL(const void* kernel_func, bool auto_disable = false)
        : kernel_func_(kernel_func), auto_disable_(auto_disable) {
        enable(kernel_func_);
    }
    ~ScopedPDL() {
        if (auto_disable_) {
            disable(kernel_func_);
        }
    }
    ScopedPDL(const ScopedPDL&) = delete;
    ScopedPDL& operator=(const ScopedPDL&) = delete;

private:
    const void* kernel_func_;
    bool auto_disable_;
};

}  // namespace pdl

}  // namespace imp
