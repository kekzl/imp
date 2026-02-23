#pragma once

#include <cuda_runtime.h>

namespace imp {

namespace pdl {

// Enable PDL on a kernel function. Must be called before the kernel is launched.
// On CUDA < 13.1, this is a no-op.
void enable(const void* kernel_func);

// Disable PDL on a kernel function (restore default behavior).
void disable(const void* kernel_func);

// Check if PDL is available on the current device/CUDA version.
bool is_available();

// Convenience: enable PDL for a __global__ function template.
// Usage: pdl::enable_kernel(my_kernel<float>);
template<typename KernelFunc>
void enable_kernel(KernelFunc func) {
    enable(reinterpret_cast<const void*>(func));
}

template<typename KernelFunc>
void disable_kernel(KernelFunc func) {
    disable(reinterpret_cast<const void*>(func));
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

} // namespace pdl

} // namespace imp
