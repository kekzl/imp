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
// THIS DOES NOT PRODUCE TAIL/HEAD OVERLAP TODAY, and this comment used to say
// that it did (#1655). Programmatic dependent launch is two halves: the host
// attribute here, and a device half in the kernels themselves. No kernel in
// src/ calls cudaTriggerProgrammaticLaunchCompletion() or
// cudaGridDependencySynchronize(), so a producer's completion event fires only
// after its last block exits, which is exactly when the default dependency
// would have released the consumer. Same schedule, extra machinery.
//
// Measured before deciding to keep it (Qwen3-8B-Q8_0, RTX 5090, 3 alternating
// rounds of `imp-cli --bench --bench-pp 512 --bench-reps 3`): runtime.no_pdl
// true against false is 12508 vs 12455 tok/s prefill and 385.8 vs 382.3 tok/s
// decode, both inside their own arms' spread. It costs nothing measurable and
// buys nothing measurable. docs/DESIGN_DECISIONS.md says why it is still here
// rather than deleted.
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
