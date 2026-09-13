#pragma once

// Programmatic Dependent Launch, device half. pdl_wait()
// (griddepcontrol.wait) must run before touching any global memory a
// predecessor may still write; pdl_trigger() (griddepcontrol.launch_dependents)
// sits after the last input read, before epilogue stores, and affects
// scheduling only, never visibility. No-ops for a non-programmatic launch
// and for the compute_120f fallback. Contract: every kernel registered via
// pdl::enable() calls pdl_wait() first.

namespace imp {

__device__ __forceinline__ void pdl_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}

__device__ __forceinline__ void pdl_trigger() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}

}  // namespace imp
