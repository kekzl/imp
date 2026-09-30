#pragma once

// Programmatic Dependent Launch, device half. No-ops for a non-programmatic launch and for the
// compute_120f fallback. Contract for every kernel registered via pdl::enable():
// - before pdl_wait() (griddepcontrol.wait): global reads of immutable data (weights) and
//   prefetch.global.L2 only. Any mutable buffer, even one written many kernels earlier, waits: a
//   predecessor that triggers before its own wait (smallm_v2) lets this grid start while older grids
//   still run, so "not the immediate predecessor" proves nothing (#2340).
// - pdl_wait() returns after the predecessor grid completed and flushed, which transitively covers
//   every earlier grid, since each registered grid waits before it completes.
// - pdl_trigger() (griddepcontrol.launch_dependents) affects scheduling only, never visibility:
//   given the rule above it may sit anywhere, also before this grid's own wait.

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
