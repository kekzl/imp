#pragma once

// Programmatic Dependent Launch, device half.
//
// The host half (runtime/pdl.h: the launch attribute, and the graph-edge
// rewrite in runtime/cuda_graph.cu) lets a registered kernel be SCHEDULED
// while its predecessor is still running. That is only correct when the
// registered kernel calls pdl_wait() before it touches any global memory a
// predecessor may still be reading or writing: griddepcontrol.wait blocks
// until every prerequisite grid has completed and its memory is visible.
// pdl_trigger() in a producer (griddepcontrol.launch_dependents) lets the
// dependent grid launch once every producer block has triggered or exited;
// it changes scheduling only, never visibility, so it sits after the block's
// last input read, before the epilogue stores.
//
// Both are no-ops for a kernel launched without a programmatic dependency
// and for the compute_120f PTX fallback (sm_90+ instructions, guarded).
// Contract: every kernel registered with pdl::enable() calls pdl_wait()
// first; a kernel that does not wait must never be registered.

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
