// The pre-cudaDeviceReset hook registry must not be silently empty (#1207).
//
// reset_static_cuda_state() frees the lazily-created module statics (cuBLAS
// handles, CUTLASS workspaces, device scratch) before cudaDeviceReset(), so
// their `if (!ptr)` guards re-arm. Before #1207 the hooks were listed by hand
// in cuda_static_reset.cpp; each owning TU now registers itself at static-init
// time, which removes the "added a static, forgot the list entry" failure —
// and introduces a new one worth pinning: if the registrars are stripped
// (--gc-sections, a link-order accident, a TU dropped from a target), the
// registry is EMPTY and reset_static_cuda_state() becomes a no-op that still
// returns cleanly. That failure is invisible until an in-process model reload
// touches freed device memory.
//
// So: assert the registry is populated. The exact count is link-dependent —
// a test binary does not link every .cu the full engine does — hence a floor
// rather than an equality.

#include "core/cuda_static_reset.h"

#include <gtest/gtest.h>

TEST(CudaStaticReset, HookRegistryIsPopulated) {
    EXPECT_GT(imp::cuda_static_reset_hook_count(), 0)
        << "no pre-cudaDeviceReset hooks registered — reset_static_cuda_state() is a no-op, "
           "so module statics will dangle after an in-process reload";
}
