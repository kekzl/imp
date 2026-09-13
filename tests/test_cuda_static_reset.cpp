// #1207: the pre-cudaDeviceReset hook registry must not be silently empty.
// reset_static_cuda_state() frees lazily-created module statics before cudaDeviceReset() so
// their guards re-arm; each owning TU now self-registers instead of a hand-kept list, which
// removes "added a static, forgot the entry" but introduces "registrars stripped
// (--gc-sections, link-order, dropped TU) -> registry EMPTY, reset becomes a silent no-op" -
// invisible until an in-process model reload touches freed device memory.
// Asserts a floor, not an exact count: a test binary doesn't link every .cu the full engine does.

#include "core/cuda_static_reset.h"

#include <gtest/gtest.h>

TEST(CudaStaticReset, HookRegistryIsPopulated) {
    EXPECT_GT(imp::cuda_static_reset_hook_count(), 0)
        << "no pre-cudaDeviceReset hooks registered — reset_static_cuda_state() is a no-op, "
           "so module statics will dangle after an in-process reload";
}
