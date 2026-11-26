// open_engine_stream (audit C-11): Engine::init refuses to load when its stream cannot be
// created instead of serving on the legacy default stream. The CPU lane has no device, so
// the create fails there: the helper must report it and leave the stream null.

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include "runtime/engine_internal.h"

namespace {

TEST(EngineStreamOpen, ResultMatchesTheStream) {
    imp::CudaStream s;
    const bool ok = imp::engine_internal::open_engine_stream(s);
    EXPECT_EQ(ok, s.get() != nullptr);
    (void)cudaGetLastError();
}

TEST(EngineStreamOpen, NoDeviceFailsTheOpen) {
    int n = 0;
    if (cudaGetDeviceCount(&n) == cudaSuccess && n > 0)
        GTEST_SKIP() << "a device is present: the create succeeds";
    (void)cudaGetLastError();
    imp::CudaStream s;
    EXPECT_FALSE(imp::engine_internal::open_engine_stream(s));
    EXPECT_EQ(s.get(), nullptr);
    (void)cudaGetLastError();
}

}  // namespace
