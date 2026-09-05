// AUDIT_arch_2026 D-1: a device fault must reach the host as a throw, not a
// "Cleared stale error" warning. Death test in threadsafe style: the child is
// a fresh process with its own CUDA context, so the poisoned context dies with
// it and the parent (and every later test in this binary) stays clean.
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "core/cuda_errors.h"
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace imp {
namespace {

__global__ void write_through_bad_pointer(int* p) { p[threadIdx.x] = 1; }

constexpr int kExitPoisoned = 42;

[[noreturn]] void fault_then_precheck() {
    write_through_bad_pointer<<<1, 32>>>(reinterpret_cast<int*>(0x10));
    const cudaError_t sync = cudaDeviceSynchronize();
    if (!cuda_error_is_unrecoverable(sync)) {
        std::fprintf(stderr, "sync after the fault returned %s, not a sticky class\n", cudaGetErrorString(sync));
        std::_Exit(1);
    }
    try {
        (void)cuda_clear_or_throw("forward");
    } catch (const std::runtime_error& e) {
        std::fprintf(stderr, "%s\n", e.what());
        std::_Exit(kExitPoisoned);
    }
    std::fprintf(stderr, "cuda_clear_or_throw cleared a sticky %s\n", cudaGetErrorString(sync));
    std::_Exit(2);
}

TEST(CudaFaultSignalTest, IllegalAddressIsStickyAndThrowsAtTheNextForward) {
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    EXPECT_EXIT(fault_then_precheck(), ::testing::ExitedWithCode(kExitPoisoned), "CUDA context poisoned");
}

// Control, in-process: a non-sticky API error is cleared and returned for the
// log line exactly as the old forward() pre-check did.
TEST(CudaFaultSignalTest, NonStickyErrorIsClearedNotThrown) {
    ASSERT_EQ(cudaSetDevice(9999), cudaErrorInvalidDevice);
    cudaError_t cleared = cudaSuccess;
    EXPECT_NO_THROW(cleared = cuda_clear_or_throw("forward"));
    EXPECT_EQ(cleared, cudaErrorInvalidDevice);
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);
}

}  // namespace
}  // namespace imp
