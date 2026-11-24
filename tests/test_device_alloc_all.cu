// device_alloc_all_async / device_free_all_async (memory/device_alloc_all.h): all-or-none acquisition
// and stream-ordered release of per-request metadata buffers.

#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "memory/device_alloc_all.h"

using namespace imp;

namespace {

uint64_t pool_used() {
    cudaMemPool_t pool = nullptr;
    EXPECT_EQ(cudaDeviceGetDefaultMemPool(&pool, 0), cudaSuccess);
    uint64_t used = 0;
    EXPECT_EQ(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemCurrent, &used), cudaSuccess);
    return used;
}

}  // namespace

// Three buffers and one request of 0 bytes: the three are usable, the fourth stays null, and the
// release (null included) returns the stream pool to the bytes it held before.
TEST(DeviceAllocAllGpu, AsyncTakesEveryBufferAndTheReleaseReturnsThem) {
    const uint64_t before = pool_used();
    int32_t* a = nullptr;
    int* b = nullptr;
    int* none = reinterpret_cast<int*>(uintptr_t{8});
    int* c = nullptr;
    ASSERT_EQ(device_alloc_all_async(nullptr, dev_req(a, 4096 * sizeof(int32_t)), dev_req(b, 1 << 20),
                                     dev_req(none, 0), dev_req(c, sizeof(int))),
              cudaSuccess);
    ASSERT_NE(a, nullptr);
    ASSERT_NE(b, nullptr);
    ASSERT_NE(c, nullptr);
    EXPECT_EQ(none, nullptr);
    EXPECT_GE(pool_used(), before + 4096 * sizeof(int32_t) + (1u << 20) + sizeof(int));
    EXPECT_EQ(cudaMemset(a, 0, 4096 * sizeof(int32_t)), cudaSuccess);
    EXPECT_EQ(cudaMemset(b, 0, 1 << 20), cudaSuccess);
    EXPECT_EQ(cudaMemset(c, 0, sizeof(int)), cudaSuccess);
    device_free_all_async(nullptr, a, b, none, c);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    EXPECT_EQ(pool_used(), before);
}

// The synchronous pair: device memory in use returns to its value before (64 MiB, above the noise).
TEST(DeviceAllocAllGpu, SyncReleaseReturnsTheBuffers) {
    size_t free0 = 0, free1 = 0, free2 = 0, total = 0;
    ASSERT_EQ(cudaMemGetInfo(&free0, &total), cudaSuccess);
    char* a = nullptr;
    char* none = nullptr;
    char* b = nullptr;
    const size_t each = size_t{64} << 20;
    ASSERT_EQ(device_alloc_all(dev_req(a, each), dev_req(none, 0), dev_req(b, each)), cudaSuccess);
    ASSERT_NE(a, nullptr);
    ASSERT_NE(b, nullptr);
    EXPECT_EQ(none, nullptr);
    ASSERT_EQ(cudaMemGetInfo(&free1, &total), cudaSuccess);
    EXPECT_LE(free1 + 2 * each, free0 + (size_t{8} << 20));
    device_free_all(a, none, b);
    ASSERT_EQ(cudaMemGetInfo(&free2, &total), cudaSuccess);
    EXPECT_GE(free2 + (size_t{8} << 20), free0);
}

// A request the device cannot serve: the error comes back, every pointer is null, the buffer taken
// before it is released and no error stays pending.
TEST(DeviceAllocAllGpu, AsyncFailureLeavesNothingBehind) {
    const uint64_t before = pool_used();
    int* a = nullptr;
    char* huge = nullptr;
    int* c = nullptr;
    EXPECT_NE(device_alloc_all_async(nullptr, dev_req(a, 1 << 20), dev_req(huge, size_t{1} << 46),
                                     dev_req(c, sizeof(int))),
              cudaSuccess);
    EXPECT_EQ(a, nullptr);
    EXPECT_EQ(huge, nullptr);
    EXPECT_EQ(c, nullptr);
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    EXPECT_EQ(pool_used(), before);
}
