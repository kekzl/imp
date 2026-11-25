// device_alloc_all_async / device_free_all_async (memory/device_alloc_all.h): all-or-none acquisition
// and stream-ordered release of per-request metadata buffers.

#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "memory/device_alloc_all.h"
#include "memory/device_owned.h"
#include "memory/vram_allocator.h"

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

// DeviceOwned on the real driver, one case per family (ownership logic: tests/test_device_owned.cpp).
// Vram: the allocator's tracked bytes return to their value before, also for the forced variant.
TEST(DeviceOwnedGpu, AllocatorFamilyReturnsTheTrackedBytes) {
    VRAMAllocator va;
    const size_t before = va.allocated();
    {
        auto a = DeviceOwned<float>::allocate(&va, 1 << 20, "device_owned_test");
        auto b = DeviceOwned<void>::allocate(&va, 1 << 16, "device_owned_test", /*force=*/true);
        ASSERT_TRUE(a);
        ASSERT_TRUE(b);
        EXPECT_EQ(a.family(), DeviceFamily::Vram);
        EXPECT_EQ(va.allocated(), before + (1 << 20) + (1 << 16));
        EXPECT_EQ(cudaMemset(a.get(), 0, a.bytes()), cudaSuccess);
        b.reset();
        EXPECT_EQ(va.allocated(), before + (1 << 20));
    }
    EXPECT_EQ(va.allocated(), before);
}

// Plain: no allocator, the driver's ledger returns to its value before.
TEST(DeviceOwnedGpu, PlainFamilyReturnsTheLedgerBytes) {
    const size_t before = device_alloc_live_bytes();
    {
        auto a = DeviceOwned<char>::allocate(nullptr, 1 << 20, "unused");
        auto b = DeviceOwned<char>::allocate_plain(1 << 16);
        ASSERT_TRUE(a);
        ASSERT_TRUE(b);
        EXPECT_EQ(a.family(), DeviceFamily::Plain);
        EXPECT_EQ(device_alloc_live_bytes(), before + (1 << 20) + (1 << 16));
    }
    EXPECT_EQ(device_alloc_live_bytes(), before);
}

// Async: the stream pool returns to its value before; an adopted pair of one all-or-none call too.
TEST(DeviceOwnedGpu, AsyncFamilyReturnsThePoolBytes) {
    const uint64_t before = pool_used();
    {
        auto a = DeviceOwned<int>::allocate_async(nullptr, 1 << 20);
        ASSERT_TRUE(a);
        EXPECT_EQ(a.family(), DeviceFamily::Async);
        int *k = nullptr, *v = nullptr;
        ASSERT_EQ(device_alloc_all_async(nullptr, dev_req(k, 1 << 16), dev_req(v, 1 << 16)), cudaSuccess);
        DeviceOwned<int> ko, vo;
        ko.adopt_async(k, 1 << 16, nullptr);
        vo.adopt_async(v, 1 << 16, nullptr);
        EXPECT_GE(pool_used(), before + (1 << 20) + (2 << 16));
    }
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    EXPECT_EQ(pool_used(), before);
}
