// RecurrentSnapshotStore — hybrid (SSM/GDN) prefix-cache snapshots.
//
// The store is plain LRU bookkeeping over fixed-size device buffers; what
// needs guarding is the lifetime contract: an entry handed out to a request
// (shared_ptr) must keep its device buffer valid across eviction, and the
// buffer must be recycled — not leaked — once the holder releases it.

#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include "memory/recurrent_snapshot_store.h"

#include <cstdint>
#include <vector>

#include "test_cuda_skip.h"

namespace imp {
namespace {

constexpr size_t kEntryBytes = 4096;

// Device buffer filled with a marker byte, for save() sources.
struct DeviceSrc {
    void* d = nullptr;
    explicit DeviceSrc(uint8_t marker) {
        EXPECT_EQ(cudaMalloc(&d, kEntryBytes), cudaSuccess);
        std::vector<uint8_t> h(kEntryBytes, marker);
        EXPECT_EQ(cudaMemcpy(d, h.data(), kEntryBytes, cudaMemcpyHostToDevice), cudaSuccess);
    }
    ~DeviceSrc() { cudaFree(d); }
};

static std::vector<uint8_t> ReadEntry(const RecurrentSnapshotEntry& e) {
    std::vector<uint8_t> h(kEntryBytes);
    EXPECT_EQ(cudaMemcpy(h.data(), e.data, kEntryBytes, cudaMemcpyDeviceToHost), cudaSuccess);
    return h;
}

TEST(RecurrentSnapshotStoreTest, SaveFindRoundTrip) {
    SKIP_IF_NO_CUDA();
    RecurrentSnapshotStore store;
    store.init(kEntryBytes, 4 * kEntryBytes);
    ASSERT_TRUE(store.enabled());
    ASSERT_EQ(store.capacity(), 4);

    DeviceSrc src(0xAB);
    ASSERT_TRUE(store.save(/*key=*/111, /*n_tokens=*/32, src.d, /*stream=*/nullptr));
    cudaStreamSynchronize(nullptr);

    auto e = store.find(111);
    ASSERT_NE(e, nullptr);
    EXPECT_EQ(e->n_tokens, 32);
    EXPECT_EQ(ReadEntry(*e), std::vector<uint8_t>(kEntryBytes, 0xAB));
    EXPECT_EQ(store.find(222), nullptr);

    // Duplicate key: no-op success (identical prefix already snapshotted).
    DeviceSrc other(0xCD);
    EXPECT_TRUE(store.save(111, 32, other.d, nullptr));
    cudaStreamSynchronize(nullptr);
    EXPECT_EQ(ReadEntry(*store.find(111)), std::vector<uint8_t>(kEntryBytes, 0xAB))
        << "duplicate save must not overwrite the stored snapshot";
}

TEST(RecurrentSnapshotStoreTest, LRUEvictionAtCapacity) {
    SKIP_IF_NO_CUDA();
    RecurrentSnapshotStore store;
    store.init(kEntryBytes, 2 * kEntryBytes);  // capacity 2
    ASSERT_EQ(store.capacity(), 2);

    DeviceSrc a(0x01), b(0x02), c(0x03);
    ASSERT_TRUE(store.save(1, 16, a.d, nullptr));
    ASSERT_TRUE(store.save(2, 16, b.d, nullptr));
    // Touch key 1 → key 2 becomes LRU.
    ASSERT_NE(store.find(1), nullptr);
    ASSERT_TRUE(store.save(3, 16, c.d, nullptr));
    cudaStreamSynchronize(nullptr);

    EXPECT_EQ(store.find(2), nullptr) << "LRU entry must have been evicted";
    ASSERT_NE(store.find(1), nullptr);
    ASSERT_NE(store.find(3), nullptr);
    EXPECT_EQ(store.size(), 2);
    EXPECT_EQ(ReadEntry(*store.find(3)), std::vector<uint8_t>(kEntryBytes, 0x03));
}

TEST(RecurrentSnapshotStoreTest, HeldEntrySurvivesEvictionThenRecycles) {
    SKIP_IF_NO_CUDA();
    RecurrentSnapshotStore store;
    store.init(kEntryBytes, kEntryBytes);  // capacity 1
    ASSERT_EQ(store.capacity(), 1);

    DeviceSrc a(0x11), b(0x22);
    ASSERT_TRUE(store.save(1, 16, a.d, nullptr));
    auto held = store.find(1);  // request holds the entry across eviction
    ASSERT_NE(held, nullptr);

    // Capacity 1 + buffer held: the save must fail (no buffer available),
    // and the held entry must stay intact — never overwritten in place.
    EXPECT_FALSE(store.save(2, 16, b.d, nullptr));
    cudaStreamSynchronize(nullptr);
    EXPECT_EQ(ReadEntry(*held), std::vector<uint8_t>(kEntryBytes, 0x11));

    // Release → buffer recycles → the next save succeeds.
    held.reset();
    EXPECT_TRUE(store.save(2, 16, b.d, nullptr));
    cudaStreamSynchronize(nullptr);
    auto e2 = store.find(2);
    ASSERT_NE(e2, nullptr);
    EXPECT_EQ(ReadEntry(*e2), std::vector<uint8_t>(kEntryBytes, 0x22));
}

TEST(RecurrentSnapshotStoreTest, ClearDropsEntriesButHeldBufferStaysValid) {
    SKIP_IF_NO_CUDA();
    RecurrentSnapshotStore store;
    store.init(kEntryBytes, 2 * kEntryBytes);

    DeviceSrc a(0x33);
    ASSERT_TRUE(store.save(7, 48, a.d, nullptr));
    auto held = store.find(7);
    ASSERT_NE(held, nullptr);

    store.clear();
    EXPECT_EQ(store.find(7), nullptr);
    EXPECT_EQ(store.size(), 0);
    cudaStreamSynchronize(nullptr);
    EXPECT_EQ(ReadEntry(*held), std::vector<uint8_t>(kEntryBytes, 0x33))
        << "held entry must stay valid after clear()";
    held.reset();  // recycles into the pool, freed by the store dtor
}

}  // namespace
}  // namespace imp
