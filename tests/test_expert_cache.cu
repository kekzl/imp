// Phase 2 (MoE host-offload + CUDA Graphs design): unit tests for the
// ExpertLRUCache device-side mirror.
//
// The host-side LRU is untouched — these tests exercise the new device
// `d_lookup_` table, asserting it stays in sync with the host slot table
// across hits, misses, evictions, and eviction-then-re-insert cycles.
// Parity is the Phase 2 invariant — Phase 3 will start consuming the
// device table from dispatch kernels.

#include <gtest/gtest.h>
#include "graph/executor.h"
#include <cuda_runtime.h>
#include <cstring>
#include <cstdint>
#include <vector>

namespace imp {
namespace {

// Synthetic test fixture: build an ExpertLRUCache without any model, drive
// it through (layer, proj, expert) keys, and check the device-side mirror
// agrees with the host-side LRU state after every operation.
class ExpertCachePhase2Test : public ::testing::Test {
   protected:
    static constexpr int kSlotBytes = 64;       // tiny — we don't care about contents
    static constexpr int kBudgetBytes = 4 * kSlotBytes;  // 4 slots
    static constexpr int kNLayers = 3;
    static constexpr int kNExperts = 8;

    void SetUp() override {
        ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream_));
        // Source bytes: we don't care what — just enough to memcpy from
        // pinned host memory to a slot. Pinned because LRU uses cudaMemcpyAsync.
        ASSERT_EQ(cudaSuccess, cudaHostAlloc(&src_, kSlotBytes, cudaHostAllocDefault));
        std::memset(src_, 0xAB, kSlotBytes);
        ASSERT_TRUE(cache_.init(kSlotBytes, kBudgetBytes, /*alloc=*/nullptr, kNLayers, kNExperts,
                                /*debug_parity=*/true));
        ASSERT_EQ(cache_.n_slots_, 4);
        ASSERT_NE(cache_.d_lookup_, nullptr);
    }

    void TearDown() override {
        cache_.destroy();
        cudaFreeHost(src_);
        cudaStreamDestroy(stream_);
    }

    // Reads the device-side mirror back to host and returns the slot_idx at
    // (layer, proj, expert). -1 means "not cached".
    int read_mirror_cell(int layer, int proj, int expert) {
        size_t off = (static_cast<size_t>(layer) * kExpertProjCount + proj) * kNExperts + expert;
        int slot_idx = 0;
        cudaMemcpyAsync(&slot_idx, cache_.d_lookup_ + off, sizeof(int), cudaMemcpyDeviceToHost,
                        stream_);
        cudaStreamSynchronize(stream_);
        return slot_idx;
    }

    ExpertLRUCache cache_;
    void* src_ = nullptr;
    cudaStream_t stream_ = nullptr;

    // Synthetic ExpertCacheKey — packed_ptr just needs to differ per (layer, proj).
    // Cast a small integer to const void* — never dereferenced.
    static const void* fake_packed_ptr(int layer, int proj) {
        uintptr_t v = (static_cast<uintptr_t>(layer + 1) << 4) | static_cast<uintptr_t>(proj + 1);
        return reinterpret_cast<const void*>(v);
    }

    void* load(int layer, ExpertProj proj, int expert) {
        ExpertCacheKey key{fake_packed_ptr(layer, static_cast<int>(proj)), expert};
        return cache_.get_or_load(layer, proj, key, src_, kSlotBytes, stream_);
    }
};

TEST_F(ExpertCachePhase2Test, InitZerosMirrorToMinusOne) {
    // After init, every cell should read -1.
    for (int l = 0; l < kNLayers; ++l)
        for (int p = 0; p < kExpertProjCount; ++p)
            for (int e = 0; e < kNExperts; ++e)
                EXPECT_EQ(read_mirror_cell(l, p, e), -1) << "(l=" << l << ", p=" << p
                                                         << ", e=" << e << ")";
}

TEST_F(ExpertCachePhase2Test, SingleInsertMirrorsToSlotZero) {
    void* ptr = load(/*layer=*/1, ExpertProj::Up, /*expert=*/3);
    EXPECT_NE(ptr, nullptr);
    EXPECT_EQ(read_mirror_cell(1, static_cast<int>(ExpertProj::Up), 3), 0);
    // All other cells stay -1.
    EXPECT_EQ(read_mirror_cell(0, 0, 0), -1);
    EXPECT_EQ(read_mirror_cell(2, 2, 7), -1);
    EXPECT_TRUE(cache_.check_parity(stream_));
}

TEST_F(ExpertCachePhase2Test, ThreeInsertsPickUnoccupiedSlots) {
    load(0, ExpertProj::Gate, 0);
    load(0, ExpertProj::Up, 0);
    load(0, ExpertProj::Down, 0);
    EXPECT_EQ(read_mirror_cell(0, 0, 0), 0);
    EXPECT_EQ(read_mirror_cell(0, 1, 0), 1);
    EXPECT_EQ(read_mirror_cell(0, 2, 0), 2);
    EXPECT_EQ(cache_.misses_, 3);
    EXPECT_EQ(cache_.hits_, 0);
    EXPECT_TRUE(cache_.check_parity(stream_));
}

TEST_F(ExpertCachePhase2Test, HitDoesNotMoveSlot) {
    void* p0 = load(0, ExpertProj::Gate, 0);
    void* p1 = load(0, ExpertProj::Gate, 0);  // hit on same key
    EXPECT_EQ(p0, p1);
    EXPECT_EQ(cache_.misses_, 1);
    EXPECT_EQ(cache_.hits_, 1);
    EXPECT_EQ(read_mirror_cell(0, 0, 0), 0);
    EXPECT_TRUE(cache_.check_parity(stream_));
}

TEST_F(ExpertCachePhase2Test, EvictionInvalidatesOldCellWritesNew) {
    // Fill all 4 slots, then trigger eviction on the 5th insert.
    load(0, ExpertProj::Gate, 0);  // -> slot 0
    load(0, ExpertProj::Up, 1);    // -> slot 1
    load(1, ExpertProj::Down, 2);  // -> slot 2
    load(2, ExpertProj::Gate, 3);  // -> slot 3
    EXPECT_TRUE(cache_.check_parity(stream_));

    // Slot 0 (layer=0, proj=Gate, expert=0) is now the LRU. The next miss
    // evicts it; that cell should read -1 and the new (layer=2, proj=Up,
    // expert=4) cell should read the freed slot index.
    load(2, ExpertProj::Up, 4);
    EXPECT_EQ(read_mirror_cell(0, 0, 0), -1);
    EXPECT_EQ(read_mirror_cell(2, 1, 4), 0);  // reused slot 0
    EXPECT_TRUE(cache_.check_parity(stream_));
}

TEST_F(ExpertCachePhase2Test, AccessMovesToFrontResistsEviction) {
    // Insert 4 entries (slots 0..3).
    load(0, ExpertProj::Gate, 0);
    load(0, ExpertProj::Up, 1);
    load(1, ExpertProj::Down, 2);
    load(2, ExpertProj::Gate, 3);

    // Touch slot 0 → it becomes most-recently-used. Next eviction should
    // hit slot 1 (now LRU), not slot 0.
    load(0, ExpertProj::Gate, 0);  // hit
    load(2, ExpertProj::Up, 4);    // miss → evict slot 1

    EXPECT_EQ(read_mirror_cell(0, 0, 0), 0);     // still resident
    EXPECT_EQ(read_mirror_cell(0, 1, 1), -1);    // evicted
    EXPECT_EQ(read_mirror_cell(2, 1, 4), 1);     // reused slot 1
    EXPECT_TRUE(cache_.check_parity(stream_));
}

TEST_F(ExpertCachePhase2Test, ParityCheckCounterAdvances) {
    // With debug_parity=true (set in SetUp), every get_or_load() runs
    // check_parity() internally. After 4 inserts the counter should be ≥4.
    int64_t before = cache_.parity_checks_ok_;
    load(0, ExpertProj::Gate, 0);
    load(0, ExpertProj::Up, 1);
    load(1, ExpertProj::Down, 2);
    load(2, ExpertProj::Gate, 3);
    EXPECT_GE(cache_.parity_checks_ok_ - before, 4);
}

TEST_F(ExpertCachePhase2Test, MirrorDisabledWhenNoLayersOrExperts) {
    // Re-init with n_layers=0 — mirror should not be allocated, but the
    // host-side LRU should still function normally.
    ExpertLRUCache c;
    ASSERT_TRUE(c.init(kSlotBytes, kBudgetBytes, /*alloc=*/nullptr, /*n_layers=*/0,
                       /*n_experts=*/kNExperts));
    EXPECT_EQ(c.d_lookup_, nullptr);
    ExpertCacheKey key{fake_packed_ptr(0, 0), 0};
    void* p = c.get_or_load(0, ExpertProj::Gate, key, src_, kSlotBytes, stream_);
    EXPECT_NE(p, nullptr);
    EXPECT_TRUE(c.check_parity(stream_));  // trivially true when mirror is null
    c.destroy();
}

}  // namespace
}  // namespace imp
