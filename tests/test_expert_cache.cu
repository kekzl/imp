// Phase 2 + Phase 3 (MoE host-offload + CUDA Graphs design): unit tests for
// the ExpertLRUCache device-side mirror + per-layer slot pool partitioning.
//
// Phase 2 invariant: every host-LRU mutation is mirrored into a device-side
// int32 table sized [n_layers × 3 × n_experts]. Cell value is the
// layer-relative slot index (Phase 3 narrows this from the old global slot
// index) or -1 if not cached.
//
// Phase 3 invariant: the slot pool is partitioned per-layer so layer L's
// cache state cannot be evicted by layer M's misses. Each layer owns
// `slots_per_layer_` slots inside the shared pool_, with independent LRU
// recency + key→slot maps.

#include <gtest/gtest.h>
#include "graph/executor.h"
#include <cuda_runtime.h>
#include <cstring>
#include <cstdint>
#include <vector>

namespace imp {
namespace {

class ExpertCachePhase3Test : public ::testing::Test {
   protected:
    // Pick a tiny budget that yields exactly 2 slots per layer with 3
    // layers — enough to test eviction within a layer (2 inserts then a
    // 3rd in the same layer evicts the LRU) and per-layer isolation
    // (filling layer 0 doesn't touch layer 1).
    static constexpr int kSlotBytes = 64;
    static constexpr int kNLayers = 3;
    static constexpr int kSlotsPerLayer = 2;
    static constexpr int kBudgetBytes = kNLayers * kSlotsPerLayer * kSlotBytes;
    static constexpr int kNExperts = 8;

    void SetUp() override {
        ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream_));
        ASSERT_EQ(cudaSuccess, cudaHostAlloc(&src_, kSlotBytes, cudaHostAllocDefault));
        std::memset(src_, 0xAB, kSlotBytes);
        ASSERT_TRUE(cache_.init(kSlotBytes, kBudgetBytes, /*alloc=*/nullptr, kNLayers, kNExperts,
                                /*debug_parity=*/true));
        ASSERT_EQ(cache_.slots_per_layer_, kSlotsPerLayer);
        ASSERT_EQ(cache_.n_slots_, kNLayers * kSlotsPerLayer);
        ASSERT_NE(cache_.d_lookup_, nullptr);
    }

    void TearDown() override {
        cache_.destroy();
        cudaFreeHost(src_);
        cudaStreamDestroy(stream_);
    }

    // Returns the layer-relative slot_idx at the (layer, proj, expert) cell,
    // or -1 if not cached.
    int read_mirror_cell(int layer, int proj, int expert) {
        size_t off = (static_cast<size_t>(layer) * kExpertProjCount + proj) * kNExperts + expert;
        int slot_idx = 0;
        cudaMemcpyAsync(&slot_idx, cache_.d_lookup_ + off, sizeof(int), cudaMemcpyDeviceToHost,
                        stream_);
        cudaStreamSynchronize(stream_);
        return slot_idx;
    }

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

    ExpertLRUCache cache_;
    void* src_ = nullptr;
    cudaStream_t stream_ = nullptr;
};

TEST_F(ExpertCachePhase3Test, InitZerosMirrorToMinusOne) {
    for (int l = 0; l < kNLayers; ++l)
        for (int p = 0; p < kExpertProjCount; ++p)
            for (int e = 0; e < kNExperts; ++e)
                EXPECT_EQ(read_mirror_cell(l, p, e), -1) << "(l=" << l << ", p=" << p
                                                         << ", e=" << e << ")";
}

TEST_F(ExpertCachePhase3Test, SingleInsertMirrorsToSlotZero) {
    void* ptr = load(/*layer=*/1, ExpertProj::Up, /*expert=*/3);
    EXPECT_NE(ptr, nullptr);
    // First insert in layer 1 picks layer-relative slot 0.
    EXPECT_EQ(read_mirror_cell(1, static_cast<int>(ExpertProj::Up), 3), 0);
    // Other cells stay -1.
    EXPECT_EQ(read_mirror_cell(0, 0, 0), -1);
    EXPECT_EQ(read_mirror_cell(2, 2, 7), -1);
    EXPECT_TRUE(cache_.check_parity(stream_));
}

TEST_F(ExpertCachePhase3Test, TwoInsertsPickUnoccupiedSlotsSameLayer) {
    load(0, ExpertProj::Gate, 0);
    load(0, ExpertProj::Up, 1);
    EXPECT_EQ(read_mirror_cell(0, 0, 0), 0);
    EXPECT_EQ(read_mirror_cell(0, 1, 1), 1);
    EXPECT_EQ(cache_.misses_, 2);
    EXPECT_EQ(cache_.hits_, 0);
    EXPECT_TRUE(cache_.check_parity(stream_));
}

TEST_F(ExpertCachePhase3Test, ThirdInsertSameLayerEvictsLRU) {
    load(0, ExpertProj::Gate, 0);    // -> slot 0 in layer 0
    load(0, ExpertProj::Up, 1);      // -> slot 1 in layer 0
    EXPECT_TRUE(cache_.check_parity(stream_));
    // Layer 0 is now full (2/2 occupied). A third insert in layer 0 evicts
    // slot 0 (the LRU): the Gate/0 cell goes to -1 and Down/2 takes slot 0.
    load(0, ExpertProj::Down, 2);
    EXPECT_EQ(read_mirror_cell(0, 0, 0), -1);
    EXPECT_EQ(read_mirror_cell(0, 1, 1), 1);
    EXPECT_EQ(read_mirror_cell(0, 2, 2), 0);
    EXPECT_TRUE(cache_.check_parity(stream_));
}

TEST_F(ExpertCachePhase3Test, PerLayerIsolation) {
    // Filling layer 0 must not affect layer 1's slots.
    load(0, ExpertProj::Gate, 0);
    load(0, ExpertProj::Up, 1);
    load(0, ExpertProj::Down, 2);  // would evict layer 0's slot 0 (LRU)
    // Layer 1 still untouched.
    for (int p = 0; p < kExpertProjCount; ++p)
        for (int e = 0; e < kNExperts; ++e)
            EXPECT_EQ(read_mirror_cell(1, p, e), -1);
    // Now insert into layer 1 — slot allocation starts fresh.
    load(1, ExpertProj::Gate, 0);
    EXPECT_EQ(read_mirror_cell(1, 0, 0), 0);
    // Layer 0's state still intact.
    EXPECT_EQ(read_mirror_cell(0, 1, 1), 1);
    EXPECT_EQ(read_mirror_cell(0, 2, 2), 0);
    EXPECT_TRUE(cache_.check_parity(stream_));
}

TEST_F(ExpertCachePhase3Test, HitDoesNotMoveSlot) {
    void* p0 = load(0, ExpertProj::Gate, 0);
    void* p1 = load(0, ExpertProj::Gate, 0);  // hit on same key
    EXPECT_EQ(p0, p1);
    EXPECT_EQ(cache_.misses_, 1);
    EXPECT_EQ(cache_.hits_, 1);
    EXPECT_EQ(read_mirror_cell(0, 0, 0), 0);
    EXPECT_TRUE(cache_.check_parity(stream_));
}

TEST_F(ExpertCachePhase3Test, AccessMovesToFrontResistsEviction) {
    load(0, ExpertProj::Gate, 0);  // slot 0
    load(0, ExpertProj::Up, 1);    // slot 1
    load(0, ExpertProj::Gate, 0);  // hit → slot 0 becomes MRU
    // Next miss in layer 0 should evict slot 1 (now LRU), not slot 0.
    load(0, ExpertProj::Down, 2);
    EXPECT_EQ(read_mirror_cell(0, 0, 0), 0);   // still resident
    EXPECT_EQ(read_mirror_cell(0, 1, 1), -1);  // evicted
    EXPECT_EQ(read_mirror_cell(0, 2, 2), 1);   // reused slot 1
    EXPECT_TRUE(cache_.check_parity(stream_));
}

TEST_F(ExpertCachePhase3Test, ParityCheckCounterAdvances) {
    int64_t before = cache_.parity_checks_ok_;
    load(0, ExpertProj::Gate, 0);
    load(0, ExpertProj::Up, 1);
    load(1, ExpertProj::Down, 2);
    load(2, ExpertProj::Gate, 3);
    EXPECT_GE(cache_.parity_checks_ok_ - before, 4);
}

TEST_F(ExpertCachePhase3Test, MirrorDisabledWhenNoExperts) {
    // Re-init with n_experts=0 → mirror skipped, host LRU still works.
    ExpertLRUCache c;
    ASSERT_TRUE(c.init(kSlotBytes, kBudgetBytes, /*alloc=*/nullptr, kNLayers,
                       /*n_experts=*/0));
    EXPECT_EQ(c.d_lookup_, nullptr);
    EXPECT_TRUE(c.host_expert_addrs_.empty());
    ExpertCacheKey key{fake_packed_ptr(0, 0), 0};
    void* p = c.get_or_load(0, ExpertProj::Gate, key, src_, kSlotBytes, stream_);
    EXPECT_NE(p, nullptr);
    EXPECT_TRUE(c.check_parity(stream_));  // trivially true when mirror is null
    c.destroy();
}

TEST_F(ExpertCachePhase3Test, HostExpertAddrsLazyPopulation) {
    // Before any get_or_load: every cell of host_expert_addrs_ is nullptr.
    ASSERT_EQ(static_cast<int>(cache_.host_expert_addrs_.size()), kNLayers);
    for (const auto& row : cache_.host_expert_addrs_)
        for (const void* p : row)
            EXPECT_EQ(p, nullptr);

    // First load stamps the cell.
    load(1, ExpertProj::Up, 3);
    size_t off = static_cast<size_t>(ExpertProj::Up) * kNExperts + 3;
    EXPECT_EQ(cache_.host_expert_addrs_[1][off], src_);
    // Same-layer different-cell is still unset.
    size_t off_other = static_cast<size_t>(ExpertProj::Down) * kNExperts + 0;
    EXPECT_EQ(cache_.host_expert_addrs_[1][off_other], nullptr);
}

TEST_F(ExpertCachePhase3Test, SlotPtrAgreesWithGpuPtr) {
    void* p = load(2, ExpertProj::Up, 5);
    int slot_in_layer = read_mirror_cell(2, static_cast<int>(ExpertProj::Up), 5);
    ASSERT_GE(slot_in_layer, 0);
    EXPECT_EQ(cache_.slot_ptr(2, slot_in_layer), p);
}

}  // namespace
}  // namespace imp
