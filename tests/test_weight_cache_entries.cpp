// Weight-cache entries (exec/weight_cache_entries.h, invariant I5): an entry releases the buffers
// it owns exactly once and never a pointer it only views (slab slice, another entry, the model's
// weights). CPU lane: the release hook replaces the driver, the buffers are host addresses.

#include <gtest/gtest.h>

#include "exec/weight_cache_entries.h"
#include "exec/weight_caches.h"
#include "exec/weight_handle.h"
#include "memory/device_owned.h"
#include "memory/vram_allocator.h"

#include <algorithm>
#include <type_traits>
#include <utility>
#include <vector>

using namespace imp;

namespace {

struct Release {
    void* ptr;
    DeviceFamily family;
    VRAMAllocator* alloc;
};

std::vector<Release>& log() {
    static std::vector<Release> l;
    return l;
}

void record(void* ptr, DeviceFamily family, VRAMAllocator* alloc, cudaStream_t /*stream*/) {
    log().push_back({ptr, family, alloc});
}

int released(const void* p) {
    return static_cast<int>(
        std::count_if(log().begin(), log().end(), [p](const Release& r) { return r.ptr == p; }));
}

class WeightCacheEntriesTest : public ::testing::Test {
protected:
    void SetUp() override {
        log().clear();
        prev_ = set_device_owned_release_hook(&record);
    }
    void TearDown() override { set_device_owned_release_hook(prev_); }

    // Distinct addresses that are never dereferenced.
    char slots_[16] = {};
    void* at(int i) { return &slots_[i]; }
    DeviceOwned<void> plain(int i, size_t bytes = 8) {
        DeviceOwned<void> o;
        o.adopt(at(i), bytes, nullptr);
        return o;
    }
    DeviceOwned<void> vram(int i, size_t bytes = 8) {
        DeviceOwned<void> o;
        o.adopt(at(i), bytes, &va_);
        return o;
    }
    Tensor f16(void* data) {
        const int64_t shape[2] = {2, 2};
        return {data, QType::F16, 2, shape, true};
    }
    VRAMAllocator va_;  // identity only: adopt() never calls it, the hook takes the release

private:
    DeviceOwnedReleaseHook prev_ = nullptr;
};

template <typename E>
constexpr bool kMoveOnly = !std::is_copy_constructible_v<E> && !std::is_copy_assignable_v<E> &&
                           std::is_nothrow_move_constructible_v<E> && std::is_nothrow_move_assignable_v<E>;
static_assert(kMoveOnly<Fp16CacheEntry>);
static_assert(kMoveOnly<NvFP4CacheEntry>);
static_assert(kMoveOnly<NvFP4MoECacheEntry>);
static_assert(kMoveOnly<CutlassNvFP4CacheEntry>);
static_assert(kMoveOnly<CutlassMxFP8CacheEntry>);
static_assert(kMoveOnly<CutlassMxFP4CacheEntry>);
static_assert(!std::is_copy_constructible_v<WeightCaches>);

}  // namespace

TEST_F(WeightCacheEntriesTest, Fp16MoveLeavesTheSourceEmptyAndReleasesOnce) {
    {
        Fp16CacheEntry a(f16(at(0)), vram(0));
        Fp16CacheEntry b(std::move(a));
        EXPECT_EQ(a.data, nullptr);  // NOLINT(bugprone-use-after-move): the moved-from state is the subject
        EXPECT_FALSE(a.owned);
        EXPECT_EQ(b.data, at(0));
        Fp16CacheEntry c;
        c = std::move(b);
        EXPECT_EQ(b.data, nullptr);  // NOLINT(bugprone-use-after-move)
        EXPECT_EQ(c.data, at(0));
        EXPECT_TRUE(log().empty());
    }
    ASSERT_EQ(log().size(), 1u);
    EXPECT_EQ(log()[0].ptr, at(0));
    EXPECT_EQ(log()[0].family, DeviceFamily::Vram);
    EXPECT_EQ(log()[0].alloc, &va_);
}

TEST_F(WeightCacheEntriesTest, Fp16MapClearReleasesOwnedCopiesAndNeverABulkSlice) {
    WeightCaches wc;
    wc.fp16_bulk_data = plain(8, 4);
    wc.fp16[at(1)] = Fp16CacheEntry(f16(at(0)), vram(0));  // private copy
    wc.fp16[at(2)] = Fp16CacheEntry(f16(at(8)));           // slice of the bulk
    wc.fp16[at(3)] = Fp16CacheEntry(f16(at(9)));           // slice of the bulk
    wc.fp16.clear();
    ASSERT_EQ(log().size(), 1u);
    EXPECT_EQ(released(at(0)), 1);
    wc.fp16_bulk_data.reset();
    ASSERT_EQ(log().size(), 2u);
    EXPECT_EQ(released(at(8)), 1);
    EXPECT_EQ(log()[1].family, DeviceFamily::Plain);
    EXPECT_EQ(released(at(9)), 0);
}

TEST_F(WeightCacheEntriesTest, MapOverwriteAndEraseReleaseTheOldEntryOnce) {
    WeightCaches wc;
    wc.fp16[at(1)] = Fp16CacheEntry(f16(at(0)), vram(0));
    wc.fp16[at(1)] = Fp16CacheEntry(f16(at(2)), vram(2));  // replacement
    EXPECT_EQ(released(at(0)), 1);
    EXPECT_EQ(released(at(2)), 0);
    wc.fp16.erase(at(1));  // eviction
    EXPECT_EQ(released(at(2)), 1);
    EXPECT_EQ(log().size(), 2u);
}

TEST_F(WeightCacheEntriesTest, Nvfp4TakesAnOwnedResultAndViewsABorrowedOne) {
    NvFP4QuantResult q;
    q.packed_data = at(0);
    q.micro_scales = at(1);
    q.N = 4;
    q.K = 16;
    q.tensor_scale = 0.5f;
    NvFP4QuantResult borrowed = q;
    borrowed.packed_data = at(2);
    borrowed.micro_scales = at(3);
    borrowed.owned = false;
    {
        NvFP4CacheEntry own(std::move(q));
        EXPECT_EQ(q.packed_data, nullptr);  // NOLINT(bugprone-use-after-move): emptied by the hand-off
        EXPECT_FALSE(q.owned);
        EXPECT_FALSE(own.owned);  // a sliced copy of the base is a view
        EXPECT_EQ(own.packed_data, at(0));
        EXPECT_FLOAT_EQ(own.tensor_scale, 0.5f);
        EXPECT_EQ(own.packed_owned.bytes(), 32u);
        NvFP4CacheEntry view(std::move(borrowed));
        EXPECT_FALSE(view.packed_owned);
        EXPECT_EQ(view.micro_scales, at(3));
        NvFP4CacheEntry moved(std::move(own));
        EXPECT_EQ(own.packed_data, nullptr);  // NOLINT(bugprone-use-after-move)
        EXPECT_FALSE(own.owned);
    }
    ASSERT_EQ(log().size(), 2u);
    EXPECT_EQ(released(at(0)), 1);
    EXPECT_EQ(released(at(1)), 1);
    EXPECT_EQ(log()[0].family, DeviceFamily::Plain);
}

TEST_F(WeightCacheEntriesTest, Nvfp4MoeOwnersReleaseThroughTheirFamilyAndABorrowedPointerNever) {
    NvFP4MoEQuantResult layout;
    layout.packed_data = at(0);   // the model's contiguous upload
    layout.micro_scales = at(1);  // contiguous copy
    layout.tensor_scales = reinterpret_cast<float*>(at(4));
    layout.n_experts = 2;
    DeviceOwned<float> ts;
    ts.adopt(reinterpret_cast<float*>(at(4)), 8, &va_);
    {
        NvFP4MoECacheEntry zero_copy(layout, DeviceOwned<void>{}, vram(1), std::move(ts));
        EXPECT_TRUE(zero_copy.borrowed);
        NvFP4MoECacheEntry moved(std::move(zero_copy));
        EXPECT_EQ(zero_copy.packed_data, nullptr);  // NOLINT(bugprone-use-after-move)
    }
    ASSERT_EQ(log().size(), 2u);
    EXPECT_EQ(released(at(0)), 0);
    EXPECT_EQ(released(at(1)), 1);
    EXPECT_EQ(released(at(4)), 1);
    EXPECT_EQ(log()[0].family, DeviceFamily::Vram);

    log().clear();
    NvFP4MoEQuantResult converted = layout;  // a converter result: plain family, all three owned
    converted.expert_stride_packed = 4;
    converted.expert_stride_ms = 2;
    { NvFP4MoECacheEntry e(std::move(converted)); }
    ASSERT_EQ(log().size(), 3u);
    EXPECT_EQ(log()[0].family, DeviceFamily::Plain);
}

TEST_F(WeightCacheEntriesTest, CutlassEntriesSliceASlabThatIsReleasedOnceAfterThem) {
    WeightCaches wc;
    wc.cutlass_sf_slab = plain(8, 4);
    wc.owned_sf_slabs.push_back(vram(12, 4));
    CutlassNvFP4Weight slab_slice;
    slab_slice.data = at(0);
    slab_slice.scale_factors = at(9);
    slab_slice.sf_borrowed = true;
    CutlassNvFP4Weight moe_slice = slab_slice;
    moe_slice.scale_factors = at(13);
    CutlassNvFP4Weight own;  // convert_nvfp4_to_cutlass: owns its SfAtom buffer
    own.data = at(0);
    own.scale_factors = at(5);
    own.sf_bytes = 4;
    wc.cutlass_nvfp4[at(1)] = CutlassNvFP4CacheEntry(std::move(slab_slice));
    wc.cutlass_nvfp4[at(2)] = CutlassNvFP4CacheEntry(std::move(moe_slice));
    wc.cutlass_nvfp4[at(3)] = CutlassNvFP4CacheEntry(std::move(own));
    EXPECT_TRUE(wc.cutlass_nvfp4.at(at(3)).sf_borrowed);  // the flag reads "not mine" in an entry
    wc.cutlass_nvfp4.clear();
    ASSERT_EQ(log().size(), 1u);
    EXPECT_EQ(released(at(5)), 1);
    EXPECT_EQ(released(at(0)), 0);  // data is always a view
    wc.cutlass_sf_slab.reset();
    wc.owned_sf_slabs.clear();
    ASSERT_EQ(log().size(), 3u);
    EXPECT_EQ(released(at(8)), 1);
    EXPECT_EQ(released(at(12)), 1);
    EXPECT_EQ(log()[2].family, DeviceFamily::Vram);
}

TEST_F(WeightCacheEntriesTest, Mxfp8RowViewReleasesItsScaleBlockAndNotTheBorrowedRows) {
    CutlassMxFP8Weight rows;
    rows.data = at(0);  // rows of the FP8 sidecar
    rows.scale_factors = at(1);
    rows.borrowed_data = true;
    CutlassMxFP8Weight copy;
    copy.data = at(2);
    copy.scale_factors = at(3);
    {
        CutlassMxFP8CacheEntry view(rows, DeviceOwned<void>{}, vram(1));
        CutlassMxFP8CacheEntry own(copy, vram(2), vram(3));
        CutlassMxFP8CacheEntry moved(std::move(view));
        EXPECT_EQ(view.data, nullptr);  // NOLINT(bugprone-use-after-move)
        EXPECT_TRUE(moved.borrowed_data);
    }
    EXPECT_EQ(log().size(), 3u);
    EXPECT_EQ(released(at(0)), 0);
    EXPECT_EQ(released(at(1)), 1);
    EXPECT_EQ(released(at(2)), 1);
    EXPECT_EQ(released(at(3)), 1);
}

TEST_F(WeightCacheEntriesTest, Mxfp4InPlaceDataIsAViewBothScaleLayoutsAreOwned) {
    CutlassMxFP4Weight w;
    w.data = at(0);  // compacted in place in the model's buffer
    w.scale_factors = at(1);
    w.linear_scales = at(2);
    w.N = 2;
    w.K = 32;
    w.hadamard_bs = 7;
    {
        CutlassMxFP4CacheEntry e(std::move(w));
        EXPECT_EQ(w.scale_factors, nullptr);  // NOLINT(bugprone-use-after-move): emptied by the hand-off
        EXPECT_EQ(e.hadamard_bs, 7);
        EXPECT_FALSE(e.data_owned);
    }
    EXPECT_EQ(log().size(), 2u);
    EXPECT_EQ(released(at(0)), 0);
    EXPECT_EQ(released(at(1)), 1);
    EXPECT_EQ(released(at(2)), 1);
}

TEST_F(WeightCacheEntriesTest, RegistryTakesAFusedOverlayAndReleasesItOnce) {
    WeightCaches wc;
    WeightRegistry reg;
    wc.fused_kv[0] = Fp16CacheEntry(f16(at(0)), vram(0, 16));
    const TensorID id = reg.reserve(TensorKind::FUSED_KV, 2, 2);
    reg.adopt_fp16_storage(id, std::move(wc.fused_kv[0].owned));
    wc.fused_kv.clear();  // a view now: releases nothing
    EXPECT_TRUE(log().empty());
    EXPECT_EQ(reg.handle(id).payload.fp16.data, at(0));
    EXPECT_EQ(reg.handle(id).owned_bytes, 16);
    EXPECT_EQ(reg.free_owned_storage(), 16u);
    EXPECT_EQ(reg.handle(id).payload.fp16.data, nullptr);
    EXPECT_EQ(reg.free_owned_storage(), 0u);  // idempotent
    ASSERT_EQ(log().size(), 1u);
    EXPECT_EQ(released(at(0)), 1);
}
