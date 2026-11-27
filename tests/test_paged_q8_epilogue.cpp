// PagedQ8Epilogue (compute/attention_paged.h, audit D-9): the split-K reduce's Q8_1 epilogue is a
// per-call launcher argument, taken once per call at batch 1 and head_dim % 32 == 0. Before, a
// process global armed per layer stayed armed when no reduce ran and leaked into the next call.

#include <gtest/gtest.h>

#include "compute/attention_paged.h"

#include <array>
#include <cstdint>
#include <type_traits>

namespace {

using imp::paged_q8_epilogue_take;
using imp::PagedQ8Epilogue;

PagedQ8Epilogue armed() {
    static int q8_slot = 0;
    static float d8_slot = 0.0f;
    PagedQ8Epilogue e;
    e.q8 = &q8_slot;
    e.d8 = &d8_slot;
    return e;
}

TEST(PagedQ8Epilogue, TakenOncePerCall) {
    PagedQ8Epilogue e = armed();
    EXPECT_TRUE(paged_q8_epilogue_take(&e, 1, 128));
    EXPECT_TRUE(e.written);
    EXPECT_FALSE(paged_q8_epilogue_take(&e, 1, 128)) << "a second reduce of the same call";
    EXPECT_TRUE(e.written);
}

TEST(PagedQ8Epilogue, NotTakenOutsideItsShape) {
    PagedQ8Epilogue e = armed();
    EXPECT_FALSE(paged_q8_epilogue_take(&e, 2, 128)) << "batch 2";
    EXPECT_FALSE(paged_q8_epilogue_take(&e, 1, 80)) << "head_dim 80";
    EXPECT_FALSE(e.written);
    EXPECT_TRUE(paged_q8_epilogue_take(&e, 1, 64));
}

TEST(PagedQ8Epilogue, NothingArmedIsNeverTaken) {
    PagedQ8Epilogue none;
    EXPECT_FALSE(paged_q8_epilogue_take(&none, 1, 128));
    EXPECT_FALSE(none.written);
    EXPECT_FALSE(paged_q8_epilogue_take(nullptr, 1, 128));
}

// PagedDecodeArgs (audit D-11): the 10 raw-pointer paged decode launchers take one aggregate built
// with designated initializers; a declaration-order swap fails the build at those call sites.
using imp::PagedDecodeArgs;
static_assert(std::is_aggregate_v<PagedDecodeArgs>);

// Defaults a launcher may rely on when a caller leaves the field out: unscaled FP8, one split,
// no window, no softcap, no sinks, auto heads per CTA, the legacy stream.
TEST(PagedDecodeArgs, OmittedFieldsTakeTheOldDefaults) {
    const PagedDecodeArgs a{.batch_size = 1};
    EXPECT_EQ(a.kv_scale, 1.0f);
    EXPECT_EQ(a.num_splits, 1);
    EXPECT_EQ(a.sliding_window, 0);
    EXPECT_EQ(a.softcap, 0.0f);
    EXPECT_EQ(a.attn_sinks, nullptr);
    EXPECT_EQ(a.heads_per_cta, 0);
    EXPECT_EQ(a.stream, nullptr);
    EXPECT_EQ(a.K_scales, nullptr);
    EXPECT_EQ(a.partial, nullptr);
}

// PagedDecodeTensorArgs (audit D-11, D-9): the 6 Tensor-level entry points take one aggregate whose
// split-K scratch is a per-call field; omitted = null = no split-K, as the old global's (nullptr, 0).
using imp::paged_splitk_bytes;
using imp::paged_splitk_cap;
using imp::PagedDecodeTensorArgs;
using imp::PagedSplitKScratch;
static_assert(std::is_aggregate_v<PagedDecodeTensorArgs>);

TEST(PagedDecodeTensorArgs, OmittedFieldsTakeTheOldDefaults) {
    const PagedDecodeTensorArgs a{.block_size = 16};
    EXPECT_EQ(a.splitk.ptr, nullptr);
    EXPECT_EQ(a.splitk.size, 0u);
    EXPECT_EQ(a.kv_scale, 1.0f);
    EXPECT_EQ((std::array{a.sliding_window, a.max_blocks_per_seq, a.n_sinks, a.v_head_dim}),
              (std::array{0, 0, 0, 0}));
    EXPECT_EQ(a.softcap, 0.0f);
    EXPECT_EQ(a.stream, nullptr);
    EXPECT_EQ(a.attn_sinks, nullptr);
    EXPECT_EQ(a.q8_epi, nullptr);
}

// 2 x 4 heads x (2 + 128) floats = 4160 bytes per split.
TEST(PagedSplitKScratch, BytesPerSplitAreBatchHeadsTimesMlO) {
    EXPECT_EQ(paged_splitk_bytes(2, 4, 128, 1), 4160u);
    EXPECT_EQ(paged_splitk_bytes(2, 4, 128, 8), 8u * 4160u);
}

TEST(PagedSplitKScratch, NullScratchNeverSplits) {
    EXPECT_EQ(paged_splitk_cap(PagedSplitKScratch{}, 2, 4, 128, 8), 1);
    EXPECT_EQ(paged_splitk_cap(PagedSplitKScratch{.ptr = nullptr, .size = 1u << 20}, 2, 4, 128, 8), 1);
}

// A scratch too small for the wanted splits caps them to what fits, never below one split.
TEST(PagedSplitKScratch, SmallScratchCapsTheSplits) {
    static float slab[1];
    const auto fits = [](size_t bytes) { return PagedSplitKScratch{.ptr = slab, .size = bytes}; };
    EXPECT_EQ(paged_splitk_cap(fits(8 * 4160), 2, 4, 128, 8), 8);  // exactly fits
    EXPECT_EQ(paged_splitk_cap(fits(64 * 4160), 2, 4, 128, 8), 8);
    EXPECT_EQ(paged_splitk_cap(fits((3 * 4160) + 4159), 2, 4, 128, 8), 3);
    EXPECT_EQ(paged_splitk_cap(fits(4159), 2, 4, 128, 8), 1);  // less than one split
}

}  // namespace
