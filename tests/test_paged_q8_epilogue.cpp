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
// with designated initializers. 23 fields: a field added or dropped fails the binding below.
using imp::PagedDecodeArgs;
static_assert(std::is_aggregate_v<PagedDecodeArgs>);

TEST(PagedDecodeArgs, CarriesEveryFieldInDeclarationOrder) {
    static const half q[1] = {};
    static const uint8_t kc[1] = {}, vc[1] = {}, ks[1] = {}, vs[1] = {};
    static half o[1] = {};
    static float part[1] = {};
    static const int bt[1] = {}, cl[1] = {};
    static const half sinks[1] = {};
    static int stream_tag = 0;
    auto* const stream = reinterpret_cast<cudaStream_t>(&stream_tag);
    const PagedDecodeArgs a{
        .Q = q,
        .K_cache = kc,
        .V_cache = vc,
        .K_scales = ks,
        .V_scales = vs,
        .O = o,
        .partial = part,
        .block_tables = bt,
        .context_lens = cl,
        .batch_size = 1,
        .n_heads = 2,
        .n_kv_heads = 3,
        .head_dim = 4,
        .block_size = 5,
        .scale = 6.0f,
        .kv_scale = 7.0f,
        .max_num_blocks = 8,
        .num_splits = 9,
        .sliding_window = 10,
        .softcap = 11.0f,
        .attn_sinks = sinks,
        .heads_per_cta = 12,
        .stream = stream,
    };
    const auto& [Q, K, V, Ks, Vs, O, P, BT, CL, b, nh, nkv, hd, bs, sc, kvs, mnb, ns, sw, cap, sk, hpc,
                 st] = a;
    EXPECT_EQ(Q, q);
    EXPECT_EQ(K, kc);
    EXPECT_EQ(V, vc);
    EXPECT_EQ(Ks, ks);
    EXPECT_EQ(Vs, vs);
    EXPECT_EQ(O, o);
    EXPECT_EQ(P, part);
    EXPECT_EQ(BT, bt);
    EXPECT_EQ(CL, cl);
    EXPECT_EQ((std::array{b, nh, nkv, hd, bs, mnb, ns, sw, hpc}), (std::array{1, 2, 3, 4, 5, 8, 9, 10, 12}));
    EXPECT_EQ((std::array{sc, kvs, cap}), (std::array{6.0f, 7.0f, 11.0f}));
    EXPECT_EQ(sk, sinks);
    EXPECT_EQ(st, stream);
}

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
