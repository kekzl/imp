// Resolved-dispatch recording (#1205) answers "which kernels did this model actually run".
// Two properties checked: (1) every dispatch_paths.h enumerator has a *_name(), so a tier
// added without a matching switch arm would silently print "?" in production; (2)
// has_prefill()/has_decode() only go true once a branch actually recorded, so a false
// positive would emit an "unset" summary on the first step. CPU-only: plain thread_local POD.

#include "compute/dispatch_paths.h"
#include "compute/dispatch_record.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

using namespace imp;

namespace {

// Keep in lock-step with compute/dispatch_paths.h. A new enumerator that is not
// listed here is not covered — but a new enumerator with no switch arm IS
// caught, which is the failure that reaches production.
const AttnPrefillPath kAllPrefillTiers[] = {
    AttnPrefillPath::FA2,
    AttnPrefillPath::FMHA_SM120,
    AttnPrefillPath::BLACKWELL,
    AttnPrefillPath::NONE,
};
const AttnPrefillOuter kAllPrefillOuter[] = {
    AttnPrefillOuter::UNSET,         AttnPrefillOuter::FA2_FP16QK, AttnPrefillOuter::CUBLAS,
    AttnPrefillOuter::CUBLAS_SLICED, AttnPrefillOuter::FMHA_CHAIN, AttnPrefillOuter::RA2,
    AttnPrefillOuter::APA,
};
const AttnDecodePath kAllDecode[] = {
    AttnDecodePath::UNSET, AttnDecodePath::FP16,  AttnDecodePath::FP8,      AttnDecodePath::INT8,
    AttnDecodePath::INT4,  AttnDecodePath::NVFP4, AttnDecodePath::MXFP4_KV,
};
const MoePrefillOuter kAllMoeOuter[] = {
    MoePrefillOuter::UNSET,     MoePrefillOuter::NONE,          MoePrefillOuter::FP16_BATCH,
    MoePrefillOuter::FP8_BATCH, MoePrefillOuter::GROUPED_NVFP4, MoePrefillOuter::NVFP4_DEQUANT,
    MoePrefillOuter::LEGACY,    MoePrefillOuter::FUSED_Q6K,     MoePrefillOuter::FUSED_Q4K_DP4A,
};
const MoePrefillPath kAllMoeTiers[] = {
    MoePrefillPath::DEVICE_ARGS,
    MoePrefillPath::GROUPED,
    MoePrefillPath::LEGACY,
};

void expect_named(const char* name) {
    ASSERT_NE(name, nullptr);
    EXPECT_STRNE(name, "?") << "enumerator has no switch arm in dispatch_paths.cpp — the "
                               "resolved-dispatch summary would print '?' for it";
    EXPECT_GT(std::string(name).size(), 0u);
}

}  // namespace

TEST(DispatchPaths, EveryEnumeratorHasAName) {
    for (auto p : kAllPrefillTiers)
        expect_named(attn_prefill_path_name(p));
    for (auto p : kAllPrefillOuter)
        expect_named(attn_prefill_outer_name(p));
    for (auto p : kAllDecode)
        expect_named(attn_decode_path_name(p));
    for (auto p : kAllMoeOuter)
        expect_named(moe_prefill_outer_name(p));
    for (auto p : kAllMoeTiers)
        expect_named(moe_prefill_path_name(p));
}

// Names are what an operator greps for, so they must be distinguishable.
TEST(DispatchPaths, NamesAreDistinctWithinEachFamily) {
    auto distinct = [](const auto& range, auto fn) {
        std::vector<std::string> seen;
        for (auto p : range) {
            std::string n = fn(p);
            for (const auto& s : seen)
                EXPECT_NE(s, n) << "duplicate path name: " << n;
            seen.push_back(n);
        }
    };
    distinct(kAllPrefillTiers, attn_prefill_path_name);
    distinct(kAllPrefillOuter, attn_prefill_outer_name);
    distinct(kAllDecode, attn_decode_path_name);
    distinct(kAllMoeOuter, moe_prefill_outer_name);
    distinct(kAllMoeTiers, moe_prefill_path_name);
}

TEST(DispatchRecord, StartsUnsetAndGatesTheDump) {
    dispatch_record::reset();
    const auto& r = dispatch_record::current();

    EXPECT_FALSE(r.has_prefill());
    EXPECT_FALSE(r.has_decode());
    EXPECT_FALSE(r.attn_prefill_tier_set);
    EXPECT_FALSE(r.moe_prefill_tier_set);

    // A prefill alone must not unlock the dump — the summary reports both.
    dispatch_record::set_attn_prefill_outer(AttnPrefillOuter::FA2_FP16QK);
    EXPECT_TRUE(r.has_prefill());
    EXPECT_FALSE(r.has_decode());

    dispatch_record::set_attn_decode(AttnDecodePath::FP8);
    EXPECT_TRUE(r.has_decode());

    dispatch_record::reset();
    EXPECT_FALSE(dispatch_record::current().has_prefill());
    EXPECT_FALSE(dispatch_record::current().has_decode());
}

// Dense GEMM ladder (D-2): every DensePath exit is named and distinct, or the
// "Resolved dense GEMM" line prints '?' or two exits under one name.
TEST(DispatchPaths, EveryDensePathHasADistinctName) {
    std::vector<std::string> seen;
    for (int p = 0; p < static_cast<int>(DensePath::COUNT); ++p) {
        const char* name = dense_path_name(static_cast<DensePath>(p));
        expect_named(name);
        for (const auto& s : seen)
            EXPECT_NE(s, name) << "duplicate dense path name: " << name;
        seen.emplace_back(name);
    }
    EXPECT_EQ(seen.size(), 53u) << "UNSET + 28 ladder exits + 24 fused / sibling exits: DensePath gained or "
                                   "lost an exit, update its recording site and this count";
}

// The first line is logged after one prefill step; decode routes (and the graph capture that records
// them) arrive later. Each step that adds an exit bit logs the summary once more, a step without one
// logs nothing.
TEST(DispatchRecord, DenseLogLineRepeatsOnlyWhenAnExitIsNew) {
    dispatch_record::reset();
    auto& r = dispatch_record::current();
    EXPECT_EQ(dispatch_record::dense_log_line(r, /*first=*/true), "Resolved dense GEMM: none recorded");

    // Prefill step: Q4_K source GEMMs without a tier entry.
    dispatch_record::set_dense(TensorKind::WQ, DensePath::UNCACHED_FALLBACK);
    dispatch_record::set_dense(TensorKind::W_DOWN, DensePath::UNCACHED_FALLBACK);
    EXPECT_EQ(dispatch_record::dense_log_line(r, /*first=*/true),
              "Resolved dense GEMM: WQ=uncached_fallback W_DOWN=uncached_fallback");
    EXPECT_EQ(dispatch_record::dense_log_line(r, /*first=*/false), "") << "no new exit, no line";

    // First decode step: the fused q|k|v GEMV and the dp4a down with residual.
    dispatch_record::set_dense(TensorKind::WQ, DensePath::QKV_DP4A);
    dispatch_record::set_dense(TensorKind::W_DOWN, DensePath::DP4A_RESIDUAL);
    EXPECT_EQ(
        dispatch_record::dense_log_line(r, /*first=*/false),
        "Resolved dense GEMM (+2 new): WQ=uncached_fallback+qkv_dp4a W_DOWN=uncached_fallback+dp4a_residual");

    // Later decode steps replay the same exits: nothing new, nothing logged.
    dispatch_record::set_dense(TensorKind::WQ, DensePath::QKV_DP4A);
    EXPECT_EQ(dispatch_record::dense_log_line(r, /*first=*/false), "");
    dispatch_record::reset();
}

// A sibling launch (q|k|v, gate|up, ssm_in|gdn_gate) is one exit for each weight in it: before the
// fused routes recorded, Qwen3.8 showed no WQ / WK / WV / W_GATE / W_UP and Qwen3-8B-FP8 printed "none".
TEST(DispatchRecord, SiblingLaunchRecordsEveryWeight) {
    dispatch_record::reset();
    const TensorKind qkv[3] = {TensorKind::WQ, TensorKind::WK, TensorKind::WV};
    dispatch_record::set_dense_each(qkv, 3, DensePath::FP8_SMALLM_MULTI);
    EXPECT_EQ(dispatch_record::dense_summary(dispatch_record::current()),
              "WQ=fp8_smallm_multi WK=fp8_smallm_multi WV=fp8_smallm_multi");
    dispatch_record::reset();
}

// set_dense keeps the last exit per kind and a bit per exit ever taken; a kind
// out of range is dropped, not written past the arrays.
TEST(DispatchRecord, DenseKeepsLastAndEverySeenExitPerKind) {
    dispatch_record::reset();
    const auto& r = dispatch_record::current();
    const int wq = static_cast<int>(TensorKind::WQ);
    const int down = static_cast<int>(TensorKind::W_DOWN);
    EXPECT_EQ(r.dense_last[wq], DensePath::UNSET);
    EXPECT_EQ(r.dense_seen[wq], 0u);
    EXPECT_EQ(dispatch_record::dense_summary(r), "");

    dispatch_record::set_dense(TensorKind::WQ, DensePath::NVFP4_TILE);
    dispatch_record::set_dense(TensorKind::WQ, DensePath::GEMV_NVFP4);
    dispatch_record::set_dense(TensorKind::W_DOWN, DensePath::SMALLM_A4);
    dispatch_record::set_dense(TensorKind::COUNT, DensePath::FP16_CACHE);
    EXPECT_EQ(r.dense_last[wq], DensePath::GEMV_NVFP4);
    EXPECT_EQ(r.dense_last[down], DensePath::SMALLM_A4);
    EXPECT_EQ(r.dense_seen[wq], (uint64_t{1} << static_cast<int>(DensePath::NVFP4_TILE)) |
                                    (uint64_t{1} << static_cast<int>(DensePath::GEMV_NVFP4)));
    // Kinds in TensorKind order, exits in DensePath order.
    EXPECT_EQ(dispatch_record::dense_summary(r), "WQ=gemv_nvfp4+nvfp4_tile W_DOWN=smallm_a4");

    dispatch_record::reset();
    EXPECT_EQ(dispatch_record::dense_summary(dispatch_record::current()), "");
}

// The two-level paths (outer → tier) must report the tier only once it was
// really recorded, otherwise the summary would claim a grouped NVFP4 tier for a
// model that never reached the grouped NVFP4 branch.
TEST(DispatchRecord, TierFlagsTrackTheirSetters) {
    dispatch_record::reset();
    const auto& r = dispatch_record::current();

    dispatch_record::set_attn_prefill_outer(AttnPrefillOuter::FMHA_CHAIN);
    EXPECT_FALSE(r.attn_prefill_tier_set) << "outer must not imply a tier";
    dispatch_record::set_attn_prefill_tier(AttnPrefillPath::FA2);
    EXPECT_TRUE(r.attn_prefill_tier_set);
    EXPECT_EQ(r.attn_prefill_tier, AttnPrefillPath::FA2);

    dispatch_record::set_moe_prefill_outer(MoePrefillOuter::GROUPED_NVFP4);
    EXPECT_FALSE(r.moe_prefill_tier_set);
    dispatch_record::set_moe_prefill_tier(MoePrefillPath::DEVICE_ARGS);
    EXPECT_TRUE(r.moe_prefill_tier_set);
    EXPECT_EQ(r.moe_prefill_tier, MoePrefillPath::DEVICE_ARGS);

    dispatch_record::reset();
}
