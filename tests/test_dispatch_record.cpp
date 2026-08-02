// Resolved-dispatch recording (#1205).
//
// The recorder in compute/dispatch_record.h is what makes "which kernels did
// this model actually run" answerable. Two properties keep it honest and both
// are checked here:
//
//   1. Every enumerator has a name. The summary is built by concatenating
//      *_name() results, so a tier added to compute/dispatch_paths.h without a
//      matching switch arm would silently print "?" in production — the exact
//      class of silent gap the recorder exists to close.
//   2. has_prefill()/has_decode() only go true once a branch has actually
//      recorded. Engine::log_resolved_dispatch_once_() gates the one-shot dump
//      on them, so a false positive there would emit a summary full of
//      "unset" on the very first step.
//
// CPU-only: the recorder is plain thread_local POD with no CUDA in it.

#include "compute/dispatch_paths.h"
#include "compute/dispatch_record.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

using namespace imp;

namespace {

// Keep in lock-step with compute/dispatch_paths.h. A new enumerator that is not
// listed here is not covered — but a new enumerator with no switch arm IS
// caught, which is the failure that reaches production.
const AttnPrefillPath kAllPrefillTiers[] = {
    AttnPrefillPath::MXFP4,      AttnPrefillPath::FA2,       AttnPrefillPath::FP8,
    AttnPrefillPath::FMHA_SM120, AttnPrefillPath::BLACKWELL, AttnPrefillPath::NONE,
};
const AttnPrefillOuter kAllPrefillOuter[] = {
    AttnPrefillOuter::UNSET,         AttnPrefillOuter::FA2_FP16QK, AttnPrefillOuter::CUBLAS,
    AttnPrefillOuter::CUBLAS_SLICED, AttnPrefillOuter::FMHA_CHAIN,
};
const AttnDecodePath kAllDecode[] = {
    AttnDecodePath::UNSET, AttnDecodePath::FP16,  AttnDecodePath::FP8,      AttnDecodePath::INT8,
    AttnDecodePath::INT4,  AttnDecodePath::NVFP4, AttnDecodePath::NVFP4_TC, AttnDecodePath::MXFP4_KV,
};
const MoePrefillOuter kAllMoeOuter[] = {
    MoePrefillOuter::UNSET,     MoePrefillOuter::NONE,      MoePrefillOuter::FP16_BATCH,
    MoePrefillOuter::FP8_BATCH, MoePrefillOuter::CUTLASS3X, MoePrefillOuter::NVFP4_DEQUANT,
    MoePrefillOuter::LEGACY,    MoePrefillOuter::FUSED_Q6K,
};
const MoePrefillPath kAllMoeTiers[] = {
    MoePrefillPath::DEVICE_ARGS,
    MoePrefillPath::SMALL_M,
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

// The two-level paths (outer → tier) must report the tier only once it was
// really recorded, otherwise the summary would claim a CUTLASS tier for a
// model that never reached the CUTLASS branch.
TEST(DispatchRecord, TierFlagsTrackTheirSetters) {
    dispatch_record::reset();
    const auto& r = dispatch_record::current();

    dispatch_record::set_attn_prefill_outer(AttnPrefillOuter::FMHA_CHAIN);
    EXPECT_FALSE(r.attn_prefill_tier_set) << "outer must not imply a tier";
    dispatch_record::set_attn_prefill_tier(AttnPrefillPath::FA2);
    EXPECT_TRUE(r.attn_prefill_tier_set);
    EXPECT_EQ(r.attn_prefill_tier, AttnPrefillPath::FA2);

    dispatch_record::set_moe_prefill_outer(MoePrefillOuter::CUTLASS3X);
    EXPECT_FALSE(r.moe_prefill_tier_set);
    dispatch_record::set_moe_prefill_tier(MoePrefillPath::DEVICE_ARGS);
    EXPECT_TRUE(r.moe_prefill_tier_set);
    EXPECT_EQ(r.moe_prefill_tier, MoePrefillPath::DEVICE_ARGS);

    dispatch_record::reset();
}
