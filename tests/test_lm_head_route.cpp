// lm_head_route (exec/lm_head_route.h, audit D-5): the one LM-head decision behind serving,
// perplexity, prompt logprobs and the draft argmax head. Each row is the arm the four ladders
// took before they shared it.

#include <gtest/gtest.h>

#include "exec/lm_head_route.h"

namespace {

using imp::DraftHeadRoute;
using imp::LmHeadFacts;
using imp::LmHeadRoute;
using imp::LmHeadUse;
using imp::StorageTier;

LmHeadFacts facts(StorageTier tier, bool fp8 = false, bool nvfp4_cache = false, bool mxfp4_linear = false,
                  bool dp4a = false) {
    LmHeadFacts f;
    f.tier = tier;
    f.fp8_cache = fp8;
    f.nvfp4_cache = nvfp4_cache;
    f.mxfp4_linear = mxfp4_linear;
    f.dp4a_ready = dp4a;
    return f;
}

struct Row {
    const char* name;
    LmHeadFacts f;
    LmHeadRoute row, batch, eval;
};

TEST(LmHeadRoute, DecisionTable) {
    using R = LmHeadRoute;
    using T = StorageTier;
    const Row rows[] = {
        {"NVFP4 tier", facts(T::NVFP4), R::Nvfp4, R::Nvfp4, R::Nvfp4},
        {"NVFP4 secondary cache of a GGUF head", facts(T::FP16, false, true, false, true), R::Nvfp4, R::Nvfp4,
         R::Nvfp4},
        {"FP8 cache beats the NVFP4 cache", facts(T::NVFP4, true, true), R::Fp8Cache, R::Handle, R::Handle},
        {"FP8 cache of a dp4a head", facts(T::Undefined, true, false, false, true), R::Fp8Cache, R::Dp4a,
         R::Handle},
        {"MXFP4 with linear scales", facts(T::MXFP4, false, false, true), R::Mxfp4Gemv, R::Handle, R::Handle},
        {"MXFP4 without linear scales", facts(T::MXFP4), R::Handle, R::Handle, R::Handle},
        {"MXFP4 tier, dp4a qtype", facts(T::MXFP4, false, false, false, true), R::Dp4a, R::Dp4a, R::Handle},
        {"raw GGUF head", facts(T::Undefined, false, false, false, true), R::Dp4a, R::Dp4a, R::Dp4a},
        {"GGUF head with an FP16-tier copy", facts(T::FP16, false, false, false, true), R::Dp4a, R::Handle,
         R::Dp4a},
        {"FP16 head", facts(T::FP16), R::Handle, R::Handle, R::Handle},
        {"FP8 tier, no cache", facts(T::FP8), R::Handle, R::Handle, R::Handle},
        {"no handle", facts(T::Undefined), R::Handle, R::Handle, R::Handle},
    };
    for (const Row& r : rows) {
        EXPECT_EQ(imp::lm_head_route(r.f, LmHeadUse::ServeRow), r.row) << r.name;
        EXPECT_EQ(imp::lm_head_route(r.f, LmHeadUse::ServeBatch), r.batch) << r.name;
        EXPECT_EQ(imp::lm_head_route(r.f, LmHeadUse::Eval), r.eval) << r.name;
    }
}

TEST(LmHeadRoute, NvFp4MeansNotFp8Cached) {
    EXPECT_TRUE(imp::lm_head_is_nvfp4(facts(StorageTier::NVFP4)));
    EXPECT_TRUE(imp::lm_head_is_nvfp4(facts(StorageTier::Undefined, false, true)));
    EXPECT_FALSE(imp::lm_head_is_nvfp4(facts(StorageTier::NVFP4, true)));
    EXPECT_FALSE(imp::lm_head_is_nvfp4(facts(StorageTier::CUTLASS_NVFP4)));
}

TEST(LmHeadRoute, DraftHead) {
    using D = DraftHeadRoute;
    // draft_copy, fp8_head, tile_ready, tile_scratch
    EXPECT_EQ(imp::draft_head_route(true, true, true, true), D::W4a16Draft);
    EXPECT_EQ(imp::draft_head_route(true, false, false, false), D::W4a16Draft);
    EXPECT_EQ(imp::draft_head_route(false, true, false, false), D::Fp8Head);
    EXPECT_EQ(imp::draft_head_route(false, true, true, true), D::TileGemm);  // built tile head wins
    EXPECT_EQ(imp::draft_head_route(false, true, true, false), D::None);
    EXPECT_EQ(imp::draft_head_route(false, false, true, true), D::TileGemm);
    EXPECT_EQ(imp::draft_head_route(false, false, true, false), D::None);
    EXPECT_EQ(imp::draft_head_route(false, false, false, true), D::None);
}

static_assert(imp::lm_head_route(LmHeadFacts{}, LmHeadUse::Eval) == LmHeadRoute::Handle);

}  // namespace
