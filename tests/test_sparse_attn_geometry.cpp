// Sparse decode attention token -> block geometry (#1819).
//
// The defect this pins: every conversion used the compile-time kKVBlockSize
// (16) while a model with n_kv_heads <= 4 runs a 32-token block. On such a
// model `attention.sparse_topk_tokens=4096` bought 8192 tokens of budget and
// `sparse_min_ctx=12288` engaged at 24576. The knob is documented in TOKENS,
// so the invariant is: budget_blocks * block_size == the configured tokens,
// at every block size.

#include <gtest/gtest.h>
#include "exec/sparse_attn_geometry.h"

namespace imp {
namespace {

constexpr int kDefaults_Sink = 16;
constexpr int kDefaults_Recent = 256;
constexpr int kDefaults_MinCtx = 12288;
constexpr int kMaxCtx = 131072;

TEST(SparseGeometry, BudgetIsTheConfiguredTokenCountAtEveryBlockSize) {
    for (int bs : {16, 32}) {
        const auto g = sparse_geometry(4096, kDefaults_Sink, kDefaults_Recent, kDefaults_MinCtx, kMaxCtx, bs);
        EXPECT_EQ(g.budget_blocks * bs, 4096) << "block_size " << bs;
        EXPECT_FALSE(g.budget_raised) << "block_size " << bs;
    }
}

TEST(SparseGeometry, EngageThresholdIsSparseMinCtxAtEveryBlockSize) {
    for (int bs : {16, 32}) {
        const auto g = sparse_geometry(4096, kDefaults_Sink, kDefaults_Recent, kDefaults_MinCtx, kMaxCtx, bs);
        EXPECT_EQ(g.engage_blocks * bs, kDefaults_MinCtx) << "block_size " << bs;
    }
}

TEST(SparseGeometry, SinkAndRecentCoverTheirConfiguredWindows) {
    for (int bs : {16, 32}) {
        const auto g = sparse_geometry(4096, kDefaults_Sink, kDefaults_Recent, kDefaults_MinCtx, kMaxCtx, bs);
        // Rounded UP to whole blocks: the window must be covered, never clipped.
        EXPECT_GE(g.sink_blocks * bs, kDefaults_Sink) << "block_size " << bs;
        EXPECT_LT((g.sink_blocks - 1) * bs, kDefaults_Sink) << "block_size " << bs;
        EXPECT_GE(g.recent_blocks * bs, kDefaults_Recent) << "block_size " << bs;
    }
}

// The regression itself, stated as the comparison that used to fail: a 32-token
// block must not double what a 16-token block delivers for the same request.
TEST(SparseGeometry, BlockSize32DoesNotDoubleTheBudgetOf16) {
    const auto g16 = sparse_geometry(4096, kDefaults_Sink, kDefaults_Recent, kDefaults_MinCtx, kMaxCtx, 16);
    const auto g32 = sparse_geometry(4096, kDefaults_Sink, kDefaults_Recent, kDefaults_MinCtx, kMaxCtx, 32);
    EXPECT_EQ(g16.budget_blocks * 16, g32.budget_blocks * 32);
    EXPECT_EQ(g16.engage_blocks * 16, g32.engage_blocks * 32);
    EXPECT_EQ(g32.budget_blocks, g16.budget_blocks / 2);
}

TEST(SparseGeometry, BudgetBelowSinkPlusRecentIsRaisedAndFlagged) {
    // 128 tokens of budget cannot hold a 16-token sink plus a 256-token recent
    // window at any block size.
    for (int bs : {16, 32}) {
        const auto g = sparse_geometry(128, kDefaults_Sink, kDefaults_Recent, kDefaults_MinCtx, kMaxCtx, bs);
        EXPECT_TRUE(g.budget_raised) << "block_size " << bs;
        EXPECT_GT(g.budget_blocks, g.sink_blocks + g.recent_blocks) << "block_size " << bs;
    }
}

TEST(SparseGeometry, RecentWindowNeverCollapsesToZeroBlocks) {
    // A zero recent window still has to keep the partial tail block, otherwise
    // the current token's own block can be selected away.
    const auto g = sparse_geometry(4096, 0, 0, kDefaults_MinCtx, kMaxCtx, 32);
    EXPECT_GE(g.recent_blocks, 1);
}

TEST(SparseGeometry, MaxCtxBlocksCarriesTheSpecVerifySlack) {
    for (int bs : {16, 32}) {
        const auto g = sparse_geometry(4096, kDefaults_Sink, kDefaults_Recent, kDefaults_MinCtx, kMaxCtx, bs);
        EXPECT_EQ(g.max_ctx_blocks, (kMaxCtx + bs - 1) / bs + 16) << "block_size " << bs;
    }
}

}  // namespace
}  // namespace imp
