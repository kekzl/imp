// kv_blocks_from_residual sizes the KV pool from what's left after weight caches are built.
// CPU-lane on purpose: until #1251 this arithmetic lived inline in
// engine_kv_cache_init.cpp, unreachable without a GPU and a 22GB checkpoint, so the one case
// that matters was never asserted.
// The case: the caller subtracts allocator headroom from the measured residual. If whoever
// reserved room for this pool didn't also reserve the headroom, the residual is entirely
// headroom, room is 0, and the pool silently floors - a rescue, not a size; `floored` is what
// tells the two apart.

#include <gtest/gtest.h>

#include "memory/vram_query.h"

namespace imp {
namespace {

constexpr size_t kMiB = 1024ULL * 1024ULL;

// Qwen3.6-35B-A3B-UD-Q4_K_M on a 32 GiB 5090: 10 attention layers, block_size
// 32, 2 KV heads, head_dim 256, FP16 -> 640 KiB per block.
constexpr size_t k35bPerBlock = 640ULL * 1024ULL;

// kv_cache.block_size is refused at resolve time, not rounded at dispatch:
// the FP8 tile kernel would silently fall back on 24, and 8 would put a
// 16-token WMMA tile across two blocks (AUDIT_arch_2026 B-5).
TEST(KvBlockSize, MultiplesOfSixteenUpTo256AreServed) {
    for (int bs : {16, 32, 48, 64, 128, 256})
        EXPECT_EQ(kv_block_size_error(bs), nullptr) << bs;
}

TEST(KvBlockSize, AnythingElseIsRefusedWithAReason) {
    for (int bs : {1, 8, 15, 24, 100, 272, 512})
        EXPECT_NE(kv_block_size_error(bs), nullptr) << bs;
}

TEST(KvResidualSizing, PlanFitsSoNothingIsClamped) {
    const auto s = kv_blocks_from_residual(8192 * kMiB, 1630 * kMiB, k35bPerBlock, 4096, 16);
    EXPECT_EQ(s.blocks, 4096);
    EXPECT_FALSE(s.clamped);
    EXPECT_FALSE(s.floored);
}

TEST(KvResidualSizing, ResidualSmallerThanPlanClampsButDoesNotFloor) {
    // 2416 MiB free - 1630 MiB headroom = 786 MiB -> 1257 blocks. These are the
    // post-fix numbers measured on the #1251 repro.
    const auto s = kv_blocks_from_residual(2416 * kMiB, 1630 * kMiB, k35bPerBlock, 4096, 16);
    EXPECT_EQ(s.blocks, 1257);
    EXPECT_TRUE(s.clamped);
    EXPECT_FALSE(s.floored) << "a pool smaller than the plan is normal — the plan is a "
                               "projection, the residual is the truth";
}

// The #1251 regression itself, with the numbers straight out of the bug report.
TEST(KvResidualSizing, HeadroomExceedingResidualFloorsAndSaysSo) {
    const auto s = kv_blocks_from_residual(1264 * kMiB, 1630 * kMiB, k35bPerBlock, 4096, 16);
    EXPECT_EQ(s.blocks, 16) << "the floor, not a computed size";
    EXPECT_TRUE(s.clamped);
    EXPECT_TRUE(s.floored) << "#1251: nothing was left to size the pool from, and the load "
                              "reported success anyway";
}

TEST(KvResidualSizing, ResidualExactlyHeadroomLeavesNothing) {
    const auto s = kv_blocks_from_residual(1630 * kMiB, 1630 * kMiB, k35bPerBlock, 4096, 16);
    EXPECT_EQ(s.blocks, 16);
    EXPECT_TRUE(s.floored);
}

// One block short of the floor still floors; exactly the floor does not.
TEST(KvResidualSizing, FloorBoundaryIsExact) {
    const size_t headroom = 1000 * kMiB;
    const auto below = kv_blocks_from_residual(headroom + 15 * k35bPerBlock, headroom,
                                               k35bPerBlock, 4096, 16);
    EXPECT_EQ(below.blocks, 16);
    EXPECT_TRUE(below.floored);

    const auto at = kv_blocks_from_residual(headroom + 16 * k35bPerBlock, headroom, k35bPerBlock,
                                            4096, 16);
    EXPECT_EQ(at.blocks, 16);
    EXPECT_TRUE(at.clamped);
    EXPECT_FALSE(at.floored) << "16 blocks that were actually computed is not the rescue path";
}

TEST(KvResidualSizing, MoreRoomThanPlannedNeverGrowsThePool) {
    const auto s = kv_blocks_from_residual(30000 * kMiB, 1630 * kMiB, k35bPerBlock, 64, 16);
    EXPECT_EQ(s.blocks, 64) << "the residual can only shrink the plan, never grow it";
    EXPECT_FALSE(s.clamped);
}

TEST(KvResidualSizing, DegenerateInputsAreInert) {
    const auto zero_block = kv_blocks_from_residual(8192 * kMiB, 1630 * kMiB, 0, 4096, 16);
    EXPECT_EQ(zero_block.blocks, 4096) << "per_block==0 must not divide";
    EXPECT_FALSE(zero_block.clamped);

    const auto no_plan = kv_blocks_from_residual(8192 * kMiB, 1630 * kMiB, k35bPerBlock, 0, 16);
    EXPECT_EQ(no_plan.blocks, 0);
    EXPECT_FALSE(no_plan.floored);
}

TEST(KvResidualSizing, FreeBelowHeadroomDoesNotUnderflow) {
    // free < headroom must saturate at 0 room, not wrap around size_t.
    const auto s = kv_blocks_from_residual(100 * kMiB, 1630 * kMiB, k35bPerBlock, 4096, 16);
    EXPECT_EQ(s.blocks, 16);
    EXPECT_TRUE(s.floored);
}

// Regression: a real-size pool can still be too small for one full-length request while
// load() reports success (#1251 covered only the floor case).

TEST(KvPoolVerdict, PoolHoldingAFullSequenceIsSufficient) {
    // 4096 blocks x 32 tokens = 131072 tokens against a 8192-token request.
    const auto s = kv_blocks_from_residual(8192 * kMiB, 1630 * kMiB, k35bPerBlock, 4096, 16);
    EXPECT_EQ(kv_pool_verdict(s, 8192, 32), KvPoolVerdict::Sufficient);
}

TEST(KvPoolVerdict, ClampedButStillShortOfOneSequenceIsReported) {
    // 2416 MiB residual, 1630 MiB headroom -> 786 MiB -> 1257 blocks, above the 16-block floor.
    // 1257 blocks x 32 tokens < what a 65536-token max_seq_len needs (2048 blocks): never admits one.
    const auto s = kv_blocks_from_residual(2416 * kMiB, 1630 * kMiB, k35bPerBlock, 4096, 16);
    ASSERT_TRUE(s.clamped);
    ASSERT_FALSE(s.floored) << "this case must not be the floor, or it is the other message";
    EXPECT_EQ(kv_pool_verdict(s, 65536, 32), KvPoolVerdict::ShortOfOneSequence);
    // Same pool, a request it can serve: nothing to report.
    EXPECT_EQ(kv_pool_verdict(s, 32768, 32), KvPoolVerdict::Sufficient);
}

TEST(KvPoolVerdict, FlooredKeepsItsOwnMessage) {
    // Floored is also short of one sequence, but it has a message of its own
    // that names the missing residual — reporting both would say it twice.
    const auto s = kv_blocks_from_residual(1264 * kMiB, 1630 * kMiB, k35bPerBlock, 4096, 16);
    ASSERT_TRUE(s.floored);
    EXPECT_EQ(kv_pool_verdict(s, 65536, 32), KvPoolVerdict::Floored);
}

TEST(KvPoolVerdict, ExactlyOneSequenceIsSufficient) {
    // The boundary decides whether a pool sized to exactly the request is
    // called a fault. It is not: one sequence fits.
    KvResidualSizing s;
    s.blocks = 64;
    EXPECT_EQ(kv_pool_verdict(s, 2048, 32), KvPoolVerdict::Sufficient);
    s.blocks = 63;
    EXPECT_EQ(kv_pool_verdict(s, 2048, 32), KvPoolVerdict::ShortOfOneSequence);
}

TEST(KvPoolVerdict, PartialTrailingBlockCounts) {
    // 2049 tokens at block_size 32 needs 65 blocks, not 64.
    EXPECT_EQ(kv_blocks_per_sequence(2049, 32), 65);
    EXPECT_EQ(kv_blocks_per_sequence(2048, 32), 64);
    KvResidualSizing s;
    s.blocks = 64;
    EXPECT_EQ(kv_pool_verdict(s, 2049, 32), KvPoolVerdict::ShortOfOneSequence);
}

TEST(KvPoolVerdict, UnsetRequirementIsNotAFault) {
    // No max_seq_len / no block size means there is nothing to check against;
    // the verdict must not turn that into a warning on every load.
    KvResidualSizing s;
    s.blocks = 1;
    EXPECT_EQ(kv_blocks_per_sequence(0, 32), 0);
    EXPECT_EQ(kv_blocks_per_sequence(2048, 0), 0);
    EXPECT_EQ(kv_pool_verdict(s, 0, 32), KvPoolVerdict::Sufficient);
    EXPECT_EQ(kv_pool_verdict(s, 2048, 0), KvPoolVerdict::Sufficient);
}

// Measured pair (MEMORY.md B8): a 3263 MiB checkpoint costs 3264 MiB of device free VRAM idle,
// 8446 MiB beside a process holding 23.4 GiB.

TEST(UploadExceedsCheckpoint, TheMeasuredPair) {
    EXPECT_FALSE(upload_exceeds_checkpoint(3264 * kMiB, 3263 * kMiB));
    EXPECT_TRUE(upload_exceeds_checkpoint(8446 * kMiB, 3263 * kMiB));
}

TEST(UploadExceedsCheckpoint, ConsumingLessIsOrdinary) {
    // Host-resident experts and dropped sources upload less than the file
    // holds. One-sided on purpose: this direction must never warn, or the
    // gpt-oss and MoE-offload paths warn on every healthy load.
    EXPECT_FALSE(upload_exceeds_checkpoint(1 * kMiB, 10000 * kMiB));
    EXPECT_FALSE(upload_exceeds_checkpoint(0, 10000 * kMiB));
}

TEST(UploadExceedsCheckpoint, TheQuarterIsInclusiveOfItsBoundary) {
    // Exactly a quarter over is still silent — the threshold has to absorb the
    // CUDA context and library growth that the upload phase also pays for.
    EXPECT_FALSE(upload_exceeds_checkpoint(1250 * kMiB, 1000 * kMiB));
    EXPECT_TRUE(upload_exceeds_checkpoint(1251 * kMiB, 1000 * kMiB));
}

TEST(UploadExceedsCheckpoint, UnknownCheckpointSizeIsNotAFault) {
    // 0 = the checkpoint could not be sized (unreadable path, an unrecognised
    // layout). There is no reference to compare against, and inventing one
    // would warn on every load of it.
    EXPECT_FALSE(upload_exceeds_checkpoint(99999 * kMiB, 0));
}

}  // namespace
}  // namespace imp
