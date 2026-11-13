// Batched MTP head pass graph cache (runtime/mtp_head_graph.h): when a pass replays, captures or runs eager,
// and the scan smem rows a captured pass sizes for. No device needed.
#include <gtest/gtest.h>

#include "runtime/mtp_head_graph.h"

namespace imp {
namespace {

TEST(MtpHeadGraph, FirstUseEagerSecondCapturesThenReplays) {
    EXPECT_EQ(mtp_head_graph_step(false, 0, 0, false), MtpHeadGraphStep::Eager);
    EXPECT_EQ(mtp_head_graph_step(false, 1, 0, false), MtpHeadGraphStep::Capture);
    EXPECT_EQ(mtp_head_graph_step(true, 1, 1, false), MtpHeadGraphStep::Replay);
}

TEST(MtpHeadGraph, FullOrDoomedCacheRunsEager) {
    EXPECT_EQ(mtp_head_graph_step(false, 5, kMtpHeadGraphMax, false), MtpHeadGraphStep::Eager);
    EXPECT_EQ(mtp_head_graph_step(false, 5, kMtpHeadGraphMax - 1, false), MtpHeadGraphStep::Capture);
    // A cached key keeps replaying when the cache is full, never once it is doomed.
    EXPECT_EQ(mtp_head_graph_step(true, 1, kMtpHeadGraphMax, false), MtpHeadGraphStep::Replay);
    EXPECT_EQ(mtp_head_graph_step(true, 1, 1, true), MtpHeadGraphStep::Eager);
}

// Every row's scores fit: rows > max_pos, a multiple of the step, never past the KV capacity.
TEST(MtpHeadGraph, ScanSmemRowsCoverTheLargestPosition) {
    EXPECT_EQ(mtp_scan_smem_rows(0, 0), kMtpScanSmemRowStep);
    EXPECT_EQ(mtp_scan_smem_rows(kMtpScanSmemRowStep - 1, 0), kMtpScanSmemRowStep);
    EXPECT_EQ(mtp_scan_smem_rows(kMtpScanSmemRowStep, 0), 2 * kMtpScanSmemRowStep);
    for (int p = 0; p < 5000; p += 37) {
        const int r = mtp_scan_smem_rows(p, 16384);
        EXPECT_GT(r, p);
        EXPECT_EQ(r % kMtpScanSmemRowStep, 0);
        EXPECT_LE(r - p, kMtpScanSmemRowStep);
    }
    EXPECT_EQ(mtp_scan_smem_rows(999, 1000), 1000);
}

// Head KV ring: the window is the last `ring` positions, minus rows a stale draft (hi - 1 > p) overwrote, and
// never empty; ring 0 = the linear cache [0, p]. A window always fits the ring, so the scan smem does too.
TEST(MtpHeadGraph, RingWindowDropsOverwrittenRows) {
    EXPECT_EQ(mtp_ring_lo(5000, 5001, 0), 0);
    EXPECT_EQ(mtp_ring_lo(100, 101, 16384), 0);
    EXPECT_EQ(mtp_ring_lo(16383, 16384, 16384), 0);
    EXPECT_EQ(mtp_ring_lo(16384, 16385, 16384), 1);
    EXPECT_EQ(mtp_ring_lo(80000, 80001, 16384), 80000 - 16383);
    EXPECT_EQ(mtp_ring_lo(80000, 80004, 16384), 80004 - 16384);  // 3 stale drafts: 3 rows fewer
    EXPECT_EQ(mtp_ring_lo(80000, 200000, 16384), 80000);         // every row overwritten: the query alone
    for (int p = 0; p < 70000; p += 997) {
        const int lo = mtp_ring_lo(p, p + 1 + (p % 20), 16384);
        EXPECT_LE(lo, p);
        EXPECT_LE(p + 1 - lo, 16384);
        EXPECT_GE(mtp_scan_smem_rows(p + (p % 20), 16384), p + 1 - lo);
        EXPECT_LT(mtp_ring_row(lo % 16384 + (p - lo), 16384), 16384);
    }
    EXPECT_EQ(mtp_ring_row(16384 + 5, 16384), 5);
    EXPECT_EQ(mtp_ring_row(20000, 0), 20000);
}

// A resume keeps the slot only while some row at or below it survived every later write.
TEST(MtpHeadGraph, RingResumeLostOnlyBehindTheRing) {
    EXPECT_FALSE(mtp_ring_resume_lost(mtp_slot_hi_bound(9000, 2), 100, 16384));  // never wrapped
    EXPECT_FALSE(mtp_ring_resume_lost(mtp_slot_hi_bound(40000, 2), 39000, 16384));
    EXPECT_TRUE(mtp_ring_resume_lost(mtp_slot_hi_bound(40000, 2), 20000, 16384));
    EXPECT_FALSE(mtp_ring_resume_lost(mtp_slot_hi_bound(40000, 2), 20000, 0));  // linear cache
    EXPECT_EQ(mtp_slot_hi_bound(100, 2), 100 + kMtpMaxChainK + 2);
    EXPECT_EQ(mtp_slot_hi_bound(100, 40), 142);
}

// Gap bind: a slot started at `floor` attends from there on, until the ring moves past it.
TEST(MtpHeadGraph, RingFloorHidesEveryRowBelowIt) {
    const int ring = 16384, floor = 52392;
    const int hi = mtp_ring_floor_hi(floor, ring);
    EXPECT_EQ(mtp_ring_lo(floor, hi, ring), floor);
    EXPECT_EQ(mtp_ring_lo(floor + 700, hi, ring), floor);
    EXPECT_EQ(mtp_ring_lo(hi - 1, hi, ring), floor);
    EXPECT_EQ(mtp_ring_lo(hi, hi + 1, ring), floor + 1) << "past the floor the ring takes over";
    EXPECT_TRUE(mtp_ring_resume_lost(hi, floor - 1, ring)) << "a resume below the floor finds no row";
    EXPECT_FALSE(mtp_ring_resume_lost(hi, floor, ring));
}

TEST(MtpHeadGraph, BindHasAGapOnlyPastTheSlotsRows) {
    EXPECT_FALSE(mtp_bind_has_gap(0, 0, false)) << "cold prompt: pair 0 is forwarded";
    EXPECT_FALSE(mtp_bind_has_gap(0, 900, true)) << "ring lost on a cold prompt: plain restart";
    EXPECT_FALSE(mtp_bind_has_gap(864, 899, false));
    EXPECT_TRUE(mtp_bind_has_gap(900, 899, false));
    EXPECT_TRUE(mtp_bind_has_gap(4224, 4300, true)) << "first seen on the shared system prompt";
}

}  // namespace
}  // namespace imp
