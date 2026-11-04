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

}  // namespace
}  // namespace imp
