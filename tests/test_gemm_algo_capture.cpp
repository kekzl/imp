// #2396: cuBLASLt algo choice when a cold shape's first call is inside a stream capture.

#include <gtest/gtest.h>

#include <array>

#include "compute/gemm_algo_capture.h"

using imp::gemm_algo_probe_allowed;
using imp::gemm_capture_safe_pick;
using imp::GemmAlgoHostCheck;

TEST(GemmAlgoCapture, ProbeOnlyWhenQueryOkAndNotCapturing) {
    EXPECT_TRUE(gemm_algo_probe_allowed(true, false));
    EXPECT_FALSE(gemm_algo_probe_allowed(true, true));
    // A failed query (e.g. legacy stream during another capture) must not probe either.
    EXPECT_FALSE(gemm_algo_probe_allowed(false, false));
    EXPECT_FALSE(gemm_algo_probe_allowed(false, true));
}

TEST(GemmAlgoCapture, PicksFirstSupportedCandidateWithinWorkspace) {
    const std::array<GemmAlgoHostCheck, 4> c{{{false, 0}, {true, 4096}, {true, 0}, {true, 0}}};
    EXPECT_EQ(gemm_capture_safe_pick(c, 8192), 1);
    // Candidate 1 needs more workspace than there is: heuristic order continues to 2.
    EXPECT_EQ(gemm_capture_safe_pick(c, 1024), 2);
}

TEST(GemmAlgoCapture, NoCandidateGivesMinusOne) {
    const std::array<GemmAlgoHostCheck, 2> none{{{false, 0}, {true, 1 << 20}}};
    EXPECT_EQ(gemm_capture_safe_pick(none, 1024), -1);
    EXPECT_EQ(gemm_capture_safe_pick(std::span<const GemmAlgoHostCheck>{}, 1024), -1);
}

static_assert(gemm_capture_safe_pick(std::array<GemmAlgoHostCheck, 1>{{{true, 0}}}, 0) == 0);
