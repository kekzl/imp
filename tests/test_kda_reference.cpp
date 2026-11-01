// The host KDA reference (tests/kda_reference.h) against fla's torch naive_recurrent_kda /
// naive_chunk_kda (tests/refs/kda_golden.h, generator tests/refs/gen_kda_golden.py).
// Tolerance 2e-5 abs: the golden is FP32 torch, the reference FP64; |o| <= ~3.

#include "kda_reference.h"
#include "refs/kda_golden.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>

namespace imp {
namespace {

TEST(KdaReference, MatchesFlaNaiveRecurrent) {
    namespace g = kda_golden;
    std::vector<float> s0;
    const auto in = kda_ref::make_inputs(g::kSeed, g::kT, g::kH, g::kK, g::kV, g::kLowerBound, &s0);
    std::vector<double> state(s0.begin(), s0.end());
    std::vector<double> out;
    kda_ref::forward(in, state, out);

    ASSERT_EQ(out.size(), std::size(g::kOut));
    ASSERT_EQ(state.size(), std::size(g::kState));
    double max_o = 0.0, max_s = 0.0, mag = 0.0;
    for (size_t i = 0; i < out.size(); ++i) {
        max_o = std::max(max_o, std::abs(out[i] - g::kOut[i]));
        mag = std::max(mag, std::abs(out[i]));
    }
    for (size_t i = 0; i < state.size(); ++i)
        max_s = std::max(max_s, std::abs(state[i] - g::kState[i]));
    EXPECT_LT(max_o, 2e-5) << "output diverges from fla";
    EXPECT_LT(max_s, 2e-5) << "final state diverges from fla";
    EXPECT_GT(mag, 0.05) << "degenerate golden: outputs near zero";
}

// The gate is the KDA-specific part: a wrong sign or a missing exp(A_log) must fail the test.
TEST(KdaReference, GateBoundedByLowerBound) {
    namespace g = kda_golden;
    auto in = kda_ref::make_inputs(g::kSeed, g::kT, g::kH, g::kK, g::kV, g::kLowerBound, nullptr);
    in.A_log.assign(in.A_log.size(), 0.0f);
    std::vector<double> s0(static_cast<size_t>(g::kH) * g::kK * g::kV, 0.0);
    std::vector<double> out;
    kda_ref::forward(in, s0, out);
    std::vector<float> s1;
    const auto ref = kda_ref::make_inputs(g::kSeed, g::kT, g::kH, g::kK, g::kV, g::kLowerBound, &s1);
    std::vector<double> st(s1.begin(), s1.end()), out_ref;
    kda_ref::forward(ref, st, out_ref);
    double diff = 0.0;
    for (size_t i = 0; i < out.size(); ++i)
        diff = std::max(diff, std::abs(out[i] - out_ref[i]));
    EXPECT_GT(diff, 1e-3) << "A_log has no effect on the reference";
}

}  // namespace
}  // namespace imp
