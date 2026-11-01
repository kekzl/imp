// The host KDA reference (tests/kda_reference.h) against fla's torch naive_recurrent_kda /
// naive_chunk_kda and naive_kda_gate (tests/refs/kda_golden.h, generator tests/refs/gen_kda_golden.py).
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

// Kimi-Linear gate g = -exp(A_log) * softplus(f + dt_bias), golden from fla's naive_kda_gate.
TEST(KdaReference, SoftplusGateMatchesFlaNaiveRecurrent) {
    namespace g = kda_golden;
    std::vector<float> s0;
    const auto in = kda_ref::make_inputs(g::kSeed, g::kT, g::kH, g::kK, g::kV, g::kLowerBound, &s0,
                                         /*softplus_gate=*/true);
    std::vector<double> state(s0.begin(), s0.end());
    std::vector<double> out;
    kda_ref::forward(in, state, out);

    ASSERT_EQ(out.size(), std::size(g::kOutSoftplus));
    ASSERT_EQ(state.size(), std::size(g::kStateSoftplus));
    double max_o = 0.0, max_s = 0.0, vs_sigmoid = 0.0;
    for (size_t i = 0; i < out.size(); ++i) {
        max_o = std::max(max_o, std::abs(out[i] - g::kOutSoftplus[i]));
        vs_sigmoid = std::max(vs_sigmoid, std::abs(static_cast<double>(g::kOutSoftplus[i]) - g::kOut[i]));
    }
    for (size_t i = 0; i < state.size(); ++i)
        max_s = std::max(max_s, std::abs(state[i] - g::kStateSoftplus[i]));
    EXPECT_LT(max_o, 2e-5) << "softplus-gate output diverges from fla";
    EXPECT_LT(max_s, 2e-5) << "softplus-gate final state diverges from fla";
    // The golden pair would not tell the gates apart if both decayed the state alike.
    EXPECT_GT(vs_sigmoid, 1e-2) << "softplus and sigmoid goldens coincide";
}

// Closed-form points of the softplus arm: torch threshold 20 switches to the identity, the
// lower bound is ignored, and the decay is unbounded below (the sigmoid arm stops at lower_bound).
TEST(KdaReference, SoftplusGateClosedForm) {
    kda_ref::Inputs in;
    in.softplus_gate = true;
    in.lower_bound = -5.0f;
    const double a = 3.0;
    EXPECT_NEAR(kda_ref::log_decay(in, a, 0.0), -a * std::log(2.0), 1e-12);
    EXPECT_NEAR(kda_ref::log_decay(in, a, 25.0), -a * 25.0, 1e-12);
    EXPECT_NEAR(kda_ref::log_decay(in, a, 19.0), -a * std::log1p(std::exp(19.0)), 1e-12);
    EXPECT_NEAR(kda_ref::log_decay(in, a, -30.0), -a * std::exp(-30.0), 1e-20);
    EXPECT_LT(kda_ref::log_decay(in, a, 10.0), in.lower_bound) << "softplus decay must not be bounded";
    in.softplus_gate = false;
    EXPECT_GT(kda_ref::log_decay(in, a, 10.0), in.lower_bound);
}

}  // namespace
}  // namespace imp
