// M-RoPE: three position axes instead of one.
//
// This touches the rotary path of EVERY model, so the first thing it has to
// prove is that it changes nothing. Two invariants carry that:
//
//   1. With M-RoPE off (null pointer) the output must be bit-identical to the
//      pre-change kernel — which is testable as "identical to the same call
//      with the parameter defaulted", since that is the only path text-only
//      models take.
//   2. With M-RoPE on but all three axes carrying the SAME position — what a
//      text-only prompt on a VL model looks like — the output must be
//      bit-identical to the single-axis path. Not merely close: the angles are
//      the same number, so every bit must match.
//
// Only then does the three-axis behaviour itself get checked, against a CPU
// reference of the interleaved layout read off `Qwen3VLTextRotaryEmbedding.
// apply_interleaved_mrope`.

#include "compute/rope.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <vector>

namespace imp {
namespace {

constexpr int kHeads = 4;
constexpr int kKvHeads = 2;
constexpr int kHeadDim = 128;
constexpr int kPairs = kHeadDim / 2;  // 64
constexpr int kTokens = 5;
constexpr float kTheta = 5000000.0f;
// Qwen3-VL's split: 24 + 20 + 20 = 64 pairs.
constexpr int kSecT = 24, kSecH = 20, kSecW = 20;

bool gpu_available() {
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

struct DeviceRun {
    std::vector<float> q, k;
};

std::vector<float> ramp(size_t n, float start) {
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i)
        v[i] = start + 0.013f * static_cast<float>(i % 97) - 0.6f;
    return v;
}

// Runs rope_forward on FP32 Q/K and brings the result back.
DeviceRun run_rope(const std::vector<float>& q_in, const std::vector<float>& k_in,
                   const std::vector<int>& positions, const std::vector<int>* mrope_positions,
                   bool interleaved) {
    float *d_q = nullptr, *d_k = nullptr;
    int *d_pos = nullptr, *d_mrope = nullptr;
    EXPECT_EQ(cudaMalloc(&d_q, q_in.size() * sizeof(float)), cudaSuccess);
    EXPECT_EQ(cudaMalloc(&d_k, k_in.size() * sizeof(float)), cudaSuccess);
    EXPECT_EQ(cudaMalloc(&d_pos, positions.size() * sizeof(int)), cudaSuccess);
    cudaMemcpy(d_q, q_in.data(), q_in.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, k_in.data(), k_in.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pos, positions.data(), positions.size() * sizeof(int), cudaMemcpyHostToDevice);

    MRopeParams mrope;
    if (mrope_positions) {
        EXPECT_EQ(cudaMalloc(&d_mrope, mrope_positions->size() * sizeof(int)), cudaSuccess);
        cudaMemcpy(d_mrope, mrope_positions->data(), mrope_positions->size() * sizeof(int),
                   cudaMemcpyHostToDevice);
        mrope.positions = d_mrope;
        mrope.stride = kTokens;
        mrope.sec_t = kSecT;
        mrope.sec_h = kSecH;
        mrope.sec_w = kSecW;
        mrope.interleaved = interleaved;
    }

    const int64_t qshape[4] = {1, kTokens, kHeads, kHeadDim};
    const int64_t kshape[4] = {1, kTokens, kKvHeads, kHeadDim};
    Tensor Q(d_q, QType::F32, 4, qshape, true);
    Tensor K(d_k, QType::F32, 4, kshape, true);
    rope_forward(Q, K, d_pos, kHeadDim, kTheta, 1.0f, /*rope_dim=*/0, /*neox=*/true, 0.0f, 1.0f, nullptr,
                 nullptr, nullptr, mrope);
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    DeviceRun out;
    out.q.resize(q_in.size());
    out.k.resize(k_in.size());
    cudaMemcpy(out.q.data(), d_q, q_in.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(out.k.data(), d_k, k_in.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_pos);
    if (d_mrope)
        cudaFree(d_mrope);
    return out;
}

void expect_bit_identical(const std::vector<float>& a, const std::vector<float>& b, const char* what) {
    ASSERT_EQ(a.size(), b.size());
    size_t differing = 0;
    for (size_t i = 0; i < a.size(); ++i)
        if (std::memcmp(&a[i], &b[i], sizeof(float)) != 0)
            ++differing;
    EXPECT_EQ(differing, 0u) << what << ": " << differing << " of " << a.size() << " floats differ";
}

// The invariant every existing model depends on.
TEST(MRope, TextOnlyPositionsAreBitIdenticalToSingleAxis) {
    if (!gpu_available())
        GTEST_SKIP() << "no CUDA device";

    const auto q = ramp(static_cast<size_t>(kTokens) * kHeads * kHeadDim, 0.4f);
    const auto k = ramp(static_cast<size_t>(kTokens) * kKvHeads * kHeadDim, -0.3f);
    std::vector<int> pos(kTokens);
    for (int i = 0; i < kTokens; ++i)
        pos[i] = 7 + i * 3;  // arbitrary, non-contiguous

    const DeviceRun single = run_rope(q, k, pos, nullptr, false);

    // All three axes carry the text position — a text-only prompt on a VL model.
    std::vector<int> three(static_cast<size_t>(3) * kTokens);
    for (int a = 0; a < 3; ++a)
        for (int i = 0; i < kTokens; ++i)
            three[static_cast<size_t>(a) * kTokens + i] = pos[i];

    for (bool interleaved : {true, false}) {
        const DeviceRun same = run_rope(q, k, pos, &three, interleaved);
        expect_bit_identical(single.q, same.q, interleaved ? "Q (interleaved)" : "Q (sectioned)");
        expect_bit_identical(single.k, same.k, interleaved ? "K (interleaved)" : "K (sectioned)");
    }
}

// CPU reference of the interleaved layout: pair j follows axis j%3 inside that
// axis's interleave region, and the text axis everywhere else.
int expected_axis_interleaved(int pair) {
    const int r = pair % 3;
    if (r == 1 && pair < 3 * kSecH)
        return 1;
    if (r == 2 && pair < 3 * kSecW)
        return 2;
    return 0;
}

int expected_axis_sectioned(int pair) {
    if (pair < kSecT)
        return 0;
    if (pair < kSecT + kSecH)
        return 1;
    return 2;
}

void check_against_reference(bool interleaved) {
    const auto q = ramp(static_cast<size_t>(kTokens) * kHeads * kHeadDim, 0.4f);
    const auto k = ramp(static_cast<size_t>(kTokens) * kKvHeads * kHeadDim, -0.3f);
    std::vector<int> pos(kTokens);
    std::vector<int> three(static_cast<size_t>(3) * kTokens);
    for (int i = 0; i < kTokens; ++i) {
        pos[i] = 100 + i;                      // T
        three[i] = pos[i];                     //
        three[kTokens + i] = 40 + 2 * i;       // H
        three[2 * kTokens + i] = 900 - 5 * i;  // W
    }

    const DeviceRun got = run_rope(q, k, pos, &three, interleaved);

    for (int t = 0; t < kTokens; ++t) {
        for (int h = 0; h < kHeads; ++h) {
            for (int p = 0; p < kPairs; ++p) {
                const int axis = interleaved ? expected_axis_interleaved(p) : expected_axis_sectioned(p);
                const double position = three[static_cast<size_t>(axis) * kTokens + t];
                const double freq = 1.0 / std::pow(static_cast<double>(kTheta),
                                                   (2.0 * p) / static_cast<double>(kHeadDim));
                const double angle = position * freq;
                const size_t base = (static_cast<size_t>(t) * kHeads + h) *
                                    kHeadDim;  // NeoX pairs (p, p + kPairs)
                const double q0 = q[base + p], q1 = q[base + p + kPairs];
                EXPECT_NEAR(got.q[base + p], q0 * std::cos(angle) - q1 * std::sin(angle), 2e-3)
                    << "token " << t << " head " << h << " pair " << p << " axis " << axis;
                EXPECT_NEAR(got.q[base + p + kPairs], q0 * std::sin(angle) + q1 * std::cos(angle), 2e-3)
                    << "token " << t << " head " << h << " pair " << p << " axis " << axis;
            }
        }
    }
}

TEST(MRope, InterleavedLayoutMatchesTheReference) {
    if (!gpu_available())
        GTEST_SKIP() << "no CUDA device";
    check_against_reference(true);
}

TEST(MRope, SectionedLayoutMatchesTheReference) {
    if (!gpu_available())
        GTEST_SKIP() << "no CUDA device";
    check_against_reference(false);
}

// The two layouts must actually differ, or "matches the reference" would be
// satisfied by either implementation.
TEST(MRope, TheTwoLayoutsDisagreeOnDistinctAxes) {
    if (!gpu_available())
        GTEST_SKIP() << "no CUDA device";

    const auto q = ramp(static_cast<size_t>(kTokens) * kHeads * kHeadDim, 0.4f);
    const auto k = ramp(static_cast<size_t>(kTokens) * kKvHeads * kHeadDim, -0.3f);
    std::vector<int> pos(kTokens);
    std::vector<int> three(static_cast<size_t>(3) * kTokens);
    for (int i = 0; i < kTokens; ++i) {
        pos[i] = 100 + i;
        three[i] = pos[i];
        three[kTokens + i] = 40 + 2 * i;
        three[2 * kTokens + i] = 900 - 5 * i;
    }
    const DeviceRun a = run_rope(q, k, pos, &three, true);
    const DeviceRun b = run_rope(q, k, pos, &three, false);
    float worst = 0.0f;
    for (size_t i = 0; i < a.q.size(); ++i)
        worst = std::max(worst, std::fabs(a.q[i] - b.q[i]));
    EXPECT_GT(worst, 1e-2f) << "interleaved and sectioned M-RoPE produced the same rotation";
}

// The H and W axes only reach their interleave regions; everything else must
// still follow the text axis. A layout that leaked H into the tail would show
// up here and nowhere else.
TEST(MRope, TailPairsBeyondTheInterleaveRegionFollowTheTextAxis) {
    if (!gpu_available())
        GTEST_SKIP() << "no CUDA device";

    const auto q = ramp(static_cast<size_t>(kTokens) * kHeads * kHeadDim, 0.4f);
    const auto k = ramp(static_cast<size_t>(kTokens) * kKvHeads * kHeadDim, -0.3f);
    std::vector<int> pos(kTokens, 11);
    std::vector<int> only_text(static_cast<size_t>(3) * kTokens);
    std::vector<int> wild_hw(static_cast<size_t>(3) * kTokens);
    for (int i = 0; i < kTokens; ++i) {
        only_text[i] = wild_hw[i] = 11;
        only_text[kTokens + i] = only_text[2 * kTokens + i] = 11;
        wild_hw[kTokens + i] = 777;  // H and W wildly different
        wild_hw[2 * kTokens + i] = 4242;
    }
    const DeviceRun base = run_rope(q, k, pos, &only_text, true);
    const DeviceRun wild = run_rope(q, k, pos, &wild_hw, true);

    const int interleave_end = 3 * (kSecH > kSecW ? kSecH : kSecW);  // 60
    for (int p = interleave_end; p < kPairs; ++p) {
        for (int t = 0; t < kTokens; ++t) {
            const size_t base_idx = (static_cast<size_t>(t) * kHeads) * kHeadDim + p;
            EXPECT_FLOAT_EQ(base.q[base_idx], wild.q[base_idx]) << "tail pair " << p << " moved with H/W";
        }
    }
    // ...and inside the region it MUST move, or the previous loop proves nothing.
    float moved = 0.0f;
    for (int p = 0; p < interleave_end; ++p)
        moved = std::max(moved, std::fabs(base.q[p] - wild.q[p]));
    EXPECT_GT(moved, 1e-2f);
}

}  // namespace
}  // namespace imp
