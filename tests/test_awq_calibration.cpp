// AWQ calibration: the file format, and the invariance the whole transform
// rests on.
//
// The interesting test here is not that the helpers run — it is that applying
// a scale to a weight's columns and the reciprocal to its producer leaves the
// product unchanged. If that ever stops holding, a "calibrated" checkpoint is
// silently a different model, and no quantization metric would say so.

#include "core/fp_bits.h"
#include "quant/awq_norm_fold.h"
#include "quant/awq_transform.h"
#include "quant/calibration_stats.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <random>
#include <vector>

using namespace imp;

namespace {

std::string temp_path(const char* stem) { return (std::filesystem::temp_directory_path() / stem).string(); }

uint16_t f2h(float f) {
    // Minimal IEEE binary32 -> binary16, round-to-nearest-even, no Inf/NaN
    // inputs expected. Kept local so the test does not depend on CUDA headers.
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = static_cast<int32_t>((x >> 23) & 0xFF) - 127 + 15;
    uint32_t man = x & 0x7FFFFFu;
    if (exp <= 0)
        return static_cast<uint16_t>(sign);
    if (exp >= 31)
        return static_cast<uint16_t>(sign | 0x7BFFu);
    uint16_t out = static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (man >> 13));
    const uint32_t rem = man & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (out & 1u)))
        out++;
    return out;
}

float h2f(uint16_t h) {
    const uint32_t sign = (h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1Fu;
    const uint32_t man = h & 0x3FFu;
    if (exp == 0) {
        if (man == 0) {
            float f;
            std::memcpy(&f, &sign, 4);
            return f;
        }
        float v = static_cast<float>(man) / 1024.0f / 16384.0f;
        return (sign ? -v : v);
    }
    const uint32_t bits = sign | ((exp - 15 + 127) << 23) | (man << 13);
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

}  // namespace

TEST(AwqCalibration, StatsRoundTrip) {
    CalibrationStats in;
    in.model_id = "/models/Qwen3-0.6B";
    in.entries.push_back({3, "WQ", 4096, {0.5f, 1.25f, 2.0f}});
    in.entries.push_back({3, "W_DOWN", 4096, {7.5f}});

    const std::string path = temp_path("imp_calib_roundtrip.bin");
    ASSERT_EQ(write_calibration_stats(path, in), "");

    CalibrationStats out;
    ASSERT_EQ(read_calibration_stats(path, out), "");
    EXPECT_EQ(out.model_id, in.model_id);
    ASSERT_EQ(out.entries.size(), 2u);
    const CalibrationEntry* q = out.find(3, "WQ");
    ASSERT_NE(q, nullptr);
    EXPECT_EQ(q->rows, 4096u);
    ASSERT_EQ(q->mean_abs.size(), 3u);
    EXPECT_FLOAT_EQ(q->mean_abs[1], 1.25f);
    EXPECT_EQ(out.find(3, "WK"), nullptr);
    EXPECT_EQ(out.find(4, "WQ"), nullptr);
    std::filesystem::remove(path);
}

TEST(AwqCalibration, RejectsForeignAndTruncatedFiles) {
    const std::string path = temp_path("imp_calib_foreign.bin");
    {
        FILE* f = std::fopen(path.c_str(), "wb");
        ASSERT_NE(f, nullptr);
        const char junk[] = "not a calibration file at all";
        std::fwrite(junk, 1, sizeof(junk), f);
        std::fclose(f);
    }
    CalibrationStats out;
    EXPECT_NE(read_calibration_stats(path, out), "");

    // A file with the right magic but a body cut short must not read as an
    // empty-but-valid calibration — that would quantize as round-to-nearest
    // while claiming to be calibrated.
    CalibrationStats good;
    good.entries.push_back({0, "WQ", 8, std::vector<float>(64, 1.0f)});
    ASSERT_EQ(write_calibration_stats(path, good), "");
    const auto full = std::filesystem::file_size(path);
    std::filesystem::resize_file(path, full - 32);
    CalibrationStats trunc;
    EXPECT_NE(read_calibration_stats(path, trunc), "");
    std::filesystem::remove(path);
}

// The core invariant: scaling a weight's input channels and dividing the
// producing norm by the same vector must not move the layer's output.
TEST(AwqCalibration, ScaleAndFoldPreserveTheProduct) {
    constexpr int64_t N = 24, K = 32;
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> wd(-0.4f, 0.4f);
    std::uniform_real_distribution<float> xd(-2.0f, 2.0f);

    std::vector<uint16_t> w(static_cast<size_t>(N * K));
    for (auto& v : w)
        v = f2h(wd(rng));
    std::vector<float> x(K), g(K);
    for (int64_t j = 0; j < K; j++) {
        x[static_cast<size_t>(j)] = xd(rng);
        g[static_cast<size_t>(j)] = 0.5f + 0.5f * std::fabs(xd(rng));
    }
    // Scales spanning a decade, which is the range AWQ actually produces.
    std::vector<float> s(K);
    for (int64_t j = 0; j < K; j++)
        s[static_cast<size_t>(j)] = 0.3f + 0.27f * static_cast<float>(j % 10);

    auto matvec = [&](const std::vector<uint16_t>& mat, const std::vector<float>& in) {
        std::vector<double> y(static_cast<size_t>(N), 0.0);
        for (int64_t i = 0; i < N; i++)
            for (int64_t j = 0; j < K; j++)
                y[static_cast<size_t>(i)] += static_cast<double>(h2f(mat[static_cast<size_t>(i * K + j)])) *
                                             static_cast<double>(in[static_cast<size_t>(j)]);
        return y;
    };

    std::vector<float> pre(K);
    for (int64_t j = 0; j < K; j++)
        pre[static_cast<size_t>(j)] = x[static_cast<size_t>(j)] * g[static_cast<size_t>(j)];
    const std::vector<double> ref = matvec(w, pre);

    // Transform: consumer columns *= s, producer (the norm weight) /= s.
    std::vector<uint16_t> w2 = w;
    awq_apply_matrix(w2, N, K, /*row_div=*/{}, /*col_scale=*/s);
    std::vector<float> g2 = g;
    ASSERT_TRUE(
        awq_apply_vector_div(reinterpret_cast<unsigned char*>(g2.data()), static_cast<size_t>(K), "F32", s));
    std::vector<float> pre2(K);
    for (int64_t j = 0; j < K; j++)
        pre2[static_cast<size_t>(j)] = x[static_cast<size_t>(j)] * g2[static_cast<size_t>(j)];
    const std::vector<double> got = matvec(w2, pre2);

    // The only error source is re-rounding W*s back to FP16, which perturbs
    // each PRODUCT by at most one ulp. So the bound is over the sum of term
    // magnitudes, not over |y| — dividing by |y| would measure how much the
    // dot product cancels, which has nothing to do with the invariant.
    for (int64_t i = 0; i < N; i++) {
        double term_mag = 0.0;
        for (int64_t j = 0; j < K; j++)
            term_mag += std::fabs(static_cast<double>(h2f(w[static_cast<size_t>(i * K + j)])) *
                                  static_cast<double>(pre[static_cast<size_t>(j)]));
        const double kFp16Eps = 4.883e-4;  // 2^-11
        EXPECT_LT(std::fabs(got[static_cast<size_t>(i)] - ref[static_cast<size_t>(i)]),
                  2.0 * kFp16Eps * term_mag)
            << "row " << i;
    }
}

// The producer-side fold for o_proj/down_proj divides ROWS of the producer;
// composing that with a column scale on the same tensor must commute with
// doing them separately, since they act on different axes.
TEST(AwqCalibration, RowAndColumnTransformsCompose) {
    constexpr int64_t N = 8, K = 16;
    std::vector<uint16_t> w(static_cast<size_t>(N * K));
    for (size_t i = 0; i < w.size(); i++)
        w[i] = f2h(0.125f * static_cast<float>((i % 7) + 1));
    std::vector<float> rows(N), cols(K);
    for (int64_t i = 0; i < N; i++)
        rows[static_cast<size_t>(i)] = 1.0f + 0.25f * static_cast<float>(i % 4);
    for (int64_t j = 0; j < K; j++)
        cols[static_cast<size_t>(j)] = 0.5f + 0.5f * static_cast<float>(j % 3);

    std::vector<uint16_t> both = w;
    awq_apply_matrix(both, N, K, rows, cols);

    std::vector<uint16_t> staged = w;
    awq_apply_matrix(staged, N, K, rows, {});
    awq_apply_matrix(staged, N, K, {}, cols);

    for (int64_t i = 0; i < N; i++)
        for (int64_t j = 0; j < K; j++) {
            const size_t idx = static_cast<size_t>(i * K + j);
            EXPECT_NEAR(h2f(both[idx]), h2f(staged[idx]), 1e-3f) << "at " << i << "," << j;
        }
}

TEST(AwqCalibration, VectorFoldRefusesWhatItCannotDo) {
    std::vector<float> v(4, 1.0f);
    auto* bytes = reinterpret_cast<unsigned char*>(v.data());
    // Length mismatch: silently folding the first n would corrupt the rest.
    EXPECT_FALSE(awq_apply_vector_div(bytes, 4, "F32", std::vector<float>(3, 2.0f)));
    // Unsupported dtype: an int-typed norm is not something to guess at.
    EXPECT_FALSE(awq_apply_vector_div(bytes, 4, "I8", std::vector<float>(4, 2.0f)));
    for (float f : v)
        EXPECT_FLOAT_EQ(f, 1.0f);
    EXPECT_TRUE(awq_apply_vector_div(bytes, 4, "F32", std::vector<float>(4, 2.0f)));
    for (float f : v)
        EXPECT_FLOAT_EQ(f, 0.5f);
}

// ---------------------------------------------------------------------------
// The unit-offset fold (Qwen3.5 / 3.6 / 3.8) and the storage hazard it carries.
// ---------------------------------------------------------------------------

// The real worst channel of Qwen3.8-27B: layer 0's post_attention_layernorm
// stores -0.99609375, so the gain the kernel applies is 2^-8.
constexpr float kWorstDelta = -0.99609375f;
constexpr float kWorstGain = 0.00390625f;

// The identity the whole family rests on: (1 + g') == (1 + g)/s.
TEST(AwqNormFold, UnitOffsetFoldDividesTheGain) {
    for (float g : {0.25f, -0.5f, 1.5f, -0.05f}) {
        for (float s : {0.5f, 1.25f, 2.0f}) {
            const float stored = awq_fold_norm_value(g, s, NormOffset::Unit);
            EXPECT_NEAR(awq_norm_gain(stored, NormOffset::Unit), (1.0f + g) / s, 1e-6f)
                << "g=" << g << " s=" << s;
            // The plain rule on the same input divides the DELTA, which is a
            // different model whenever the offset is real.
            EXPECT_NEAR(awq_fold_norm_value(g, s, NormOffset::Plain), g / s, 1e-6f);
        }
    }
}

// Storing that identity in BF16 is where it breaks, and by how much.
TEST(AwqNormFold, BF16LosesTheNearZeroGainChannel) {
    struct Case {
        float s;
        float min_loss;
    };
    // Measured on the checkpoint: 25 % at 1.25, 50 % at 1.5, and at s >= 2 the
    // stored delta saturates to exactly -1 and the channel is deleted.
    for (const Case c : {Case{1.25f, 0.24f}, Case{1.5f, 0.49f}, Case{2.0f, 0.99f}}) {
        const float err = awq_fold_gain_error(kWorstDelta, c.s, NormOffset::Unit, "BF16");
        EXPECT_GT(err, c.min_loss) << "s=" << c.s;
    }
    // s = 2 is the annihilation case: the recovered gain is exactly zero.
    const float stored = awq_round_to_dtype(awq_fold_norm_value(kWorstDelta, 2.0f, NormOffset::Unit), "BF16");
    EXPECT_FLOAT_EQ(stored, -1.0f);
    EXPECT_FLOAT_EQ(awq_norm_gain(stored, NormOffset::Unit), 0.0f);
    // F32 storage is not the fix (the loader applies +1 on BF16 paths only) but
    // it is the proof that the dtype, not the algebra, is what fails.
    EXPECT_LT(awq_fold_gain_error(kWorstDelta, 2.0f, NormOffset::Unit, "F32"), 1e-6f);
}

// The clamp keeps the channel instead of deleting it: never a gain of zero,
// always inside the bound, and never a divisor further from 1 than asked for.
TEST(AwqNormFold, ClampKeepsTheChannelInsideTheBound) {
    constexpr float kTol = 0.01f;
    for (float s : {1.25f, 1.5f, 2.0f, 4.0f}) {
        const float clamped = awq_clamp_norm_divisor(kWorstDelta, s, NormOffset::Unit, "BF16", kTol);
        EXPECT_LT(clamped, s) << "s=" << s;
        EXPECT_GE(clamped, 1.0f) << "s=" << s;
        EXPECT_LE(awq_fold_gain_error(kWorstDelta, clamped, NormOffset::Unit, "BF16"), kTol) << "s=" << s;
        const float stored = awq_round_to_dtype(awq_fold_norm_value(kWorstDelta, clamped, NormOffset::Unit),
                                                "BF16");
        EXPECT_GT(awq_norm_gain(stored, NormOffset::Unit), 0.0f) << "channel deleted at s=" << s;
    }
    // A healthy channel is not touched: the bound already holds, so the search
    // must return the divisor unchanged rather than something near it.
    EXPECT_FLOAT_EQ(awq_clamp_norm_divisor(0.25f, 1.5f, NormOffset::Unit, "BF16", kTol), 1.5f);
}

// A plain norm stores g/s, whose relative error is a half-ulp by construction,
// so the clamp is a no-op there. That is what makes it safe to run the same
// code on every architecture.
TEST(AwqNormFold, PlainSitesAreNeverClamped) {
    for (float g : {kWorstDelta, 0.87f, -0.5f}) {
        for (float s : {1.25f, 2.0f, 4.0f}) {
            EXPECT_LT(awq_fold_gain_error(g, s, NormOffset::Plain, "BF16"), 0.01f);
            EXPECT_FLOAT_EQ(awq_clamp_norm_divisor(g, s, NormOffset::Plain, "BF16", 0.01f), s);
        }
    }
}

TEST(AwqNormFold, ClampReportCountsAndBoundsWhatItMoved) {
    // Source values as a BF16 tensor holds them: a fold reads stored bytes, and
    // -0.9f is not a BF16 value, so a raw literal would charge the fold for the
    // rounding the checkpoint had already done.
    std::vector<float> g;
    for (float v : {kWorstDelta, 0.25f, -0.9f, -0.99f})
        g.push_back(bf16_to_float(float_to_bf16(v)));
    std::vector<float> div = {2.0f, 2.0f, 2.0f, 2.0f};
    NormFoldReport rep;
    awq_clamp_norm_divisors(g.data(), g.size(), NormOffset::Unit, "BF16", 0.01f, div, rep);
    EXPECT_EQ(rep.channels, 4u);
    EXPECT_GE(rep.clamped, 1u);
    EXPECT_LT(rep.clamped, 4u) << "a healthy channel must not be clamped";
    EXPECT_LE(rep.worst_rel_err, 0.01f);
    EXPECT_FLOAT_EQ(div[1], 2.0f) << "g = 0.25 is representable at s = 2";
    EXPECT_LT(div[0], 2.0f);
    // The clamped divisor is the one the consumer must use, so the product it
    // reconstructs is the folded gain times that divisor.
    for (size_t j = 0; j < g.size(); j++) {
        const float stored = awq_round_to_dtype(awq_fold_norm_value(g[j], div[j], NormOffset::Unit), "BF16");
        const float recovered = awq_norm_gain(stored, NormOffset::Unit) * div[j];
        const float want = awq_norm_gain(g[j], NormOffset::Unit);
        EXPECT_NEAR(recovered, want, 0.01f * std::fabs(want) + 1e-7f) << "channel " << j;
    }
}

// The writer's side: the same offset must reach the bytes, or the plan and the
// checkpoint describe different models.
TEST(AwqNormFold, VectorFoldWritesTheOffsetAwareValue) {
    std::vector<uint16_t> bf16 = {float_to_bf16(0.25f), float_to_bf16(kWorstDelta)};
    std::vector<uint16_t> plain = bf16;
    const std::vector<float> div = {2.0f, 1.0f};
    auto* off_bytes = reinterpret_cast<unsigned char*>(bf16.data());
    auto* plain_bytes = reinterpret_cast<unsigned char*>(plain.data());
    ASSERT_TRUE(awq_apply_vector_div(off_bytes, 2, "BF16", div, NormOffset::Unit));
    ASSERT_TRUE(awq_apply_vector_div(plain_bytes, 2, "BF16", div, NormOffset::Plain));
    EXPECT_NE(bf16[0], plain[0]) << "the offset must change what is stored";
    EXPECT_NEAR(1.0f + bf16_to_float(bf16[0]), 1.25f / 2.0f, 0.01f);
    EXPECT_NEAR(bf16_to_float(plain[0]), 0.25f / 2.0f, 0.01f);
    // Divisor 1 is a no-op under both conventions.
    EXPECT_EQ(bf16[1], float_to_bf16(kWorstDelta));
    EXPECT_EQ(plain[1], float_to_bf16(kWorstDelta));
}

// A wrong-length plan vector must be ignored rather than partially applied:
// the writer applies the plan to every tensor and relies on that.
TEST(AwqCalibration, MatrixTransformIgnoresMismatchedVectors) {
    constexpr int64_t N = 4, K = 8;
    std::vector<uint16_t> w(static_cast<size_t>(N * K), f2h(1.0f));
    const std::vector<uint16_t> before = w;
    awq_apply_matrix(w, N, K, std::vector<float>(N + 1, 2.0f), std::vector<float>(K - 1, 3.0f));
    EXPECT_EQ(w, before);
    awq_apply_matrix(w, N, K, {}, std::vector<float>(K, 2.0f));
    EXPECT_NEAR(h2f(w[0]), 2.0f, 1e-3f);
}
