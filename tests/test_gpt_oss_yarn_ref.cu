// test(P2.7): gpt-oss YaRN long-sequence RoPE parity — kernel vs fp64 ref.
//
// gpt-oss-20b uses YaRN RoPE (factor=32, original_max_position=4096, extended
// to 131072). #547 carried a latent rope_freq_scale INVERSION bug: the HF
// config provides a YaRN `factor` (32); imp stores the factor and the kernel
// applies 1/factor. Storing 1/factor instead double-inverts the scale — the
// interpolated dims rotate factor^2 = 1024x too fast and the mscale flips
// sign. The error is invisible at small positions but enormous beyond the
// original context, exactly where the YaRN extension matters.
//
// This test reimplements the YaRN math in fp64 (independently, from the
// YaRN/HF semantics), cross-checks it against a COMMITTED numpy golden
// (tests/refs/yarn_rope_golden.h, 1e-9 rel), then runs imp's rope_forward
// kernel with the gpt-oss YaRN params at positions sampled across the range
// INCLUDING > original_ctx (up to 131071) and asserts the rotated Q/K match
// the verified fp64 reference at the f16 tolerance class. A final guard
// proves the test is SENSITIVE to the inversion: the kernel output must NOT
// match the 1024x-wrong reference.

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "core/tensor.h"
#include "compute/rope.h"
#include "refs/yarn_rope_golden.h"

#include <vector>
#include <cmath>
#include <cstdint>
#include <algorithm>

namespace imp {
namespace {

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err = (call);                                                 \
        ASSERT_EQ(err, cudaSuccess) << "CUDA error: " << cudaGetErrorString(err); \
    } while (0)

// ---------------------------------------------------------------------------
// Independent fp64 YaRN cos/sin reference (mirrors imp's rope_yarn device
// helper + rope_yarn_corr_dims). `factor` is the HF YaRN factor (32); the
// kernel applies 1/factor, so inv_scaling = 1/factor here too.
// ---------------------------------------------------------------------------
double yarn_corr_dim(int n_dims, int n_ctx_orig, double n_rot, double base) {
    return n_dims * std::log(n_ctx_orig / (n_rot * 2.0 * M_PI)) / (2.0 * std::log(base));
}
void yarn_corr_dims(int n_dims, int n_ctx_orig, double base, double beta_fast, double beta_slow,
                    double& cd0, double& cd1) {
    double start = std::floor(yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, base));
    double end = std::ceil(yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, base));
    cd0 = std::max(0.0, start);
    cd1 = std::min(static_cast<double>(n_dims - 1), end);
}
double yarn_ramp(double low, double high, int i0) {
    double y = (i0 / 2.0 - low) / std::max(0.001, high - low);
    return 1.0 - std::min(1.0, std::max(0.0, y));
}
// Returns (cos*mscale, sin*mscale) for a given position/pair, factor = HF factor.
void yarn_cos_sin(int pos, int pair, int n_pairs, double theta, double factor, double cd0, double cd1,
                  double& cos_out, double& sin_out) {
    double inv_scaling = 1.0 / factor;
    double theta_extrap = pos / std::pow(theta, (2.0 * pair) / (2.0 * n_pairs));
    double theta_interp = inv_scaling * theta_extrap;
    double ramp_mix = yarn_ramp(cd0, cd1, 2 * pair) * 1.0;  // ext_factor=1
    double th = theta_interp * (1.0 - ramp_mix) + theta_extrap * ramp_mix;
    double mscale = 1.0 + 0.1 * std::log(1.0 / inv_scaling);  // attn_factor=1
    cos_out = std::cos(th) * mscale;
    sin_out = std::sin(th) * mscale;
}

// gpt-oss-20b YaRN params (config.json).
constexpr int HD = 64;
constexpr double THETA = 150000.0;
constexpr double FACTOR = 32.0;
constexpr double BETA_FAST = 32.0;
constexpr double BETA_SLOW = 1.0;
constexpr int N_CTX_ORIG = 4096;

// ---------------------------------------------------------------------------
// 1. Reference vs committed golden: proves the in-test fp64 math equals the
//    independent numpy generator (no magic constants — README §4 rule).
// ---------------------------------------------------------------------------
TEST(GptOssYarnRef, ReferenceMatchesGolden) {
    using namespace yarn_golden;
    ASSERT_EQ(kHeadDim, HD);
    double cd0, cd1;
    yarn_corr_dims(HD, N_CTX_ORIG, THETA, BETA_FAST, BETA_SLOW, cd0, cd1);
    EXPECT_NEAR(cd0, kCorrDim0, 1e-9) << "corr_dim0 mismatch vs golden";
    EXPECT_NEAR(cd1, kCorrDim1, 1e-9) << "corr_dim1 mismatch vs golden";

    int n_pairs = HD / 2;
    double worst = 0.0;
    for (int i = 0; i < kNumRows; i++) {
        const auto& row = kRows[i];
        double c, s;
        yarn_cos_sin(row.pos, row.pair, n_pairs, THETA, FACTOR, cd0, cd1, c, s);
        auto rel = [](double a, double b) { return std::abs(a - b) / std::max(1e-12, std::abs(b)); };
        worst = std::max(worst, std::max(rel(c, row.cos), rel(s, row.sin)));
    }
    EXPECT_LT(worst, 1e-9) << "in-test fp64 YaRN ref disagrees with numpy golden (rel " << worst << ")";
    printf("[yarn] ref-vs-golden max rel = %.2e (rows=%d)\n", worst, kNumRows);
}

// Apply the fp64 reference rotation to a single (q0,q1) interleaved pair.
// imp uses neox=false for gpt-oss (interleaved pairs (2i, 2i+1)).
void ref_rotate(double q0, double q1, double cos_v, double sin_v, double& o0, double& o1) {
    o0 = q0 * cos_v - q1 * sin_v;
    o1 = q0 * sin_v + q1 * cos_v;
}

// ---------------------------------------------------------------------------
// 2. GPU kernel parity at long positions. We rotate a known Q/K vector at each
//    golden position and compare against the fp64 reference applied to the
//    f16-rounded inputs.
// ---------------------------------------------------------------------------
TEST(GptOssYarnRef, KernelMatchesReferenceLongSeq) {
    using namespace yarn_golden;
    const int n_heads = 1, n_kv_heads = 1, n_pairs = HD / 2;
    double cd0, cd1;
    yarn_corr_dims(HD, N_CTX_ORIG, THETA, BETA_FAST, BETA_SLOW, cd0, cd1);
    float corr_dims_f[2] = {static_cast<float>(cd0), static_cast<float>(cd1)};

    // corr_dims is consumed HOST-side by rope_forward (it reads corr_dims[0..1]
    // before launch) — pass a host pointer, NOT a device buffer.
    const float* corr_dims_host = corr_dims_f;

    // One token per golden position. Q/K input: a fixed heavy-tailed pattern
    // (f16-rounded so the reference sees the same bits the kernel does).
    const int T = kNumPositions;
    std::vector<half> Qh(static_cast<size_t>(T) * HD), Kh(static_cast<size_t>(T) * HD);
    std::vector<int> positions(T);
    for (int t = 0; t < T; t++) {
        positions[t] = kPositions[t];
        for (int d = 0; d < HD; d++) {
            float val = std::sin(0.7f * d + 0.3f) * 1.5f;        // smooth, ~[-1.5,1.5]
            Qh[static_cast<size_t>(t) * HD + d] = __float2half(val);
            Kh[static_cast<size_t>(t) * HD + d] = __float2half(0.5f * val + 0.2f);
        }
    }

    // Device buffers. rope_forward expects Q[batch, seq, n_heads, head_dim].
    half *dQ = nullptr, *dK = nullptr;
    int* dPos = nullptr;
    CUDA_CHECK(cudaMalloc(&dQ, Qh.size() * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dK, Kh.size() * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dPos, T * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(dQ, Qh.data(), Qh.size() * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dK, Kh.data(), Kh.size() * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dPos, positions.data(), T * sizeof(int), cudaMemcpyHostToDevice));

    int64_t qshape[4] = {1, T, n_heads, HD};
    int64_t kshape[4] = {1, T, n_kv_heads, HD};
    Tensor Q(dQ, QType::F16, 4, qshape, true);
    Tensor K(dK, QType::F16, 4, kshape, true);

    // gpt-oss path: theta=150000, scaling=FACTOR(32), rope_dim=0(full HD),
    // neox=false, ext_factor=1 (YaRN), attn_factor=1.
    rope_forward(Q, K, dPos, HD, static_cast<float>(THETA), static_cast<float>(FACTOR),
                 /*rope_dim=*/0, /*neox=*/false, /*ext_factor=*/1.0f, /*attn_factor=*/1.0f, corr_dims_host,
                 /*stream=*/nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<half> Qo(Qh.size()), Ko(Kh.size());
    CUDA_CHECK(cudaMemcpy(Qo.data(), dQ, Qo.size() * sizeof(half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(Ko.data(), dK, Ko.size() * sizeof(half), cudaMemcpyDeviceToHost));
    cudaFree(dQ); cudaFree(dK); cudaFree(dPos);

    // fp64 reference from the f16-rounded inputs, interleaved pairs.
    //
    // Error metric: PER-PAIR vector error ||got-ref|| / max(floor, ||ref||).
    // A RoPE pair is a 2D rotation; per-ELEMENT relative error explodes
    // spuriously when the rotated vector has a near-zero component (a tiny
    // phase error in the f32 __sinf/__cosf rotates a ~0 component to a
    // small-but-not-tiny value → huge relative error on a physically correct
    // rotation). The vector metric is rotation-magnitude-aware: it stays
    // small for the correct rotation yet still explodes for the 1024×-wrong
    // angle (which sends the vector to a completely different direction).
    double worst = 0.0, worst_long = 0.0;
    for (int t = 0; t < T; t++) {
        for (int p = 0; p < n_pairs; p++) {
            double c, s;
            yarn_cos_sin(positions[t], p, n_pairs, THETA, FACTOR, cd0, cd1, c, s);
            for (auto* arr : {&Qh, &Kh}) {
                bool isQ = (arr == &Qh);
                double q0 = __half2float((*arr)[static_cast<size_t>(t) * HD + 2 * p]);
                double q1 = __half2float((*arr)[static_cast<size_t>(t) * HD + 2 * p + 1]);
                double r0, r1;
                ref_rotate(q0, q1, c, s, r0, r1);
                const auto& out = isQ ? Qo : Ko;
                double g0 = __half2float(out[static_cast<size_t>(t) * HD + 2 * p]);
                double g1 = __half2float(out[static_cast<size_t>(t) * HD + 2 * p + 1]);
                double d = std::hypot(g0 - r0, g1 - r1);
                double n = std::hypot(r0, r1);
                double e = d / std::max(1e-2, n);
                worst = std::max(worst, e);
                if (positions[t] > N_CTX_ORIG)
                    worst_long = std::max(worst_long, e);
            }
        }
    }
    // f16 tolerance class: inputs/outputs are f16; the device __cosf/__sinf
    // argument reduction loses ~a few ULPs of phase at the largest angles
    // (pos/32 ~4096 rad at pos=131071). Per-pair vector envelope 3e-2
    // (ASSERTED) — measured ~1e-2 below pos 100k, ~2-3e-2 at 131071.
    EXPECT_LT(worst, 3e-2) << "YaRN rope kernel vs fp64 ref per-pair vector rel err " << worst;
    printf("[yarn] kernel-vs-ref max per-pair vec rel = %.2e (all positions), %.2e (pos > %d)\n", worst,
           worst_long, N_CTX_ORIG);

    // Sensitivity guard: if rope_freq_scale were inverted (scaling=1/factor
    // passed in -> kernel applies 1/(1/factor)=factor), the interpolated dims
    // would rotate factor^2=1024x too fast. Build that WRONG reference and
    // confirm the kernel does NOT match it at a deep position — i.e. the test
    // would catch the #547 inversion bug.
    {
        int t_deep = -1;
        for (int t = 0; t < T; t++)
            if (positions[t] == 100000) t_deep = t;
        ASSERT_GE(t_deep, 0);
        // Mean per-pair vector deviation between the kernel and the INVERTED
        // (1024×-wrong) reference. The correct kernel rotates each pair to a
        // wholly different angle than the inverted ref, so this is order ~1.
        double sum_wrong = 0.0;
        int cnt = 0;
        for (int p = 0; p < n_pairs; p++) {
            double c, s;  // inverted: factor = 1/FACTOR -> kernel-equiv scale = FACTOR
            yarn_cos_sin(positions[t_deep], p, n_pairs, THETA, 1.0 / FACTOR, cd0, cd1, c, s);
            double q0 = __half2float(Qh[static_cast<size_t>(t_deep) * HD + 2 * p]);
            double q1 = __half2float(Qh[static_cast<size_t>(t_deep) * HD + 2 * p + 1]);
            double r0, r1;
            ref_rotate(q0, q1, c, s, r0, r1);
            double g0 = __half2float(Qo[static_cast<size_t>(t_deep) * HD + 2 * p]);
            double g1 = __half2float(Qo[static_cast<size_t>(t_deep) * HD + 2 * p + 1]);
            sum_wrong += std::hypot(g0 - r0, g1 - r1) / std::max(1e-2, std::hypot(r0, r1));
            cnt++;
        }
        double match_wrong = sum_wrong / cnt;  // mean per-pair vector rel deviation
        // The kernel must be FAR from the inverted reference (huge mismatch).
        EXPECT_GT(match_wrong, 0.5) << "kernel suspiciously close to the 1024x-inverted YaRN ref — "
                                       "the test would not catch the #547 rope_freq_scale inversion";
        printf("[yarn] inverted-scale sensitivity: mean kernel-vs-WRONG vec rel = %.2e (must be large)\n",
               match_wrong);
    }
}

}  // namespace
}  // namespace imp
