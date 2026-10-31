// qknorm_rope_fused (#1957) shipped with zero test references, backed only by an e2e
// throughput number; a wrong norm epsilon, pair layout or per-head stride still produces
// plausible activations, and its rows (n<=64) are exactly what a batch-1 greedy lock never
// enters. Reference: host RMSNorm + standalone rope_forward (each covered elsewhere), so a
// disagreement here is the fusion. Shape: Qwen3.8-27B (head_dim 256, rope 64, GQA 16/2) at
// real dispatch rows 1/32/64.

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "core/tensor.h"
#include "compute/rope.h"

#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

using namespace imp;

namespace {

#define QK_CUDA_CHECK(x)                                                                   \
    do {                                                                                   \
        cudaError_t e_ = (x);                                                              \
        ASSERT_EQ(e_, cudaSuccess) << #x << " -> " << cudaGetErrorString(e_);              \
    } while (0)

bool cuda_available() {
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

// Deterministic, sign-mixed, and away from zero so a dropped term shows.
float sample(int64_t i) { return 0.4f * std::sin(0.017f * static_cast<float>(i)) + 0.05f; }

std::vector<half> to_half(const std::vector<float>& v) {
    std::vector<half> h(v.size());
    for (size_t i = 0; i < v.size(); ++i)
        h[i] = __float2half(v[i]);
    return h;
}

std::vector<float> to_float(const std::vector<half>& v) {
    std::vector<float> f(v.size());
    for (size_t i = 0; i < v.size(); ++i)
        f[i] = __half2float(v[i]);
    return f;
}

void* upload(const void* src, size_t bytes) {
    void* d = nullptr;
    if (cudaMalloc(&d, bytes) != cudaSuccess)
        return nullptr;
    if (cudaMemcpy(d, src, bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(d);
        return nullptr;
    }
    return d;
}

// Per-head RMSNorm over head_dim, in FP32, exactly as the standalone kernel
// does it: x * rsqrt(mean(x^2) + eps) * w.
void host_qk_norm(std::vector<float>& x, const std::vector<float>& w, int n_tokens, int heads,
                  int head_dim, float eps) {
    for (int t = 0; t < n_tokens; ++t) {
        for (int h = 0; h < heads; ++h) {
            float* row = x.data() + (static_cast<int64_t>(t) * heads + h) * head_dim;
            double ss = 0.0;
            for (int d = 0; d < head_dim; ++d)
                ss += static_cast<double>(row[d]) * static_cast<double>(row[d]);
            const float scale =
                1.0f / std::sqrt(static_cast<float>(ss / head_dim) + eps);
            for (int d = 0; d < head_dim; ++d)
                row[d] = row[d] * scale * w[d];
        }
    }
}

struct FusedCase {
    int n_tokens;
};

class QkNormRopeFusedTest : public ::testing::TestWithParam<FusedCase> {};

TEST_P(QkNormRopeFusedTest, MatchesNormThenStandaloneRope) {
    if (!cuda_available())
        GTEST_SKIP() << "no CUDA device";

    const int n_tokens = GetParam().n_tokens;
    const int n_heads = 16;
    const int n_kv_heads = 2;
    const int head_dim = 256;
    const int rope_dim = 64;
    const float theta = 10000.0f;
    const float eps = 1e-6f;

    const size_t q_count = static_cast<size_t>(n_tokens) * n_heads * head_dim;
    const size_t k_count = static_cast<size_t>(n_tokens) * n_kv_heads * head_dim;

    std::vector<float> q_in(q_count), k_in(k_count), qw(head_dim), kw(head_dim);
    for (size_t i = 0; i < q_count; ++i)
        q_in[i] = sample(static_cast<int64_t>(i));
    for (size_t i = 0; i < k_count; ++i)
        k_in[i] = sample(static_cast<int64_t>(i) + 7919);
    for (int d = 0; d < head_dim; ++d) {
        qw[d] = 0.8f + 0.4f * sample(d);          // never 1.0: a dropped weight would pass
        kw[d] = 1.2f - 0.3f * sample(d + 131);
    }

    // Positions past the first block, so the angle is not near zero.
    std::vector<int> pos(n_tokens);
    for (int t = 0; t < n_tokens; ++t)
        pos[t] = 1000 + t;

    const std::vector<half> q_h = to_half(q_in), k_h = to_half(k_in);
    const std::vector<half> qw_h = to_half(qw), kw_h = to_half(kw);

    // ---- arm A: the fused kernel
    void* d_q_fused = upload(q_h.data(), q_h.size() * sizeof(half));
    void* d_k_fused = upload(k_h.data(), k_h.size() * sizeof(half));
    void* d_qw = upload(qw_h.data(), qw_h.size() * sizeof(half));
    void* d_kw = upload(kw_h.data(), kw_h.size() * sizeof(half));
    void* d_pos = upload(pos.data(), pos.size() * sizeof(int));
    ASSERT_NE(d_q_fused, nullptr);
    ASSERT_NE(d_k_fused, nullptr);
    ASSERT_NE(d_qw, nullptr);
    ASSERT_NE(d_kw, nullptr);
    ASSERT_NE(d_pos, nullptr);

    qknorm_rope_fused(static_cast<half*>(d_q_fused), static_cast<half*>(d_k_fused),
                      static_cast<const half*>(d_qw), static_cast<const half*>(d_kw), n_heads,
                      n_kv_heads, head_dim, eps, static_cast<const int*>(d_pos), theta,
                      /*scaling=*/1.0f, rope_dim, /*neox=*/true, /*stream=*/nullptr,
                      /*weight_offset=*/0.0f, /*ext_factor=*/0.0f, /*attn_factor=*/1.0f,
                      /*corr_dims=*/nullptr, /*longrope_inv_freqs=*/nullptr, MRopeParams{},
                      n_tokens);
    QK_CUDA_CHECK(cudaDeviceSynchronize());

    // ---- arm B: host RMSNorm, then the standalone RoPE kernel
    std::vector<float> q_ref(q_in), k_ref(k_in);
    host_qk_norm(q_ref, qw, n_tokens, n_heads, head_dim, eps);
    host_qk_norm(k_ref, kw, n_tokens, n_kv_heads, head_dim, eps);
    const std::vector<half> q_ref_h = to_half(q_ref), k_ref_h = to_half(k_ref);

    void* d_q_ref = upload(q_ref_h.data(), q_ref_h.size() * sizeof(half));
    void* d_k_ref = upload(k_ref_h.data(), k_ref_h.size() * sizeof(half));
    ASSERT_NE(d_q_ref, nullptr);
    ASSERT_NE(d_k_ref, nullptr);

    int64_t q_shape[4] = {1, n_tokens, n_heads, head_dim};
    int64_t k_shape[4] = {1, n_tokens, n_kv_heads, head_dim};
    Tensor Q(d_q_ref, QType::F16, 4, q_shape, true);
    Tensor K(d_k_ref, QType::F16, 4, k_shape, true);
    rope_forward(Q, K, static_cast<const int*>(d_pos), head_dim, theta, /*scaling=*/1.0f, rope_dim,
                 /*neox=*/true);
    QK_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<half> q_a(q_count), k_a(k_count), q_b(q_count), k_b(k_count);
    QK_CUDA_CHECK(cudaMemcpy(q_a.data(), d_q_fused, q_count * sizeof(half), cudaMemcpyDeviceToHost));
    QK_CUDA_CHECK(cudaMemcpy(k_a.data(), d_k_fused, k_count * sizeof(half), cudaMemcpyDeviceToHost));
    QK_CUDA_CHECK(cudaMemcpy(q_b.data(), d_q_ref, q_count * sizeof(half), cudaMemcpyDeviceToHost));
    QK_CUDA_CHECK(cudaMemcpy(k_b.data(), d_k_ref, k_count * sizeof(half), cudaMemcpyDeviceToHost));

    const std::vector<float> qa = to_float(q_a), kb_a = to_float(k_a);
    const std::vector<float> qb = to_float(q_b), kb_b = to_float(k_b);

    // FP16 storage on both arms, FP32 math on both: the residue is the rounding
    // of one intermediate the fused kernel keeps in registers.
    const float tol = 4e-3f;
    for (size_t i = 0; i < q_count; ++i)
        ASSERT_NEAR(qa[i], qb[i], tol) << "Q mismatch at " << i << " (n_tokens=" << n_tokens << ")";
    for (size_t i = 0; i < k_count; ++i)
        ASSERT_NEAR(kb_a[i], kb_b[i], tol) << "K mismatch at " << i << " (n_tokens=" << n_tokens << ")";

    // Dims past rope_dim are normed but not rotated; asserting they moved at all separates "the
    // fusion works" from "the kernel wrote the input back unchanged" (rope_dim 64 of 256 means
    // 3/4 of each head would survive a no-op RoPE unnoticed).
    bool tail_changed = false;
    for (size_t t = 0; t < static_cast<size_t>(n_tokens) && !tail_changed; ++t) {
        const size_t base = t * n_heads * head_dim;
        for (int d = rope_dim; d < head_dim; ++d)
            if (std::fabs(qa[base + d] - q_in[base + d]) > 1e-3f) {
                tail_changed = true;
                break;
            }
    }
    EXPECT_TRUE(tail_changed) << "dims >= rope_dim came back unchanged: the QK-norm half did nothing";

    cudaFree(d_q_fused);
    cudaFree(d_k_fused);
    cudaFree(d_q_ref);
    cudaFree(d_k_ref);
    cudaFree(d_qw);
    cudaFree(d_kw);
    cudaFree(d_pos);
}

INSTANTIATE_TEST_SUITE_P(RowCounts, QkNormRopeFusedTest,
                         ::testing::Values(FusedCase{1}, FusedCase{32}, FusedCase{64}),
                         [](const ::testing::TestParamInfo<FusedCase>& i) {
                             return "n" + std::to_string(i.param.n_tokens);
                         });

// The warp-per-head kernel must reproduce the CTA kernel bit for bit (same per-element math and
// sum order), on every rotary variant it serves: neox full/partial, interleaved, weight offset,
// YaRN, precomputed frequencies.
struct WarpCase {
    const char* name;
    int n_tokens, n_heads, n_kv_heads, head_dim, rope_dim;
    bool neox;
    float weight_offset, ext_factor;
    bool longrope;
};

class QkNormRopeWarpTest : public ::testing::TestWithParam<WarpCase> {};

TEST_P(QkNormRopeWarpTest, BitIdenticalToCtaKernel) {
    if (!cuda_available())
        GTEST_SKIP() << "no CUDA device";
    const WarpCase c = GetParam();
    const size_t q_count = static_cast<size_t>(c.n_tokens) * c.n_heads * c.head_dim;
    const size_t k_count = static_cast<size_t>(c.n_tokens) * c.n_kv_heads * c.head_dim;
    std::vector<float> q_in(q_count), k_in(k_count), qw(c.head_dim), kw(c.head_dim), freqs(c.rope_dim / 2);
    for (size_t i = 0; i < q_count; ++i)
        q_in[i] = 3.0f * sample(static_cast<int64_t>(i) * 13);
    for (size_t i = 0; i < k_count; ++i)
        k_in[i] = 3.0f * sample(static_cast<int64_t>(i) * 13 + 7919);
    for (int d = 0; d < c.head_dim; ++d) {
        qw[d] = 0.8f + 0.4f * sample(d);
        kw[d] = 1.2f - 0.3f * sample(d + 131);
    }
    for (int p = 0; p < c.rope_dim / 2; ++p)
        freqs[p] = std::pow(1e6f, -2.0f * p / c.rope_dim) / (1.0f + 0.1f * (p % 7));
    std::vector<int> pos(c.n_tokens);
    for (int t = 0; t < c.n_tokens; ++t)
        pos[t] = 37 + 61 * t;
    const float corr_dims[2] = {4.0f, 40.0f};

    const std::vector<half> q_h = to_half(q_in), k_h = to_half(k_in), qw_h = to_half(qw), kw_h = to_half(kw);
    void* d_q[2] = {upload(q_h.data(), q_count * sizeof(half)), upload(q_h.data(), q_count * sizeof(half))};
    void* d_k[2] = {upload(k_h.data(), k_count * sizeof(half)), upload(k_h.data(), k_count * sizeof(half))};
    void* d_qw = upload(qw_h.data(), qw_h.size() * sizeof(half));
    void* d_kw = upload(kw_h.data(), kw_h.size() * sizeof(half));
    void* d_pos = upload(pos.data(), pos.size() * sizeof(int));
    void* d_freqs = upload(freqs.data(), freqs.size() * sizeof(float));
    for (void* p : {d_q[0], d_q[1], d_k[0], d_k[1], d_qw, d_kw, d_pos, d_freqs})
        ASSERT_NE(p, nullptr);

    using Fn = decltype(&qknorm_rope_fused);
    const Fn fns[2] = {&qknorm_rope_fused_cta, &qknorm_rope_fused};
    for (int a = 0; a < 2; ++a)
        fns[a](static_cast<half*>(d_q[a]), static_cast<half*>(d_k[a]), static_cast<const half*>(d_qw),
               static_cast<const half*>(d_kw), c.n_heads, c.n_kv_heads, c.head_dim, 1e-6f,
               static_cast<const int*>(d_pos), 1e6f, /*scaling=*/c.ext_factor > 0.0f ? 4.0f : 1.0f,
               c.rope_dim, c.neox, nullptr, c.weight_offset, c.ext_factor, /*attn_factor=*/1.1f,
               c.ext_factor > 0.0f ? corr_dims : nullptr,
               c.longrope ? static_cast<const float*>(d_freqs) : nullptr, MRopeParams{}, c.n_tokens);
    QK_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<uint16_t> out[2][2];
    for (int a = 0; a < 2; ++a) {
        out[a][0].resize(q_count);
        out[a][1].resize(k_count);
        QK_CUDA_CHECK(cudaMemcpy(out[a][0].data(), d_q[a], q_count * 2, cudaMemcpyDeviceToHost));
        QK_CUDA_CHECK(cudaMemcpy(out[a][1].data(), d_k[a], k_count * 2, cudaMemcpyDeviceToHost));
    }
    size_t changed = 0;
    for (size_t i = 0; i < q_count; ++i) {
        ASSERT_EQ(out[0][0][i], out[1][0][i]) << c.name << ": Q bits differ at " << i;
        changed += out[1][0][i] != reinterpret_cast<const uint16_t*>(q_h.data())[i];
    }
    for (size_t i = 0; i < k_count; ++i)
        ASSERT_EQ(out[0][1][i], out[1][1][i]) << c.name << ": K bits differ at " << i;
    EXPECT_GT(changed, q_count / 2) << c.name << ": both kernels left Q mostly untouched";

    for (void* p : {d_q[0], d_q[1], d_k[0], d_k[1], d_qw, d_kw, d_pos, d_freqs})
        cudaFree(p);
}

INSTANTIATE_TEST_SUITE_P(
    Variants, QkNormRopeWarpTest,
    ::testing::Values(WarpCase{"qwen3_decode", 32, 40, 8, 128, 128, true, 0.0f, 0.0f, false},
                      WarpCase{"qwen3_prefill", 2048, 40, 8, 128, 128, true, 0.0f, 0.0f, false},
                      WarpCase{"qwen38_partial", 64, 16, 2, 256, 64, true, 0.0f, 0.0f, false},
                      WarpCase{"interleaved_partial", 33, 8, 4, 64, 32, false, 0.0f, 0.0f, false},
                      WarpCase{"gemma_offset_longrope", 17, 8, 1, 512, 512, true, 1.0f, 0.0f, true},
                      WarpCase{"yarn", 9, 12, 4, 128, 128, true, 0.0f, 1.0f, false},
                      WarpCase{"cta_fallback_hd96", 5, 4, 2, 96, 96, true, 0.0f, 0.0f, false}),
    [](const ::testing::TestParamInfo<WarpCase>& i) { return std::string(i.param.name); });

}  // namespace
