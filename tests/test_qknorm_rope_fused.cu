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

}  // namespace
