// Qwen4Exp QSA indexer kernels (compute/qsa_indexer.h) against CPU references.
//   AllTrueSelectionMatchesDensePaged: below 512 complete blocks the selection is every
//     token in order, so select + gather + paged kernel on the scratch must equal the paged
//     kernel on the original cache byte for byte.
//   SelectMatchesCpuTopk: > 512 blocks, scores and the top-k set (ties: lowest index) and
//     the tail against a CPU reference.
//   PrepQueriesMatchesCpu: (1+w) norm + NeoX RoPE on the first rope_dim dims of a head.
#include <gtest/gtest.h>
#include "compute/attention_paged.h"
#include "compute/qsa_indexer.h"
#include "core/tensor.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

using namespace imp;

namespace {

constexpr int D = kQsaDim;

template <class T>
T* up(const std::vector<T>& h) {
    T* d = nullptr;
    cudaMalloc(&d, h.size() * sizeof(T));
    cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}
template <class T>
std::vector<T> down(const T* d, size_t n) {
    std::vector<T> h(n);
    cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost);
    return h;
}
std::vector<half> rand_half(size_t n, std::mt19937& rng, float lo = -1.0f, float hi = 1.0f) {
    std::uniform_real_distribution<float> u(lo, hi);
    std::vector<half> v(n);
    for (auto& x : v)
        x = __float2half(u(rng));
    return v;
}
Tensor f16(void* p, std::initializer_list<int64_t> s) {
    std::vector<int64_t> sh(s);
    return Tensor(p, QType::F16, static_cast<int>(sh.size()), sh.data(), true);
}
QsaGeom geom() {
    QsaGeom g{};
    g.n_heads = 4;
    g.ratio = 4;
    g.budget = 2048;
    g.rope_dim = 64;
    g.theta = 1.0e7f;
    g.eps = 1e-6f;
    return g;
}

}  // namespace

TEST(QsaIndexer, AllTrueSelectionMatchesDensePaged) {
    const int nh = 24, nkv = 2, hd = 256, bs = 16, rows = 6;
    const int positions_h[rows] = {0, 1, 3, 17, 157, 1300};
    const int max_ctx = 1301, num_blocks = (max_ctx + bs - 1) / bs;
    const QsaGeom g = geom();
    const int cap = g.budget + g.ratio - 1, bpr = (cap + bs - 1) / bs;
    std::mt19937 rng(7);
    auto K = rand_half(static_cast<size_t>(num_blocks) * bs * nkv * hd, rng);
    auto V = rand_half(K.size(), rng);
    auto Q = rand_half(static_cast<size_t>(rows) * nh * hd, rng);
    auto Qi = rand_half(static_cast<size_t>(rows) * g.n_heads * D, rng);       // indexer q (unused at <= 512)
    auto BK = rand_half(static_cast<size_t>(max_ctx / g.ratio + 1) * D, rng);  // block keys
    std::vector<int> bt(static_cast<size_t>(rows) * num_blocks), ctx(rows), pos(positions_h, positions_h + rows);
    for (int r = 0; r < rows; ++r) {
        std::iota(bt.begin() + static_cast<size_t>(r) * num_blocks, bt.begin() + static_cast<size_t>(r + 1) * num_blocks, 0);
        ctx[r] = pos[r] + 1;
    }
    half *dK = up(K), *dV = up(V), *dQ = up(Q), *dQi = up(Qi), *dBK = up(BK);
    int *dbt = up(bt), *dctx = up(ctx), *dpos = up(pos);
    half* dO_dense = nullptr;
    half* dO_sel = nullptr;
    cudaMalloc(&dO_dense, Q.size() * sizeof(half));
    cudaMalloc(&dO_sel, Q.size() * sizeof(half));
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    paged_attention_set_splitk_scratch(nullptr, 0);
    Tensor tQ = f16(dQ, {rows, 1, nh, hd}), tK = f16(dK, {num_blocks, bs, nkv, hd}),
           tV = f16(dV, {num_blocks, bs, nkv, hd}), tOd = f16(dO_dense, {rows, 1, nh, hd});
    paged_attention_decode(tQ, tK, tV, tOd, dbt, dctx, bs, scale, max_ctx, 0, 0.0f, nullptr, num_blocks, 0,
                           nullptr, hd);

    // Selected path.
    const int nb_max = max_ctx / g.ratio + 1;
    float* d_scores = nullptr;
    int32_t *d_sel = nullptr, *d_cnt = nullptr, *d_sbt = nullptr, *d_sctx = nullptr;
    half *dKs = nullptr, *dVs = nullptr;
    cudaMalloc(&d_scores, static_cast<size_t>(rows) * nb_max * sizeof(float));
    cudaMalloc(&d_sel, static_cast<size_t>(rows) * cap * sizeof(int32_t));
    cudaMalloc(&d_cnt, rows * sizeof(int32_t));
    cudaMalloc(&d_sbt, static_cast<size_t>(rows) * bpr * sizeof(int32_t));
    cudaMalloc(&d_sctx, rows * sizeof(int32_t));
    const size_t scr = static_cast<size_t>(rows) * bpr * bs * nkv * hd;
    cudaMalloc(&dKs, scr * sizeof(half));
    cudaMalloc(&dVs, scr * sizeof(half));
    cudaMemset(dKs, 0x55, scr * sizeof(half));  // stale bytes: the kernel must not read past ctx
    cudaMemset(dVs, 0x55, scr * sizeof(half));
    qsa_init_scratch_bt(d_sbt, rows, bpr, nullptr);
    qsa_select(dQi, dpos, dBK, d_scores, nb_max, d_sel, d_cnt, rows, g, nullptr);
    qsa_gather_kv(dK, dV, dbt, bs, nkv, hd, d_sel, d_cnt, cap, dKs, dVs, bpr, d_sctx, rows, nullptr);
    Tensor tKs = f16(dKs, {rows * bpr, bs, nkv, hd}), tVs = f16(dVs, {rows * bpr, bs, nkv, hd}),
           tOs = f16(dO_sel, {rows, 1, nh, hd});
    paged_attention_decode(tQ, tKs, tVs, tOs, d_sbt, d_sctx, bs, scale, cap, 0, 0.0f, nullptr, bpr, 0, nullptr,
                           hd);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    auto cnt = down(d_cnt, rows);
    auto sel = down(d_sel, static_cast<size_t>(rows) * cap);
    for (int r = 0; r < rows; ++r) {
        ASSERT_EQ(cnt[r], pos[r] + 1) << "row " << r;
        for (int j = 0; j <= pos[r]; ++j)
            ASSERT_EQ(sel[static_cast<size_t>(r) * cap + j], j) << "row " << r << " slot " << j;
    }
    auto od = down(dO_dense, Q.size());
    auto os = down(dO_sel, Q.size());
    size_t mism = 0;
    for (size_t i = 0; i < od.size(); ++i)
        mism += (__half_as_ushort(od[i]) != __half_as_ushort(os[i]));
    EXPECT_EQ(mism, 0u) << "selected path differs from dense paged in " << mism << " of " << od.size() << " halfs";
}

TEST(QsaIndexer, SelectMatchesCpuTopk) {
    const QsaGeom g = geom();
    const int rows = 2;
    const int positions_h[rows] = {2051, 3001};  // 512 and 750 complete blocks
    const int nb_max = 3001 / g.ratio + 2, cap = g.budget + g.ratio - 1;
    std::mt19937 rng(11);
    auto Qi = rand_half(static_cast<size_t>(rows) * g.n_heads * D, rng);
    auto BK = rand_half(static_cast<size_t>(nb_max) * D, rng);
    // Force ties: blocks 100..109 of row 1 identical keys.
    for (int b = 101; b < 110; ++b)
        std::copy(BK.begin() + 100 * D, BK.begin() + 101 * D, BK.begin() + static_cast<size_t>(b) * D);
    std::vector<int> pos(positions_h, positions_h + rows);
    half *dQi = up(Qi), *dBK = up(BK);
    int* dpos = up(pos);
    float* d_scores = nullptr;
    int32_t *d_sel = nullptr, *d_cnt = nullptr;
    cudaMalloc(&d_scores, static_cast<size_t>(rows) * nb_max * sizeof(float));
    cudaMalloc(&d_sel, static_cast<size_t>(rows) * cap * sizeof(int32_t));
    cudaMalloc(&d_cnt, rows * sizeof(int32_t));
    qsa_select(dQi, dpos, dBK, d_scores, nb_max, d_sel, d_cnt, rows, g, nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    auto cnt = down(d_cnt, rows);
    auto sel = down(d_sel, static_cast<size_t>(rows) * cap);
    auto sc = down(d_scores, static_cast<size_t>(rows) * nb_max);
    const int topk = g.budget / g.ratio;
    for (int r = 0; r < rows; ++r) {
        const int p = pos[r], nb = (p + 1) / g.ratio;
        std::vector<float> ref(nb);
        for (int b = 0; b < nb; ++b) {
            float s = 0.0f;
            for (int h = 0; h < g.n_heads; ++h) {
                float dot = 0.0f;
                for (int d = 0; d < D; ++d)
                    dot += __half2float(Qi[(static_cast<size_t>(r) * g.n_heads + h) * D + d]) *
                           __half2float(BK[static_cast<size_t>(b) * D + d]);
                s += std::max(dot, 0.0f);
            }
            ref[b] = s / std::sqrt(static_cast<float>(D));
            EXPECT_NEAR(sc[static_cast<size_t>(r) * nb_max + b], ref[b], 1e-3f * (1.0f + std::fabs(ref[b])))
                << "row " << r << " block " << b;
        }
        // CPU top-k on the KERNEL's scores (exact float ordering), ties by lowest index.
        std::vector<int> order(nb);
        std::iota(order.begin(), order.end(), 0);
        const float* ks = sc.data() + static_cast<size_t>(r) * nb_max;
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return ks[a] > ks[b]; });
        const int k = std::min(topk, nb);
        std::vector<int> want(order.begin(), order.begin() + k);
        std::sort(want.begin(), want.end());
        ASSERT_EQ(cnt[r], k * g.ratio + (p + 1 - nb * g.ratio)) << "row " << r;
        for (int i = 0; i < k; ++i)
            for (int j = 0; j < g.ratio; ++j)
                ASSERT_EQ(sel[static_cast<size_t>(r) * cap + i * g.ratio + j], want[i] * g.ratio + j)
                    << "row " << r << " selected block " << i;
        for (int t = nb * g.ratio; t <= p; ++t)
            ASSERT_EQ(sel[static_cast<size_t>(r) * cap + k * g.ratio + (t - nb * g.ratio)], t) << "tail " << t;
    }
}

TEST(QsaIndexer, PrepQueriesMatchesCpu) {
    const QsaGeom g = geom();
    const int rows = 3;
    const int positions_h[rows] = {0, 5, 4097};
    std::mt19937 rng(3);
    auto qk = rand_half(static_cast<size_t>(rows) * (g.n_heads + 1) * D, rng);
    auto w = rand_half(D, rng, -0.5f, 0.5f);
    std::vector<int> pos(positions_h, positions_h + rows);
    half *dqk = up(qk), *dw = up(w);
    int* dpos = up(pos);
    half *dq = nullptr, *draw = nullptr;
    cudaMalloc(&dq, static_cast<size_t>(rows) * g.n_heads * D * sizeof(half));
    cudaMalloc(&draw, static_cast<size_t>(4098) * D * sizeof(half));
    qsa_prep_queries(dqk, dpos, dw, dq, draw, rows, g, nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    auto q = down(dq, static_cast<size_t>(rows) * g.n_heads * D);
    auto raw = down(draw, static_cast<size_t>(4098) * D);
    const int stride = (g.n_heads + 1) * D;
    for (int r = 0; r < rows; ++r) {
        for (int d = 0; d < D; ++d)
            EXPECT_EQ(__half_as_ushort(raw[static_cast<size_t>(pos[r]) * D + d]),
                      __half_as_ushort(qk[static_cast<size_t>(r) * stride + g.n_heads * D + d]));
        for (int h = 0; h < g.n_heads; ++h) {
            std::vector<float> x(D), y(D);
            double ss = 0.0;
            for (int d = 0; d < D; ++d) {
                x[d] = __half2float(qk[static_cast<size_t>(r) * stride + h * D + d]);
                ss += static_cast<double>(x[d]) * x[d];
            }
            const float inv = 1.0f / std::sqrt(static_cast<float>(ss / D) + g.eps);
            for (int d = 0; d < D; ++d)
                y[d] = __half2float(__float2half(x[d] * inv * (1.0f + __half2float(w[d]))));
            for (int d = 0; d < D; ++d) {
                float out = y[d];
                if (d < g.rope_dim) {
                    const int half_dim = g.rope_dim / 2, i = d < half_dim ? d : d - half_dim;
                    const float inv_freq = std::pow(g.theta, -2.0f * i / g.rope_dim);
                    const float ang = static_cast<float>(pos[r]) * inv_freq;
                    const float c = std::cos(ang), s = std::sin(ang);
                    out = d < half_dim ? y[d] * c - y[d + half_dim] * s : y[d] * c + y[d - half_dim] * s;
                }
                const float got = __half2float(q[(static_cast<size_t>(r) * g.n_heads + h) * D + d]);
                EXPECT_NEAR(got, out, 2e-2f * (1.0f + std::fabs(out))) << "row " << r << " head " << h << " d " << d;
            }
        }
    }
}

// The selected path runs with split-K enabled (executor_qsa.cu), and
// compute_splitk_splits() takes max_context_len as an input: passing the constant cap
// there instead of the real context makes the SAME bytes reduce in a different order.
// Below the budget the selection is every token in order, so the two must agree with
// the dense paged kernel bit for bit once both are told the same context.
TEST(QsaIndexer, SplitKBelowBudgetMatchesDensePaged) {
    const int nh = 24, nkv = 2, hd = 256, bs = 16, rows = 1;
    // 100 tokens = 7 context blocks: the dense path splits 7 ways, cap (2051 = 129
    // blocks) splits 29. Above ~464 tokens both saturate at 29 and the bug is invisible.
    const int pos = 100;
    const int max_ctx = pos + 1, num_blocks = (max_ctx + bs - 1) / bs;
    const QsaGeom g = geom();
    const int cap = g.budget + g.ratio - 1, bpr = (cap + bs - 1) / bs;
    std::mt19937 rng(11);
    auto K = rand_half(static_cast<size_t>(num_blocks) * bs * nkv * hd, rng);
    auto V = rand_half(K.size(), rng);
    auto Q = rand_half(static_cast<size_t>(rows) * nh * hd, rng);
    auto Qi = rand_half(static_cast<size_t>(rows) * g.n_heads * D, rng);
    auto BK = rand_half(static_cast<size_t>(max_ctx / g.ratio + 1) * D, rng);
    std::vector<int> bt(num_blocks), ctx(rows, max_ctx), p(rows, pos);
    std::iota(bt.begin(), bt.end(), 0);
    half *dK = up(K), *dV = up(V), *dQ = up(Q), *dQi = up(Qi), *dBK = up(BK);
    int *dbt = up(bt), *dctx = up(ctx), *dpos = up(p);
    half *dO_dense = nullptr, *dO_sel = nullptr;
    cudaMalloc(&dO_dense, Q.size() * sizeof(half));
    cudaMalloc(&dO_sel, Q.size() * sizeof(half));
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));

    // Split-K scratch sized as the executor's: rows x heads x splits x (2 + hd) floats.
    void* splitk = nullptr;
    const size_t splitk_sz = static_cast<size_t>(rows) * nh * 32 * (2 + hd) * sizeof(float);
    cudaMalloc(&splitk, splitk_sz);

    Tensor tQ = f16(dQ, {rows, 1, nh, hd}), tK = f16(dK, {num_blocks, bs, nkv, hd}),
           tV = f16(dV, {num_blocks, bs, nkv, hd}), tOd = f16(dO_dense, {rows, 1, nh, hd});
    paged_attention_set_splitk_scratch(splitk, splitk_sz);
    paged_attention_decode(tQ, tK, tV, tOd, dbt, dctx, bs, scale, max_ctx, 0, 0.0f, nullptr, num_blocks, 0,
                           nullptr, hd);

    const int nb_max = max_ctx / g.ratio + 1;
    float* d_scores = nullptr;
    int32_t *d_sel = nullptr, *d_cnt = nullptr, *d_sbt = nullptr, *d_sctx = nullptr;
    half *dKs = nullptr, *dVs = nullptr;
    cudaMalloc(&d_scores, static_cast<size_t>(rows) * nb_max * sizeof(float));
    cudaMalloc(&d_sel, static_cast<size_t>(rows) * cap * sizeof(int32_t));
    cudaMalloc(&d_cnt, rows * sizeof(int32_t));
    cudaMalloc(&d_sbt, static_cast<size_t>(rows) * bpr * sizeof(int32_t));
    cudaMalloc(&d_sctx, rows * sizeof(int32_t));
    const size_t scr = static_cast<size_t>(rows) * bpr * bs * nkv * hd;
    cudaMalloc(&dKs, scr * sizeof(half));
    cudaMalloc(&dVs, scr * sizeof(half));
    qsa_init_scratch_bt(d_sbt, rows, bpr, nullptr);
    qsa_select(dQi, dpos, dBK, d_scores, nb_max, d_sel, d_cnt, rows, g, nullptr);
    qsa_gather_kv(dK, dV, dbt, bs, nkv, hd, d_sel, d_cnt, cap, dKs, dVs, bpr, d_sctx, rows, nullptr);
    Tensor tKs = f16(dKs, {rows * bpr, bs, nkv, hd}), tVs = f16(dVs, {rows * bpr, bs, nkv, hd}),
           tOs = f16(dO_sel, {rows, 1, nh, hd});
    paged_attention_set_splitk_scratch(splitk, splitk_sz);
    paged_attention_decode(tQ, tKs, tVs, tOs, d_sbt, d_sctx, bs, scale, std::min(cap, max_ctx), 0, 0.0f,
                           nullptr, bpr, 0, nullptr, hd);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    // The scratch pointer is global state: leaving it set hands a foreign buffer to
    // whatever test runs next in this binary.
    paged_attention_set_splitk_scratch(nullptr, 0);
    cudaFree(splitk);

    ASSERT_EQ(down(d_cnt, rows)[0], pos + 1);
    auto od = down(dO_dense, Q.size());
    auto os = down(dO_sel, Q.size());
    size_t mism = 0;
    for (size_t i = 0; i < od.size(); ++i)
        mism += (__half_as_ushort(od[i]) != __half_as_ushort(os[i]));
    EXPECT_EQ(mism, 0u) << "split-K selected path differs from dense paged in " << mism << " of " << od.size()
                        << " halfs";
}
