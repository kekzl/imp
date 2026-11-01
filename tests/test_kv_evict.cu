// KV eviction harness kernels (kv_cache.evict_keep_ratio, src/exec/kv_evict.cu): the compacted
// block table makes the unmodified paged decode kernel equal reference attention over the kept
// tokens only; the TriAttention score and the pre-RoPE query statistics match CPU references.

#include <gtest/gtest.h>
#include "compute/attention_paged.h"
#include "core/tensor.h"
#include "exec/kv_evict.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

namespace imp {
namespace {

constexpr int kBS = 16;

template <typename T>
T* dmalloc(size_t n) {
    T* p = nullptr;
    EXPECT_EQ(cudaMalloc(&p, n * sizeof(T)), cudaSuccess);
    return p;
}
template <typename T>
void dcopy(T* dst, const std::vector<T>& src) {
    EXPECT_EQ(cudaMemcpy(dst, src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice), cudaSuccess);
}
template <typename T>
std::vector<T> dread(const T* src, size_t n) {
    std::vector<T> out(n);
    EXPECT_EQ(cudaMemcpy(out.data(), src, n * sizeof(T), cudaMemcpyDeviceToHost), cudaSuccess);
    return out;
}

float decode_ordered(unsigned e) {
    const unsigned b = (e & 0x80000000u) ? (e & 0x7FFFFFFFu) : ~e;
    float f;
    std::memcpy(&f, &b, sizeof(f));
    return f;
}

// Row 0 has holes at logical blocks 2, 3, 7 (evicted); row 1 has none. Decode over the compacted
// tables must equal softmax attention over exactly the live tokens of each row.
TEST(KvEvictCompact, PagedDecodeEqualsReferenceOverKeptTokens) {
    const int nh = 8, nkv = 2, hd = 128, group = nh / nkv;
    const int ctx = (12 * kBS) + 5;  // 13 logical blocks, partial tail
    const int n_logical = (ctx + kBS - 1) / kBS;
    const int stride = 16, pool = 32, rows = 2;
    const size_t row_elems = static_cast<size_t>(nkv) * hd;
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
    std::vector<half> k_h(static_cast<size_t>(pool) * kBS * row_elems), v_h(k_h.size());
    for (size_t i = 0; i < k_h.size(); ++i) {
        k_h[i] = __float2half(uni(rng));
        v_h[i] = __float2half(uni(rng));
    }
    // Row 0 uses physical blocks 3 + 2*b (b < 13), row 1 uses 4 + 2*b capped into the pool.
    std::vector<int> bt(static_cast<size_t>(rows) * stride, -1);
    for (int b = 0; b < n_logical; ++b) {
        bt[b] = (b == 2 || b == 3 || b == 7) ? -1 : (3 + (2 * b)) % pool;
        bt[stride + b] = (4 + (2 * b)) % pool;
    }
    std::vector<half> q_h(static_cast<size_t>(rows) * nh * hd);
    for (auto& x : q_h)
        x = __float2half(uni(rng));
    const std::vector<int> ctx_h = {ctx, ctx};

    half* d_k = dmalloc<half>(k_h.size());
    half* d_v = dmalloc<half>(v_h.size());
    half* d_q = dmalloc<half>(q_h.size());
    half* d_o = dmalloc<half>(q_h.size());
    int* d_bt = dmalloc<int>(bt.size());
    int* d_ctx = dmalloc<int>(rows);
    int* d_cbt = dmalloc<int>(bt.size());
    int* d_cctx = dmalloc<int>(rows);
    dcopy(d_k, k_h);
    dcopy(d_v, v_h);
    dcopy(d_q, q_h);
    dcopy(d_bt, bt);
    dcopy(d_ctx, ctx_h);

    kv_evict_compact_tables(d_bt, d_ctx, rows, stride, kBS, d_cbt, d_cctx, nullptr);
    int64_t qd[4] = {rows, 1, nh, hd};
    int64_t cs[4] = {pool, kBS, nkv, hd};
    Tensor Q(d_q, QType::F16, 4, qd, true);
    Tensor K(d_k, QType::F16, 4, cs, true);
    Tensor V(d_v, QType::F16, 4, cs, true);
    Tensor O(d_o, QType::F16, 4, qd, true);
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    paged_attention_decode(Q, K, V, O, d_cbt, d_cctx, kBS, scale, ctx, 0, 0.0f, nullptr, stride);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    const auto cctx = dread(d_cctx, rows);
    EXPECT_EQ(cctx[0], ctx - (3 * kBS));
    EXPECT_EQ(cctx[1], ctx);
    const auto cbt = dread(d_cbt, bt.size());
    EXPECT_EQ(cbt[2], bt[4]);  // compacted: logical 4 moves to slot 2
    const auto o = dread(d_o, q_h.size());

    double max_err = 0.0;
    for (int r = 0; r < rows; ++r) {
        for (int h = 0; h < nh; ++h) {
            const int kvh = h / group;
            std::vector<double> logit, acc(static_cast<size_t>(hd), 0.0);
            std::vector<int> toks;
            for (int t = 0; t < ctx; ++t) {
                const int phys = bt[(r * stride) + (t / kBS)];
                if (phys < 0)
                    continue;
                const size_t base = ((static_cast<size_t>(phys) * kBS + (t % kBS)) * row_elems) + (kvh * hd);
                double s = 0.0;
                for (int d = 0; d < hd; ++d)
                    s += static_cast<double>(__half2float(q_h[((r * nh) + h) * hd + d])) *
                         __half2float(k_h[base + d]);
                logit.push_back(s * scale);
                toks.push_back(static_cast<int>(base));
            }
            const double m = *std::max_element(logit.begin(), logit.end());
            double z = 0.0;
            for (size_t i = 0; i < logit.size(); ++i) {
                const double p = std::exp(logit[i] - m);
                z += p;
                for (int d = 0; d < hd; ++d)
                    acc[d] += p * __half2float(v_h[static_cast<size_t>(toks[i]) + d]);
            }
            for (int d = 0; d < hd; ++d) {
                const double ref = acc[d] / z;
                const double got = __half2float(o[((r * nh) + h) * hd + d]);
                max_err = std::max(max_err, std::abs(ref - got));
            }
        }
    }
    EXPECT_LT(max_err, 2e-3) << "paged decode over the compacted table drifts from the kept-token reference";

    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_q);
    cudaFree(d_o);
    cudaFree(d_bt);
    cudaFree(d_ctx);
    cudaFree(d_cbt);
    cudaFree(d_cctx);
}

// Score pass 0 + 1 + layer accumulate vs the CPU formula: s_h(t) = u_h . k + sum_f w_h[f] |k_f|,
// z-normalized per head over the scored range, max over every head.
TEST(KvEvictScore, MatchesCpuTriAttentionScore) {
    const int nh = 8, nkv = 2, hd = 128, pairs = hd / 2, group = nh / nkv;
    const int n_blocks = 6, ctx = n_blocks * kBS, t0 = 16, t1 = 80;
    const size_t row_elems = static_cast<size_t>(nkv) * hd;
    std::mt19937 rng(11);
    std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
    std::vector<half> k_h(static_cast<size_t>(n_blocks) * kBS * row_elems);
    for (auto& x : k_h)
        x = __float2half(uni(rng));
    std::vector<float> u(static_cast<size_t>(nh) * hd), w(static_cast<size_t>(nh) * pairs);
    for (auto& x : u)
        x = uni(rng);
    for (auto& x : w)
        x = 0.5f * (uni(rng) + 1.0f);
    const std::vector<int> bt = {5, 4, 3, 2, 1, 0};  // reversed physical order

    half* d_k = dmalloc<half>(k_h.size());
    float* d_u = dmalloc<float>(u.size());
    float* d_w = dmalloc<float>(w.size());
    int* d_bt = dmalloc<int>(bt.size());
    double* d_hs = dmalloc<double>(static_cast<size_t>(nh) * 2);
    unsigned* d_tm = dmalloc<unsigned>(ctx);
    float* d_ta = dmalloc<float>(ctx);
    dcopy(d_k, k_h);
    dcopy(d_u, u);
    dcopy(d_w, w);
    dcopy(d_bt, bt);
    cudaMemset(d_hs, 0, static_cast<size_t>(nh) * 2 * sizeof(double));
    cudaMemset(d_tm, 0, ctx * sizeof(unsigned));
    cudaMemset(d_ta, 0, ctx * sizeof(float));

    KvEvictKeyView kv;
    kv.dtype = QType::F16;
    kv.k_base = d_k;
    kv.block_stride = static_cast<int64_t>(kBS * row_elems * sizeof(half));
    kv.n_kv_heads = nkv;
    kv.head_dim = hd;
    kv.block_size = kBS;
    kv_evict_score_pass(0, kv, d_u, d_w, d_bt, t0, t1, nh, d_hs, d_tm, nullptr);
    kv_evict_score_pass(1, kv, d_u, d_w, d_bt, t0, t1, nh, d_hs, d_tm, nullptr);
    const auto tm = dread(d_tm, ctx);
    kv_evict_accumulate_layer(d_tm, d_ta, t0, t1, nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    const auto ta = dread(d_ta, ctx);

    std::vector<double> s(static_cast<size_t>(nh) * ctx, 0.0);
    for (int t = t0; t < t1; ++t) {
        const size_t base = (static_cast<size_t>(bt[t / kBS]) * kBS + (t % kBS)) * row_elems;
        for (int h = 0; h < nh; ++h) {
            const int kvh = h / group;
            double acc = 0.0;
            for (int f = 0; f < pairs; ++f) {
                const double a = __half2float(k_h[base + (kvh * hd) + f]);
                const double b = __half2float(k_h[base + (kvh * hd) + f + pairs]);
                acc += (u[(h * hd) + f] * a) + (u[(h * hd) + f + pairs] * b) +
                       (w[(h * pairs) + f] * std::sqrt((a * a) + (b * b)));
            }
            s[(static_cast<size_t>(h) * ctx) + t] = acc;
        }
    }
    const int n = t1 - t0;
    for (int t = t0; t < t1; ++t) {
        double zmax = -1e30;
        for (int h = 0; h < nh; ++h) {
            double m = 0.0, m2 = 0.0;
            for (int i = t0; i < t1; ++i) {
                m += s[(static_cast<size_t>(h) * ctx) + i];
                m2 += s[(static_cast<size_t>(h) * ctx) + i] * s[(static_cast<size_t>(h) * ctx) + i];
            }
            m /= n;
            const double sd = std::sqrt(std::max((m2 / n) - (m * m), 1e-12));
            zmax = std::max(zmax, (s[(static_cast<size_t>(h) * ctx) + t] - m) / sd);
        }
        EXPECT_NEAR(decode_ordered(tm[t]), zmax, 2e-3) << "token " << t;
        EXPECT_NEAR(ta[t], zmax, 2e-3) << "token " << t;
    }
    EXPECT_EQ(tm[0], 0u);  // outside the range: untouched
    EXPECT_EQ(ta[t1], 0.0f);

    cudaFree(d_k);
    cudaFree(d_u);
    cudaFree(d_w);
    cudaFree(d_bt);
    cudaFree(d_hs);
    cudaFree(d_tm);
    cudaFree(d_ta);
}

// Query statistics undo NeoX RoPE: sums of the de-rotated pairs and their magnitudes equal the
// CPU sums over the same F16 post-RoPE rows.
TEST(KvEvictQStats, UndoesRope) {
    const int nh = 4, hd = 64, pairs = hd / 2, n = 100;
    const float theta = 1.0e6f;
    std::mt19937 rng(3);
    std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
    std::vector<int> pos(n);
    for (int i = 0; i < n; ++i)
        pos[i] = 1000 + (37 * i);
    std::vector<half> q_h(static_cast<size_t>(n) * nh * hd);
    std::vector<double> want(static_cast<size_t>(nh) * pairs * 3, 0.0);
    for (int t = 0; t < n; ++t)
        for (int h = 0; h < nh; ++h)
            for (int f = 0; f < pairs; ++f) {
                const double a = 0.3 + uni(rng), b = -0.2 + uni(rng);  // non-zero centre
                const float freq = 1.0f / std::pow(theta, (2.0f * f) / static_cast<float>(hd));
                const double ang = static_cast<double>(pos[t]) * freq;
                const half a2 = __float2half(static_cast<float>((a * std::cos(ang)) - (b * std::sin(ang))));
                const half b2 = __float2half(static_cast<float>((a * std::sin(ang)) + (b * std::cos(ang))));
                q_h[(((static_cast<size_t>(t) * nh) + h) * hd) + f] = a2;
                q_h[(((static_cast<size_t>(t) * nh) + h) * hd) + f + pairs] = b2;
                const double ra = (__half2float(a2) * std::cos(ang)) + (__half2float(b2) * std::sin(ang));
                const double rb = (__half2float(b2) * std::cos(ang)) - (__half2float(a2) * std::sin(ang));
                double* acc = want.data() + (((h * pairs) + f) * 3);
                acc[0] += ra;
                acc[1] += rb;
                acc[2] += std::sqrt((ra * ra) + (rb * rb));
            }
    const int64_t nf = kv_evict_qstats_floats(nh, hd);
    half* d_q = dmalloc<half>(q_h.size());
    int* d_pos = dmalloc<int>(n);
    float* d_st = dmalloc<float>(static_cast<size_t>(nf));
    dcopy(d_q, q_h);
    dcopy(d_pos, pos);
    cudaMemset(d_st, 0, static_cast<size_t>(nf) * sizeof(float));
    ASSERT_TRUE(kv_evict_qstats_accumulate(d_q, d_pos, n, nh, hd, true, theta, 1.0f, d_st, nullptr));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    const auto st = dread(d_st, static_cast<size_t>(nf));
    for (size_t i = 0; i < want.size(); ++i)
        EXPECT_NEAR(st[i], want[i], 0.05) << "stat " << i;
    EXPECT_EQ(st[static_cast<size_t>(nh) * pairs * 3], static_cast<float>(n));
    cudaFree(d_q);
    cudaFree(d_pos);
    cudaFree(d_st);
}

}  // namespace
}  // namespace imp
