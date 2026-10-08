// RA2 FP4 prefill (compute/attention_ra2.cu) against an FP32 CPU oracle: GQA, hd 64/128/256, odd n,
// chunk continuation (q_offset > 0), and the declines the executor relies on.

#include "compute/attention_apa.h"
#include "compute/attention_ra2.h"
#include "compute/apa/apa.cuh"
#include "compute/ra2/ra2.cuh"
#include "exec/workspace_sizes.h"
#include "scoped_engine_arena.h"

#include <cuda_fp16.h>
#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

namespace imp {
namespace {

struct Ra2Case {
    int n, kv_len, nh, nkv, hd;
    float apa_eps = 0.f;  // > 0: attention_apa_prefill with this eps instead of RA2
};

// Causal attention, Q row i at position q_offset + i (q_offset = kv_len - n), FP32 throughout.
std::vector<float> reference(const std::vector<half>& q, const std::vector<half>& k,
                             const std::vector<half>& v, const Ra2Case& c, float scale) {
    const int q_offset = c.kv_len - c.n, g = c.nh / c.nkv;
    std::vector<float> o((size_t)c.n * c.nh * c.hd), s(c.kv_len);
    for (int i = 0; i < c.n; ++i)
        for (int h = 0; h < c.nh; ++h) {
            const int hk = h / g, last = q_offset + i;
            const half* qr = &q[((size_t)i * c.nh + h) * c.hd];
            float mx = -INFINITY;
            for (int j = 0; j <= last; ++j) {
                const half* kr = &k[((size_t)j * c.nkv + hk) * c.hd];
                float a = 0.f;
                for (int d = 0; d < c.hd; ++d)
                    a += __half2float(qr[d]) * __half2float(kr[d]);
                s[j] = a * scale;
                mx = std::max(mx, s[j]);
            }
            float l = 0.f;
            for (int j = 0; j <= last; ++j)
                l += (s[j] = std::exp(s[j] - mx));
            float* orow = &o[((size_t)i * c.nh + h) * c.hd];
            for (int d = 0; d < c.hd; ++d) {
                float a = 0.f;
                for (int j = 0; j <= last; ++j)
                    a += s[j] * __half2float(v[((size_t)j * c.nkv + hk) * c.hd + d]);
                orow[d] = a / l;
            }
        }
    return o;
}

struct Ra2Result {
    bool accepted = false;
    double cos = 0.0, rel_l1 = 0.0;
    size_t nonfinite = 0;
};

Ra2Result run_case(const Ra2Case& c) {
    ScopedEngineArena arena(64ull << 20);
    std::mt19937 rng(1234 + c.hd + c.n);
    std::normal_distribution<float> nd(0.f, 1.f);
    const size_t nq = (size_t)c.n * c.nh * c.hd, nkv = (size_t)c.kv_len * c.nkv * c.hd;
    std::vector<half> hq(nq), hk(nkv), hv(nkv);
    for (auto& x : hq)
        x = __float2half(nd(rng));
    for (auto& x : hk)
        x = __float2half(nd(rng));
    for (auto& x : hv)
        x = __float2half(nd(rng));
    half *dq, *dk, *dv, *dout;
    EXPECT_EQ(cudaMalloc(&dq, nq * 2), cudaSuccess);
    EXPECT_EQ(cudaMalloc(&dk, nkv * 2), cudaSuccess);
    EXPECT_EQ(cudaMalloc(&dv, nkv * 2), cudaSuccess);
    EXPECT_EQ(cudaMalloc(&dout, nq * 2), cudaSuccess);
    cudaMemcpy(dq, hq.data(), nq * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dk, hk.data(), nkv * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dv, hv.data(), nkv * 2, cudaMemcpyHostToDevice);
    const int64_t qs[2] = {c.n, (int64_t)c.nh * c.hd}, ks[2] = {c.kv_len, (int64_t)c.nkv * c.hd};
    Tensor tq(dq, QType::F16, 2, qs, true), tk(dk, QType::F16, 2, ks, true), tv(dv, QType::F16, 2, ks, true);
    Tensor to(dout, QType::F16, 2, qs, true);
    const float scale = 1.f / std::sqrt((float)c.hd);

    Ra2Result r;
    r.accepted = c.apa_eps > 0.f ? attention_apa_prefill(tq, tk, tv, to, c.n, c.kv_len, c.nh, c.nkv, c.hd,
                                                         scale, c.kv_len - c.n, c.apa_eps, nullptr)
                                 : attention_ra2_prefill(tq, tk, tv, to, c.n, c.kv_len, c.nh, c.nkv, c.hd,
                                                         scale, c.kv_len - c.n, nullptr);
    if (r.accepted) {
        EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        std::vector<half> ho(nq);
        cudaMemcpy(ho.data(), dout, nq * 2, cudaMemcpyDeviceToHost);
        const std::vector<float> ref = reference(hq, hk, hv, c, scale);
        double dot = 0, na = 0, nb = 0, l1 = 0, l1r = 0;
        for (size_t i = 0; i < nq; ++i) {
            const double a = ref[i], b = __half2float(ho[i]);
            if (!std::isfinite(b))
                ++r.nonfinite;
            dot += a * b, na += a * a, nb += b * b, l1 += std::fabs(a - b), l1r += std::fabs(a);
        }
        r.cos = dot / std::sqrt(na * nb);
        r.rel_l1 = l1 / l1r;
    }
    cudaFree(dq);
    cudaFree(dk);
    cudaFree(dv);
    cudaFree(dout);
    return r;
}

// Tolerance: Q, K, V and P are all E2M1 with a UE4M3 scale per 16 values. On N(0,1) inputs the
// standalone ra2 matrix (108 cases, S <= 3000, GQA 1/4/8) measured cosine 0.9745..0.977 and
// rel. L1 0.20..0.22; 0.96 / 0.30 leave room for the smaller shapes here, not for a layout bug
// (a wrong slot permutation or swizzle drops the cosine below 0.8).
void expect_close(const Ra2Case& c) {
    const Ra2Result r = run_case(c);
    ASSERT_TRUE(r.accepted) << "hd " << c.hd << " n " << c.n << " kv " << c.kv_len;
    EXPECT_EQ(r.nonfinite, 0u);
    EXPECT_GE(r.cos, 0.96) << "hd " << c.hd;
    EXPECT_LE(r.rel_l1, 0.30) << "hd " << c.hd;
}

TEST(Ra2PrefillTest, MatchesReference_Hd64) { expect_close({320, 320, 8, 2, 64}); }
TEST(Ra2PrefillTest, MatchesReference_Hd128) { expect_close({320, 320, 8, 2, 128}); }
TEST(Ra2PrefillTest, MatchesReference_Hd256) { expect_close({320, 320, 8, 2, 256}); }

// Chunk continuation: Q rows start at position 512 of a 712-token context (FA2's q_offset contract).
TEST(Ra2PrefillTest, ChunkContinuation) { expect_close({200, 712, 8, 2, 128}); }

// The T2 plan (exec_ra2_workspace_bytes, CPU lane) must equal what the kernel carves.
TEST(Ra2PrefillTest, PlannedWorkspaceMatchesTheCarve) {
    const int shapes[][5] = {{320, 320, 8, 2, 64},
                             {200, 712, 8, 2, 128},
                             {4096, 131072, 32, 8, 128},
                             {4096, 32768, 16, 2, 256}};
    for (const auto& c : shapes) {
        const ra2::Problem p{1, c[0], c[1], c[2], c[3], c[4], c[1] - c[0], true, 1.f};
        EXPECT_EQ(exec_ra2_workspace_bytes(c[0], c[1], c[2], c[3], c[4]), ra2::workspace_bytes(p))
            << c[0] << "x" << c[1];
    }
}

TEST(Ra2PrefillTest, DeclinesUnsupported) {
    EXPECT_FALSE(run_case({128, 128, 8, 2, 96}).accepted);   // head dim
    EXPECT_FALSE(run_case({128, 128, 6, 4, 128}).accepted);  // nh % nkv != 0
}

// APA 0.2.0: K column sums in a fixed order, so reruns of the same call are bit-identical.
TEST(ApaPrefillTest, RerunsAreBitIdentical) {
    ScopedEngineArena arena(64ull << 20);
    constexpr int n = 320, kv = 1024, nh = 8, nkv = 2, hd = 128;
    const size_t nq = (size_t)n * nh * hd, nk = (size_t)kv * nkv * hd;
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<half> hq(nq), hk(nk), hv(nk), h0(nq), h1(nq);
    for (auto* v : {&hq, &hk, &hv})
        for (auto& x : *v)
            x = __float2half(nd(rng));
    half *dq, *dk, *dv, *dout;
    ASSERT_EQ(cudaMalloc(&dq, nq * 2), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dk, nk * 2), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dv, nk * 2), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dout, nq * 2), cudaSuccess);
    cudaMemcpy(dq, hq.data(), nq * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dk, hk.data(), nk * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dv, hv.data(), nk * 2, cudaMemcpyHostToDevice);
    const int64_t qs[2] = {n, (int64_t)nh * hd}, ks[2] = {kv, (int64_t)nkv * hd};
    Tensor tq(dq, QType::F16, 2, qs, true), tk(dk, QType::F16, 2, ks, true), tv(dv, QType::F16, 2, ks, true);
    Tensor to(dout, QType::F16, 2, qs, true);
    size_t diff = 0;
    for (int run = 0; run < 3; ++run) {
        ASSERT_TRUE(attention_apa_prefill(tq, tk, tv, to, n, kv, nh, nkv, hd, 1.f / std::sqrt((float)hd),
                                          kv - n, 5e-3f, nullptr));
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        cudaMemcpy(run ? h1.data() : h0.data(), dout, nq * 2, cudaMemcpyDeviceToHost);
        if (run)
            for (size_t i = 0; i < nq; ++i)
                diff += __half_as_ushort(h0[i]) != __half_as_ushort(h1[i]);
    }
    EXPECT_EQ(diff, 0u);
    for (void* p : {(void*)dq, (void*)dk, (void*)dv, (void*)dout})
        cudaFree(p);
}

// APA: eps 1e-9 sends every tile to the exact FP16 pass 2 (FP16-accumulated QK and PV, FP32 softmax),
// eps 1e30 none (pure FP4 pass 1, RA2 tolerance), 1e-2 a mix; chunk continuation as above.
TEST(ApaPrefillTest, AllHotIsExact) {
    const Ra2Result r = run_case({320, 320, 8, 2, 128, 1e-9f});
    ASSERT_TRUE(r.accepted);
    EXPECT_EQ(r.nonfinite, 0u);
    EXPECT_GE(r.cos, 0.9999);
    EXPECT_LE(r.rel_l1, 0.02);
}
TEST(ApaPrefillTest, AllColdMatchesReference) { expect_close({320, 320, 8, 2, 128, 1e30f}); }
TEST(ApaPrefillTest, MixedMatchesReference) { expect_close({320, 320, 8, 2, 128, 1e-2f}); }
TEST(ApaPrefillTest, ChunkContinuation) { expect_close({200, 712, 8, 2, 128, 1e-2f}); }
TEST(ApaPrefillTest, DeclinesUnsupported) {
    EXPECT_FALSE(run_case({128, 128, 8, 2, 64, 1e-2f}).accepted);   // hd 128 only
    EXPECT_FALSE(run_case({128, 128, 6, 4, 128, 1e-2f}).accepted);  // nh % nkv != 0
}
// Paged FP16 cache (shuffled 16-slot blocks for keys [0, 512), current chunk flat): same output as the
// flat call on the same K/V up to the float-atomic order of the K/V stats.
TEST(ApaPrefillTest, PagedMatchesFlat) {
    ScopedEngineArena arena(64ull << 20);
    constexpr int n = 200, kv = 712, nh = 8, nkv = 2, hd = 128, bs = 16, tail = kv - n, nblk = tail / bs;
    const size_t row = (size_t)nkv * hd, nq = (size_t)n * nh * hd;
    std::mt19937 rng(99);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<half> hq(nq), hk(kv * row), hv(kv * row);
    for (auto* v : {&hq, &hk, &hv})
        for (auto& x : *v)
            x = __float2half(nd(rng));
    std::vector<int> bt(nblk);
    for (int i = 0; i < nblk; ++i)
        bt[i] = (i * 7) % nblk;  // 7 and 32 coprime: a permutation
    std::vector<half> pk(tail * row), pv(tail * row);
    for (int i = 0; i < nblk; ++i)
        for (size_t e = 0; e < bs * row; ++e) {
            pk[bt[i] * bs * row + e] = hk[i * bs * row + e];
            pv[bt[i] * bs * row + e] = hv[i * bs * row + e];
        }
    half *dq, *dk, *dv, *dpk, *dpv, *o1, *o2;
    int* dbt;
    ASSERT_EQ(cudaMalloc(&dq, nq * 2), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dk, kv * row * 2), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dv, kv * row * 2), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dpk, tail * row * 2), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dpv, tail * row * 2), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&o1, nq * 2), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&o2, nq * 2), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dbt, nblk * 4), cudaSuccess);
    cudaMemcpy(dq, hq.data(), nq * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dk, hk.data(), kv * row * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dv, hv.data(), kv * row * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dpk, pk.data(), tail * row * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dpv, pv.data(), tail * row * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dbt, bt.data(), nblk * 4, cudaMemcpyHostToDevice);
    const int64_t qs[2] = {n, (int64_t)nh * hd}, ks[2] = {kv, (int64_t)row}, ts[2] = {n, (int64_t)row};
    Tensor tq(dq, QType::F16, 2, qs, true), tk(dk, QType::F16, 2, ks, true), tv(dv, QType::F16, 2, ks, true);
    Tensor t1(o1, QType::F16, 2, qs, true), t2(o2, QType::F16, 2, qs, true);
    Tensor kt(dk + tail * row, QType::F16, 2, ts, true), vt(dv + tail * row, QType::F16, 2, ts, true);
    const float scale = 1.f / std::sqrt((float)hd);
    ASSERT_TRUE(attention_apa_prefill(tq, tk, tv, t1, n, kv, nh, nkv, hd, scale, tail, 1e-2f, nullptr));
    ASSERT_TRUE(attention_apa_prefill_paged(tq, dpk, dpv, dbt, bs, kt, vt, tail, t2, n, kv, nh, nkv, hd,
                                            scale, tail, 1e-2f, /*kv_layer=*/-1, /*owner=*/-1, nullptr));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::vector<half> h1(nq), h2(nq);
    cudaMemcpy(h1.data(), o1, nq * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(h2.data(), o2, nq * 2, cudaMemcpyDeviceToHost);
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < nq; ++i) {
        const double a = __half2float(h1[i]), b = __half2float(h2[i]);
        dot += a * b, na += a * a, nb += b * b;
    }
    EXPECT_GE(dot / std::sqrt(na * nb), 0.9999);

    // Tile cache of layer 0: chunk [312, 512) quantizes tiles 0..7, chunk [512, 712) only 8..11 (stats frozen
    // at 512 keys). Same output as the flat call up to those stats.
    attention_apa_set_kv_states(1, nkv, 1024);
    const int64_t ts0[2] = {0, (int64_t)row};
    Tensor kt0(dk, QType::F16, 2, ts0, true), vt0(dv, QType::F16, 2, ts0, true);
    ASSERT_TRUE(attention_apa_prefill_paged(tq, dpk, dpv, dbt, bs, kt0, vt0, tail, t2, n, tail, nh, nkv, hd,
                                            scale, tail - n, 1e-2f, /*kv_layer=*/0, /*owner=*/1, nullptr));
    ASSERT_TRUE(attention_apa_prefill_paged(tq, dpk, dpv, dbt, bs, kt, vt, tail, t2, n, kv, nh, nkv, hd,
                                            scale, tail, 1e-2f, /*kv_layer=*/0, /*owner=*/1, nullptr));
    auto cos_vs_flat = [&] {
        EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        cudaMemcpy(h2.data(), o2, nq * 2, cudaMemcpyDeviceToHost);
        double d = 0, a2 = 0, b2 = 0;
        for (size_t i = 0; i < nq; ++i) {
            const double a = __half2float(h1[i]), b = __half2float(h2[i]);
            d += a * b, a2 += a * a, b2 += b * b;
        }
        return d / std::sqrt(a2 * b2);
    };
    EXPECT_GE(cos_vs_flat(), 0.999) << "cached tiles";

    // Another request (owner 2) at the same length on other keys: all keys but the fingerprint's (0, len/2,
    // len-1) negated. Its tiles must not serve owner 3. eps 0.5: tiles cold, so stale tiles would show.
    ASSERT_TRUE(attention_apa_prefill(tq, tk, tv, t1, n, kv, nh, nkv, hd, scale, tail, 0.5f, nullptr));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    cudaMemcpy(h1.data(), o1, nq * 2, cudaMemcpyDeviceToHost);
    std::vector<half> pk2(pk), pv2(pv);
    for (int i = 0; i < nblk; ++i)
        for (int s = 0; s < bs; ++s)
            if (const int key = i * bs + s; key != 0 && key != tail / 2 && key != tail - 1)
                for (size_t e = 0; e < row; ++e) {
                    const size_t at = (bt[i] * bs + s) * row + e;
                    pk2[at] = __float2half(-__half2float(pk[at]));
                    pv2[at] = __float2half(-__half2float(pv[at]));
                }
    cudaMemcpy(dpk, pk2.data(), tail * row * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dpv, pv2.data(), tail * row * 2, cudaMemcpyHostToDevice);
    ASSERT_TRUE(attention_apa_prefill_paged(tq, dpk, dpv, dbt, bs, kt0, vt0, tail, t2, n, tail, nh, nkv, hd,
                                            scale, tail - n, 0.5f, /*kv_layer=*/0, /*owner=*/2, nullptr));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    cudaMemcpy(dpk, pk.data(), tail * row * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dpv, pv.data(), tail * row * 2, cudaMemcpyHostToDevice);
    ASSERT_TRUE(attention_apa_prefill_paged(tq, dpk, dpv, dbt, bs, kt, vt, tail, t2, n, kv, nh, nkv, hd,
                                            scale, tail, 0.5f, /*kv_layer=*/0, /*owner=*/3, nullptr));
    EXPECT_GE(cos_vs_flat(), 0.999) << "owner change restarts the cache";
    attention_apa_set_kv_states(0, 0, 0);
    for (void* p :
         {(void*)dq, (void*)dk, (void*)dv, (void*)dpk, (void*)dpv, (void*)o1, (void*)o2, (void*)dbt})
        cudaFree(p);
}

TEST(ApaPrefillTest, PlannedWorkspaceMatchesTheCarve) {
    const int shapes[][4] = {{320, 320, 8, 2}, {200, 712, 8, 2}, {4096, 131072, 32, 8}, {4096, 32768, 24, 8}};
    for (const auto& c : shapes) {
        const apa::Problem p{1, c[0], c[1], c[2], c[3], 128, c[1] - c[0], true, 1.f};
        EXPECT_EQ(exec_apa_workspace_bytes(c[0], c[1], c[2], c[3], 128), apa::workspace_bytes(p))
            << c[0] << "x" << c[1];
    }
    for (int cap : {712, 32768, 131072})
        EXPECT_EQ(exec_apa_kv_state_bytes(8, 128, cap), apa::kv_state_bytes(1, 8, 128, cap)) << cap;
}

}  // namespace
}  // namespace imp
