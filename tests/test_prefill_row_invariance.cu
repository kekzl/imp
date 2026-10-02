// Row invariance of the MoE prefill kernels (#2167): a prompt row's bits must not depend on
// which other rows share its batch or chunk. Each test computes the same row inside batches of
// different size and composition and asserts identical bytes.

#include <gtest/gtest.h>
#include "compute/attention_fmha_sm120.h"
#include "compute/gemm.h"
#include "compute/gemm_cutlass_sm120.h"
#include "compute/quantize_fp16_nvfp4_moe_native.h"
#include "core/tensor.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace imp {
namespace {

bool is_sm120() {
    int dev = 0, major = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    return major >= 12;
}

void fill(std::vector<half>& v, uint32_t seed, float amp) {
    uint32_t s = seed * 2654435761u + 1u;
    for (auto& h : v) {
        s = s * 1664525u + 1013904223u;
        float u = ((s >> 8) & 0xFFFFFF) * (1.0f / 16777216.0f);
        h = __float2half((2.0f * u - 1.0f) * amp);
    }
}

template <class T>
T* to_dev(const std::vector<T>& h) {
    T* d = nullptr;
    cudaMalloc(&d, h.size() * sizeof(T));
    cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}

template <class T>
std::vector<T> to_host(const T* d, size_t n) {
    std::vector<T> h(n);
    cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost);
    return h;
}

// Router logits: rows of a 401-row prefill vs the same rows as 288+113, 33-row and 1-row batches.
TEST(PrefillRowInvariance, RouterLogitsIndependentOfBatch) {
    if (!is_sm120())
        GTEST_SKIP() << "SM120 required";
    const int K = 2816, ne = 128, N = 401;
    std::vector<half> W(static_cast<size_t>(ne) * K), X(static_cast<size_t>(N) * K);
    fill(W, 1, 0.05f);
    fill(X, 2, 3.0f);
    half* dW = to_dev(W);
    half* dX = to_dev(X);
    float* dY = nullptr;
    cudaMalloc(&dY, sizeof(float) * N * ne);
    gemm_gate_fp32_rows(dW, dX, dY, N, ne, K, nullptr);
    const auto full = to_host(dY, static_cast<size_t>(N) * ne);

    struct Part {
        int r0, n;
    };
    for (Part p : {Part{0, 288}, Part{288, 113}, Part{100, 33}, Part{7, 1}, Part{400, 1}, Part{13, 17}}) {
        cudaMemset(dY, 0xFF, sizeof(float) * N * ne);
        gemm_gate_fp32_rows(dW, dX + static_cast<size_t>(p.r0) * K, dY, p.n, ne, K, nullptr);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        const auto part = to_host(dY, static_cast<size_t>(p.n) * ne);
        EXPECT_EQ(0, std::memcmp(part.data(), full.data() + static_cast<size_t>(p.r0) * ne,
                                 part.size() * sizeof(float)))
            << "rows [" << p.r0 << ", " << p.r0 + p.n << ") differ from the 401-row batch";
    }
    // Sanity: fp64 reference within FP32 accumulation error.
    for (int r : {0, 200, 400}) {
        for (int e : {0, 77, 127}) {
            double ref = 0.0;
            for (int k = 0; k < K; ++k)
                ref += static_cast<double>(__half2float(W[static_cast<size_t>(e) * K + k])) *
                       __half2float(X[static_cast<size_t>(r) * K + k]);
            EXPECT_NEAR(full[static_cast<size_t>(r) * ne + e], ref, 1e-3 + 1e-4 * std::abs(ref));
        }
    }
    cudaFree(dW);
    cudaFree(dX);
    cudaFree(dY);
}

// Host replica of sfatom_offset (gemm_cutlass_sm120.cu): SfAtom 128 rows x 4 K-groups, 512 B.
int sfatom_off(int row, int kg, int n_k_tiles) {
    const int rl = row % 128;
    return ((row / 128) * n_k_tiles + kg / 4) * 512 + (rl % 32) * 16 + (rl / 32) * 4 + kg % 4;
}

struct MoeQuantOut {
    std::vector<uint8_t> packed;  // probe row, K/2 bytes
    std::vector<uint8_t> sf;      // probe row, K/16 bytes
};

// Quantizes `rows` (expert-sorted, offsets per expert) with the default-path MoE quantize and
// returns the probe row's packed bytes and scale bytes.
MoeQuantOut quantize_moe_probe(const std::vector<half>& rows, const std::vector<int>& offsets, int K,
                               int probe_row) {
    const int ne = static_cast<int>(offsets.size()) - 1;
    const int expanded = offsets.back();
    const int n_k_tiles = (K + 63) / 64;
    half* d_src = to_dev(rows);
    int* d_off = to_dev(offsets);
    uint8_t* d_packed = nullptr;
    cudaMalloc(&d_packed, static_cast<size_t>(expanded) * K / 2);
    std::vector<uint8_t*> bases(ne);
    std::vector<uint8_t*> owned;
    for (int e = 0; e < ne; ++e) {
        const size_t sz = cutlass_nvfp4_sf_size(offsets[e + 1] - offsets[e], K);
        uint8_t* p = nullptr;
        cudaMalloc(&p, sz > 0 ? sz : 1);
        cudaMemset(p, 0, sz > 0 ? sz : 1);
        bases[e] = p;
        owned.push_back(p);
    }
    uint8_t** d_bases = to_dev(bases);
    quantize_fp16_to_nvfp4_cutlass_moe(d_src, d_packed, d_bases, d_off, expanded, K, ne, nullptr);
    cudaDeviceSynchronize();

    MoeQuantOut out;
    out.packed.resize(static_cast<size_t>(K) / 2);
    cudaMemcpy(out.packed.data(), d_packed + static_cast<size_t>(probe_row) * K / 2, K / 2,
               cudaMemcpyDeviceToHost);
    int e = 0;
    while (offsets[e + 1] <= probe_row)
        ++e;
    const int local = probe_row - offsets[e];
    const size_t sz = cutlass_nvfp4_sf_size(offsets[e + 1] - offsets[e], K);
    const auto slab = to_host(bases[e], sz);
    for (int kg = 0; kg < K / 16; ++kg)
        out.sf.push_back(slab[static_cast<size_t>(sfatom_off(local, kg, n_k_tiles))]);
    for (auto* p : owned)
        cudaFree(p);
    cudaFree(d_bases);
    cudaFree(d_src);
    cudaFree(d_off);
    cudaFree(d_packed);
    return out;
}

// MoE activation quantize (default device-args path): the probe row lands at a different expert
// slot and row among neighbours 100x larger; its FP4 bytes and UE4M3 scales must not move.
TEST(PrefillRowInvariance, MoeActivationQuantIndependentOfBatch) {
    if (!is_sm120())
        GTEST_SKIP() << "SM120 required";
    const int K = 2816;
    std::vector<half> probe(K);
    fill(probe, 3, 2.0f);

    auto build = [&](const std::vector<int>& offsets, int probe_row, uint32_t seed, float amp) {
        std::vector<half> rows(static_cast<size_t>(offsets.back()) * K);
        fill(rows, seed, amp);
        std::copy(probe.begin(), probe.end(), rows.begin() + static_cast<size_t>(probe_row) * K);
        return quantize_moe_probe(rows, offsets, K, probe_row);
    };
    const auto a = build({0, 3, 3, 9, 12}, 5, 11, 0.5f);         // expert 2, local row 2, 12 rows
    const auto b = build({0, 200, 290, 291}, 130, 12, 200.0f);   // expert 0, local row 130, 291 rows
    const auto c = build({0, 1}, 0, 13, 1.0f);                   // alone
    EXPECT_EQ(a.packed, b.packed);
    EXPECT_EQ(a.packed, c.packed);
    EXPECT_EQ(a.sf, b.sf);
    EXPECT_EQ(a.sf, c.sf);
}

// Opt-in smallM path (moe.nvfp4_smallM): the native MoE quantize took a per-expert batch absmax as
// tensor scale, so the probe row's bytes moved with its expert mates. Now a fixed scale.
TEST(PrefillRowInvariance, SmallMNativeActivationQuantIndependentOfBatch) {
    if (!is_sm120())
        GTEST_SKIP() << "SM120 required";
    const int K = 2816;
    std::vector<half> probe(K);
    fill(probe, 4, 2.0f);
    auto run = [&](int m_e, int probe_local, uint32_t seed, float amp) {
        std::vector<half> rows(static_cast<size_t>(m_e) * K);
        fill(rows, seed, amp);
        std::copy(probe.begin(), probe.end(), rows.begin() + static_cast<size_t>(probe_local) * K);
        half* d_src = to_dev(rows);
        const std::vector<int> offsets = {0, m_e};
        int* d_off = to_dev(offsets);
        void* packed = nullptr;
        void* sf = nullptr;
        cudaMalloc(&packed, static_cast<size_t>(m_e) * K / 2);
        cudaMalloc(&sf, static_cast<size_t>(m_e) * K / 16);
        float* d_ts = nullptr;
        cudaMalloc(&d_ts, sizeof(float));
        void* hp[1] = {packed};
        void* hs[1] = {sf};
        EXPECT_TRUE(quantize_fp16_to_nvfp4_moe_native_with_scales(d_src, hp, hs, d_ts, d_off, m_e, K, 1,
                                                                  /*d_ptr_scratch=*/nullptr, nullptr));
        cudaDeviceSynchronize();
        MoeQuantOut out;
        out.packed.resize(static_cast<size_t>(K) / 2);
        out.sf.resize(static_cast<size_t>(K) / 16);
        cudaMemcpy(out.packed.data(), static_cast<uint8_t*>(packed) + static_cast<size_t>(probe_local) * K / 2,
                   K / 2, cudaMemcpyDeviceToHost);
        cudaMemcpy(out.sf.data(), static_cast<uint8_t*>(sf) + static_cast<size_t>(probe_local) * K / 16, K / 16,
                   cudaMemcpyDeviceToHost);
        float ts = 0.0f;
        cudaMemcpy(&ts, d_ts, sizeof(float), cudaMemcpyDeviceToHost);
        EXPECT_EQ(ts, 1.0f);
        cudaFree(d_src);
        cudaFree(d_off);
        cudaFree(packed);
        cudaFree(sf);
        cudaFree(d_ts);
        return out;
    };
    const auto a = run(1, 0, 31, 1.0f);
    const auto b = run(40, 17, 32, 300.0f);
    EXPECT_EQ(a.packed, b.packed);
    EXPECT_EQ(a.sf, b.sf);
}

// hd=512 prefill attention with a fixed KV order: rows of a single 150-row prefill vs the same
// rows computed as chunk continuations (q_offset > 0) against the full K/V.
TEST(PrefillRowInvariance, Hd512AttentionIndependentOfChunk) {
    if (!is_sm120())
        GTEST_SKIP() << "SM120 required";
    const int S = 150, NH = 4, NKV = 2, HD = 512;
    const float scale = 1.0f;  // Gemma-4 folds the scale into QK-norm
    std::vector<half> Q(static_cast<size_t>(S) * NH * HD), K(static_cast<size_t>(S) * NKV * HD),
        V(static_cast<size_t>(S) * NKV * HD);
    fill(Q, 21, 0.2f);
    fill(K, 22, 0.2f);
    fill(V, 23, 1.0f);
    half* dQ = to_dev(Q);
    half* dK = to_dev(K);
    half* dV = to_dev(V);
    half* dO = nullptr;
    const size_t row_elems = static_cast<size_t>(NH) * HD;
    cudaMalloc(&dO, sizeof(half) * S * row_elems);

    auto run = [&](int q0, int n, int kv_len) {
        int64_t qs[4] = {1, n, NH, HD};
        int64_t ks[4] = {1, kv_len, NKV, HD};
        Tensor q(dQ + static_cast<size_t>(q0) * row_elems, QType::F16, 4, qs, true);
        Tensor k(dK, QType::F16, 4, ks, true);
        Tensor v(dV, QType::F16, 4, ks, true);
        Tensor o(dO, QType::F16, 4, qs, true);
        cudaMemset(dO, 0, sizeof(half) * S * row_elems);
        EXPECT_TRUE(fmha_sm120_prefill(q, k, v, o, scale, /*causal=*/true, /*sliding_window=*/0,
                                       /*softcap=*/0.0f, nullptr, q0, nullptr, /*fixed_kv_order=*/true));
        EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        return to_host(dO, static_cast<size_t>(n) * row_elems);
    };
    const auto full = run(0, S, S);
    for (int c : {16, 37, 64, 100, 149}) {
        const auto head = run(0, c, c);
        EXPECT_EQ(0, std::memcmp(head.data(), full.data(), head.size() * sizeof(half)))
            << "first chunk of " << c << " rows differs";
        const auto tail = run(c, S - c, S);
        EXPECT_EQ(0, std::memcmp(tail.data(), full.data() + static_cast<size_t>(c) * row_elems,
                                 tail.size() * sizeof(half)))
            << "continuation at q_offset " << c << " differs";
    }
    cudaFree(dQ);
    cudaFree(dK);
    cudaFree(dV);
    cudaFree(dO);
}

}  // namespace
}  // namespace imp
