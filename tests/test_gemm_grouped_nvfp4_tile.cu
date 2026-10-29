// gemm_grouped_nvfp4_tile (own grouped NVFP4 MoE GEMM, ROADMAP row 123) against the CUTLASS 3.x
// grouped GEMM on the same device-args operands (bit-exact), B as a slab (mode a) and as pointer
// arrays (mode b); plus a timing table on Qwen3-30B-A3B expert shapes (informational).

#include "compute/gemm_cutlass_grouped_3x.h"
#include "compute/gemm_cutlass_sm120.h"
#include "compute/gemm_grouped_nvfp4_tile.h"
#include "core/tensor.h"
#include "quant/nvfp4_quant.h"
#include "scoped_engine_arena.h"

#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <memory>
#include <random>
#include <vector>

namespace imp {
namespace {

__global__ void fill_kernel(half* p, size_t n, uint32_t seed, float amp) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
        uint32_t h = static_cast<uint32_t>(i) * 2654435761u ^ seed;
        h ^= h >> 13;
        h *= 0x5bd1e995u;
        h ^= h >> 15;
        p[i] = __float2half(amp * ((static_cast<float>(h & 0xffff) / 32768.0f) - 1.0f));
    }
}

// Expert-sorted grouped operands as the MoE prefill builds them (contiguous A / SFA / D slabs, B and
// SFB as one slab with a per-expert stride, plus device pointer arrays into it).
struct Grouped {
    int ne = 0, N = 0, K = 0, M_total = 0;
    std::vector<int> M_per;
    std::vector<float> alpha;
    void *a = nullptr, *a_sf = nullptr, *b = nullptr, *b_sf = nullptr;
    size_t b_stride = 0, sfb_stride = 0;
    int32_t *d_M = nullptr, *d_off = nullptr;
    int64_t* d_sfa_off = nullptr;
    float* d_alpha = nullptr;
    const void** d_bptr = nullptr;
    const void** d_sfbptr = nullptr;
    std::vector<int> off;
    std::vector<int64_t> sfa_off;

    void build(const std::vector<int>& m_per, int n, int k, cudaStream_t s) {
        M_per = m_per;
        ne = static_cast<int>(m_per.size());
        N = n;
        K = k;
        off.assign(ne + 1, 0);
        sfa_off.assign(ne + 1, 0);
        for (int e = 0; e < ne; ++e) {
            off[e + 1] = off[e] + M_per[e];
            sfa_off[e + 1] = sfa_off[e] + static_cast<int64_t>(cutlass_nvfp4_sf_size(M_per[e], K));
        }
        M_total = off[ne];
        b_stride = size_t(N) * K / 2;
        sfb_stride = cutlass_nvfp4_sf_size(N, K);
        cudaMalloc(&b, b_stride * ne);
        cudaMalloc(&b_sf, sfb_stride * ne);
        half* w = nullptr;
        cudaMalloc(&w, size_t(N) * K * sizeof(half));
        alpha.resize(ne);
        for (int e = 0; e < ne; ++e) {
            fill_kernel<<<256, 256, 0, s>>>(w, size_t(N) * K, 0x100u + e, 0.1f);
            int64_t ws[2] = {N, K};
            Tensor wt(w, QType::F16, 2, ws, true);
            NvFP4QuantResult q;
            quantize_fp16_to_nvfp4(wt, q, s);
            CutlassNvFP4Weight cw;
            convert_nvfp4_to_cutlass(q, cw, s);
            cudaMemcpyAsync(static_cast<char*>(b) + e * b_stride, cw.data, b_stride, cudaMemcpyDeviceToDevice,
                            s);
            cudaMemcpyAsync(static_cast<char*>(b_sf) + e * sfb_stride, cw.scale_factors, sfb_stride,
                            cudaMemcpyDeviceToDevice, s);
            alpha[e] = cw.tensor_scale;
            cudaStreamSynchronize(s);
            free_cutlass_nvfp4_weight(cw);
            free_nvfp4_result(q);
        }
        half* x = nullptr;
        cudaMalloc(&x, size_t(M_total) * K * sizeof(half));
        fill_kernel<<<1024, 256, 0, s>>>(x, size_t(M_total) * K, 0x77u, 1.0f);
        cudaMalloc(&a, size_t(M_total) * K / 2 + 16);
        cudaMalloc(&a_sf, sfa_off[ne] + 16);
        cudaMemsetAsync(a_sf, 0, sfa_off[ne] + 16, s);
        for (int e = 0; e < ne; ++e)
            if (M_per[e] > 0)
                quantize_fp16_to_nvfp4_cutlass(x + size_t(off[e]) * K,
                                               static_cast<char*>(a) + size_t(off[e]) * K / 2,
                                               static_cast<char*>(a_sf) + sfa_off[e], M_per[e], K, s);
        cudaMalloc(&d_M, ne * sizeof(int32_t));
        cudaMalloc(&d_off, (ne + 1) * sizeof(int32_t));
        cudaMalloc(&d_sfa_off, (ne + 1) * sizeof(int64_t));
        cudaMalloc(&d_alpha, ne * sizeof(float));
        cudaMemcpy(d_M, M_per.data(), ne * sizeof(int32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_off, off.data(), (ne + 1) * sizeof(int32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_sfa_off, sfa_off.data(), (ne + 1) * sizeof(int64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_alpha, alpha.data(), ne * sizeof(float), cudaMemcpyHostToDevice);
        std::vector<const void*> hb(ne), hsfb(ne);
        for (int e = 0; e < ne; ++e) {
            hb[e] = static_cast<const char*>(b) + e * b_stride;
            hsfb[e] = static_cast<const char*>(b_sf) + e * sfb_stride;
        }
        cudaMalloc(&d_bptr, ne * sizeof(void*));
        cudaMalloc(&d_sfbptr, ne * sizeof(void*));
        cudaMemcpy(d_bptr, hb.data(), ne * sizeof(void*), cudaMemcpyHostToDevice);
        cudaMemcpy(d_sfbptr, hsfb.data(), ne * sizeof(void*), cudaMemcpyHostToDevice);
        cudaStreamSynchronize(s);
        cudaFree(w);
        cudaFree(x);
    }
    GroupedNvfp4DeviceArgs args(void* d, bool ptr_mode) const {
        GroupedNvfp4DeviceArgs g{};
        g.d_M_per = d_M;
        g.d_expert_offsets = d_off;
        g.d_sfa_offsets = d_sfa_off;
        g.d_alpha = d_alpha;
        g.base_A_packed = a;
        g.base_A_sf = a_sf;
        g.base_B_packed = b;
        g.b_expert_stride_packed = static_cast<int64_t>(b_stride);
        g.base_B_sf = b_sf;
        g.b_expert_stride_sf = static_cast<int64_t>(sfb_stride);
        g.d_B_ptrs = ptr_mode ? d_bptr : nullptr;
        g.d_SFB_ptrs = ptr_mode ? d_sfbptr : nullptr;
        g.base_D = d;
        return g;
    }
    void release() {
        for (void* p : {a, a_sf, b, b_sf, static_cast<void*>(d_M), static_cast<void*>(d_off),
                        static_cast<void*>(d_sfa_off), static_cast<void*>(d_alpha),
                        static_cast<void*>(d_bptr), static_cast<void*>(d_sfbptr)})
            cudaFree(p);
    }
};

class GroupedNvfp4TileTest : public ::testing::Test {
protected:
    void SetUp() override {
        arena_ = std::make_unique<ScopedEngineArena>();
        cudaStreamCreate(&s_);
    }
    void TearDown() override {
        cudaStreamDestroy(s_);
        gemm_grouped_3x_nvfp4_cleanup();
        arena_.reset();
    }
    std::unique_ptr<ScopedEngineArena> arena_;
    cudaStream_t s_ = nullptr;
};

}  // namespace

// Tails (7, 1 rows), several M tiles per expert (300), N tail (N=384 = 3 tiles), both B modes;
// K = 704 / 2880 (Gemma-4 / gpt-oss widths) end on a half 64-K stage.
class GroupedNvfp4TileK : public GroupedNvfp4TileTest, public ::testing::WithParamInterface<int> {};

TEST_P(GroupedNvfp4TileK, MatchesCutlassGrouped) {
    if (!cutlass_grouped_3x_nvfp4_available())
        GTEST_SKIP() << "CUTLASS 3x grouped NVFP4 disabled";
    Grouped g;
    g.build({130, 7, 300, 64, 1}, 384, GetParam(), s_);
    const size_t out_bytes = size_t(g.M_total) * g.N * sizeof(half);
    void *ref = nullptr, *own = nullptr;
    cudaMalloc(&ref, out_bytes);
    cudaMalloc(&own, out_bytes);
    ASSERT_TRUE(gemm_grouped_cutlass_3x_nvfp4_device_args(g.ne, g.N, g.K, g.args(ref, false), s_));
    std::vector<half> r(size_t(g.M_total) * g.N), o(r.size());
    ASSERT_EQ(cudaStreamSynchronize(s_), cudaSuccess);
    cudaMemcpy(r.data(), ref, out_bytes, cudaMemcpyDeviceToHost);
    for (bool ptr_mode : {false, true}) {
        cudaMemset(own, 0xff, out_bytes);
        ASSERT_TRUE(gemm_grouped_nvfp4_tile(g.ne, g.N, g.K, g.M_total, g.args(own, ptr_mode), s_));
        ASSERT_EQ(cudaStreamSynchronize(s_), cudaSuccess);
        cudaMemcpy(o.data(), own, out_bytes, cudaMemcpyDeviceToHost);
        size_t mism = 0;
        double max_ref = 0.0;
        for (size_t i = 0; i < r.size(); ++i) {
            mism += __half_as_ushort(r[i]) != __half_as_ushort(o[i]) ? 1 : 0;
            max_ref = std::max(max_ref, std::fabs((double)__half2float(r[i])));
        }
        std::fprintf(stderr, "[grouped tile] mode %c: %zu / %zu mismatches, max|ref| %.4g\n",
                     ptr_mode ? 'b' : 'a', mism, r.size(), max_ref);
        EXPECT_GT(max_ref, 0.0);
        EXPECT_EQ(mism, 0u) << "mode " << (ptr_mode ? 'b' : 'a');
    }
    cudaFree(ref);
    cudaFree(own);
    g.release();
}

INSTANTIATE_TEST_SUITE_P(K, GroupedNvfp4TileK, ::testing::Values(768, 704, 2880));

// Qwen3-30B-A3B experts: 128 x (N=768, K=2048) gate/up and (N=2048, K=768) down; rows per
// expert from a skewed draw around pp512 (avg 32) and pp4096 (avg 256) routing.
TEST_F(GroupedNvfp4TileTest, BenchVsCutlass) {
    if (!cutlass_grouped_3x_nvfp4_available())
        GTEST_SKIP() << "CUTLASS 3x grouped NVFP4 disabled";
    std::fprintf(stderr, "%6s %6s %6s %10s %10s\n", "avg_M", "N", "K", "cutlass_us", "own_us");
    for (int avg : {32, 256})
        for (const auto& nk : {std::pair<int, int>{768, 2048}, std::pair<int, int>{2048, 768}}) {
            std::mt19937 gen(7);
            std::gamma_distribution<float> dist(2.0f, avg / 2.0f);
            std::vector<int> m(128);
            for (int& v : m)
                v = std::max(1, static_cast<int>(dist(gen)));
            Grouped g;
            g.build(m, nk.first, nk.second, s_);
            void* d = nullptr;
            cudaMalloc(&d, size_t(g.M_total) * g.N * sizeof(half));
            auto time_us = [&](auto&& fn) {
                for (int i = 0; i < 10; ++i)
                    fn();
                cudaEvent_t e0, e1;
                cudaEventCreate(&e0);
                cudaEventCreate(&e1);
                cudaEventRecord(e0, s_);
                for (int i = 0; i < 20; ++i)
                    fn();
                cudaEventRecord(e1, s_);
                cudaEventSynchronize(e1);
                float ms = 0.0f;
                cudaEventElapsedTime(&ms, e0, e1);
                cudaEventDestroy(e0);
                cudaEventDestroy(e1);
                return 1000.0 * ms / 20;
            };
            const double t_ct = time_us([&] {
                (void)gemm_grouped_cutlass_3x_nvfp4_device_args(g.ne, g.N, g.K, g.args(d, true), s_);
            });
            const double t_own = time_us(
                [&] { (void)gemm_grouped_nvfp4_tile(g.ne, g.N, g.K, g.M_total, g.args(d, true), s_); });
            ASSERT_EQ(cudaStreamSynchronize(s_), cudaSuccess);
            std::fprintf(stderr, "%6d %6d %6d %10.1f %10.1f\n", avg, g.N, g.K, t_ct, t_own);
            cudaFree(d);
            g.release();
        }
}

}  // namespace imp
