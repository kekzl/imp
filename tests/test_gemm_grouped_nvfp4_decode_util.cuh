#pragma once
// Shared by tests/test_gemm_grouped_nvfp4_decode.cu and tests/bench/test_gemm_grouped_nvfp4_decode_bench.cu.
#include "compute/nvfp4_sfatom_quant.h"
#include "compute/gemm_grouped_nvfp4_decode.h"
#include "compute/gemm_grouped_nvfp4_tile.h"
#include "compute/quantize_fp16_nvfp4_moe_native.h"
#include "core/tensor.h"
#include "model/model_config.h"
#include "nvfp4_fp64_ref.h"
#include "compute/nvfp4_sfatom_owned.h"
#include "quant/nvfp4_owned.h"
#include "quant/nvfp4_quant.h"

#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
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

// ne experts of one projection [N, K]: packed FP4 + SfAtom SF slabs, device pointer arrays, alpha.
struct Weights {
    int ne = 0, N = 0, K = 0;
    size_t stride = 0, sf_stride = 0;
    void *b = nullptr, *sf = nullptr;
    const void** d_b = nullptr;
    const void** d_sf = nullptr;
    float* d_alpha = nullptr;
    std::vector<float> alpha;

    void build(int ne_, int n, int k, uint32_t seed, float amp, cudaStream_t s) {
        ne = ne_;
        N = n;
        K = k;
        stride = size_t(N) * K / 2;
        sf_stride = nvfp4_sfatom_size(N, K);
        cudaMalloc(&b, stride * ne);
        cudaMalloc(&sf, sf_stride * ne);
        half* w = nullptr;
        cudaMalloc(&w, size_t(N) * K * sizeof(half));
        alpha.resize(ne);
        for (int e = 0; e < ne; ++e) {
            fill_kernel<<<256, 256, 0, s>>>(w, size_t(N) * K, seed + e,
                                            amp * static_cast<float>(1 + (e % 4)));
            int64_t ws[2] = {N, K};
            Tensor wt(w, QType::F16, 2, ws, true);
            NvFP4Owned q;
            quantize_fp16_to_nvfp4(wt, q, s);
            NvFP4SfAtomOwned cw;
            convert_nvfp4_to_sfatom(q, cw, s);
            cudaMemcpyAsync(static_cast<char*>(b) + e * stride, cw.data, stride, cudaMemcpyDeviceToDevice, s);
            cudaMemcpyAsync(static_cast<char*>(sf) + e * sf_stride, cw.scale_factors, sf_stride,
                            cudaMemcpyDeviceToDevice, s);
            alpha[e] = cw.tensor_scale;
            cudaStreamSynchronize(s);
        }
        cudaFree(w);
        set_ptrs(b, sf);
    }
    // Pointer arrays into (base, sf_base), e.g. a copy of the slabs at another address.
    void set_ptrs(const void* base, const void* sf_base) {
        std::vector<const void*> hb(ne), hs(ne);
        for (int e = 0; e < ne; ++e) {
            hb[e] = static_cast<const char*>(base) + e * stride;
            hs[e] = static_cast<const char*>(sf_base) + e * sf_stride;
        }
        if (d_b == nullptr) {
            cudaMalloc(&d_b, ne * sizeof(void*));
            cudaMalloc(&d_sf, ne * sizeof(void*));
            cudaMalloc(&d_alpha, ne * sizeof(float));
        }
        cudaMemcpy(d_b, hb.data(), ne * sizeof(void*), cudaMemcpyHostToDevice);
        cudaMemcpy(d_sf, hs.data(), ne * sizeof(void*), cudaMemcpyHostToDevice);
        cudaMemcpy(d_alpha, alpha.data(), ne * sizeof(float), cudaMemcpyHostToDevice);
    }
    void release() {
        for (void* p :
             {b, sf, static_cast<void*>(d_b), static_cast<void*>(d_sf), static_cast<void*>(d_alpha)})
            cudaFree(p);
    }
};

// Expert-sorted activations [M_total, K] as NVFP4 + SfAtom slabs (the gate/up input of the MoE path).
struct Acts {
    int ne = 0, K = 0, M_total = 0;
    std::vector<int> M_per, off;
    std::vector<int64_t> sfa_off;
    void *a = nullptr, *a_sf = nullptr;
    int32_t *d_M = nullptr, *d_off = nullptr;
    int64_t* d_sfa_off = nullptr;

    void build(const std::vector<int>& m_per, int k, cudaStream_t s) {
        M_per = m_per;
        ne = static_cast<int>(m_per.size());
        K = k;
        off.assign(ne + 1, 0);
        sfa_off.assign(ne + 1, 0);
        for (int e = 0; e < ne; ++e) {
            off[e + 1] = off[e] + M_per[e];
            sfa_off[e + 1] = sfa_off[e] + static_cast<int64_t>(nvfp4_sfatom_size(M_per[e], K));
        }
        M_total = off[ne];
        half* x = nullptr;
        cudaMalloc(&x, size_t(M_total) * K * sizeof(half));
        fill_kernel<<<1024, 256, 0, s>>>(x, size_t(M_total) * K, 0x77u, 1.0f);
        cudaMalloc(&a, size_t(M_total) * K / 2 + 16);
        cudaMalloc(&a_sf, sfa_off[ne] + 16);
        cudaMemsetAsync(a_sf, 0, sfa_off[ne] + 16, s);
        for (int e = 0; e < ne; ++e)
            if (M_per[e] > 0)
                quantize_fp16_to_nvfp4_sfatom(x + size_t(off[e]) * K,
                                              static_cast<char*>(a) + size_t(off[e]) * K / 2,
                                              static_cast<char*>(a_sf) + sfa_off[e], M_per[e], K, s);
        cudaMalloc(&d_M, ne * sizeof(int32_t));
        cudaMalloc(&d_off, (ne + 1) * sizeof(int32_t));
        cudaMalloc(&d_sfa_off, (ne + 1) * sizeof(int64_t));
        cudaMemcpy(d_M, M_per.data(), ne * sizeof(int32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_off, off.data(), (ne + 1) * sizeof(int32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_sfa_off, sfa_off.data(), (ne + 1) * sizeof(int64_t), cudaMemcpyHostToDevice);
        cudaStreamSynchronize(s);
        cudaFree(x);
    }
    void release() {
        for (void* p :
             {a, a_sf, static_cast<void*>(d_M), static_cast<void*>(d_off), static_cast<void*>(d_sfa_off)})
            cudaFree(p);
    }
};

GroupedNvfp4DeviceArgs tile_args(const Acts& x, const Weights& w, void* d) {
    GroupedNvfp4DeviceArgs g{};
    g.d_M_per = x.d_M;
    g.d_expert_offsets = x.d_off;
    g.d_sfa_offsets = x.d_sfa_off;
    g.d_alpha = w.d_alpha;
    g.base_A_packed = x.a;
    g.base_A_sf = x.a_sf;
    g.d_B_ptrs = w.d_b;
    g.d_SFB_ptrs = w.d_sf;
    g.base_D = d;
    return g;
}

GroupedNvfp4DecodeArgs dec_args(const Acts& x, const Weights& up, const Weights* gate, void* d, void* d_sf,
                                bool own_sfa_offsets) {
    GroupedNvfp4DecodeArgs g{};
    g.d_M_per = x.d_M;
    g.d_expert_offsets = x.d_off;
    g.d_sfa_offsets = own_sfa_offsets ? nullptr : x.d_sfa_off;
    g.base_A_packed = x.a;
    g.base_A_sf = x.a_sf;
    g.d_B_ptrs = up.d_b;
    g.d_SFB_ptrs = up.d_sf;
    g.d_alpha = up.d_alpha;
    if (gate != nullptr) {
        g.d_G_ptrs = gate->d_b;
        g.d_SFG_ptrs = gate->d_sf;
        g.d_galpha = gate->d_alpha;
    }
    g.base_D = d;
    g.base_D_sf = d_sf;
    return g;
}

// T decode tokens, each routed to top_k distinct experts (skewed popularity), as rows per expert.
std::vector<int> decode_routing(int ne, int tokens, int top_k, uint32_t seed) {
    std::mt19937 gen(seed);
    std::vector<double> wgt(ne);
    for (int e = 0; e < ne; ++e)
        wgt[e] = 1.0 / std::pow(1.0 + ((e * 37) % ne), 0.6);
    std::vector<int> m(ne, 0);
    for (int t = 0; t < tokens; ++t) {
        std::vector<double> w = wgt;
        for (int k = 0; k < top_k; ++k) {
            std::discrete_distribution<int> d(w.begin(), w.end());
            const int e = d(gen);
            ++m[e];
            w[e] = 0.0;
        }
    }
    return m;
}

class GroupedNvfp4DecodeTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaStreamCreate(&s_);
        gemm_grouped_nvfp4_decode_prewarm();
    }
    void TearDown() override { cudaStreamDestroy(s_); }
    cudaStream_t s_ = nullptr;
};

}  // namespace
}  // namespace imp
