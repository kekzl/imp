#pragma once
// Shared by tests/test_gemm_grouped_nvfp4_tile.cu and tests/bench/test_gemm_grouped_nvfp4_tile_bench.cu.
#include "compute/gemm_cutlass_sm120.h"
#include "compute/gemm_grouped_nvfp4_tile.h"
#include "core/tensor.h"
#include "nvfp4_fp64_ref.h"
#include "compute/cutlass_nvfp4_owned.h"
#include "quant/nvfp4_owned.h"
#include "quant/nvfp4_quant.h"

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
            // Distinct amplitude per expert: distinct alpha, so a wrong expert's alpha shows.
            fill_kernel<<<256, 256, 0, s>>>(w, size_t(N) * K, 0x100u + e,
                                            0.05f * static_cast<float>(1 + (e % 4)));
            int64_t ws[2] = {N, K};
            Tensor wt(w, QType::F16, 2, ws, true);
            NvFP4Owned q;
            quantize_fp16_to_nvfp4(wt, q, s);
            CutlassNvFP4Owned cw;
            convert_nvfp4_to_cutlass(q, cw, s);
            cudaMemcpyAsync(static_cast<char*>(b) + e * b_stride, cw.data, b_stride, cudaMemcpyDeviceToDevice,
                            s);
            cudaMemcpyAsync(static_cast<char*>(b_sf) + e * sfb_stride, cw.scale_factors, sfb_stride,
                            cudaMemcpyDeviceToDevice, s);
            alpha[e] = cw.tensor_scale;
            cudaStreamSynchronize(s);
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
    // Host copy of every bit the kernel reads, for the fp64 reference.
    nvfp4_ref::GroupedHost host() const {
        nvfp4_ref::GroupedHost h;
        h.ne = ne;
        h.N = N;
        h.K = K;
        h.offsets = off;
        h.sfa_off.assign(sfa_off.begin(), sfa_off.end() - 1);
        h.alpha = alpha;
        h.a.resize(size_t(M_total) * K / 2);
        h.a_sf.resize(static_cast<size_t>(sfa_off[ne]));
        cudaMemcpy(h.a.data(), a, h.a.size(), cudaMemcpyDeviceToHost);
        cudaMemcpy(h.a_sf.data(), a_sf, h.a_sf.size(), cudaMemcpyDeviceToHost);
        h.b.resize(ne);
        h.b_sf.resize(ne);
        for (int e = 0; e < ne; ++e) {
            h.b[e].resize(b_stride);
            h.b_sf[e].resize(sfb_stride);
            cudaMemcpy(h.b[e].data(), static_cast<const char*>(b) + e * b_stride, b_stride,
                       cudaMemcpyDeviceToHost);
            cudaMemcpy(h.b_sf[e].data(), static_cast<const char*>(b_sf) + e * sfb_stride, sfb_stride,
                       cudaMemcpyDeviceToHost);
        }
        return h;
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
    void SetUp() override { cudaStreamCreate(&s_); }
    void TearDown() override { cudaStreamDestroy(s_); }
    cudaStream_t s_ = nullptr;
};

}  // namespace
}  // namespace imp
