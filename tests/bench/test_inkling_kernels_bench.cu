// Timing cases of tests/test_inkling_kernels.cu, built into test-bench (make test-bench), not into the
// default GPU run.
#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "compute/gemm.h"
#include "compute/gemm_grouped_nvfp4_f16_tile.h"
#include "compute/gemv_i8b32.h"
#include "core/tensor.h"
#include "compute/inkling_ops.h"
#include "quant/nvfp4_gemm.h"
#include "quant/nvfp4_quant.h"

using namespace imp;

#include "test_inkling_kernels_util.cuh"

// Bench (run with --gtest_also_run_disabled_tests): the MXFP8 slot GEMV against sort + grouped GEMM on
// decode-sized calls, experts rotating over 680 MB so every launch reads cold weights.
TEST(InklingPrefill, DISABLED_Mxfp8SlotVsGroupedBench) {
    const int NE = 128, N = 2304, K = 2304, RUNS = 200;
    int32_t* d_grp = group_scratch();
    const size_t wn = static_cast<size_t>(NE) * N * K, sn = wn / 32;
    uint8_t *d_w = nullptr, *d_sc = nullptr;
    ASSERT_EQ(cudaMalloc(&d_w, wn), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_sc, sn), cudaSuccess);
    cudaMemset(d_w, 0x3A, wn);
    cudaMemset(d_sc, 118, sn);
    int32_t *d_bias = nullptr, *d_idx = nullptr, *d_perm = nullptr, *d_off = nullptr;
    float* d_ts = nullptr;
    half *d_x = nullptr, *d_xs = nullptr, *d_y = nullptr;
    const int MAXS = 64;
    ASSERT_EQ(cudaMalloc(&d_bias, NE * 4), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_ts, NE * 4), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_idx, static_cast<size_t>(RUNS) * MAXS * 4), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_perm, MAXS * 4), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_off, (NE + 1) * 4), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_x, static_cast<size_t>(MAXS) * K * 2), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_xs, static_cast<size_t>(MAXS) * K * 2), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_y, static_cast<size_t>(MAXS) * N * 2), cudaSuccess);
    cudaMemset(d_x, 0, static_cast<size_t>(MAXS) * K * 2);
    inkling::mxfp8_expert_bias(d_sc, NE, static_cast<size_t>(N) * (K / 32), d_bias, d_ts, nullptr);
    struct Case {
        const char* name;
        int slots, distinct;
    };
    for (const Case& cs : {Case{"8 slots, 8 experts", 8, 8}, Case{"16 slots, 12 experts", 16, 12},
                           Case{"32 slots, 19 experts", 32, 19}, Case{"64 slots, 29 experts", 64, 29},
                           Case{"64 slots, 64 experts", 64, 64}}) {
        std::vector<int32_t> idx(static_cast<size_t>(RUNS) * MAXS);
        for (int r = 0; r < RUNS; ++r)
            for (int j = 0; j < cs.slots; ++j)
                idx[static_cast<size_t>(r) * MAXS + j] = (r * 37 + (j % cs.distinct) * 3) % NE;
        cudaMemcpy(d_idx, idx.data(), idx.size() * 4, cudaMemcpyHostToDevice);
        for (int form = 0; form < 3; ++form) {
            double best = 1e30;
            for (int round = 0; round < 5; ++round) {
                cudaDeviceSynchronize();
                const auto t0 = std::chrono::steady_clock::now();
                for (int r = 0; r < RUNS; ++r) {
                    const int32_t* ix = d_idx + static_cast<size_t>(r) * MAXS;
                    if (form == 0) {
                        inkling::gemv_mxfp8_slots(d_w, d_sc, ix, d_x, d_y, cs.slots, N, K, 1, nullptr);
                    } else if (form == 2) {
                        ASSERT_TRUE(inkling::moe_group(ix, cs.slots, d_grp, nullptr));
                        inkling::MoeSmallArgs m;
                        m.x = d_x, m.y = d_y, m.groups = d_grp, m.w = d_w,
                        m.w_stride = static_cast<size_t>(N) * K;
                        m.sc = d_sc, m.sc_stride = static_cast<size_t>(N) * (K / 32), m.ts = d_ts,
                        m.bias = d_bias;
                        m.slots = cs.slots, m.N = N, m.K = K, m.fp8 = true;
                        ASSERT_TRUE(inkling::moe_small(m, nullptr));
                    } else {
                        ASSERT_TRUE(inkling::moe_sort(ix, cs.slots, NE, d_perm, d_off, nullptr));
                        inkling::moe_rows_copy(d_x, d_perm, d_xs, cs.slots, K, 1, true, nullptr);
                        GroupedNvfp4F16Args g;
                        g.a = d_xs, g.rows = cs.slots, g.N = N, g.K = K, g.ne = NE, g.d_offsets = d_off;
                        g.w = d_w, g.w_stride = static_cast<size_t>(N) * K, g.sc = d_sc;
                        g.sc_stride = static_cast<size_t>(N) * (K / 32), g.d_tensor_scales = d_ts, g.d = d_y;
                        ASSERT_TRUE(gemm_grouped_mxfp8_f16_tile(g, d_bias, nullptr));
                    }
                }
                cudaDeviceSynchronize();
                const double us =
                    std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() /
                    RUNS;
                best = std::min(best, us);
            }
            const double mb = static_cast<double>(cs.distinct) * N * K * (1.0 + 1.0 / 32) / 1e6;
            std::printf("mxfp8 %s, %s: %.1f us per call (%.0f MB of weights, %.0f GB/s)\n", cs.name,
                        form == 0   ? "slot GEMV"
                        : form == 1 ? "sort + grouped GEMM"
                                    : "small tiles",
                        best, mb, mb / best * 1e3);
        }
    }
    for (void* p : {static_cast<void*>(d_w), static_cast<void*>(d_sc), static_cast<void*>(d_bias),
                    static_cast<void*>(d_ts), static_cast<void*>(d_idx), static_cast<void*>(d_perm),
                    static_cast<void*>(d_off), static_cast<void*>(d_x), static_cast<void*>(d_xs),
                    static_cast<void*>(d_y)})
        cudaFree(p);
}

// The same for the NVFP4 experts: gemv_nvfp4_moe_decode against sort + grouped GEMM.
TEST(InklingPrefill, DISABLED_Nvfp4SlotVsGroupedBench) {
    const int NE = 128, RUNS = 200, MAXS = 64;
    int32_t* d_grp = group_scratch();
    struct Shape {
        const char* name;
        int N, K;
    };
    for (const Shape& sh : {Shape{"w13 2304 x 2304", 2304, 2304}, Shape{"w2 2304 x 1152", 2304, 1152}}) {
        const int N = sh.N, K = sh.K;
        const size_t wn = static_cast<size_t>(NE) * N * (K / 2), sn = static_cast<size_t>(NE) * N * (K / 16);
        uint8_t *d_w = nullptr, *d_sc = nullptr;
        ASSERT_EQ(cudaMalloc(&d_w, wn), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_sc, sn), cudaSuccess);
        cudaMemset(d_w, 0x21, wn);
        cudaMemset(d_sc, 0x38, sn);
        int32_t *d_idx = nullptr, *d_perm = nullptr, *d_off = nullptr;
        float* d_ts = nullptr;
        half *d_x = nullptr, *d_xs = nullptr, *d_y = nullptr;
        ASSERT_EQ(cudaMalloc(&d_ts, NE * 4), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_idx, static_cast<size_t>(RUNS) * MAXS * 4), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_perm, MAXS * 4), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_off, (NE + 1) * 4), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_x, static_cast<size_t>(MAXS) * K * 2), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_xs, static_cast<size_t>(MAXS) * K * 2), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_y, static_cast<size_t>(MAXS) * N * 2), cudaSuccess);
        cudaMemset(d_x, 0, static_cast<size_t>(MAXS) * K * 2);
        std::vector<float> ts(NE, 0.01f);
        cudaMemcpy(d_ts, ts.data(), NE * 4, cudaMemcpyHostToDevice);
        NvFP4MoEQuantResult q;
        q.packed_data = d_w, q.micro_scales = d_sc, q.tensor_scales = d_ts, q.n_experts = NE, q.N = N,
        q.K = K;
        q.expert_stride_packed = static_cast<size_t>(N) * (K / 2);
        q.expert_stride_ms = static_cast<size_t>(N) * (K / 16);
        q.borrowed = true;
        struct Case {
            const char* name;
            int slots, distinct;
        };
        for (const Case& cs : {Case{"8 slots, 8 experts", 8, 8}, Case{"16 slots, 12 experts", 16, 12},
                               Case{"32 slots, 19 experts", 32, 19}, Case{"64 slots, 29 experts", 64, 29},
                               Case{"64 slots, 64 experts", 64, 64}}) {
            std::vector<int32_t> idx(static_cast<size_t>(RUNS) * MAXS);
            for (int r = 0; r < RUNS; ++r)
                for (int j = 0; j < cs.slots; ++j)
                    idx[static_cast<size_t>(r) * MAXS + j] = (r * 37 + (j % cs.distinct) * 3) % NE;
            cudaMemcpy(d_idx, idx.data(), idx.size() * 4, cudaMemcpyHostToDevice);
            for (int form = 0; form < 3; ++form) {
                double best = 1e30;
                for (int round = 0; round < 5; ++round) {
                    cudaDeviceSynchronize();
                    const auto t0 = std::chrono::steady_clock::now();
                    for (int r = 0; r < RUNS; ++r) {
                        const int32_t* ix = d_idx + static_cast<size_t>(r) * MAXS;
                        if (form == 0) {
                            gemv_nvfp4_moe_decode(q, ix, d_x, d_y, N, K, K, cs.slots, nullptr, 1);
                        } else if (form == 2) {
                            ASSERT_TRUE(inkling::moe_group(ix, cs.slots, d_grp, nullptr));
                            inkling::MoeSmallArgs m;
                            m.x = d_x, m.y = d_y, m.groups = d_grp, m.w = d_w,
                            m.w_stride = q.expert_stride_packed;
                            m.sc = d_sc, m.sc_stride = q.expert_stride_ms, m.ts = d_ts;
                            m.slots = cs.slots, m.N = N, m.K = K;
                            ASSERT_TRUE(inkling::moe_small(m, nullptr));
                        } else {
                            ASSERT_TRUE(inkling::moe_sort(ix, cs.slots, NE, d_perm, d_off, nullptr));
                            inkling::moe_rows_copy(d_x, d_perm, d_xs, cs.slots, K, 1, true, nullptr);
                            GroupedNvfp4F16Args g;
                            g.a = d_xs, g.rows = cs.slots, g.N = N, g.K = K, g.ne = NE, g.d_offsets = d_off;
                            g.w = d_w, g.w_stride = q.expert_stride_packed, g.sc = d_sc;
                            g.sc_stride = q.expert_stride_ms, g.d_tensor_scales = d_ts, g.d = d_y;
                            ASSERT_TRUE(gemm_grouped_nvfp4_f16_tile(g, nullptr));
                        }
                    }
                    cudaDeviceSynchronize();
                    best = std::min(best, std::chrono::duration<double, std::micro>(
                                              std::chrono::steady_clock::now() - t0)
                                                  .count() /
                                              RUNS);
                }
                const double mb = static_cast<double>(cs.distinct) * N * K * (0.5 + 1.0 / 16) / 1e6;
                std::printf("nvfp4 %s, %s, %s: %.1f us per call (%.0f MB of weights, %.0f GB/s)\n", sh.name,
                            cs.name,
                            form == 0   ? "slot GEMV"
                            : form == 1 ? "sort + grouped GEMM"
                                        : "small tiles",
                            best, mb, mb / best * 1e3);
            }
        }
        for (void* p : {static_cast<void*>(d_w), static_cast<void*>(d_sc), static_cast<void*>(d_ts),
                        static_cast<void*>(d_idx), static_cast<void*>(d_perm), static_cast<void*>(d_off),
                        static_cast<void*>(d_x), static_cast<void*>(d_xs), static_cast<void*>(d_y)})
            cudaFree(p);
    }
}
