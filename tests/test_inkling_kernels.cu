// Inkling2-Loop kernels against their reference forms on synthetic data (no checkpoint): the grouped
// MXFP8 expert GEMM against the slot GEMV, the few-row and many-row attention against the row kernel.

#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "compute/gemm_grouped_nvfp4_f16_tile.h"
#include "compute/inkling_ops.h"
#include "quant/nvfp4_gemm.h"
#include "quant/nvfp4_quant.h"

using namespace imp;

namespace {
int32_t* group_scratch() {
    static int32_t* p = nullptr;
    if (!p)
        cudaMalloc(&p, inkling::kMoeSmallGroupInts * sizeof(int32_t));
    return p;
}
}  // namespace

// The grouped MXFP8 GEMM against the slot GEMV on random experts whose exponents span 20 binades.
TEST(InklingPrefill, GroupedMxfp8MatchesTheSlotGemv) {
    const int NE = 5, N = 72, K = 192, ROWS = 150;
    uint32_t seed = 777u;
    const auto rnd = [&]() {
        seed = seed * 1664525u + 1013904223u;
        return seed >> 8;
    };
    std::vector<uint8_t> w(static_cast<size_t>(NE) * N * K), sc(static_cast<size_t>(NE) * N * (K / 32));
    for (auto& v : w) {
        v = static_cast<uint8_t>(rnd());
        if ((v & 0x7F) == 0x7F)
            v ^= 1;  // no NaN codes
    }
    for (auto& v : sc)
        v = static_cast<uint8_t>(105 + rnd() % 20);
    std::vector<int32_t> idx(ROWS), off(NE + 1, 0);
    for (int r = 0; r < ROWS; ++r)
        idx[r] = (r * NE) / ROWS == 3 ? 4 : (r * NE) / ROWS;  // sorted, expert 3 without rows
    for (int r = 0; r < ROWS; ++r)
        ++off[idx[r] + 1];
    for (int e = 0; e < NE; ++e)
        off[e + 1] += off[e];
    std::vector<half> x(static_cast<size_t>(ROWS) * K);
    for (auto& v : x)
        v = __float2half_rn(static_cast<float>(rnd()) / 16777216.0f - 0.5f);
    uint8_t *d_w = nullptr, *d_sc = nullptr;
    int32_t *d_idx = nullptr, *d_off = nullptr, *d_bias = nullptr;
    float* d_ts = nullptr;
    half *d_x = nullptr, *d_y = nullptr;
    const size_t yn = static_cast<size_t>(ROWS) * N;
    ASSERT_EQ(cudaMalloc(&d_w, w.size()), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_sc, sc.size()), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_idx, idx.size() * 4), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_off, off.size() * 4), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_bias, NE * 4), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_ts, NE * 4), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_x, x.size() * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_y, yn * sizeof(half)), cudaSuccess);
    cudaMemcpy(d_w, w.data(), w.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sc, sc.data(), sc.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_idx, idx.data(), idx.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_off, off.data(), off.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x.data(), x.size() * sizeof(half), cudaMemcpyHostToDevice);
    std::vector<half> want(yn), got(yn);
    inkling::gemv_mxfp8_slots(d_w, d_sc, d_idx, d_x, d_y, ROWS, N, K, 1, nullptr);
    cudaMemcpy(want.data(), d_y, yn * sizeof(half), cudaMemcpyDeviceToHost);
    cudaMemset(d_y, 0, yn * sizeof(half));
    inkling::mxfp8_expert_bias(d_sc, NE, static_cast<size_t>(N) * (K / 32), d_bias, d_ts, nullptr);
    GroupedNvfp4F16Args g;
    g.a = d_x, g.rows = ROWS, g.N = N, g.K = K, g.ne = NE, g.d_offsets = d_off;
    g.w = d_w, g.w_stride = static_cast<size_t>(N) * K, g.sc = d_sc,
    g.sc_stride = static_cast<size_t>(N) * (K / 32);
    g.d_tensor_scales = d_ts, g.d = d_y;
    ASSERT_TRUE(gemm_grouped_mxfp8_f16_tile(g, d_bias, nullptr));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    cudaMemcpy(got.data(), d_y, yn * sizeof(half), cudaMemcpyDeviceToHost);
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < yn; ++i) {
        const double a = __half2float(want[i]), b = __half2float(got[i]);
        ASSERT_TRUE(std::isfinite(b)) << i;
        num += (a - b) * (a - b);
        den += a * a;
    }
    std::printf("grouped MXFP8: %d rows of %d experts, rel. error %.3g against the slot GEMV\n", ROWS, NE,
                std::sqrt(num / den));
    EXPECT_GT(den, 0.0);
    EXPECT_LT(std::sqrt(num / den), 1e-3);
    for (void* p : {static_cast<void*>(d_w), static_cast<void*>(d_sc), static_cast<void*>(d_idx),
                    static_cast<void*>(d_off), static_cast<void*>(d_bias), static_cast<void*>(d_ts),
                    static_cast<void*>(d_x), static_cast<void*>(d_y)})
        cudaFree(p);
}

// The small-tile expert GEMM of 2..8-row calls against the slot GEMVs, NVFP4 and MXFP8: random weights,
// 40 slots over 9 experts (one expert 8 times, two once), input rows shared by groups of 8 slots or not.
TEST(InklingDecode, SmallTileExpertsMatchTheSlotGemvs) {
    const int NE = 12, N = 128, K = 192, SLOTS = 40;
    uint32_t seed = 4242u;
    const auto rnd = [&]() {
        seed = seed * 1664525u + 1013904223u;
        return seed >> 8;
    };
    int32_t* d_grp = group_scratch();
    std::vector<int32_t> idx(SLOTS);
    for (int i = 0; i < SLOTS; ++i)
        idx[i] = i < 8 ? 7 : (i == 8 ? 0 : (i == 9 ? 11 : 1 + (i * 5) % 6));
    for (const int x_group : {1, 8}) {
        const int xrows = SLOTS / x_group;
        std::vector<half> x(static_cast<size_t>(xrows) * K);
        for (auto& v : x)
            v = __float2half_rn(static_cast<float>(rnd()) / 16777216.0f - 0.5f);
        for (const bool fp8 : {false, true}) {
            const size_t wrow = fp8 ? K : K / 2, srow = fp8 ? K / 32 : K / 16;
            std::vector<uint8_t> w(static_cast<size_t>(NE) * N * wrow),
                sc(static_cast<size_t>(NE) * N * srow);
            for (auto& v : w) {
                v = static_cast<uint8_t>(rnd());
                if (fp8 && (v & 0x7F) == 0x7F)
                    v ^= 1;
            }
            for (auto& v : sc)
                v = fp8 ? static_cast<uint8_t>(110 + rnd() % 12) : static_cast<uint8_t>(0x30 + rnd() % 0x20);
            std::vector<float> ts(NE);
            for (auto& v : ts)
                v = 0.001f + static_cast<float>(rnd() % 100) * 1e-5f;
            uint8_t *d_w = nullptr, *d_sc = nullptr;
            int32_t *d_idx = nullptr, *d_bias = nullptr;
            float* d_ts = nullptr;
            half *d_x = nullptr, *d_y = nullptr;
            const size_t yn = static_cast<size_t>(SLOTS) * N;
            ASSERT_EQ(cudaMalloc(&d_w, w.size()), cudaSuccess);
            ASSERT_EQ(cudaMalloc(&d_sc, sc.size()), cudaSuccess);
            ASSERT_EQ(cudaMalloc(&d_idx, SLOTS * 4), cudaSuccess);
            ASSERT_EQ(cudaMalloc(&d_bias, NE * 4), cudaSuccess);
            ASSERT_EQ(cudaMalloc(&d_ts, NE * 4), cudaSuccess);
            ASSERT_EQ(cudaMalloc(&d_x, x.size() * sizeof(half)), cudaSuccess);
            ASSERT_EQ(cudaMalloc(&d_y, yn * sizeof(half)), cudaSuccess);
            cudaMemcpy(d_w, w.data(), w.size(), cudaMemcpyHostToDevice);
            cudaMemcpy(d_sc, sc.data(), sc.size(), cudaMemcpyHostToDevice);
            cudaMemcpy(d_idx, idx.data(), SLOTS * 4, cudaMemcpyHostToDevice);
            cudaMemcpy(d_ts, ts.data(), NE * 4, cudaMemcpyHostToDevice);
            cudaMemcpy(d_x, x.data(), x.size() * sizeof(half), cudaMemcpyHostToDevice);
            std::vector<half> want(yn), got(yn);
            if (fp8) {
                inkling::gemv_mxfp8_slots(d_w, d_sc, d_idx, d_x, d_y, SLOTS, N, K, x_group, nullptr);
                inkling::mxfp8_expert_bias(d_sc, NE, static_cast<size_t>(N) * srow, d_bias, d_ts, nullptr);
            } else {
                NvFP4MoEQuantResult q;
                q.packed_data = d_w, q.micro_scales = d_sc, q.tensor_scales = d_ts, q.n_experts = NE;
                q.N = N, q.K = K, q.expert_stride_packed = static_cast<size_t>(N) * wrow;
                q.expert_stride_ms = static_cast<size_t>(N) * srow, q.borrowed = true;
                gemv_nvfp4_moe_decode(q, d_idx, d_x, d_y, N, K, K, SLOTS, nullptr, x_group);
            }
            cudaMemcpy(want.data(), d_y, yn * sizeof(half), cudaMemcpyDeviceToHost);
            cudaMemset(d_y, 0, yn * sizeof(half));
            ASSERT_TRUE(inkling::moe_group(d_idx, SLOTS, d_grp, nullptr));
            inkling::MoeSmallArgs m;
            m.x = d_x, m.y = d_y, m.groups = d_grp, m.w = d_w, m.w_stride = static_cast<size_t>(N) * wrow;
            m.sc = d_sc, m.sc_stride = static_cast<size_t>(N) * srow, m.ts = d_ts,
            m.bias = fp8 ? d_bias : nullptr;
            m.slots = SLOTS, m.N = N, m.K = K, m.x_group = x_group, m.fp8 = fp8;
            ASSERT_TRUE(inkling::moe_small(m, nullptr));
            ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
            cudaMemcpy(got.data(), d_y, yn * sizeof(half), cudaMemcpyDeviceToHost);
            double num = 0.0, den = 0.0;
            for (size_t i = 0; i < yn; ++i) {
                const double a = __half2float(want[i]), b = __half2float(got[i]);
                ASSERT_TRUE(std::isfinite(b)) << i;
                num += (a - b) * (a - b);
                den += a * a;
            }
            std::printf("small tiles %s, input rows per %d slots: rel. error %.3g against the slot GEMV\n",
                        fp8 ? "MXFP8" : "NVFP4", x_group, std::sqrt(num / den));
            EXPECT_GT(den, 0.0);
            EXPECT_LT(std::sqrt(num / den), 2e-3);
            for (void* p : {static_cast<void*>(d_w), static_cast<void*>(d_sc), static_cast<void*>(d_idx),
                            static_cast<void*>(d_bias), static_cast<void*>(d_ts), static_cast<void*>(d_x),
                            static_cast<void*>(d_y)})
                cudaFree(p);
        }
    }
}

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

TEST(InklingAttention, FewRowsFormMatchesTheRowForm) {
    using inkling::AttnArgs;
    const int NKV = 8, HD = 128, DR = 16, EXT = 1024, T = 96;
    struct Case {
        const char* name;
        int heads, window, ring, ctx, pos0;
        float alpha;
    };
    const Case cases[] = {{"sliding", 24, 1024, 2048, 4096, 3001, 0.0f},
                          {"full", 24, 0, 0, 4096, 2500, 0.1f},
                          {"start", 24, 1024, 2048, 4096, 5, 0.0f},
                          {"sliding across the ring end", 24, 1024, 2048, 4096, 2040, 0.0f},
                          {"full start", 24, 0, 0, 4096, 0, 0.1f},
                          {"two heads per KV head", 16, 1024, 2048, 4096, 3001, 0.0f}};
    uint32_t seed = 12345u;
    const auto rnd = [&]() {
        seed = seed * 1664525u + 1013904223u;
        return static_cast<float>(seed >> 8) / 16777216.0f - 0.5f;
    };
    for (const Case& cs : cases) {
        const int NH = cs.heads, QD = NH * HD, RD = NH * DR, stride = 2 * QD + RD + NH;
        ASSERT_EQ(stride % 8, 0);
        const size_t cache_rows = static_cast<size_t>(cs.ring > 0 ? cs.ring : cs.ctx);
        std::vector<half> x(static_cast<size_t>(T) * stride), kv(2 * cache_rows * NKV * HD);
        std::vector<float> qs(static_cast<size_t>(NH) * HD), rs(static_cast<size_t>(NH) * DR),
            proj(static_cast<size_t>(EXT) * DR);
        for (auto& v : x)
            v = __float2half_rn(4.0f * rnd());
        for (auto& v : kv)
            v = __float2half_rn(2.0f * rnd());
        for (auto& v : qs)
            v = 1.0f + rnd();
        for (auto& v : rs)
            v = 1.0f + rnd();
        for (auto& v : proj)
            v = rnd();
        const size_t part_n = inkling::attention_part_floats(NH, HD, cs.window, cs.ctx);
        half *d_x = nullptr, *d_kv = nullptr, *d_out = nullptr;
        float *d_qs = nullptr, *d_rs = nullptr, *d_proj = nullptr, *d_part = nullptr;
        int* d_pos = nullptr;
        ASSERT_EQ(cudaMalloc(&d_x, x.size() * sizeof(half)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_kv, kv.size() * sizeof(half)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_out, static_cast<size_t>(T) * QD * sizeof(half)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_qs, qs.size() * sizeof(float)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_rs, rs.size() * sizeof(float)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_proj, proj.size() * sizeof(float)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_part, part_n * sizeof(float)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_pos, sizeof(int)), cudaSuccess);
        cudaMemcpy(d_x, x.data(), x.size() * sizeof(half), cudaMemcpyHostToDevice);
        cudaMemcpy(d_kv, kv.data(), kv.size() * sizeof(half), cudaMemcpyHostToDevice);
        cudaMemcpy(d_qs, qs.data(), qs.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_rs, rs.data(), rs.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_proj, proj.data(), proj.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_pos, &cs.pos0, sizeof(int), cudaMemcpyHostToDevice);
        AttnArgs a;
        a.q = d_x, a.rel = d_x + QD, a.sink = d_x + QD + RD, a.gate = d_x + QD + RD + NH;
        a.q_scale = d_qs, a.rel_scale = d_rs, a.proj = d_proj;
        a.k = d_kv, a.v = d_kv + cache_rows * NKV * HD, a.out = d_out, a.d_pos = d_pos;
        a.max_ctx = cs.ctx;
        a.stride = stride, a.n_heads = NH, a.n_kv = NKV, a.hd = HD, a.d_rel = DR, a.rel_extent = EXT;
        a.window = cs.window, a.ring = cs.ring;
        a.scale = 1.0f / std::sqrt(static_cast<float>(HD)), a.q_thr = 22.6f, a.rel_thr = 8.0f;
        a.log_alpha = cs.alpha, a.log_floor = 1024.0f;
        for (int rows : {1, 2, 3, 5, 8}) {
            const size_t n = static_cast<size_t>(rows) * QD;
            std::vector<half> want(n), got(n);
            a.T = rows;
            a.part = nullptr;
            inkling::attention(a, nullptr);
            cudaMemcpy(want.data(), d_out, n * sizeof(half), cudaMemcpyDeviceToHost);
            cudaMemset(d_out, 0, n * sizeof(half));
            a.part = d_part;
            inkling::attention(a, nullptr);
            ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
            cudaMemcpy(got.data(), d_out, n * sizeof(half), cudaMemcpyDeviceToHost);
            double num = 0.0, den = 0.0;
            for (size_t i = 0; i < n; ++i) {
                const double w = __half2float(want[i]), gv = __half2float(got[i]);
                num += (gv - w) * (gv - w);
                den += w * w;
            }
            EXPECT_GT(den, 1.0) << cs.name << " rows " << rows;
            EXPECT_LT(std::sqrt(num / den), 1e-4) << cs.name << " rows " << rows;
            std::printf("attention %s, %d rows: rel. error %.3g against the row form\n", cs.name, rows,
                        std::sqrt(num / den));
        }
        // Many rows: the GEMM form against the row form, as one sub-batch and as several.
        const size_t wb = inkling::attention_rows_bytes(NH, HD, DR, cs.window, cs.ctx, T);
        void* d_work = nullptr;
        ASSERT_EQ(cudaMalloc(&d_work, wb), cudaSuccess);
        for (const size_t bytes : {wb, wb / 7}) {
            const size_t n = static_cast<size_t>(T) * QD;
            std::vector<half> want(n), got(n);
            a.T = T, a.part = nullptr, a.work = nullptr;
            inkling::attention(a, nullptr);
            cudaMemcpy(want.data(), d_out, n * sizeof(half), cudaMemcpyDeviceToHost);
            cudaMemset(d_out, 0, n * sizeof(half));
            a.work = d_work, a.work_bytes = bytes, a.pos0 = cs.pos0;
            ASSERT_TRUE(inkling::attention_rows(a, nullptr)) << cs.name;
            ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
            cudaMemcpy(got.data(), d_out, n * sizeof(half), cudaMemcpyDeviceToHost);
            double num = 0.0, den = 0.0;
            for (size_t i = 0; i < n; ++i) {
                const double w = __half2float(want[i]), gv = __half2float(got[i]);
                num += (gv - w) * (gv - w);
                den += w * w;
            }
            EXPECT_GT(den, 1.0) << cs.name;
            EXPECT_LT(std::sqrt(num / den), 2e-3) << cs.name << " scratch " << bytes;
            std::printf("attention %s, %d rows, %zu B scratch: rel. error %.3g against the row form\n",
                        cs.name, T, bytes, std::sqrt(num / den));
        }
        cudaFree(d_work);
        for (void* p : {static_cast<void*>(d_x), static_cast<void*>(d_kv), static_cast<void*>(d_out),
                        static_cast<void*>(d_qs), static_cast<void*>(d_rs), static_cast<void*>(d_proj),
                        static_cast<void*>(d_part), static_cast<void*>(d_pos)})
            cudaFree(p);
    }
}
