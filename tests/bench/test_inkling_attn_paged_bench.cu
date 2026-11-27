// Timing cases of tests/test_inkling_attn_paged.cu, built into test-bench (make test-bench), not into the
// default GPU run.
#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

#include "compute/inkling_ops.h"

using namespace imp;

#include "test_inkling_attn_paged_util.cuh"

// Hot/cold timing (skill kernel-tuning, section 3) of one decode attention call (1 row, 24 heads) at
// contexts 1k..80k: contiguous, identity table, shuffled table. Cold rotates over > 400 MB of K/V.
TEST(InklingAttnPaged, DISABLED_DecodeHotColdBench) {
    constexpr int kIters = 400, kRounds = 7, nh = 24;
    Inputs in = make_inputs(nh, 1, 5u);
    half* d_out = nullptr;
    int* d_pos = nullptr;
    ASSERT_EQ(cudaMalloc(&d_out, static_cast<size_t>(nh) * kHd * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_pos, sizeof(int)), cudaSuccess);
    for (const int ctx : {1024, 8192, 32768, 81920}) {
        const size_t rows = static_cast<size_t>(ctx), kv_bytes = 2 * rows * kKd * sizeof(half);
        const int copies = static_cast<int>((static_cast<size_t>(400) << 20) / kv_bytes) + 2;
        std::vector<half> kv(2 * rows * kKd);
        std::mt19937 rng(11u);
        std::uniform_real_distribution<float> u(-1.0f, 1.0f);
        for (auto& v : kv)
            v = __float2half_rn(u(rng));
        const Paged pg = make_paged(kv, ctx, 16, 13u);
        std::vector<int> ident(pg.tbl.size());
        std::iota(ident.begin(), ident.end(), 0);
        std::vector<void*> owned;
        const int* d_ident = upload(ident, owned);
        const int* d_tbl = upload(pg.tbl, owned);
        std::vector<half*> contig(static_cast<size_t>(copies)), pool(static_cast<size_t>(copies));
        for (int i = 0; i < copies; ++i) {
            contig[static_cast<size_t>(i)] = upload(kv, owned);
            pool[static_cast<size_t>(i)] = upload(pg.pool, owned);
            ASSERT_TRUE(contig[static_cast<size_t>(i)] && pool[static_cast<size_t>(i)]) << "ctx " << ctx;
        }
        const size_t part_n = inkling::attention_part_floats(nh, kHd, 0, ctx);
        float* d_part = nullptr;
        ASSERT_EQ(cudaMalloc(&d_part, part_n * sizeof(float)), cudaSuccess);
        owned.push_back(d_part);
        const int pos = ctx - 1;
        cudaMemcpy(d_pos, &pos, sizeof(int), cudaMemcpyHostToDevice);
        inkling::AttnArgs base = base_args(in, nh, ctx, d_out, d_pos);
        base.T = 1, base.part = d_part;
        struct Arm {
            const char* tag;
            bool paged;
            const int* tbl;
        };
        const Arm arms[] = {{"contiguous", false, nullptr},
                            {"identity", false, d_ident},
                            {"shuffled", true, d_tbl}};
        const auto args_for = [&](const Arm& arm, int copy) {
            inkling::AttnArgs a = base;
            half* b = arm.paged ? pool[static_cast<size_t>(copy)] : contig[static_cast<size_t>(copy)];
            const size_t r = arm.paged ? pg.rows : rows;
            a.k = b, a.v = b + r * kKd, a.blocks = arm.tbl;
            return a;
        };
        const std::vector<half> ref = run(args_for(arms[0], 0), false);
        for (const Arm& arm : arms)
            EXPECT_EQ(differing(ref, run(args_for(arm, 0), false)), 0u) << arm.tag << " ctx " << ctx;
        cudaEvent_t t0, t1;
        cudaEventCreate(&t0);
        cudaEventCreate(&t1);
        for (int i = 0; i < 4000; ++i)  // clocks up
            inkling::attention(args_for(arms[0], i % copies), nullptr);
        cudaDeviceSynchronize();
        for (const Arm& arm : arms) {
            std::vector<double> hot, cold;
            for (int round = 0; round < kRounds; ++round) {
                float ms_h = 0.0f, ms_c = 0.0f;
                const inkling::AttnArgs a0 = args_for(arm, 0);
                cudaEventRecord(t0);
                for (int i = 0; i < kIters; ++i)
                    inkling::attention(a0, nullptr);
                cudaEventRecord(t1);
                cudaEventSynchronize(t1);
                cudaEventElapsedTime(&ms_h, t0, t1);
                std::vector<inkling::AttnArgs> rot;
                for (int c = 0; c < copies; ++c)
                    rot.push_back(args_for(arm, c));
                cudaEventRecord(t0);
                for (int i = 0; i < kIters; ++i)
                    inkling::attention(rot[static_cast<size_t>(i % copies)], nullptr);
                cudaEventRecord(t1);
                cudaEventSynchronize(t1);
                cudaEventElapsedTime(&ms_c, t0, t1);
                hot.push_back(ms_h * 1000.0 / kIters), cold.push_back(ms_c * 1000.0 / kIters);
            }
            std::sort(hot.begin(), hot.end());
            std::sort(cold.begin(), cold.end());
            std::printf(
                "inkling decode attn ctx %6d %-10s hot min/med/max %7.2f %7.2f %7.2f us  cold %7.2f %7.2f "
                "%7.2f us (%5.0f GB/s K/V)\n",
                ctx, arm.tag, hot[0], hot[kRounds / 2], hot[kRounds - 1], cold[0], cold[kRounds / 2],
                cold[kRounds - 1], static_cast<double>(kv_bytes) / cold[kRounds / 2] / 1e3);
        }
        cudaEventDestroy(t0);
        cudaEventDestroy(t1);
        for (void* q : owned)
            cudaFree(q);
    }
    cudaFree(d_out);
    cudaFree(d_pos);
    for (void* q : in.owned)
        cudaFree(q);
}
