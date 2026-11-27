// Inkling2-Loop full-attention kernels reading K/V through a block table (B-111 S2a) against the same
// kernels on the contiguous cache, synthetic data (no checkpoint). The table is SHUFFLED: an ascending
// one (what BlockPool hands out first) would hide a kernel that ignores it.
// Hot/cold bench: tests/bench/test_inkling_attn_paged_bench.cu.

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

// Decode paths (fixed-shape rows, runtime rows, per-head chunks, row form) and the GEMM form (one sub-batch
// and several, key ranges across the 8192-key tile) on the contiguous cache vs a shuffled block table:
// bitwise equal. Context 12300 (not a multiple of 16), block sizes 16 and 32.
TEST(InklingAttnPaged, MatchesContiguousOnShuffledBlocks) {
    const int ctx = 12300, max_rows = 40;
    std::mt19937 rng(91u);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<half> kv(2 * static_cast<size_t>(ctx) * kKd);
    for (auto& v : kv)
        v = __float2half_rn(u(rng));
    std::vector<void*> owned;
    half* d_kv = upload(kv, owned);
    ASSERT_NE(d_kv, nullptr);
    half* d_out = nullptr;
    int* d_pos = nullptr;
    ASSERT_EQ(cudaMalloc(&d_out, static_cast<size_t>(max_rows) * 24 * kHd * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_pos, sizeof(int)), cudaSuccess);
    owned.push_back(d_out), owned.push_back(d_pos);
    for (const int block : {16, 32}) {
        const Paged pg = make_paged(kv, ctx, block, 7u + block);
        size_t moved = 0;
        for (size_t b = 0; b < pg.tbl.size(); ++b)
            moved += pg.tbl[b] != static_cast<int>(b);
        ASSERT_GT(moved, pg.tbl.size() / 2) << "table barely shuffled";
        half* d_pool = upload(pg.pool, owned);
        const int* d_tbl = upload(pg.tbl, owned);
        ASSERT_TRUE(d_pool && d_tbl);
        for (const int nh : {24, 16}) {
            Inputs in = make_inputs(nh, max_rows, 3u + nh);
            const size_t part_n = inkling::attention_part_floats(nh, kHd, 0, ctx);
            float* d_part = nullptr;
            ASSERT_EQ(cudaMalloc(&d_part, part_n * sizeof(float)), cudaSuccess);
            inkling::AttnArgs c = base_args(in, nh, ctx, d_out, d_pos);
            c.k = d_kv, c.v = d_kv + static_cast<size_t>(ctx) * kKd;
            inkling::AttnArgs p = c;
            p.k = d_pool, p.v = d_pool + pg.rows * kKd, p.blocks = d_tbl, p.kv_block = block;
            // Decode: 1..8 rows ending at contexts below one block, across 64-key chunks and 8192 keys.
            for (const int pos : {0, 15, 1000, 4097, 8185, 8191, 9001, ctx - 8}) {
                cudaMemcpy(d_pos, &pos, sizeof(int), cudaMemcpyHostToDevice);
                for (const int rows : {1, 3, 8}) {
                    c.T = p.T = rows;
                    for (const bool fixed : {true, false}) {
                        for (const bool part : {true, false}) {
                            c.part = p.part = part ? d_part : nullptr;
                            inkling::set_fixed_shape_kernels(fixed);
                            const std::vector<half> want = run(c, false), got = run(p, false);
                            inkling::set_fixed_shape_kernels(true);
                            EXPECT_EQ(differing(want, got), 0u)
                                << "decode block " << block << " heads " << nh << " pos " << pos << " rows "
                                << rows << (fixed ? " fixed" : " runtime")
                                << (part ? " chunked" : " row form");
                        }
                    }
                }
            }
            // GEMM form: 40 rows; work for one sub-batch and for several (same rows per sub-batch both ways).
            const size_t gb = inkling::attention_gather_bytes(kNkv, kHd, ctx);
            const size_t wb = inkling::attention_rows_bytes(nh, kNkv, kHd, kDr, 0, ctx, max_rows);
            void* d_work = nullptr;
            ASSERT_EQ(cudaMalloc(&d_work, wb), cudaSuccess);
            c.T = p.T = max_rows, c.part = p.part = nullptr, c.work = p.work = d_work;
            for (const int pos : {100, 8170, 9001, ctx - max_rows - 3}) {
                c.pos0 = p.pos0 = pos;
                cudaMemcpy(d_pos, &pos, sizeof(int), cudaMemcpyHostToDevice);
                for (const size_t bytes : {wb, gb + (wb - gb) / 5}) {
                    c.work_bytes = bytes - gb, p.work_bytes = bytes;
                    const std::vector<half> want = run(c, true), got = run(p, true);
                    EXPECT_EQ(differing(want, got), 0u)
                        << "gemm block " << block << " heads " << nh << " pos " << pos << " work " << bytes;
                    double den = 0.0;
                    for (const half h : want)
                        den += std::fabs(__half2float(h));
                    EXPECT_TRUE(std::isfinite(den) && den > 1.0) << "gemm pos " << pos;
                }
            }
            cudaFree(d_work);
            cudaFree(d_part);
            for (void* q : in.owned)
                cudaFree(q);
        }
    }
    for (void* q : owned)
        cudaFree(q);
}
