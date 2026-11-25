// Inkling2-Loop full-attention kernels reading K/V through a block table (B-111 S2a) against the same
// kernels on the contiguous cache, synthetic data (no checkpoint). The table is SHUFFLED: an ascending
// one (what BlockPool hands out first) would hide a kernel that ignores it.
// Hot/cold bench, disabled: test-compute --gtest_also_run_disabled_tests --gtest_filter='InklingAttnPaged.*'

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

namespace {

constexpr int kNkv = 8, kHd = 128, kDr = 16, kExt = 1024, kKd = kNkv * kHd;

template <typename T>
T* upload(const std::vector<T>& h, std::vector<void*>& owned) {
    T* d = nullptr;
    if (cudaMalloc(&d, h.size() * sizeof(T)) != cudaSuccess)
        return nullptr;
    cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
    owned.push_back(d);
    return d;
}

// A shuffled table of ceil(ctx / block) entries over ctx / block + 5 pool blocks, and the pool with key k
// of kv ([2][ctx, kKd]: K then V) at row tbl[k / block] * block + k % block; unused rows hold other noise.
struct Paged {
    std::vector<int> tbl;
    std::vector<half> pool;  // [2][pool rows, kKd]
    size_t rows = 0;
};

Paged make_paged(const std::vector<half>& kv, int ctx, int block, uint32_t seed) {
    Paged p;
    const int n_log = (ctx + block - 1) / block, n_phys = n_log + 5;
    std::vector<int> ids(static_cast<size_t>(n_phys));
    std::iota(ids.begin(), ids.end(), 0);
    std::mt19937 rng(seed);
    std::shuffle(ids.begin(), ids.end(), rng);
    p.tbl.assign(ids.begin(), ids.begin() + n_log);
    p.rows = static_cast<size_t>(n_phys) * block;
    p.pool.resize(2 * p.rows * kKd);
    std::uniform_real_distribution<float> noise(-3.0f, 3.0f);
    for (auto& v : p.pool)
        v = __float2half_rn(noise(rng));
    for (int side = 0; side < 2; ++side)
        for (int k = 0; k < ctx; ++k) {
            const size_t at = static_cast<size_t>(p.tbl[static_cast<size_t>(k / block)]) * block + k % block;
            std::memcpy(&p.pool[(side * p.rows + at) * kKd], &kv[(static_cast<size_t>(side) * ctx + k) * kKd],
                        kKd * sizeof(half));
        }
    return p;
}

// Inputs of one attention call: rows of q | rel | sink | gate, scales, distance table.
struct Inputs {
    std::vector<void*> owned;
    half* x = nullptr;
    float *qs = nullptr, *rs = nullptr, *proj = nullptr;
    int stride = 0;
};

Inputs make_inputs(int nh, int rows, uint32_t seed) {
    Inputs in;
    const int qd = nh * kHd, rd = nh * kDr;
    in.stride = 2 * qd + rd + nh;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(-0.5f, 0.5f);
    std::vector<half> x(static_cast<size_t>(rows) * in.stride);
    std::vector<float> qs(static_cast<size_t>(nh) * kHd), rs(static_cast<size_t>(nh) * kDr),
        proj(static_cast<size_t>(kExt) * kDr);
    for (auto& v : x)
        v = __float2half_rn(4.0f * u(rng));
    for (auto& v : qs)
        v = 1.0f + u(rng);
    for (auto& v : rs)
        v = 1.0f + u(rng);
    for (auto& v : proj)
        v = u(rng);
    in.x = upload(x, in.owned), in.qs = upload(qs, in.owned), in.rs = upload(rs, in.owned);
    in.proj = upload(proj, in.owned);
    return in;
}

inkling::AttnArgs base_args(const Inputs& in, int nh, int ctx, half* out, const int* d_pos) {
    inkling::AttnArgs a;
    const int qd = nh * kHd, rd = nh * kDr;
    a.q = in.x, a.rel = in.x + qd, a.sink = in.x + qd + rd, a.gate = in.x + qd + rd + nh;
    a.q_scale = in.qs, a.rel_scale = in.rs, a.proj = in.proj;
    a.out = out, a.d_pos = d_pos, a.max_ctx = ctx;
    a.stride = in.stride, a.n_heads = nh, a.n_kv = kNkv, a.hd = kHd, a.d_rel = kDr, a.rel_extent = kExt;
    a.scale = 1.0f / std::sqrt(static_cast<float>(kHd)), a.q_thr = 22.6f, a.rel_thr = 8.0f;
    a.log_alpha = 0.1f, a.log_floor = 1024.0f;
    return a;
}

// Output of one call; out is filled with NaN first so a path that writes nothing cannot match.
std::vector<half> run(const inkling::AttnArgs& a, bool gemm) {
    const size_t n = static_cast<size_t>(a.T) * a.n_heads * a.hd;
    cudaMemset(a.out, 0xff, n * sizeof(half));
    if (gemm)
        EXPECT_TRUE(inkling::attention_rows(a, nullptr));
    else
        inkling::attention(a, nullptr);
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::vector<half> h(n);
    cudaMemcpy(h.data(), a.out, n * sizeof(half), cudaMemcpyDeviceToHost);
    return h;
}

size_t differing(const std::vector<half>& x, const std::vector<half>& y) {
    size_t d = x.size() == y.size() ? 0 : x.size() + y.size();
    for (size_t i = 0; i < std::min(x.size(), y.size()); ++i)
        d += std::memcmp(&x[i], &y[i], sizeof(half)) != 0;
    return d;
}

}  // namespace

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
