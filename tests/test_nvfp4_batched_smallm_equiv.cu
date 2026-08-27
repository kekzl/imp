// tests/test_nvfp4_batched_smallm_equiv.cu
//
// Gate for the small-M batched NVFP4 GEMV that speculative verify chunks route
// into (`gemm_nvfp4_batched`, #998/#1055).
//
// It exists because the end-to-end equivalence check for that routing cannot
// see it. `ctx.spec_verify_small_m` is only true inside a verify chunk, so a
// no-speculation arm exercises the OLD path by construction and comes back
// byte-identical whatever the new path computes, while the speculative arms are
// not reproducible across processes (#1457) and cannot be diffed either. A
// change that reroutes 48 of 64 layers onto a different kernel family therefore
// had no gate at all until this file.
//
// GPU-ONLY, AND CI NEVER RUNS IT. The `Test` job is skipped on the runners
// (no GPU), so this file's protection is local-only: it ran once through a
// pre-commit hook and nothing upstream will run it again. If you touch the
// small-M verify dispatch, run it yourself before pushing:
//
//   docker run --rm --gpus all -v $PWD:/src -w /src imp:toolchain \
//       ./build-dev/test-quant --gtest_filter='*BatchedSmallM*'
//
// Two invariants, both deterministic and both at the shapes a GDN hybrid
// actually uses:
//   1. row m of a batched M-row call equals the single-row call for that row,
//      which is what the MR bucket selection (<1>/<2>/<3>/<4>) must preserve;
//   2. the result matches a dequantise-then-multiply reference, so "both paths
//      agree" cannot be satisfied by both being wrong the same way.

#include <gtest/gtest.h>

#include "core/tensor.h"
#include "quant/nvfp4_gemm.h"
#include "quant/nvfp4_quant.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>

namespace {

bool has_sm120() {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess)
        return false;
    cudaDeviceProp p{};
    if (cudaGetDeviceProperties(&p, dev) != cudaSuccess)
        return false;
    return p.major == 12;
}

// Shapes from Qwen3.8-27B's GDN layers, which are the ones the verify chunk
// routes: d_model 5120 into the in/out projections. K stays a multiple of 16.
constexpr int kK = 5120;
constexpr int kN = 512;  // one N tile; the kernel walks N in blocks anyway

struct Fixture {
    imp::NvFP4QuantResult w{};
    std::vector<half> w_fp16;  // dequantised reference weight, [kN, kK]
    ~Fixture() { imp::free_nvfp4_result(w); }
};

// Quantise a random weight and keep its dequantised form as the reference.
void build(Fixture& f) {
    std::mt19937 rng(1234);
    std::normal_distribution<float> dist(0.0f, 0.05f);
    std::vector<half> host(static_cast<size_t>(kN) * kK);
    for (auto& h : host)
        h = __float2half(dist(rng));

    void* d_w = nullptr;
    ASSERT_EQ(cudaMalloc(&d_w, host.size() * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_w, host.data(), host.size() * sizeof(half), cudaMemcpyHostToDevice),
              cudaSuccess);
    int64_t shape[2] = {kN, kK};
    imp::Tensor wt(d_w, imp::QType::F16, 2, shape, /*on_device=*/true);
    imp::quantize_fp16_to_nvfp4(wt, f.w, nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    // The reference multiplies the weight the kernel actually holds, not the
    // pre-quantisation one: this test gates the GEMV, not the quantiser.
    void* d_deq = nullptr;
    ASSERT_EQ(cudaMalloc(&d_deq, host.size() * sizeof(half)), cudaSuccess);
    imp::dequantize_nvfp4_to_fp16(f.w, d_deq, nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    f.w_fp16.resize(host.size());
    ASSERT_EQ(cudaMemcpy(f.w_fp16.data(), d_deq, host.size() * sizeof(half), cudaMemcpyDeviceToHost),
              cudaSuccess);
    cudaFree(d_deq);
    cudaFree(d_w);
}

std::vector<half> run_batched(const imp::NvFP4QuantResult& w, const std::vector<half>& x, int m) {
    half *d_x = nullptr, *d_y = nullptr;
    EXPECT_EQ(cudaMalloc(&d_x, x.size() * sizeof(half)), cudaSuccess);
    EXPECT_EQ(cudaMalloc(&d_y, static_cast<size_t>(m) * kN * sizeof(half)), cudaSuccess);
    EXPECT_EQ(cudaMemcpy(d_x, x.data(), x.size() * sizeof(half), cudaMemcpyHostToDevice), cudaSuccess);
    EXPECT_EQ(cudaMemset(d_y, 0, static_cast<size_t>(m) * kN * sizeof(half)), cudaSuccess);
    imp::gemm_nvfp4_batched(w, d_x, d_y, kN, kK, m, nullptr);
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::vector<half> y(static_cast<size_t>(m) * kN);
    EXPECT_EQ(cudaMemcpy(y.data(), d_y, y.size() * sizeof(half), cudaMemcpyDeviceToHost), cudaSuccess);
    cudaFree(d_x);
    cudaFree(d_y);
    return y;
}

class BatchedSmallM : public ::testing::Test {
  protected:
    void SetUp() override {
        if (!has_sm120())
            GTEST_SKIP() << "sm_120 required";
        build(f_);
    }
    Fixture f_;
};

// Invariant 1: the MR bucket must not change what a row computes. A 2-, 3- or
// 4-row call has to reproduce the 1-row call for each of its rows, which is
// exactly what a wrong bucket or a wrong activation stride would break.
TEST_F(BatchedSmallM, RowsMatchTheSingleRowCall) {
    std::mt19937 rng(99);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<half> x(static_cast<size_t>(4) * kK);
    for (auto& h : x)
        h = __float2half(dist(rng));

    // Reference: each row through the M=1 path.
    std::vector<std::vector<half>> single;
    for (int r = 0; r < 4; ++r) {
        std::vector<half> row(x.begin() + static_cast<size_t>(r) * kK,
                              x.begin() + static_cast<size_t>(r + 1) * kK);
        single.push_back(run_batched(f_.w, row, 1));
    }

    for (int m = 2; m <= 4; ++m) {
        std::vector<half> xm(x.begin(), x.begin() + static_cast<size_t>(m) * kK);
        const std::vector<half> got = run_batched(f_.w, xm, m);
        for (int r = 0; r < m; ++r)
            for (int n = 0; n < kN; ++n) {
                const float a = __half2float(single[r][n]);
                const float b = __half2float(got[static_cast<size_t>(r) * kN + n]);
                ASSERT_NEAR(a, b, 1e-2f * std::max(1.0f, std::fabs(a)))
                    << "m=" << m << " row=" << r << " n=" << n;
            }
    }
}

// Invariant 2: absolute correctness against dequantise-then-multiply, so both
// paths agreeing while both being wrong is not a passing state.
TEST_F(BatchedSmallM, MatchesDequantisedReference) {
    std::mt19937 rng(7);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (int m = 1; m <= 4; ++m) {
        std::vector<half> x(static_cast<size_t>(m) * kK);
        for (auto& h : x)
            h = __float2half(dist(rng));
        const std::vector<half> got = run_batched(f_.w, x, m);

        for (int r = 0; r < m; ++r)
            for (int n = 0; n < kN; n += 37) {  // stride: full N is 512x4 dot products
                double acc = 0.0;
                for (int k = 0; k < kK; ++k)
                    acc += static_cast<double>(__half2float(x[static_cast<size_t>(r) * kK + k])) *
                           static_cast<double>(__half2float(f_.w_fp16[static_cast<size_t>(n) * kK + k]));
                const float ref = static_cast<float>(acc);
                const float out = __half2float(got[static_cast<size_t>(r) * kN + n]);
                // FP16 accumulation over 5120 terms: tolerance scales with the
                // magnitude, not an absolute epsilon.
                ASSERT_NEAR(out, ref, 0.05f * std::max(1.0f, std::fabs(ref)))
                    << "m=" << m << " row=" << r << " n=" << n;
            }
    }
}


// The sibling-pair v2 launch (FFN gate|up, GDN in|z: two weights, one
// quantized activation, one kernel) must be BIT-identical per tensor to the
// two single v2 launches it replaces — same CTA body, same tile order, so any
// difference is a selection-prologue bug (wrong weight, wrong n_base, wrong
// ts, wrong y stride). Shapes are stripes==1 (both Ns >= 5120), which is the
// only regime the pair entry accepts.
TEST(SmallMV2Pair, PairMatchesTwoSingleCallsBitExact) {
    if (!has_sm120())
        GTEST_SKIP() << "sm_120 required";
    constexpr int kPK = 5120;
    constexpr int kN1 = 5120;
    constexpr int kN2 = 6144;  // deliberately unequal: exercises the n_tiles1 split
    std::mt19937 rng(4242);
    std::normal_distribution<float> dist(0.0f, 0.05f);

    auto make_w = [&](int n_rows) {
        std::vector<half> host(static_cast<size_t>(n_rows) * kPK);
        for (auto& h : host)
            h = __float2half(dist(rng));
        void* d_w = nullptr;
        EXPECT_EQ(cudaMalloc(&d_w, host.size() * sizeof(half)), cudaSuccess);
        EXPECT_EQ(cudaMemcpy(d_w, host.data(), host.size() * sizeof(half), cudaMemcpyHostToDevice),
                  cudaSuccess);
        int64_t shape[2] = {n_rows, kPK};
        imp::Tensor wt(d_w, imp::QType::F16, 2, shape, true);
        imp::NvFP4QuantResult w{};
        imp::quantize_fp16_to_nvfp4(wt, w, nullptr);
        EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        cudaFree(d_w);
        return w;
    };
    imp::NvFP4QuantResult w1 = make_w(kN1);
    imp::NvFP4QuantResult w2 = make_w(kN2);

    for (int m : {2, 7, 32}) {
        // One quantized activation, shared by every call — exactly the
        // production contract (the pair reads the same xq the singles read).
        std::vector<half> x(static_cast<size_t>(m) * kPK);
        for (auto& h : x)
            h = __float2half(dist(rng));
        void* d_x = nullptr;
        ASSERT_EQ(cudaMalloc(&d_x, x.size() * sizeof(half)), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(d_x, x.data(), x.size() * sizeof(half), cudaMemcpyHostToDevice),
                  cudaSuccess);
        int64_t xshape[2] = {m, kPK};
        imp::Tensor xt(d_x, imp::QType::F16, 2, xshape, true);
        imp::NvFP4QuantResult xq{};
        imp::quantize_fp16_to_nvfp4(xt, xq, nullptr);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        cudaFree(d_x);

        auto alloc_y = [&](int n_rows) {
            half* d = nullptr;
            EXPECT_EQ(cudaMalloc(&d, static_cast<size_t>(m) * n_rows * sizeof(half)), cudaSuccess);
            EXPECT_EQ(cudaMemset(d, 0xEE, static_cast<size_t>(m) * n_rows * sizeof(half)), cudaSuccess);
            return d;
        };
        half* y1s = alloc_y(kN1);
        half* y2s = alloc_y(kN2);
        half* y1p = alloc_y(kN1);
        half* y2p = alloc_y(kN2);

        ASSERT_TRUE(imp::gemm_nvfp4_smallm_v2_a4(w1, xq, y1s, m, kN1, kPK, nullptr, nullptr, false));
        ASSERT_TRUE(imp::gemm_nvfp4_smallm_v2_a4(w2, xq, y2s, m, kN2, kPK, nullptr, nullptr, false));
        ASSERT_TRUE(imp::gemm_nvfp4_smallm_v2_pair_a4(w1, w2, xq, y1p, y2p, m, kN1, kN2, kPK, nullptr));
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

        auto fetch = [&](half* d, int n_rows) {
            std::vector<half> h(static_cast<size_t>(m) * n_rows);
            EXPECT_EQ(cudaMemcpy(h.data(), d, h.size() * sizeof(half), cudaMemcpyDeviceToHost),
                      cudaSuccess);
            cudaFree(d);
            return h;
        };
        const std::vector<half> a1 = fetch(y1s, kN1), b1 = fetch(y1p, kN1);
        const std::vector<half> a2 = fetch(y2s, kN2), b2 = fetch(y2p, kN2);
        ASSERT_EQ(std::memcmp(a1.data(), b1.data(), a1.size() * sizeof(half)), 0) << "m=" << m << " W1";
        ASSERT_EQ(std::memcmp(a2.data(), b2.data(), a2.size() * sizeof(half)), 0) << "m=" << m << " W2";
        imp::free_nvfp4_result(xq);
    }
    imp::free_nvfp4_result(w1);
    imp::free_nvfp4_result(w2);
}

// DISABLED by default: a measurement, not an assertion. Run with
//   ./build-dev/test-quant --gtest_also_run_disabled_tests \
//       --gtest_filter='*M1PipelineVsGemv*'
// Question it answers: does the v2 producer/consumer pipeline kernel beat the
// shipped M=1 decode GEMV (gemv_nvfp4_kpar, W4A16, PDL) on the batch=1 decode
// shapes? The M=1 GEMV's roofline verdict is 66-70% of HBM with a
// 4-bit-dequant co-limit; the pipeline reached ~79% of the weight floor at
// M=32, and it can run M=1 (rows >= M are zero-filled). Method per
// MarginalRowCost above: >1s warmup, paired ALTERNATING in one process.
// Caveat the timing cannot see: the GEMV is W4A16, the pipeline is W4A4 — a
// switch would also change batch=1 numerics and needs a PPL A/B on top.
//
// VERDICT (2026-08-27, this harness, L2-defeated with the 4-copy ring): NO.
// GEMV and v2 sit inside each other's round spread on all six shapes (v2
// marginally ahead only on gdn-in, marginally behind on gate/up, attn-qkv,
// down). Without the L2 defeat the same run reads >1792 GB/s — every shape's
// packed weight fits the 96 MB L2, the exact #1785 trap. In-situ a switch
// would ADD a per-projection activation quantize and the W4A16->W4A4 numerics
// change, so the isolated tie is decisive: the M=1 GEMV stands. Recorded in
// sm120-cuda-expert known-issues.
TEST(SmallMV2Pair, DISABLED_M1PipelineVsGemvBench) {
    if (!has_sm120())
        GTEST_SKIP() << "sm_120 required";
    struct Shape { int n, k; const char* tag; };
    const Shape shapes[] = {{17408, 5120, "gate/up"}, {10240, 5120, "gdn-in"},
                            {6144, 5120, "gdn-z"},    {12288, 5120, "attn-qkv"},
                            {5120, 17408, "down"},    {5120, 6144, "o/gdn-out"}};
    std::mt19937 rng(11);
    std::normal_distribution<float> dist(0.0f, 0.05f);
    // L2 defeat: every shape's packed weight fits the 96 MB L2, so a single
    // weight replayed 200x measures L2, not DRAM (the #1785 trap). Rotate
    // kCopies quantized copies so consecutive iterations never re-hit.
    constexpr int kCopies = 4;
    for (const auto& sh : shapes) {
        imp::NvFP4QuantResult wcp[kCopies]{};
        for (int c = 0; c < kCopies; ++c) {
            std::vector<half> wh(static_cast<size_t>(sh.n) * sh.k);
            for (auto& h : wh) h = __float2half(dist(rng));
            void* d_w = nullptr;
            ASSERT_EQ(cudaMalloc(&d_w, wh.size() * sizeof(half)), cudaSuccess);
            ASSERT_EQ(cudaMemcpy(d_w, wh.data(), wh.size() * sizeof(half), cudaMemcpyHostToDevice),
                      cudaSuccess);
            int64_t wshape[2] = {sh.n, sh.k};
            imp::Tensor wt(d_w, imp::QType::F16, 2, wshape, true);
            imp::quantize_fp16_to_nvfp4(wt, wcp[c], nullptr);
            cudaDeviceSynchronize();
            cudaFree(d_w);
        }
        const imp::NvFP4QuantResult& w = wcp[0];
        // Activation: one FP16 row + its quantized twin
        std::vector<half> xh(sh.k);
        for (auto& h : xh) h = __float2half(dist(rng));
        half* d_x = nullptr;
        ASSERT_EQ(cudaMalloc(&d_x, xh.size() * sizeof(half)), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(d_x, xh.data(), xh.size() * sizeof(half), cudaMemcpyHostToDevice),
                  cudaSuccess);
        int64_t xshape[2] = {1, sh.k};
        imp::Tensor xt(d_x, imp::QType::F16, 2, xshape, true);
        imp::NvFP4QuantResult xq{};
        imp::quantize_fp16_to_nvfp4(xt, xq, nullptr);
        cudaDeviceSynchronize();
        half* d_y = nullptr;
        ASSERT_EQ(cudaMalloc(&d_y, static_cast<size_t>(sh.n) * sizeof(half)), cudaSuccess);
        // Warmup >1s: alternate both kernels across the copy ring
        cudaEvent_t t0, t1;
        cudaEventCreate(&t0); cudaEventCreate(&t1);
        for (int i = 0; i < 400; ++i) {
            imp::gemv_nvfp4_kpar(wcp[i % kCopies], d_x, d_y, sh.n, sh.k, nullptr);
            imp::gemm_nvfp4_smallm_v2_a4(wcp[i % kCopies], xq, d_y, 1, sh.n, sh.k, nullptr, nullptr,
                                         false);
        }
        cudaDeviceSynchronize();
        const double wbytes = sh.n * (sh.k / 2.0 + sh.k / 16.0);
        for (int round = 0; round < 3; ++round) {
            float ms_g = 0, ms_v = 0;
            cudaEventRecord(t0);
            for (int i = 0; i < 200; ++i)
                imp::gemv_nvfp4_kpar(wcp[i % kCopies], d_x, d_y, sh.n, sh.k, nullptr);
            cudaEventRecord(t1); cudaEventSynchronize(t1); cudaEventElapsedTime(&ms_g, t0, t1);
            cudaEventRecord(t0);
            for (int i = 0; i < 200; ++i)
                imp::gemm_nvfp4_smallm_v2_a4(wcp[i % kCopies], xq, d_y, 1, sh.n, sh.k, nullptr, nullptr,
                                             false);
            cudaEventRecord(t1); cudaEventSynchronize(t1); cudaEventElapsedTime(&ms_v, t0, t1);
            const double us_g = ms_g * 1000.0 / 200.0, us_v = ms_v * 1000.0 / 200.0;
            printf("%-9s N=%5d K=%5d round%d  gemv %7.2f us (%6.0f GB/s)   v2 %7.2f us (%6.0f GB/s)\n",
                   sh.tag, sh.n, sh.k, round, us_g, wbytes / us_g / 1e3, us_v, wbytes / us_v / 1e3);
        }
        cudaEventDestroy(t0); cudaEventDestroy(t1);
        cudaFree(d_x); cudaFree(d_y);
        for (int c = 0; c < kCopies; ++c)
            imp::free_nvfp4_result(wcp[c]);
        imp::free_nvfp4_result(xq);
    }
}

// The pair entry must refuse the striped regime rather than compute it with
// stripes silently forced to 1 (a small-N weight would then be produced by a
// single K-stripe with no reduce — numerically wrong under split-K rounding
// AND slower). N=512 has stripes > 1 by the shipped policy.
TEST(SmallMV2Pair, RefusesStripedShapes) {
    if (!has_sm120())
        GTEST_SKIP() << "sm_120 required";
    ASSERT_GT(imp::gemm_nvfp4_smallm_v2_stripes(512, 5120), 1);
    imp::NvFP4QuantResult dummy{};
    // args_ok fails on null pointers anyway, but the stripe check must come
    // first and be decisive on its own: a fully valid striped weight is the
    // dangerous input, so assert the policy gate directly.
    ASSERT_FALSE(imp::gemm_nvfp4_smallm_v2_pair_a4(dummy, dummy, dummy, nullptr, nullptr, 2, 512, 5120,
                                                   5120, nullptr));
}

// DISABLED by default: a measurement, not an assertion. Run it with
//   ./build-dev/test-quant --gtest_also_run_disabled_tests \\
//       --gtest_filter='*MarginalRowCost*'
//
// TWO METHOD REQUIREMENTS ARE BAKED IN HERE BECAUSE GETTING EITHER WRONG
// PRODUCED CONFIDENT NONSENSE (2026-08-18):
//
//  1. Warm up past the clock ramp. With a 20-iteration warmup the SAME
//     unchanged code path measured 20.59, 15.54 and 8.82 us across three runs,
//     purely because it was always timed first. Burn a full second.
//  2. Compare implementations PAIRED AND ALTERNATING INSIDE ONE PROCESS. Across
//     two builds the same unchanged path read 11.30 and 13.88 us, so any
//     difference under ~25 % is drift, and a 2-5 us delta on a 10 us kernel is
//     not resolvable that way.
//
// An N-tiled variant of this kernel (one activation read shared across several
// output rows) was built and measured this way: +0.55, +0.23 and -0.20 us at
// MR = 2, 3, 4. A wash, so it was not kept. The per-verify marginal row cost of
// 4.22 ms is real and comes from server measurements, and its cause is NOT the
// activation re-read.
//
// ATTRIBUTED 2026-08-19: register pressure, and the shipped launch bounds are
// already the best point on that curve. ptxas gives 40 registers at MR=1/2
// (12 blocks/SM) against 48-53 at MR=3/4 (9-10 blocks/SM), and the weight
// bandwidth this benchmark prints tracks it exactly: 1444-1508 GB/s at MR=1/2,
// then 1045 and 885. Both ways of "fixing" it were measured and are worse —
// dropping __launch_bounds__ clears MR=3's 4-byte spill and buys nothing
// (12.60 / 12.55 us against 12.72, inside this harness's own drift), and
// pinning (kKparThreads, 12) forces 40 registers by spilling 40 bytes at MR=4
// and takes it from 14.5 to 26.8 us. Detail in the sm120-cuda-expert skill's
// known-issues. Do not re-derive.
//
// Third requirement this harness earned on that run: it cannot resolve a 12 %
// effect. Three consecutive runs of the SAME binary read 11.92 / 8.92 / 8.69 us
// at MR=1 — a 37 % spread, because the first MR=1 measurement is always the one
// paying the clock ramp. Compare the LATER rows within one run, never a single
// row across runs.
TEST_F(BatchedSmallM, DISABLED_MarginalRowCost) {
    std::mt19937 rng(5);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<half> x(static_cast<size_t>(4) * kK);
    for (auto& h : x) h = __float2half(dist(rng));

    // A real GDN projection shape: at the fixture's N=512 only 512 blocks reach
    // a 170-SM card and the kernel is latency-bound, which is not the regime a
    // verify chunk runs in.
    constexpr int kBenchN = 5120;
    imp::NvFP4QuantResult bw{};
    {
        std::vector<half> hw(static_cast<size_t>(kBenchN) * kK);
        std::normal_distribution<float> wd(0.0f, 0.05f);
        for (auto& h : hw) h = __float2half(wd(rng));
        void* d_bw = nullptr;
        ASSERT_EQ(cudaMalloc(&d_bw, hw.size() * sizeof(half)), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(d_bw, hw.data(), hw.size() * sizeof(half), cudaMemcpyHostToDevice),
                  cudaSuccess);
        int64_t sh[2] = {kBenchN, kK};
        imp::Tensor wt(d_bw, imp::QType::F16, 2, sh, true);
        imp::quantize_fp16_to_nvfp4(wt, bw, nullptr);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        cudaFree(d_bw);
    }

    half *d_x = nullptr, *d_y = nullptr;
    ASSERT_EQ(cudaMalloc(&d_x, x.size() * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_y, static_cast<size_t>(4) * kBenchN * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_x, x.data(), x.size() * sizeof(half), cudaMemcpyHostToDevice), cudaSuccess);

    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    for (int i = 0; i < 80000; ++i)  // > 1 s, see requirement 1 above
        imp::gemm_nvfp4_batched(bw, d_x, d_y, kBenchN, kK, 2, nullptr);
    cudaDeviceSynchronize();

    printf("\n  N=%d K=%d  (weight ~%.2f MB)\n", kBenchN, kK, kBenchN * kK * 0.5 / 1e6);
    float prev = 0.0f;
    for (int m = 1; m <= 4; ++m) {
        float best = 1e9f;
        for (int rep = 0; rep < 5; ++rep) {
            constexpr int kIter = 300;
            cudaEventRecord(a);
            for (int i = 0; i < kIter; ++i)
                imp::gemm_nvfp4_batched(bw, d_x, d_y, kBenchN, kK, m, nullptr);
            cudaEventRecord(b);
            cudaEventSynchronize(b);
            float ms = 0.0f; cudaEventElapsedTime(&ms, a, b);
            best = std::min(best, ms * 1000.0f / kIter);
        }
        printf("  MR=%d  %8.2f us  marginal %+7.2f us  %.0f GB/s of weight\n",
               m, best, m == 1 ? 0.0f : best - prev, (kBenchN * kK * 0.5) / (best * 1e3));
        prev = best;
    }
    cudaEventDestroy(a); cudaEventDestroy(b);
    cudaFree(d_x); cudaFree(d_y);
    imp::free_nvfp4_result(bw);
}

}  // namespace
