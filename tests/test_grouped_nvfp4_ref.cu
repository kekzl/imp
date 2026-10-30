// TEST_AUDIT(retired) P1.2/R1.2 (#576): own grouped NVFP4 MoE GEMM (gemm_grouped_nvfp4_host_args) vs an
// INDEPENDENT fp64 CPU reference decoding the exact NVFP4 bits the GPU consumes (nvfp4_fp64_ref.h), so
// quantization error cancels and only accumulation + f16 store differ.
// Tolerance: f16 class, rel < 1e-2 (abs floor 1e-3), an order under the NVFP4 1e-1 floor, no NaN/Inf.
// Cases: empty / 1-token / 128-row-crossing experts, K = 64..256 (K % 128 == 64 half stages), and 512
// experts so every CTA walks several experts (tensormap slot rotation).

#include <gtest/gtest.h>
#include "compute/gemm_cutlass_sm120.h"
#include "compute/gemm_grouped_nvfp4_tile.h"
#include "core/tensor.h"
#include "nvfp4_fp64_ref.h"
#include "quant/nvfp4_quant.h"
#include "scoped_engine_arena.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <vector>
#include <cmath>
#include <random>

namespace imp {
namespace {

struct SyntheticExpert {
    NvFP4QuantResult nvfp4{};
    CutlassNvFP4Weight cutlass_w{};
};

static void make_expert(SyntheticExpert& e, int N, int K, float wscale, uint64_t seed, cudaStream_t stream) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-wscale, wscale);
    std::vector<half> w(static_cast<size_t>(N) * K);
    for (auto& v : w)
        v = __float2half(dist(gen));
    void* d_w_fp16 = nullptr;
    cudaMalloc(&d_w_fp16, w.size() * sizeof(half));
    cudaMemcpy(d_w_fp16, w.data(), w.size() * sizeof(half), cudaMemcpyHostToDevice);
    int64_t w_shape[2] = {N, K};
    Tensor w_input(d_w_fp16, QType::F16, 2, w_shape, true);
    quantize_fp16_to_nvfp4(w_input, e.nvfp4, stream);
    convert_nvfp4_to_cutlass(e.nvfp4, e.cutlass_w, stream);
    cudaStreamSynchronize(stream);
    cudaFree(d_w_fp16);
}

static void free_expert(SyntheticExpert& e) {
    free_cutlass_nvfp4_weight(e.cutlass_w);
    free_nvfp4_result(e.nvfp4);
}

class GroupedNvfp4RefTest : public ::testing::Test {
protected:
    void SetUp() override {
        // T2 arena for the host-args staging slice; Engine::init opens it in production.
        arena_ = std::make_unique<ScopedEngineArena>();
        gemm_grouped_nvfp4_prewarm();
        cudaStreamCreate(&stream_);
        int dev = 0;
        cudaGetDevice(&dev);
        int major = 0, minor = 0;
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
        sm_ = major * 10 + minor;
    }
    void TearDown() override {
        cudaStreamDestroy(stream_);
        arena_.reset();
    }
    std::unique_ptr<ScopedEngineArena> arena_;
    cudaStream_t stream_ = nullptr;
    int sm_ = 0;

    // Host-args grouped GEMM on the given per-expert distribution vs the fp64 reference.
    void run_case(const std::vector<int>& M_per, int N, int K, uint64_t seed) {
        ASSERT_GE(sm_, 120) << "SM120 required";
        const int ne = static_cast<int>(M_per.size());
        ASSERT_TRUE(gemm_grouped_nvfp4_supports(ne, N, K));

        nvfp4_ref::GroupedHost h;
        h.ne = ne;
        h.N = N;
        h.K = K;
        h.offsets.assign(ne + 1, 0);
        h.sfa_off.assign(ne, 0);
        size_t sfa_total = 0;
        for (int e = 0; e < ne; ++e) {
            h.offsets[e + 1] = h.offsets[e] + M_per[e];
            h.sfa_off[e] = sfa_total;
            sfa_total += cutlass_nvfp4_sf_size(M_per[e], K);
        }
        const int M_total = h.offsets[ne];
        ASSERT_GT(M_total, 0) << "all-empty distribution is meaningless";

        // Experts with distinct weight scales, so a wrong expert's alpha shows.
        std::vector<SyntheticExpert> experts(ne);
        std::vector<const void*> hB(ne), hSFB(ne);
        h.alpha.resize(ne);
        h.b.resize(ne);
        h.b_sf.resize(ne);
        const size_t b_bytes = static_cast<size_t>(N) * K / 2;
        const size_t sfb_bytes = cutlass_nvfp4_sf_size(N, K);
        for (int i = 0; i < ne; ++i) {
            make_expert(experts[i], N, K, /*wscale=*/0.25f * static_cast<float>(1 + (i % 5)), seed + 1000 + i,
                        stream_);
            hB[i] = experts[i].cutlass_w.data;
            hSFB[i] = experts[i].cutlass_w.scale_factors;
            h.alpha[i] = experts[i].cutlass_w.tensor_scale;
            h.b[i].resize(b_bytes);
            h.b_sf[i].resize(sfb_bytes);
            cudaMemcpy(h.b[i].data(), hB[i], b_bytes, cudaMemcpyDeviceToHost);
            cudaMemcpy(h.b_sf[i].data(), hSFB[i], sfb_bytes, cudaMemcpyDeviceToHost);
        }

        // FP16 activations [M_total, K], quantized per expert into contiguous A / SFA slabs.
        std::mt19937 agen(static_cast<unsigned>(seed));
        std::uniform_real_distribution<float> adist(-1.0f, 1.0f);
        std::vector<half> h_A(static_cast<size_t>(M_total) * K);
        for (auto& v : h_A)
            v = __float2half(adist(agen));
        void* d_A_fp16 = nullptr;
        void* d_a = nullptr;
        void* d_a_sf = nullptr;
        cudaMalloc(&d_A_fp16, h_A.size() * sizeof(half));
        cudaMemcpy(d_A_fp16, h_A.data(), h_A.size() * sizeof(half), cudaMemcpyHostToDevice);
        cudaMalloc(&d_a, static_cast<size_t>(M_total) * K / 2);
        cudaMalloc(&d_a_sf, sfa_total);
        cudaMemset(d_a_sf, 0, sfa_total);
        for (int e = 0; e < ne; ++e)
            if (M_per[e] > 0)
                quantize_fp16_to_nvfp4_cutlass(
                    static_cast<const half*>(d_A_fp16) + (static_cast<size_t>(h.offsets[e]) * K),
                    static_cast<char*>(d_a) + (static_cast<size_t>(h.offsets[e]) * K / 2),
                    static_cast<char*>(d_a_sf) + h.sfa_off[e], M_per[e], K, stream_);
        ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
        h.a.resize(static_cast<size_t>(M_total) * K / 2);
        h.a_sf.resize(sfa_total);
        cudaMemcpy(h.a.data(), d_a, h.a.size(), cudaMemcpyDeviceToHost);
        cudaMemcpy(h.a_sf.data(), d_a_sf, h.a_sf.size(), cudaMemcpyDeviceToHost);

        const size_t out_bytes = static_cast<size_t>(M_total) * N * sizeof(half);
        void* d_out = nullptr;
        cudaMalloc(&d_out, out_bytes);
        cudaMemset(d_out, 0xff, out_bytes);  // NaN fill: a row the kernel misses fails the check
        GroupedNvfp4HostArgs args{};
        args.offsets = h.offsets.data();
        args.sfa_offsets = h.sfa_off.data();
        args.B = hB.data();
        args.SFB = hSFB.data();
        args.alpha = h.alpha.data();
        args.base_A_packed = d_a;
        args.base_A_sf = d_a_sf;
        args.base_D = d_out;
        ASSERT_TRUE(gemm_grouped_nvfp4_host_args(ne, N, K, M_total, args, stream_))
            << "host-args dispatch failed";
        ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
        std::vector<half> out(static_cast<size_t>(M_total) * N);
        cudaMemcpy(out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost);

        const nvfp4_ref::GroupedErr err = nvfp4_ref::check_grouped(h, out, 1e-2);
        std::printf(
            "[grouped-ref] ne=%d N=%d K=%d M_total=%d checked=%zu bad=%zu nonfinite=%zu max_rel=%.3e "
            "max_abs=%.3e max|ref|=%.3g\n",
            ne, N, K, M_total, err.checked, err.bad, err.nonfinite, err.max_rel, err.max_abs, err.max_ref);
        EXPECT_EQ(err.checked, static_cast<size_t>(M_total) * N);
        EXPECT_GT(err.max_ref, 0.0);
        EXPECT_EQ(err.bad, 0u) << err.nonfinite << " non-finite, max_rel " << err.max_rel;

        for (auto& e : experts)
            free_expert(e);
        cudaFree(d_A_fp16);
        cudaFree(d_a);
        cudaFree(d_a_sf);
        cudaFree(d_out);
    }
};

// Even token distribution, multiple experts.
TEST_F(GroupedNvfp4RefTest, EvenDistribution) {
    run_case({32, 16, 48, 64}, /*N=*/256, /*K=*/256, /*seed=*/42);
}

// An EMPTY expert (M=0), a 1-token expert, and a large-M expert crossing the 128-row SfAtom tile.
TEST_F(GroupedNvfp4RefTest, BoundaryDistribution) {
    run_case({0, 1, 200, 7}, /*N=*/128, /*K=*/256, /*seed=*/7);
}

// Single active expert between empty neighbours.
TEST_F(GroupedNvfp4RefTest, SingleActiveExpert) { run_case({0, 33, 0}, /*N=*/192, /*K=*/256, /*seed=*/123); }

// K below the old CUTLASS 384 floor; K % 128 == 64 ends on a half stage (one SF atom). N = 200: N tail.
class GroupedNvfp4RefSmallK : public GroupedNvfp4RefTest, public ::testing::WithParamInterface<int> {};

TEST_P(GroupedNvfp4RefSmallK, MatchesFp64Reference) {
    run_case({5, 0, 130, 17}, /*N=*/200, GetParam(), /*seed=*/900 + GetParam());
}

INSTANTIATE_TEST_SUITE_P(K, GroupedNvfp4RefSmallK, ::testing::Values(64, 128, 192, 256));

// 512 experts (the kernel maximum), 0-6 rows each, N = 256: ~2 tiles per active expert over ~170 CTAs, so a
// CTA walks several experts and cycles all 4 tensormap slots; K = 64 (one stage per tile) reuses a slot
// soonest.
class GroupedNvfp4RefManyExperts : public GroupedNvfp4RefTest, public ::testing::WithParamInterface<int> {};

TEST_P(GroupedNvfp4RefManyExperts, MatchesFp64Reference) {
    std::mt19937 gen(31);
    std::uniform_int_distribution<int> dist(0, 6);
    std::vector<int> m(512);
    for (int& v : m)
        v = dist(gen);
    run_case(m, /*N=*/256, GetParam(), /*seed=*/77 + GetParam());
}

INSTANTIATE_TEST_SUITE_P(K, GroupedNvfp4RefManyExperts, ::testing::Values(64, 192));

}  // namespace
}  // namespace imp
