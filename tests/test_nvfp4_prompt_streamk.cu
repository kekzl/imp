// gemm.nvfp4_prompt_streamk off (auto on MoE): prompt rows (chunks >= 33) of the CUTLASS_NVFP4 registry
// kernel never take stream-K, whose K split follows M (Gemma-4-26B global q_proj N 8192 K 2816: 218 rows
// data-parallel, 42 rows split in 2; prefix resend 0.179 nats).

#include "compute/gemm_cutlass_sm120.h"  // CutlassNvFP4Weight, convert_nvfp4_to_cutlass
#include "core/process_diag.h"
#include "core/tensor.h"
#include "exec/gemm_kernel_registry.h"
#include "quant/nvfp4_gemm_tile.h"  // gemm_nvfp4_tile, gemm_tile_streamk_workspace_bytes
#include "quant/nvfp4_quant.h"

#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

using namespace imp;

namespace {

// One weight [N,K], `big` activation rows; the last `small` rows dispatch alone and on forced tile variants.
// Prompt stream-K is off for the rig's lifetime.
struct PromptRowsRig {
    int N, K, big, small;
    bool prompt_sk = process_diag_nvfp4_prompt_streamk();
    cudaStream_t stream = nullptr;
    __half *d_x = nullptr, *d_w = nullptr;
    void *d_act = nullptr, *d_act_sf = nullptr, *d_ws = nullptr;
    size_t sf_bytes = 0, ws_bytes = gemm_tile_streamk_workspace_bytes();
    NvFP4QuantResult nv4{};
    CutlassNvFP4Weight cw{};

    PromptRowsRig(int n, int k, int m_big, int m_small, uint32_t seed)
        : N(n), K(k), big(m_big), small(m_small) {
        process_diag_set_nvfp4_prompt_streamk(false);
        EXPECT_EQ(cudaStreamCreate(&stream), cudaSuccess);
        std::vector<__half> h_x(size_t(big) * K), h_w(size_t(N) * K);
        uint32_t st = seed;
        auto unit = [&] {  // xorshift, uniform in [-1, 1)
            st ^= st << 13;
            st ^= st >> 17;
            st ^= st << 5;
            return static_cast<float>(st >> 8) * (2.0f / 16777216.0f) - 1.0f;
        };
        for (size_t i = 0; i < h_x.size(); ++i)  // 2^-6..2^6 per 32 channels: K sums depend on their order
            h_x[i] = __float2half(unit() * std::ldexp(1.0f, static_cast<int>((i % K) / 32 * 7 % 13) - 6));
        for (auto& v : h_w)
            v = __float2half(unit() * 0.05f);
        cudaMalloc(&d_x, h_x.size() * sizeof(__half));
        cudaMalloc(&d_w, h_w.size() * sizeof(__half));
        cudaMemcpy(d_x, h_x.data(), h_x.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cudaMemcpy(d_w, h_w.data(), h_w.size() * sizeof(__half), cudaMemcpyHostToDevice);
        int64_t w_shape[2] = {N, K};
        Tensor weight_fp16(d_w, QType::F16, 2, w_shape, /*on_device=*/true);
        quantize_fp16_to_nvfp4(weight_fp16, nv4, stream);
        convert_nvfp4_to_cutlass(nv4, cw, stream);
        sf_bytes = cutlass_nvfp4_sf_size(big, K);
        cudaMalloc(&d_act, size_t(big) * K / 2);
        cudaMalloc(&d_act_sf, sf_bytes);
        cudaMalloc(&d_ws, ws_bytes);
    }
    ~PromptRowsRig() {
        free_cutlass_nvfp4_weight(cw);
        free_nvfp4_result(nv4);
        cudaFree(d_act);
        cudaFree(d_act_sf);
        cudaFree(d_ws);
        cudaFree(d_x);
        cudaFree(d_w);
        cudaStreamDestroy(stream);
        process_diag_set_nvfp4_prompt_streamk(prompt_sk);
    }
    PromptRowsRig(const PromptRowsRig&) = delete;
    PromptRowsRig& operator=(const PromptRowsRig&) = delete;

    const __half* tail() const { return d_x + size_t(big - small) * K; }

    // The registry dispatch (auto plan, workspace offered) of m rows from x, copied to the host.
    std::vector<__half> dispatch(const __half* x, int m) {
        __half* d_y = nullptr;
        cudaMalloc(&d_y, size_t(m) * N * sizeof(__half));
        int64_t in_shape[2] = {m, K};
        int64_t out_shape[2] = {m, N};
        Tensor input(const_cast<__half*>(x), QType::F16, 2, in_shape, /*on_device=*/true);
        Tensor output(d_y, QType::F16, 2, out_shape, /*on_device=*/true);
        cudaMemsetAsync(d_act_sf, 0, sf_bytes, stream);
        GemmKernelArgs args{};
        args.input = &input;
        args.output = &output;
        args.stream = stream;
        args.beta = 0.0f;
        args.weight_payload = &cw;
        args.cutlass_act_data = d_act;
        args.cutlass_act_sf = d_act_sf;
        args.cutlass_workspace = d_ws;
        args.cutlass_workspace_size = ws_bytes;
        GemmStrategy strat{StorageTier::CUTLASS_NVFP4, QType::F16, /*m_is_one=*/false};
        EXPECT_EQ(GemmKernelRegistry::instance().dispatch(strat, args), GemmDispatchResult::Ok) << "M=" << m;
        return to_host(d_y, m);
    }
    // The tail rows on one forced tile variant (with the workspace).
    std::vector<__half> forced(int variant) {
        __half* d_y = nullptr;
        cudaMalloc(&d_y, size_t(small) * N * sizeof(__half));
        cudaMemsetAsync(d_act_sf, 0, sf_bytes, stream);
        quantize_fp16_to_nvfp4_cutlass(tail(), d_act, d_act_sf, small, K, stream);
        EXPECT_TRUE(gemm_nvfp4_tile(d_act, d_act_sf, cw.data, cw.scale_factors, cw.tensor_scale, d_y, small,
                                    N, K, stream, variant, d_ws, ws_bytes))
            << "variant " << variant;
        return to_host(d_y, small);
    }
    std::vector<__half> to_host(__half* d_y, int m) {
        EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        std::vector<__half> h(size_t(m) * N);
        cudaMemcpy(h.data(), d_y, h.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        cudaFree(d_y);
        return h;
    }
    // Tail rows of the M=big result equal y bitwise.
    bool tail_equals(const std::vector<__half>& big_y, const std::vector<__half>& y) const {
        return std::memcmp(big_y.data() + size_t(big - small) * N, y.data(), y.size() * sizeof(__half)) == 0;
    }
};

bool same_bits(const std::vector<__half>& a, const std::vector<__half>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(__half)) == 0;
}

// Rows 176..217 of the M=218 dispatch must equal the M=42 dispatch bitwise; the forced split-2 tile at
// M=42 must differ, else the activations sum exactly in fp32 and this checks nothing.
TEST(Nvfp4PromptStreamK, PromptRowsInvariantToM) {
    PromptRowsRig r(/*N=*/8192, /*K=*/2816, /*big=*/218, /*small=*/42, 0x9E3779B9u);
    const auto big = r.dispatch(r.d_x, r.big);
    const auto small = r.dispatch(r.tail(), r.small);
    const auto sk = r.forced(/*variant=*/4);  // hybrid stream-K, as the dispatch picked before
    EXPECT_TRUE(r.tail_equals(big, small)) << "prompt rows depend on M (stream-K on a prompt chunk)";
    EXPECT_FALSE(same_bits(sk, small))
        << "split-2 matched data-parallel bitwise: the activations sum exactly, nothing is checked";
}

// Up to 0.4 wave of 128x128 tiles the no-workspace plan runs 128x64 tiles (Qwen3-30B-A3B o_proj M 512
// N 2048 K 4096: 64 -> 128 CTAs). Its rows must equal the 128x128 rows of the M=1024 dispatch bitwise.
TEST(Nvfp4PromptStreamK, NarrowTileRowsMatchWideTile) {
    PromptRowsRig r(/*N=*/2048, /*K=*/4096, /*big=*/1024, /*small=*/512, 0x2545F491u);
    const auto big = r.dispatch(r.d_x, r.big);
    const auto small = r.dispatch(r.tail(), r.small);
    const auto narrow = r.forced(/*variant=*/5);  // 128x64 data-parallel
    const auto sk = r.forced(/*variant=*/4);      // split-2 stream-K
    EXPECT_TRUE(r.tail_equals(big, small)) << "prompt rows depend on M (narrow tile changed the k order)";
    EXPECT_TRUE(r.tail_equals(big, narrow)) << "128x64 rows differ from the 128x128 rows of M=1024";
    EXPECT_FALSE(same_bits(sk, small))
        << "split-2 matched data-parallel bitwise: the activations sum exactly, nothing is checked";
}

}  // namespace
