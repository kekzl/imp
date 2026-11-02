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

// Rows 176..217 of the M=218 dispatch must equal the M=42 dispatch bitwise; the forced split-2 tile at
// M=42 must differ, else the activations sum exactly in fp32 and this checks nothing.
TEST(Nvfp4PromptStreamK, PromptRowsInvariantToM) {
    const bool prompt_sk = process_diag_nvfp4_prompt_streamk();
    process_diag_set_nvfp4_prompt_streamk(false);
    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    constexpr int N = 8192, K = 2816, kBig = 218, kSmall = 42, kTail = kBig - kSmall;
    std::vector<__half> h_x(size_t(kBig) * K), h_w(size_t(N) * K);
    uint32_t st = 0x9E3779B9u;
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
    __half *d_x = nullptr, *d_w = nullptr, *d_big = nullptr, *d_small = nullptr, *d_sk = nullptr;
    cudaMalloc(&d_x, h_x.size() * sizeof(__half));
    cudaMalloc(&d_w, h_w.size() * sizeof(__half));
    cudaMalloc(&d_big, size_t(kBig) * N * sizeof(__half));
    cudaMalloc(&d_small, size_t(kSmall) * N * sizeof(__half));
    cudaMalloc(&d_sk, size_t(kSmall) * N * sizeof(__half));
    cudaMemcpy(d_x, h_x.data(), h_x.size() * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, h_w.data(), h_w.size() * sizeof(__half), cudaMemcpyHostToDevice);
    int64_t w_shape[2] = {N, K};
    Tensor weight_fp16(d_w, QType::F16, 2, w_shape, /*on_device=*/true);
    NvFP4QuantResult nv4{};
    quantize_fp16_to_nvfp4(weight_fp16, nv4, stream);
    CutlassNvFP4Weight cw{};
    convert_nvfp4_to_cutlass(nv4, cw, stream);
    const size_t sf_bytes = cutlass_nvfp4_sf_size(kBig, K);
    const size_t ws_bytes = gemm_tile_streamk_workspace_bytes();
    void *d_act = nullptr, *d_act_sf = nullptr, *d_ws = nullptr;
    cudaMalloc(&d_act, size_t(kBig) * K / 2);
    cudaMalloc(&d_act_sf, sf_bytes);
    cudaMalloc(&d_ws, ws_bytes);

    auto dispatch = [&](const __half* x, __half* y, int m) {
        int64_t in_shape[2] = {m, K};
        int64_t out_shape[2] = {m, N};
        Tensor input(const_cast<__half*>(x), QType::F16, 2, in_shape, /*on_device=*/true);
        Tensor output(y, QType::F16, 2, out_shape, /*on_device=*/true);
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
    };
    dispatch(d_x, d_big, kBig);
    dispatch(d_x + size_t(kTail) * K, d_small, kSmall);
    // The same 42 rows on the hybrid stream-K plan (variant 4), as the dispatch picked before.
    cudaMemsetAsync(d_act_sf, 0, sf_bytes, stream);
    quantize_fp16_to_nvfp4_cutlass(d_x + size_t(kTail) * K, d_act, d_act_sf, kSmall, K, stream);
    EXPECT_TRUE(gemm_nvfp4_tile(d_act, d_act_sf, cw.data, cw.scale_factors, cw.tensor_scale, d_sk, kSmall, N,
                                K, stream, /*variant=*/4, d_ws, ws_bytes));
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    std::vector<__half> h_big(size_t(kBig) * N), h_small(size_t(kSmall) * N), h_sk(h_small.size());
    cudaMemcpy(h_big.data(), d_big, h_big.size() * sizeof(__half), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_small.data(), d_small, h_small.size() * sizeof(__half), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_sk.data(), d_sk, h_sk.size() * sizeof(__half), cudaMemcpyDeviceToHost);
    EXPECT_EQ(0,
              std::memcmp(h_small.data(), h_big.data() + size_t(kTail) * N, h_small.size() * sizeof(__half)))
        << "prompt rows depend on M (stream-K on a prompt chunk)";
    EXPECT_NE(0, std::memcmp(h_sk.data(), h_small.data(), h_sk.size() * sizeof(__half)))
        << "split-2 matched data-parallel bitwise: the activations sum exactly, nothing is checked";

    free_cutlass_nvfp4_weight(cw);
    free_nvfp4_result(nv4);
    cudaFree(d_act);
    cudaFree(d_act_sf);
    cudaFree(d_ws);
    cudaFree(d_x);
    cudaFree(d_w);
    cudaFree(d_big);
    cudaFree(d_small);
    cudaFree(d_sk);
    cudaStreamDestroy(stream);
    process_diag_set_nvfp4_prompt_streamk(prompt_sk);
}

}  // namespace
