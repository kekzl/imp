#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "compute/ssm.h"
#include "core/tensor.h"

#include <vector>
#include <cmath>

#define SKIP_IF_NO_CUDA() do { int dev; if (cudaGetDevice(&dev) != cudaSuccess) GTEST_SKIP(); } while(0)

namespace imp {
namespace {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
Tensor make_fp16_gpu(const float* host, std::initializer_list<int64_t> shape) {
    Tensor t;
    t.qtype = QType::F16;
    t.ndim  = static_cast<int>(shape.size());
    int i = 0;
    for (auto s : shape) t.shape[i++] = s;
    t.compute_strides();
    t.on_device = true;
    cudaMalloc(&t.data, t.nbytes());
    std::vector<half> h(t.numel());
    for (int64_t j = 0; j < t.numel(); j++)
        h[j] = __float2half(host[j]);
    cudaMemcpy(t.data, h.data(), t.nbytes(), cudaMemcpyHostToDevice);
    return t;
}

Tensor alloc_fp16_gpu(std::initializer_list<int64_t> shape) {
    Tensor t;
    t.qtype = QType::F16;
    t.ndim  = static_cast<int>(shape.size());
    int i = 0;
    for (auto s : shape) t.shape[i++] = s;
    t.compute_strides();
    t.on_device = true;
    cudaMalloc(&t.data, t.nbytes());
    cudaMemset(t.data, 0, t.nbytes());
    return t;
}

Tensor make_empty_tensor() {
    Tensor t;
    t.data = nullptr;
    t.ndim = 0;
    t.qtype = QType::F16;
    return t;
}

std::vector<float> read_fp16(const Tensor& t) {
    std::vector<half> h(t.numel());
    cudaMemcpy(h.data(), t.data, t.nbytes(), cudaMemcpyDeviceToHost);
    std::vector<float> r(t.numel());
    for (int64_t j = 0; j < t.numel(); j++)
        r[j] = __half2float(h[j]);
    return r;
}

void free_tensor(Tensor& t) {
    if (t.data) { cudaFree(t.data); t.data = nullptr; }
}

// ===========================================================================
// Test 1: Conv1d decode — shift state and convolve
// ===========================================================================
TEST(SSMConv1dTest, DecodeShiftAndConvolve) {
    SKIP_IF_NO_CUDA();

    constexpr int channels = 4;
    constexpr int kernel_size = 4;

    // Conv state: [channels, kernel_size] in FP32
    std::vector<float> h_state(channels * kernel_size);
    for (int ch = 0; ch < channels; ch++)
        for (int k = 0; k < kernel_size; k++)
            h_state[ch * kernel_size + k] = static_cast<float>(ch * 10 + k);

    // Weight: [channels, kernel_size] in FP16
    std::vector<float> h_weight(channels * kernel_size);
    for (int i = 0; i < channels * kernel_size; i++)
        h_weight[i] = 0.25f;  // uniform weight for easy verification

    // Input token: [channels]
    std::vector<float> h_x(channels);
    for (int ch = 0; ch < channels; ch++)
        h_x[ch] = 100.0f + static_cast<float>(ch);

    // Allocate GPU buffers
    float* d_state;
    cudaMalloc(&d_state, channels * kernel_size * sizeof(float));
    cudaMemcpy(d_state, h_state.data(), channels * kernel_size * sizeof(float),
               cudaMemcpyHostToDevice);

    Tensor d_x = make_fp16_gpu(h_x.data(), {channels});
    Tensor d_w = make_fp16_gpu(h_weight.data(), {channels, kernel_size});
    Tensor d_out = alloc_fp16_gpu({channels});
    Tensor d_bias = make_empty_tensor();

    ssm_conv1d_decode(d_state, d_x, d_w, d_bias, d_out, kernel_size, nullptr);
    cudaDeviceSynchronize();

    // CPU reference: shift, insert, convolve
    auto out = read_fp16(d_out);
    for (int ch = 0; ch < channels; ch++) {
        // After shift: state = [old[1], old[2], old[3], new_x]
        float shifted[4];
        shifted[0] = h_state[ch * kernel_size + 1];
        shifted[1] = h_state[ch * kernel_size + 2];
        shifted[2] = h_state[ch * kernel_size + 3];
        shifted[3] = h_x[ch];

        float expected = 0.0f;
        for (int k = 0; k < kernel_size; k++)
            expected += shifted[k] * 0.25f;

        EXPECT_NEAR(out[ch], expected, 0.5f) << "Channel " << ch;
    }

    cudaFree(d_state);
    free_tensor(d_x);
    free_tensor(d_w);
    free_tensor(d_out);
}

// ===========================================================================
// Test 2: Conv1d prefill — causal (no future leakage)
// ===========================================================================
TEST(SSMConv1dTest, PrefillCausal) {
    SKIP_IF_NO_CUDA();

    constexpr int n_tokens = 6;
    constexpr int channels = 2;
    constexpr int kernel_size = 3;

    // Input: [n_tokens, channels] — simple ascending
    std::vector<float> h_x(n_tokens * channels);
    for (int i = 0; i < n_tokens * channels; i++)
        h_x[i] = static_cast<float>(i + 1);

    // Weight: [channels, kernel_size] = all 1.0 (sum kernel)
    std::vector<float> h_w(channels * kernel_size, 1.0f);

    // Conv state
    float* d_state;
    cudaMalloc(&d_state, channels * kernel_size * sizeof(float));
    cudaMemset(d_state, 0, channels * kernel_size * sizeof(float));

    Tensor d_x   = make_fp16_gpu(h_x.data(), {n_tokens, channels});
    Tensor d_w   = make_fp16_gpu(h_w.data(), {channels, kernel_size});
    Tensor d_out = alloc_fp16_gpu({n_tokens, channels});
    Tensor d_bias = make_empty_tensor();

    ssm_conv1d_prefill(d_state, d_x, d_w, d_bias, d_out, kernel_size, nullptr);
    cudaDeviceSynchronize();

    auto out = read_fp16(d_out);

    // CPU reference: causal conv with zero-padding
    for (int t = 0; t < n_tokens; t++) {
        for (int ch = 0; ch < channels; ch++) {
            float expected = 0.0f;
            for (int k = 0; k < kernel_size; k++) {
                int src_t = t - (kernel_size - 1) + k;
                if (src_t >= 0)
                    expected += h_x[src_t * channels + ch] * h_w[ch * kernel_size + k];
            }
            EXPECT_NEAR(out[t * channels + ch], expected, 0.5f)
                << "Token " << t << " channel " << ch;
        }
    }

    // Verify no future leakage: token 0 should only see itself (kernel pos 2)
    // With sum kernel and zero-pad, out[0][ch] = x[0][ch]
    for (int ch = 0; ch < channels; ch++) {
        EXPECT_NEAR(out[ch], h_x[ch], 0.5f) << "Token 0 should only see itself";
    }

    cudaFree(d_state);
    free_tensor(d_x);
    free_tensor(d_w);
    free_tensor(d_out);
}

// ===========================================================================
// Test 3: Prefill final state matches sequential decode states
// ===========================================================================
TEST(SSMConv1dTest, StateConsistency) {
    SKIP_IF_NO_CUDA();

    constexpr int n_tokens = 4;
    constexpr int channels = 2;
    constexpr int kernel_size = 3;

    std::vector<float> h_x(n_tokens * channels);
    for (int i = 0; i < n_tokens * channels; i++)
        h_x[i] = static_cast<float>(i + 1) * 0.5f;
    std::vector<float> h_w(channels * kernel_size, 0.5f);

    // --- Prefill path ---
    float* d_state_prefill;
    cudaMalloc(&d_state_prefill, channels * kernel_size * sizeof(float));
    cudaMemset(d_state_prefill, 0, channels * kernel_size * sizeof(float));

    Tensor d_x_pf   = make_fp16_gpu(h_x.data(), {n_tokens, channels});
    Tensor d_w_pf   = make_fp16_gpu(h_w.data(), {channels, kernel_size});
    Tensor d_out_pf = alloc_fp16_gpu({n_tokens, channels});
    Tensor d_bias   = make_empty_tensor();

    ssm_conv1d_prefill(d_state_prefill, d_x_pf, d_w_pf, d_bias, d_out_pf,
                       kernel_size, nullptr);
    cudaDeviceSynchronize();

    std::vector<float> prefill_state(channels * kernel_size);
    cudaMemcpy(prefill_state.data(), d_state_prefill,
               channels * kernel_size * sizeof(float), cudaMemcpyDeviceToHost);

    // --- Sequential decode path ---
    float* d_state_decode;
    cudaMalloc(&d_state_decode, channels * kernel_size * sizeof(float));
    cudaMemset(d_state_decode, 0, channels * kernel_size * sizeof(float));

    Tensor d_w_dec = make_fp16_gpu(h_w.data(), {channels, kernel_size});
    for (int t = 0; t < n_tokens; t++) {
        Tensor d_x_t   = make_fp16_gpu(h_x.data() + t * channels, {channels});
        Tensor d_out_t = alloc_fp16_gpu({channels});
        ssm_conv1d_decode(d_state_decode, d_x_t, d_w_dec, d_bias, d_out_t,
                          kernel_size, nullptr);
        cudaDeviceSynchronize();
        free_tensor(d_x_t);
        free_tensor(d_out_t);
    }

    std::vector<float> decode_state(channels * kernel_size);
    cudaMemcpy(decode_state.data(), d_state_decode,
               channels * kernel_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Compare states
    for (int i = 0; i < channels * kernel_size; i++) {
        EXPECT_NEAR(prefill_state[i], decode_state[i], 1e-2f)
            << "State mismatch at index " << i;
    }

    cudaFree(d_state_prefill);
    cudaFree(d_state_decode);
    free_tensor(d_x_pf);
    free_tensor(d_w_pf);
    free_tensor(d_out_pf);
    free_tensor(d_w_dec);
}

// ===========================================================================
// Test 4: Fused conv1d + SiLU FP32 output path
// ===========================================================================
TEST(SSMConv1dTest, FP32SiLUFused) {
    SKIP_IF_NO_CUDA();

    constexpr int channels = 4;
    constexpr int kernel_size = 4;

    // All ones state -> conv output = sum of weights per channel
    std::vector<float> h_state(channels * kernel_size, 1.0f);
    std::vector<float> h_w(channels * kernel_size, 0.5f);
    std::vector<float> h_x(channels, 1.0f);

    float* d_state;
    cudaMalloc(&d_state, channels * kernel_size * sizeof(float));
    cudaMemcpy(d_state, h_state.data(), channels * kernel_size * sizeof(float),
               cudaMemcpyHostToDevice);

    Tensor d_x    = make_fp16_gpu(h_x.data(), {channels});
    Tensor d_w    = make_fp16_gpu(h_w.data(), {channels, kernel_size});
    Tensor d_bias = make_empty_tensor();

    float* d_out_f32;
    cudaMalloc(&d_out_f32, channels * sizeof(float));

    ssm_conv1d_decode_f32_silu(d_state, d_x, d_w, d_bias, d_out_f32,
                                kernel_size, nullptr);
    cudaDeviceSynchronize();

    std::vector<float> h_out(channels);
    cudaMemcpy(h_out.data(), d_out_f32, channels * sizeof(float),
               cudaMemcpyDeviceToHost);

    // After shift: state = [old[1], old[2], old[3], new_x] = [1,1,1,1]
    // conv = sum(1.0 * 0.5) * 4 = 2.0
    // SiLU(2.0) = 2.0 / (1 + exp(-2.0)) = 2.0 * 0.8808 = 1.7616
    float conv_val = 4.0f * 0.5f;  // = 2.0
    float expected = conv_val / (1.0f + std::exp(-conv_val));

    for (int ch = 0; ch < channels; ch++) {
        EXPECT_NEAR(h_out[ch], expected, 0.05f) << "Channel " << ch;
    }

    cudaFree(d_state);
    cudaFree(d_out_f32);
    free_tensor(d_x);
    free_tensor(d_w);
}

} // namespace
} // namespace imp
