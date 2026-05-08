#include "compute/attention_cublas.h"
#include "core/tensor.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <gtest/gtest.h>
#include <vector>
#include <cmath>

namespace imp {

// Helper: allocate FP16 device tensor [d0, d1].
static Tensor make_fp16_tensor_2d(int d0, int d1) {
    int64_t shape[2] = {d0, d1};
    half* p = nullptr;
    cudaMalloc(&p, (size_t)d0 * d1 * sizeof(half));
    return Tensor(p, QType::F16, 2, shape, /*on_device=*/true);
}

static Tensor make_fp16_tensor_3d(int d0, int d1, int d2) {
    int64_t shape[3] = {d0, d1, d2};
    half* p = nullptr;
    cudaMalloc(&p, (size_t)d0 * d1 * d2 * sizeof(half));
    return Tensor(p, QType::F16, 3, shape, /*on_device=*/true);
}

static void fill_fp16_random(half* d_ptr, size_t n, uint32_t seed) {
    std::vector<half> h(n);
    std::srand(seed);
    for (size_t i = 0; i < n; i++) {
        h[i] = __float2half(((float)std::rand() / RAND_MAX) * 0.1f - 0.05f);
    }
    cudaMemcpy(d_ptr, h.data(), n * sizeof(half), cudaMemcpyHostToDevice);
}

// Sanity: square causal attention at q_offset=0 produces finite, nonzero output.
TEST(AttentionChunkedTest, RectangularEqualsSquareAtZeroOffset) {
    const int seq = 64, nh = 4, nkv = 4, hd = 32;
    const float scale = 1.0f / std::sqrt((float)hd);

    Tensor Q = make_fp16_tensor_2d(seq, nh * hd);
    Tensor K = make_fp16_tensor_2d(seq, nkv * hd);
    Tensor V = make_fp16_tensor_2d(seq, nkv * hd);
    Tensor O = make_fp16_tensor_2d(seq, nh * hd);
    // S: [nh, seq, seq] — fp32_elems = nh*seq*seq = 16384, buf_fp16 = 16384 → FP16 path
    Tensor S = make_fp16_tensor_3d(nh, seq, seq);

    fill_fp16_random((half*)Q.data, seq * nh * hd, 1);
    fill_fp16_random((half*)K.data, seq * nkv * hd, 2);
    fill_fp16_random((half*)V.data, seq * nkv * hd, 3);

    attention_cublas_prefill(Q, K, V, O, S, nh, nkv, hd, scale, /*causal=*/true,
                             /*softcap=*/0.0f, /*q_offset=*/0, 0);
    cudaDeviceSynchronize();

    std::vector<half> h_o((size_t)seq * nh * hd);
    cudaMemcpy(h_o.data(), O.data, h_o.size() * sizeof(half), cudaMemcpyDeviceToHost);

    // Sanity: no NaN, magnitudes plausible (not all zero).
    float sum_abs = 0.0f;
    for (size_t i = 0; i < h_o.size(); i++) {
        float v = __half2float(h_o[i]);
        ASSERT_FALSE(std::isnan(v));
        sum_abs += std::fabs(v);
    }
    EXPECT_GT(sum_abs, 0.0f);

    cudaFree(Q.data); cudaFree(K.data); cudaFree(V.data);
    cudaFree(O.data); cudaFree(S.data);
}

// Synthesized Q/K to verify the offset-aware causal mask. K is one-hot at column 0
// (only position 0 has nonzero K, all others are zero), so attention scores are
// nonzero only when Q attends to position 0. With q_offset=128 and q_len=64,
// Q[i]'s absolute position is 128 + i — should attend to position 0 (causal: 0 <= 128+i).
// O[:, 0] should equal V[0][0] = 7 because softmax over the visible K positions
// collapses to weight 1.0 on j=0 (only nonzero score).
TEST(AttentionChunkedTest, OffsetAwareCausalMask) {
    const int q_len = 64, kv_len = 192, q_offset = 128, nh = 1, nkv = 1, hd = 16;
    const float scale = 1.0f;

    Tensor Q = make_fp16_tensor_2d(q_len, nh * hd);
    Tensor K = make_fp16_tensor_2d(kv_len, nkv * hd);
    Tensor V = make_fp16_tensor_2d(kv_len, nkv * hd);
    Tensor O = make_fp16_tensor_2d(q_len, nh * hd);
    // S: [nh, kv_len, kv_len] — fp32_elems = 1*64*192 = 12288, buf_fp16 = 1*192*192 = 36864
    // use_fp32_s = (12288*2 <= 36864) = true → FP32 path (more accurate)
    Tensor S = make_fp16_tensor_3d(nh, kv_len, kv_len);

    // Q: dim 0 = 10, zero elsewhere
    std::vector<half> h_q(q_len * nh * hd, __float2half(0.f));
    for (int i = 0; i < q_len; i++) h_q[i * nh * hd + 0] = __float2half(10.f);

    // K: K[0][0] = 10, rest zero. Score = Q·K*scale: j=0 → 100, j>0 → 0.
    // The FP32 softmax subtracts max(=100), so exp(-100) ≈ 0 for j>0.
    // After softmax, weight[0] ≈ 1.0, weight[j>0] ≈ 0. So O[:, 0] ≈ V[0][0].
    std::vector<half> h_k(kv_len * nkv * hd, __float2half(0.f));
    h_k[0 * nkv * hd + 0] = __float2half(10.f);

    // V: V[j][0] = 7 at j=0, zero elsewhere. After softmax over visible K positions,
    // P will have weight ~1.0 on j=0 (dominant score), so O[:, 0] ≈ 7.
    std::vector<half> h_v(kv_len * nkv * hd, __float2half(0.f));
    h_v[0 * nkv * hd + 0] = __float2half(7.f);

    cudaMemcpy(Q.data, h_q.data(), h_q.size() * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(K.data, h_k.data(), h_k.size() * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(V.data, h_v.data(), h_v.size() * sizeof(half), cudaMemcpyHostToDevice);

    attention_cublas_prefill(Q, K, V, O, S, nh, nkv, hd, scale, /*causal=*/true,
                             /*softcap=*/0.0f, /*q_offset=*/q_offset, 0);
    cudaDeviceSynchronize();

    std::vector<half> h_o(q_len * nh * hd);
    cudaMemcpy(h_o.data(), O.data, h_o.size() * sizeof(half), cudaMemcpyDeviceToHost);

    // Every output row's component 0 equals 7.0 (because Q saw V[0]).
    for (int i = 0; i < q_len; i++) {
        float val = __half2float(h_o[i * nh * hd + 0]);
        EXPECT_NEAR(val, 7.0f, 0.05f) << "row " << i;
    }

    cudaFree(Q.data); cudaFree(K.data); cudaFree(V.data);
    cudaFree(O.data); cudaFree(S.data);
}

// GQA with ratio=4 and non-zero q_offset: just verify no NaN in output.
TEST(AttentionChunkedTest, GQA_Ratio4) {
    const int q_len = 32, kv_len = 64, nh = 16, nkv = 4, hd = 32;
    const float scale = 1.0f / std::sqrt((float)hd);

    Tensor Q = make_fp16_tensor_2d(q_len, nh * hd);
    Tensor K = make_fp16_tensor_2d(kv_len, nkv * hd);
    Tensor V = make_fp16_tensor_2d(kv_len, nkv * hd);
    Tensor O = make_fp16_tensor_2d(q_len, nh * hd);
    // S: [nh, kv_len, kv_len] — fp32_elems = 16*32*64 = 32768, buf_fp16 = 16*64*64 = 65536
    // use_fp32_s = (32768*2 <= 65536) = true → FP32 path
    Tensor S = make_fp16_tensor_3d(nh, kv_len, kv_len);

    fill_fp16_random((half*)Q.data, q_len * nh * hd, 7);
    fill_fp16_random((half*)K.data, kv_len * nkv * hd, 8);
    fill_fp16_random((half*)V.data, kv_len * nkv * hd, 9);

    attention_cublas_prefill(Q, K, V, O, S, nh, nkv, hd, scale, /*causal=*/true,
                             /*softcap=*/0.0f, /*q_offset=*/16, 0);
    cudaDeviceSynchronize();

    std::vector<half> h_o((size_t)q_len * nh * hd);
    cudaMemcpy(h_o.data(), O.data, h_o.size() * sizeof(half), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < h_o.size(); i++) {
        ASSERT_FALSE(std::isnan(__half2float(h_o[i])));
    }

    cudaFree(Q.data); cudaFree(K.data); cudaFree(V.data);
    cudaFree(O.data); cudaFree(S.data);
}

}  // namespace imp
