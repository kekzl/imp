#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <cstdlib>

#include "compute/ggml_mmvq.h"
#include "compute/gemm.h"

namespace imp {

static half* make_random_fp16(int n, unsigned seed = 42) {
    std::srand(seed);
    std::vector<half> host(n);
    for (int i = 0; i < n; i++)
        host[i] = __float2half((std::rand() / (float)RAND_MAX - 0.5f) * 2.0f);
    half* dev = nullptr;
    cudaMalloc(&dev, n * sizeof(half));
    cudaMemcpy(dev, host.data(), n * sizeof(half), cudaMemcpyHostToDevice);
    return dev;
}

static void* make_q8_0_weights(int N, int K, unsigned seed = 123) {
    int blocks_per_row = K / 32;
    size_t total_bytes = (size_t)N * blocks_per_row * 34;
    std::vector<uint8_t> host(total_bytes);
    std::srand(seed);
    for (int row = 0; row < N; row++) {
        for (int blk = 0; blk < blocks_per_row; blk++) {
            uint8_t* bp = host.data() + (row * blocks_per_row + blk) * 34;
            float amax = 0;
            int8_t qs[32];
            for (int i = 0; i < 32; i++) {
                float v = (std::rand() / (float)RAND_MAX - 0.5f) * 4.0f;
                qs[i] = (int8_t)std::max(-127.0f, std::min(127.0f, std::round(v * 32.0f)));
                amax = std::max(amax, std::abs((float)qs[i]));
            }
            float d = amax / 127.0f;
            half d_h = __float2half(d);
            memcpy(bp, &d_h, 2);
            memcpy(bp + 2, qs, 32);
        }
    }
    void* dev = nullptr;
    cudaMalloc(&dev, total_bytes);
    cudaMemcpy(dev, host.data(), total_bytes, cudaMemcpyHostToDevice);
    return dev;
}

static void* make_random_packed_weights(int N, int K, int block_bytes, int block_elems, unsigned seed = 99) {
    int blocks_per_row = K / block_elems;
    size_t total_bytes = (size_t)N * blocks_per_row * block_bytes;
    std::vector<uint8_t> host(total_bytes);
    std::srand(seed);
    for (size_t i = 0; i < total_bytes; i++)
        host[i] = std::rand() & 0xFF;
    for (int row = 0; row < N; row++) {
        for (int blk = 0; blk < blocks_per_row; blk++) {
            uint8_t* bp = host.data() + (row * blocks_per_row + blk) * block_bytes;
            half d = __float2half(0.01f + 0.001f * (std::rand() % 100));
            memcpy(bp, &d, 2);
            if (block_bytes >= 4) {
                half dmin = __float2half(0.001f + 0.0001f * (std::rand() % 100));
                memcpy(bp + 2, &dmin, 2);
            }
        }
    }
    void* dev = nullptr;
    cudaMalloc(&dev, total_bytes);
    cudaMemcpy(dev, host.data(), total_bytes, cudaMemcpyHostToDevice);
    return dev;
}

static void compare_dp4a_vs_mmvq(const char* name, void* W, int N, int K, half* x_fp16,
                                 void (*mmvq_fn)(const void*, const half*, half*, int, int, int, void*,
                                                 size_t, cudaStream_t),
                                 void (*dp4a_fn)(const void*, const block_q8_1*, const float*, half*, int,
                                                 int, cudaStream_t)) {
    half *out_dp4a = nullptr, *out_mmvq = nullptr;
    cudaMalloc(&out_dp4a, N * sizeof(half));
    cudaMalloc(&out_mmvq, N * sizeof(half));

    block_q8_1* q8_buf = nullptr;
    float* d8_buf = nullptr;
    int padded_blocks = ((K + 255) / 256) * 8;
    cudaMalloc(&q8_buf, padded_blocks * sizeof(block_q8_1));
    cudaMalloc(&d8_buf, padded_blocks * sizeof(float));
    cudaMemset(q8_buf, 0, padded_blocks * sizeof(block_q8_1));
    cudaMemset(d8_buf, 0, padded_blocks * sizeof(float));
    quantize_fp16_to_q8_1(x_fp16, q8_buf, d8_buf, K, nullptr);
    dp4a_fn(W, q8_buf, d8_buf, out_dp4a, N, K, nullptr);

    int q8_blocks = (K + 31) / 32;
    size_t scratch_size = (size_t)q8_blocks * 36 * 2;
    void* scratch = nullptr;
    cudaMalloc(&scratch, scratch_size);
    mmvq_fn(W, x_fp16, out_mmvq, 1, N, K, scratch, scratch_size, nullptr);

    cudaDeviceSynchronize();

    std::vector<half> h_dp4a(N), h_mmvq(N);
    cudaMemcpy(h_dp4a.data(), out_dp4a, N * sizeof(half), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_mmvq.data(), out_mmvq, N * sizeof(half), cudaMemcpyDeviceToHost);

    float max_abs_err = 0, max_rel_err = 0;
    int worst_idx = 0;
    for (int i = 0; i < N; i++) {
        float vd = __half2float(h_dp4a[i]);
        float vm = __half2float(h_mmvq[i]);
        float abs_err = std::abs(vd - vm);
        float rel_err = (std::abs(vd) > 1e-6f) ? abs_err / std::abs(vd) : abs_err;
        if (abs_err > max_abs_err) {
            max_abs_err = abs_err;
            max_rel_err = rel_err;
            worst_idx = i;
        }
    }

    printf("[%s] N=%d K=%d max_abs_err=%.6f max_rel_err=%.4f%% at idx=%d (dp4a=%.4f mmvq=%.4f)\n", name, N, K,
           max_abs_err, max_rel_err * 100.0f, worst_idx, __half2float(h_dp4a[worst_idx]),
           __half2float(h_mmvq[worst_idx]));

    EXPECT_LT(max_rel_err, 0.002f) << name << ": relative error too large";
    // abs_err threshold 2.0 allows FP16 rounding at large magnitudes (~1024+);
    // the relative error check above is the meaningful correctness gate.
    EXPECT_LE(max_abs_err, 2.0f) << name << ": absolute error too large";

    cudaFree(out_dp4a);
    cudaFree(out_mmvq);
    cudaFree(q8_buf);
    cudaFree(d8_buf);
    cudaFree(scratch);
}

TEST(MMVQ, Q4_K_MatchesDp4a) {
    const int N = 64, K = 256;
    void* W = make_random_packed_weights(N, K, 144, 256);
    half* x = make_random_fp16(K);
    compare_dp4a_vs_mmvq("Q4_K", W, N, K, x, ggml_mmvq_q4k, gemv_q4_k_q8_1);
    cudaFree(W);
    cudaFree(x);
}

TEST(MMVQ, Q5_K_MatchesDp4a) {
    const int N = 64, K = 256;
    void* W = make_random_packed_weights(N, K, 176, 256);
    half* x = make_random_fp16(K);
    compare_dp4a_vs_mmvq("Q5_K", W, N, K, x, ggml_mmvq_q5k, gemv_q5_k_q8_1);
    cudaFree(W);
    cudaFree(x);
}

TEST(MMVQ, Q8_0_MatchesDp4a) {
    const int N = 64, K = 256;
    void* W = make_q8_0_weights(N, K);
    half* x = make_random_fp16(K);
    compare_dp4a_vs_mmvq("Q8_0", W, N, K, x, ggml_mmvq_q8_0, gemv_q8_0_q8_1);
    cudaFree(W);
    cudaFree(x);
}

TEST(MMVQ, Q4_K_LargerDims) {
    const int N = 2048, K = 2816;
    void* W = make_random_packed_weights(N, K, 144, 256);
    half* x = make_random_fp16(K);
    compare_dp4a_vs_mmvq("Q4_K_large", W, N, K, x, ggml_mmvq_q4k, gemv_q4_k_q8_1);
    cudaFree(W);
    cudaFree(x);
}

TEST(MMVQ, Q5_K_LargerDims) {
    const int N = 2816, K = 4096;
    void* W = make_random_packed_weights(N, K, 176, 256);
    half* x = make_random_fp16(K);
    compare_dp4a_vs_mmvq("Q5_K_large", W, N, K, x, ggml_mmvq_q5k, gemv_q5_k_q8_1);
    cudaFree(W);
    cudaFree(x);
}

TEST(MMVQ, Q8_0_LargerDims) {
    const int N = 2816, K = 2112;
    void* W = make_q8_0_weights(N, K);
    half* x = make_random_fp16(K);
    compare_dp4a_vs_mmvq("Q8_0_large", W, N, K, x, ggml_mmvq_q8_0, gemv_q8_0_q8_1);
    cudaFree(W);
    cudaFree(x);
}

}  // namespace imp
