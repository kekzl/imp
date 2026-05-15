// Unit tests for the tiled Q4_K MMQ kernel (src/compute/mmq_q4k.cu).
// Reference: ggml_mmvq_q4k computed on the same (W, x). Both kernels use
// identical Q8_1 quantization and the same vec_dot_q4_K_q8_1 dp4a sequence,
// so outputs should differ only by dp4a accumulation order at large K.

#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "compute/ggml_mmvq.h"
#include "compute/mmq_q4k.h"

namespace imp {

namespace {

// Build a random Q4_K weight tensor. Block layout matches the on-disk GGUF
// format exactly (144 bytes per 256 elements). For the test we only need
// the bytes to be self-consistent — random nibbles + a plausible scale/min
// give meaningful dot products without needing a real GGUF.
std::vector<uint8_t> make_random_q4k(int N, int K, unsigned seed) {
    const int blocks_per_row = K / 256;
    const size_t total = static_cast<size_t>(N) * blocks_per_row * 144;
    std::vector<uint8_t> host(total);

    std::srand(seed);
    for (int row = 0; row < N; ++row) {
        for (int blk = 0; blk < blocks_per_row; ++blk) {
            uint8_t* bp = host.data() + (static_cast<size_t>(row) * blocks_per_row + blk) * 144;
            // d, dmin: small positive halfs
            half d = __float2half(0.005f + 0.01f * (std::rand() % 100) / 100.0f);
            half dmin = __float2half(0.001f + 0.005f * (std::rand() % 100) / 100.0f);
            std::memcpy(bp + 0, &d, 2);
            std::memcpy(bp + 2, &dmin, 2);
            // scales: 12 bytes of pseudo-random 6-bit data
            for (int i = 0; i < 12; ++i) bp[4 + i] = static_cast<uint8_t>(std::rand() & 0x3F);
            // qs: 128 bytes of 4-bit nibble pairs
            for (int i = 0; i < 128; ++i) bp[16 + i] = static_cast<uint8_t>(std::rand() & 0xFF);
        }
    }
    return host;
}

half* upload_fp16(const std::vector<half>& host) {
    half* dev = nullptr;
    cudaMalloc(&dev, host.size() * sizeof(half));
    cudaMemcpy(dev, host.data(), host.size() * sizeof(half), cudaMemcpyHostToDevice);
    return dev;
}

void* upload_bytes(const std::vector<uint8_t>& host) {
    void* dev = nullptr;
    cudaMalloc(&dev, host.size());
    cudaMemcpy(dev, host.data(), host.size(), cudaMemcpyHostToDevice);
    return dev;
}

void compare(int M, int N, int K, unsigned seed) {
    auto w_host = make_random_q4k(N, K, seed);
    std::vector<half> x_host(static_cast<size_t>(M) * K);
    std::srand(seed ^ 0x9e3779b9);
    for (auto& v : x_host)
        v = __float2half((std::rand() / static_cast<float>(RAND_MAX) - 0.5f) * 2.0f);

    void* W_dev = upload_bytes(w_host);
    half* x_dev = upload_fp16(x_host);
    half* y_ref = nullptr;
    half* y_test = nullptr;
    cudaMalloc(&y_ref, static_cast<size_t>(M) * N * sizeof(half));
    cudaMalloc(&y_test, static_cast<size_t>(M) * N * sizeof(half));
    cudaMemset(y_ref, 0, static_cast<size_t>(M) * N * sizeof(half));
    cudaMemset(y_test, 0, static_cast<size_t>(M) * N * sizeof(half));

    const size_t scratch_size = mmq_q4k_scratch_bytes(M, K);
    void* scratch = nullptr;
    cudaMalloc(&scratch, scratch_size);

    // Reference
    ggml_mmvq_q4k(W_dev, x_dev, y_ref, M, N, K, scratch, scratch_size, nullptr);
    // Tested kernel
    mmq_q4k(W_dev, x_dev, y_test, M, N, K, scratch, scratch_size, nullptr);
    cudaDeviceSynchronize();

    std::vector<half> h_ref(static_cast<size_t>(M) * N);
    std::vector<half> h_test(static_cast<size_t>(M) * N);
    cudaMemcpy(h_ref.data(), y_ref, h_ref.size() * sizeof(half), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_test.data(), y_test, h_test.size() * sizeof(half), cudaMemcpyDeviceToHost);

    float max_abs = 0.0f, max_rel = 0.0f;
    int worst = 0;
    for (size_t i = 0; i < h_ref.size(); ++i) {
        float vr = __half2float(h_ref[i]);
        float vt = __half2float(h_test[i]);
        float ae = std::abs(vr - vt);
        float re = std::abs(vr) > 1e-6f ? ae / std::abs(vr) : ae;
        if (ae > max_abs) {
            max_abs = ae;
            max_rel = re;
            worst = static_cast<int>(i);
        }
    }
    printf("[mmq_q4k M=%d N=%d K=%d] max_abs=%.5f max_rel=%.4f%% at idx=%d (ref=%.4f test=%.4f)\n",
           M, N, K, max_abs, max_rel * 100.0f, worst,
           __half2float(h_ref[worst]), __half2float(h_test[worst]));

    // dp4a accumulation is order-independent at FP32. The only error is FP16
    // store rounding at the output magnitude — which scales with K. Relative
    // error is the meaningful gate.
    EXPECT_LT(max_rel, 0.005f) << "rel error too large";  // 0.5 %, FP16 ulp territory

    cudaFree(W_dev);
    cudaFree(x_dev);
    cudaFree(y_ref);
    cudaFree(y_test);
    cudaFree(scratch);
}

}  // namespace

TEST(MmqQ4K, Small_M32_N64_K256) { compare(32, 64, 256, 0xa1); }
TEST(MmqQ4K, Small_M32_N128_K512) { compare(32, 128, 512, 0xa2); }
TEST(MmqQ4K, Mid_M64_N128_K1024) { compare(64, 128, 1024, 0xa3); }
TEST(MmqQ4K, Mid_M128_N256_K1024) { compare(128, 256, 1024, 0xa4); }
TEST(MmqQ4K, Prefill_M256_N512_K2560) { compare(256, 512, 2560, 0xa5); }
TEST(MmqQ4K, Prefill_M512_N512_K5120) { compare(512, 512, 5120, 0xa6); }
// Non-aligned M / N — exercise the bounds-check branches.
TEST(MmqQ4K, Pad_M33_N65_K256) { compare(33, 65, 256, 0xa7); }
TEST(MmqQ4K, Pad_M50_N100_K512) { compare(50, 100, 512, 0xa8); }

}  // namespace imp
