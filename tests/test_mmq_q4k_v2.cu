// Unit tests for mmq_q4k v2 Phase 1a — eff_scale / eff_min precompute kernel.
// Validates that the GPU kernel produces bit-equivalent FP16 outputs against
// a CPU reference using the same 6-bit unpack from scales[12].

#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "compute/mmq_q4k_v2.h"

namespace imp {

namespace {

// Random Q4_K super-blocks: random nibbles in qs, random 6-bit values in scales,
// random small positive halfs for d and dmin. Matches the spec layout.
std::vector<uint8_t> make_random_q4k(int N, int K, unsigned seed) {
    const int blocks = K / 256;
    std::vector<uint8_t> h(static_cast<size_t>(N) * blocks * 144);
    std::srand(seed);
    for (int row = 0; row < N; ++row) {
        for (int b = 0; b < blocks; ++b) {
            uint8_t* bp = h.data() + (static_cast<size_t>(row) * blocks + b) * 144;
            half d = __float2half(0.005f + 0.01f * (std::rand() % 100) / 100.0f);
            half dmin = __float2half(0.001f + 0.005f * (std::rand() % 100) / 100.0f);
            std::memcpy(bp + 0, &d, 2);
            std::memcpy(bp + 2, &dmin, 2);
            for (int i = 0; i < 12; ++i) bp[4 + i] = static_cast<uint8_t>(std::rand() & 0xFF);
            for (int i = 0; i < 128; ++i) bp[16 + i] = static_cast<uint8_t>(std::rand() & 0xFF);
        }
    }
    return h;
}

// CPU reference — same unpack as the GPU device function. Returns sc[8] and m[8]
// (0..63 each) for one super-block's scales[12].
void cpu_unpack_q4k_scales_mins(const uint8_t* scales12, uint8_t sc_out[8],
                                uint8_t m_out[8]) {
    const uint16_t* scales = reinterpret_cast<const uint16_t*>(scales12);
    for (int bo_step = 0; bo_step < 4; ++bo_step) {
        const int bq8_offset = 2 * bo_step;
        uint16_t aux[2];
        const int j = bo_step;
        if (j < 2) {
            aux[0] = scales[j + 0] & 0x3f3f;
            aux[1] = scales[j + 2] & 0x3f3f;
        } else {
            aux[0] = ((scales[j + 2] >> 0) & 0x0f0f) | ((scales[j - 2] & 0xc0c0) >> 2);
            aux[1] = ((scales[j + 2] >> 4) & 0x0f0f) | ((scales[j - 0] & 0xc0c0) >> 2);
        }
        const uint8_t* sc = reinterpret_cast<const uint8_t*>(aux);
        const uint8_t* m = sc + 2;
        sc_out[bq8_offset + 0] = sc[0];
        sc_out[bq8_offset + 1] = sc[1];
        m_out [bq8_offset + 0] = m[0];
        m_out [bq8_offset + 1] = m[1];
    }
}

void run_check(int N, int K, unsigned seed) {
    ASSERT_EQ(K % 256, 0);
    auto w_host = make_random_q4k(N, K, seed);
    void* w_dev = nullptr;
    cudaMalloc(&w_dev, w_host.size());
    cudaMemcpy(w_dev, w_host.data(), w_host.size(), cudaMemcpyHostToDevice);

    const size_t out_bytes = q4k_eff_scale_bytes(N, K);
    half *eff_scale_dev = nullptr, *eff_min_dev = nullptr;
    cudaMalloc(&eff_scale_dev, out_bytes);
    cudaMalloc(&eff_min_dev, out_bytes);
    cudaMemset(eff_scale_dev, 0, out_bytes);
    cudaMemset(eff_min_dev, 0, out_bytes);

    q4k_precompute_eff_scales(w_dev, eff_scale_dev, eff_min_dev, N, K, nullptr);
    cudaDeviceSynchronize();

    const int K_blocks = K / 256;
    std::vector<half> h_scale(static_cast<size_t>(N) * K_blocks * 8);
    std::vector<half> h_min  (static_cast<size_t>(N) * K_blocks * 8);
    cudaMemcpy(h_scale.data(), eff_scale_dev, out_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_min.data(),   eff_min_dev,   out_bytes, cudaMemcpyDeviceToHost);

    // CPU reference: same affine math (d*sc, dmin*m), same FP16 rounding.
    int mismatches = 0;
    float max_abs = 0.0f;
    int worst_n = 0, worst_k = 0;
    for (int n = 0; n < N; ++n) {
        for (int kbx = 0; kbx < K_blocks; ++kbx) {
            const uint8_t* sb =
                w_host.data() + (static_cast<size_t>(n) * K_blocks + kbx) * 144;
            half d, dmin;
            std::memcpy(&d, sb, 2);
            std::memcpy(&dmin, sb + 2, 2);
            uint8_t sc[8], m[8];
            cpu_unpack_q4k_scales_mins(sb + 4, sc, m);
            const float d_f = __half2float(d);
            const float dmin_f = __half2float(dmin);
            for (int i = 0; i < 8; ++i) {
                half ref_scale = __float2half(d_f    * static_cast<float>(sc[i]));
                half ref_min   = __float2half(dmin_f * static_cast<float>(m[i]));
                const size_t off = (static_cast<size_t>(n) * K_blocks + kbx) * 8 + i;
                if (__half2float(h_scale[off]) != __half2float(ref_scale)) {
                    ++mismatches;
                    float diff = std::abs(__half2float(h_scale[off]) -
                                          __half2float(ref_scale));
                    if (diff > max_abs) {
                        max_abs = diff;
                        worst_n = n;
                        worst_k = kbx * 8 + i;
                    }
                }
                if (__half2float(h_min[off]) != __half2float(ref_min)) {
                    ++mismatches;
                    float diff = std::abs(__half2float(h_min[off]) -
                                          __half2float(ref_min));
                    if (diff > max_abs) {
                        max_abs = diff;
                        worst_n = n;
                        worst_k = kbx * 8 + i;
                    }
                }
            }
        }
    }

    printf("[mmq_q4k_v2 N=%d K=%d] mismatches=%d max_abs=%.6g (at n=%d k=%d)\n",
           N, K, mismatches, max_abs, worst_n, worst_k);
    EXPECT_EQ(mismatches, 0) << "GPU != CPU reference";

    cudaFree(w_dev);
    cudaFree(eff_scale_dev);
    cudaFree(eff_min_dev);
}

}  // namespace

TEST(MmqQ4KV2Scales, Small_N4_K256)   { run_check(4, 256, 0xb1); }
TEST(MmqQ4KV2Scales, Small_N32_K512)  { run_check(32, 512, 0xb2); }
TEST(MmqQ4KV2Scales, Mid_N128_K1024)  { run_check(128, 1024, 0xb3); }
TEST(MmqQ4KV2Scales, Mid_N512_K2560)  { run_check(512, 2560, 0xb4); }
TEST(MmqQ4KV2Scales, Large_N5120_K5120) { run_check(5120, 5120, 0xb5); }
// Non-aligned N to make sure the 1-thread-per-super-block grid handles tails.
TEST(MmqQ4KV2Scales, Pad_N33_K256)    { run_check(33, 256, 0xb6); }
TEST(MmqQ4KV2Scales, Pad_N100_K1280)  { run_check(100, 1280, 0xb7); }

}  // namespace imp
