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

#include "compute/mmq_q4k.h"
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

// ---------------------------------------------------------------------------
// Phase 1b: layout permutation tests
// ---------------------------------------------------------------------------

namespace {

// Read a Q4 nibble from canonical qs[128] for element index i in [0, 256).
uint8_t canonical_q4(const uint8_t* qs, int i) {
    const int qs_byte = (i / 64) * 32 + (i % 32);
    const int use_high = (i / 32) & 1;
    const uint8_t packed = qs[qs_byte];
    return use_high ? ((packed >> 4) & 0xF) : (packed & 0xF);
}

// Read a Q4 nibble from the permuted eff_q4 layout for element index i in
// [0, 256) within one super-block.
uint8_t permuted_q4(const uint8_t* eff_q4_super, int i) {
    const int sub = i / 32;
    const int pos = i % 32;
    const int byte_idx = pos / 2;
    const int high = pos & 1;  // low nibble = even pos, high = odd
    const uint8_t b = eff_q4_super[sub * 16 + byte_idx];
    return high ? ((b >> 4) & 0xF) : (b & 0xF);
}

void run_permute_check(int N, int K, unsigned seed) {
    ASSERT_EQ(K % 256, 0);
    auto w_host = make_random_q4k(N, K, seed);
    void* w_dev = nullptr;
    cudaMalloc(&w_dev, w_host.size());
    cudaMemcpy(w_dev, w_host.data(), w_host.size(), cudaMemcpyHostToDevice);

    const size_t out_bytes = q4k_eff_q4_bytes(N, K);
    uint8_t* eff_q4_dev = nullptr;
    cudaMalloc(&eff_q4_dev, out_bytes);
    cudaMemset(eff_q4_dev, 0, out_bytes);

    q4k_permute_to_v2_layout(w_dev, eff_q4_dev, N, K, nullptr);
    cudaDeviceSynchronize();

    std::vector<uint8_t> h_eff_q4(out_bytes);
    cudaMemcpy(h_eff_q4.data(), eff_q4_dev, out_bytes, cudaMemcpyDeviceToHost);

    // Test 1: every quant value reads back identically from both layouts.
    const int K_blocks = K / 256;
    int nibble_mismatches = 0;
    int first_n = -1, first_kbx = -1, first_i = -1;
    int can_val = 0, perm_val = 0;
    for (int n = 0; n < N; ++n) {
        for (int kbx = 0; kbx < K_blocks; ++kbx) {
            const uint8_t* canon_qs =
                w_host.data() + (static_cast<size_t>(n) * K_blocks + kbx) * 144 + 16;
            const uint8_t* perm_super =
                h_eff_q4.data() + (static_cast<size_t>(n) * K_blocks * 8 + kbx * 8) * 16;
            for (int i = 0; i < 256; ++i) {
                uint8_t cv = canonical_q4(canon_qs, i);
                uint8_t pv = permuted_q4(perm_super, i);
                if (cv != pv) {
                    if (nibble_mismatches == 0) {
                        first_n = n;
                        first_kbx = kbx;
                        first_i = i;
                        can_val = cv;
                        perm_val = pv;
                    }
                    ++nibble_mismatches;
                }
            }
        }
    }
    printf("[mmq_q4k_v2 permute N=%d K=%d] nibble_mismatches=%d", N, K, nibble_mismatches);
    if (nibble_mismatches > 0)
        printf(" first @ (n=%d kbx=%d i=%d): canon=%d permuted=%d", first_n, first_kbx,
               first_i, can_val, perm_val);
    printf("\n");
    EXPECT_EQ(nibble_mismatches, 0) << "permuted layout does not preserve nibble values";

    cudaFree(w_dev);
    cudaFree(eff_q4_dev);
}

}  // namespace

TEST(MmqQ4KV2Permute, Small_N4_K256)   { run_permute_check(4, 256, 0xc1); }
TEST(MmqQ4KV2Permute, Small_N32_K512)  { run_permute_check(32, 512, 0xc2); }
TEST(MmqQ4KV2Permute, Mid_N128_K1024)  { run_permute_check(128, 1024, 0xc3); }
TEST(MmqQ4KV2Permute, Mid_N512_K2560)  { run_permute_check(512, 2560, 0xc4); }
TEST(MmqQ4KV2Permute, Large_N5120_K5120) { run_permute_check(5120, 5120, 0xc5); }
TEST(MmqQ4KV2Permute, Pad_N33_K256)    { run_permute_check(33, 256, 0xc6); }

// ---------------------------------------------------------------------------
// Phase 2: HMMA kernel correctness test
//
// End-to-end check: build a Q4_K W, run Phase 1a + Phase 1b, then the v2 HMMA
// kernel, and compare against the v1 dp4a tiled kernel which is already
// validated against ggml_mmvq_q4k. Bit-exact equality is not expected
// (HMMA FP32 accumulation vs chained dp4a int math takes different rounding
// paths), so we accept a small relative tolerance.
// ---------------------------------------------------------------------------

namespace {

void fill_random_fp16(std::vector<half>& v, unsigned seed) {
    std::srand(seed);
    for (auto& x : v) {
        x = __float2half((std::rand() / static_cast<float>(RAND_MAX) - 0.5f) * 2.0f);
    }
}

// CPU reference: dequant each Q4_K block on the host and compute
// y[m, n] = sum_k x[m, k] * W_dequant[n, k] in FP32. Slow but ground truth.
std::vector<float> cpu_reference_gemm(const std::vector<uint8_t>& h_W,
                                      const std::vector<half>& h_x, int M, int N,
                                      int K) {
    const int K_blocks = K / 256;
    std::vector<float> y(static_cast<size_t>(M) * N, 0.0f);
    // Dequant W on host into FP32 [N, K]
    std::vector<float> W_dq(static_cast<size_t>(N) * K, 0.0f);
    for (int n = 0; n < N; ++n) {
        for (int kbx = 0; kbx < K_blocks; ++kbx) {
            const uint8_t* sb =
                h_W.data() + (static_cast<size_t>(n) * K_blocks + kbx) * 144;
            half d, dmin;
            std::memcpy(&d, sb, 2);
            std::memcpy(&dmin, sb + 2, 2);
            uint8_t sc[8], m[8];
            cpu_unpack_q4k_scales_mins(sb + 4, sc, m);
            const float df = __half2float(d);
            const float dmf = __half2float(dmin);
            const uint8_t* qs = sb + 16;
            // Sub-block s ∈ [0, 8). qs bytes [(s>>1)*32 .. (s>>1)*32 + 32)
            // hold sub-block s (low nibbles if s even, high if s odd).
            for (int s = 0; s < 8; ++s) {
                const float eff_scale = df * sc[s];
                const float eff_min = dmf * m[s];
                const int byte_base = (s >> 1) * 32;
                const bool use_high = (s & 1) != 0;
                for (int k_in_sub = 0; k_in_sub < 32; ++k_in_sub) {
                    uint8_t byte = qs[byte_base + k_in_sub];
                    uint8_t nib =
                        use_high ? ((byte >> 4) & 0xF) : (byte & 0xF);
                    int k_global = kbx * 256 + s * 32 + k_in_sub;
                    W_dq[(size_t)n * K + k_global] = nib * eff_scale - eff_min;
                }
            }
        }
    }
    // Matmul
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                acc += __half2float(h_x[(size_t)m * K + k]) *
                       W_dq[(size_t)n * K + k];
            }
            y[(size_t)m * N + n] = acc;
        }
    }
    return y;
}

void cmp_to_ref(const char* label, const std::vector<half>& h_y,
                const std::vector<float>& ref, int M, int N, float abs_tol,
                float rel_tol, int* mismatches_out, float* max_abs_out,
                float* max_rel_out) {
    int mismatches = 0;
    float max_abs = 0.0f, max_rel = 0.0f;
    int worst_m = 0, worst_n = 0;
    for (int i = 0; i < M * N; ++i) {
        float a = __half2float(h_y[i]);
        float b = ref[i];
        float d = std::abs(a - b);
        float scale = std::max(std::abs(a), std::abs(b)) + 1e-3f;
        float rel = d / scale;
        if (d > abs_tol && rel > rel_tol) {
            ++mismatches;
            if (d > max_abs) {
                max_abs = d;
                worst_m = i / N;
                worst_n = i % N;
            }
        }
        if (rel > max_rel) max_rel = rel;
    }
    printf("  %s vs CPU ref: mismatches=%d max_abs=%.4f max_rel=%.4f "
           "(worst @ m=%d n=%d)\n",
           label, mismatches, max_abs, max_rel, worst_m, worst_n);
    if (mismatches_out) *mismatches_out = mismatches;
    if (max_abs_out) *max_abs_out = max_abs;
    if (max_rel_out) *max_rel_out = max_rel;
}

// Validates v2 HMMA against a CPU FP32 reference. v1 (dp4a) introduces Q8_1
// quantization error on the activations (per-element abs ~0.5 at K=256) so it
// is reported for context but not gated.
void run_v2_vs_v1_check(int M, int N, int K, unsigned seed,
                        float rel_tol = 0.02f, float abs_tol = 0.15f) {
    // abs_tol picked to absorb FP16-dequant rounding: the scaffold dequants
    // weights to FP16 in sB before the MMA, so per-element |W_fp16 - W_fp32|
    // ≈ |W| · 2^-10. Random-sign accumulation over K elements stays ≲ 0.1.
    ASSERT_EQ(K % 256, 0);
    auto h_W = make_random_q4k(N, K, seed);
    std::vector<half> h_x(static_cast<size_t>(M) * K);
    fill_random_fp16(h_x, seed ^ 0xa5a5);

    void* W_dev = nullptr;
    half* x_dev = nullptr;
    half* y_v1 = nullptr;
    half* y_v2 = nullptr;
    void* scratch_v1 = nullptr;
    uint8_t* eff_q4 = nullptr;
    half* eff_scale = nullptr;
    half* eff_min = nullptr;
    const size_t bytes_y = static_cast<size_t>(M) * N * sizeof(half);
    const size_t scratch_bytes = mmq_q4k_scratch_bytes(M, K);

    cudaMalloc(&W_dev, h_W.size());
    cudaMalloc(&x_dev, h_x.size() * sizeof(half));
    cudaMalloc(&y_v1, bytes_y);
    cudaMalloc(&y_v2, bytes_y);
    cudaMalloc(&scratch_v1, scratch_bytes);
    cudaMalloc(&eff_q4, q4k_eff_q4_bytes(N, K));
    cudaMalloc(&eff_scale, q4k_eff_scale_bytes(N, K));
    cudaMalloc(&eff_min, q4k_eff_scale_bytes(N, K));

    cudaMemcpy(W_dev, h_W.data(), h_W.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(x_dev, h_x.data(), h_x.size() * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemset(y_v1, 0, bytes_y);
    cudaMemset(y_v2, 0, bytes_y);

    q4k_precompute_eff_scales(W_dev, eff_scale, eff_min, N, K, nullptr);
    q4k_permute_to_v2_layout(W_dev, eff_q4, N, K, nullptr);

    mmq_q4k(W_dev, x_dev, y_v1, M, N, K, scratch_v1, scratch_bytes, nullptr);
    mmq_q4k_v2(x_dev, eff_q4, eff_scale, eff_min, y_v2, M, N, K, nullptr);
    cudaDeviceSynchronize();

    std::vector<half> h_y1(static_cast<size_t>(M) * N);
    std::vector<half> h_y2(static_cast<size_t>(M) * N);
    cudaMemcpy(h_y1.data(), y_v1, bytes_y, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_y2.data(), y_v2, bytes_y, cudaMemcpyDeviceToHost);

    auto ref = cpu_reference_gemm(h_W, h_x, M, N, K);
    printf("[M=%d N=%d K=%d] ref[0,0]=%.4f v1[0,0]=%.4f v2[0,0]=%.4f\n", M, N,
           K, ref[0], __half2float(h_y1[0]), __half2float(h_y2[0]));

    int m1, m2;
    float ma1, ma2, mr1, mr2;
    cmp_to_ref("v1", h_y1, ref, M, N, /*abs_tol=*/2.0f, /*rel_tol=*/0.10f, &m1,
               &ma1, &mr1);  // v1 carries Q8_1 quant error
    cmp_to_ref("v2", h_y2, ref, M, N, abs_tol, rel_tol, &m2, &ma2, &mr2);

    EXPECT_EQ(m2, 0) << "v2 HMMA kernel diverges from CPU FP32 reference";

    cudaFree(W_dev);
    cudaFree(x_dev);
    cudaFree(y_v1);
    cudaFree(y_v2);
    cudaFree(scratch_v1);
    cudaFree(eff_q4);
    cudaFree(eff_scale);
    cudaFree(eff_min);
}

}  // namespace

TEST(MmqQ4KV2HMMA, AlignedTile_M64_N64_K256)     { run_v2_vs_v1_check(64, 64, 256, 0xd1); }
TEST(MmqQ4KV2HMMA, MultiTile_M128_N128_K512)     { run_v2_vs_v1_check(128, 128, 512, 0xd2); }
TEST(MmqQ4KV2HMMA, NonMultipleM_M40_N64_K256)    { run_v2_vs_v1_check(40, 64, 256, 0xd3); }
TEST(MmqQ4KV2HMMA, NonMultipleN_M64_N96_K512)    { run_v2_vs_v1_check(64, 96, 512, 0xd4); }
TEST(MmqQ4KV2HMMA, LongK_M64_N64_K2560)          { run_v2_vs_v1_check(64, 64, 2560, 0xd5); }
TEST(MmqQ4KV2HMMA, Realistic_M256_N512_K1024)    { run_v2_vs_v1_check(256, 512, 1024, 0xd6); }

// Cover the BN=128 dispatch branch — large enough blocks_bn128 to trigger it.
TEST(MmqQ4KV2HMMA, Bn128_M512_N256_K512) {
    run_v2_vs_v1_check(512, 256, 512, 0xd7, /*rel_tol=*/0.02f, /*abs_tol=*/0.15f);
}
TEST(MmqQ4KV2HMMA, Bn128_M256_N1024_K512) {
    run_v2_vs_v1_check(256, 1024, 512, 0xd8, 0.02f, 0.15f);
}
TEST(MmqQ4KV2HMMA, Bn128_NonMultiple_M200_N300_K768) {
    run_v2_vs_v1_check(200, 300, 768, 0xd9, 0.02f, 0.15f);
}

// ---------------------------------------------------------------------------
// Phase 6: Q5_K v2 — kernel + Phase 1a/1b correctness vs CPU FP32 reference
// ---------------------------------------------------------------------------

namespace {

std::vector<uint8_t> make_random_q5k(int N, int K, unsigned seed) {
    const int blocks = K / 256;
    std::vector<uint8_t> h(static_cast<size_t>(N) * blocks * 176);
    std::srand(seed);
    for (int row = 0; row < N; ++row) {
        for (int b = 0; b < blocks; ++b) {
            uint8_t* bp = h.data() + (static_cast<size_t>(row) * blocks + b) * 176;
            half d = __float2half(0.005f + 0.01f * (std::rand() % 100) / 100.0f);
            half dmin = __float2half(0.001f + 0.005f * (std::rand() % 100) / 100.0f);
            std::memcpy(bp + 0, &d, 2);
            std::memcpy(bp + 2, &dmin, 2);
            for (int i = 0; i < 12; ++i) bp[4 + i] = static_cast<uint8_t>(std::rand() & 0xFF);
            // qh[32] at [16, 48)
            for (int i = 0; i < 32; ++i) bp[16 + i] = static_cast<uint8_t>(std::rand() & 0xFF);
            // qs[128] at [48, 176)
            for (int i = 0; i < 128; ++i) bp[48 + i] = static_cast<uint8_t>(std::rand() & 0xFF);
        }
    }
    return h;
}

std::vector<float> cpu_reference_q5k_gemm(const std::vector<uint8_t>& h_W,
                                          const std::vector<half>& h_x, int M,
                                          int N, int K) {
    const int K_blocks = K / 256;
    std::vector<float> y(static_cast<size_t>(M) * N, 0.0f);
    std::vector<float> W_dq(static_cast<size_t>(N) * K, 0.0f);
    for (int n = 0; n < N; ++n) {
        for (int kbx = 0; kbx < K_blocks; ++kbx) {
            const uint8_t* sb =
                h_W.data() + (static_cast<size_t>(n) * K_blocks + kbx) * 176;
            half d, dmin;
            std::memcpy(&d, sb, 2);
            std::memcpy(&dmin, sb + 2, 2);
            uint8_t sc[8], m[8];
            cpu_unpack_q4k_scales_mins(sb + 4, sc, m);  // Q5_K shares Q4_K scales layout
            const float df = __half2float(d);
            const float dmf = __half2float(dmin);
            const uint8_t* qh = sb + 16;
            const uint8_t* qs = sb + 48;
            for (int s = 0; s < 8; ++s) {
                const float eff_scale = df * sc[s];
                const float eff_min = dmf * m[s];
                const int byte_base = (s >> 1) * 32;
                const bool use_high = (s & 1) != 0;
                for (int k_in_sub = 0; k_in_sub < 32; ++k_in_sub) {
                    uint8_t byte = qs[byte_base + k_in_sub];
                    uint8_t nib = use_high ? ((byte >> 4) & 0xF) : (byte & 0xF);
                    int k_global_in_super = s * 32 + k_in_sub;
                    int qh_byte = k_global_in_super / 8;
                    int qh_bit = k_global_in_super & 7;
                    int hi = (qh[qh_byte] >> qh_bit) & 1;
                    int q5 = nib | (hi << 4);
                    int k_global = kbx * 256 + k_global_in_super;
                    W_dq[(size_t)n * K + k_global] = q5 * eff_scale - eff_min;
                }
            }
        }
    }
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                acc += __half2float(h_x[(size_t)m * K + k]) *
                       W_dq[(size_t)n * K + k];
            }
            y[(size_t)m * N + n] = acc;
        }
    }
    return y;
}

void run_q5k_v2_check(int M, int N, int K, unsigned seed,
                      float rel_tol = 0.02f, float abs_tol = 0.20f) {
    ASSERT_EQ(K % 256, 0);
    auto h_W = make_random_q5k(N, K, seed);
    std::vector<half> h_x(static_cast<size_t>(M) * K);
    fill_random_fp16(h_x, seed ^ 0xbeef);

    void* W_dev = nullptr;
    half* x_dev = nullptr;
    half* y_dev = nullptr;
    uint8_t* eff_ql = nullptr;
    uint8_t* eff_qh = nullptr;
    half* eff_scale = nullptr;
    half* eff_min = nullptr;
    const size_t bytes_y = static_cast<size_t>(M) * N * sizeof(half);

    cudaMalloc(&W_dev, h_W.size());
    cudaMalloc(&x_dev, h_x.size() * sizeof(half));
    cudaMalloc(&y_dev, bytes_y);
    cudaMalloc(&eff_ql, q5k_eff_ql_bytes(N, K));
    cudaMalloc(&eff_qh, q5k_eff_qh_bytes(N, K));
    cudaMalloc(&eff_scale, q4k_eff_scale_bytes(N, K));
    cudaMalloc(&eff_min, q4k_eff_scale_bytes(N, K));

    cudaMemcpy(W_dev, h_W.data(), h_W.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(x_dev, h_x.data(), h_x.size() * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemset(y_dev, 0, bytes_y);

    q5k_precompute_eff_scales(W_dev, eff_scale, eff_min, N, K, nullptr);
    q5k_permute_to_v2_layout(W_dev, eff_ql, eff_qh, N, K, nullptr);
    mmq_q5k_v2(x_dev, eff_ql, eff_qh, eff_scale, eff_min, y_dev, M, N, K, nullptr);
    cudaDeviceSynchronize();

    std::vector<half> h_y(static_cast<size_t>(M) * N);
    cudaMemcpy(h_y.data(), y_dev, bytes_y, cudaMemcpyDeviceToHost);

    auto ref = cpu_reference_q5k_gemm(h_W, h_x, M, N, K);
    printf("[Q5K v2 M=%d N=%d K=%d] ref[0,0]=%.4f gpu[0,0]=%.4f\n", M, N, K,
           ref[0], __half2float(h_y[0]));
    int m_cnt;
    float ma, mr;
    cmp_to_ref("Q5K_v2", h_y, ref, M, N, abs_tol, rel_tol, &m_cnt, &ma, &mr);
    EXPECT_EQ(m_cnt, 0) << "Q5_K v2 diverges from CPU FP32 reference";

    cudaFree(W_dev);
    cudaFree(x_dev);
    cudaFree(y_dev);
    cudaFree(eff_ql);
    cudaFree(eff_qh);
    cudaFree(eff_scale);
    cudaFree(eff_min);
}

}  // namespace

TEST(MmqQ5KV2HMMA, AlignedTile_M64_N64_K256)     { run_q5k_v2_check(64, 64, 256, 0xe1); }
TEST(MmqQ5KV2HMMA, MultiTile_M128_N128_K512)     { run_q5k_v2_check(128, 128, 512, 0xe2); }
TEST(MmqQ5KV2HMMA, NonMultipleM_M40_N64_K256)    { run_q5k_v2_check(40, 64, 256, 0xe3); }
TEST(MmqQ5KV2HMMA, NonMultipleN_M64_N96_K512)    { run_q5k_v2_check(64, 96, 512, 0xe4); }
TEST(MmqQ5KV2HMMA, LongK_M64_N64_K2560)          { run_q5k_v2_check(64, 64, 2560, 0xe5); }
TEST(MmqQ5KV2HMMA, Realistic_M256_N512_K1024)    { run_q5k_v2_check(256, 512, 1024, 0xe6); }
TEST(MmqQ5KV2HMMA, Bn128_M512_N256_K512)         { run_q5k_v2_check(512, 256, 512, 0xe7); }

// ---------------------------------------------------------------------------
// Phase 6b: Q6_K v2 — kernel + Phase 1 (byte expansion + scale precompute)
// ---------------------------------------------------------------------------

namespace {

std::vector<uint8_t> make_random_q6k(int N, int K, unsigned seed) {
    const int blocks = K / 256;
    std::vector<uint8_t> h(static_cast<size_t>(N) * blocks * 210);
    std::srand(seed);
    for (int row = 0; row < N; ++row) {
        for (int b = 0; b < blocks; ++b) {
            uint8_t* bp = h.data() + (static_cast<size_t>(row) * blocks + b) * 210;
            for (int i = 0; i < 128; ++i) bp[i] = static_cast<uint8_t>(std::rand() & 0xFF);  // ql
            for (int i = 0; i < 64; ++i) bp[128 + i] = static_cast<uint8_t>(std::rand() & 0xFF);  // qh
            for (int i = 0; i < 16; ++i) {
                int8_t s = static_cast<int8_t>((std::rand() % 121) - 60);  // signed [-60, 60]
                std::memcpy(bp + 192 + i, &s, 1);
            }
            half d = __float2half(0.001f + 0.005f * (std::rand() % 100) / 100.0f);
            std::memcpy(bp + 208, &d, 2);
        }
    }
    return h;
}

int cpu_q6k_decode_unsigned(const uint8_t* bp, int i) {
    int group = i >> 7;
    int within = i & 127;
    int quad = within >> 5;
    int l = within & 31;
    int ql_idx = (group << 6) + ((quad & 1) << 5) + l;
    int qh_idx = (group << 5) + l;
    uint8_t ql_byte = bp[ql_idx];
    uint8_t low4 = (quad >= 2) ? ((ql_byte >> 4) & 0xF) : (ql_byte & 0xF);
    uint8_t high2 = (bp[128 + qh_idx] >> (quad * 2)) & 0x3;
    return static_cast<int>((high2 << 4) | low4);
}

std::vector<float> cpu_reference_q6k_gemm(const std::vector<uint8_t>& h_W,
                                          const std::vector<half>& h_x, int M,
                                          int N, int K) {
    const int K_blocks = K / 256;
    std::vector<float> y(static_cast<size_t>(M) * N, 0.0f);
    std::vector<float> W_dq(static_cast<size_t>(N) * K, 0.0f);
    for (int n = 0; n < N; ++n) {
        for (int kbx = 0; kbx < K_blocks; ++kbx) {
            const uint8_t* bp = h_W.data() + (static_cast<size_t>(n) * K_blocks + kbx) * 210;
            const int8_t* scales = reinterpret_cast<const int8_t*>(bp + 192);
            half d_h;
            std::memcpy(&d_h, bp + 208, 2);
            float d = __half2float(d_h);
            for (int i = 0; i < 256; ++i) {
                int q_unsigned = cpu_q6k_decode_unsigned(bp, i);
                int q_signed = q_unsigned - 32;
                int s = i >> 4;  // sub-block of 16
                float w = d * static_cast<float>(scales[s]) * static_cast<float>(q_signed);
                W_dq[(size_t)n * K + kbx * 256 + i] = w;
            }
        }
    }
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                acc += __half2float(h_x[(size_t)m * K + k]) *
                       W_dq[(size_t)n * K + k];
            }
            y[(size_t)m * N + n] = acc;
        }
    }
    return y;
}

void run_q6k_v2_check(int M, int N, int K, unsigned seed,
                      float rel_tol = 0.02f, float abs_tol = 0.30f) {
    ASSERT_EQ(K % 256, 0);
    auto h_W = make_random_q6k(N, K, seed);
    std::vector<half> h_x(static_cast<size_t>(M) * K);
    fill_random_fp16(h_x, seed ^ 0xcafe);

    void* W_dev = nullptr;
    half* x_dev = nullptr;
    half* y_dev = nullptr;
    uint8_t* eff_q6 = nullptr;
    half* eff_scale = nullptr;
    half* eff_min = nullptr;
    const size_t bytes_y = static_cast<size_t>(M) * N * sizeof(half);

    cudaMalloc(&W_dev, h_W.size());
    cudaMalloc(&x_dev, h_x.size() * sizeof(half));
    cudaMalloc(&y_dev, bytes_y);
    cudaMalloc(&eff_q6, q6k_eff_q6_bytes(N, K));
    cudaMalloc(&eff_scale, q6k_eff_scale_bytes(N, K));
    cudaMalloc(&eff_min, q6k_eff_scale_bytes(N, K));

    cudaMemcpy(W_dev, h_W.data(), h_W.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(x_dev, h_x.data(), h_x.size() * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemset(y_dev, 0, bytes_y);

    q6k_prepare_v2_layout(W_dev, eff_q6, eff_scale, eff_min, N, K, nullptr);
    mmq_q6k_v2(x_dev, eff_q6, eff_scale, eff_min, y_dev, M, N, K, nullptr);
    cudaDeviceSynchronize();

    std::vector<half> h_y(static_cast<size_t>(M) * N);
    cudaMemcpy(h_y.data(), y_dev, bytes_y, cudaMemcpyDeviceToHost);

    auto ref = cpu_reference_q6k_gemm(h_W, h_x, M, N, K);
    printf("[Q6K v2 M=%d N=%d K=%d] ref[0,0]=%.4f gpu[0,0]=%.4f\n", M, N, K,
           ref[0], __half2float(h_y[0]));
    int m_cnt;
    float ma, mr;
    cmp_to_ref("Q6K_v2", h_y, ref, M, N, abs_tol, rel_tol, &m_cnt, &ma, &mr);
    EXPECT_EQ(m_cnt, 0) << "Q6_K v2 diverges from CPU FP32 reference";

    cudaFree(W_dev);
    cudaFree(x_dev);
    cudaFree(y_dev);
    cudaFree(eff_q6);
    cudaFree(eff_scale);
    cudaFree(eff_min);
}

}  // namespace

TEST(MmqQ6KV2HMMA, AlignedTile_M64_N64_K256)     { run_q6k_v2_check(64, 64, 256, 0xf1); }
TEST(MmqQ6KV2HMMA, MultiTile_M128_N128_K512)     { run_q6k_v2_check(128, 128, 512, 0xf2); }
TEST(MmqQ6KV2HMMA, NonMultipleM_M40_N64_K256)    { run_q6k_v2_check(40, 64, 256, 0xf3); }
TEST(MmqQ6KV2HMMA, NonMultipleN_M64_N96_K512)    { run_q6k_v2_check(64, 96, 512, 0xf4); }
TEST(MmqQ6KV2HMMA, LongK_M64_N64_K2560)          { run_q6k_v2_check(64, 64, 2560, 0xf5); }
TEST(MmqQ6KV2HMMA, Realistic_M256_N512_K1024)    { run_q6k_v2_check(256, 512, 1024, 0xf6); }
TEST(MmqQ6KV2HMMA, Bn128_M512_N256_K512)         { run_q6k_v2_check(512, 256, 512, 0xf7); }

}  // namespace imp
