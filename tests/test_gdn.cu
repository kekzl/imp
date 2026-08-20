#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "compute/gdn.h"

#include <vector>
#include <cmath>

namespace imp {
namespace {

// CPU reference: delta rule scan for a single head, single token.
// Mutates h_state in-place, writes y_out.
static void gdn_scan_cpu(const float* V, const float* K, const float* Q, float alpha, float beta_raw,
                         float A_log, float dt_bias, float* h_state, float* y_out, int head_dim,
                         int state_size) {
    float dt_val = alpha + dt_bias;
    dt_val = (dt_val > 20.0f) ? dt_val : logf(1.0f + expf(dt_val));
    float g_t = expf(fmaxf(A_log * dt_val, -20.0f));
    float beta = 1.0f / (1.0f + expf(-fmaxf(fminf(beta_raw, 20.0f), -20.0f)));

    float k_sq = 0, q_sq = 0;
    for (int s = 0; s < state_size; s++) {
        k_sq += K[s] * K[s];
        q_sq += Q[s] * Q[s];
    }
    float k_inv = 1.0f / sqrtf(k_sq + 1e-6f);
    float q_inv = 1.0f / sqrtf(q_sq + 1e-6f);

    std::vector<float> k_n(state_size), q_n(state_size);
    for (int s = 0; s < state_size; s++) {
        k_n[s] = K[s] * k_inv;
        q_n[s] = Q[s] * q_inv;
    }
    float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

    for (int d = 0; d < head_dim; d++) {
        float kv = 0;
        for (int s = 0; s < state_size; s++)
            kv += h_state[s * head_dim + d] * k_n[s];
        float delta = (V[d] - g_t * kv) * beta;
        float y_partial = 0;
        for (int s = 0; s < state_size; s++) {
            float h_new = g_t * h_state[s * head_dim + d] + k_n[s] * delta;
            h_state[s * head_dim + d] = h_new;
            y_partial += h_new * q_n[s];
        }
        y_out[d] = y_partial * scale;
    }
}

// =========================================================================
// Test 1: Single token CPU vs GPU
// =========================================================================

TEST(GDNScanTest, SingleTokenCPUvsGPU) {
    constexpr int n_heads = 4, head_dim = 128, state_size = 128, n_groups = 4;
    constexpr int inner = n_heads * head_dim;
    constexpr int BC_size = n_groups * state_size;

    srand(42);
    std::vector<float> h_V(inner), h_K(BC_size), h_Q(BC_size);
    std::vector<float> h_alpha(n_heads), h_beta(n_heads);
    std::vector<float> h_A_log(n_heads, -0.5f), h_dt_bias(n_heads, 0.5f);
    std::vector<float> state_cpu(n_heads * state_size * head_dim, 0.0f);

    for (auto& v : h_V)
        v = (rand() % 200 - 100) / 100.0f;
    for (auto& v : h_K)
        v = (rand() % 200 - 100) / 100.0f;
    for (auto& v : h_Q)
        v = (rand() % 200 - 100) / 100.0f;
    for (int i = 0; i < n_heads; i++) {
        h_alpha[i] = (rand() % 200 - 100) / 100.0f;
        h_beta[i] = (rand() % 200 - 100) / 100.0f;
    }

    // CPU reference
    std::vector<float> y_cpu(inner);
    for (int h = 0; h < n_heads; h++) {
        int g = h % n_groups;
        gdn_scan_cpu(h_V.data() + h * head_dim, h_K.data() + g * state_size, h_Q.data() + g * state_size,
                     h_alpha[h], h_beta[h], h_A_log[h], h_dt_bias[h],
                     state_cpu.data() + h * state_size * head_dim, y_cpu.data() + h * head_dim, head_dim,
                     state_size);
    }

    // GPU
    float *d_V, *d_K, *d_Q, *d_A, *d_dt, *d_state;
    half *d_alpha, *d_beta, *d_y;
    cudaMalloc(&d_V, inner * sizeof(float));
    cudaMalloc(&d_K, BC_size * sizeof(float));
    cudaMalloc(&d_Q, BC_size * sizeof(float));
    cudaMalloc(&d_A, n_heads * sizeof(float));
    cudaMalloc(&d_dt, n_heads * sizeof(float));
    cudaMalloc(&d_state, n_heads * state_size * head_dim * sizeof(float));
    cudaMalloc(&d_alpha, n_heads * sizeof(half));
    cudaMalloc(&d_beta, n_heads * sizeof(half));
    cudaMalloc(&d_y, inner * sizeof(half));

    cudaMemcpy(d_V, h_V.data(), inner * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K.data(), BC_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Q, h_Q.data(), BC_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, h_A_log.data(), n_heads * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dt, h_dt_bias.data(), n_heads * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_state, 0, n_heads * state_size * head_dim * sizeof(float));

    std::vector<half> ha(n_heads), hb(n_heads);
    for (int i = 0; i < n_heads; i++) {
        ha[i] = __float2half(h_alpha[i]);
        hb[i] = __float2half(h_beta[i]);
    }
    cudaMemcpy(d_alpha, ha.data(), n_heads * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, hb.data(), n_heads * sizeof(half), cudaMemcpyHostToDevice);

    gdn_scan_decode_f32(d_V, d_K, d_Q, d_alpha, d_beta, d_A, d_dt, d_state, d_y, nullptr, n_heads, head_dim,
                        state_size, n_groups, nullptr);
    cudaDeviceSynchronize();

    std::vector<half> hy(inner);
    cudaMemcpy(hy.data(), d_y, inner * sizeof(half), cudaMemcpyDeviceToHost);
    float max_err = 0;
    for (int i = 0; i < inner; i++) {
        float err = fabsf(__half2float(hy[i]) - y_cpu[i]);
        max_err = fmaxf(max_err, err);
    }
    EXPECT_LT(max_err, 1e-2f) << "Single-token CPU vs GPU max error too large";

    cudaFree(d_V);
    cudaFree(d_K);
    cudaFree(d_Q);
    cudaFree(d_A);
    cudaFree(d_dt);
    cudaFree(d_state);
    cudaFree(d_alpha);
    cudaFree(d_beta);
    cudaFree(d_y);
}

// =========================================================================
// Test 2: Multi-token sequential — 5 tokens, verify state accumulation
// =========================================================================

TEST(GDNScanTest, MultiTokenSequential) {
    constexpr int n_heads = 4, head_dim = 128, state_size = 128, n_groups = 4;
    constexpr int inner = n_heads * head_dim, BC_size = n_groups * state_size;
    constexpr int n_tok = 5;

    srand(42);
    std::vector<float> all_V(n_tok * inner), all_K(n_tok * BC_size), all_Q(n_tok * BC_size);
    std::vector<float> all_alpha(n_tok * n_heads), all_beta(n_tok * n_heads);
    std::vector<float> h_A_log(n_heads, -0.5f), h_dt_bias(n_heads, 0.5f);
    for (auto& v : all_V)
        v = (rand() % 200 - 100) / 100.0f;
    for (auto& v : all_K)
        v = (rand() % 200 - 100) / 100.0f;
    for (auto& v : all_Q)
        v = (rand() % 200 - 100) / 100.0f;
    for (auto& v : all_alpha)
        v = (rand() % 200 - 100) / 100.0f;
    for (auto& v : all_beta)
        v = (rand() % 200 - 100) / 100.0f;

    // CPU: sequential per-token processing
    std::vector<float> state_cpu(n_heads * state_size * head_dim, 0.0f);
    std::vector<float> y_cpu(inner);
    for (int t = 0; t < n_tok; t++) {
        for (int h = 0; h < n_heads; h++) {
            int g = h % n_groups;
            gdn_scan_cpu(&all_V[t * inner + h * head_dim], &all_K[t * BC_size + g * state_size],
                         &all_Q[t * BC_size + g * state_size], all_alpha[t * n_heads + h],
                         all_beta[t * n_heads + h], h_A_log[h], h_dt_bias[h],
                         &state_cpu[h * state_size * head_dim], y_cpu.data() + h * head_dim, head_dim,
                         state_size);
        }
    }

    // GPU: per-token loop via gdn_scan_decode_f32
    float *d_V, *d_K, *d_Q, *d_A, *d_dt, *d_state;
    half *d_alpha, *d_beta, *d_y;
    cudaMalloc(&d_V, inner * sizeof(float));
    cudaMalloc(&d_K, BC_size * sizeof(float));
    cudaMalloc(&d_Q, BC_size * sizeof(float));
    cudaMalloc(&d_A, n_heads * sizeof(float));
    cudaMalloc(&d_dt, n_heads * sizeof(float));
    cudaMalloc(&d_state, n_heads * state_size * head_dim * sizeof(float));
    cudaMalloc(&d_alpha, n_heads * sizeof(half));
    cudaMalloc(&d_beta, n_heads * sizeof(half));
    cudaMalloc(&d_y, inner * sizeof(half));

    cudaMemcpy(d_A, h_A_log.data(), n_heads * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dt, h_dt_bias.data(), n_heads * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_state, 0, n_heads * state_size * head_dim * sizeof(float));

    for (int t = 0; t < n_tok; t++) {
        cudaMemcpy(d_V, all_V.data() + t * inner, inner * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_K, all_K.data() + t * BC_size, BC_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Q, all_Q.data() + t * BC_size, BC_size * sizeof(float), cudaMemcpyHostToDevice);
        std::vector<half> ha(n_heads), hb(n_heads);
        for (int i = 0; i < n_heads; i++) {
            ha[i] = __float2half(all_alpha[t * n_heads + i]);
            hb[i] = __float2half(all_beta[t * n_heads + i]);
        }
        cudaMemcpy(d_alpha, ha.data(), n_heads * sizeof(half), cudaMemcpyHostToDevice);
        cudaMemcpy(d_beta, hb.data(), n_heads * sizeof(half), cudaMemcpyHostToDevice);
        gdn_scan_decode_f32(d_V, d_K, d_Q, d_alpha, d_beta, d_A, d_dt, d_state, d_y, nullptr, n_heads,
                            head_dim, state_size, n_groups, nullptr);
        cudaDeviceSynchronize();
    }

    std::vector<half> hy(inner);
    cudaMemcpy(hy.data(), d_y, inner * sizeof(half), cudaMemcpyDeviceToHost);
    float max_err = 0;
    for (int i = 0; i < inner; i++)
        max_err = fmaxf(max_err, fabsf(__half2float(hy[i]) - y_cpu[i]));
    EXPECT_LT(max_err, 0.05f) << "Multi-token sequential max error too large";

    cudaFree(d_V);
    cudaFree(d_K);
    cudaFree(d_Q);
    cudaFree(d_A);
    cudaFree(d_dt);
    cudaFree(d_state);
    cudaFree(d_alpha);
    cudaFree(d_beta);
    cudaFree(d_y);
}

// =========================================================================
// Test 3: Fused kernel matches legacy per-token kernel
// =========================================================================

TEST(GDNScanTest, FusedKernelMatchesLegacy) {
    constexpr int n_heads = 4, head_dim = 128, state_size = 128, n_groups = 4;
    constexpr int inner = n_heads * head_dim, BC_size = n_groups * state_size;
    constexpr int n_tok = 3;
    constexpr int conv_channels = BC_size + BC_size + inner;  // Q|K|V layout

    srand(42);
    // Build conv_f32: [n_tok, conv_channels] where each row = [Q(BC), K(BC), V(inner)]
    std::vector<float> conv_f32(n_tok * conv_channels);
    std::vector<float> all_alpha(n_tok * n_heads), all_beta(n_tok * n_heads);
    std::vector<float> h_A_log(n_heads, -0.5f), h_dt_bias(n_heads, 0.5f);
    for (auto& v : conv_f32)
        v = (rand() % 200 - 100) / 100.0f;
    for (auto& v : all_alpha)
        v = (rand() % 200 - 100) / 100.0f;
    for (auto& v : all_beta)
        v = (rand() % 200 - 100) / 100.0f;

    // Convert alpha/beta to FP16
    std::vector<half> h_alpha_h(n_tok * n_heads), h_beta_h(n_tok * n_heads);
    for (int i = 0; i < n_tok * n_heads; i++) {
        h_alpha_h[i] = __float2half(all_alpha[i]);
        h_beta_h[i] = __float2half(all_beta[i]);
    }

    // Allocate GPU buffers
    float *d_conv, *d_A, *d_dt, *d_state1, *d_state2;
    half *d_alpha, *d_beta, *d_y1, *d_y2;
    cudaMalloc(&d_conv, n_tok * conv_channels * sizeof(float));
    cudaMalloc(&d_A, n_heads * sizeof(float));
    cudaMalloc(&d_dt, n_heads * sizeof(float));
    cudaMalloc(&d_state1, n_heads * state_size * head_dim * sizeof(float));
    cudaMalloc(&d_state2, n_heads * state_size * head_dim * sizeof(float));
    cudaMalloc(&d_alpha, n_tok * n_heads * sizeof(half));
    cudaMalloc(&d_beta, n_tok * n_heads * sizeof(half));
    cudaMalloc(&d_y1, n_tok * inner * sizeof(half));
    cudaMalloc(&d_y2, n_tok * inner * sizeof(half));

    cudaMemcpy(d_conv, conv_f32.data(), n_tok * conv_channels * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, h_A_log.data(), n_heads * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dt, h_dt_bias.data(), n_heads * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_alpha, h_alpha_h.data(), n_tok * n_heads * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, h_beta_h.data(), n_tok * n_heads * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemset(d_state1, 0, n_heads * state_size * head_dim * sizeof(float));
    cudaMemset(d_state2, 0, n_heads * state_size * head_dim * sizeof(float));

    // Fused: single kernel for all tokens
    gdn_scan_fused_f32(d_conv, conv_channels, d_alpha, d_beta, d_A, d_dt, d_state1, d_y1, n_tok, n_heads,
                       head_dim, state_size, n_groups, nullptr);
    cudaDeviceSynchronize();

    // Legacy: per-token using gdn_scan_prefill_f32 (which internally loops)
    // We need to decompose conv_f32 into separate Q, K, V arrays per token
    // and use gdn_scan_decode_f32 per token.
    float *d_V, *d_K, *d_Q;
    half *d_a1, *d_b1;
    cudaMalloc(&d_V, inner * sizeof(float));
    cudaMalloc(&d_K, BC_size * sizeof(float));
    cudaMalloc(&d_Q, BC_size * sizeof(float));
    cudaMalloc(&d_a1, n_heads * sizeof(half));
    cudaMalloc(&d_b1, n_heads * sizeof(half));

    for (int t = 0; t < n_tok; t++) {
        const float* row = conv_f32.data() + t * conv_channels;
        // conv layout: [Q(BC_size), K(BC_size), V(inner)]
        cudaMemcpy(d_Q, row, BC_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_K, row + BC_size, BC_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_V, row + 2 * BC_size, inner * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_a1, h_alpha_h.data() + t * n_heads, n_heads * sizeof(half), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b1, h_beta_h.data() + t * n_heads, n_heads * sizeof(half), cudaMemcpyHostToDevice);
        gdn_scan_decode_f32(d_V, d_K, d_Q, d_a1, d_b1, d_A, d_dt, d_state2, d_y2 + t * inner, nullptr,
                            n_heads, head_dim, state_size, n_groups, nullptr);
        cudaDeviceSynchronize();
    }

    // Compare outputs
    std::vector<half> hy1(n_tok * inner), hy2(n_tok * inner);
    cudaMemcpy(hy1.data(), d_y1, n_tok * inner * sizeof(half), cudaMemcpyDeviceToHost);
    cudaMemcpy(hy2.data(), d_y2, n_tok * inner * sizeof(half), cudaMemcpyDeviceToHost);
    float max_err = 0;
    for (int i = 0; i < n_tok * inner; i++)
        max_err = fmaxf(max_err, fabsf(__half2float(hy1[i]) - __half2float(hy2[i])));
    EXPECT_LT(max_err, 1e-2f) << "Fused vs legacy max error too large";

    cudaFree(d_conv);
    cudaFree(d_A);
    cudaFree(d_dt);
    cudaFree(d_state1);
    cudaFree(d_state2);
    cudaFree(d_alpha);
    cudaFree(d_beta);
    cudaFree(d_y1);
    cudaFree(d_y2);
    cudaFree(d_V);
    cudaFree(d_K);
    cudaFree(d_Q);
    cudaFree(d_a1);
    cudaFree(d_b1);
}

// =========================================================================
// Test 3b: Chunk-boundary handoff equivalence
// -------------------------------------------------------------------------
// Phase 1a of the chunkwise SSD scan refactor (design plan retired with
// docs/plans/). Establishes the precondition for Phase 1b:
// splitting a sequential GDN scan at any token boundary, saving the H state
// at the split, and resuming with that saved state must produce bit-
// equivalent output to a single monolithic scan.
//
// This isn't just a sanity test — it's the regression gate for the Phase 1b
// parallel-within-chunk SSD kernel. ANY chunkwise replacement must preserve
// the same chunk-end-state and intra-chunk-y semantics. If this test passes
// for two implementations (sequential and chunkwise), they're functionally
// interchangeable at the dispatch layer.
//
// Setup: n_tok=16 (= 2 chunks of 8), n_heads=4, head_dim=128, state=128.
//   Run A: gdn_scan_fused_f32(tokens[0..16], state=zeros)  → Y_full, state_full
//   Run B: gdn_scan_fused_f32(tokens[0..8],  state=zeros)  → Y_chunk1, state_mid
//          gdn_scan_fused_f32(tokens[8..16], state=state_mid) → Y_chunk2, state_end
// Assert: Y_full[0..8] ≈ Y_chunk1, Y_full[8..16] ≈ Y_chunk2, state_full ≈ state_end.
// =========================================================================

TEST(GDNScanTest, ChunkBoundaryHandoff) {
    constexpr int n_heads = 4, head_dim = 128, state_size = 128, n_groups = 4;
    constexpr int inner = n_heads * head_dim, BC_size = n_groups * state_size;
    constexpr int conv_channels = 2 * BC_size + inner;
    constexpr int n_tok = 16;
    constexpr int chunk = n_tok / 2;

    srand(123);
    std::vector<float> conv_f32(n_tok * conv_channels);
    std::vector<float> all_alpha(n_tok * n_heads), all_beta(n_tok * n_heads);
    std::vector<float> h_A_log(n_heads, -0.5f), h_dt_bias(n_heads, 0.5f);
    for (auto& v : conv_f32)
        v = (rand() % 200 - 100) / 100.0f;
    for (auto& v : all_alpha)
        v = (rand() % 200 - 100) / 100.0f;
    for (auto& v : all_beta)
        v = (rand() % 200 - 100) / 100.0f;

    std::vector<half> h_alpha_h(n_tok * n_heads), h_beta_h(n_tok * n_heads);
    for (int i = 0; i < n_tok * n_heads; i++) {
        h_alpha_h[i] = __float2half(all_alpha[i]);
        h_beta_h[i] = __float2half(all_beta[i]);
    }

    // GPU buffers
    const int state_floats = n_heads * state_size * head_dim;
    float *d_conv, *d_A, *d_dt, *d_state_full, *d_state_mid;
    half *d_alpha, *d_beta, *d_y_full, *d_y_chunk1, *d_y_chunk2;
    cudaMalloc(&d_conv, n_tok * conv_channels * sizeof(float));
    cudaMalloc(&d_A, n_heads * sizeof(float));
    cudaMalloc(&d_dt, n_heads * sizeof(float));
    cudaMalloc(&d_state_full, state_floats * sizeof(float));
    cudaMalloc(&d_state_mid, state_floats * sizeof(float));
    cudaMalloc(&d_alpha, n_tok * n_heads * sizeof(half));
    cudaMalloc(&d_beta, n_tok * n_heads * sizeof(half));
    cudaMalloc(&d_y_full, n_tok * inner * sizeof(half));
    cudaMalloc(&d_y_chunk1, chunk * inner * sizeof(half));
    cudaMalloc(&d_y_chunk2, chunk * inner * sizeof(half));

    cudaMemcpy(d_conv, conv_f32.data(), n_tok * conv_channels * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, h_A_log.data(), n_heads * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dt, h_dt_bias.data(), n_heads * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_alpha, h_alpha_h.data(), n_tok * n_heads * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, h_beta_h.data(), n_tok * n_heads * sizeof(half), cudaMemcpyHostToDevice);

    // === Run A: monolithic full scan ===
    cudaMemset(d_state_full, 0, state_floats * sizeof(float));
    gdn_scan_fused_f32(d_conv, conv_channels, d_alpha, d_beta, d_A, d_dt, d_state_full, d_y_full, n_tok,
                       n_heads, head_dim, state_size, n_groups, nullptr);
    cudaDeviceSynchronize();

    // === Run B: chunked scan with mid-state handoff ===
    cudaMemset(d_state_mid, 0, state_floats * sizeof(float));
    // Chunk 1: tokens [0..chunk), uses zero state. d_state_mid mutates in-place.
    gdn_scan_fused_f32(d_conv, conv_channels, d_alpha, d_beta, d_A, d_dt, d_state_mid, d_y_chunk1, chunk,
                       n_heads, head_dim, state_size, n_groups, nullptr);
    // Chunk 2: tokens [chunk..n_tok), uses d_state_mid carried from chunk 1.
    // Offset alpha/beta by chunk * n_heads; conv_f32 by chunk * conv_channels.
    gdn_scan_fused_f32(d_conv + static_cast<size_t>(chunk) * conv_channels, conv_channels,
                       d_alpha + chunk * n_heads, d_beta + chunk * n_heads, d_A, d_dt, d_state_mid,
                       d_y_chunk2, chunk, n_heads, head_dim, state_size, n_groups, nullptr);
    cudaDeviceSynchronize();

    // === Compare ===
    std::vector<half> y_full(n_tok * inner), y_chunk1(chunk * inner), y_chunk2(chunk * inner);
    std::vector<float> state_full(state_floats), state_mid_host(state_floats);
    cudaMemcpy(y_full.data(), d_y_full, n_tok * inner * sizeof(half), cudaMemcpyDeviceToHost);
    cudaMemcpy(y_chunk1.data(), d_y_chunk1, chunk * inner * sizeof(half), cudaMemcpyDeviceToHost);
    cudaMemcpy(y_chunk2.data(), d_y_chunk2, chunk * inner * sizeof(half), cudaMemcpyDeviceToHost);
    cudaMemcpy(state_full.data(), d_state_full, state_floats * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(state_mid_host.data(), d_state_mid, state_floats * sizeof(float), cudaMemcpyDeviceToHost);

    // Y_full[0..chunk] vs Y_chunk1
    float max_diff_chunk1 = 0;
    for (int i = 0; i < chunk * inner; i++) {
        float d = std::abs(__half2float(y_full[i]) - __half2float(y_chunk1[i]));
        if (d > max_diff_chunk1)
            max_diff_chunk1 = d;
    }
    // Y_full[chunk..n_tok] vs Y_chunk2
    float max_diff_chunk2 = 0;
    for (int i = 0; i < chunk * inner; i++) {
        float d = std::abs(__half2float(y_full[chunk * inner + i]) - __half2float(y_chunk2[i]));
        if (d > max_diff_chunk2)
            max_diff_chunk2 = d;
    }
    // state_full vs state_mid (after chunk 2)
    float max_diff_state = 0;
    for (int i = 0; i < state_floats; i++) {
        float d = std::abs(state_full[i] - state_mid_host[i]);
        if (d > max_diff_state)
            max_diff_state = d;
    }

    std::printf("\n  ChunkBoundaryHandoff: max_diff Y_chunk1 = %.6e\n", max_diff_chunk1);
    std::printf("  ChunkBoundaryHandoff: max_diff Y_chunk2 = %.6e\n", max_diff_chunk2);
    std::printf("  ChunkBoundaryHandoff: max_diff state    = %.6e\n", max_diff_state);

    // FP16 output: max-abs-diff < 1e-3 covers FP16 rounding. State is FP32 → tighter.
    EXPECT_LT(max_diff_chunk1, 1e-3f);
    EXPECT_LT(max_diff_chunk2, 1e-3f);
    EXPECT_LT(max_diff_state, 1e-5f);

    // === Run C: gdn_scan_chunkwise_f32 — the Phase 1b scaffolding ===
    // Currently a chunk-iterating sequential wrapper. Once Phase 1b.1 lands
    // the real SSD matmul kernel, this same test catches any deviation.
    float* d_state_cw;
    half* d_y_cw;
    cudaMalloc(&d_state_cw, state_floats * sizeof(float));
    cudaMalloc(&d_y_cw, n_tok * inner * sizeof(half));
    cudaMemset(d_state_cw, 0, state_floats * sizeof(float));
    gdn_scan_chunkwise_f32(d_conv, conv_channels, d_alpha, d_beta, d_A, d_dt, d_state_cw, d_y_cw, n_tok,
                           n_heads, head_dim, state_size, n_groups, nullptr,
                           /*chunk_size=*/chunk, /*grouped_layout=*/0);
    cudaDeviceSynchronize();

    std::vector<half> y_cw(n_tok * inner);
    std::vector<float> state_cw_host(state_floats);
    cudaMemcpy(y_cw.data(), d_y_cw, n_tok * inner * sizeof(half), cudaMemcpyDeviceToHost);
    cudaMemcpy(state_cw_host.data(), d_state_cw, state_floats * sizeof(float), cudaMemcpyDeviceToHost);

    float max_diff_cw_y = 0;
    for (int i = 0; i < n_tok * inner; i++) {
        float d = std::abs(__half2float(y_full[i]) - __half2float(y_cw[i]));
        if (d > max_diff_cw_y)
            max_diff_cw_y = d;
    }
    float max_diff_cw_state = 0;
    for (int i = 0; i < state_floats; i++) {
        float d = std::abs(state_full[i] - state_cw_host[i]);
        if (d > max_diff_cw_state)
            max_diff_cw_state = d;
    }
    std::printf("  ChunkBoundaryHandoff: max_diff Y_chunkwise = %.6e\n", max_diff_cw_y);
    std::printf("  ChunkBoundaryHandoff: max_diff state_chunkwise = %.6e\n", max_diff_cw_state);
    // Same tolerance budgets as Run B (chunkwise scaffold is a chunk-iterating
    // wrapper around gdn_scan_fused_f32, so for now this should also be bit-
    // exact 0.0; once Phase 1b.1 ships the SSD matmul kernel, FMA-order
    // differences may push the FP16 output toward the 1e-3 budget).
    EXPECT_LT(max_diff_cw_y, 1e-3f);
    EXPECT_LT(max_diff_cw_state, 1e-5f);

    cudaFree(d_conv);
    cudaFree(d_A);
    cudaFree(d_dt);
    cudaFree(d_state_full);
    cudaFree(d_state_mid);
    cudaFree(d_state_cw);
    cudaFree(d_alpha);
    cudaFree(d_beta);
    cudaFree(d_y_full);
    cudaFree(d_y_chunk1);
    cudaFree(d_y_chunk2);
    cudaFree(d_y_cw);
}

// =========================================================================
// Test 3c: Chunkwise SSD prototype matches sequential fused scan
// -------------------------------------------------------------------------
// Phase 1b.1 of the chunkwise SSD scan refactor (design plan retired with
// docs/plans/). Exercises the chunked-shared-memory kernel
// path inside gdn_scan_chunkwise_f32 (chunk_size=64), which is structurally
// distinct from the existing per-token sequential kernel.
//
// Phase 1b's wrapper path (chunk_size != 64) is covered by ChunkBoundary
// Handoff above. This test specifically validates the new
// gdn_scan_chunkwise_kernel<128, 128, 64> at the production GDN shape
// (Qwen 3.5 / 3.6: HD=SS=128). Tolerance budgets from Phase 1a:
//   - FP16 y:   1e-3 max-abs-diff
//   - FP32 H state: 1e-5 max-abs-diff
// =========================================================================

TEST(GDNScanTest, ChunkwiseProtoMatchesFused) {
    constexpr int n_heads = 4, head_dim = 128, state_size = 128, n_groups = 4;
    constexpr int inner = n_heads * head_dim, BC_size = n_groups * state_size;
    constexpr int conv_channels = 2 * BC_size + inner;
    constexpr int n_tok = 128;  // 2 chunks of 64 — exercises chunk-loop + boundary
    constexpr int chunk_size = 64;

    srand(7);
    std::vector<float> conv_f32(n_tok * conv_channels);
    std::vector<float> all_alpha(n_tok * n_heads), all_beta(n_tok * n_heads);
    std::vector<float> h_A_log(n_heads, -0.5f), h_dt_bias(n_heads, 0.5f);
    for (auto& v : conv_f32)
        v = (rand() % 200 - 100) / 100.0f;
    for (auto& v : all_alpha)
        v = (rand() % 200 - 100) / 100.0f;
    for (auto& v : all_beta)
        v = (rand() % 200 - 100) / 100.0f;

    std::vector<half> h_alpha_h(n_tok * n_heads), h_beta_h(n_tok * n_heads);
    for (int i = 0; i < n_tok * n_heads; i++) {
        h_alpha_h[i] = __float2half(all_alpha[i]);
        h_beta_h[i] = __float2half(all_beta[i]);
    }

    const int state_floats = n_heads * state_size * head_dim;
    float *d_conv, *d_A, *d_dt, *d_state_ref, *d_state_cw;
    half *d_alpha, *d_beta, *d_y_ref, *d_y_cw;
    cudaMalloc(&d_conv, n_tok * conv_channels * sizeof(float));
    cudaMalloc(&d_A, n_heads * sizeof(float));
    cudaMalloc(&d_dt, n_heads * sizeof(float));
    cudaMalloc(&d_state_ref, state_floats * sizeof(float));
    cudaMalloc(&d_state_cw, state_floats * sizeof(float));
    cudaMalloc(&d_alpha, n_tok * n_heads * sizeof(half));
    cudaMalloc(&d_beta, n_tok * n_heads * sizeof(half));
    cudaMalloc(&d_y_ref, n_tok * inner * sizeof(half));
    cudaMalloc(&d_y_cw, n_tok * inner * sizeof(half));

    cudaMemcpy(d_conv, conv_f32.data(), n_tok * conv_channels * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, h_A_log.data(), n_heads * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dt, h_dt_bias.data(), n_heads * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_alpha, h_alpha_h.data(), n_tok * n_heads * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, h_beta_h.data(), n_tok * n_heads * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemset(d_state_ref, 0, state_floats * sizeof(float));
    cudaMemset(d_state_cw, 0, state_floats * sizeof(float));

    // Reference: monolithic sequential fused scan.
    gdn_scan_fused_f32(d_conv, conv_channels, d_alpha, d_beta, d_A, d_dt, d_state_ref, d_y_ref, n_tok,
                       n_heads, head_dim, state_size, n_groups, nullptr);
    // Prototype: chunkwise SSD kernel (chunk_size=64 hits the new kernel path).
    gdn_scan_chunkwise_f32(d_conv, conv_channels, d_alpha, d_beta, d_A, d_dt, d_state_cw, d_y_cw, n_tok,
                           n_heads, head_dim, state_size, n_groups, nullptr,
                           /*chunk_size=*/chunk_size, /*grouped_layout=*/0);
    cudaDeviceSynchronize();

    std::vector<half> y_ref(n_tok * inner), y_cw(n_tok * inner);
    std::vector<float> state_ref(state_floats), state_cw(state_floats);
    cudaMemcpy(y_ref.data(), d_y_ref, n_tok * inner * sizeof(half), cudaMemcpyDeviceToHost);
    cudaMemcpy(y_cw.data(), d_y_cw, n_tok * inner * sizeof(half), cudaMemcpyDeviceToHost);
    cudaMemcpy(state_ref.data(), d_state_ref, state_floats * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(state_cw.data(), d_state_cw, state_floats * sizeof(float), cudaMemcpyDeviceToHost);

    float max_diff_y = 0;
    for (int i = 0; i < n_tok * inner; i++) {
        float d = std::abs(__half2float(y_ref[i]) - __half2float(y_cw[i]));
        if (d > max_diff_y)
            max_diff_y = d;
    }
    float max_diff_state = 0;
    for (int i = 0; i < state_floats; i++) {
        float d = std::abs(state_ref[i] - state_cw[i]);
        if (d > max_diff_state)
            max_diff_state = d;
    }

    std::printf("\n  ChunkwiseProto: max_diff Y = %.6e\n", max_diff_y);
    std::printf("  ChunkwiseProto: max_diff state = %.6e\n", max_diff_state);

    // Phase 1a tolerance budgets.
    EXPECT_LT(max_diff_y, 1e-3f);
    EXPECT_LT(max_diff_state, 1e-5f);

    cudaFree(d_conv);
    cudaFree(d_A);
    cudaFree(d_dt);
    cudaFree(d_state_ref);
    cudaFree(d_state_cw);
    cudaFree(d_alpha);
    cudaFree(d_beta);
    cudaFree(d_y_ref);
    cudaFree(d_y_cw);
}

// =========================================================================
// Test 3e: Phase 2a WY-rep prototype matches sequential
// -------------------------------------------------------------------------
// Validates the new `gdn_scan_chunkwise_wy_f32` kernel — factorises the
// chunk-internal sequential dependency into a forward triangular solve +
// matrix-matrix products (WY representation, Yang et al. 2024). Same math
// as the sequential kernel but reorganised; expected to be numerically
// close (FMA-order may differ → ~5e-3 FP16 tolerance, ~1e-2 FP32 state).
// =========================================================================

TEST(GDNScanTest, ChunkwiseWyMatchesFused) {
    constexpr int n_heads = 4, head_dim = 128, state_size = 128, n_groups = 4;
    constexpr int inner = n_heads * head_dim, BC_size = n_groups * state_size;
    constexpr int conv_channels = 2 * BC_size + inner;
    constexpr int n_tok = 64;  // 2 WY chunks of 32 — exercises chunk loop + state propagation

    srand(17);
    std::vector<float> conv_f32(n_tok * conv_channels);
    std::vector<float> all_alpha(n_tok * n_heads), all_beta(n_tok * n_heads);
    std::vector<float> h_A_log(n_heads, -0.5f), h_dt_bias(n_heads, 0.5f);
    for (auto& v : conv_f32)
        v = (rand() % 200 - 100) / 100.0f;
    for (auto& v : all_alpha)
        v = (rand() % 200 - 100) / 100.0f;
    for (auto& v : all_beta)
        v = (rand() % 200 - 100) / 100.0f;

    std::vector<half> h_alpha_h(n_tok * n_heads), h_beta_h(n_tok * n_heads);
    for (int i = 0; i < n_tok * n_heads; i++) {
        h_alpha_h[i] = __float2half(all_alpha[i]);
        h_beta_h[i] = __float2half(all_beta[i]);
    }

    const int state_floats = n_heads * state_size * head_dim;
    float *d_conv, *d_A, *d_dt, *d_state_ref, *d_state_wy;
    half *d_alpha, *d_beta, *d_y_ref, *d_y_wy;
    cudaMalloc(&d_conv, n_tok * conv_channels * sizeof(float));
    cudaMalloc(&d_A, n_heads * sizeof(float));
    cudaMalloc(&d_dt, n_heads * sizeof(float));
    cudaMalloc(&d_state_ref, state_floats * sizeof(float));
    cudaMalloc(&d_state_wy, state_floats * sizeof(float));
    cudaMalloc(&d_alpha, n_tok * n_heads * sizeof(half));
    cudaMalloc(&d_beta, n_tok * n_heads * sizeof(half));
    cudaMalloc(&d_y_ref, n_tok * inner * sizeof(half));
    cudaMalloc(&d_y_wy, n_tok * inner * sizeof(half));

    cudaMemcpy(d_conv, conv_f32.data(), n_tok * conv_channels * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, h_A_log.data(), n_heads * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dt, h_dt_bias.data(), n_heads * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_alpha, h_alpha_h.data(), n_tok * n_heads * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, h_beta_h.data(), n_tok * n_heads * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemset(d_state_ref, 0, state_floats * sizeof(float));
    cudaMemset(d_state_wy, 0, state_floats * sizeof(float));

    gdn_scan_fused_f32(d_conv, conv_channels, d_alpha, d_beta, d_A, d_dt, d_state_ref, d_y_ref, n_tok,
                       n_heads, head_dim, state_size, n_groups, nullptr);
    gdn_scan_chunkwise_wy_f32(d_conv, conv_channels, d_alpha, d_beta, d_A, d_dt, d_state_wy, d_y_wy, n_tok,
                              n_heads, head_dim, state_size, n_groups, nullptr, /*grouped_layout=*/0);
    cudaDeviceSynchronize();

    std::vector<half> y_ref(n_tok * inner), y_wy(n_tok * inner);
    std::vector<float> state_ref(state_floats), state_wy(state_floats);
    cudaMemcpy(y_ref.data(), d_y_ref, n_tok * inner * sizeof(half), cudaMemcpyDeviceToHost);
    cudaMemcpy(y_wy.data(), d_y_wy, n_tok * inner * sizeof(half), cudaMemcpyDeviceToHost);
    cudaMemcpy(state_ref.data(), d_state_ref, state_floats * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(state_wy.data(), d_state_wy, state_floats * sizeof(float), cudaMemcpyDeviceToHost);

    float max_diff_y = 0;
    for (int i = 0; i < n_tok * inner; i++) {
        float diff = std::abs(__half2float(y_ref[i]) - __half2float(y_wy[i]));
        if (diff > max_diff_y)
            max_diff_y = diff;
    }
    float max_diff_state = 0;
    for (int i = 0; i < state_floats; i++) {
        float diff = std::abs(state_ref[i] - state_wy[i]);
        if (diff > max_diff_state)
            max_diff_state = diff;
    }

    std::printf("\n  ChunkwiseWy: max_diff Y = %.6e\n", max_diff_y);
    std::printf("  ChunkwiseWy: max_diff state = %.6e\n", max_diff_state);

    // The WY reformulation differs from the sequential FMA order, but the
    // chunk-internal matmuls + log-space cumulative decay land within FP32
    // reordering noise: ~4e-6 max-abs on Y (FP16 quantisation floor),
    // ~6e-8 max-abs on FP32 state at 2 chunks of 32 tokens. These are well
    // inside Phase 1a's FP16 1e-3 / FP32 1e-5 budgets.
    EXPECT_LT(max_diff_y, 1e-3f);
    EXPECT_LT(max_diff_state, 1e-5f);

    cudaFree(d_conv);
    cudaFree(d_A);
    cudaFree(d_dt);
    cudaFree(d_state_ref);
    cudaFree(d_state_wy);
    cudaFree(d_alpha);
    cudaFree(d_beta);
    cudaFree(d_y_ref);
    cudaFree(d_y_wy);
}

// =========================================================================
// Test 3f: Phase 2b Tensor-Core WY-rep prototype matches sequential
// -------------------------------------------------------------------------
// Validates `gdn_scan_chunkwise_wy_tc_f32` — Phase 2a's WY-rep math with
// the four chunk-internal matmuls (KK, QK, KH, QH) replaced by WMMA TC
// dispatches. FP16 storage of K̃/Q̃/H_0 introduces a small precision drop
// vs Phase 2a's FP32 storage; outputs land within FP16 1e-2 (output) /
// FP32 1e-2 (state) — the looser tolerance covers the FP16 truncation
// on K̃/Q̃ (each ~3-4 mantissa bits lost vs FP32) and the cumulative
// effect on the rank-L state update.
// =========================================================================

TEST(GDNScanTest, ChunkwiseWyTcMatchesFused) {
    constexpr int n_heads = 4, head_dim = 128, state_size = 128, n_groups = 4;
    constexpr int inner = n_heads * head_dim, BC_size = n_groups * state_size;
    constexpr int conv_channels = 2 * BC_size + inner;
    constexpr int n_tok = 32;  // 2 WY-TC chunks of 16

    srand(23);
    std::vector<float> conv_f32(n_tok * conv_channels);
    std::vector<float> all_alpha(n_tok * n_heads), all_beta(n_tok * n_heads);
    std::vector<float> h_A_log(n_heads, -0.5f), h_dt_bias(n_heads, 0.5f);
    for (auto& v : conv_f32)
        v = (rand() % 200 - 100) / 100.0f;
    for (auto& v : all_alpha)
        v = (rand() % 200 - 100) / 100.0f;
    for (auto& v : all_beta)
        v = (rand() % 200 - 100) / 100.0f;

    std::vector<half> h_alpha_h(n_tok * n_heads), h_beta_h(n_tok * n_heads);
    for (int i = 0; i < n_tok * n_heads; i++) {
        h_alpha_h[i] = __float2half(all_alpha[i]);
        h_beta_h[i] = __float2half(all_beta[i]);
    }

    const int state_floats = n_heads * state_size * head_dim;
    float *d_conv, *d_A, *d_dt, *d_state_ref, *d_state_tc;
    half *d_alpha, *d_beta, *d_y_ref, *d_y_tc;
    cudaMalloc(&d_conv, n_tok * conv_channels * sizeof(float));
    cudaMalloc(&d_A, n_heads * sizeof(float));
    cudaMalloc(&d_dt, n_heads * sizeof(float));
    cudaMalloc(&d_state_ref, state_floats * sizeof(float));
    cudaMalloc(&d_state_tc, state_floats * sizeof(float));
    cudaMalloc(&d_alpha, n_tok * n_heads * sizeof(half));
    cudaMalloc(&d_beta, n_tok * n_heads * sizeof(half));
    cudaMalloc(&d_y_ref, n_tok * inner * sizeof(half));
    cudaMalloc(&d_y_tc, n_tok * inner * sizeof(half));

    cudaMemcpy(d_conv, conv_f32.data(), n_tok * conv_channels * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, h_A_log.data(), n_heads * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dt, h_dt_bias.data(), n_heads * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_alpha, h_alpha_h.data(), n_tok * n_heads * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, h_beta_h.data(), n_tok * n_heads * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemset(d_state_ref, 0, state_floats * sizeof(float));
    cudaMemset(d_state_tc, 0, state_floats * sizeof(float));

    gdn_scan_fused_f32(d_conv, conv_channels, d_alpha, d_beta, d_A, d_dt, d_state_ref, d_y_ref, n_tok,
                       n_heads, head_dim, state_size, n_groups, nullptr);
    gdn_scan_chunkwise_wy_tc_f32(d_conv, conv_channels, d_alpha, d_beta, d_A, d_dt, d_state_tc, d_y_tc,
                                 n_tok, n_heads, head_dim, state_size, n_groups, nullptr,
                                 /*grouped_layout=*/0);
    cudaError_t launch_err = cudaPeekAtLastError();
    cudaError_t sync_err = cudaDeviceSynchronize();
    if (launch_err != cudaSuccess)
        std::printf("\n  WY-TC launch error: %s\n", cudaGetErrorString(launch_err));
    if (sync_err != cudaSuccess)
        std::printf("  WY-TC sync error: %s\n", cudaGetErrorString(sync_err));

    std::vector<half> y_ref(n_tok * inner), y_tc(n_tok * inner);
    std::vector<float> state_ref(state_floats), state_tc(state_floats);
    cudaMemcpy(y_ref.data(), d_y_ref, n_tok * inner * sizeof(half), cudaMemcpyDeviceToHost);
    cudaMemcpy(y_tc.data(), d_y_tc, n_tok * inner * sizeof(half), cudaMemcpyDeviceToHost);
    cudaMemcpy(state_ref.data(), d_state_ref, state_floats * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(state_tc.data(), d_state_tc, state_floats * sizeof(float), cudaMemcpyDeviceToHost);

    float max_diff_y = 0;
    for (int i = 0; i < n_tok * inner; i++) {
        float diff = std::abs(__half2float(y_ref[i]) - __half2float(y_tc[i]));
        if (diff > max_diff_y)
            max_diff_y = diff;
    }
    float max_diff_state = 0;
    for (int i = 0; i < state_floats; i++) {
        float diff = std::abs(state_ref[i] - state_tc[i]);
        if (diff > max_diff_state)
            max_diff_state = diff;
    }

    std::printf("\n  ChunkwiseWyTc: max_diff Y = %.6e\n", max_diff_y);
    std::printf("  ChunkwiseWyTc: max_diff state = %.6e\n", max_diff_state);

    // FP16 storage of K̃ / Q̃ / H_0 (vs Phase 2a's FP32) costs ~3-4 mantissa
    // bits on the operands. WMMA accumulates in FP32 so the per-matmul
    // result keeps full precision, but the round-trip through FP16 storage
    // does cap the achievable accuracy. Phase 1a's FP16 1e-3 budget for Y
    // may or may not be met — the looser 1e-2 / 1e-2 tolerance lets the
    // test catch outright wrong outputs without forcing rebalancing the
    // FP16 storage choices now (Phase 2c could revisit).
    EXPECT_LT(max_diff_y, 1e-2f);
    EXPECT_LT(max_diff_state, 1e-2f);

    cudaFree(d_conv);
    cudaFree(d_A);
    cudaFree(d_dt);
    cudaFree(d_state_ref);
    cudaFree(d_state_tc);
    cudaFree(d_alpha);
    cudaFree(d_beta);
    cudaFree(d_y_ref);
    cudaFree(d_y_tc);
}

// =========================================================================
// Test 3g: Phase 2c fully-tuned WY-TC-MMA matches sequential
// -------------------------------------------------------------------------
// CHUNK=32, all 5 chunk-internal matmuls (KK, QK, KH, QH, H_L) on WMMA.
// Same tolerance as Phase 2b — FP16 storage costs ~3-4 mantissa bits but
// FP32 WMMA accumulation preserves per-matmul precision.
// =========================================================================

TEST(GDNScanTest, ChunkwiseWyTc2MatchesFused) {
    constexpr int n_heads = 4, head_dim = 128, state_size = 128, n_groups = 4;
    constexpr int inner = n_heads * head_dim, BC_size = n_groups * state_size;
    constexpr int conv_channels = 2 * BC_size + inner;
    constexpr int n_tok = 64;  // 2 Phase-2c chunks of 32

    srand(29);
    std::vector<float> conv_f32(n_tok * conv_channels);
    std::vector<float> all_alpha(n_tok * n_heads), all_beta(n_tok * n_heads);
    std::vector<float> h_A_log(n_heads, -0.5f), h_dt_bias(n_heads, 0.5f);
    for (auto& v : conv_f32)
        v = (rand() % 200 - 100) / 100.0f;
    for (auto& v : all_alpha)
        v = (rand() % 200 - 100) / 100.0f;
    for (auto& v : all_beta)
        v = (rand() % 200 - 100) / 100.0f;

    std::vector<half> h_alpha_h(n_tok * n_heads), h_beta_h(n_tok * n_heads);
    for (int i = 0; i < n_tok * n_heads; i++) {
        h_alpha_h[i] = __float2half(all_alpha[i]);
        h_beta_h[i] = __float2half(all_beta[i]);
    }

    const int state_floats = n_heads * state_size * head_dim;
    float *d_conv, *d_A, *d_dt, *d_state_ref, *d_state_tc2;
    half *d_alpha, *d_beta, *d_y_ref, *d_y_tc2;
    cudaMalloc(&d_conv, n_tok * conv_channels * sizeof(float));
    cudaMalloc(&d_A, n_heads * sizeof(float));
    cudaMalloc(&d_dt, n_heads * sizeof(float));
    cudaMalloc(&d_state_ref, state_floats * sizeof(float));
    cudaMalloc(&d_state_tc2, state_floats * sizeof(float));
    cudaMalloc(&d_alpha, n_tok * n_heads * sizeof(half));
    cudaMalloc(&d_beta, n_tok * n_heads * sizeof(half));
    cudaMalloc(&d_y_ref, n_tok * inner * sizeof(half));
    cudaMalloc(&d_y_tc2, n_tok * inner * sizeof(half));

    cudaMemcpy(d_conv, conv_f32.data(), n_tok * conv_channels * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, h_A_log.data(), n_heads * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dt, h_dt_bias.data(), n_heads * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_alpha, h_alpha_h.data(), n_tok * n_heads * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, h_beta_h.data(), n_tok * n_heads * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemset(d_state_ref, 0, state_floats * sizeof(float));
    cudaMemset(d_state_tc2, 0, state_floats * sizeof(float));

    gdn_scan_fused_f32(d_conv, conv_channels, d_alpha, d_beta, d_A, d_dt, d_state_ref, d_y_ref, n_tok,
                       n_heads, head_dim, state_size, n_groups, nullptr);
    gdn_scan_chunkwise_wy_tc2_f32(d_conv, conv_channels, d_alpha, d_beta, d_A, d_dt, d_state_tc2, d_y_tc2,
                                  n_tok, n_heads, head_dim, state_size, n_groups, nullptr,
                                  /*grouped_layout=*/0);
    cudaError_t launch_err = cudaPeekAtLastError();
    cudaError_t sync_err = cudaDeviceSynchronize();
    if (launch_err != cudaSuccess)
        std::printf("\n  WY-TC2 launch error: %s\n", cudaGetErrorString(launch_err));
    if (sync_err != cudaSuccess)
        std::printf("  WY-TC2 sync error: %s\n", cudaGetErrorString(sync_err));

    std::vector<half> y_ref(n_tok * inner), y_tc2(n_tok * inner);
    std::vector<float> state_ref(state_floats), state_tc2(state_floats);
    cudaMemcpy(y_ref.data(), d_y_ref, n_tok * inner * sizeof(half), cudaMemcpyDeviceToHost);
    cudaMemcpy(y_tc2.data(), d_y_tc2, n_tok * inner * sizeof(half), cudaMemcpyDeviceToHost);
    cudaMemcpy(state_ref.data(), d_state_ref, state_floats * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(state_tc2.data(), d_state_tc2, state_floats * sizeof(float), cudaMemcpyDeviceToHost);

    float max_diff_y = 0;
    for (int i = 0; i < n_tok * inner; i++) {
        float diff = std::abs(__half2float(y_ref[i]) - __half2float(y_tc2[i]));
        if (diff > max_diff_y)
            max_diff_y = diff;
    }
    float max_diff_state = 0;
    for (int i = 0; i < state_floats; i++) {
        float diff = std::abs(state_ref[i] - state_tc2[i]);
        if (diff > max_diff_state)
            max_diff_state = diff;
    }

    std::printf("\n  ChunkwiseWyTc2: max_diff Y = %.6e\n", max_diff_y);
    std::printf("  ChunkwiseWyTc2: max_diff state = %.6e\n", max_diff_state);

    EXPECT_LT(max_diff_y, 1e-2f);
    EXPECT_LT(max_diff_state, 1e-2f);

    cudaFree(d_conv);
    cudaFree(d_A);
    cudaFree(d_dt);
    cudaFree(d_state_ref);
    cudaFree(d_state_tc2);
    cudaFree(d_alpha);
    cudaFree(d_beta);
    cudaFree(d_y_ref);
    cudaFree(d_y_tc2);
}

// =========================================================================
// Test 3d: Chunkwise prototype microbench (Phase 1b.1).
// -------------------------------------------------------------------------
// Times the chunkwise SSD prototype vs the sequential fused scan at the
// Qwen 3.6 prefill shape (n_tokens=4096, n_heads=32, HD=SS=128, n_groups=16).
// Gated behind IMP_GDN_MICROBENCH=1 because it's a perf probe, not a
// correctness gate — only useful when explicitly comparing the two kernels.
// =========================================================================

TEST(GDNScanTest, ChunkwiseProtoMicrobench) {
    if (!std::getenv("IMP_GDN_MICROBENCH")) {
        GTEST_SKIP() << "Set IMP_GDN_MICROBENCH=1 to run the chunkwise perf probe";
    }
    constexpr int n_heads = 32, head_dim = 128, state_size = 128, n_groups = 16;
    constexpr int inner = n_heads * head_dim, BC_size = n_groups * state_size;
    constexpr int conv_channels = 2 * BC_size + inner;
    constexpr int n_tok = 4096;
    constexpr int chunk_size = 64;

    srand(11);
    std::vector<float> conv_f32(static_cast<size_t>(n_tok) * conv_channels);
    std::vector<float> all_alpha(n_tok * n_heads), all_beta(n_tok * n_heads);
    std::vector<float> h_A_log(n_heads, -0.5f), h_dt_bias(n_heads, 0.5f);
    for (auto& v : conv_f32)
        v = (rand() % 200 - 100) / 100.0f;
    for (auto& v : all_alpha)
        v = (rand() % 200 - 100) / 100.0f;
    for (auto& v : all_beta)
        v = (rand() % 200 - 100) / 100.0f;

    std::vector<half> h_alpha_h(n_tok * n_heads), h_beta_h(n_tok * n_heads);
    for (int i = 0; i < n_tok * n_heads; i++) {
        h_alpha_h[i] = __float2half(all_alpha[i]);
        h_beta_h[i] = __float2half(all_beta[i]);
    }

    const int state_floats = n_heads * state_size * head_dim;
    float *d_conv, *d_A, *d_dt, *d_state;
    half *d_alpha, *d_beta, *d_y;
    cudaMalloc(&d_conv, static_cast<size_t>(n_tok) * conv_channels * sizeof(float));
    cudaMalloc(&d_A, n_heads * sizeof(float));
    cudaMalloc(&d_dt, n_heads * sizeof(float));
    cudaMalloc(&d_state, state_floats * sizeof(float));
    cudaMalloc(&d_alpha, n_tok * n_heads * sizeof(half));
    cudaMalloc(&d_beta, n_tok * n_heads * sizeof(half));
    cudaMalloc(&d_y, n_tok * inner * sizeof(half));

    cudaMemcpy(d_conv, conv_f32.data(), static_cast<size_t>(n_tok) * conv_channels * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, h_A_log.data(), n_heads * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dt, h_dt_bias.data(), n_heads * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_alpha, h_alpha_h.data(), n_tok * n_heads * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, h_beta_h.data(), n_tok * n_heads * sizeof(half), cudaMemcpyHostToDevice);

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);

    auto bench = [&](auto fn, const char* label, int reps = 20, int warmup = 3) {
        for (int r = 0; r < warmup; r++) {
            cudaMemset(d_state, 0, state_floats * sizeof(float));
            fn();
        }
        cudaDeviceSynchronize();
        float total_ms = 0;
        for (int r = 0; r < reps; r++) {
            cudaMemset(d_state, 0, state_floats * sizeof(float));
            cudaDeviceSynchronize();
            cudaEventRecord(e0);
            fn();
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            float ms = 0;
            cudaEventElapsedTime(&ms, e0, e1);
            total_ms += ms;
        }
        float avg_ms = total_ms / reps;
        float us_per_tok = avg_ms * 1000.0f / n_tok;
        std::printf("  %s: %.3f ms avg over %d tokens = %.3f us/token (%d reps)\n", label, avg_ms, n_tok,
                    us_per_tok, reps);
        return avg_ms;
    };

    float ms_fused = bench(
        [&]() {
            gdn_scan_fused_f32(d_conv, conv_channels, d_alpha, d_beta, d_A, d_dt, d_state, d_y, n_tok,
                               n_heads, head_dim, state_size, n_groups, nullptr);
        },
        "gdn_scan_fused_f32 (sequential)");

    float ms_chunkwise = bench(
        [&]() {
            gdn_scan_chunkwise_f32(d_conv, conv_channels, d_alpha, d_beta, d_A, d_dt, d_state, d_y, n_tok,
                                   n_heads, head_dim, state_size, n_groups, nullptr,
                                   /*chunk_size=*/chunk_size, /*grouped_layout=*/0);
        },
        "gdn_scan_chunkwise_f32 (Phase 1b.1 proto)");

    float ms_wy = bench(
        [&]() {
            gdn_scan_chunkwise_wy_f32(d_conv, conv_channels, d_alpha, d_beta, d_A, d_dt, d_state, d_y, n_tok,
                                      n_heads, head_dim, state_size, n_groups, nullptr,
                                      /*grouped_layout=*/0);
        },
        "gdn_scan_chunkwise_wy_f32 (Phase 2a WY-rep)");

    float ms_wy_tc = bench(
        [&]() {
            gdn_scan_chunkwise_wy_tc_f32(d_conv, conv_channels, d_alpha, d_beta, d_A, d_dt, d_state, d_y,
                                         n_tok, n_heads, head_dim, state_size, n_groups, nullptr,
                                         /*grouped_layout=*/0);
        },
        "gdn_scan_chunkwise_wy_tc_f32 (Phase 2b TC-MMA)");

    float ms_wy_tc2 = bench(
        [&]() {
            gdn_scan_chunkwise_wy_tc2_f32(d_conv, conv_channels, d_alpha, d_beta, d_A, d_dt, d_state, d_y,
                                          n_tok, n_heads, head_dim, state_size, n_groups, nullptr,
                                          /*grouped_layout=*/0);
        },
        "gdn_scan_chunkwise_wy_tc2_f32 (Phase 2c CHUNK=32 + H_L TC)");

    std::printf("  Ratio Phase1b.1/fused = %.3fx\n", ms_chunkwise / ms_fused);
    std::printf("  Ratio Phase2a-WY/fused = %.3fx (naive shared-mem matmul)\n", ms_wy / ms_fused);
    std::printf("  Ratio Phase2b-WY-TC/fused = %.3fx (KK/QK/KH/QH on TC, CHUNK=16)\n",
                ms_wy_tc / ms_fused);
    std::printf("  Ratio Phase2c-WY-TC2/fused = %.3fx (all 5 matmuls on TC, CHUNK=32)\n",
                ms_wy_tc2 / ms_fused);

    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    cudaFree(d_conv);
    cudaFree(d_A);
    cudaFree(d_dt);
    cudaFree(d_state);
    cudaFree(d_alpha);
    cudaFree(d_beta);
    cudaFree(d_y);
}

// =========================================================================
// Test 4: Zero state + one token produces non-zero output
// =========================================================================

TEST(GDNScanTest, ZeroState) {
    constexpr int n_heads = 2, head_dim = 128, state_size = 128, n_groups = 2;
    constexpr int inner = n_heads * head_dim, BC_size = n_groups * state_size;

    srand(42);
    std::vector<float> h_V(inner), h_K(BC_size), h_Q(BC_size);
    for (auto& v : h_V)
        v = 1.0f;
    for (auto& v : h_K)
        v = 0.5f;
    for (auto& v : h_Q)
        v = 0.3f;

    float *d_V, *d_K, *d_Q, *d_A, *d_dt, *d_state;
    half *d_alpha, *d_beta, *d_y;
    cudaMalloc(&d_V, inner * sizeof(float));
    cudaMalloc(&d_K, BC_size * sizeof(float));
    cudaMalloc(&d_Q, BC_size * sizeof(float));
    cudaMalloc(&d_A, n_heads * sizeof(float));
    cudaMalloc(&d_dt, n_heads * sizeof(float));
    cudaMalloc(&d_state, n_heads * state_size * head_dim * sizeof(float));
    cudaMalloc(&d_alpha, n_heads * sizeof(half));
    cudaMalloc(&d_beta, n_heads * sizeof(half));
    cudaMalloc(&d_y, inner * sizeof(half));

    cudaMemcpy(d_V, h_V.data(), inner * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K.data(), BC_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Q, h_Q.data(), BC_size * sizeof(float), cudaMemcpyHostToDevice);
    std::vector<float> A(n_heads, -0.5f), dt(n_heads, 0.5f);
    cudaMemcpy(d_A, A.data(), n_heads * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dt, dt.data(), n_heads * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_state, 0, n_heads * state_size * head_dim * sizeof(float));
    std::vector<half> ha(n_heads), hb(n_heads);
    for (int i = 0; i < n_heads; i++) {
        ha[i] = __float2half(0.5f);
        hb[i] = __float2half(0.5f);
    }
    cudaMemcpy(d_alpha, ha.data(), n_heads * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, hb.data(), n_heads * sizeof(half), cudaMemcpyHostToDevice);

    gdn_scan_decode_f32(d_V, d_K, d_Q, d_alpha, d_beta, d_A, d_dt, d_state, d_y, nullptr, n_heads, head_dim,
                        state_size, n_groups, nullptr);
    cudaDeviceSynchronize();

    std::vector<half> hy(inner);
    cudaMemcpy(hy.data(), d_y, inner * sizeof(half), cudaMemcpyDeviceToHost);
    bool any_nonzero = false;
    for (int i = 0; i < inner; i++) {
        if (fabsf(__half2float(hy[i])) > 1e-6f) {
            any_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(any_nonzero) << "Zero state + non-zero input should produce non-zero output";

    cudaFree(d_V);
    cudaFree(d_K);
    cudaFree(d_Q);
    cudaFree(d_A);
    cudaFree(d_dt);
    cudaFree(d_state);
    cudaFree(d_alpha);
    cudaFree(d_beta);
    cudaFree(d_y);
}

// =========================================================================
// Test 5: RMSNormGatedSiLU kernel
// =========================================================================

TEST(GDNScanTest, RMSNormGatedSiLU) {
    constexpr int n_tokens = 2, n_heads = 4, head_dim = 64;
    constexpr int total = n_tokens * n_heads * head_dim;

    srand(42);
    std::vector<float> h_y(total), h_gate(total), h_weight(head_dim);
    for (auto& v : h_y)
        v = (rand() % 200 - 100) / 100.0f;
    for (auto& v : h_gate)
        v = (rand() % 200 - 100) / 100.0f;
    for (auto& v : h_weight)
        v = 0.5f + (rand() % 100) / 200.0f;
    const float eps = 1e-5f;

    // CPU reference
    std::vector<float> y_ref(total);
    for (int t = 0; t < n_tokens; t++) {
        for (int h = 0; h < n_heads; h++) {
            int base = t * n_heads * head_dim + h * head_dim;
            float sq_sum = 0;
            for (int d = 0; d < head_dim; d++)
                sq_sum += h_y[base + d] * h_y[base + d];
            float inv_rms = 1.0f / sqrtf(sq_sum / head_dim + eps);
            for (int d = 0; d < head_dim; d++) {
                float normed = h_y[base + d] * inv_rms * h_weight[d];
                float g = h_gate[base + d];
                float silu_g = g / (1.0f + expf(-g));
                y_ref[base + d] = normed * silu_g;
            }
        }
    }

    // GPU
    half *d_y, *d_gate, *d_weight;
    cudaMalloc(&d_y, total * sizeof(half));
    cudaMalloc(&d_gate, total * sizeof(half));
    cudaMalloc(&d_weight, head_dim * sizeof(half));

    std::vector<half> hy(total), hg(total), hw(head_dim);
    for (int i = 0; i < total; i++)
        hy[i] = __float2half(h_y[i]);
    for (int i = 0; i < total; i++)
        hg[i] = __float2half(h_gate[i]);
    for (int i = 0; i < head_dim; i++)
        hw[i] = __float2half(h_weight[i]);
    cudaMemcpy(d_y, hy.data(), total * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gate, hg.data(), total * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, hw.data(), head_dim * sizeof(half), cudaMemcpyHostToDevice);

    gdn_rmsnorm_gated_silu(d_y, d_gate, d_weight, eps, n_tokens, n_heads, head_dim, nullptr);
    cudaDeviceSynchronize();

    cudaMemcpy(hy.data(), d_y, total * sizeof(half), cudaMemcpyDeviceToHost);
    float max_err = 0;
    for (int i = 0; i < total; i++)
        max_err = fmaxf(max_err, fabsf(__half2float(hy[i]) - y_ref[i]));
    EXPECT_LT(max_err, 0.02f) << "RMSNormGatedSiLU max error too large";

    cudaFree(d_y);
    cudaFree(d_gate);
    cudaFree(d_weight);
}

// ===========================================================================
// Test: Padded verify chunk (#847), GDN scans — with d_real_n set, y is
// produced for every row but h_state must stop advancing after the real
// last row (bit-equal to a plain run over the real rows). Covers the fused
// kernel, the chunkwise entry (routes padded chunks through one fused call)
// and the reference kernel.
// ===========================================================================
TEST(GDNScanTest, PaddedChunkDeviceLength) {
    constexpr int n_heads = 4;
    constexpr int n_groups = 2;
    constexpr int head_dim = 64;
    constexpr int state_size = 64;
    constexpr int inner = n_heads * head_dim;
    constexpr int BC_size = n_groups * state_size;
    constexpr int conv_channels = 2 * BC_size + inner;
    constexpr int n_real = 5;
    constexpr int n_padded = 12;

    std::vector<float> conv_f32(n_padded * conv_channels);
    std::vector<half> h_alpha(n_padded * n_heads), h_beta(n_padded * n_heads);
    std::vector<float> h_A_log(n_heads, -0.5f), h_dt_bias(n_heads, 0.5f);
    for (size_t i = 0; i < conv_f32.size(); i++)
        conv_f32[i] = std::sin(0.37f * i);
    for (size_t i = 0; i < h_alpha.size(); i++) {
        h_alpha[i] = __float2half(std::cos(0.21f * i));
        h_beta[i] = __float2half(std::sin(0.11f * i + 1.0f));
    }

    float *d_conv, *d_A, *d_dt;
    half *d_alpha, *d_beta;
    cudaMalloc(&d_conv, conv_f32.size() * sizeof(float));
    cudaMalloc(&d_A, n_heads * sizeof(float));
    cudaMalloc(&d_dt, n_heads * sizeof(float));
    cudaMalloc(&d_alpha, h_alpha.size() * sizeof(half));
    cudaMalloc(&d_beta, h_beta.size() * sizeof(half));
    cudaMemcpy(d_conv, conv_f32.data(), conv_f32.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, h_A_log.data(), n_heads * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dt, h_dt_bias.data(), n_heads * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_alpha, h_alpha.data(), h_alpha.size() * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, h_beta.data(), h_beta.size() * sizeof(half), cudaMemcpyHostToDevice);

    int* d_real_n;
    cudaMalloc(&d_real_n, sizeof(int));
    int real_n = n_real;
    cudaMemcpy(d_real_n, &real_n, sizeof(int), cudaMemcpyHostToDevice);

    const size_t h_elems = static_cast<size_t>(n_heads) * state_size * head_dim;

    // scan_fn(n_tokens, d_real_n, d_state, d_y)
    auto check_variant = [&](const char* name, auto scan_fn) {
        float* d_state;
        half* d_y;
        cudaMalloc(&d_state, h_elems * sizeof(float));
        cudaMalloc(&d_y, static_cast<size_t>(n_padded) * inner * sizeof(half));

        // Reference: plain run over the real rows only.
        cudaMemset(d_state, 0, h_elems * sizeof(float));
        scan_fn(n_real, static_cast<const int*>(nullptr), d_state, d_y);
        cudaDeviceSynchronize();
        std::vector<float> h_ref(h_elems);
        std::vector<half> y_ref(static_cast<size_t>(n_real) * inner);
        cudaMemcpy(h_ref.data(), d_state, h_elems * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(y_ref.data(), d_y, y_ref.size() * sizeof(half), cudaMemcpyDeviceToHost);

        // Padded run with the device-side real length.
        cudaMemset(d_state, 0, h_elems * sizeof(float));
        scan_fn(n_padded, static_cast<const int*>(d_real_n), d_state, d_y);
        cudaDeviceSynchronize();
        std::vector<float> h_pad(h_elems);
        std::vector<half> y_pad(static_cast<size_t>(n_padded) * inner);
        cudaMemcpy(h_pad.data(), d_state, h_elems * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(y_pad.data(), d_y, y_pad.size() * sizeof(half), cudaMemcpyDeviceToHost);

        for (int t = 0; t < n_real; t++)
            for (int i = 0; i < inner; i++)
                EXPECT_EQ(__half2float(y_ref[t * inner + i]), __half2float(y_pad[t * inner + i]))
                    << name << ": y mismatch at t=" << t << " i=" << i;
        for (size_t i = 0; i < h_elems; i++)
            EXPECT_EQ(h_ref[i], h_pad[i]) << name << ": h_state mismatch at i=" << i;

        cudaFree(d_state);
        cudaFree(d_y);
    };

    check_variant("fused", [&](int n_tok, const int* d_rn, float* d_state, half* d_y) {
        gdn_scan_fused_f32(d_conv, conv_channels, d_alpha, d_beta, d_A, d_dt, d_state, d_y, n_tok, n_heads,
                           head_dim, state_size, n_groups, nullptr, /*grouped_layout=*/0, d_rn);
    });
    check_variant("chunkwise", [&](int n_tok, const int* d_rn, float* d_state, half* d_y) {
        gdn_scan_chunkwise_f32(d_conv, conv_channels, d_alpha, d_beta, d_A, d_dt, d_state, d_y, n_tok,
                               n_heads, head_dim, state_size, n_groups, nullptr, /*chunk_size=*/64,
                               /*grouped_layout=*/0, d_rn);
    });
    check_variant("reference", [&](int n_tok, const int* d_rn, float* d_state, half* d_y) {
        gdn_scan_reference_f32(d_conv, conv_channels, d_alpha, d_beta, d_A, d_dt, d_state, d_y, n_tok,
                               n_heads, head_dim, state_size, n_groups, nullptr, /*grouped_layout=*/0, d_rn);
    });

    cudaFree(d_real_n);
    cudaFree(d_conv);
    cudaFree(d_A);
    cudaFree(d_dt);
    cudaFree(d_alpha);
    cudaFree(d_beta);
}

}  // namespace
}  // namespace imp
