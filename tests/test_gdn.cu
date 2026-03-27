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
static void gdn_scan_cpu(
    const float* V, const float* K, const float* Q,
    float alpha, float beta_raw, float A_log, float dt_bias,
    float* h_state, float* y_out, int head_dim, int state_size) {
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

    for (auto& v : h_V) v = (rand() % 200 - 100) / 100.0f;
    for (auto& v : h_K) v = (rand() % 200 - 100) / 100.0f;
    for (auto& v : h_Q) v = (rand() % 200 - 100) / 100.0f;
    for (int i = 0; i < n_heads; i++) {
        h_alpha[i] = (rand() % 200 - 100) / 100.0f;
        h_beta[i] = (rand() % 200 - 100) / 100.0f;
    }

    // CPU reference
    std::vector<float> y_cpu(inner);
    for (int h = 0; h < n_heads; h++) {
        int g = h % n_groups;
        gdn_scan_cpu(h_V.data() + h * head_dim, h_K.data() + g * state_size,
                      h_Q.data() + g * state_size, h_alpha[h], h_beta[h],
                      h_A_log[h], h_dt_bias[h],
                      state_cpu.data() + h * state_size * head_dim,
                      y_cpu.data() + h * head_dim, head_dim, state_size);
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

    gdn_scan_decode_f32(d_V, d_K, d_Q, d_alpha, d_beta, d_A, d_dt,
                         d_state, d_y, nullptr,
                         n_heads, head_dim, state_size, n_groups, nullptr);
    cudaDeviceSynchronize();

    std::vector<half> hy(inner);
    cudaMemcpy(hy.data(), d_y, inner * sizeof(half), cudaMemcpyDeviceToHost);
    float max_err = 0;
    for (int i = 0; i < inner; i++) {
        float err = fabsf(__half2float(hy[i]) - y_cpu[i]);
        max_err = fmaxf(max_err, err);
    }
    EXPECT_LT(max_err, 1e-2f) << "Single-token CPU vs GPU max error too large";

    cudaFree(d_V); cudaFree(d_K); cudaFree(d_Q);
    cudaFree(d_A); cudaFree(d_dt); cudaFree(d_state);
    cudaFree(d_alpha); cudaFree(d_beta); cudaFree(d_y);
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
    for (auto& v : all_V) v = (rand() % 200 - 100) / 100.0f;
    for (auto& v : all_K) v = (rand() % 200 - 100) / 100.0f;
    for (auto& v : all_Q) v = (rand() % 200 - 100) / 100.0f;
    for (auto& v : all_alpha) v = (rand() % 200 - 100) / 100.0f;
    for (auto& v : all_beta) v = (rand() % 200 - 100) / 100.0f;

    // CPU: sequential per-token processing
    std::vector<float> state_cpu(n_heads * state_size * head_dim, 0.0f);
    std::vector<float> y_cpu(inner);
    for (int t = 0; t < n_tok; t++) {
        for (int h = 0; h < n_heads; h++) {
            int g = h % n_groups;
            gdn_scan_cpu(&all_V[t * inner + h * head_dim],
                          &all_K[t * BC_size + g * state_size],
                          &all_Q[t * BC_size + g * state_size],
                          all_alpha[t * n_heads + h], all_beta[t * n_heads + h],
                          h_A_log[h], h_dt_bias[h],
                          &state_cpu[h * state_size * head_dim],
                          y_cpu.data() + h * head_dim, head_dim, state_size);
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
        gdn_scan_decode_f32(d_V, d_K, d_Q, d_alpha, d_beta, d_A, d_dt,
                             d_state, d_y, nullptr,
                             n_heads, head_dim, state_size, n_groups, nullptr);
        cudaDeviceSynchronize();
    }

    std::vector<half> hy(inner);
    cudaMemcpy(hy.data(), d_y, inner * sizeof(half), cudaMemcpyDeviceToHost);
    float max_err = 0;
    for (int i = 0; i < inner; i++)
        max_err = fmaxf(max_err, fabsf(__half2float(hy[i]) - y_cpu[i]));
    EXPECT_LT(max_err, 0.05f) << "Multi-token sequential max error too large";

    cudaFree(d_V); cudaFree(d_K); cudaFree(d_Q);
    cudaFree(d_A); cudaFree(d_dt); cudaFree(d_state);
    cudaFree(d_alpha); cudaFree(d_beta); cudaFree(d_y);
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
    for (auto& v : conv_f32) v = (rand() % 200 - 100) / 100.0f;
    for (auto& v : all_alpha) v = (rand() % 200 - 100) / 100.0f;
    for (auto& v : all_beta) v = (rand() % 200 - 100) / 100.0f;

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
    gdn_scan_fused_f32(d_conv, conv_channels, d_alpha, d_beta, d_A, d_dt,
                        d_state1, d_y1, n_tok, n_heads, head_dim,
                        state_size, n_groups, nullptr);
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
        gdn_scan_decode_f32(d_V, d_K, d_Q, d_a1, d_b1, d_A, d_dt,
                             d_state2, d_y2 + t * inner, nullptr,
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

    cudaFree(d_conv); cudaFree(d_A); cudaFree(d_dt);
    cudaFree(d_state1); cudaFree(d_state2);
    cudaFree(d_alpha); cudaFree(d_beta);
    cudaFree(d_y1); cudaFree(d_y2);
    cudaFree(d_V); cudaFree(d_K); cudaFree(d_Q);
    cudaFree(d_a1); cudaFree(d_b1);
}

// =========================================================================
// Test 4: Zero state + one token produces non-zero output
// =========================================================================

TEST(GDNScanTest, ZeroState) {
    constexpr int n_heads = 2, head_dim = 128, state_size = 128, n_groups = 2;
    constexpr int inner = n_heads * head_dim, BC_size = n_groups * state_size;

    srand(42);
    std::vector<float> h_V(inner), h_K(BC_size), h_Q(BC_size);
    for (auto& v : h_V) v = 1.0f;
    for (auto& v : h_K) v = 0.5f;
    for (auto& v : h_Q) v = 0.3f;

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

    gdn_scan_decode_f32(d_V, d_K, d_Q, d_alpha, d_beta, d_A, d_dt,
                         d_state, d_y, nullptr,
                         n_heads, head_dim, state_size, n_groups, nullptr);
    cudaDeviceSynchronize();

    std::vector<half> hy(inner);
    cudaMemcpy(hy.data(), d_y, inner * sizeof(half), cudaMemcpyDeviceToHost);
    bool any_nonzero = false;
    for (int i = 0; i < inner; i++) {
        if (fabsf(__half2float(hy[i])) > 1e-6f) { any_nonzero = true; break; }
    }
    EXPECT_TRUE(any_nonzero) << "Zero state + non-zero input should produce non-zero output";

    cudaFree(d_V); cudaFree(d_K); cudaFree(d_Q);
    cudaFree(d_A); cudaFree(d_dt); cudaFree(d_state);
    cudaFree(d_alpha); cudaFree(d_beta); cudaFree(d_y);
}

// =========================================================================
// Test 5: RMSNormGatedSiLU kernel
// =========================================================================

TEST(GDNScanTest, RMSNormGatedSiLU) {
    constexpr int n_tokens = 2, n_heads = 4, head_dim = 64;
    constexpr int total = n_tokens * n_heads * head_dim;

    srand(42);
    std::vector<float> h_y(total), h_gate(total), h_weight(head_dim);
    for (auto& v : h_y) v = (rand() % 200 - 100) / 100.0f;
    for (auto& v : h_gate) v = (rand() % 200 - 100) / 100.0f;
    for (auto& v : h_weight) v = 0.5f + (rand() % 100) / 200.0f;
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
    for (int i = 0; i < total; i++) hy[i] = __float2half(h_y[i]);
    for (int i = 0; i < total; i++) hg[i] = __float2half(h_gate[i]);
    for (int i = 0; i < head_dim; i++) hw[i] = __float2half(h_weight[i]);
    cudaMemcpy(d_y, hy.data(), total * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gate, hg.data(), total * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, hw.data(), head_dim * sizeof(half), cudaMemcpyHostToDevice);

    gdn_rmsnorm_gated_silu(d_y, d_gate, d_weight, eps,
                             n_tokens, n_heads, head_dim, nullptr);
    cudaDeviceSynchronize();

    cudaMemcpy(hy.data(), d_y, total * sizeof(half), cudaMemcpyDeviceToHost);
    float max_err = 0;
    for (int i = 0; i < total; i++)
        max_err = fmaxf(max_err, fabsf(__half2float(hy[i]) - y_ref[i]));
    EXPECT_LT(max_err, 0.02f) << "RMSNormGatedSiLU max error too large";

    cudaFree(d_y); cudaFree(d_gate); cudaFree(d_weight);
}

} // namespace
} // namespace imp
