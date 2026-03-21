// Minimal test for GDN delta rule scan kernel.
// Compares GPU kernel output with CPU reference for a single token.
#include <cstdio>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "compute/gdn.h"

using namespace imp;

// CPU reference implementation of the delta rule scan
void gdn_scan_cpu(
    const float* V, const float* K, const float* Q,
    float alpha, float beta_raw,
    float A_log, float dt_bias,
    float* h_state,  // [state_size, head_dim]
    float* y_out,    // [head_dim]
    int head_dim, int state_size)
{
    // Compute decay
    float dt_val = alpha + dt_bias;
    dt_val = (dt_val > 20.0f) ? dt_val : logf(1.0f + expf(dt_val));
    float g_t = expf(fmaxf(A_log * dt_val, -20.0f));

    // Beta = sigmoid
    float beta = 1.0f / (1.0f + expf(-fmaxf(fminf(beta_raw, 20.0f), -20.0f)));

    // L2-normalize K and Q
    float k_norm[256], q_norm[256];  // max state_size
    float k_sq = 0, q_sq = 0;
    for (int s = 0; s < state_size; s++) {
        k_sq += K[s] * K[s];
        q_sq += Q[s] * Q[s];
    }
    float k_inv = 1.0f / sqrtf(k_sq + 1e-6f);
    float q_inv = 1.0f / sqrtf(q_sq + 1e-6f);
    for (int s = 0; s < state_size; s++) {
        k_norm[s] = K[s] * k_inv;
        q_norm[s] = Q[s] * q_inv;
    }

    float scale = 1.0f / sqrtf((float)head_dim);

    for (int d = 0; d < head_dim; d++) {
        // kv = S^T @ k_norm
        float kv = 0;
        for (int s = 0; s < state_size; s++)
            kv += h_state[s * head_dim + d] * k_norm[s];

        // delta = (v - g*kv) * beta
        float delta = (V[d] - g_t * kv) * beta;

        // Update S and compute output
        float y_partial = 0;
        for (int s = 0; s < state_size; s++) {
            float h_new = g_t * h_state[s * head_dim + d] + k_norm[s] * delta;
            h_state[s * head_dim + d] = h_new;
            y_partial += h_new * q_norm[s];
        }
        y_out[d] = y_partial * scale;
    }
}

int main() {
    const int n_heads = 32;
    const int head_dim = 128;  // realistic dimensions for Qwen3.5-4B
    const int state_size = 128;
    const int n_groups = 16;
    const int inner = n_heads * head_dim;
    const int BC_size = n_groups * state_size;

    // Initialize test data
    float h_V[inner], h_K[BC_size], h_Q[BC_size];
    float h_alpha[n_heads], h_beta[n_heads];
    float h_A_log[n_heads], h_dt_bias[n_heads];
    float h_state_cpu[n_heads * state_size * head_dim] = {};
    float h_state_gpu[n_heads * state_size * head_dim] = {};

    srand(42);
    for (int i = 0; i < inner; i++) h_V[i] = (rand() % 200 - 100) / 100.0f;
    for (int i = 0; i < BC_size; i++) h_K[i] = (rand() % 200 - 100) / 100.0f;
    for (int i = 0; i < BC_size; i++) h_Q[i] = (rand() % 200 - 100) / 100.0f;
    for (int i = 0; i < n_heads; i++) {
        h_alpha[i] = (rand() % 200 - 100) / 100.0f;
        h_beta[i] = (rand() % 200 - 100) / 100.0f;
        h_A_log[i] = -0.5f;  // negative (pre-converted -exp(A_log))
        h_dt_bias[i] = 0.5f;
    }

    // CPU reference
    float y_cpu[inner];
    for (int h = 0; h < n_heads; h++) {
        int g = h % n_groups;
        gdn_scan_cpu(
            h_V + h * head_dim, h_K + g * state_size, h_Q + g * state_size,
            h_alpha[h], h_beta[h], h_A_log[h], h_dt_bias[h],
            h_state_cpu + h * state_size * head_dim,
            y_cpu + h * head_dim,
            head_dim, state_size);
    }

    // GPU kernel
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

    cudaMemcpy(d_V, h_V, inner * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, BC_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Q, h_Q, BC_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, h_A_log, n_heads * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dt, h_dt_bias, n_heads * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_state, h_state_gpu, n_heads * state_size * head_dim * sizeof(float), cudaMemcpyHostToDevice);

    // Convert alpha/beta to FP16
    half h_alpha_h[n_heads], h_beta_h[n_heads];
    for (int i = 0; i < n_heads; i++) {
        h_alpha_h[i] = __float2half(h_alpha[i]);
        h_beta_h[i] = __float2half(h_beta[i]);
    }
    cudaMemcpy(d_alpha, h_alpha_h, n_heads * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, h_beta_h, n_heads * sizeof(half), cudaMemcpyHostToDevice);

    gdn_scan_decode_f32(d_V, d_K, d_Q, d_alpha, d_beta, d_A, d_dt,
                         d_state, d_y, nullptr,
                         n_heads, head_dim, state_size, n_groups, 0);
    cudaError_t err2 = cudaGetLastError();
    if (err2 != cudaSuccess) {
        printf("CUDA error after kernel: %s\n", cudaGetErrorString(err2));
    }
    cudaDeviceSynchronize();
    err2 = cudaGetLastError();
    if (err2 != cudaSuccess) {
        printf("CUDA error after sync: %s\n", cudaGetErrorString(err2));
    }

    // Read back GPU output
    half h_y_gpu_h[inner];
    cudaMemcpy(h_y_gpu_h, d_y, inner * sizeof(half), cudaMemcpyDeviceToHost);

    // Compare
    printf("=== GDN Kernel Test (n_heads=%d, head_dim=%d, state=%d, groups=%d) ===\n",
           n_heads, head_dim, state_size, n_groups);
    float max_err = 0;
    int max_err_idx = 0;
    for (int i = 0; i < inner; i++) {
        float gpu_val = __half2float(h_y_gpu_h[i]);
        float cpu_val = y_cpu[i];
        float err = fabsf(gpu_val - cpu_val);
        if (err > max_err) { max_err = err; max_err_idx = i; }
    }
    printf("  First 4: CPU=[%.6f, %.6f, %.6f, %.6f] GPU=[%.6f, %.6f, %.6f, %.6f]\n",
           y_cpu[0], y_cpu[1], y_cpu[2], y_cpu[3],
           __half2float(h_y_gpu_h[0]), __half2float(h_y_gpu_h[1]),
           __half2float(h_y_gpu_h[2]), __half2float(h_y_gpu_h[3]));
    printf("  Max error: %.6f at index %d (CPU=%.6f GPU=%.6f)\n",
           max_err, max_err_idx, y_cpu[max_err_idx], __half2float(h_y_gpu_h[max_err_idx]));
    printf("Result: %s\n", max_err < 1e-2f ? "PASS" : "FAIL");

    // Test 2: Multi-token sequential (simulate prefill)
    printf("\n=== Multi-token sequential test (5 tokens) ===\n");
    {
        const int n_tok = 5;
        std::vector<float> all_V(n_tok * inner), all_K(n_tok * BC_size), all_Q(n_tok * BC_size);
        std::vector<float> all_alpha(n_tok * n_heads), all_beta(n_tok * n_heads);
        std::vector<float> state_cpu2(n_heads * state_size * head_dim, 0.0f);
        std::vector<float> state_gpu2(n_heads * state_size * head_dim, 0.0f);
        float y_cpu2[inner], y_gpu2_f[inner];

        for (int i = 0; i < n_tok * inner; i++) all_V[i] = (rand() % 200 - 100) / 100.0f;
        for (int i = 0; i < n_tok * BC_size; i++) all_K[i] = (rand() % 200 - 100) / 100.0f;
        for (int i = 0; i < n_tok * BC_size; i++) all_Q[i] = (rand() % 200 - 100) / 100.0f;
        for (int i = 0; i < n_tok * n_heads; i++) all_alpha[i] = (rand() % 200 - 100) / 100.0f;
        for (int i = 0; i < n_tok * n_heads; i++) all_beta[i] = (rand() % 200 - 100) / 100.0f;

        // CPU: process each token sequentially
        for (int t = 0; t < n_tok; t++) {
            for (int h2 = 0; h2 < n_heads; h2++) {
                int g2 = h2 % n_groups;
                gdn_scan_cpu(
                    &all_V[t * inner + h2 * head_dim],
                    &all_K[t * BC_size + g2 * state_size],
                    &all_Q[t * BC_size + g2 * state_size],
                    all_alpha[t * n_heads + h2], all_beta[t * n_heads + h2],
                    h_A_log[h2 % n_heads], h_dt_bias[h2 % n_heads],
                    &state_cpu2[h2 * state_size * head_dim],
                    y_cpu2 + h2 * head_dim,
                    head_dim, state_size);
            }
        }

        // GPU: process each token via gdn_scan_decode_f32
        cudaMemcpy(d_state, state_gpu2.data(), n_heads * state_size * head_dim * sizeof(float), cudaMemcpyHostToDevice);
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
                                 n_heads, head_dim, state_size, n_groups, 0);
            cudaDeviceSynchronize();
        }

        // Compare final output
        half hy[inner];
        cudaMemcpy(hy, d_y, inner * sizeof(half), cudaMemcpyDeviceToHost);
        float max_err2 = 0;
        for (int i = 0; i < inner; i++) {
            float ge = __half2float(hy[i]);
            float ce = y_cpu2[i];
            float e2 = fabsf(ge - ce);
            if (e2 > max_err2) max_err2 = e2;
        }
        printf("  Max error after %d tokens: %.6f\n", n_tok, max_err2);
        printf("  Result: %s\n", max_err2 < 0.05f ? "PASS" : "FAIL");
    }

    cudaFree(d_V); cudaFree(d_K); cudaFree(d_Q);
    cudaFree(d_A); cudaFree(d_dt); cudaFree(d_state);
    cudaFree(d_alpha); cudaFree(d_beta); cudaFree(d_y);
    return max_err < 1e-3f ? 0 : 1;
}
