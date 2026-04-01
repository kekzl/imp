#include "compute/gdn.h"
#include "core/logging.h"
#include <cmath>

namespace imp {

// Forward declarations
__global__ void gdn_scan_decode_kernel(
    const float*, const float*, const float*,
    const half*, const half*, const float*, const float*,
    float*, half*, const half*,
    int, int, int, int);

// ---------------------------------------------------------------------------
// Fused multi-token GDN Delta Rule Scan kernel.
//
// Processes ALL tokens in a single kernel launch. The recurrent state is
// cached in registers (128 floats per thread = 512 bytes), eliminating
// per-token global memory round-trips.
//
// Grid:  (n_heads)
// Block: (head_dim_ssm)  — typically 128 threads
//
// Each block handles one head, processing all tokens sequentially.
// Each thread owns one column (d) of the state matrix H[state_size, head_dim].
// Thread d holds H[0..state_size-1, d] in register array.
//
// Shared memory: K_norm[state_size] + Q_norm[state_size] + reduce[block_dim]
// ---------------------------------------------------------------------------
template <int HD, int SS>
__global__ void __launch_bounds__(HD, 2)
gdn_scan_fused_kernel(
    const float* __restrict__ conv_f32,   // [n_tokens, conv_channels] FP32
    const half*  __restrict__ alpha_all,  // [n_tokens, n_heads] FP16
    const half*  __restrict__ beta_all,   // [n_tokens, n_heads] FP16
    const float* __restrict__ A_log,      // [n_heads] FP32
    const float* __restrict__ dt_bias,    // [n_heads] FP32
    float*       __restrict__ h_state,    // [n_heads, SS, HD] FP32
    half*        __restrict__ y_out,      // [n_tokens, n_heads * HD] FP16
    int n_tokens, int n_heads, int n_groups, int conv_channels)
{
    const int h = blockIdx.x;
    if (h >= n_heads) return;
    const int d = threadIdx.x;

    const int g = h % n_groups;
    const int inner = n_heads * HD;
    const int BC_size = n_groups * SS;
    const float scale = rsqrtf(static_cast<float>(HD));

    // Load per-head constants
    const float A_h = A_log[h];
    const float dtb_h = dt_bias[h];

    // Load state into registers — the critical optimization.
    // Each thread holds SS floats = one column of H[SS, HD].
    float H_reg[SS];
    {
        const float* H_col = h_state + static_cast<size_t>(h) * SS * HD + d;
        #pragma unroll
        for (int s = 0; s < SS; s++)
            H_reg[s] = H_col[s * HD];
    }

    // Shared memory: K_norm[SS] + Q_norm[SS] + reduce_buf[HD]
    extern __shared__ float smem[];
    float* s_k = smem;
    float* s_q = smem + SS;
    float* s_reduce = smem + 2 * SS;

    // Process each token
    for (int t = 0; t < n_tokens; t++) {
        const float* row = conv_f32 + static_cast<size_t>(t) * conv_channels;
        const float* Q_g = row + g * SS;
        const float* K_g = row + BC_size + g * SS;
        const float* V_base = row + 2 * BC_size;

        // Load V for this thread's d index
        float v_d = V_base[h * HD + d];

        // Compute alpha → decay gate
        float alpha_h = __half2float(alpha_all[t * n_heads + h]);
        float dt_val = alpha_h + dtb_h;
        dt_val = (dt_val > 20.0f) ? dt_val : logf(1.0f + expf(dt_val));
        float g_t = expf(fmaxf(A_h * dt_val, -20.0f));

        // Compute beta → learning rate
        float beta_h = __half2float(beta_all[t * n_heads + h]);
        beta_h = 1.0f / (1.0f + expf(-fmaxf(fminf(beta_h, 20.0f), -20.0f)));

        // Parallel L2-normalize K and Q.
        // Each thread loads SS/HD elements and contributes to the reduction.
        // With HD=128 threads and SS=128 elements, each thread loads 1 element.
        {
            // Load K and Q into shared memory
            if (d < SS) {
                s_k[d] = K_g[d];
                s_q[d] = Q_g[d];
            }
            __syncthreads();

            // Parallel sum-of-squares reduction
            float k_sq = 0.0f, q_sq = 0.0f;
            for (int i = d; i < SS; i += HD) {
                k_sq += s_k[i] * s_k[i];
                q_sq += s_q[i] * s_q[i];
            }
            // Block reduction for k_sq
            s_reduce[d] = k_sq;
            __syncthreads();
            for (int stride = HD / 2; stride > 0; stride >>= 1) {
                if (d < stride) s_reduce[d] += s_reduce[d + stride];
                __syncthreads();
            }
            float k_inv = rsqrtf(s_reduce[0] + 1e-6f);

            s_reduce[d] = q_sq;
            __syncthreads();
            for (int stride = HD / 2; stride > 0; stride >>= 1) {
                if (d < stride) s_reduce[d] += s_reduce[d + stride];
                __syncthreads();
            }
            float q_inv = rsqrtf(s_reduce[0] + 1e-6f);

            // Normalize in-place
            if (d < SS) {
                s_k[d] *= k_inv;
                s_q[d] *= q_inv;
            }
            __syncthreads();
        }

        // Delta rule scan — all in registers, no global memory access
        // Step 1: kv = H^T @ k_norm (dot product of state column with k)
        float kv_d = 0.0f;
        #pragma unroll
        for (int s = 0; s < SS; s++)
            kv_d += H_reg[s] * s_k[s];

        // Step 2: delta = (v - g*kv) * beta
        float delta_d = (v_d - g_t * kv_d) * beta_h;

        // Step 3: Update state + compute output
        float y_partial = 0.0f;
        #pragma unroll
        for (int s = 0; s < SS; s++) {
            float h_new = g_t * H_reg[s] + s_k[s] * delta_d;
            H_reg[s] = h_new;
            y_partial += h_new * s_q[s];
        }

        y_out[t * inner + h * HD + d] = __float2half(y_partial * scale);

        // Sync before next token — the next iteration overwrites s_k/s_q in
        // shared memory. Without this barrier, fast threads can overwrite
        // s_k/s_q while slow threads are still reading them in the loops above.
        if (t + 1 < n_tokens) __syncthreads();
    }

    // Store state back to global memory (once, at the end)
    {
        float* H_col = h_state + static_cast<size_t>(h) * SS * HD + d;
        #pragma unroll
        for (int s = 0; s < SS; s++)
            H_col[s * HD] = H_reg[s];
    }
}

// ---------------------------------------------------------------------------
// Fused RMSNormGated + SiLU kernel.
// Computes: y[t,h,:] = rmsnorm(y[t,h,:], weight) * silu(gate[t,h,:])
//
// Grid:  (n_tokens, n_heads)
// Block: (head_dim)
// ---------------------------------------------------------------------------
__global__ void gdn_rmsnorm_gated_silu_kernel(
    half* __restrict__ y,            // [n_tokens, n_heads * head_dim] in/out
    const half* __restrict__ gate,   // [n_tokens, n_heads * head_dim]
    const half* __restrict__ weight, // [head_dim] shared norm weight
    float eps, int n_heads, int head_dim)
{
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const int d = threadIdx.x;
    if (d >= head_dim) return;

    const int inner = n_heads * head_dim;
    const int base = t * inner + h * head_dim;

    // Load y value
    float val = __half2float(y[base + d]);

    // Parallel sum-of-squares for RMSNorm
    extern __shared__ float s_buf[];
    s_buf[d] = val * val;
    __syncthreads();
    for (int stride = head_dim / 2; stride > 0; stride >>= 1) {
        if (d < stride) s_buf[d] += s_buf[d + stride];
        __syncthreads();
    }
    float inv_rms = rsqrtf(s_buf[0] / static_cast<float>(head_dim) + eps);

    // RMSNorm: normalize and scale by weight
    float normed = val * inv_rms * __half2float(weight[d]);

    // SiLU on gate and multiply
    float g = __half2float(gate[base + d]);
    float silu_g = g / (1.0f + expf(-g));

    y[base + d] = __float2half(normed * silu_g);
}

// ---------------------------------------------------------------------------
// V-head reorder: tiled → grouped (undo GGUF converter reorder for ssm_out)
// ---------------------------------------------------------------------------
__global__ void vhead_tiled_to_grouped_kernel(
    const half* __restrict__ src,
    half* __restrict__       dst,
    int n_tokens, int n_heads, int head_dim, int n_groups)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_tokens * n_heads * head_dim;
    if (tid >= total) return;

    int d = tid % head_dim;
    int h_tiled = (tid / head_dim) % n_heads;
    int t = tid / (n_heads * head_dim);

    int n_v_per_k = n_heads / n_groups;
    int replica = h_tiled / n_groups;
    int group = h_tiled % n_groups;
    int h_grouped = group * n_v_per_k + replica;

    dst[t * n_heads * head_dim + h_grouped * head_dim + d] =
        src[t * n_heads * head_dim + h_tiled * head_dim + d];
}

void vhead_tiled_to_grouped(const half* src, half* dst,
                             int n_tokens, int n_heads, int head_dim, int n_groups,
                             cudaStream_t stream) {
    if (n_heads == n_groups) return;
    int total = n_tokens * n_heads * head_dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    vhead_tiled_to_grouped_kernel<<<blocks, threads, 0, stream>>>(
        src, dst, n_tokens, n_heads, head_dim, n_groups);
}

// ---------------------------------------------------------------------------
// Host launchers
// ---------------------------------------------------------------------------

// Fused scan: processes all tokens in one kernel launch.
// conv_f32: [n_tokens, conv_channels] FP32 — full conv output (Q|K|V interleaved per token)
void gdn_scan_fused_f32(const float* conv_f32, int conv_channels,
                         const half* alpha, const half* beta,
                         const float* A_log, const float* dt_bias,
                         float* h_state, half* y,
                         int n_tokens, int n_heads, int head_dim_ssm,
                         int state_size, int n_groups,
                         cudaStream_t stream) {
    // Shared memory: K_norm[SS] + Q_norm[SS] + reduce[HD]
    size_t smem = (2 * state_size + head_dim_ssm) * sizeof(float);

    // Template dispatch for common sizes
    if (head_dim_ssm == 128 && state_size == 128) {
        gdn_scan_fused_kernel<128, 128><<<n_heads, 128, smem, stream>>>(
            conv_f32, alpha, beta, A_log, dt_bias, h_state, y,
            n_tokens, n_heads, n_groups, conv_channels);
    } else if (head_dim_ssm == 64 && state_size == 64) {
        gdn_scan_fused_kernel<64, 64><<<n_heads, 64, smem, stream>>>(
            conv_f32, alpha, beta, A_log, dt_bias, h_state, y,
            n_tokens, n_heads, n_groups, conv_channels);
    } else {
        // Fallback: per-token loop (for unsupported HD/SS sizes)
        int inner = n_heads * head_dim_ssm;
        int BC_size = n_groups * state_size;
        size_t smem_old = 2 * state_size * sizeof(float) + 2 * sizeof(float);
        for (int t = 0; t < n_tokens; t++) {
            const float* row = conv_f32 + static_cast<size_t>(t) * conv_channels;
            gdn_scan_decode_kernel<<<n_heads, head_dim_ssm, smem_old, stream>>>(
                row + 2 * BC_size, row + BC_size, row,
                alpha + t * n_heads, beta + t * n_heads,
                A_log, dt_bias, h_state,
                y + t * inner, nullptr,
                n_heads, head_dim_ssm, state_size, n_groups);
        }
    }
}

// Fused RMSNormGated + SiLU
void gdn_rmsnorm_gated_silu(half* y, const half* gate, const half* weight,
                              float eps, int n_tokens, int n_heads, int head_dim,
                              cudaStream_t stream) {
    size_t smem = head_dim * sizeof(float);
    dim3 grid(n_tokens, n_heads);
    gdn_rmsnorm_gated_silu_kernel<<<grid, head_dim, smem, stream>>>(
        y, gate, weight, eps, n_heads, head_dim);
}

// ---------------------------------------------------------------------------
// Legacy interfaces (kept for backward compatibility)
// ---------------------------------------------------------------------------

// Old per-token decode kernel (still available for reference)
__global__ void gdn_scan_decode_kernel(
    const float* __restrict__ x,
    const float* __restrict__ B_in,
    const float* __restrict__ C_in,
    const half*  __restrict__ alpha_raw,
    const half*  __restrict__ beta_raw,
    const float* __restrict__ A_log,
    const float* __restrict__ dt_bias,
    float*       __restrict__ h_state,
    half*        __restrict__ y,
    const half*  __restrict__ z,
    int n_heads, int head_dim_ssm, int state_size, int n_groups)
{
    const int h = blockIdx.x;
    if (h >= n_heads) return;
    const int d = threadIdx.x;
    if (d >= head_dim_ssm) return;

    const int g = h % n_groups;
    float* H = h_state + static_cast<size_t>(h) * state_size * head_dim_ssm;
    const float* K_g = B_in + g * state_size;
    const float* Q_g = C_in + g * state_size;

    float v_d = x[h * head_dim_ssm + d];
    float alpha_h = __half2float(alpha_raw[h]);
    float dt_val = alpha_h + dt_bias[h];
    dt_val = (dt_val > 20.0f) ? dt_val : logf(1.0f + expf(dt_val));
    float g_t = expf(fmaxf(A_log[h] * dt_val, -20.0f));

    float beta_h = __half2float(beta_raw[h]);
    beta_h = 1.0f / (1.0f + expf(-fmaxf(fminf(beta_h, 20.0f), -20.0f)));

    extern __shared__ float smem[];
    float* s_k = smem;
    float* s_q = smem + state_size;
    __shared__ float s_k_inv, s_q_inv;
    if (d == 0) {
        float k_sq = 0.0f, q_sq = 0.0f;
        for (int s = 0; s < state_size; s++) {
            float ks = K_g[s]; float qs = Q_g[s];
            s_k[s] = ks; s_q[s] = qs;
            k_sq += ks * ks; q_sq += qs * qs;
        }
        s_k_inv = rsqrtf(k_sq + 1e-6f);
        s_q_inv = rsqrtf(q_sq + 1e-6f);
        for (int s = 0; s < state_size; s++) {
            s_k[s] *= s_k_inv; s_q[s] *= s_q_inv;
        }
    }
    __syncthreads();

    const float scale = rsqrtf(static_cast<float>(head_dim_ssm));
    float kv_d = 0.0f;
    for (int s = 0; s < state_size; s++)
        kv_d += H[s * head_dim_ssm + d] * s_k[s];
    float delta_d = (v_d - g_t * kv_d) * beta_h;
    float y_partial = 0.0f;
    for (int s = 0; s < state_size; s++) {
        float h_new = g_t * H[s * head_dim_ssm + d] + s_k[s] * delta_d;
        H[s * head_dim_ssm + d] = h_new;
        y_partial += h_new * s_q[s];
    }
    y[h * head_dim_ssm + d] = __float2half(y_partial * scale);
}

void gdn_scan_decode_f32(const float* x, const float* B, const float* C,
                         const half* alpha, const half* beta,
                         const float* A_log, const float* dt_bias,
                         float* h_state, half* y, const half* z,
                         int n_heads, int head_dim_ssm,
                         int state_size, int n_groups,
                         cudaStream_t stream) {
    size_t smem = 2 * state_size * sizeof(float) + 2 * sizeof(float);
    gdn_scan_decode_kernel<<<n_heads, head_dim_ssm, smem, stream>>>(
        x, B, C, alpha, beta, A_log, dt_bias, h_state, y, z,
        n_heads, head_dim_ssm, state_size, n_groups);
}

void gdn_scan_prefill_f32(const float* x, const float* B, const float* C,
                          const half* alpha, const half* beta,
                          const float* A_log, const float* dt_bias,
                          float* h_state, half* y, const half* z,
                          int n_tokens, int n_heads, int head_dim_ssm,
                          int state_size, int n_groups,
                          cudaStream_t stream) {
    int inner = n_heads * head_dim_ssm;
    int BC_size = n_groups * state_size;
    size_t smem = 2 * state_size * sizeof(float) + 2 * sizeof(float);
    for (int t = 0; t < n_tokens; t++) {
        gdn_scan_decode_kernel<<<n_heads, head_dim_ssm, smem, stream>>>(
            x + t * inner, B + t * BC_size, C + t * BC_size,
            alpha + t * n_heads, beta + t * n_heads,
            A_log, dt_bias, h_state, y + t * inner, nullptr,
            n_heads, head_dim_ssm, state_size, n_groups);
    }
}

// Legacy stubs
void gdn_scan_decode(const half*, const half*, const half*,
                     const half*, const half*, const float*, const float*,
                     float*, half*, const half*,
                     int, int, int, int, cudaStream_t) {}
void gdn_scan_prefill(const half*, const half*, const half*,
                      const half*, const half*, const float*, const float*,
                      float*, half*, const half*,
                      int, int, int, int, int, cudaStream_t) {}
void gdn_decode(const half*, const half*, const half*,
                const half*, const half*, float*, half*, const half*,
                int, int, int, int, cudaStream_t) {}
void gdn_prefill(const half*, const half*, const half*,
                 const half*, const half*, float*, half*, const half*,
                 int, int, int, int, int, cudaStream_t) {}

} // namespace imp
