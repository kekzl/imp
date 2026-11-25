#include <algorithm>

#include "compute/gdn_internal.cuh"
#include "core/logging.h"

namespace imp {

// Phase 1b.1 chunkwise SSD scan prototype (Mamba2 SSD adapted to the GDN delta rule). Same
// math as gdn_scan_fused_kernel; caches CHUNK tokens' normalised K,Q in shared memory, then
// sweeps the within-chunk delta-rule update. The WY-rep parallel form is gdn_scan_chunkpar.cu
// (Yang et al. 2024).
// Shared memory: s_k[CHUNK*SS], s_q[CHUNK*SS], s_reduce[HD]; at HD=SS=128, CHUNK=64 = 65 KiB,
// needs the dynamic-shared-memory opt-in. Grid (n_heads), block (HD).
template <int HD, int SS, int CHUNK, typename YOut>
__global__ void __launch_bounds__(HD, 1) gdn_scan_chunkwise_kernel(
    const float* __restrict__ conv_f32,  // [n_tokens, conv_channels] FP32
    const half* __restrict__ alpha_all,  // [n_tokens, n_heads] FP16
    const half* __restrict__ beta_all,   // [n_tokens, n_heads] FP16
    const float* __restrict__ A_log,     // [n_heads] FP32
    const float* __restrict__ dt_bias,   // [n_heads] FP32
    float* __restrict__ h_state,         // [n_heads, SS, HD] FP32
    YOut* __restrict__ y_out,            // [n_tokens, n_heads * HD] FP16 or FP32
    int n_tokens, int n_heads, int n_groups, int conv_channels, int grouped_layout) {
    const int h = blockIdx.x;
    if (h >= n_heads)
        return;
    const int d = threadIdx.x;

    const int g = grouped_layout ? (h / (n_heads / n_groups)) : (h % n_groups);
    const int inner = n_heads * HD;
    const int BC_size = n_groups * SS;
    const float scale = rsqrtf(static_cast<float>(HD));

    const float A_h = A_log[h];
    const float dtb_h = dt_bias[h];

    // State in registers (one column per thread).
    float H_reg[SS];
    {
        const float* H_col = h_state + (static_cast<size_t>(h) * SS * HD) + d;
#pragma unroll
        for (int s = 0; s < SS; s++)
            H_reg[s] = H_col[static_cast<ptrdiff_t>(s * HD)];
    }

    // #2218 bounded: HD, SS, CHUNK template args (128/128/64, 64/64/64 dispatch below): smem offsets
    // <= 2 * CHUNK * SS <= 16384 floats, H_col s * HD < SS * HD <= 16384, t_local * SS < CHUNK * SS <= 8192
    extern __shared__ float smem[];
    float* s_k = smem;                               // [CHUNK * SS]
    float* s_q = smem + static_cast<ptrdiff_t>(CHUNK * SS);           // [CHUNK * SS]
    float* s_reduce = smem + static_cast<ptrdiff_t>(2 * CHUNK * SS);  // [HD]

    int t_chunk_start = 0;
    while (t_chunk_start < n_tokens) {
        const int L = (t_chunk_start + CHUNK <= n_tokens) ? CHUNK : (n_tokens - t_chunk_start);

        // Phase 1: load this chunk's raw K,Q into shared memory; thread d stores one element per
        // token, looping over tokens.
        if (d < SS) {
            for (int t_local = 0; t_local < L; t_local++) {
                const int t_global = t_chunk_start + t_local;
                const float* row = conv_f32 + (static_cast<size_t>(t_global) * conv_channels);
                s_q[(t_local * SS) + d] = row[(g * SS) + d];
                s_k[(t_local * SS) + d] = row[BC_size + (g * SS) + d];
            }
        }
        __syncthreads();

        // Phase 2: L2-normalise K,Q per chunk token (sequential across tokens, parallel across SS).
        // Uses rsqrt(max(sum_sq,1e-12)), same formula as gdn_scan_fused_kernel, for bit-equivalent numerics.
        for (int t_local = 0; t_local < L; t_local++) {
            float* k_row = s_k + static_cast<ptrdiff_t>(t_local * SS);
            float* q_row = s_q + static_cast<ptrdiff_t>(t_local * SS);

            float k_sq = 0.0f, q_sq = 0.0f;
            for (int i = d; i < SS; i += HD) {
                k_sq += k_row[i] * k_row[i];
                q_sq += q_row[i] * q_row[i];
            }
            float k_inv = rsqrtf(fmaxf(gdn_block_sum(s_reduce, k_sq, d, HD), 1e-12f));
            float q_inv = rsqrtf(fmaxf(gdn_block_sum(s_reduce, q_sq, d, HD), 1e-12f));

            if (d < SS) {
                k_row[d] *= k_inv;
                q_row[d] *= q_inv;
            }
            __syncthreads();
        }

        // Phase 3: sequential per-token delta-rule update within the chunk, reading K~,Q~ from
        // shared memory; same math as gdn_scan_fused_kernel. The WY-rep form is gdn_scan_chunkpar.cu.
        for (int t_local = 0; t_local < L; t_local++) {
            const int t_global = t_chunk_start + t_local;
            const float* row = conv_f32 + (static_cast<size_t>(t_global) * conv_channels);
            const float* V_base = row + (static_cast<int64_t>(2) * BC_size);
            const float v_d = V_base[(h * HD) + d];

            const float* k_row = s_k + static_cast<ptrdiff_t>(t_local * SS);
            const float* q_row = s_q + static_cast<ptrdiff_t>(t_local * SS);

            float alpha_h = __half2float(alpha_all[(t_global * n_heads) + h]);
            float dt_val = alpha_h + dtb_h;
            dt_val = (dt_val > 20.0f) ? dt_val : logf(1.0f + expf(dt_val));
            const float g_t = expf(fmaxf(A_h * dt_val, -20.0f));

            float beta_h = __half2float(beta_all[(t_global * n_heads) + h]);
            beta_h = 1.0f / (1.0f + expf(-fmaxf(fminf(beta_h, 20.0f), -20.0f)));

            float kv_d = 0.0f;
#pragma unroll
            for (int s = 0; s < SS; s++)
                kv_d += H_reg[s] * k_row[s];

            const float delta_d = (v_d - (g_t * kv_d)) * beta_h;

            float y_partial = 0.0f;
#pragma unroll
            for (int s = 0; s < SS; s++) {
                const float h_new = g_t * H_reg[s] + k_row[s] * delta_d;
                H_reg[s] = h_new;
                y_partial += h_new * q_row[s];
            }

            const float out_val = y_partial * scale;
            if constexpr (std::is_same_v<YOut, float>) {
                y_out[(t_global * inner) + (h * HD) + d] = out_val;
            } else {
                y_out[(t_global * inner) + (h * HD) + d] = __float2half(out_val);
            }
        }

        t_chunk_start += L;
        __syncthreads();  // Before re-using s_k / s_q for the next chunk.
    }

    // Write final state back to global memory.
    {
        float* H_col = h_state + (static_cast<size_t>(h) * SS * HD) + d;
#pragma unroll
        for (int s = 0; s < SS; s++)
            H_col[static_cast<ptrdiff_t>(s * HD)] = H_reg[s];
    }
}

// ---------------------------------------------------------------------------
// Host launchers
// ---------------------------------------------------------------------------

// Phase 1b.1 chunkwise SSD scan host launcher: dispatches to gdn_scan_chunkwise_kernel
// <HD,SS,CHUNK> for supported shapes, else falls back to a chunk-iterating wrapper around
// gdn_scan_fused_f32. Output is bit-near-equivalent to gdn_scan_fused_kernel (FP16 1e-3,
// FP32 state 1e-5 tolerance), validated by ChunkBoundaryHandoff + ChunkwiseProtoMatchesFused.
// Templated dispatcher of gdn_scan_chunkwise_f32 (YOut=half; the YOut=float launcher left with
// gdn.fp32_scan, 2026-11-14).
template <typename YOut, typename FusedFallback>
static void gdn_scan_chunkwise_dispatch(const float* conv_f32, int conv_channels, const half* alpha,
                                        const half* beta, const float* A_log, const float* dt_bias,
                                        float* h_state, YOut* y, int n_tokens, int n_heads, int head_dim_ssm,
                                        int state_size, int n_groups, cudaStream_t stream, int chunk_size,
                                        int grouped_layout, const int* d_real_n, FusedFallback fused) {
    // Padded verify chunk (#847): host cannot split a device-side length across chunks, and
    // neither the chunkwise kernel nor the wrapper loop is device-length-aware. The whole
    // (small, <= k_max+1 token) padded chunk runs through one sequential fused call, committing
    // state at the real last row.
    if (d_real_n != nullptr) {
        fused(conv_f32, alpha, beta, h_state, y, n_tokens, d_real_n);
        return;
    }
    if (chunk_size <= 0)
        chunk_size = 64;
    chunk_size = std::min(chunk_size, n_tokens);

    // Direct chunkwise kernel dispatch for the supported (HD, SS, CHUNK) shapes.
    // Phase 1b.1 covers chunk_size==64 + HD==SS in {128, 64}, the production
    // GDN shapes (Qwen 3.5 / 3.6). Other sizes fall through to the wrapper.
    if (chunk_size == 64 && n_tokens >= 64) {
        if (head_dim_ssm == 128 && state_size == 128) {
            constexpr int HD = 128, SS = 128, CHUNK = 64;
            const size_t smem = ((2 * CHUNK * SS) + HD) * sizeof(float);
            static bool attr_set = false;
            if (!attr_set) {
                IMP_CUDA_CHECK_LOG(cudaFuncSetAttribute(
                    reinterpret_cast<const void*>(&gdn_scan_chunkwise_kernel<HD, SS, CHUNK, YOut>),
                    cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024));
                attr_set = true;
            }
            gdn_scan_chunkwise_kernel<HD, SS, CHUNK, YOut><<<n_heads, HD, smem, stream>>>(
                conv_f32, alpha, beta, A_log, dt_bias, h_state, y, n_tokens, n_heads, n_groups, conv_channels,
                grouped_layout);
            IMP_CUDA_CHECK_LAUNCH();
            return;
        }
        if (head_dim_ssm == 64 && state_size == 64) {
            constexpr int HD = 64, SS = 64, CHUNK = 64;
            const size_t smem = ((2 * CHUNK * SS) + HD) * sizeof(float);
            // 2 * 64 * 64 * 4 + 64 * 4 = 32 KiB + 256 B — within default static cap.
            gdn_scan_chunkwise_kernel<HD, SS, CHUNK, YOut><<<n_heads, HD, smem, stream>>>(
                conv_f32, alpha, beta, A_log, dt_bias, h_state, y, n_tokens, n_heads, n_groups, conv_channels,
                grouped_layout);
            IMP_CUDA_CHECK_LAUNCH();
            return;
        }
    }

    // Fallback: chunk-iterating wrapper around the sequential fused kernel, used for non-default
    // chunk sizes, unsupported HD/SS combos, and tail chunks (n_tokens < chunk_size). h_state
    // mutates in-place across per-chunk calls; same-stream submission keeps ordering.
    const int inner = n_heads * head_dim_ssm;
    int t = 0;
    while (t < n_tokens) {
        const int this_chunk = (t + chunk_size <= n_tokens) ? chunk_size : (n_tokens - t);
        fused(conv_f32 + (static_cast<size_t>(t) * conv_channels),
              alpha + (static_cast<ptrdiff_t>(t) * n_heads), beta + (static_cast<ptrdiff_t>(t) * n_heads),
              h_state, y + static_cast<size_t>(t) * inner, this_chunk, nullptr);
        t += this_chunk;
    }
}

void gdn_scan_chunkwise_f32(const float* conv_f32, int conv_channels, const half* alpha, const half* beta,
                            const float* A_log, const float* dt_bias, float* h_state, half* y, int n_tokens,
                            int n_heads, int head_dim_ssm, int state_size, int n_groups, cudaStream_t stream,
                            int chunk_size, int grouped_layout, const int* d_real_n, float* h_snap,
                            const int* d_snap_n, int64_t snap2_off) {
    // Snapshot row d_snap_n is in whole-range coordinates and the chunkwise kernel has no snapshot
    // write: a snapshot request runs the whole range as one fused scan (#2214).
    if (h_snap) {
        gdn_scan_fused_f32(conv_f32, conv_channels, alpha, beta, A_log, dt_bias, h_state, y, n_tokens,
                           n_heads, head_dim_ssm, state_size, n_groups, stream, grouped_layout, d_real_n,
                           h_snap, d_snap_n, snap2_off);
        return;
    }
    gdn_scan_chunkwise_dispatch<half>(
        conv_f32, conv_channels, alpha, beta, A_log, dt_bias, h_state, y, n_tokens, n_heads, head_dim_ssm,
        state_size, n_groups, stream, chunk_size, grouped_layout, d_real_n,
        [&](const float* row_conv, const half* row_alpha, const half* row_beta, float* h_state_, half* y_,
            int n_tok_chunk, const int* d_real_n_chunk) {
            gdn_scan_fused_f32(row_conv, conv_channels, row_alpha, row_beta, A_log, dt_bias, h_state_, y_,
                               n_tok_chunk, n_heads, head_dim_ssm, state_size, n_groups, stream,
                               grouped_layout, d_real_n_chunk, h_snap, d_snap_n);
        });
}

}  // namespace imp
