#include "compute/gdn_internal.cuh"
#include "core/logging.h"

#include <stdexcept>
#include <string>

namespace imp {

// Forward declarations
__global__ void gdn_scan_decode_kernel(const float*, const float*, const float*, const half*, const half*,
                                       const float*, const float*, float*, half*, const half*, int, int, int,
                                       int, int);

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
// SPLIT: how many threads cooperate on one state column. 1 is the original
// shape — one thread owns H[0..SS-1, d] entirely, which at SS=128 costs 128
// registers for the state alone and made ptxas spill (88 B stack, 128 B spill
// stores, 255 registers used, 2 blocks/SM, 8.3 % occupancy). With SPLIT=2 each
// thread owns half a column, the two partners sit in ADJACENT lanes so the two
// dot products reduce with a single __shfl_xor_sync, and the register pressure
// halves.
template <int HD, int SS, typename YOut, int SPLIT = 1>
__global__ void __launch_bounds__(HD * SPLIT, 1) gdn_scan_fused_kernel(
    const float* __restrict__ conv_f32,  // [n_tokens, conv_channels] FP32
    const half* __restrict__ alpha_all,  // [n_tokens, n_heads] FP16
    const half* __restrict__ beta_all,   // [n_tokens, n_heads] FP16
    const float* __restrict__ A_log,     // [n_heads] FP32
    const float* __restrict__ dt_bias,   // [n_heads] FP32
    float* __restrict__ h_state,         // [n_heads, SS, HD] FP32
    YOut* __restrict__ y_out,            // [n_tokens, n_heads * HD] FP16 or FP32
    int n_tokens, int n_heads, int n_groups, int conv_channels, int grouped_layout,
    const int* __restrict__ d_real_n,    // device chunk length (padded verify chunk) or nullptr
    float* __restrict__ h_snap,          // second state output, or nullptr
    const int* __restrict__ d_snap_n,    // row count h_snap is taken at
    // --- batched decode over independent sequences (#GDN-batch) ---
    // seq_slots: per-blockIdx.y recurrent-state slot id, or nullptr for the
    // single-sequence call (then h_state is already the caller's slot pointer
    // and gridDim.y is 1, so every offset below is zero and the emitted code
    // is the one this kernel always ran).
    // Concurrent GDN sequences are INDEPENDENT — each owns its own state slot —
    // so they parallelise across blockIdx.y exactly like heads do across
    // blockIdx.x. Tokens within one sequence stay sequential, which is the part
    // that genuinely cannot batch.
    const int* __restrict__ seq_slots = nullptr,
    int64_t h_state_seq_stride = 0) {
    const int h = blockIdx.x;
    if (h >= n_heads)
        return;
    const int seq = blockIdx.y;
    // SPLIT partners are ADJACENT lanes (d = tid / SPLIT), so the cross-partner
    // reductions below are a warp shuffle rather than a shared-memory round.
    const int d = (SPLIT == 1) ? static_cast<int>(threadIdx.x) : static_cast<int>(threadIdx.x) / SPLIT;
    const int part = (SPLIT == 1) ? 0 : static_cast<int>(threadIdx.x) % SPLIT;
    constexpr int SS_PER = SS / SPLIT;   // state rows this thread owns
    const int s_base = part * SS_PER;
    // Padded verify chunk (#847): y is produced for every row (padding rows
    // are causally invisible downstream), but the committed h_state is the
    // register snapshot at the real last row — H_reg keeps evolving through
    // the pads only to define their (discarded) y values.
    const int real_n = d_real_n ? min(n_tokens, __ldg(d_real_n)) : n_tokens;
    // Second commit row for the speculative verify: the state as of snap_n
    // rows, written alongside the one at real_n. 0 disables it.
    const int snap_n = (h_snap && d_snap_n) ? min(n_tokens, __ldg(d_snap_n)) : 0;

    // Head-to-K-group mapping. GGUF stores heads in tiled layout where head h's
    // group is `h % n_groups`. HF SafeTensors (Qwen3.5/3.6) stores heads in
    // grouped layout where head h's group is `h / (n_heads / n_groups)`.
    // grouped_layout=1 selects the HF formula.
    const int g = grouped_layout ? (h / (n_heads / n_groups)) : (h % n_groups);
    const int inner = n_heads * HD;

    // Rebase every per-sequence pointer. With gridDim.y == 1 and no slot table
    // these are all +0.
    if (seq_slots) {
        h_state += static_cast<size_t>(seq_slots[seq]) * static_cast<size_t>(h_state_seq_stride);
    }
    conv_f32 += static_cast<size_t>(seq) * n_tokens * conv_channels;
    alpha_all += static_cast<size_t>(seq) * n_tokens * n_heads;
    beta_all += static_cast<size_t>(seq) * n_tokens * n_heads;
    y_out += static_cast<size_t>(seq) * n_tokens * inner;
    const int BC_size = n_groups * SS;
    const float scale = rsqrtf(static_cast<float>(HD));

    // Load per-head constants
    const float A_h = A_log[h];
    const float dtb_h = dt_bias[h];

    // Load state into registers — the critical optimization.
    // Each thread holds SS floats = one column of H[SS, HD].
    float H_reg[SS_PER];
    {
        const float* H_col = h_state + static_cast<size_t>(h) * SS * HD + d;
#pragma unroll
        for (int s = 0; s < SS_PER; s++)
            H_reg[s] = H_col[(s_base + s) * HD];
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
            // Indexed by the REAL thread id, not by the column: with SPLIT > 1
            // two threads share `d` and would both write s_k[d] and s_reduce[d].
            constexpr int NT = HD * SPLIT;
            const int tid = static_cast<int>(threadIdx.x);
            // Load K and Q into shared memory
            if (tid < SS) {
                s_k[tid] = K_g[tid];
                s_q[tid] = Q_g[tid];
            }
            __syncthreads();

            // Parallel sum-of-squares reduction
            float k_sq = 0.0f, q_sq = 0.0f;
            for (int i = tid; i < SS; i += NT) {
                k_sq += s_k[i] * s_k[i];
                q_sq += s_q[i] * s_q[i];
            }
            // Block reduction for k_sq
            s_reduce[tid] = k_sq;
            __syncthreads();
            for (int stride = NT / 2; stride > 0; stride >>= 1) {
                if (tid < stride)
                    s_reduce[tid] += s_reduce[tid + stride];
                __syncthreads();
            }
            // PyTorch-style L2 norm (matches llama's ggml_l2_norm): rsqrtf(max(sum_sq, eps^2)).
            // Additive eps (sum + eps) over-clamps near-zero heads and produces
            // 100-1000x too-small normalization scale vs llama, which breaks Qwen 3.6 scan
            // outputs at layers where some heads have near-zero K (e.g. L1 h19/20/22/25/29).
            float k_inv = rsqrtf(fmaxf(s_reduce[0], 1e-12f));

            // Every thread has to finish READING s_reduce[0] above before any
            // thread overwrites s_reduce for the q reduction. Without this
            // barrier thread 0 can store q_sq into s_reduce[0] while a slower
            // warp is still loading k_inv from it, and that warp normalises K
            // by Q's norm.
            //
            // Latent since the kernel was written and invisible at the shipped
            // grid: (n_heads) is at most 48 blocks here, well under the 170 SMs,
            // so every block had an SM to itself and its four warps ran in
            // lockstep. Batching decode over sequences makes the grid
            // (n_heads x n_seq) — at 256 blocks the race fires and one block's
            // state comes out different run to run (measured: stable through
            // 128 blocks, 16384 of 4194304 floats differing at 256).
            __syncthreads();
            s_reduce[tid] = q_sq;
            __syncthreads();
            for (int stride = NT / 2; stride > 0; stride >>= 1) {
                if (tid < stride)
                    s_reduce[tid] += s_reduce[tid + stride];
                __syncthreads();
            }
            float q_inv = rsqrtf(fmaxf(s_reduce[0], 1e-12f));

            // Normalize in-place
            __syncthreads();
            if (tid < SS) {
                s_k[tid] *= k_inv;
                s_q[tid] *= q_inv;
            }
            __syncthreads();
        }

        // Delta rule scan — all in registers, no global memory access
        // Step 1: kv = H^T @ k_norm (dot product of state column with k)
        float kv_d = 0.0f;
#pragma unroll
        for (int s = 0; s < SS_PER; s++)
            kv_d += H_reg[s] * s_k[s_base + s];
        if constexpr (SPLIT > 1) {
            // Sum the partners' halves. Adjacent lanes, so one shuffle.
#pragma unroll
            for (int off = 1; off < SPLIT; off <<= 1)
                kv_d += __shfl_xor_sync(0xffffffffu, kv_d, off);
        }

        // Step 2: delta = (v - g*kv) * beta
        float delta_d = (v_d - g_t * kv_d) * beta_h;

        // Step 3: Update state + compute output
        float y_partial = 0.0f;
#pragma unroll
        for (int s = 0; s < SS_PER; s++) {
            float h_new = g_t * H_reg[s] + s_k[s_base + s] * delta_d;
            H_reg[s] = h_new;
            y_partial += h_new * s_q[s_base + s];
        }
        if constexpr (SPLIT > 1) {
#pragma unroll
            for (int off = 1; off < SPLIT; off <<= 1)
                y_partial += __shfl_xor_sync(0xffffffffu, y_partial, off);
        }

        if (part == 0) {
            const float out_val = y_partial * scale;
            if constexpr (std::is_same_v<YOut, float>) {
                y_out[t * inner + h * HD + d] = out_val;
            } else {
                y_out[t * inner + h * HD + d] = __float2half(out_val);
            }
        }

        // Commit the state at the real last row (per-thread register column,
        // no barrier needed). For the unpadded case real_n == n_tokens and
        // this is the single end-of-scan store the kernel always did.
        if (t + 1 == real_n) {
            float* H_col = h_state + static_cast<size_t>(h) * SS * HD + d;
#pragma unroll
            for (int s = 0; s < SS_PER; s++)
                H_col[(s_base + s) * HD] = H_reg[s];
        }
        if (t + 1 == snap_n) {
            float* S_col = h_snap + static_cast<size_t>(h) * SS * HD + d;
#pragma unroll
            for (int s = 0; s < SS_PER; s++)
                S_col[(s_base + s) * HD] = H_reg[s];
        }

        // Sync before next token — the next iteration overwrites s_k/s_q in
        // shared memory. Without this barrier, fast threads can overwrite
        // s_k/s_q while slow threads are still reading them in the loops above.
        if (t + 1 < n_tokens)
            __syncthreads();
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
    half* __restrict__ y,             // [n_tokens, n_heads * head_dim] in/out
    const half* __restrict__ gate,    // [n_tokens, n_heads * head_dim]
    const half* __restrict__ weight,  // [head_dim] shared norm weight
    float eps, int n_heads, int head_dim) {
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const int d = threadIdx.x;
    if (d >= head_dim)
        return;

    const int inner = n_heads * head_dim;
    const int base = t * inner + h * head_dim;

    // Load y value
    float val = __half2float(y[base + d]);

    // Parallel sum-of-squares for RMSNorm
    extern __shared__ float s_buf[];
    s_buf[d] = val * val;
    __syncthreads();
    for (int stride = head_dim / 2; stride > 0; stride >>= 1) {
        if (d < stride)
            s_buf[d] += s_buf[d + stride];
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
// V-head reorder: tiled -> grouped (undo GGUF converter reorder for ssm_out)
//
// FP16 twin removed 2026-08-21: it and its kernel were declared, defined and
// called nowhere. The only consumer of `gdn.vhead_reorder`
// (executor_ssm_gdn.cu:421) gathers V out of the FP32 conv1d output, so it has
// only ever called the _f32 variant below.
// ---------------------------------------------------------------------------
// FP32 variant for conv1d-SiLU output (= scan V input). Same math as FP16,
// different element type. Used when the GGUF stored V in tiled layout and the
// scan kernel reads V[h*HD+d] assuming grouped layout.
__global__ void vhead_tiled_to_grouped_f32_kernel(const float* __restrict__ src, float* __restrict__ dst,
                                                  int n_tokens, int n_heads, int head_dim, int n_groups) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_tokens * n_heads * head_dim;
    if (tid >= total)
        return;

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

void vhead_tiled_to_grouped_f32(const float* src, float* dst, int n_tokens, int n_heads, int head_dim,
                                int n_groups, cudaStream_t stream) {
    if (n_heads == n_groups)
        return;
    int total = n_tokens * n_heads * head_dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    vhead_tiled_to_grouped_f32_kernel<<<blocks, threads, 0, stream>>>(src, dst, n_tokens, n_heads, head_dim,
                                                                      n_groups);
    IMP_CUDA_CHECK_LAUNCH();
}

// ---------------------------------------------------------------------------
// Host launchers
// ---------------------------------------------------------------------------


// Batched fused scan over N INDEPENDENT sequences, n_tokens rows each.
//
// This is what lets concurrent GDN decode batch. The recurrent scan is
// sequential in TOKENS, which is why a single sequence cannot be parallelised
// over its own timeline — but separate sequences share nothing except the
// weights, so they map onto blockIdx.y the same way heads map onto blockIdx.x.
// At decode n_tokens is 1 and the launch goes from (n_heads) blocks, which
// leaves an RTX 5090 almost idle, to (n_heads x n_seq).
//
// h_state_pool: base of the recurrent-state pool; seq_slots[i] selects the slot
// for sequence i (device pointer, n_seq ints). h_state_seq_stride is the
// distance between slots in FLOATS.
// conv_f32 / alpha / beta / y are [n_seq * n_tokens, ...] with sequence-major
// rows — the layout a batched decode step already produces.
void gdn_scan_fused_f32_batched(const float* conv_f32, int conv_channels, const half* alpha,
                                const half* beta, const float* A_log, const float* dt_bias,
                                float* h_state_pool, const int* seq_slots, int64_t h_state_seq_stride,
                                half* y, int n_seq, int n_tokens, int n_heads, int head_dim_ssm,
                                int state_size, int n_groups, cudaStream_t stream, int grouped_layout,
                                const int* d_real_n) {
    if (n_seq <= 0)
        return;
    dim3 grid(n_heads, n_seq);
    if (head_dim_ssm == 128 && state_size == 128) {
        // SPLIT=2: two threads per state column. At SS=128 one-thread-per-column
        // needs 128 registers for the state alone and ptxas spills (255 used,
        // 88 B stack); halving the slice removes the spill and doubles the
        // blocks that fit per SM.
        constexpr int SPLIT = 2;
        const size_t smem = (2 * 128 + 128 * SPLIT) * sizeof(float);
        gdn_scan_fused_kernel<128, 128, half, SPLIT><<<grid, 128 * SPLIT, smem, stream>>>(
            conv_f32, alpha, beta, A_log, dt_bias, h_state_pool, y, n_tokens, n_heads, n_groups,
            conv_channels, grouped_layout, d_real_n, nullptr, nullptr, seq_slots, h_state_seq_stride);
        IMP_CUDA_CHECK_LAUNCH();
    } else if (head_dim_ssm == 64 && state_size == 64) {
        const size_t smem = (2 * 64 + 64) * sizeof(float);
        gdn_scan_fused_kernel<64, 64, half><<<grid, 64, smem, stream>>>(
            conv_f32, alpha, beta, A_log, dt_bias, h_state_pool, y, n_tokens, n_heads, n_groups,
            conv_channels, grouped_layout, d_real_n, nullptr, nullptr, seq_slots, h_state_seq_stride);
        IMP_CUDA_CHECK_LAUNCH();
    } else {
        IMP_LOG_ERROR("gdn_scan_fused_f32_batched: unsupported head_dim=%d state_size=%d", head_dim_ssm,
                      state_size);
    }
}

// Fused scan: processes all tokens in one kernel launch.
// conv_f32: [n_tokens, conv_channels] FP32 — full conv output (Q|K|V interleaved per token)
// grouped_layout: 0 = GGUF tiled (g = h % n_groups), 1 = HF SafeTensors grouped
//                 (g = h / n_v_per_k). See kernel comment for details.
void gdn_scan_fused_f32(const float* conv_f32, int conv_channels, const half* alpha, const half* beta,
                        const float* A_log, const float* dt_bias, float* h_state, half* y, int n_tokens,
                        int n_heads, int head_dim_ssm, int state_size, int n_groups, cudaStream_t stream,
                        int grouped_layout, const int* d_real_n) {
    // Shared memory: K_norm[SS] + Q_norm[SS] + reduce[HD]
    size_t smem = (2 * state_size + head_dim_ssm) * sizeof(float);

    // Template dispatch for common sizes
    if (head_dim_ssm == 128 && state_size == 128) {
        gdn_scan_fused_kernel<128, 128, half>
            <<<n_heads, 128, smem, stream>>>(conv_f32, alpha, beta, A_log, dt_bias, h_state, y, n_tokens,
                                             n_heads, n_groups, conv_channels, grouped_layout, d_real_n,
                                             nullptr, nullptr);
        IMP_CUDA_CHECK_LAUNCH();
    } else if (head_dim_ssm == 64 && state_size == 64) {
        gdn_scan_fused_kernel<64, 64, half>
            <<<n_heads, 64, smem, stream>>>(conv_f32, alpha, beta, A_log, dt_bias, h_state, y, n_tokens,
                                            n_heads, n_groups, conv_channels, grouped_layout, d_real_n,
                                            nullptr, nullptr);
        IMP_CUDA_CHECK_LAUNCH();
    } else {
        // Fallback: per-token loop (for unsupported HD/SS sizes). The host
        // loop cannot bound itself by a device-side length — reject padded
        // verify chunks loudly instead of advancing state through pad rows.
        if (d_real_n != nullptr)
            throw std::runtime_error("gdn_scan_fused_f32: padded verify chunk unsupported for HD=" +
                                     std::to_string(head_dim_ssm) + " SS=" + std::to_string(state_size));
        int inner = n_heads * head_dim_ssm;
        int BC_size = n_groups * state_size;
        size_t smem_old = 2 * state_size * sizeof(float) + 2 * sizeof(float);
        for (int t = 0; t < n_tokens; t++) {
            const float* row = conv_f32 + static_cast<size_t>(t) * conv_channels;
            gdn_scan_decode_kernel<<<n_heads, head_dim_ssm, smem_old, stream>>>(
                row + 2 * BC_size, row + BC_size, row, alpha + t * n_heads, beta + t * n_heads, A_log,
                dt_bias, h_state, y + t * inner, nullptr, n_heads, head_dim_ssm, state_size, n_groups,
                grouped_layout);
            IMP_CUDA_CHECK_LAUNCH();
        }
    }
}

// FP32-output variant — writes scan result as FP32 for downstream
// FP32-input RMSNorm+Gate (avoids FP16 subnormal-truncation at ~6e-5).
void gdn_scan_fused_fp32out(const float* conv_f32, int conv_channels, const half* alpha, const half* beta,
                            const float* A_log, const float* dt_bias, float* h_state, float* y_fp32,
                            int n_tokens, int n_heads, int head_dim_ssm, int state_size, int n_groups,
                            cudaStream_t stream, int grouped_layout, const int* d_real_n, float* h_snap,
                            const int* d_snap_n) {
    size_t smem = (2 * state_size + head_dim_ssm) * sizeof(float);
    if (head_dim_ssm == 128 && state_size == 128) {
        gdn_scan_fused_kernel<128, 128, float>
            <<<n_heads, 128, smem, stream>>>(conv_f32, alpha, beta, A_log, dt_bias, h_state, y_fp32, n_tokens,
                                             n_heads, n_groups, conv_channels, grouped_layout, d_real_n,
                                             h_snap, d_snap_n);
        IMP_CUDA_CHECK_LAUNCH();
    } else if (head_dim_ssm == 64 && state_size == 64) {
        gdn_scan_fused_kernel<64, 64, float>
            <<<n_heads, 64, smem, stream>>>(conv_f32, alpha, beta, A_log, dt_bias, h_state, y_fp32, n_tokens,
                                            n_heads, n_groups, conv_channels, grouped_layout, d_real_n,
                                            h_snap, d_snap_n);
        IMP_CUDA_CHECK_LAUNCH();
    } else {
        // Refuse, do not approximate. This branch used to run the FP16 decode
        // kernel into a scratch buffer, log a WARN that the FP32 conversion was
        // "not implemented", and free the scratch — leaving `y_fp32` exactly as
        // the caller passed it in. The scan silently contributed nothing, the
        // step still returned, and the only trace was one warning in a log
        // nobody reads per token. The same failure shape as #654: an unchecked
        // fallback that keeps serving.
        //
        // Every GDN checkpoint staged here is HD=SS=128 (Qwen3.5/3.6/3.8:
        // linear_key_head_dim = linear_value_head_dim = 128), so nothing reaches
        // this today — which is precisely why it could sit unfinished. A shape
        // that does arrive now says so at the first token instead of producing
        // an answer nobody can tell from a correct one. Mamba2 checkpoints with
        // other shapes (Nemotron-3.5: mamba_head_dim 64, ssm_state_size 128) use
        // the Mamba2 scan, not this one.
        throw std::runtime_error("gdn_scan_fused_fp32out: no kernel for HD=" + std::to_string(head_dim_ssm) +
                                 " SS=" + std::to_string(state_size) +
                                 " (supported: 128/128 and 64/64). The FP16 decode fallback never wrote "
                                 "its FP32 output, so serving this shape would return a scan that did "
                                 "nothing.");
    }
}

// ---------------------------------------------------------------------------
// Reference scan kernel — unfused semantics for validation.
//
// Design: one CUDA block per v_head, block size = head_dim_v. Each block owns
// the full [state_size, head_dim_v] state slab for one head. State is kept in
// shared memory for the duration of the per-token loop and written back to
// global at the end. Differences vs. `gdn_scan_fused_kernel`:
//   - State in SHARED memory (not per-thread registers). Easier to reason
//     about; no cross-thread register ownership of state columns.
//   - Q, K, V, alpha, beta loaded afresh each token from global via shared
//     (no reuse across tokens).
//   - L2-norm of Q, K uses the standard block-reduce pattern.
//   - No `#pragma unroll` over state_size — keeps the inner loop predictable.
// Math is identical to the fused kernel. If outputs differ, the fused
// kernel has a correctness bug (register lifetime, sync, or dataflow).
// ---------------------------------------------------------------------------
__global__ void gdn_scan_reference_kernel(
    const float* __restrict__ conv_f32,  // [n_tokens, conv_channels] FP32
    const half* __restrict__ alpha_all,  // [n_tokens, n_heads] FP16
    const half* __restrict__ beta_all,   // [n_tokens, n_heads] FP16
    const float* __restrict__ A_log,     // [n_heads] FP32
    const float* __restrict__ dt_bias,   // [n_heads] FP32
    float* __restrict__ h_state,         // [n_heads, state_size, head_dim_v] FP32
    half* __restrict__ y_out,            // [n_tokens, n_heads * head_dim_v] FP16
    int n_tokens, int n_heads, int n_groups, int head_dim_v, int state_size, int conv_channels,
    int grouped_layout, const int* __restrict__ d_real_n) {
    const int h = blockIdx.x;
    if (h >= n_heads)
        return;
    const int d = threadIdx.x;  // [0, head_dim_v)
    const int SS = state_size;
    const int HD = head_dim_v;
    // Padded verify chunk (#847): commit the state at the real last row; pads
    // only define their (discarded) y rows. See gdn_scan_fused_kernel.
    const int real_n = d_real_n ? min(n_tokens, __ldg(d_real_n)) : n_tokens;

    const int g = grouped_layout ? (h / (n_heads / n_groups)) : (h % n_groups);
    const int inner = n_heads * HD;
    const int BC_size = n_groups * SS;

    const float A_h = A_log[h];
    const float dtb_h = dt_bias[h];

    // Shared memory layout:
    //   s_H[SS * HD]       — state slab (owned by this block, this head)
    //   s_k[SS]            — K for current token (after L2-norm)
    //   s_q[SS]            — Q for current token (after L2-norm)
    //   s_v[HD]            — V for current token
    //   s_reduce[HD]       — block reduction scratch
    extern __shared__ float smem[];
    float* s_H = smem;
    float* s_k = s_H + SS * HD;
    float* s_q = s_k + SS;
    float* s_v = s_q + SS;
    float* s_reduce = s_v + HD;

    // Load state from global into shared.
    {
        const float* H_src = h_state + static_cast<size_t>(h) * SS * HD;
        for (int idx = d; idx < SS * HD; idx += HD) {
            s_H[idx] = H_src[idx];
        }
    }
    __syncthreads();

    for (int t = 0; t < n_tokens; t++) {
        const float* row = conv_f32 + static_cast<size_t>(t) * conv_channels;
        const float* Q_g = row + g * SS;
        const float* K_g = row + BC_size + g * SS;
        const float* V_base = row + 2 * BC_size + h * HD;

        // Load V (one element per thread)
        s_v[d] = V_base[d];

        // Load Q, K into shared (only first SS threads)
        if (d < SS) {
            s_q[d] = Q_g[d];
            s_k[d] = K_g[d];
        }
        __syncthreads();

        // Per-head scalars
        float alpha_h = __half2float(alpha_all[t * n_heads + h]);
        float dt_val = alpha_h + dtb_h;
        dt_val = (dt_val > 20.0f) ? dt_val : logf(1.0f + expf(dt_val));
        float g_t = expf(fmaxf(A_h * dt_val, -20.0f));

        float beta_h = __half2float(beta_all[t * n_heads + h]);
        beta_h = 1.0f / (1.0f + expf(-fmaxf(fminf(beta_h, 20.0f), -20.0f)));

        // L2-normalize Q, K across state_size.
        // Each thread contributes one element (if d < SS) to sum-of-squares.
        float k_sq = (d < SS) ? s_k[d] * s_k[d] : 0.0f;
        float q_sq = (d < SS) ? s_q[d] * s_q[d] : 0.0f;

        s_reduce[d] = k_sq;
        __syncthreads();
        for (int stride = HD / 2; stride > 0; stride >>= 1) {
            if (d < stride)
                s_reduce[d] += s_reduce[d + stride];
            __syncthreads();
        }
        // PyTorch-style L2 norm (see note in fused kernel above).
        float k_inv = rsqrtf(fmaxf(s_reduce[0], 1e-12f));

        s_reduce[d] = q_sq;
        __syncthreads();
        for (int stride = HD / 2; stride > 0; stride >>= 1) {
            if (d < stride)
                s_reduce[d] += s_reduce[d + stride];
            __syncthreads();
        }
        float q_inv = rsqrtf(fmaxf(s_reduce[0], 1e-12f));

        if (d < SS) {
            s_k[d] *= k_inv;
            s_q[d] *= q_inv;
        }
        __syncthreads();

        // Delta rule scan for this token.
        // Each thread owns column d of the state: s_H[s * HD + d] for s in [0, SS).
        //
        // kv[d] = sum_s H[s, d] * k_norm[s]
        float kv_d = 0.0f;
        for (int s = 0; s < SS; s++) {
            kv_d += s_H[s * HD + d] * s_k[s];
        }

        // delta[d] = (v[d] - g*kv[d]) * beta
        float delta_d = (s_v[d] - g_t * kv_d) * beta_h;

        // H_new[s, d] = g * H[s, d] + k_norm[s] * delta[d]
        // y[d] = sum_s H_new[s, d] * q_norm[s]
        float y_partial = 0.0f;
        for (int s = 0; s < SS; s++) {
            float h_new = g_t * s_H[s * HD + d] + s_k[s] * delta_d;
            s_H[s * HD + d] = h_new;
            y_partial += h_new * s_q[s];
        }

        y_out[t * inner + h * HD + d] = __float2half(y_partial * rsqrtf(static_cast<float>(HD)));

        // Commit the state at the real last row. Each thread copies exactly
        // the column (d, d+HD, ...) it wrote above — no barrier needed.
        if (t + 1 == real_n) {
            float* H_dst = h_state + static_cast<size_t>(h) * SS * HD;
            for (int idx = d; idx < SS * HD; idx += HD) {
                H_dst[idx] = s_H[idx];
            }
        }

        __syncthreads();  // before next token overwrites s_k/s_q/s_v
    }
}

void gdn_scan_reference_f32(const float* conv_f32, int conv_channels, const half* alpha, const half* beta,
                            const float* A_log, const float* dt_bias, float* h_state, half* y, int n_tokens,
                            int n_heads, int head_dim_ssm, int state_size, int n_groups, cudaStream_t stream,
                            int grouped_layout, const int* d_real_n) {
    // Shared memory: state slab [SS*HD] + K[SS] + Q[SS] + V[HD] + reduce[HD]
    // For Qwen 3.6 (SS=128, HD=128): ~66 KB — exceeds 48 KB default per block,
    // needs the opt-in dynamic-shared attribute.
    size_t smem = (state_size * head_dim_ssm + 2 * state_size + 2 * head_dim_ssm) * sizeof(float);
    static bool attr_set = false;
    if (!attr_set) {
        cudaFuncSetAttribute(reinterpret_cast<const void*>(&gdn_scan_reference_kernel),
                             cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024);
        attr_set = true;
    }
    gdn_scan_reference_kernel<<<n_heads, head_dim_ssm, smem, stream>>>(conv_f32, alpha, beta, A_log, dt_bias,
                                                                       h_state, y, n_tokens, n_heads,
                                                                       n_groups, head_dim_ssm, state_size,
                                                                       conv_channels, grouped_layout,
                                                                       d_real_n);
    IMP_CUDA_CHECK_LAUNCH();
}

// FP32-input variant: reads y as FP32, writes FP16. Used together with
// `gdn_scan_fused_fp32out` so the RMS reduction sees full-precision scan output
// (without FP16 subnormal truncation at ~6e-5).
__global__ void gdn_rmsnorm_gated_silu_fp32in_kernel(
    half* __restrict__ y_fp16_out,        // [n_tokens, n_heads * head_dim]
    const float* __restrict__ y_fp32_in,  // [n_tokens, n_heads * head_dim]
    const half* __restrict__ gate, const half* __restrict__ weight, float eps, int n_heads, int head_dim) {
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const int d = threadIdx.x;
    if (d >= head_dim)
        return;

    const int inner = n_heads * head_dim;
    const int base = t * inner + h * head_dim;

    float val = y_fp32_in[base + d];

    extern __shared__ float s_buf[];
    s_buf[d] = val * val;
    __syncthreads();
    for (int stride = head_dim / 2; stride > 0; stride >>= 1) {
        if (d < stride)
            s_buf[d] += s_buf[d + stride];
        __syncthreads();
    }
    float inv_rms = rsqrtf(s_buf[0] / static_cast<float>(head_dim) + eps);

    float normed = val * inv_rms * __half2float(weight[d]);

    float g = __half2float(gate[base + d]);
    float silu_g = g / (1.0f + expf(-g));

    y_fp16_out[base + d] = __float2half(normed * silu_g);
}

void gdn_rmsnorm_gated_silu_fp32in(half* y_fp16_out, const float* y_fp32_in, const half* gate,
                                   const half* weight, float eps, int n_tokens, int n_heads, int head_dim,
                                   cudaStream_t stream) {
    size_t smem = head_dim * sizeof(float);
    dim3 grid(n_tokens, n_heads);
    gdn_rmsnorm_gated_silu_fp32in_kernel<<<grid, head_dim, smem, stream>>>(y_fp16_out, y_fp32_in, gate,
                                                                           weight, eps, n_heads, head_dim);
    IMP_CUDA_CHECK_LAUNCH();
}

// FP32-in, FP32-out: keeps full precision through gated norm so ssm_out GEMM
// sees FP32 input (fixes 6% accumulation drift in FP16-input matmul).
__global__ void gdn_rmsnorm_gated_silu_fp32inout_kernel(float* __restrict__ y_fp32_out,
                                                        const float* __restrict__ y_fp32_in,
                                                        const half* __restrict__ gate,
                                                        const half* __restrict__ weight, float eps,
                                                        int n_heads, int head_dim) {
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const int d = threadIdx.x;
    if (d >= head_dim)
        return;

    const int inner = n_heads * head_dim;
    const int base = t * inner + h * head_dim;

    float val = y_fp32_in[base + d];

    extern __shared__ float s_buf[];
    s_buf[d] = val * val;
    __syncthreads();
    for (int stride = head_dim / 2; stride > 0; stride >>= 1) {
        if (d < stride)
            s_buf[d] += s_buf[d + stride];
        __syncthreads();
    }
    float inv_rms = rsqrtf(s_buf[0] / static_cast<float>(head_dim) + eps);

    float normed = val * inv_rms * __half2float(weight[d]);

    float g = __half2float(gate[base + d]);
    float silu_g = g / (1.0f + expf(-g));

    y_fp32_out[base + d] = normed * silu_g;
}

void gdn_rmsnorm_gated_silu_fp32inout(float* y_fp32_out, const float* y_fp32_in, const half* gate,
                                      const half* weight, float eps, int n_tokens, int n_heads, int head_dim,
                                      cudaStream_t stream) {
    size_t smem = head_dim * sizeof(float);
    dim3 grid(n_tokens, n_heads);
    gdn_rmsnorm_gated_silu_fp32inout_kernel<<<grid, head_dim, smem, stream>>>(y_fp32_out, y_fp32_in, gate,
                                                                              weight, eps, n_heads, head_dim);
    IMP_CUDA_CHECK_LAUNCH();
}

// Fused RMSNormGated + SiLU
void gdn_rmsnorm_gated_silu(half* y, const half* gate, const half* weight, float eps, int n_tokens,
                            int n_heads, int head_dim, cudaStream_t stream) {
    size_t smem = head_dim * sizeof(float);
    dim3 grid(n_tokens, n_heads);
    gdn_rmsnorm_gated_silu_kernel<<<grid, head_dim, smem, stream>>>(y, gate, weight, eps, n_heads, head_dim);
    IMP_CUDA_CHECK_LAUNCH();
}

// ---------------------------------------------------------------------------
// Legacy interfaces (kept for backward compatibility)
// ---------------------------------------------------------------------------

// Old per-token decode kernel (still available for reference)
__global__ void gdn_scan_decode_kernel(const float* __restrict__ x, const float* __restrict__ B_in,
                                       const float* __restrict__ C_in, const half* __restrict__ alpha_raw,
                                       const half* __restrict__ beta_raw, const float* __restrict__ A_log,
                                       const float* __restrict__ dt_bias, float* __restrict__ h_state,
                                       half* __restrict__ y, const half* __restrict__ z, int n_heads,
                                       int head_dim_ssm, int state_size, int n_groups, int grouped_layout) {
    const int h = blockIdx.x;
    if (h >= n_heads)
        return;
    const int d = threadIdx.x;
    if (d >= head_dim_ssm)
        return;

    const int g = grouped_layout ? (h / (n_heads / n_groups)) : (h % n_groups);
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
            float ks = K_g[s];
            float qs = Q_g[s];
            s_k[s] = ks;
            s_q[s] = qs;
            k_sq += ks * ks;
            q_sq += qs * qs;
        }
        // PyTorch-style L2 norm: rsqrtf(max(sum_sq, eps^2)), matches llama's ggml_l2_norm.
        s_k_inv = rsqrtf(fmaxf(k_sq, 1e-12f));
        s_q_inv = rsqrtf(fmaxf(q_sq, 1e-12f));
        for (int s = 0; s < state_size; s++) {
            s_k[s] *= s_k_inv;
            s_q[s] *= s_q_inv;
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

void gdn_scan_decode_f32(const float* x, const float* B, const float* C, const half* alpha, const half* beta,
                         const float* A_log, const float* dt_bias, float* h_state, half* y, const half* z,
                         int n_heads, int head_dim_ssm, int state_size, int n_groups, cudaStream_t stream,
                         int grouped_layout) {
    size_t smem = 2 * state_size * sizeof(float) + 2 * sizeof(float);
    gdn_scan_decode_kernel<<<n_heads, head_dim_ssm, smem, stream>>>(x, B, C, alpha, beta, A_log, dt_bias,
                                                                    h_state, y, z, n_heads, head_dim_ssm,
                                                                    state_size, n_groups, grouped_layout);
    IMP_CUDA_CHECK_LAUNCH();
}

void gdn_scan_prefill_f32(const float* x, const float* B, const float* C, const half* alpha, const half* beta,
                          const float* A_log, const float* dt_bias, float* h_state, half* y, const half* z,
                          int n_tokens, int n_heads, int head_dim_ssm, int state_size, int n_groups,
                          cudaStream_t stream, int grouped_layout) {
    int inner = n_heads * head_dim_ssm;
    int BC_size = n_groups * state_size;
    size_t smem = 2 * state_size * sizeof(float) + 2 * sizeof(float);
    for (int t = 0; t < n_tokens; t++) {
        gdn_scan_decode_kernel<<<n_heads, head_dim_ssm, smem, stream>>>(x + t * inner, B + t * BC_size,
                                                                        C + t * BC_size, alpha + t * n_heads,
                                                                        beta + t * n_heads, A_log, dt_bias,
                                                                        h_state, y + t * inner, nullptr,
                                                                        n_heads, head_dim_ssm, state_size,
                                                                        n_groups, grouped_layout);
        IMP_CUDA_CHECK_LAUNCH();
    }
}

}  // namespace imp
