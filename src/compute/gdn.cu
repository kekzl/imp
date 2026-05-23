#include "compute/gdn.h"
#include "core/logging.h"
#include <cmath>
#include <type_traits>
#include <mma.h>

namespace imp {

namespace wmma = nvcuda::wmma;

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
template <int HD, int SS, typename YOut>
__global__ void __launch_bounds__(HD, 1) gdn_scan_fused_kernel(
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

    // Head-to-K-group mapping. GGUF stores heads in tiled layout where head h's
    // group is `h % n_groups`. HF SafeTensors (Qwen3.5/3.6) stores heads in
    // grouped layout where head h's group is `h / (n_heads / n_groups)`.
    // grouped_layout=1 selects the HF formula.
    const int g = grouped_layout ? (h / (n_heads / n_groups)) : (h % n_groups);
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
                if (d < stride)
                    s_reduce[d] += s_reduce[d + stride];
                __syncthreads();
            }
            // PyTorch-style L2 norm (matches llama's ggml_l2_norm): rsqrtf(max(sum_sq, eps^2)).
            // Additive eps (sum + eps) over-clamps near-zero heads and produces
            // 100-1000x too-small normalization scale vs llama, which breaks Qwen 3.6 scan
            // outputs at layers where some heads have near-zero K (e.g. L1 h19/20/22/25/29).
            float k_inv = rsqrtf(fmaxf(s_reduce[0], 1e-12f));

            s_reduce[d] = q_sq;
            __syncthreads();
            for (int stride = HD / 2; stride > 0; stride >>= 1) {
                if (d < stride)
                    s_reduce[d] += s_reduce[d + stride];
                __syncthreads();
            }
            float q_inv = rsqrtf(fmaxf(s_reduce[0], 1e-12f));

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

        {
            const float out_val = y_partial * scale;
            if constexpr (std::is_same_v<YOut, float>) {
                y_out[t * inner + h * HD + d] = out_val;
            } else {
                y_out[t * inner + h * HD + d] = __float2half(out_val);
            }
        }

        // Sync before next token — the next iteration overwrites s_k/s_q in
        // shared memory. Without this barrier, fast threads can overwrite
        // s_k/s_q while slow threads are still reading them in the loops above.
        if (t + 1 < n_tokens)
            __syncthreads();
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
// Phase 1b.1 — Standalone chunkwise SSD scan prototype.
//
// Structural prototype for the Mamba2 SSD (Structured State-space Duality)
// algorithm adapted for the GDN delta rule. Same numerical math as
// `gdn_scan_fused_kernel`, but reorganised into per-chunk passes that cache
// all CHUNK tokens' normalised K and Q in shared memory upfront, then sweep
// the within-chunk delta-rule update.
//
// Phase 2 will replace the sequential within-chunk loop with the WY-rep
// parallel matmul update (Yang et al. 2024, "Parallel Linear Attention With
// The Delta Rule"). The chunk-cached K, Q layout established here is the
// prerequisite — both Q · K^T (chunk-internal masked-attention) and the
// cumulative decay propagation need all CHUNK tokens' K, Q resident at once.
//
// Per-block shared memory:
//   s_k[CHUNK * SS]  — normalised K, all tokens in chunk
//   s_q[CHUNK * SS]  — normalised Q, all tokens in chunk
//   s_reduce[HD]     — block-reduction scratch (reused per L2 norm)
//
// At HD=SS=128, CHUNK=64 this is 2 * 64 * 128 * 4 + 128 * 4 = 65 KiB,
// requiring the dynamic shared-memory opt-in (cudaFuncAttributeMaxDynamicShared
// MemorySize). Host launcher sets it once.
//
// Grid:  (n_heads)              — one block per head
// Block: (HD)                   — typically 128 threads
// ---------------------------------------------------------------------------
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
        const float* H_col = h_state + static_cast<size_t>(h) * SS * HD + d;
#pragma unroll
        for (int s = 0; s < SS; s++)
            H_reg[s] = H_col[s * HD];
    }

    extern __shared__ float smem[];
    float* s_k = smem;                               // [CHUNK * SS]
    float* s_q = smem + CHUNK * SS;                  // [CHUNK * SS]
    float* s_reduce = smem + 2 * CHUNK * SS;         // [HD]

    int t_chunk_start = 0;
    while (t_chunk_start < n_tokens) {
        const int L = (t_chunk_start + CHUNK <= n_tokens) ? CHUNK : (n_tokens - t_chunk_start);

        // -------------------------------------------------------------------
        // Phase 1: Load this chunk's K, Q (raw) into shared memory.
        // Each thread d stores one element per token, looping over tokens.
        // -------------------------------------------------------------------
        if (d < SS) {
            for (int t_local = 0; t_local < L; t_local++) {
                const int t_global = t_chunk_start + t_local;
                const float* row = conv_f32 + static_cast<size_t>(t_global) * conv_channels;
                s_q[t_local * SS + d] = row[g * SS + d];
                s_k[t_local * SS + d] = row[BC_size + g * SS + d];
            }
        }
        __syncthreads();

        // -------------------------------------------------------------------
        // Phase 2: L2-normalise K, Q for each token in the chunk.
        // Per-token reduction (sequential across tokens, parallel across SS).
        // Uses the SAME formula as gdn_scan_fused_kernel (rsqrt of max(sum_sq,
        // 1e-12)) for bit-equivalent numerics.
        // -------------------------------------------------------------------
        for (int t_local = 0; t_local < L; t_local++) {
            float* k_row = s_k + t_local * SS;
            float* q_row = s_q + t_local * SS;

            float k_sq = 0.0f, q_sq = 0.0f;
            for (int i = d; i < SS; i += HD) {
                k_sq += k_row[i] * k_row[i];
                q_sq += q_row[i] * q_row[i];
            }
            s_reduce[d] = k_sq;
            __syncthreads();
            for (int stride = HD / 2; stride > 0; stride >>= 1) {
                if (d < stride)
                    s_reduce[d] += s_reduce[d + stride];
                __syncthreads();
            }
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
                k_row[d] *= k_inv;
                q_row[d] *= q_inv;
            }
            __syncthreads();
        }

        // -------------------------------------------------------------------
        // Phase 3: Sequential per-token delta-rule update within chunk.
        // Reads K̃, Q̃ from shared memory; same math as gdn_scan_fused_kernel.
        // Phase 2 of the design doc replaces this loop with the WY-rep
        // parallel matmul update.
        // -------------------------------------------------------------------
        for (int t_local = 0; t_local < L; t_local++) {
            const int t_global = t_chunk_start + t_local;
            const float* row = conv_f32 + static_cast<size_t>(t_global) * conv_channels;
            const float* V_base = row + 2 * BC_size;
            const float v_d = V_base[h * HD + d];

            const float* k_row = s_k + t_local * SS;
            const float* q_row = s_q + t_local * SS;

            float alpha_h = __half2float(alpha_all[t_global * n_heads + h]);
            float dt_val = alpha_h + dtb_h;
            dt_val = (dt_val > 20.0f) ? dt_val : logf(1.0f + expf(dt_val));
            const float g_t = expf(fmaxf(A_h * dt_val, -20.0f));

            float beta_h = __half2float(beta_all[t_global * n_heads + h]);
            beta_h = 1.0f / (1.0f + expf(-fmaxf(fminf(beta_h, 20.0f), -20.0f)));

            float kv_d = 0.0f;
#pragma unroll
            for (int s = 0; s < SS; s++)
                kv_d += H_reg[s] * k_row[s];

            const float delta_d = (v_d - g_t * kv_d) * beta_h;

            float y_partial = 0.0f;
#pragma unroll
            for (int s = 0; s < SS; s++) {
                const float h_new = g_t * H_reg[s] + k_row[s] * delta_d;
                H_reg[s] = h_new;
                y_partial += h_new * q_row[s];
            }

            const float out_val = y_partial * scale;
            if constexpr (std::is_same_v<YOut, float>) {
                y_out[t_global * inner + h * HD + d] = out_val;
            } else {
                y_out[t_global * inner + h * HD + d] = __float2half(out_val);
            }
        }

        t_chunk_start += L;
        __syncthreads();  // Before re-using s_k / s_q for the next chunk.
    }

    // Write final state back to global memory.
    {
        float* H_col = h_state + static_cast<size_t>(h) * SS * HD + d;
#pragma unroll
        for (int s = 0; s < SS; s++)
            H_col[s * HD] = H_reg[s];
    }
}

// ---------------------------------------------------------------------------
// Phase 2a — WY-representation parallel delta-rule scan prototype.
//
// Numerically equivalent to gdn_scan_fused_kernel but factors the
// chunk-internal sequential dependency into a forward triangular solve +
// matrix-matrix products. Reference: Yang et al. 2024, "Parallel Linear
// Attention With The Delta Rule"; imp-specific derivation in
// docs/plans/gdn_chunkwise_scan_design_2026_05_23.md §"Phase 2a".
//
// Algorithm per chunk of L tokens:
//   1. Cache K̃, Q̃ in shared memory (post L2 norm)
//   2. Compute Gram matrices KK, QK, KH, QH (matmuls vs the in-register H_0)
//   3. Build the L×L triangular coefficient matrix T and bias vectors c_t
//   4. Forward solve u_t = c_t - Σ_{j<t} T[t,j] u_j (sequential over t,
//      parallel over HD output dim — thread d owns column d of U)
//   5. Compute y_t = scale · (D[0..t+1] QH[t,:] + Σ_{j≤t} D[j+1..t+1] · QK[t,j] · u_j)
//   6. Update H_L = D[0..L] H_0 + Σ_t D[t+1..L] k̃_t u_t^T
//
// Cumulative decay D[a..b] = Π_{i=a..b-1} g_i carried in log-space to dodge
// underflow over L=32 tokens with possibly tiny g (sequential kernel caps
// g_t at e^-20).
//
// CHUNK=32 (not 64) to fit the L^2 + L×HD scratch buffers within the 100 KiB
// sm_120 per-block opt-in cap. At HD=SS=128, CHUNK=32 → ~92 KiB dynamic smem.
//
// Phase 2b will swap the explicit per-thread shared-memory matmul loops for
// CUTLASS / cute MMA tile dispatches. The numerical structure here mirrors
// what the Tensor Core path needs, so 2b is a localized replacement.
// ---------------------------------------------------------------------------
template <int HD, int SS, int CHUNK>
__global__ void __launch_bounds__(HD, 1) gdn_scan_chunkwise_wy_kernel(
    const float* __restrict__ conv_f32, const half* __restrict__ alpha_all,
    const half* __restrict__ beta_all, const float* __restrict__ A_log,
    const float* __restrict__ dt_bias, float* __restrict__ h_state, half* __restrict__ y_out,
    int n_tokens, int n_heads, int n_groups, int conv_channels, int grouped_layout) {
    const int h = blockIdx.x;
    if (h >= n_heads)
        return;
    const int d = threadIdx.x;

    const int g_idx = grouped_layout ? (h / (n_heads / n_groups)) : (h % n_groups);
    const int inner = n_heads * HD;
    const int BC_size = n_groups * SS;
    const float scale = rsqrtf(static_cast<float>(HD));
    const float A_h = A_log[h];
    const float dtb_h = dt_bias[h];

    // State in registers (one column per thread).
    float H_reg[SS];
    {
        const float* H_col = h_state + static_cast<size_t>(h) * SS * HD + d;
#pragma unroll
        for (int s = 0; s < SS; s++)
            H_reg[s] = H_col[s * HD];
    }

    // Shared memory layout (sized for CHUNK, SS, HD; opt-in dynamic smem).
    extern __shared__ float smem[];
    float* s_k = smem;                          // [CHUNK * SS]    normalized K
    float* s_q = s_k + CHUNK * SS;              // [CHUNK * SS]    normalized Q
    float* s_u = s_q + CHUNK * SS;              // [CHUNK * HD]    triangular-solve output
    float* s_kh = s_u + CHUNK * HD;             // [CHUNK * HD]    K̃ H_0
    float* s_qh = s_kh + CHUNK * HD;            // [CHUNK * HD]    Q̃ H_0
    float* s_kk = s_qh + CHUNK * HD;            // [CHUNK * CHUNK] K̃ K̃^T (lower-tri only used)
    float* s_qk = s_kk + CHUNK * CHUNK;         // [CHUNK * CHUNK] Q̃ K̃^T (lower-tri only used)
    float* s_g = s_qk + CHUNK * CHUNK;          // [CHUNK]         per-token decay
    float* s_beta = s_g + CHUNK;                // [CHUNK]         per-token learning rate
    float* s_logD = s_beta + CHUNK;             // [CHUNK + 1]     cumulative log decay
    float* s_reduce = s_logD + CHUNK + 1;       // [HD]            block-reduction scratch

    int t_chunk_start = 0;
    while (t_chunk_start < n_tokens) {
        const int L = (t_chunk_start + CHUNK <= n_tokens) ? CHUNK : (n_tokens - t_chunk_start);

        // ---------------- STEP 1: load + normalize K, Q ----------------
        if (d < SS) {
            for (int t_loc = 0; t_loc < L; t_loc++) {
                const int t = t_chunk_start + t_loc;
                const float* row = conv_f32 + static_cast<size_t>(t) * conv_channels;
                s_q[t_loc * SS + d] = row[g_idx * SS + d];
                s_k[t_loc * SS + d] = row[BC_size + g_idx * SS + d];
            }
        }
        __syncthreads();

        for (int t_loc = 0; t_loc < L; t_loc++) {
            float k_sq = 0.0f, q_sq = 0.0f;
            for (int i = d; i < SS; i += HD) {
                k_sq += s_k[t_loc * SS + i] * s_k[t_loc * SS + i];
                q_sq += s_q[t_loc * SS + i] * s_q[t_loc * SS + i];
            }
            s_reduce[d] = k_sq;
            __syncthreads();
            for (int stride = HD / 2; stride > 0; stride >>= 1) {
                if (d < stride)
                    s_reduce[d] += s_reduce[d + stride];
                __syncthreads();
            }
            const float k_inv = rsqrtf(fmaxf(s_reduce[0], 1e-12f));

            s_reduce[d] = q_sq;
            __syncthreads();
            for (int stride = HD / 2; stride > 0; stride >>= 1) {
                if (d < stride)
                    s_reduce[d] += s_reduce[d + stride];
                __syncthreads();
            }
            const float q_inv = rsqrtf(fmaxf(s_reduce[0], 1e-12f));

            if (d < SS) {
                s_k[t_loc * SS + d] *= k_inv;
                s_q[t_loc * SS + d] *= q_inv;
            }
            __syncthreads();
        }

        // ---------------- STEP 2a: per-token g_t, β_t, log_D[0..t+1] ----------------
        // Thread 0 owns the sequential dependency (small, L=32 iterations).
        if (d == 0) {
            float log_D = 0.0f;
            s_logD[0] = 0.0f;
            for (int t_loc = 0; t_loc < L; t_loc++) {
                const int t = t_chunk_start + t_loc;
                float alpha_h = __half2float(alpha_all[t * n_heads + h]);
                float dt_val = alpha_h + dtb_h;
                dt_val = (dt_val > 20.0f) ? dt_val : logf(1.0f + expf(dt_val));
                float lg_t = fmaxf(A_h * dt_val, -20.0f);  // log(g_t)
                s_g[t_loc] = expf(lg_t);
                log_D += lg_t;
                s_logD[t_loc + 1] = log_D;
                float beta_h = __half2float(beta_all[t * n_heads + h]);
                s_beta[t_loc] = 1.0f / (1.0f + expf(-fmaxf(fminf(beta_h, 20.0f), -20.0f)));
            }
        }
        __syncthreads();

        // ---------------- STEP 2b: KH = K̃ H_0, QH = Q̃ H_0 ----------------
        // Thread d computes column d of KH and QH (H_0 column d is in this thread's H_reg).
        for (int t_loc = 0; t_loc < L; t_loc++) {
            float kh = 0.0f, qh = 0.0f;
#pragma unroll
            for (int s = 0; s < SS; s++) {
                kh += s_k[t_loc * SS + s] * H_reg[s];
                qh += s_q[t_loc * SS + s] * H_reg[s];
            }
            s_kh[t_loc * HD + d] = kh;
            s_qh[t_loc * HD + d] = qh;
        }
        // No sync needed yet — STEP 3 reads from s_kh/s_qh after a sync at end of STEP 2c.

        // ---------------- STEP 2c: KK and QK Gram matrices ----------------
        // Lower-triangular only (j ≤ i). HD=128 threads cooperate on L*L pairs.
        for (int idx = d; idx < L * L; idx += HD) {
            const int i = idx / L;
            const int j = idx % L;
            if (j > i) {
                s_kk[idx] = 0.0f;
                s_qk[idx] = 0.0f;
                continue;
            }
            float kk = 0.0f, qk = 0.0f;
#pragma unroll
            for (int s = 0; s < SS; s++) {
                const float k_is = s_k[i * SS + s];
                const float k_js = s_k[j * SS + s];
                const float q_is = s_q[i * SS + s];
                kk += k_is * k_js;
                qk += q_is * k_js;
            }
            s_kk[idx] = kk;
            s_qk[idx] = qk;
        }
        __syncthreads();  // STEP 3 reads s_kh, s_kk; STEP 4 reads s_u; STEP 5 reads s_qh, s_qk.

        // ---------------- STEP 3+4: triangular solve for u_t ----------------
        // Sequential over t (intra-chunk), parallel over HD output dim (one per thread).
        // u_t[d] = c_t[d] - Σ_{j<t} T[t,j] u_j[d]
        // with c_t[d] = β_t v_t[d] - β_t g_t D[0..t] KH[t,d]
        // and  T[t,j] = β_t g_t D[j+1..t] KK[t,j]
        for (int t_loc = 0; t_loc < L; t_loc++) {
            const int t = t_chunk_start + t_loc;
            const float* row = conv_f32 + static_cast<size_t>(t) * conv_channels;
            const float v_d = row[2 * BC_size + h * HD + d];

            const float g_t = s_g[t_loc];
            const float beta_t = s_beta[t_loc];
            const float logD_0t = s_logD[t_loc];
            const float D_0t = expf(logD_0t);

            float u_t = beta_t * v_d - beta_t * g_t * D_0t * s_kh[t_loc * HD + d];
            for (int j = 0; j < t_loc; j++) {
                const float logD_j1t = s_logD[t_loc] - s_logD[j + 1];
                const float coef = beta_t * g_t * expf(logD_j1t) * s_kk[t_loc * L + j];
                u_t -= coef * s_u[j * HD + d];
            }
            s_u[t_loc * HD + d] = u_t;
            __syncthreads();
        }

        // ---------------- STEP 5: Y[t] for all chunk tokens ----------------
        // y_t[d] = scale · (D[0..t+1] QH[t,d] + Σ_{j≤t} D[j+1..t+1] QK[t,j] u_j[d])
        for (int t_loc = 0; t_loc < L; t_loc++) {
            const int t = t_chunk_start + t_loc;
            const float logD_0t1 = s_logD[t_loc + 1];

            float y = expf(logD_0t1) * s_qh[t_loc * HD + d];
            for (int j = 0; j <= t_loc; j++) {
                const float logD_j1t1 = s_logD[t_loc + 1] - s_logD[j + 1];
                y += expf(logD_j1t1) * s_qk[t_loc * L + j] * s_u[j * HD + d];
            }
            y_out[static_cast<size_t>(t) * inner + h * HD + d] = __float2half(y * scale);
        }

        // ---------------- STEP 6: H_L = D[0..L] H_0 + Σ_t D[t+1..L] k̃_t u_t^T ----------------
        // Thread d updates column d of H. For each state row s: H_reg[s] *= D[0..L] then add Σ_t D[t+1..L] k̃_t[s] u_t[d].
        {
            const float D_0L = expf(s_logD[L]);
#pragma unroll
            for (int s = 0; s < SS; s++) {
                float add = 0.0f;
                for (int t_loc = 0; t_loc < L; t_loc++) {
                    const float logD_t1L = s_logD[L] - s_logD[t_loc + 1];
                    add += expf(logD_t1L) * s_k[t_loc * SS + s] * s_u[t_loc * HD + d];
                }
                H_reg[s] = D_0L * H_reg[s] + add;
            }
        }

        t_chunk_start += L;
        __syncthreads();  // Before reusing s_k / s_q / etc. for the next chunk.
    }

    // Write final state back to global memory.
    {
        float* H_col = h_state + static_cast<size_t>(h) * SS * HD + d;
#pragma unroll
        for (int s = 0; s < SS; s++)
            H_col[s * HD] = H_reg[s];
    }
}

// ---------------------------------------------------------------------------
// Phase 2b — Tensor Core MMA prototype on top of the Phase 2a WY-rep math.
//
// Replaces the four chunk-internal scalar shared-memory matmuls in Phase 2a
// (KK, QK, KH, QH) with WMMA 16×16×16 FP16→FP32 Tensor Core dispatches.
// The H_L update (Step 6) stays scalar but with hoisted exp-of-cumulative-
// decay caching — independent optimisation that drops ~SS × L exp calls per
// chunk per head down to L per chunk per head.
//
// CHUNK=16 (not 32 like Phase 2a) — the WMMA shape is exactly the 16×16
// fragment and the smem budget at CHUNK=32 with FP16 K̃/Q̃ + FP16 H_0 +
// FP32 outputs blows past the 99 KiB sm_120 opt-in cap. CHUNK=16 lands at
// ~69 KiB.
//
// Smem layout (HD=SS=128, CHUNK=16):
//   s_k_fp16[L*SS]    = 4 KiB     normalised K (FP16 for WMMA matmul)
//   s_q_fp16[L*SS]    = 4 KiB     normalised Q (FP16 for WMMA matmul)
//   s_h0_fp16[SS*HD]  = 32 KiB    H_0 materialised from registers as FP16
//   s_kh_fp32[L*HD]   = 8 KiB     KH = K̃ H_0 (WMMA output, FP32 accum)
//   s_qh_fp32[L*HD]   = 8 KiB     QH = Q̃ H_0
//   s_kk_fp32[L*L]    = 1 KiB     KK = K̃ K̃^T
//   s_qk_fp32[L*L]    = 1 KiB     QK = Q̃ K̃^T
//   s_u_fp32[L*HD]    = 8 KiB     U (output of triangular solve)
//   s_D[L+1]          = 68 B      cumulative decay (exp space, not log)
//   s_g, s_beta       = 128 B
//   s_reduce[HD]      = 512 B
// Total ~67 KiB → fits with the 96 KiB opt-in (same as Phase 1b.1/2a).
//
// WMMA operand setup:
//   - K̃ K̃^T (KK):   A=K̃ row_major [L,SS], B=K̃ col_major [SS,L]
//   - Q̃ K̃^T (QK):   A=Q̃ row_major, B=K̃ col_major
//   - K̃ H_0 (KH):    A=K̃ row_major, B=H_0 row_major [SS,HD]
//   - Q̃ H_0 (QH):    A=Q̃ row_major, B=H_0 row_major
//
// Phase 2b is the FIRST imp GDN-side TC-MMA path. The H_L update remains
// scalar; integrating WMMA on H_L would require an extra ~16 KiB temp tile
// buffer or careful warp-fragment-back-to-register choreography that
// doesn't fit in this initial prototype.
// ---------------------------------------------------------------------------
template <int HD, int SS, int CHUNK>
__global__ void __launch_bounds__(HD, 1) gdn_scan_chunkwise_wy_tc_kernel(
    const float* __restrict__ conv_f32, const half* __restrict__ alpha_all,
    const half* __restrict__ beta_all, const float* __restrict__ A_log,
    const float* __restrict__ dt_bias, float* __restrict__ h_state, half* __restrict__ y_out,
    int n_tokens, int n_heads, int n_groups, int conv_channels, int grouped_layout) {
    static_assert(CHUNK == 16, "WMMA prototype tuned for CHUNK=16 only");
    static_assert(HD == 128 && SS == 128, "WMMA prototype tuned for HD=SS=128 only");

    const int h = blockIdx.x;
    if (h >= n_heads)
        return;
    const int d = threadIdx.x;
    const int warp_id = d / 32;
    const int n_warps = HD / 32;  // 4 warps per block

    const int g_idx = grouped_layout ? (h / (n_heads / n_groups)) : (h % n_groups);
    const int inner = n_heads * HD;
    const int BC_size = n_groups * SS;
    const float scale = rsqrtf(static_cast<float>(HD));
    const float A_h = A_log[h];
    const float dtb_h = dt_bias[h];

    // State in registers (one column per thread).
    float H_reg[SS];
    {
        const float* H_col = h_state + static_cast<size_t>(h) * SS * HD + d;
#pragma unroll
        for (int s = 0; s < SS; s++)
            H_reg[s] = H_col[s * HD];
    }

    extern __shared__ float smem[];
    half* s_k_fp16 = reinterpret_cast<half*>(smem);                      // [L * SS]
    half* s_q_fp16 = s_k_fp16 + CHUNK * SS;                              // [L * SS]
    half* s_h0_fp16 = s_q_fp16 + CHUNK * SS;                             // [SS * HD]
    float* s_kh_fp32 = reinterpret_cast<float*>(s_h0_fp16 + SS * HD);    // [L * HD]
    float* s_qh_fp32 = s_kh_fp32 + CHUNK * HD;                           // [L * HD]
    float* s_kk_fp32 = s_qh_fp32 + CHUNK * HD;                           // [L * L]
    float* s_qk_fp32 = s_kk_fp32 + CHUNK * CHUNK;                        // [L * L]
    float* s_u_fp32 = s_qk_fp32 + CHUNK * CHUNK;                         // [L * HD]
    float* s_D = s_u_fp32 + CHUNK * HD;                                  // [L + 1]   exp-cumulative decay
    float* s_g = s_D + (CHUNK + 1);                                      // [L]
    float* s_beta = s_g + CHUNK;                                         // [L]
    float* s_reduce = s_beta + CHUNK;                                    // [HD]

    int t_chunk_start = 0;
    while (t_chunk_start < n_tokens) {
        const int L = (t_chunk_start + CHUNK <= n_tokens) ? CHUNK : (n_tokens - t_chunk_start);

        // ---------------- STEP 1: load K, Q (FP32→FP16) and L2-normalise ----------------
        // First load raw values into FP16 (FP16 has 11 mantissa bits — sufficient for the
        // post-norm values which are ≤ 1 in magnitude after rsqrt scaling).
        if (d < SS) {
            for (int t_loc = 0; t_loc < L; t_loc++) {
                const int t = t_chunk_start + t_loc;
                const float* row = conv_f32 + static_cast<size_t>(t) * conv_channels;
                s_q_fp16[t_loc * SS + d] = __float2half(row[g_idx * SS + d]);
                s_k_fp16[t_loc * SS + d] = __float2half(row[BC_size + g_idx * SS + d]);
            }
        }
        __syncthreads();

        // Per-token L2-norm reduction (parallel across SS via block reduction).
        for (int t_loc = 0; t_loc < L; t_loc++) {
            float k_sq = 0.0f, q_sq = 0.0f;
            for (int i = d; i < SS; i += HD) {
                float k = __half2float(s_k_fp16[t_loc * SS + i]);
                float q = __half2float(s_q_fp16[t_loc * SS + i]);
                k_sq += k * k;
                q_sq += q * q;
            }
            s_reduce[d] = k_sq;
            __syncthreads();
            for (int stride = HD / 2; stride > 0; stride >>= 1) {
                if (d < stride)
                    s_reduce[d] += s_reduce[d + stride];
                __syncthreads();
            }
            const float k_inv = rsqrtf(fmaxf(s_reduce[0], 1e-12f));

            s_reduce[d] = q_sq;
            __syncthreads();
            for (int stride = HD / 2; stride > 0; stride >>= 1) {
                if (d < stride)
                    s_reduce[d] += s_reduce[d + stride];
                __syncthreads();
            }
            const float q_inv = rsqrtf(fmaxf(s_reduce[0], 1e-12f));

            if (d < SS) {
                s_k_fp16[t_loc * SS + d] = __float2half(__half2float(s_k_fp16[t_loc * SS + d]) * k_inv);
                s_q_fp16[t_loc * SS + d] = __float2half(__half2float(s_q_fp16[t_loc * SS + d]) * q_inv);
            }
            __syncthreads();
        }

        // ---------------- STEP 2: g_t, β_t, cumulative decay D[0..t+1] ----------------
        if (d == 0) {
            float D_cum = 1.0f;
            s_D[0] = 1.0f;
            for (int t_loc = 0; t_loc < L; t_loc++) {
                const int t = t_chunk_start + t_loc;
                float alpha_h = __half2float(alpha_all[t * n_heads + h]);
                float dt_val = alpha_h + dtb_h;
                dt_val = (dt_val > 20.0f) ? dt_val : logf(1.0f + expf(dt_val));
                float lg_t = fmaxf(A_h * dt_val, -20.0f);
                s_g[t_loc] = expf(lg_t);
                D_cum *= s_g[t_loc];
                s_D[t_loc + 1] = D_cum;
                float beta_h = __half2float(beta_all[t * n_heads + h]);
                s_beta[t_loc] = 1.0f / (1.0f + expf(-fmaxf(fminf(beta_h, 20.0f), -20.0f)));
            }
        }

        // ---------------- STEP 3: materialise H_0 in shared memory as FP16 ----------------
        // Each thread d writes its column to s_h0_fp16[s * HD + d].
        // The static H_reg[SS] array doesn't allow trivial cp.async; do per-element stores.
#pragma unroll
        for (int s = 0; s < SS; s++) {
            s_h0_fp16[s * HD + d] = __float2half(H_reg[s]);
        }
        __syncthreads();

        // ---------------- STEP 4: WMMA matmuls KK, QK, KH, QH ----------------
        // All four use the same SS-dim inner reduction (L×SS × SS×{L,HD}).
        // Output tiles distributed across warps. Standard m=n=k=16 fragments.

        // KK and QK (L×L outputs, single 16×16 tile each). Warp 0 does KK, warp 1 does QK.
        if (warp_id == 0) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
            wmma::fill_fragment(c_frag, 0.0f);
            for (int k = 0; k < SS / 16; k++) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
                wmma::load_matrix_sync(a_frag, s_k_fp16 + k * 16, SS);
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
                wmma::load_matrix_sync(b_frag, s_k_fp16 + k * 16, SS);
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
            wmma::store_matrix_sync(s_kk_fp32, c_frag, CHUNK, wmma::mem_row_major);
        } else if (warp_id == 1) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
            wmma::fill_fragment(c_frag, 0.0f);
            for (int k = 0; k < SS / 16; k++) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
                wmma::load_matrix_sync(a_frag, s_q_fp16 + k * 16, SS);
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
                wmma::load_matrix_sync(b_frag, s_k_fp16 + k * 16, SS);
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
            wmma::store_matrix_sync(s_qk_fp32, c_frag, CHUNK, wmma::mem_row_major);
        }

        // KH and QH (L×HD outputs, 1 M-tile × 8 N-tiles = 8 tiles each, 16 total).
        // All 4 warps process tiles in stride-4 fashion.
        const int kh_n_tiles = HD / 16;  // 8
        for (int tile_idx = warp_id; tile_idx < kh_n_tiles; tile_idx += n_warps) {
            const int n_offset = tile_idx * 16;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
            wmma::fill_fragment(c_frag, 0.0f);
            for (int k = 0; k < SS / 16; k++) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
                wmma::load_matrix_sync(a_frag, s_k_fp16 + k * 16, SS);
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
                wmma::load_matrix_sync(b_frag, s_h0_fp16 + k * 16 * HD + n_offset, HD);
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
            wmma::store_matrix_sync(s_kh_fp32 + n_offset, c_frag, HD, wmma::mem_row_major);
        }
        for (int tile_idx = warp_id; tile_idx < kh_n_tiles; tile_idx += n_warps) {
            const int n_offset = tile_idx * 16;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
            wmma::fill_fragment(c_frag, 0.0f);
            for (int k = 0; k < SS / 16; k++) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
                wmma::load_matrix_sync(a_frag, s_q_fp16 + k * 16, SS);
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
                wmma::load_matrix_sync(b_frag, s_h0_fp16 + k * 16 * HD + n_offset, HD);
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
            wmma::store_matrix_sync(s_qh_fp32 + n_offset, c_frag, HD, wmma::mem_row_major);
        }
        __syncthreads();

        // ---------------- STEP 5: triangular solve for u_t ----------------
        // Sequential over t (intra-chunk), parallel over HD output dim.
        // u_t[d] = c_t[d] - Σ_{j<t} T[t,j] u_j[d]
        // c_t[d] = β_t v_t[d] - β_t g_t D[0..t] KH[t,d]
        // T[t,j] = β_t g_t (D[0..t]/D[0..j+1]) KK[t,j]
        for (int t_loc = 0; t_loc < L; t_loc++) {
            const int t = t_chunk_start + t_loc;
            const float* row = conv_f32 + static_cast<size_t>(t) * conv_channels;
            const float v_d = row[2 * BC_size + h * HD + d];

            const float g_t = s_g[t_loc];
            const float beta_t = s_beta[t_loc];
            const float D_0t = s_D[t_loc];

            float u_t = beta_t * v_d - beta_t * g_t * D_0t * s_kh_fp32[t_loc * HD + d];
            const float bg_Dt = beta_t * g_t * D_0t;
            for (int j = 0; j < t_loc; j++) {
                const float D_inv_j1 = 1.0f / s_D[j + 1];
                const float coef = bg_Dt * D_inv_j1 * s_kk_fp32[t_loc * CHUNK + j];
                u_t -= coef * s_u_fp32[j * HD + d];
            }
            s_u_fp32[t_loc * HD + d] = u_t;
            __syncthreads();
        }

        // ---------------- STEP 6: Y[t] for all chunk tokens ----------------
        // y_t[d] = scale · (D[0..t+1] QH[t,d] + Σ_{j≤t} (D[0..t+1]/D[0..j+1]) QK[t,j] u_j[d])
        for (int t_loc = 0; t_loc < L; t_loc++) {
            const int t = t_chunk_start + t_loc;
            const float D_0t1 = s_D[t_loc + 1];

            float y = D_0t1 * s_qh_fp32[t_loc * HD + d];
            for (int j = 0; j <= t_loc; j++) {
                const float coef = (D_0t1 / s_D[j + 1]) * s_qk_fp32[t_loc * CHUNK + j];
                y += coef * s_u_fp32[j * HD + d];
            }
            y_out[static_cast<size_t>(t) * inner + h * HD + d] = __float2half(y * scale);
        }

        // ---------------- STEP 7: H_L = D[0..L] H_0 + Σ_t (D[0..L]/D[0..t+1]) k̃_t u_t^T ----------------
        // Precompute D[t+1..L] = D[0..L] / D[0..t+1] into s_g (reusable scratch, no longer needed).
        // Saves SS × L − L expf calls per chunk per head.
        const float D_0L = s_D[L];
        if (d < L) {
            s_g[d] = D_0L / s_D[d + 1];  // D[d+1..L]
        }
        __syncthreads();

#pragma unroll
        for (int s = 0; s < SS; s++) {
            float add = 0.0f;
            for (int t_loc = 0; t_loc < L; t_loc++) {
                add += s_g[t_loc] * __half2float(s_k_fp16[t_loc * SS + s]) * s_u_fp32[t_loc * HD + d];
            }
            H_reg[s] = D_0L * H_reg[s] + add;
        }

        t_chunk_start += L;
        __syncthreads();  // Before reusing smem for the next chunk.
    }

    // Write final state back to global memory.
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
// V-head reorder: tiled → grouped (undo GGUF converter reorder for ssm_out)
// ---------------------------------------------------------------------------
__global__ void vhead_tiled_to_grouped_kernel(const half* __restrict__ src, half* __restrict__ dst,
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

void vhead_tiled_to_grouped(const half* src, half* dst, int n_tokens, int n_heads, int head_dim, int n_groups,
                            cudaStream_t stream) {
    if (n_heads == n_groups)
        return;
    int total = n_tokens * n_heads * head_dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    vhead_tiled_to_grouped_kernel<<<blocks, threads, 0, stream>>>(src, dst, n_tokens, n_heads, head_dim,
                                                                  n_groups);
}

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
}

// ---------------------------------------------------------------------------
// Host launchers
// ---------------------------------------------------------------------------

// Fused scan: processes all tokens in one kernel launch.
// conv_f32: [n_tokens, conv_channels] FP32 — full conv output (Q|K|V interleaved per token)
// grouped_layout: 0 = GGUF tiled (g = h % n_groups), 1 = HF SafeTensors grouped
//                 (g = h / n_v_per_k). See kernel comment for details.
void gdn_scan_fused_f32(const float* conv_f32, int conv_channels, const half* alpha, const half* beta,
                        const float* A_log, const float* dt_bias, float* h_state, half* y, int n_tokens,
                        int n_heads, int head_dim_ssm, int state_size, int n_groups, cudaStream_t stream,
                        int grouped_layout) {
    // Shared memory: K_norm[SS] + Q_norm[SS] + reduce[HD]
    size_t smem = (2 * state_size + head_dim_ssm) * sizeof(float);

    // Template dispatch for common sizes
    if (head_dim_ssm == 128 && state_size == 128) {
        gdn_scan_fused_kernel<128, 128, half>
            <<<n_heads, 128, smem, stream>>>(conv_f32, alpha, beta, A_log, dt_bias, h_state, y, n_tokens,
                                             n_heads, n_groups, conv_channels, grouped_layout);
    } else if (head_dim_ssm == 64 && state_size == 64) {
        gdn_scan_fused_kernel<64, 64, half>
            <<<n_heads, 64, smem, stream>>>(conv_f32, alpha, beta, A_log, dt_bias, h_state, y, n_tokens,
                                            n_heads, n_groups, conv_channels, grouped_layout);
    } else {
        // Fallback: per-token loop (for unsupported HD/SS sizes)
        int inner = n_heads * head_dim_ssm;
        int BC_size = n_groups * state_size;
        size_t smem_old = 2 * state_size * sizeof(float) + 2 * sizeof(float);
        for (int t = 0; t < n_tokens; t++) {
            const float* row = conv_f32 + static_cast<size_t>(t) * conv_channels;
            gdn_scan_decode_kernel<<<n_heads, head_dim_ssm, smem_old, stream>>>(
                row + 2 * BC_size, row + BC_size, row, alpha + t * n_heads, beta + t * n_heads, A_log,
                dt_bias, h_state, y + t * inner, nullptr, n_heads, head_dim_ssm, state_size, n_groups,
                grouped_layout);
        }
    }
}

// Phase 1b.1 — Standalone chunkwise SSD scan host launcher.
//
// Dispatches to `gdn_scan_chunkwise_kernel<HD, SS, CHUNK>` for the supported
// (HD, SS, CHUNK) combinations and falls back to a chunk-iterating wrapper
// around `gdn_scan_fused_f32` otherwise. The chunkwise kernel produces output
// bit-near-equivalent to `gdn_scan_fused_kernel` (FP16 1e-3, FP32 state 1e-5
// tolerances per Phase 1a), validated by ChunkBoundaryHandoff +
// ChunkwiseProtoMatchesFused tests.
//
// Phase 2 will replace the within-chunk sequential delta-rule loop in
// `gdn_scan_chunkwise_kernel` with the WY-rep parallel matmul update (Yang
// et al. 2024, "Parallel Linear Attention With The Delta Rule"). Until then
// the prototype is structural-only and gives no perf win over the sequential
// fused kernel; it establishes the chunked shared-memory layout that the
// SSD matmul will need.
//
// Phase 0 verdict (ncu): docs/plans/gdn_chunkwise_scan_design_2026_05_23.md
//   Memory 5.47 % peak / Compute 5.47 % peak / Achieved Occ 8.33 % → PROCEED.
// Templated dispatcher used by both gdn_scan_chunkwise_f32 (YOut=half) and
// gdn_scan_chunkwise_fp32out (YOut=float). Keeps the kernel-template + smem
// + opt-in logic in one place — the two host launchers only differ in the
// y_out element type and the fallback they use for unsupported shapes.
template <typename YOut, typename FusedFallback>
static void gdn_scan_chunkwise_dispatch(const float* conv_f32, int conv_channels, const half* alpha,
                                        const half* beta, const float* A_log, const float* dt_bias,
                                        float* h_state, YOut* y, int n_tokens, int n_heads, int head_dim_ssm,
                                        int state_size, int n_groups, cudaStream_t stream, int chunk_size,
                                        int grouped_layout, FusedFallback fused) {
    if (chunk_size <= 0)
        chunk_size = 64;
    if (chunk_size > n_tokens)
        chunk_size = n_tokens;

    // Direct chunkwise kernel dispatch for the supported (HD, SS, CHUNK) shapes.
    // Phase 1b.1 covers chunk_size==64 + HD==SS in {128, 64}, the production
    // GDN shapes (Qwen 3.5 / 3.6). Other sizes fall through to the wrapper.
    if (chunk_size == 64 && n_tokens >= 64) {
        if (head_dim_ssm == 128 && state_size == 128) {
            constexpr int HD = 128, SS = 128, CHUNK = 64;
            const size_t smem = (2 * CHUNK * SS + HD) * sizeof(float);
            static bool attr_set = false;
            if (!attr_set) {
                cudaFuncSetAttribute(
                    reinterpret_cast<const void*>(&gdn_scan_chunkwise_kernel<HD, SS, CHUNK, YOut>),
                    cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024);
                attr_set = true;
            }
            gdn_scan_chunkwise_kernel<HD, SS, CHUNK, YOut><<<n_heads, HD, smem, stream>>>(
                conv_f32, alpha, beta, A_log, dt_bias, h_state, y, n_tokens, n_heads, n_groups, conv_channels,
                grouped_layout);
            return;
        }
        if (head_dim_ssm == 64 && state_size == 64) {
            constexpr int HD = 64, SS = 64, CHUNK = 64;
            const size_t smem = (2 * CHUNK * SS + HD) * sizeof(float);
            // 2 * 64 * 64 * 4 + 64 * 4 = 32 KiB + 256 B — within default static cap.
            gdn_scan_chunkwise_kernel<HD, SS, CHUNK, YOut><<<n_heads, HD, smem, stream>>>(
                conv_f32, alpha, beta, A_log, dt_bias, h_state, y, n_tokens, n_heads, n_groups, conv_channels,
                grouped_layout);
            return;
        }
    }

    // Fallback: chunk-iterating wrapper around the sequential fused kernel.
    // Used for non-default chunk sizes, unsupported HD/SS combos, and the
    // tail-chunk path where n_tokens < chunk_size. h_state mutates in-place
    // across the per-chunk calls; same-stream submission keeps ordering.
    const int inner = n_heads * head_dim_ssm;
    int t = 0;
    while (t < n_tokens) {
        const int this_chunk = (t + chunk_size <= n_tokens) ? chunk_size : (n_tokens - t);
        fused(conv_f32 + static_cast<size_t>(t) * conv_channels, alpha + t * n_heads, beta + t * n_heads,
              h_state, y + static_cast<size_t>(t) * inner, this_chunk);
        t += this_chunk;
    }
}

void gdn_scan_chunkwise_f32(const float* conv_f32, int conv_channels, const half* alpha, const half* beta,
                            const float* A_log, const float* dt_bias, float* h_state, half* y,
                            int n_tokens, int n_heads, int head_dim_ssm, int state_size, int n_groups,
                            cudaStream_t stream, int chunk_size, int grouped_layout) {
    gdn_scan_chunkwise_dispatch<half>(
        conv_f32, conv_channels, alpha, beta, A_log, dt_bias, h_state, y, n_tokens, n_heads, head_dim_ssm,
        state_size, n_groups, stream, chunk_size, grouped_layout,
        [&](const float* row_conv, const half* row_alpha, const half* row_beta, float* h_state_, half* y_,
            int n_tok_chunk) {
            gdn_scan_fused_f32(row_conv, conv_channels, row_alpha, row_beta, A_log, dt_bias, h_state_, y_,
                               n_tok_chunk, n_heads, head_dim_ssm, state_size, n_groups, stream,
                               grouped_layout);
        });
}

// FP32-output chunkwise launcher. Mirrors `gdn_scan_chunkwise_f32` for the
// `gdn.fp32_scan` path where the scan output must stay FP32 all the way
// through RMSNorm+Gate+SiLU (Qwen 3.6 L0 sign-flip root cause; see comment
// at executor_ssm_gdn.cu:483-486).
void gdn_scan_chunkwise_fp32out(const float* conv_f32, int conv_channels, const half* alpha, const half* beta,
                                const float* A_log, const float* dt_bias, float* h_state, float* y_fp32,
                                int n_tokens, int n_heads, int head_dim_ssm, int state_size, int n_groups,
                                cudaStream_t stream, int chunk_size, int grouped_layout) {
    gdn_scan_chunkwise_dispatch<float>(
        conv_f32, conv_channels, alpha, beta, A_log, dt_bias, h_state, y_fp32, n_tokens, n_heads, head_dim_ssm,
        state_size, n_groups, stream, chunk_size, grouped_layout,
        [&](const float* row_conv, const half* row_alpha, const half* row_beta, float* h_state_, float* y_,
            int n_tok_chunk) {
            gdn_scan_fused_fp32out(row_conv, conv_channels, row_alpha, row_beta, A_log, dt_bias, h_state_, y_,
                                   n_tok_chunk, n_heads, head_dim_ssm, state_size, n_groups, stream,
                                   grouped_layout);
        });
}

// Phase 2a WY-rep host launcher. Currently HD=SS=128 + CHUNK=32 only; other
// shapes fall back to `gdn_scan_fused_f32`. Output is FP16 only — Phase 2a
// is a correctness reference; the FP32-out and Phase 2b TC-MMA variants come
// later.
void gdn_scan_chunkwise_wy_f32(const float* conv_f32, int conv_channels, const half* alpha, const half* beta,
                               const float* A_log, const float* dt_bias, float* h_state, half* y,
                               int n_tokens, int n_heads, int head_dim_ssm, int state_size, int n_groups,
                               cudaStream_t stream, int grouped_layout) {
    if (head_dim_ssm == 128 && state_size == 128 && n_tokens >= 1) {
        constexpr int HD = 128, SS = 128, CHUNK = 32;
        // Shared-memory budget for the WY kernel:
        //   s_k + s_q       = 2 * CHUNK * SS         = 32 KiB
        //   s_u + s_kh + s_qh = 3 * CHUNK * HD       = 48 KiB
        //   s_kk + s_qk     = 2 * CHUNK * CHUNK      =  8 KiB
        //   s_g + s_beta + s_logD + s_reduce        ≈  1 KiB
        // Total ~89 KiB → needs the dynamic-shared opt-in.
        const size_t smem =
            (2 * CHUNK * SS + 3 * CHUNK * HD + 2 * CHUNK * CHUNK + 2 * CHUNK + (CHUNK + 1) + HD) *
            sizeof(float);
        static bool attr_set = false;
        if (!attr_set) {
            // sm_120 caps `cudaFuncAttributeMaxDynamicSharedMemorySize` at 99 KiB
            // (sharedMemPerBlockOptin = 101376 B). Setting above that returns
            // cudaErrorInvalidValue and the kernel falls back to the 48 KiB
            // default → kernel launch fails with "invalid argument" since the
            // request (~89 KiB) exceeds the default. Use 96 KiB (matches the
            // existing reference kernel's opt-in).
            cudaFuncSetAttribute(
                reinterpret_cast<const void*>(&gdn_scan_chunkwise_wy_kernel<HD, SS, CHUNK>),
                cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024);
            attr_set = true;
        }
        gdn_scan_chunkwise_wy_kernel<HD, SS, CHUNK><<<n_heads, HD, smem, stream>>>(
            conv_f32, alpha, beta, A_log, dt_bias, h_state, y, n_tokens, n_heads, n_groups, conv_channels,
            grouped_layout);
        return;
    }
    // Unsupported shape — fall back to the sequential kernel for safety.
    gdn_scan_fused_f32(conv_f32, conv_channels, alpha, beta, A_log, dt_bias, h_state, y, n_tokens, n_heads,
                       head_dim_ssm, state_size, n_groups, stream, grouped_layout);
}

// Phase 2b host launcher. HD=SS=128 + CHUNK=16 path; other shapes fall back
// to the sequential kernel.
void gdn_scan_chunkwise_wy_tc_f32(const float* conv_f32, int conv_channels, const half* alpha,
                                  const half* beta, const float* A_log, const float* dt_bias,
                                  float* h_state, half* y, int n_tokens, int n_heads, int head_dim_ssm,
                                  int state_size, int n_groups, cudaStream_t stream, int grouped_layout) {
    if (head_dim_ssm == 128 && state_size == 128 && n_tokens >= 1) {
        constexpr int HD = 128, SS = 128, CHUNK = 16;
        // Shared-memory budget for the TC kernel (FP16 storage of K̃/Q̃/H_0,
        // FP32 outputs):
        //   s_k_fp16 + s_q_fp16 = 2 * CHUNK * SS * 2 = 8 KiB
        //   s_h0_fp16           = SS * HD * 2        = 32 KiB
        //   s_kh + s_qh         = 2 * CHUNK * HD * 4 = 16 KiB
        //   s_kk + s_qk         = 2 * CHUNK^2 * 4    = 2 KiB
        //   s_u_fp32            = CHUNK * HD * 4     = 8 KiB
        //   s_D + s_g + s_beta + s_reduce ≈ 1 KiB
        // Total ~67 KiB — fits within the 96 KiB opt-in.
        const size_t smem =
            (2 * CHUNK * SS) * sizeof(half) + (SS * HD) * sizeof(half) +
            (2 * CHUNK * HD + 2 * CHUNK * CHUNK + CHUNK * HD + (CHUNK + 1) + 2 * CHUNK + HD) * sizeof(float);
        static bool attr_set = false;
        if (!attr_set) {
            cudaFuncSetAttribute(
                reinterpret_cast<const void*>(&gdn_scan_chunkwise_wy_tc_kernel<HD, SS, CHUNK>),
                cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024);
            attr_set = true;
        }
        gdn_scan_chunkwise_wy_tc_kernel<HD, SS, CHUNK><<<n_heads, HD, smem, stream>>>(
            conv_f32, alpha, beta, A_log, dt_bias, h_state, y, n_tokens, n_heads, n_groups, conv_channels,
            grouped_layout);
        return;
    }
    gdn_scan_fused_f32(conv_f32, conv_channels, alpha, beta, A_log, dt_bias, h_state, y, n_tokens, n_heads,
                       head_dim_ssm, state_size, n_groups, stream, grouped_layout);
}

// FP32-output variant — writes scan result as FP32 for downstream
// FP32-input RMSNorm+Gate (avoids FP16 subnormal-truncation at ~6e-5).
void gdn_scan_fused_fp32out(const float* conv_f32, int conv_channels, const half* alpha, const half* beta,
                            const float* A_log, const float* dt_bias, float* h_state, float* y_fp32,
                            int n_tokens, int n_heads, int head_dim_ssm, int state_size, int n_groups,
                            cudaStream_t stream, int grouped_layout) {
    size_t smem = (2 * state_size + head_dim_ssm) * sizeof(float);
    if (head_dim_ssm == 128 && state_size == 128) {
        gdn_scan_fused_kernel<128, 128, float>
            <<<n_heads, 128, smem, stream>>>(conv_f32, alpha, beta, A_log, dt_bias, h_state, y_fp32, n_tokens,
                                             n_heads, n_groups, conv_channels, grouped_layout);
    } else if (head_dim_ssm == 64 && state_size == 64) {
        gdn_scan_fused_kernel<64, 64, float>
            <<<n_heads, 64, smem, stream>>>(conv_f32, alpha, beta, A_log, dt_bias, h_state, y_fp32, n_tokens,
                                            n_heads, n_groups, conv_channels, grouped_layout);
    } else {
        // No FP32 fallback: supported sizes are HD=SS=128 / 64 only.
        // Fall back to FP16 path + post-hoc upcast (precision loss intact).
        int inner = n_heads * head_dim_ssm;
        int BC_size = n_groups * state_size;
        size_t smem_old = 2 * state_size * sizeof(float) + 2 * sizeof(float);
        std::vector<half> scratch_host(n_tokens * inner);
        half* scratch_dev = nullptr;
        cudaMallocAsync(&scratch_dev, n_tokens * inner * sizeof(half), stream);
        for (int t = 0; t < n_tokens; t++) {
            const float* row = conv_f32 + static_cast<size_t>(t) * conv_channels;
            gdn_scan_decode_kernel<<<n_heads, head_dim_ssm, smem_old, stream>>>(
                row + 2 * BC_size, row + BC_size, row, alpha + t * n_heads, beta + t * n_heads, A_log,
                dt_bias, h_state, scratch_dev + t * inner, nullptr, n_heads, head_dim_ssm, state_size,
                n_groups, grouped_layout);
        }
        // Convert FP16 scratch → FP32. Simple elementwise cast kernel not
        // present; do it via cuda memcpy + cast on device via small kernel.
        // Skip for now; assume 128/64 paths cover practical models.
        IMP_LOG_WARN("gdn_scan_fused_fp32out: unsupported HD=%d SS=%d — fallback not implemented",
                     head_dim_ssm, state_size);
        cudaFreeAsync(scratch_dev, stream);
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
    int grouped_layout) {
    const int h = blockIdx.x;
    if (h >= n_heads)
        return;
    const int d = threadIdx.x;  // [0, head_dim_v)
    const int SS = state_size;
    const int HD = head_dim_v;

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

        __syncthreads();  // before next token overwrites s_k/s_q/s_v
    }

    // Store state back to global.
    {
        float* H_dst = h_state + static_cast<size_t>(h) * SS * HD;
        for (int idx = d; idx < SS * HD; idx += HD) {
            H_dst[idx] = s_H[idx];
        }
    }
}

void gdn_scan_reference_f32(const float* conv_f32, int conv_channels, const half* alpha, const half* beta,
                            const float* A_log, const float* dt_bias, float* h_state, half* y, int n_tokens,
                            int n_heads, int head_dim_ssm, int state_size, int n_groups, cudaStream_t stream,
                            int grouped_layout) {
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
                                                                       conv_channels, grouped_layout);
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
}

// Fused RMSNormGated + SiLU
void gdn_rmsnorm_gated_silu(half* y, const half* gate, const half* weight, float eps, int n_tokens,
                            int n_heads, int head_dim, cudaStream_t stream) {
    size_t smem = head_dim * sizeof(float);
    dim3 grid(n_tokens, n_heads);
    gdn_rmsnorm_gated_silu_kernel<<<grid, head_dim, smem, stream>>>(y, gate, weight, eps, n_heads, head_dim);
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
    }
}

// Legacy stubs
void gdn_scan_decode(const half*, const half*, const half*, const half*, const half*, const float*,
                     const float*, float*, half*, const half*, int, int, int, int, cudaStream_t) {}
void gdn_scan_prefill(const half*, const half*, const half*, const half*, const half*, const float*,
                      const float*, float*, half*, const half*, int, int, int, int, int, cudaStream_t) {}
void gdn_decode(const half*, const half*, const half*, const half*, const half*, float*, half*, const half*,
                int, int, int, int, cudaStream_t) {}
void gdn_prefill(const half*, const half*, const half*, const half*, const half*, float*, half*, const half*,
                 int, int, int, int, int, cudaStream_t) {}

}  // namespace imp
