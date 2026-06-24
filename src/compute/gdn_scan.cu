#include "compute/gdn_internal.cuh"

namespace imp {

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
        //
        // NB: thread d writes s_u[t·HD + d] and only reads s_u[j·HD + d] for
        // j < t — its OWN column from previous iterations. No cross-thread
        // reads of s_u inside the loop, so the per-iteration __syncthreads()
        // is unnecessary. One sync after the loop is needed before Step 5
        // (which DOES read other threads' u columns via QK matmul).
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
        }
        __syncthreads();  // Before Step 5 reads cross-thread u columns.

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
        // Tuning (2026-05-24):
        //   - Hoist D[t+1..L] = exp(logD[L] - logD[t+1]) out of the (s, t) double
        //     loop. The decay factor depends only on t, so computing it L*SS times
        //     per chunk per head is wasted work. Precompute once into s_g (which
        //     is reusable scratch by this point in the chunk).
        //   - Loop interchange (t outer, s inner). The inner loop now walks
        //     s_k[t*SS + s] with stride 1 in s, instead of the natural (s, t)
        //     order's stride-SS column access. Sequential reads → much better
        //     L1/coalescing.
        //   - Hoist the per-thread per-t coefficient (D[t+1..L] · u_t[d]) out of
        //     the s loop so the inner s loop is one FMA per element.
        {
            const float D_0L = expf(s_logD[L]);
            if (d < L) {
                s_g[d] = expf(s_logD[L] - s_logD[d + 1]);  // D[d+1..L], shared
            }
            __syncthreads();
#pragma unroll
            for (int s = 0; s < SS; s++) {
                H_reg[s] *= D_0L;
            }
            for (int t_loc = 0; t_loc < L; t_loc++) {
                const float coef = s_g[t_loc] * s_u[t_loc * HD + d];
                const float* k_row = s_k + t_loc * SS;
#pragma unroll
                for (int s = 0; s < SS; s++) {
                    H_reg[s] += coef * k_row[s];
                }
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
// Host launchers
// ---------------------------------------------------------------------------

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

}  // namespace imp
