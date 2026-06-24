#include "compute/gdn_internal.cuh"
#include <mma.h>

namespace imp {

namespace wmma = nvcuda::wmma;

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
        // NB: own-column-only reads (s_u_fp32[j·HD + d] is thread d's own
        // column) — no per-iteration __syncthreads() needed inside the loop.
        // The H_L step at the chunk end DOES read cross-thread u, so add
        // one sync after the loop.
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
        }
        __syncthreads();

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

        // Loop interchange (t outer, s inner) — sequential smem access to s_k_fp16
        // along s (stride 1) instead of stride-SS column access. Plus hoist the
        // per-thread per-t coefficient (s_g[t] · u_t[d]) out of the s loop.
        // Halves the H_L step's effective memory traffic vs the natural (s, t)
        // ordering and eliminates SS × L redundant scalar multiplications per
        // thread per chunk.
#pragma unroll
        for (int s = 0; s < SS; s++) {
            H_reg[s] *= D_0L;
        }
        for (int t_loc = 0; t_loc < L; t_loc++) {
            const float coef = s_g[t_loc] * s_u_fp32[t_loc * HD + d];
            const half* k_row = s_k_fp16 + t_loc * SS;
#pragma unroll
            for (int s = 0; s < SS; s++) {
                H_reg[s] += coef * __half2float(k_row[s]);
            }
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
// Phase 2c — fully-tuned WY-rep + TC-MMA, including H_L update.
//
// Builds on Phase 2b with:
//   1. CHUNK=32 (2× larger than Phase 2b) — half as many chunks per prefill,
//      half the per-chunk setup / sync / decay-precompute overhead.
//   2. Drops the persistent s_qh buffer. The s_kh buffer's 16 KiB is reused
//      twice across a chunk's lifetime: first holds KH (Step 4), then is
//      overwritten with QH (Step 5), then split into s_u_fp16 (8 KiB) +
//      s_strip_out_fp32 (8 KiB) for the Phase 2c TC-MMA H_L step.
//   3. **Phase 2c**: TC-MMA on the H_L update. The Σ_t (D[t+1..L] k̃_t u_t^T)
//      term is computed as a K̃^T · U_scaled matmul (M=SS=128, K=L=32,
//      N=HD=128). Processed in M-strips of 16 rows: per strip, all 8 N-tiles
//      computed in parallel (4 warps × 2 N-tiles each), output written to an
//      8 KiB s_strip_out_fp32 buffer, then each thread updates 16 elements
//      of H_reg from its column slice. Total: 8 strips × 8 N-tiles × 2 K-tiles
//      = 128 WMMA dispatches per chunk — same as KH but with different
//      operand layout.
//
// Smem layout (CHUNK=32, HD=SS=128, all bytes):
//   s_k_fp16[L*SS]          = 8 KiB
//   s_q_fp16[L*SS]          = 8 KiB
//   s_h0_fp16[SS*HD]        = 32 KiB
//   s_kh_buf[L*HD]          = 16 KiB  (KH → QH → s_u_fp16 + s_strip_out)
//   s_kk_fp32[L*L]          = 4 KiB
//   s_qk_fp32[L*L]          = 4 KiB
//   s_u_fp32[L*HD]          = 16 KiB
//   s_D[L+1] + s_g + s_beta + s_reduce ≈ 1 KiB
// Total ~89 KiB — fits in the 96 KiB sm_120 opt-in.
//
// Numerics: FP16 storage of K̃/Q̃/H_0/u_scaled introduces ~3-4 mantissa-bit
// drop on operands; WMMA FP32 accumulation preserves per-matmul precision.
// Expected output ≈ Phase 2b numerics (max_diff_y ~1e-5).
// ---------------------------------------------------------------------------
template <int HD, int SS, int CHUNK>
__global__ void __launch_bounds__(HD, 1) gdn_scan_chunkwise_wy_tc2_kernel(
    const float* __restrict__ conv_f32, const half* __restrict__ alpha_all,
    const half* __restrict__ beta_all, const float* __restrict__ A_log,
    const float* __restrict__ dt_bias, float* __restrict__ h_state, half* __restrict__ y_out,
    int n_tokens, int n_heads, int n_groups, int conv_channels, int grouped_layout) {
    static_assert(CHUNK == 32, "Phase 2c kernel tuned for CHUNK=32 only");
    static_assert(HD == 128 && SS == 128, "Phase 2c kernel tuned for HD=SS=128 only");
    static_assert(CHUNK % 16 == 0 && SS % 16 == 0 && HD % 16 == 0, "WMMA tiles must align");

    const int h = blockIdx.x;
    if (h >= n_heads)
        return;
    const int d = threadIdx.x;
    const int warp_id = d / 32;
    const int n_warps = HD / 32;  // 4 warps

    const int g_idx = grouped_layout ? (h / (n_heads / n_groups)) : (h % n_groups);
    const int inner = n_heads * HD;
    const int BC_size = n_groups * SS;
    const float scale = rsqrtf(static_cast<float>(HD));
    const float A_h = A_log[h];
    const float dtb_h = dt_bias[h];

    // State in registers.
    float H_reg[SS];
    {
        const float* H_col = h_state + static_cast<size_t>(h) * SS * HD + d;
#pragma unroll
        for (int s = 0; s < SS; s++)
            H_reg[s] = H_col[s * HD];
    }

    extern __shared__ float smem[];
    half* s_k_fp16 = reinterpret_cast<half*>(smem);                  // [L*SS]
    half* s_q_fp16 = s_k_fp16 + CHUNK * SS;                          // [L*SS]
    half* s_h0_fp16 = s_q_fp16 + CHUNK * SS;                         // [SS*HD]
    // s_kh_buf is the multipurpose 16 KiB region:
    //   - Step 4: KH (FP32 [L, HD])
    //   - Between Step 4 and 5: recomputed as QH (FP32 [L, HD])
    //   - Step 7: split as s_u_fp16 (FP16 [L, HD] = 8 KiB) + s_strip_out (FP32 [16, HD] = 8 KiB)
    float* s_kh_fp32 = reinterpret_cast<float*>(s_h0_fp16 + SS * HD);  // [L*HD]
    float* s_kk_fp32 = s_kh_fp32 + CHUNK * HD;                       // [L*L]
    float* s_qk_fp32 = s_kk_fp32 + CHUNK * CHUNK;                    // [L*L]
    float* s_u_fp32 = s_qk_fp32 + CHUNK * CHUNK;                     // [L*HD]
    float* s_D = s_u_fp32 + CHUNK * HD;                              // [L+1]
    float* s_g = s_D + (CHUNK + 1);                                  // [L]
    float* s_beta = s_g + CHUNK;                                     // [L]
    float* s_reduce = s_beta + CHUNK;                                // [HD]

    // Phase 7 buffer aliases (s_kh region reused).
    half* s_u_fp16 = reinterpret_cast<half*>(s_kh_fp32);              // [L*HD] (= 8 KiB)
    float* s_strip_out = reinterpret_cast<float*>(s_u_fp16 + CHUNK * HD);  // [16*HD] (= 8 KiB)

    int t_chunk_start = 0;
    while (t_chunk_start < n_tokens) {
        const int L = (t_chunk_start + CHUNK <= n_tokens) ? CHUNK : (n_tokens - t_chunk_start);

        // ---------------- STEP 1: load K, Q (FP32→FP16) and L2-normalise ----------------
        if (d < SS) {
            for (int t_loc = 0; t_loc < L; t_loc++) {
                const int t = t_chunk_start + t_loc;
                const float* row = conv_f32 + static_cast<size_t>(t) * conv_channels;
                s_q_fp16[t_loc * SS + d] = __float2half(row[g_idx * SS + d]);
                s_k_fp16[t_loc * SS + d] = __float2half(row[BC_size + g_idx * SS + d]);
            }
        }
        __syncthreads();

        // Per-token L2-norm reduction (same pattern as Phase 2b; could be
        // warp-parallelised but the sequential block-reduce is simple and
        // hasn't shown up as a bottleneck in profiling).
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

        // ---------------- STEP 2: g_t, β_t, cumulative decay ----------------
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
#pragma unroll
        for (int s = 0; s < SS; s++) {
            s_h0_fp16[s * HD + d] = __float2half(H_reg[s]);
        }
        __syncthreads();

        // ---------------- STEP 4a: KK, QK Gram (TC-MMA) ----------------
        // Warp 0: KK, Warp 1: QK. CHUNK=32 → 2x2 = 4 m,n tiles per L*L Gram.
        if (warp_id == 0 || warp_id == 1) {
            const half* a_src = (warp_id == 0) ? s_k_fp16 : s_q_fp16;
            const half* b_src = s_k_fp16;
            float* out = (warp_id == 0) ? s_kk_fp32 : s_qk_fp32;
            for (int m_tile = 0; m_tile < CHUNK / 16; m_tile++) {
                const int m_offset = m_tile * 16;
                for (int n_tile = 0; n_tile < CHUNK / 16; n_tile++) {
                    const int n_offset = n_tile * 16;
                    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
                    wmma::fill_fragment(c_frag, 0.0f);
                    for (int k = 0; k < SS / 16; k++) {
                        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
                        wmma::load_matrix_sync(a_frag, a_src + m_offset * SS + k * 16, SS);
                        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
                        wmma::load_matrix_sync(b_frag, b_src + n_offset * SS + k * 16, SS);
                        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
                    }
                    wmma::store_matrix_sync(out + m_offset * CHUNK + n_offset, c_frag, CHUNK,
                                            wmma::mem_row_major);
                }
            }
        }

        // ---------------- STEP 4b: KH (TC-MMA) ----------------
        // Output [L, HD] = [32, 128]. CHUNK/16 × HD/16 = 2 × 8 = 16 tiles.
        // 4 warps × 4 tiles each.
        {
            const int n_tiles_kh = (CHUNK / 16) * (HD / 16);  // 16
            for (int tile_idx = warp_id; tile_idx < n_tiles_kh; tile_idx += n_warps) {
                const int m_tile = tile_idx / (HD / 16);
                const int n_tile = tile_idx % (HD / 16);
                const int m_offset = m_tile * 16;
                const int n_offset = n_tile * 16;
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
                wmma::fill_fragment(c_frag, 0.0f);
                for (int k = 0; k < SS / 16; k++) {
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
                    wmma::load_matrix_sync(a_frag, s_k_fp16 + m_offset * SS + k * 16, SS);
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
                    wmma::load_matrix_sync(b_frag, s_h0_fp16 + k * 16 * HD + n_offset, HD);
                    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
                }
                wmma::store_matrix_sync(s_kh_fp32 + m_offset * HD + n_offset, c_frag, HD,
                                        wmma::mem_row_major);
            }
        }
        __syncthreads();

        // ---------------- STEP 5: triangular solve for u_t (uses KH from s_kh) ----------------
        // Own-column-only reads — no per-iteration sync needed.
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
        }
        __syncthreads();  // Before Step 6a TC-MMA overwrites s_kh.

        // ---------------- STEP 6a: overwrite s_kh with QH via TC-MMA ----------------
        // (s_kh's KH data is no longer needed after Step 5; reuse same 16 KiB for QH.)
        {
            const int n_tiles_qh = (CHUNK / 16) * (HD / 16);
            for (int tile_idx = warp_id; tile_idx < n_tiles_qh; tile_idx += n_warps) {
                const int m_tile = tile_idx / (HD / 16);
                const int n_tile = tile_idx % (HD / 16);
                const int m_offset = m_tile * 16;
                const int n_offset = n_tile * 16;
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
                wmma::fill_fragment(c_frag, 0.0f);
                for (int k = 0; k < SS / 16; k++) {
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
                    wmma::load_matrix_sync(a_frag, s_q_fp16 + m_offset * SS + k * 16, SS);
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
                    wmma::load_matrix_sync(b_frag, s_h0_fp16 + k * 16 * HD + n_offset, HD);
                    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
                }
                wmma::store_matrix_sync(s_kh_fp32 + m_offset * HD + n_offset, c_frag, HD,
                                        wmma::mem_row_major);
            }
        }
        __syncthreads();
        // Alias for clarity in the Y step.
        float* s_qh_fp32 = s_kh_fp32;

        // ---------------- STEP 6b: Y[t] using QH + QK + u ----------------
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
        __syncthreads();

        // ---------------- STEP 7: H_L update via TC-MMA on K̃^T · U_scaled ----------------
        // Pre-scale U by D[t+1..L] and convert to FP16 (in-place into s_kh_buf's first half).
        const float D_0L = s_D[L];
        if (d < L) {
            s_g[d] = D_0L / s_D[d + 1];  // D[d+1..L]
        }
        __syncthreads();
        for (int t_loc = 0; t_loc < L; t_loc++) {
            s_u_fp16[t_loc * HD + d] = __float2half(s_g[t_loc] * s_u_fp32[t_loc * HD + d]);
        }
        __syncthreads();

        // Pre-scale H_reg by D_0L.
#pragma unroll
        for (int s = 0; s < SS; s++) {
            H_reg[s] *= D_0L;
        }

        // M-strip loop: process 16 rows of H_L_add at a time.
        // Per strip: 8 N-tiles distributed across 4 warps (2 tiles per warp).
        // K iterations: L/16 = 32/16 = 2.
        const int n_strips = SS / 16;        // 8
        const int n_tiles_n = HD / 16;       // 8
        const int n_k_tiles = CHUNK / 16;    // 2
        for (int m_strip = 0; m_strip < n_strips; m_strip++) {
            const int m_offset = m_strip * 16;
            for (int n_tile = warp_id; n_tile < n_tiles_n; n_tile += n_warps) {
                const int n_offset = n_tile * 16;
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
                wmma::fill_fragment(c_frag, 0.0f);
                for (int k = 0; k < n_k_tiles; k++) {
                    // A = K̃^T loaded col_major from K̃ row-major storage.
                    // K̃[k_global, m_global] = s_k_fp16[k_global*SS + m_global].
                    // For col_major matrix_a with ld=SS: A[m_local, k_local] at
                    //   base[m_local + k_local*SS] where base = s_k_fp16 + k_offset*SS + m_offset.
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> a_frag;
                    wmma::load_matrix_sync(a_frag, s_k_fp16 + k * 16 * SS + m_offset, SS);
                    // B = U_scaled[k_global, n_global] at s_u_fp16[k_global*HD + n_global].
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
                    wmma::load_matrix_sync(b_frag, s_u_fp16 + k * 16 * HD + n_offset, HD);
                    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
                }
                wmma::store_matrix_sync(s_strip_out + n_offset, c_frag, HD, wmma::mem_row_major);
            }
            __syncthreads();

            // Each thread updates 16 elements of its H_reg column.
#pragma unroll
            for (int m_local = 0; m_local < 16; m_local++) {
                H_reg[m_offset + m_local] += s_strip_out[m_local * HD + d];
            }
            __syncthreads();  // Before next strip overwrites s_strip_out.
        }

        t_chunk_start += L;
    }

    // Write final state back.
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

// Phase 2c host launcher. HD=SS=128 + CHUNK=32 path; other shapes fall back.
void gdn_scan_chunkwise_wy_tc2_f32(const float* conv_f32, int conv_channels, const half* alpha,
                                   const half* beta, const float* A_log, const float* dt_bias,
                                   float* h_state, half* y, int n_tokens, int n_heads, int head_dim_ssm,
                                   int state_size, int n_groups, cudaStream_t stream, int grouped_layout) {
    if (head_dim_ssm == 128 && state_size == 128 && n_tokens >= 1) {
        constexpr int HD = 128, SS = 128, CHUNK = 32;
        // Smem (see kernel header for full breakdown): ~89 KiB total.
        const size_t smem = (2 * CHUNK * SS) * sizeof(half) + (SS * HD) * sizeof(half) +
                            (CHUNK * HD + 2 * CHUNK * CHUNK + CHUNK * HD + (CHUNK + 1) + 2 * CHUNK + HD) *
                                sizeof(float);
        static bool attr_set = false;
        if (!attr_set) {
            cudaFuncSetAttribute(
                reinterpret_cast<const void*>(&gdn_scan_chunkwise_wy_tc2_kernel<HD, SS, CHUNK>),
                cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024);
            attr_set = true;
        }
        gdn_scan_chunkwise_wy_tc2_kernel<HD, SS, CHUNK><<<n_heads, HD, smem, stream>>>(
            conv_f32, alpha, beta, A_log, dt_bias, h_state, y, n_tokens, n_heads, n_groups, conv_channels,
            grouped_layout);
        return;
    }
    gdn_scan_fused_f32(conv_f32, conv_channels, alpha, beta, A_log, dt_bias, h_state, y, n_tokens, n_heads,
                       head_dim_ssm, state_size, n_groups, stream, grouped_layout);
}

}  // namespace imp
