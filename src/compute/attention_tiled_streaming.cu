// =============================================================================
// attention_tiled_streaming.cu — Track E v2: tiled FA2 attention for sm_120a
// =============================================================================
//
// Flash-Attention-2 style tiled streaming attention.  The key design goal is
// numerical agreement with the cuBLAS reference to < 5e-3 absolute error
// even for magnitude-1.0 Q/K/V inputs.
//
// Numerical strategy (differs from attention_fmha_sm120.cu):
//   • QK^T GEMM uses WMMA with FP32 accumulator → stored in S_tile as FP32.
//   • Online softmax step keeps P = exp(S − m) as FP32 in S_tile (no FP16
//     conversion, no per-tile normalization by l).
//   • PV GEMM is a scalar FP32 loop (no WMMA).  V is loaded as FP16 and
//     upcast to FP32 on the fly.  O_acc accumulates entirely in FP32.
//   • Final output = O_acc / l_final, converted to FP16 at store time.
//
// This matches cuBLAS's numerical path (FP32 attention weights, FP32 PV dot)
// far more closely than a kernel that stores P as FP16.
//
// Supported configs:
//   head_dim ∈ {64, 96, 128, 256}          (512 → returns false)
//   dtype: FP16 Q/K/V/O
//   GQA, causal, sliding_window, softcap, chunked-prefill (q_offset > 0)
// =============================================================================

#include "compute/attention_tiled_streaming.h"
#include "compute/attention_paged_common.cuh"
#include "core/logging.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <float.h>
#include <mma.h>

using namespace nvcuda;

namespace imp {

// =============================================================================
// Constants
// =============================================================================

static constexpr int kTE_WARP_SIZE     = 32;
static constexpr int kTE_NUM_WARPS     = 8;
static constexpr int kTE_BLOCK_THREADS = kTE_WARP_SIZE * kTE_NUM_WARPS;  // 256
static constexpr int kTE_Bkv           = 64;
static constexpr int kTE_WMMA_M        = 16;
static constexpr int kTE_WMMA_N        = 16;
static constexpr int kTE_WMMA_K        = 16;

// =============================================================================
// Kernel
// =============================================================================

template <int Bq, int HD>
__global__ void __launch_bounds__(kTE_BLOCK_THREADS, 1) track_e_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half*       __restrict__ O,
    int batch_size, int seq_q, int seq_kv,
    int n_heads, int n_kv_heads,
    float scale, bool causal, int sliding_window, float softcap,
    int q_offset)
{
    constexpr int Bkv      = kTE_Bkv;
    constexpr int head_dim = HD;
    // threads-per-row for the parallel softmax + PV scatter
    constexpr int TPR = kTE_BLOCK_THREADS / Bq;
    static_assert(TPR >= 1 && (TPR & (TPR-1)) == 0, "TPR must be pow2");

    // ---- block / thread indices -----------------------------------------------
    const int tile_q     = blockIdx.x;
    const int batch_head = blockIdx.y;
    const int batch_idx  = batch_head / n_heads;
    const int head_idx   = batch_head % n_heads;
    const int kv_head    = head_idx / (n_heads / n_kv_heads);

    const int tid     = threadIdx.x + threadIdx.y * blockDim.x;
    const int warp_id = tid / kTE_WARP_SIZE;
    const int q_start = tile_q * Bq;           // tile start within this chunk
    const int abs_q_start = q_offset + q_start; // absolute position of Q[0] of tile

    // parallel-softmax addressing
    const int sm_row  = tid / TPR;
    const int sm_lane = tid % TPR;

    // ---- global memory strides: [batch, seq, heads, head_dim] ----------------
    const int64_t q_row_stride  = (int64_t)n_heads    * head_dim;
    const int64_t kv_row_stride = (int64_t)n_kv_heads * head_dim;

    const half* Q_ptr = Q
        + (int64_t)batch_idx * seq_q  * q_row_stride
        + (int64_t)q_start            * q_row_stride
        + (int64_t)head_idx           * head_dim;
    const half* K_ptr = K
        + (int64_t)batch_idx * seq_kv * kv_row_stride
        + (int64_t)kv_head            * head_dim;
    const half* V_ptr = V
        + (int64_t)batch_idx * seq_kv * kv_row_stride
        + (int64_t)kv_head            * head_dim;
    half* O_ptr = O
        + (int64_t)batch_idx * seq_q  * q_row_stride
        + (int64_t)q_start            * q_row_stride
        + (int64_t)head_idx           * head_dim;

    // ---- shared memory layout -------------------------------------------------
    //   Q_tile  : half  [Bq  × HD]
    //   KV_tile : half  [Bkv × HD]   (K then V reuse same buffer)
    //   S_tile  : float [Bq  × Bkv]  (FP32 scores → FP32 probabilities, no FP16 cast)
    //   O_acc   : float [Bq  × HD]   (FP32 accumulator, normalised at final write)
    //   row_m   : float [Bq]
    //   row_l   : float [Bq]
    extern __shared__ char smem[];
    half*  Q_tile  = reinterpret_cast<half*>(smem);
    half*  KV_tile = Q_tile  + Bq  * head_dim;
    float* S_tile  = reinterpret_cast<float*>(KV_tile + Bkv * head_dim);
    float* O_acc   = S_tile  + Bq  * Bkv;
    float* row_m   = O_acc   + Bq  * head_dim;
    float* row_l   = row_m   + Bq;

    // ---- load Q tile (vectorised float4 = 8 halves per iter) ------------------
    {
        const int total_vec8 = (Bq * head_dim) / 8;
        for (int vi = tid; vi < total_vec8; vi += kTE_BLOCK_THREADS) {
            int i = vi * 8;
            int r = i / head_dim;
            int d = i % head_dim;
            float4* dst = reinterpret_cast<float4*>(&Q_tile[i]);
            if (q_start + r < seq_q) {
                const float4* src = reinterpret_cast<const float4*>(
                    &Q_ptr[(int64_t)r * q_row_stride + d]);
                *dst = *src;
            } else {
                *dst = make_float4(0.f, 0.f, 0.f, 0.f);
            }
        }
    }

    // ---- zero O_acc + init running softmax state ------------------------------
    {
        const int total_vec4 = (Bq * head_dim) / 4;
        const float4 zero = make_float4(0.f, 0.f, 0.f, 0.f);
        for (int vi = tid; vi < total_vec4; vi += kTE_BLOCK_THREADS)
            reinterpret_cast<float4*>(O_acc)[vi] = zero;
    }
    if (tid < Bq) {
        row_m[tid] = -FLT_MAX;
        row_l[tid] = 0.f;
    }
    __syncthreads();

    // ---- KV tile loop bounds --------------------------------------------------
    // compute_kv_tile_bounds uses absolute q_start for causal tile pruning.
    int num_kv_tiles, first_kv_tile;
    compute_kv_tile_bounds(abs_q_start, Bq, Bkv,
                           q_offset + seq_q,   // absolute end of this chunk
                           seq_kv,
                           causal, sliding_window,
                           first_kv_tile, num_kv_tiles);

    // ---- derived WMMA tiling constants ----------------------------------------
    const int hd_chunks   = head_dim / kTE_WMMA_K;    // QK^T k-reduction chunks
    const int s_row_tiles = Bq  / kTE_WMMA_M;
    const int s_col_tiles = Bkv / kTE_WMMA_N;
    const int s_total     = s_row_tiles * s_col_tiles;

    // ==========================================================================
    // Main loop over KV tiles
    // ==========================================================================
    for (int j = first_kv_tile; j < num_kv_tiles; j++) {
        const int kv_start = j * Bkv;

        // ---- load K tile -----------------------------------------------------
        {
            const int total_vec8 = (Bkv * head_dim) / 8;
            for (int vi = tid; vi < total_vec8; vi += kTE_BLOCK_THREADS) {
                int i = vi * 8;
                int r = i / head_dim;
                int d = i % head_dim;
                float4* dst = reinterpret_cast<float4*>(&KV_tile[i]);
                if (kv_start + r < seq_kv) {
                    const float4* src = reinterpret_cast<const float4*>(
                        &K_ptr[(int64_t)(kv_start + r) * kv_row_stride + d]);
                    *dst = *src;
                } else {
                    *dst = make_float4(0.f, 0.f, 0.f, 0.f);
                }
            }
        }
        __syncthreads();

        // ---- Phase 1: S = Q @ K^T  via WMMA (FP32 accumulator) ---------------
        for (int tile_idx = warp_id; tile_idx < s_total; tile_idx += kTE_NUM_WARPS) {
            int ri = tile_idx / s_col_tiles;
            int ci = tile_idx % s_col_tiles;

            wmma::fragment<wmma::accumulator,
                           kTE_WMMA_M, kTE_WMMA_N, kTE_WMMA_K, float> acc;
            wmma::fill_fragment(acc, 0.f);

            for (int k = 0; k < hd_chunks; k++) {
                wmma::fragment<wmma::matrix_a,
                               kTE_WMMA_M, kTE_WMMA_N, kTE_WMMA_K,
                               half, wmma::row_major> a_frag;
                wmma::load_matrix_sync(
                    a_frag,
                    Q_tile + ri * kTE_WMMA_M * head_dim + k * kTE_WMMA_K,
                    head_dim);

                wmma::fragment<wmma::matrix_b,
                               kTE_WMMA_M, kTE_WMMA_N, kTE_WMMA_K,
                               half, wmma::col_major> b_frag;
                wmma::load_matrix_sync(
                    b_frag,
                    KV_tile + ci * kTE_WMMA_N * head_dim + k * kTE_WMMA_K,
                    head_dim);

                wmma::mma_sync(acc, a_frag, b_frag, acc);
            }

            wmma::store_matrix_sync(
                S_tile + ri * kTE_WMMA_M * Bkv + ci * kTE_WMMA_N,
                acc, Bkv, wmma::mem_row_major);
        }
        __syncthreads();

        // ---- apply scale, softcap, causal/SWA mask ----------------------------
        // Pass abs_q_start so that chunked-prefill causal masking is correct.
        apply_score_masks(S_tile, Bq, Bkv, kTE_BLOCK_THREADS, tid,
                          abs_q_start, kv_start,
                          q_offset + seq_q, seq_kv,
                          scale, softcap, causal, sliding_window);
        __syncthreads();

        // ---- Phase 2: online softmax (fully FP32, P stored in S_tile) ---------
        // Strategy:  keep P = exp(S − m_new) as FP32 in S_tile.
        // O_acc is rescaled by exp(m_old − m_new) each tile.
        // Final normalization (÷ row_l) is deferred to the output write step.
        {
            const int r = sm_row;
            const bool row_valid = (r < Bq) && (q_start + r < seq_q);

            // Row max reduction across TPR lanes
            float partial_max = -FLT_MAX;
            if (row_valid) {
                for (int c = sm_lane; c < Bkv; c += TPR)
                    partial_max = fmaxf(partial_max, S_tile[r * Bkv + c]);
            }
#pragma unroll
            for (int off = TPR / 2; off >= 1; off >>= 1)
                partial_max = fmaxf(partial_max,
                    __shfl_xor_sync(0xffffffff, partial_max, off));
            float m_ij = partial_max;

            float m_old = row_valid ? row_m[r] : -FLT_MAX;
            float m_new = fmaxf(m_old, m_ij);
            float alpha = __expf(m_old - m_new);   // correction for O_acc

            // exp + sum; map masked (−∞) values to 0 explicitly
            float partial_sum = 0.f;
            if (row_valid) {
                for (int c = sm_lane; c < Bkv; c += TPR) {
                    float s_val = S_tile[r * Bkv + c];
                    float p = (s_val <= -FLT_MAX * 0.5f)
                                  ? 0.f : __expf(s_val - m_new);
                    partial_sum += p;
                    S_tile[r * Bkv + c] = p;  // store FP32 prob (unnormalised)
                }
            } else if (r < Bq) {
                // zero out padding rows so PV accumulation stays clean
                for (int c = sm_lane; c < Bkv; c += TPR)
                    S_tile[r * Bkv + c] = 0.f;
            }
#pragma unroll
            for (int off = TPR / 2; off >= 1; off >>= 1)
                partial_sum += __shfl_xor_sync(0xffffffff, partial_sum, off);

            float l_old = row_valid ? row_l[r] : 0.f;
            float l_new = alpha * l_old + partial_sum;
            if (sm_lane == 0 && row_valid) {
                row_m[r] = m_new;
                row_l[r] = l_new;
            }

            // Rescale O_acc rows: O_old *= exp(m_old - m_new)
            if (row_valid && alpha != 1.f) {
                for (int d = sm_lane; d < head_dim; d += TPR)
                    O_acc[r * head_dim + d] *= alpha;
            }
        }
        __syncthreads();

        // ---- load V tile into KV_tile ----------------------------------------
        {
            const int total_vec8 = (Bkv * head_dim) / 8;
            for (int vi = tid; vi < total_vec8; vi += kTE_BLOCK_THREADS) {
                int i = vi * 8;
                int r = i / head_dim;
                int d = i % head_dim;
                float4* dst = reinterpret_cast<float4*>(&KV_tile[i]);
                if (kv_start + r < seq_kv) {
                    const float4* src = reinterpret_cast<const float4*>(
                        &V_ptr[(int64_t)(kv_start + r) * kv_row_stride + d]);
                    *dst = *src;
                } else {
                    *dst = make_float4(0.f, 0.f, 0.f, 0.f);
                }
            }
        }
        __syncthreads();

        // ---- Phase 3: O_acc += P @ V  (scalar FP32, P unnormalised) ----------
        // S_tile[r][c] = P[r][c] = exp(S[r][c] - m_new)  (FP32)
        // KV_tile[c][d] = V[c][d]  (FP16, upcast inline)
        // Each thread row sm_row handles columns sm_lane, sm_lane+TPR, ...
        {
            const int r = sm_row;
            if (r < Bq) {
                for (int d = sm_lane; d < head_dim; d += TPR) {
                    float acc = O_acc[r * head_dim + d];
                    for (int c = 0; c < Bkv; c++) {
                        float p = S_tile[r * Bkv + c];
                        float v = __half2float(KV_tile[c * head_dim + d]);
                        acc += p * v;
                    }
                    O_acc[r * head_dim + d] = acc;
                }
            }
        }
        __syncthreads();
    }

    // ---- write final output: O = O_acc / row_l --------------------------------
    {
        const int r = sm_row;
        if (r < Bq && q_start + r < seq_q) {
            float inv_l = (row_l[r] > 0.f) ? (1.f / row_l[r]) : 0.f;
            for (int d = sm_lane; d < head_dim; d += TPR) {
                float v = O_acc[r * head_dim + d] * inv_l;
                O_ptr[(int64_t)r * q_row_stride + d] = __float2half(v);
            }
        }
    }
}

// =============================================================================
// Shared memory size helper
// =============================================================================

static size_t compute_smem_track_e(int Bq, int Bkv, int head_dim) {
    return (size_t)Bq  * head_dim * sizeof(half)   // Q_tile
         + (size_t)Bkv * head_dim * sizeof(half)   // KV_tile
         + (size_t)Bq  * Bkv     * sizeof(float)   // S_tile
         + (size_t)Bq  * head_dim * sizeof(float)  // O_acc
         + 2 * (size_t)Bq        * sizeof(float);  // row_m + row_l
}

// =============================================================================
// Public launcher
// =============================================================================

bool attention_tiled_streaming_prefill(const Tensor& Q, const Tensor& K,
                                       const Tensor& V, Tensor& O, float scale,
                                       bool causal, int sliding_window,
                                       float softcap, int q_offset,
                                       cudaStream_t stream) {
    if (Q.qtype != QType::F16 || K.qtype != QType::F16 || V.qtype != QType::F16)
        return false;
    if (Q.ndim != 4)
        return false;

    const int batch_size = static_cast<int>(Q.shape[0]);
    const int seq_q      = static_cast<int>(Q.shape[1]);
    const int n_heads    = static_cast<int>(Q.shape[2]);
    const int head_dim   = static_cast<int>(Q.shape[3]);
    const int seq_kv     = static_cast<int>(K.shape[1]);
    const int n_kv_heads = static_cast<int>(K.shape[2]);

    if (n_kv_heads == 0 || n_heads % n_kv_heads != 0) return false;
    if (seq_q == 0 || seq_kv == 0)                     return false;
    if (head_dim % kTE_WMMA_K != 0)                    return false;

    int device = 0;
    cudaGetDevice(&device);
    int max_smem = 0;
    cudaDeviceGetAttribute(&max_smem,
        cudaDevAttrMaxSharedMemoryPerBlockOptin, device);

    int Bq;
    {
        size_t smem128 = compute_smem_track_e(128, kTE_Bkv, head_dim);
        size_t smem64  = compute_smem_track_e( 64, kTE_Bkv, head_dim);
        size_t smem32  = compute_smem_track_e( 32, kTE_Bkv, head_dim);
        if      (smem128 <= (size_t)max_smem) Bq = 128;
        else if (smem64  <= (size_t)max_smem) Bq = 64;
        else if (smem32  <= (size_t)max_smem) Bq = 32;
        else {
            IMP_LOG_DEBUG("TrackE: no Bq fits smem (hd=%d)", head_dim);
            return false;
        }
    }
    const size_t smem = compute_smem_track_e(Bq, kTE_Bkv, head_dim);

    const int num_q_tiles = (seq_q + Bq - 1) / Bq;
    dim3 grid(num_q_tiles, batch_size * n_heads);
    dim3 block(kTE_WARP_SIZE, kTE_NUM_WARPS);

    IMP_LOG_DEBUG("TrackE: Sq=%d Skv=%d nh=%d hd=%d Bq=%d q_off=%d",
                  seq_q, seq_kv, n_heads, head_dim, Bq, q_offset);

#define LAUNCH_TE(BQ, HD)                                                            \
    do {                                                                             \
        cudaError_t _e = cudaFuncSetAttribute(                                       \
            track_e_kernel<BQ, HD>,                                                  \
            cudaFuncAttributeMaxDynamicSharedMemorySize,                              \
            static_cast<int>(smem));                                                 \
        if (_e != cudaSuccess) { return false; }                                     \
        cudaFuncSetAttribute(track_e_kernel<BQ, HD>,                                 \
                             cudaFuncAttributePreferredSharedMemoryCarveout,          \
                             cudaSharedmemCarveoutMaxShared);                        \
        track_e_kernel<BQ, HD><<<grid, block, smem, stream>>>(                       \
            reinterpret_cast<const half*>(Q.data),                                   \
            reinterpret_cast<const half*>(K.data),                                   \
            reinterpret_cast<const half*>(V.data),                                   \
            reinterpret_cast<half*>(O.data),                                         \
            batch_size, seq_q, seq_kv, n_heads, n_kv_heads,                          \
            scale, causal, sliding_window, softcap, q_offset);                       \
    } while (0)

    if (Bq == 128) {
        switch (head_dim) {
            case  64: LAUNCH_TE(128,  64); return true;
            case  96: LAUNCH_TE(128,  96); return true;
            case 128: LAUNCH_TE(128, 128); return true;
            case 256: LAUNCH_TE(128, 256); return true;
            default: break;
        }
    } else if (Bq == 64) {
        switch (head_dim) {
            case  64: LAUNCH_TE(64,  64); return true;
            case  96: LAUNCH_TE(64,  96); return true;
            case 128: LAUNCH_TE(64, 128); return true;
            case 256: LAUNCH_TE(64, 256); return true;
            default: break;
        }
    } else {
        switch (head_dim) {
            case  64: LAUNCH_TE(32,  64); return true;
            case  96: LAUNCH_TE(32,  96); return true;
            case 128: LAUNCH_TE(32, 128); return true;
            case 256: LAUNCH_TE(32, 256); return true;
            default: break;
        }
    }

#undef LAUNCH_TE

    return false;  // hd=512 or unsupported
}

}  // namespace imp
