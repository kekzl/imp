// =============================================================================
// attention_blackwell.cu -- Optimized WMMA attention for sm_120 (Blackwell)
// =============================================================================
//
// Flash Attention 2 kernel using WMMA tensor cores with 8 warps (256 threads),
// double-buffered KV tiles, and adaptive Q tile height.
//
// Key improvements over the 64x64 / 4-warp WMMA path (attention_tc.cu):
//   - 8 warps: 2x WMMA parallelism, each warp handles fewer tiles
//   - Double-buffered KV: overlaps next-K prefetch with current-tile computation
//   - S/P union shared memory: float S and half P share the same region
//   - Adaptive Br: 128-row Q tiles when shared memory allows (head_dim <= 64),
//     64-row Q tiles otherwise — both with 8 warps for better utilisation
//
// The RTX 5090 (sm_120) has 100 KB shared memory per SM with 99 KB opt-in max.
// Layout for Br=128, Bc=64, HD=64: ~96.5 KB (fits)
// Layout for Br=64, Bc=64, HD=128:  ~96.3 KB (fits)
// =============================================================================

#include "compute/attention_tc.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <float.h>
#include <mma.h>

using namespace nvcuda;

namespace imp {

// Fixed tile/thread parameters
static constexpr int BW_Bc             = 64;   // key tile cols (always 64)
static constexpr int BW_WARP_SIZE      = 32;
static constexpr int BW_NUM_WARPS      = 8;
static constexpr int BW_BLOCK_THREADS  = BW_WARP_SIZE * BW_NUM_WARPS;  // 256

// WMMA tile dimensions
static constexpr int WMMA_M = 16;
static constexpr int WMMA_N = 16;
static constexpr int WMMA_K = 16;

// ---- kernel (templated on Br) -----------------------------------------------

template <int Br, int HD>
__global__ void flash_attention_blackwell_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half*       __restrict__ O,
    int   batch_size,
    int   seq_q,
    int   seq_kv,
    int   n_heads,
    int   n_kv_heads,
    float scale,
    bool  causal,
    int   sliding_window,
    float softcap)
{
    constexpr int head_dim = HD;  // compile-time head_dim for optimized div/mod

    // ---- index computation --------------------------------------------------
    const int tile_q     = blockIdx.x;
    const int batch_head = blockIdx.y;
    const int batch_idx  = batch_head / n_heads;
    const int head_idx   = batch_head % n_heads;
    const int kv_head    = head_idx / (n_heads / n_kv_heads);

    const int tid     = threadIdx.x + threadIdx.y * blockDim.x;  // [0,256)
    const int warp_id = tid / BW_WARP_SIZE;  // [0,8)
    const int q_start = tile_q * Br;

    // Global memory strides (row-major [batch, seq, heads, head_dim]).
    const int64_t q_row_stride  = (int64_t)n_heads    * head_dim;
    const int64_t kv_row_stride = (int64_t)n_kv_heads * head_dim;

    const half* Q_ptr = Q + (int64_t)batch_idx * seq_q  * q_row_stride
                          + (int64_t)q_start   * q_row_stride
                          + (int64_t)head_idx  * head_dim;
    const half* K_ptr = K + (int64_t)batch_idx * seq_kv * kv_row_stride
                          + (int64_t)kv_head   * head_dim;
    const half* V_ptr = V + (int64_t)batch_idx * seq_kv * kv_row_stride
                          + (int64_t)kv_head   * head_dim;
    half* O_ptr       = O + (int64_t)batch_idx * seq_q  * q_row_stride
                          + (int64_t)q_start   * q_row_stride
                          + (int64_t)head_idx  * head_dim;

    // ---- shared memory layout -----------------------------------------------
    //
    // Q_tile      : half  [Br  × hd]
    // KV_buf[0]   : half  [Bc  × hd]   double-buffer slot 0
    // KV_buf[1]   : half  [Bc  × hd]   double-buffer slot 1
    // SP_tile     : union { float [Br×Bc], half [Br×Bc] }
    // O_acc       : float [Br  × hd]
    // scale_pv    : float [Br]
    extern __shared__ char smem[];

    half*  Q_tile    = reinterpret_cast<half*>(smem);
    half*  KV_buf0   = Q_tile  + Br * head_dim;
    half*  KV_buf1   = KV_buf0 + BW_Bc * head_dim;
    // SP_tile: union of float[Br*Bc] and half[Br*Bc]
    float* SP_float  = reinterpret_cast<float*>(KV_buf1 + BW_Bc * head_dim);
    half*  SP_half   = reinterpret_cast<half*>(SP_float);
    float* O_acc     = reinterpret_cast<float*>(
                           reinterpret_cast<char*>(SP_float) +
                           Br * BW_Bc * sizeof(float));
    float* scale_pv_shared = reinterpret_cast<float*>(O_acc + Br * head_dim);

    half* KV_bufs[2] = { KV_buf0, KV_buf1 };

    // ---- load Q tile --------------------------------------------------------
    {
        const int total = Br * head_dim;
        for (int i = tid; i < total; i += BW_BLOCK_THREADS) {
            int r = i / head_dim;
            int d = i % head_dim;
            if (q_start + r < seq_q) {
                Q_tile[i] = Q_ptr[(int64_t)r * q_row_stride + d];
            } else {
                Q_tile[i] = __float2half(0.0f);
            }
        }
    }

    // ---- zero output accumulator --------------------------------------------
    {
        const int total = Br * head_dim;
        for (int i = tid; i < total; i += BW_BLOCK_THREADS) {
            O_acc[i] = 0.0f;
        }
    }
    __syncthreads();

    // ---- per-row softmax running state (thread-local) ----
    float m_i = -FLT_MAX;
    float l_i = 0.0f;

    // ---- number of KV tiles to iterate ----
    int num_kv_tiles = (seq_kv + BW_Bc - 1) / BW_Bc;
    int first_kv_tile = 0;
    if (causal) {
        int max_q = q_start + Br - 1;
        if (max_q >= seq_q) max_q = seq_q - 1;
        int furthest_kv_tile = (max_q + BW_Bc) / BW_Bc;
        if (furthest_kv_tile < num_kv_tiles) num_kv_tiles = furthest_kv_tile;
    }
    if (sliding_window > 0) {
        int earliest_kv = q_start - sliding_window + 1;
        if (earliest_kv > 0) {
            first_kv_tile = earliest_kv / BW_Bc;
        }
    }

    // Derived constants for WMMA tiling
    const int hd_chunks     = head_dim / WMMA_K;
    const int s_row_tiles   = Br / WMMA_M;
    const int s_col_tiles   = BW_Bc / WMMA_N;             // 4
    const int s_total_tiles = s_row_tiles * s_col_tiles;
    const int o_row_tiles   = Br / WMMA_M;
    const int o_col_tiles   = head_dim / WMMA_N;
    const int o_total_tiles = o_row_tiles * o_col_tiles;
    const int pv_chunks     = BW_Bc / WMMA_K;             // 4

    // ---- prefetch first K tile into buf[0] ----
    int cur_buf = 0;
    if (first_kv_tile < num_kv_tiles) {
        const int kv_start = first_kv_tile * BW_Bc;
        const int total = BW_Bc * head_dim;
        for (int i = tid; i < total; i += BW_BLOCK_THREADS) {
            int r = i / head_dim;
            int d = i % head_dim;
            if (kv_start + r < seq_kv) {
                KV_bufs[0][i] = K_ptr[(int64_t)(kv_start + r) * kv_row_stride + d];
            } else {
                KV_bufs[0][i] = __float2half(0.0f);
            }
        }
    }
    __syncthreads();

    // ================================================================
    // Main loop over KV tiles
    // ================================================================
    for (int j = first_kv_tile; j < num_kv_tiles; j++) {
        const int kv_start = j * BW_Bc;

        // K[j] is in KV_bufs[cur_buf], ready.

        // ============================================================
        // Phase 1: S = Q_tile @ KV_buf[cur]^T  [Br, Bc] using WMMA
        // ============================================================
        for (int tile_idx = warp_id; tile_idx < s_total_tiles; tile_idx += BW_NUM_WARPS) {
            int ri = tile_idx / s_col_tiles;
            int ci = tile_idx % s_col_tiles;

            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
            wmma::fill_fragment(acc, 0.0f);

            for (int k = 0; k < hd_chunks; k++) {
                wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                               half, wmma::row_major> a_frag;
                wmma::load_matrix_sync(a_frag,
                    Q_tile + ri * WMMA_M * head_dim + k * WMMA_K,
                    head_dim);

                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                               half, wmma::col_major> b_frag;
                wmma::load_matrix_sync(b_frag,
                    KV_bufs[cur_buf] + ci * WMMA_N * head_dim + k * WMMA_K,
                    head_dim);

                wmma::mma_sync(acc, a_frag, b_frag, acc);
            }

            wmma::store_matrix_sync(
                SP_float + ri * WMMA_M * BW_Bc + ci * WMMA_N,
                acc, BW_Bc, wmma::mem_row_major);
        }
        __syncthreads();

        // ---- Apply scale, softcap, and causal/sliding_window mask ----
        {
            const int total = Br * BW_Bc;
            for (int i = tid; i < total; i += BW_BLOCK_THREADS) {
                int r = i / BW_Bc;
                int c = i % BW_Bc;
                int gq = q_start  + r;
                int gk = kv_start + c;

                if (gq < seq_q && gk < seq_kv) {
                    float val = SP_float[i] * scale;
                    if (softcap > 0.0f) val = softcap * tanhf(val / softcap);
                    if (causal && gq < gk) val = -FLT_MAX;
                    if (sliding_window > 0 && (gq - gk) >= sliding_window) val = -FLT_MAX;
                    SP_float[i] = val;
                } else {
                    SP_float[i] = -FLT_MAX;
                }
            }
        }
        __syncthreads();

        // ============================================================
        // Phase 2: Online softmax + rescale O accumulator
        // ============================================================
        if (tid < Br && (q_start + tid) < seq_q) {
            const int r = tid;

            float m_ij = -FLT_MAX;
            for (int c = 0; c < BW_Bc; c++) {
                m_ij = fmaxf(m_ij, SP_float[r * BW_Bc + c]);
            }

            float m_new = fmaxf(m_i, m_ij);

            float p_sum = 0.0f;
            for (int c = 0; c < BW_Bc; c++) {
                float p = expf(SP_float[r * BW_Bc + c] - m_new);
                SP_float[r * BW_Bc + c] = p;
                p_sum += p;
            }

            float alpha = expf(m_i - m_new) * l_i;
            float l_new = alpha + p_sum;

            float rescale = (l_new > 0.0f) ? (alpha / l_new) : 0.0f;
            float spv     = (l_new > 0.0f) ? (1.0f / l_new)  : 0.0f;

            for (int d = 0; d < head_dim; d++) {
                O_acc[r * head_dim + d] *= rescale;
            }

            scale_pv_shared[r] = spv;

            m_i = m_new;
            l_i = l_new;
        }
        __syncthreads();

        // ============================================================
        // Phase 3: Convert SP_float -> SP_half with scale_pv baked in
        // ============================================================
        // SP_half aliases SP_float. Since sizeof(half) < sizeof(float),
        // writing half[i] doesn't corrupt later float[i] reads because
        // we process in order and each element is read (float) then
        // written (half) exactly once.
        {
            const int total = Br * BW_Bc;
            for (int i = tid; i < total; i += BW_BLOCK_THREADS) {
                int r = i / BW_Bc;
                float spv = scale_pv_shared[r];
                float val = SP_float[i];
                SP_half[i] = __float2half(val * spv);
            }
        }
        __syncthreads();

        // ---- Load V tile into KV_bufs[cur_buf] (K no longer needed) ----
        {
            const int total = BW_Bc * head_dim;
            for (int i = tid; i < total; i += BW_BLOCK_THREADS) {
                int r = i / head_dim;
                int d = i % head_dim;
                if (kv_start + r < seq_kv) {
                    KV_bufs[cur_buf][i] = V_ptr[(int64_t)(kv_start + r) * kv_row_stride + d];
                } else {
                    KV_bufs[cur_buf][i] = __float2half(0.0f);
                }
            }
        }
        __syncthreads();

        // ============================================================
        // Phase 4: O += P @ V  [Br, head_dim] using WMMA
        // ============================================================
        for (int tile_idx = warp_id; tile_idx < o_total_tiles; tile_idx += BW_NUM_WARPS) {
            int ri = tile_idx / o_col_tiles;
            int di = tile_idx % o_col_tiles;

            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> o_frag;
            wmma::load_matrix_sync(o_frag,
                O_acc + ri * WMMA_M * head_dim + di * WMMA_N,
                head_dim, wmma::mem_row_major);

            for (int k = 0; k < pv_chunks; k++) {
                wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                               half, wmma::row_major> p_frag;
                wmma::load_matrix_sync(p_frag,
                    SP_half + ri * WMMA_M * BW_Bc + k * WMMA_K,
                    BW_Bc);

                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                               half, wmma::row_major> v_frag;
                wmma::load_matrix_sync(v_frag,
                    KV_bufs[cur_buf] + k * WMMA_N * head_dim + di * WMMA_N,
                    head_dim);

                wmma::mma_sync(o_frag, p_frag, v_frag, o_frag);
            }

            wmma::store_matrix_sync(
                O_acc + ri * WMMA_M * head_dim + di * WMMA_N,
                o_frag, head_dim, wmma::mem_row_major);
        }
        __syncthreads();

        // ---- Prefetch K[j+1] into KV_bufs[1-cur_buf] ----
        int next_buf = 1 - cur_buf;
        if (j + 1 < num_kv_tiles) {
            const int next_kv_start = (j + 1) * BW_Bc;
            const int total = BW_Bc * head_dim;
            for (int i = tid; i < total; i += BW_BLOCK_THREADS) {
                int r = i / head_dim;
                int d = i % head_dim;
                if (next_kv_start + r < seq_kv) {
                    KV_bufs[next_buf][i] = K_ptr[(int64_t)(next_kv_start + r) * kv_row_stride + d];
                } else {
                    KV_bufs[next_buf][i] = __float2half(0.0f);
                }
            }
        }
        cur_buf = next_buf;
        __syncthreads();
    }

    // ---- write final output to global memory ----
    {
        const int total = Br * head_dim;
        for (int i = tid; i < total; i += BW_BLOCK_THREADS) {
            int r = i / head_dim;
            if (q_start + r < seq_q) {
                O_ptr[(int64_t)r * q_row_stride + (i % head_dim)] =
                    __float2half(O_acc[i]);
            }
        }
    }
}

// Compute shared memory for a given Br and head_dim
static size_t compute_smem(int Br, int head_dim) {
    return (size_t)Br * head_dim * sizeof(half)          // Q_tile
         + 2 * (size_t)BW_Bc * head_dim * sizeof(half)  // KV_buf[0] + KV_buf[1]
         + (size_t)Br * BW_Bc * sizeof(float)            // SP_tile (float union)
         + (size_t)Br * head_dim * sizeof(float)          // O_acc
         + (size_t)Br * sizeof(float);                    // scale_pv
}

// ===== Host-side launcher ====================================================

void flash_attention_blackwell(
    const Tensor& Q, const Tensor& K, const Tensor& V, Tensor& O,
    float scale, bool causal, int sliding_window, float softcap, cudaStream_t stream)
{
    const int batch_size = static_cast<int>(Q.shape[0]);
    const int seq_q      = static_cast<int>(Q.shape[1]);
    const int n_heads    = static_cast<int>(Q.shape[2]);
    const int head_dim   = static_cast<int>(Q.shape[3]);
    const int seq_kv     = static_cast<int>(K.shape[1]);
    const int n_kv_heads = static_cast<int>(K.shape[2]);

    // Query device shared memory limit
    int device = 0;
    cudaGetDevice(&device);
    int max_smem = 0;
    cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);

    // Choose Br: prefer 128 if shared memory fits, else 64
    const size_t smem_128 = compute_smem(128, head_dim);
    const size_t smem_64  = compute_smem(64,  head_dim);
    const bool use_br128  = (smem_128 <= (size_t)max_smem);

    // Dispatch macro: Br x HD template instantiation
    #define LAUNCH_BW(BR, HD) do { \
        cudaFuncSetAttribute( \
            flash_attention_blackwell_kernel<BR, HD>, \
            cudaFuncAttributeMaxDynamicSharedMemorySize, \
            static_cast<int>(smem)); \
        flash_attention_blackwell_kernel<BR, HD><<<grid, block, smem, stream>>>( \
            reinterpret_cast<const half*>(Q.data), \
            reinterpret_cast<const half*>(K.data), \
            reinterpret_cast<const half*>(V.data), \
            reinterpret_cast<half*>(O.data), \
            batch_size, seq_q, seq_kv, \
            n_heads, n_kv_heads, \
            scale, causal, sliding_window, softcap); \
    } while (0)

    // Select Br and compute grid
    auto launch = [&](int Br, size_t smem) -> bool {
        const int num_q_tiles = (seq_q + Br - 1) / Br;
        dim3 grid(num_q_tiles, batch_size * n_heads);
        dim3 block(BW_WARP_SIZE, BW_NUM_WARPS);

        bool launched = false;
        if (Br == 128) {
            switch (head_dim) {
                case 64:  LAUNCH_BW(128, 64);  launched = true; break;
                case 96:  LAUNCH_BW(128, 96);  launched = true; break;
                default: break;
            }
        } else {
            switch (head_dim) {
                case 64:  LAUNCH_BW(64, 64);   launched = true; break;
                case 96:  LAUNCH_BW(64, 96);   launched = true; break;
                case 128: LAUNCH_BW(64, 128);  launched = true; break;
                case 256: LAUNCH_BW(64, 256);  launched = true; break;
                default: break;
            }
        }
        return launched;
    };

    bool launched = false;
    if (use_br128) {
        launched = launch(128, smem_128);
    } else if (smem_64 <= (size_t)max_smem) {
        launched = launch(64, smem_64);
    }
    if (!launched) {
        // Unsupported head_dim or smem too small; fall back to tc path
        flash_attention_prefill_tc(Q, K, V, O, scale, causal, sliding_window, softcap, stream);
    }
    #undef LAUNCH_BW
}

} // namespace imp
