#include "compute/attention.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <float.h>

namespace imp {

// ---------------------------------------------------------------------------
// Flash Attention 2 -- Prefill Kernel
// ---------------------------------------------------------------------------
//
// Tile sizes: Br = 64 (query rows), Bc = 64 (key cols).
// Block: 128 threads (32 x 4).
//
// Memory layout [batch, seq, heads, head_dim] -- row-major.
//
// Shared memory regions (contiguous):
//   S_tile  : float [Br * Bc]       -- attention scores / softmax probs
//   Q_tile  : half  [Br * head_dim] -- query tile (loaded once)
//   KV_tile : half  [Bc * head_dim] -- key tile, then reused for value tile
//   O_acc   : float [Br * head_dim] -- output accumulator
//
// Each thread "owns" one row of the Br-tile (tid < Br) and performs all
// per-row work for that row.  Threads with tid >= Br assist only with
// cooperative shared-memory loads.
// ---------------------------------------------------------------------------

static constexpr int Br = 64;
static constexpr int Bc = 64;
static constexpr int WARP_SIZE = 32;
static constexpr int BLOCK_THREADS = 128;
static constexpr int NUM_WARPS = BLOCK_THREADS / WARP_SIZE;

__global__ void flash_attention_prefill_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    int batch_size,
    int seq_q,
    int seq_kv,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    float scale,
    bool causal,
    int sliding_window)
{
    // ---- index mapping ----
    const int tile_q     = blockIdx.x;                   // query-tile index
    const int batch_head = blockIdx.y;                   // flat (batch, head)
    const int batch_idx  = batch_head / n_heads;
    const int head_idx   = batch_head % n_heads;
    const int kv_head    = head_idx / (n_heads / n_kv_heads);  // GQA

    const int tid     = threadIdx.x + threadIdx.y * blockDim.x;  // [0,128)
    const int q_start = tile_q * Br;  // first query row in this tile

    // ---- global pointers (row-major strides) ----
    const int64_t q_row_stride  = (int64_t)n_heads    * head_dim;
    const int64_t kv_row_stride = (int64_t)n_kv_heads * head_dim;

    const half* Q_ptr = Q + (int64_t)batch_idx * seq_q  * q_row_stride
                          + (int64_t)q_start   * q_row_stride
                          + (int64_t)head_idx  * head_dim;

    const half* K_ptr = K + (int64_t)batch_idx * seq_kv * kv_row_stride
                          + (int64_t)kv_head   * head_dim;

    const half* V_ptr = V + (int64_t)batch_idx * seq_kv * kv_row_stride
                          + (int64_t)kv_head   * head_dim;

    half* O_ptr = O + (int64_t)batch_idx * seq_q * q_row_stride
                    + (int64_t)q_start   * q_row_stride
                    + (int64_t)head_idx  * head_dim;

    // ---- shared memory ----
    extern __shared__ char smem[];
    float* S_tile  = reinterpret_cast<float*>(smem);
    half*  Q_tile  = reinterpret_cast<half*>(S_tile + Br * Bc);
    half*  KV_tile = Q_tile + Br * head_dim;
    float* O_acc   = reinterpret_cast<float*>(KV_tile + Bc * head_dim);

    // ---- Load Q tile into shared memory (once) ----
    {
        const int total = Br * head_dim;
        for (int i = tid; i < total; i += BLOCK_THREADS) {
            int r = i / head_dim;
            int d = i % head_dim;
            if (q_start + r < seq_q) {
                Q_tile[i] = Q_ptr[(int64_t)r * q_row_stride + d];
            } else {
                Q_tile[i] = __float2half(0.0f);
            }
        }
    }

    // ---- Initialise O accumulator to zero ----
    {
        const int total = Br * head_dim;
        for (int i = tid; i < total; i += BLOCK_THREADS) {
            O_acc[i] = 0.0f;
        }
    }
    __syncthreads();

    // ---- Per-row softmax running state (thread-local) ----
    float m_i = -FLT_MAX;   // running max
    float l_i = 0.0f;       // running denominator

    // ---- Number of KV tiles to iterate ----
    int num_kv_tiles = (seq_kv + Bc - 1) / Bc;
    int first_kv_tile = 0;
    if (causal) {
        // No need to look at KV positions beyond the furthest query in this tile
        int max_q = q_start + Br - 1;
        if (max_q >= seq_q) max_q = seq_q - 1;
        int furthest_kv_tile = (max_q + Bc) / Bc;
        if (furthest_kv_tile < num_kv_tiles) num_kv_tiles = furthest_kv_tile;
    }
    if (sliding_window > 0) {
        // Skip KV tiles entirely before the window: earliest relevant KV pos
        // is q_start - sliding_window + 1 (for the first query in this tile).
        int earliest_kv = q_start - sliding_window + 1;
        if (earliest_kv > 0) {
            first_kv_tile = earliest_kv / Bc;
        }
    }

    for (int j = first_kv_tile; j < num_kv_tiles; j++) {
        const int kv_start = j * Bc;

        // ---- Load K tile into KV_tile ----
        {
            const int total = Bc * head_dim;
            for (int i = tid; i < total; i += BLOCK_THREADS) {
                int r = i / head_dim;
                int d = i % head_dim;
                if (kv_start + r < seq_kv) {
                    KV_tile[i] = K_ptr[(int64_t)(kv_start + r) * kv_row_stride + d];
                } else {
                    KV_tile[i] = __float2half(0.0f);
                }
            }
        }
        __syncthreads();

        // ---- Compute S = Q_tile @ KV_tile^T * scale  [Br, Bc] ----
        {
            const int total = Br * Bc;
            for (int i = tid; i < total; i += BLOCK_THREADS) {
                int r = i / Bc;  // query row within tile
                int c = i % Bc;  // key   col within tile
                int gq = q_start  + r;
                int gk = kv_start + c;

                float dot = 0.0f;
                if (gq < seq_q && gk < seq_kv) {
                    for (int d = 0; d < head_dim; d++) {
                        dot += __half2float(Q_tile[r * head_dim + d])
                             * __half2float(KV_tile[c * head_dim + d]);
                    }
                    dot *= scale;
                    if (causal && gq < gk) dot = -FLT_MAX;
                    if (sliding_window > 0 && (gq - gk) >= sliding_window) dot = -FLT_MAX;
                } else {
                    dot = -FLT_MAX;
                }
                S_tile[i] = dot;
            }
        }
        __syncthreads();

        // ---- Online softmax update + prepare P (per owning row) ----
        // Only threads with tid < Br and valid query row participate.
        // After this block: S_tile[row*Bc+c] holds P_{ij}(r,c) = exp(S - m_new).
        // m_i and l_i are updated to the new running state.
        // O_acc[row*hd+d] is rescaled by (alpha / l_new).

        float scale_pv = 0.0f;  // 1/l_new, used later in P@V

        if (tid < Br && (q_start + tid) < seq_q) {
            const int r = tid;

            // Row max of S tile
            float m_ij = -FLT_MAX;
            for (int c = 0; c < Bc; c++) {
                m_ij = fmaxf(m_ij, S_tile[r * Bc + c]);
            }

            float m_new = fmaxf(m_i, m_ij);

            // Compute P = exp(S - m_new) and its row sum
            float p_sum = 0.0f;
            for (int c = 0; c < Bc; c++) {
                float p = expf(S_tile[r * Bc + c] - m_new);
                S_tile[r * Bc + c] = p;
                p_sum += p;
            }

            // Rescale previous accumulator
            float alpha = expf(m_i - m_new) * l_i;
            float l_new = alpha + p_sum;

            float rescale = (l_new > 0.0f) ? (alpha / l_new) : 0.0f;
            scale_pv      = (l_new > 0.0f) ? (1.0f / l_new) : 0.0f;

            for (int d = 0; d < head_dim; d++) {
                O_acc[r * head_dim + d] *= rescale;
            }

            m_i = m_new;
            l_i = l_new;
        }
        __syncthreads();  // all S_tile writes (P values) visible; safe to reload KV

        // ---- Load V tile into KV_tile ----
        {
            const int total = Bc * head_dim;
            for (int i = tid; i < total; i += BLOCK_THREADS) {
                int r = i / head_dim;
                int d = i % head_dim;
                if (kv_start + r < seq_kv) {
                    KV_tile[i] = V_ptr[(int64_t)(kv_start + r) * kv_row_stride + d];
                } else {
                    KV_tile[i] = __float2half(0.0f);
                }
            }
        }
        __syncthreads();

        // ---- Accumulate P @ V into O_acc ----
        if (tid < Br && (q_start + tid) < seq_q) {
            const int r = tid;
            for (int d = 0; d < head_dim; d++) {
                float v_acc = 0.0f;
                for (int c = 0; c < Bc; c++) {
                    v_acc += S_tile[r * Bc + c] * __half2float(KV_tile[c * head_dim + d]);
                }
                O_acc[r * head_dim + d] += scale_pv * v_acc;
            }
        }
        __syncthreads();
    }

    // ---- Write final output ----
    if (tid < Br && (q_start + tid) < seq_q) {
        for (int d = 0; d < head_dim; d++) {
            O_ptr[(int64_t)tid * q_row_stride + d] = __float2half(O_acc[tid * head_dim + d]);
        }
    }
}

// ---------------------------------------------------------------------------
// Host launcher
// ---------------------------------------------------------------------------
void flash_attention_prefill(
    const Tensor& Q, const Tensor& K, const Tensor& V, Tensor& O,
    float scale, bool causal, int sliding_window, cudaStream_t stream)
{
    const int batch_size = static_cast<int>(Q.shape[0]);
    const int seq_q      = static_cast<int>(Q.shape[1]);
    const int n_heads    = static_cast<int>(Q.shape[2]);
    const int head_dim   = static_cast<int>(Q.shape[3]);
    const int seq_kv     = static_cast<int>(K.shape[1]);
    const int n_kv_heads = static_cast<int>(K.shape[2]);

    const int num_q_tiles = (seq_q + Br - 1) / Br;
    dim3 grid(num_q_tiles, batch_size * n_heads);
    dim3 block(WARP_SIZE, NUM_WARPS);  // 32 x 4 = 128

    size_t smem_bytes = Br * Bc * sizeof(float)           // S_tile
                      + Br * head_dim * sizeof(half)      // Q_tile
                      + Bc * head_dim * sizeof(half)      // KV_tile
                      + Br * head_dim * sizeof(float);    // O_acc

    cudaFuncSetAttribute(flash_attention_prefill_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         static_cast<int>(smem_bytes));

    flash_attention_prefill_kernel<<<grid, block, smem_bytes, stream>>>(
        reinterpret_cast<const half*>(Q.data),
        reinterpret_cast<const half*>(K.data),
        reinterpret_cast<const half*>(V.data),
        reinterpret_cast<half*>(O.data),
        batch_size, seq_q, seq_kv, n_heads, n_kv_heads, head_dim,
        scale, causal, sliding_window);
}

} // namespace imp
