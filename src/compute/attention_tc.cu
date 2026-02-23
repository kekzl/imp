#include "compute/attention_tc.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <float.h>
#include <mma.h>

using namespace nvcuda;

namespace imp {

// ---------------------------------------------------------------------------
// Flash Attention 2 -- Tensor-Core (WMMA) Prefill Kernel
// ---------------------------------------------------------------------------
//
// Same tiling strategy as the scalar kernel but replaces scalar dot-products
// with WMMA 16x16x16 fp16 -> fp32 matrix multiply-accumulate operations.
//
// Tile sizes: Br = 64 (query rows), Bc = 64 (key cols).
// Block: 128 threads = 4 warps.
// WMMA tile: 16x16x16 (M=16, N=16, K=16).
//
// For S = Q @ K^T  [Br x Bc]:
//   - Br/16 = 4 row-tiles, Bc/16 = 4 col-tiles -> 16 output tiles (4x4).
//   - 4 warps handle 4 output tiles each, cycling through all 16 tiles.
//   - Each output tile accumulates head_dim/16 WMMA ops.
//
// For O += P @ V  [Br x head_dim]:
//   - Br/16 = 4 row-tiles, head_dim/16 col-tiles.
//   - Similarly distributed across 4 warps.
//
// Shared memory layout (contiguous):
//   Q_tile  : half  [Br * head_dim]   -- query tile (loaded once)
//   KV_tile : half  [Bc * head_dim]   -- key tile, then reused for value tile
//   S_tile  : float [Br * Bc]         -- score / probability matrix
//   O_acc   : float [Br * head_dim]   -- output accumulator
//   P_half  : half  [Br * Bc]         -- half-precision P for WMMA P@V
//   scale_pv_shared : float [Br]      -- per-row 1/l_new broadcast array
//
// This file must be compiled with -arch=sm_90 (or higher) to enable WMMA.
// The host launcher performs a runtime check and falls back gracefully.
// ---------------------------------------------------------------------------

static constexpr int TC_Br = 64;
static constexpr int TC_Bc = 64;
static constexpr int TC_WARP_SIZE = 32;
static constexpr int TC_BLOCK_THREADS = 128;
static constexpr int TC_NUM_WARPS = TC_BLOCK_THREADS / TC_WARP_SIZE;

// WMMA tile dimensions
static constexpr int WMMA_M = 16;
static constexpr int WMMA_N = 16;
static constexpr int WMMA_K = 16;

__global__ void flash_attention_prefill_tc_kernel(
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
    bool causal)
{
    // ---- index mapping ----
    const int tile_q     = blockIdx.x;                   // query-tile index
    const int batch_head = blockIdx.y;                   // flat (batch, head)
    const int batch_idx  = batch_head / n_heads;
    const int head_idx   = batch_head % n_heads;
    const int kv_head    = head_idx / (n_heads / n_kv_heads);  // GQA

    const int tid       = threadIdx.x + threadIdx.y * blockDim.x;  // [0,128)
    const int warp_id   = tid / TC_WARP_SIZE;   // [0,4)
    const int q_start   = tile_q * TC_Br;       // first query row in this tile

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

    // ---- shared memory layout ----
    //   [Q_tile | KV_tile | S_tile | O_acc | P_half | scale_pv_shared]
    extern __shared__ char smem[];
    half*  Q_tile  = reinterpret_cast<half*>(smem);
    half*  KV_tile = Q_tile + TC_Br * head_dim;
    float* S_tile  = reinterpret_cast<float*>(KV_tile + TC_Bc * head_dim);
    float* O_acc   = S_tile + TC_Br * TC_Bc;
    half*  P_half  = reinterpret_cast<half*>(O_acc + TC_Br * head_dim);
    float* scale_pv_shared = reinterpret_cast<float*>(P_half + TC_Br * TC_Bc);

    // ---- Load Q tile into shared memory (once) ----
    // Q_tile layout: [Br, head_dim] row-major in shared memory.
    {
        const int total = TC_Br * head_dim;
        for (int i = tid; i < total; i += TC_BLOCK_THREADS) {
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
        const int total = TC_Br * head_dim;
        for (int i = tid; i < total; i += TC_BLOCK_THREADS) {
            O_acc[i] = 0.0f;
        }
    }
    __syncthreads();

    // ---- Per-row softmax running state (thread-local) ----
    // Only threads with tid < TC_Br own a row.
    float m_i = -FLT_MAX;   // running max
    float l_i = 0.0f;       // running denominator

    // ---- Number of KV tiles to iterate ----
    int num_kv_tiles = (seq_kv + TC_Bc - 1) / TC_Bc;
    if (causal) {
        int max_q = q_start + TC_Br - 1;
        if (max_q >= seq_q) max_q = seq_q - 1;
        int furthest_kv_tile = (max_q + TC_Bc) / TC_Bc;
        if (furthest_kv_tile < num_kv_tiles) num_kv_tiles = furthest_kv_tile;
    }

    // Derived constants for WMMA tiling
    const int hd_chunks   = head_dim / WMMA_K;              // e.g. 128/16 = 8
    const int s_row_tiles = TC_Br / WMMA_M;                 // 4
    const int s_col_tiles = TC_Bc / WMMA_N;                 // 4
    const int s_total_tiles = s_row_tiles * s_col_tiles;     // 16
    const int o_row_tiles = TC_Br / WMMA_M;                 // 4
    const int o_col_tiles = head_dim / WMMA_N;               // head_dim/16
    const int o_total_tiles = o_row_tiles * o_col_tiles;
    const int pv_chunks   = TC_Bc / WMMA_K;                 // 4

    // ================================================================
    // Main loop over KV tiles
    // ================================================================
    for (int j = 0; j < num_kv_tiles; j++) {
        const int kv_start = j * TC_Bc;

        // ---- Load K tile into KV_tile ----
        // KV_tile layout: [Bc, head_dim] row-major.
        {
            const int total = TC_Bc * head_dim;
            for (int i = tid; i < total; i += TC_BLOCK_THREADS) {
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

        // ============================================================
        // Phase 1: S = Q_tile @ KV_tile^T  [Br, Bc] using WMMA
        // ============================================================
        //
        // S[Br, Bc] is decomposed into 4x4 = 16 tiles of 16x16.
        // 4 warps each handle 4 tiles (round-robin by tile index).
        //
        // For each 16x16 output tile S[ri, ci]:
        //   acc = 0
        //   for k in 0..hd_chunks-1:
        //     a_frag = Q_tile[ri*16..(ri+1)*16, k*16..(k+1)*16]  (row_major)
        //     b_frag = KV_tile[ci*16..(ci+1)*16, k*16..(k+1)*16] (col_major => transpose)
        //     acc += a_frag @ b_frag
        //   S_tile[ri*16..ci*16..] = acc
        //
        // Q_tile row-major [Br, head_dim], ldm = head_dim -> matrix_a row_major
        // KV_tile row-major [Bc, head_dim], loaded as col_major with ldm = head_dim
        //   This effectively reads K^T because col_major element(i,j) = mem[j*ldm+i],
        //   so element(i,j) of the 16x16 fragment = KV_tile[ci*16+j, k*16+i] = K[ci*16+j, k*16+i],
        //   which is K^T[k*16+i, ci*16+j]. The mma computes A @ B where B is this
        //   transposed view, yielding Q @ K^T.
        // ============================================================

        for (int tile_idx = warp_id; tile_idx < s_total_tiles; tile_idx += TC_NUM_WARPS) {
            int ri = tile_idx / s_col_tiles;   // row-tile index [0,4)
            int ci = tile_idx % s_col_tiles;   // col-tile index [0,4)

            // Accumulator fragment for this 16x16 output tile
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
            wmma::fill_fragment(acc, 0.0f);

            // Iterate over head_dim in chunks of 16
            for (int k = 0; k < hd_chunks; k++) {
                // Q fragment: rows [ri*16, ri*16+16), cols [k*16, k*16+16)
                // Pointer: Q_tile + ri*16*head_dim + k*16, ldm = head_dim
                wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                               half, wmma::row_major> a_frag;
                wmma::load_matrix_sync(a_frag,
                    Q_tile + ri * WMMA_M * head_dim + k * WMMA_K,
                    head_dim);

                // K^T fragment via col_major load of KV_tile
                // Pointer: KV_tile + ci*16*head_dim + k*16, ldm = head_dim
                // col_major: element(i,j) = mem[j*head_dim + i]
                //          = KV_tile[(ci*16+j)*head_dim + (k*16+i)]
                //          = K[ci*16+j, k*16+i] = K^T[k*16+i, ci*16+j]
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                               half, wmma::col_major> b_frag;
                wmma::load_matrix_sync(b_frag,
                    KV_tile + ci * WMMA_N * head_dim + k * WMMA_K,
                    head_dim);

                wmma::mma_sync(acc, a_frag, b_frag, acc);
            }

            // Store accumulated S tile to shared memory
            // S_tile is [Br, Bc] row-major, ldm = Bc
            wmma::store_matrix_sync(
                S_tile + ri * WMMA_M * TC_Bc + ci * WMMA_N,
                acc, TC_Bc, wmma::mem_row_major);
        }
        __syncthreads();

        // ---- Apply scale and causal mask to S_tile ----
        {
            const int total = TC_Br * TC_Bc;
            for (int i = tid; i < total; i += TC_BLOCK_THREADS) {
                int r = i / TC_Bc;
                int c = i % TC_Bc;
                int gq = q_start  + r;
                int gk = kv_start + c;

                if (gq < seq_q && gk < seq_kv) {
                    float val = S_tile[i] * scale;
                    if (causal && gq < gk) val = -FLT_MAX;
                    S_tile[i] = val;
                } else {
                    S_tile[i] = -FLT_MAX;
                }
            }
        }
        __syncthreads();

        // ============================================================
        // Phase 2: Online softmax + rescale O accumulator
        // ============================================================
        // Only threads with tid < TC_Br and valid query row participate.
        // After this block:
        //   - S_tile[row*Bc+c] holds P(r,c) = exp(S - m_new)
        //   - m_i, l_i updated to new running state
        //   - O_acc rescaled by (alpha / l_new)
        //   - scale_pv_shared[row] = 1/l_new for use in P@V
        // ============================================================

        if (tid < TC_Br && (q_start + tid) < seq_q) {
            const int r = tid;

            // Row max of S tile
            float m_ij = -FLT_MAX;
            for (int c = 0; c < TC_Bc; c++) {
                m_ij = fmaxf(m_ij, S_tile[r * TC_Bc + c]);
            }

            float m_new = fmaxf(m_i, m_ij);

            // Compute P = exp(S - m_new) and its row sum
            float p_sum = 0.0f;
            for (int c = 0; c < TC_Bc; c++) {
                float p = expf(S_tile[r * TC_Bc + c] - m_new);
                S_tile[r * TC_Bc + c] = p;
                p_sum += p;
            }

            // Rescale previous accumulator
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
        // Phase 3: Convert P to half with scale_pv baked in (for WMMA P@V)
        // ============================================================
        // P_half[r, c] = half(S_tile[r, c] * scale_pv_shared[r])
        // This combines the softmax probability with the 1/l_new normalisation,
        // so the subsequent P@V accumulation directly produces the correctly
        // normalised output contribution.
        // ============================================================
        {
            const int total = TC_Br * TC_Bc;
            for (int i = tid; i < total; i += TC_BLOCK_THREADS) {
                int r = i / TC_Bc;
                float spv = scale_pv_shared[r];
                P_half[i] = __float2half(S_tile[i] * spv);
            }
        }
        __syncthreads();

        // ---- Load V tile into KV_tile ----
        {
            const int total = TC_Bc * head_dim;
            for (int i = tid; i < total; i += TC_BLOCK_THREADS) {
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

        // ============================================================
        // Phase 4: O += P @ V  [Br, head_dim] using WMMA
        // ============================================================
        // P_half: [Br, Bc] row-major half  -> matrix_a row_major, ldm = Bc
        // V (KV_tile): [Bc, head_dim] row-major half -> matrix_b row_major, ldm = head_dim
        //
        // O[Br, head_dim] is decomposed into (Br/16) x (head_dim/16) tiles.
        // Each 16x16 output tile accumulates over Bc/16 WMMA ops:
        //   O[ri, di] += sum_{k=0}^{Bc/16-1} P[ri, k] @ V[k, di]
        // ============================================================

        for (int tile_idx = warp_id; tile_idx < o_total_tiles; tile_idx += TC_NUM_WARPS) {
            int ri = tile_idx / o_col_tiles;   // row-tile [0, Br/16)
            int di = tile_idx % o_col_tiles;   // col-tile [0, head_dim/16)

            // Load current O_acc 16x16 tile into accumulator fragment
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> o_frag;
            wmma::load_matrix_sync(o_frag,
                O_acc + ri * WMMA_M * head_dim + di * WMMA_N,
                head_dim, wmma::mem_row_major);

            // Accumulate P[ri_block, k_block] @ V[k_block, di_block]
            for (int k = 0; k < pv_chunks; k++) {
                // P fragment: P_half[ri*16..(ri+1)*16, k*16..(k+1)*16], row-major
                wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                               half, wmma::row_major> p_frag;
                wmma::load_matrix_sync(p_frag,
                    P_half + ri * WMMA_M * TC_Bc + k * WMMA_K,
                    TC_Bc);

                // V fragment: KV_tile[k*16..(k+1)*16, di*16..(di+1)*16], row-major
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                               half, wmma::row_major> v_frag;
                wmma::load_matrix_sync(v_frag,
                    KV_tile + k * WMMA_N * head_dim + di * WMMA_N,
                    head_dim);

                wmma::mma_sync(o_frag, p_frag, v_frag, o_frag);
            }

            // Store updated O fragment back to shared memory
            wmma::store_matrix_sync(
                O_acc + ri * WMMA_M * head_dim + di * WMMA_N,
                o_frag, head_dim, wmma::mem_row_major);
        }
        __syncthreads();
    }

    // ---- Write final output to global memory ----
    {
        const int total = TC_Br * head_dim;
        for (int i = tid; i < total; i += TC_BLOCK_THREADS) {
            int r = i / head_dim;
            if (q_start + r < seq_q) {
                O_ptr[(int64_t)r * q_row_stride + (i % head_dim)] =
                    __float2half(O_acc[i]);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Host launcher
// ---------------------------------------------------------------------------
void flash_attention_prefill_tc(
    const Tensor& Q, const Tensor& K, const Tensor& V, Tensor& O,
    float scale, bool causal, cudaStream_t stream)
{
    const int batch_size = static_cast<int>(Q.shape[0]);
    const int seq_q      = static_cast<int>(Q.shape[1]);
    const int n_heads    = static_cast<int>(Q.shape[2]);
    const int head_dim   = static_cast<int>(Q.shape[3]);
    const int seq_kv     = static_cast<int>(K.shape[1]);
    const int n_kv_heads = static_cast<int>(K.shape[2]);

    const int num_q_tiles = (seq_q + TC_Br - 1) / TC_Br;
    dim3 grid(num_q_tiles, batch_size * n_heads);
    dim3 block(TC_WARP_SIZE, TC_NUM_WARPS);  // 32 x 4 = 128

    // Shared memory budget:
    //   Q_tile  : half  [Br * head_dim]
    //   KV_tile : half  [Bc * head_dim]
    //   S_tile  : float [Br * Bc]
    //   O_acc   : float [Br * head_dim]
    //   P_half  : half  [Br * Bc]           -- half P for WMMA P@V
    //   scale_pv_shared : float [Br]        -- per-row 1/l_new
    size_t smem_bytes = TC_Br * head_dim * sizeof(half)      // Q_tile
                      + TC_Bc * head_dim * sizeof(half)      // KV_tile
                      + TC_Br * TC_Bc    * sizeof(float)     // S_tile
                      + TC_Br * head_dim * sizeof(float)     // O_acc
                      + TC_Br * TC_Bc    * sizeof(half)      // P_half
                      + TC_Br            * sizeof(float);    // scale_pv_shared

    cudaFuncSetAttribute(flash_attention_prefill_tc_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         static_cast<int>(smem_bytes));

    flash_attention_prefill_tc_kernel<<<grid, block, smem_bytes, stream>>>(
        reinterpret_cast<const half*>(Q.data),
        reinterpret_cast<const half*>(K.data),
        reinterpret_cast<const half*>(V.data),
        reinterpret_cast<half*>(O.data),
        batch_size, seq_q, seq_kv, n_heads, n_kv_heads, head_dim,
        scale, causal);
}

// ---------------------------------------------------------------------------
// Check if tensor-core attention is available on the current device
// ---------------------------------------------------------------------------
bool tc_attention_available()
{
    int device = -1;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess || device < 0) return false;

    int major = 0, minor = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);

    // Require sm_90+ (Hopper / Blackwell)
    return (major > 9) || (major == 9 && minor >= 0);
}

} // namespace imp
