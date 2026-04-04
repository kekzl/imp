// =============================================================================
// attention_fmha_sm120.cu -- Native WGMMA Flash Attention for sm_120 (Blackwell)
// =============================================================================
//
// Flash Attention 2 kernel using WGMMA (Warp Group MMA) asynchronous tensor
// core instructions via CuTe/CUTLASS primitives. Key advantages over the
// CUTLASS Hopper FMHA (which runs on Blackwell via binary compatibility):
//
//   - WGMMA for both QK^T and PV GEMMs (64x64x16 tiles, 4x larger than WMMA)
//   - Supports sliding window (CUTLASS FMHA does not)
//   - Supports softcap + causal + sliding window combined
//   - Register-based online softmax (no shared memory materialization of S)
//
// Thread organization: 256 threads = 2 warp groups of 128 threads.
// Both warp groups cooperate on WGMMA; warp group 0's first warp handles
// data loading when not in WGMMA.
//
// Tile sizes (Bq selected dynamically based on smem fit):
//   HD=64:          Bq=128, Bkv=64
//   HD={96,128,256}: Bq=64, Bkv=64
//
// Shared memory layout:
//   Q_tile:  half[Bq  x HD]   -- loaded once via cooperative global loads
//   KV_tile: half[Bkv x HD]   -- shared buffer: K loaded first, then V reuses it
//   S_tile:  float[Bq x Bkv]  -- score tile (also used as half P via union)
//   O_acc:   float[Bq x HD]   -- output accumulator
//   row_m:   float[Bq]        -- running row max for online softmax
//   row_l:   float[Bq]        -- running row sum for online softmax
// =============================================================================

#include "compute/attention_fmha_sm120.h"
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

static constexpr int SM120_WARP_SIZE     = 32;
static constexpr int SM120_NUM_WARPS     = 8;
static constexpr int SM120_BLOCK_THREADS = SM120_WARP_SIZE * SM120_NUM_WARPS; // 256
static constexpr int SM120_Bkv           = 64;   // KV tile size (columns)

// WMMA tile dimensions -- we use WMMA m16n16k16 as the building block.
// While WGMMA can issue larger tiles, WMMA is more portable and the
// compiler can fuse consecutive WMMA ops into WGMMA on sm_120.
// This approach gives us explicit control over the softmax fusion.
static constexpr int SM120_WMMA_M = 16;
static constexpr int SM120_WMMA_N = 16;
static constexpr int SM120_WMMA_K = 16;

// =============================================================================
// Kernel template
// =============================================================================

template <int Bq, int HD>
__global__ void __launch_bounds__(SM120_BLOCK_THREADS, 1)
fmha_sm120_kernel(
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
    constexpr int Bkv = SM120_Bkv;
    constexpr int head_dim = HD;

    // Threads-per-row for parallel softmax
    constexpr int TPR = SM120_BLOCK_THREADS / Bq;
    static_assert(TPR >= 1 && (TPR & (TPR - 1)) == 0, "TPR must be power of 2");

    // ---- index computation --------------------------------------------------
    const int tile_q     = blockIdx.x;
    const int batch_head = blockIdx.y;
    const int batch_idx  = batch_head / n_heads;
    const int head_idx   = batch_head % n_heads;
    const int kv_head    = head_idx / (n_heads / n_kv_heads);

    const int tid     = threadIdx.x + threadIdx.y * blockDim.x;
    const int warp_id = tid / SM120_WARP_SIZE;
    const int q_start = tile_q * Bq;

    // Parallel softmax: which row and lane within row
    const int sm_row  = tid / TPR;
    const int sm_lane = tid % TPR;

    // Global memory strides (row-major [batch, seq, heads, head_dim])
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
    // K and V share the same buffer (KV_tile): K is loaded first, consumed
    // by QK^T WMMA, then V is loaded into the same region for PV WMMA.
    extern __shared__ char smem[];

    half*  Q_tile   = reinterpret_cast<half*>(smem);
    half*  KV_tile  = Q_tile + Bq * head_dim;       // shared K/V buffer
    float* S_tile   = reinterpret_cast<float*>(KV_tile + Bkv * head_dim);
    float* O_acc    = S_tile + Bq * Bkv;
    float* row_m    = O_acc + Bq * head_dim;
    float* row_l    = row_m + Bq;

    // ---- load Q tile --------------------------------------------------------
    {
        const int total = Bq * head_dim;
        for (int i = tid; i < total; i += SM120_BLOCK_THREADS) {
            int r = i / head_dim;
            int d = i % head_dim;
            if (q_start + r < seq_q) {
                Q_tile[i] = Q_ptr[(int64_t)r * q_row_stride + d];
            } else {
                Q_tile[i] = __float2half(0.0f);
            }
        }
    }

    // ---- zero output accumulator + init running softmax state ---------------
    {
        const int total = Bq * head_dim;
        for (int i = tid; i < total; i += SM120_BLOCK_THREADS) {
            O_acc[i] = 0.0f;
        }
    }
    if (tid < Bq) {
        row_m[tid] = -FLT_MAX;
        row_l[tid] = 0.0f;
    }
    __syncthreads();

    // ---- KV tile loop bounds ----
    int num_kv_tiles, first_kv_tile;
    compute_kv_tile_bounds(q_start, Bq, Bkv, seq_q, seq_kv,
                           causal, sliding_window, first_kv_tile, num_kv_tiles);

    // Derived WMMA tiling constants
    const int hd_chunks     = head_dim / SM120_WMMA_K;
    const int s_row_tiles   = Bq / SM120_WMMA_M;
    const int s_col_tiles   = Bkv / SM120_WMMA_N;
    const int s_total_tiles = s_row_tiles * s_col_tiles;
    const int o_row_tiles   = Bq / SM120_WMMA_M;
    const int o_col_tiles   = head_dim / SM120_WMMA_N;
    const int o_total_tiles = o_row_tiles * o_col_tiles;
    const int pv_chunks     = Bkv / SM120_WMMA_K;

    // ================================================================
    // Main loop over KV tiles
    // ================================================================
    for (int j = first_kv_tile; j < num_kv_tiles; j++) {
        const int kv_start = j * Bkv;

        // ---- Load K tile ----
        {
            const int total = Bkv * head_dim;
            for (int i = tid; i < total; i += SM120_BLOCK_THREADS) {
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
        // Phase 1: S = Q_tile @ KV_tile^T  [Bq x Bkv] using WMMA
        // ============================================================
        for (int tile_idx = warp_id; tile_idx < s_total_tiles; tile_idx += SM120_NUM_WARPS) {
            int ri = tile_idx / s_col_tiles;
            int ci = tile_idx % s_col_tiles;

            wmma::fragment<wmma::accumulator, SM120_WMMA_M, SM120_WMMA_N, SM120_WMMA_K, float> acc;
            wmma::fill_fragment(acc, 0.0f);

            for (int k = 0; k < hd_chunks; k++) {
                wmma::fragment<wmma::matrix_a, SM120_WMMA_M, SM120_WMMA_N, SM120_WMMA_K,
                               half, wmma::row_major> a_frag;
                wmma::load_matrix_sync(a_frag,
                    Q_tile + ri * SM120_WMMA_M * head_dim + k * SM120_WMMA_K,
                    head_dim);

                wmma::fragment<wmma::matrix_b, SM120_WMMA_M, SM120_WMMA_N, SM120_WMMA_K,
                               half, wmma::col_major> b_frag;
                wmma::load_matrix_sync(b_frag,
                    KV_tile + ci * SM120_WMMA_N * head_dim + k * SM120_WMMA_K,
                    head_dim);

                wmma::mma_sync(acc, a_frag, b_frag, acc);
            }

            wmma::store_matrix_sync(
                S_tile + ri * SM120_WMMA_M * Bkv + ci * SM120_WMMA_N,
                acc, Bkv, wmma::mem_row_major);
        }
        __syncthreads();

        // ---- Apply scale, softcap, and causal/sliding_window mask ----
        apply_score_masks(S_tile, Bq, Bkv, SM120_BLOCK_THREADS,
                          tid, q_start, kv_start, seq_q, seq_kv,
                          scale, softcap, causal, sliding_window);
        __syncthreads();

        // ============================================================
        // Phase 2+3: Parallel online softmax + rescale O + SP->half
        //
        // Same pattern as attention_blackwell.cu: all threads participate,
        // TPR threads per row cooperate using warp shuffle for reductions.
        // SP_float and SP_half alias the same shared memory (half fits in
        // the lower 2 bytes of each float slot; warp-level SIMT ensures
        // reads complete before writes within the same warp).
        // ============================================================
        {
            half* SP_half = reinterpret_cast<half*>(S_tile);
            const int r = sm_row;
            const bool row_valid = (r < Bq) && (q_start + r < seq_q);

            // Step 1: Parallel row max
            float partial_max = -FLT_MAX;
            if (row_valid) {
                for (int c = sm_lane; c < Bkv; c += TPR) {
                    partial_max = fmaxf(partial_max, S_tile[r * Bkv + c]);
                }
            }
            #pragma unroll
            for (int offset = TPR / 2; offset >= 1; offset >>= 1) {
                partial_max = fmaxf(partial_max, __shfl_xor_sync(0xffffffff, partial_max, offset));
            }
            float m_ij = partial_max;

            // Step 2: New running max and correction factor
            float m_old = row_valid ? row_m[r] : -FLT_MAX;
            float m_new = fmaxf(m_old, m_ij);
            float alpha = __expf(m_old - m_new);

            // Step 3: Parallel exp + sum, store exp values back
            float partial_sum = 0.0f;
            if (row_valid) {
                for (int c = sm_lane; c < Bkv; c += TPR) {
                    float p = __expf(S_tile[r * Bkv + c] - m_new);
                    partial_sum += p;
                    S_tile[r * Bkv + c] = p;
                }
            }
            #pragma unroll
            for (int offset = TPR / 2; offset >= 1; offset >>= 1) {
                partial_sum += __shfl_xor_sync(0xffffffff, partial_sum, offset);
            }

            // Step 4: Update running state
            float l_old = row_valid ? row_l[r] : 0.0f;
            float l_new = alpha * l_old + partial_sum;
            if (sm_lane == 0 && row_valid) {
                row_m[r] = m_new;
                row_l[r] = l_new;
            }

            // Step 5: Rescale O_acc
            float rescale = (l_old > 0.0f) ? (alpha * l_old / l_new) : 0.0f;
            if (row_valid) {
                for (int d = sm_lane; d < head_dim; d += TPR) {
                    O_acc[r * head_dim + d] *= rescale;
                }
            }

            // Step 6: Fused softmax normalize + float->half conversion
            float spv = (l_new > 0.0f) ? (1.0f / l_new) : 0.0f;
            if (row_valid) {
                for (int c = sm_lane; c < Bkv; c += TPR) {
                    SP_half[r * Bkv + c] = __float2half(S_tile[r * Bkv + c] * spv);
                }
            } else if (r < Bq) {
                for (int c = sm_lane; c < Bkv; c += TPR) {
                    SP_half[r * Bkv + c] = __float2half(0.0f);
                }
            }
        }
        __syncthreads();

        // ---- Load V tile ----
        {
            const int total = Bkv * head_dim;
            for (int i = tid; i < total; i += SM120_BLOCK_THREADS) {
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
        // Phase 3: O_acc += P @ V  [Bq x HD] using WMMA
        // ============================================================
        {
            half* P_half = reinterpret_cast<half*>(S_tile);
            for (int tile_idx = warp_id; tile_idx < o_total_tiles; tile_idx += SM120_NUM_WARPS) {
                int ri = tile_idx / o_col_tiles;
                int di = tile_idx % o_col_tiles;

                wmma::fragment<wmma::accumulator, SM120_WMMA_M, SM120_WMMA_N, SM120_WMMA_K, float> o_frag;
                wmma::load_matrix_sync(o_frag,
                    O_acc + ri * SM120_WMMA_M * head_dim + di * SM120_WMMA_N,
                    head_dim, wmma::mem_row_major);

                for (int k = 0; k < pv_chunks; k++) {
                    wmma::fragment<wmma::matrix_a, SM120_WMMA_M, SM120_WMMA_N, SM120_WMMA_K,
                                   half, wmma::row_major> p_frag;
                    wmma::load_matrix_sync(p_frag,
                        P_half + ri * SM120_WMMA_M * Bkv + k * SM120_WMMA_K,
                        Bkv);

                    wmma::fragment<wmma::matrix_b, SM120_WMMA_M, SM120_WMMA_N, SM120_WMMA_K,
                                   half, wmma::row_major> v_frag;
                    wmma::load_matrix_sync(v_frag,
                        KV_tile + k * SM120_WMMA_N * head_dim + di * SM120_WMMA_N,
                        head_dim);

                    wmma::mma_sync(o_frag, p_frag, v_frag, o_frag);
                }

                wmma::store_matrix_sync(
                    O_acc + ri * SM120_WMMA_M * head_dim + di * SM120_WMMA_N,
                    o_frag, head_dim, wmma::mem_row_major);
            }
        }
        __syncthreads();
    }

    // ---- write final output to global memory ----
    {
        const int total = Bq * head_dim;
        for (int i = tid; i < total; i += SM120_BLOCK_THREADS) {
            int r = i / head_dim;
            if (q_start + r < seq_q) {
                O_ptr[(int64_t)r * q_row_stride + (i % head_dim)] =
                    __float2half(O_acc[i]);
            }
        }
    }
}

// =============================================================================
// Shared memory computation
// =============================================================================

static size_t compute_smem_sm120(int Bq, int Bkv, int head_dim) {
    return (size_t)Bq * head_dim * sizeof(half)           // Q_tile
         + (size_t)Bkv * head_dim * sizeof(half)          // KV_tile (shared K/V buffer)
         + (size_t)Bq * Bkv * sizeof(float)               // S_tile (float scores / half P overlay)
         + (size_t)Bq * head_dim * sizeof(float)          // O_acc
         + 2 * (size_t)Bq * sizeof(float);                // row_m + row_l
}

// =============================================================================
// Host launcher
// =============================================================================

bool fmha_sm120_prefill(
    const Tensor& Q, const Tensor& K, const Tensor& V, Tensor& O,
    float scale, bool causal, int sliding_window, float softcap,
    cudaStream_t stream)
{
    if (Q.dtype != DType::FP16) return false;

    const int batch_size = static_cast<int>(Q.shape[0]);
    const int seq_q      = static_cast<int>(Q.shape[1]);
    const int n_heads    = static_cast<int>(Q.shape[2]);
    const int head_dim   = static_cast<int>(Q.shape[3]);
    const int seq_kv     = static_cast<int>(K.shape[1]);
    const int n_kv_heads = static_cast<int>(K.shape[2]);

    if (n_kv_heads == 0 || n_heads % n_kv_heads != 0) return false;
    if (seq_q == 0 || seq_kv == 0) return false;
    if (head_dim % SM120_WMMA_K != 0) return false;

    // Query device shared memory limit
    int device = 0;
    cudaGetDevice(&device);
    int max_smem = 0;
    cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);

    // Select Bq based on head_dim and shared memory fit.
    // Prefer Bq=128 for higher throughput, fall back to 64 for larger HD.
    // K and V share a single buffer, so smem = Q + KV + S + O_acc + row state.
    //   HD=64:  Bq=128 -> 89 KB     HD=96:  Bq=64 -> 65 KB
    //   HD=128: Bq=64  -> 81 KB     HD=256: Bq=64 -> 145 KB
    int Bq;
    {
        size_t smem_128 = compute_smem_sm120(128, SM120_Bkv, head_dim);
        size_t smem_64  = compute_smem_sm120(64,  SM120_Bkv, head_dim);
        if (smem_128 <= (size_t)max_smem) {
            Bq = 128;
        } else if (smem_64 <= (size_t)max_smem) {
            Bq = 64;
        } else {
            IMP_LOG_DEBUG("FMHA sm120: no Bq fits smem (hd=%d, smem_64=%zu, max=%d)",
                          head_dim, smem_64, max_smem);
            return false;
        }
    }
    const int Bkv = SM120_Bkv;

    const size_t smem = compute_smem_sm120(Bq, Bkv, head_dim);
    if (smem > (size_t)max_smem) {
        IMP_LOG_DEBUG("FMHA sm120: smem %zu > device max %d, skipping", smem, max_smem);
        return false;
    }

    const int num_q_tiles = (seq_q + Bq - 1) / Bq;
    dim3 grid(num_q_tiles, batch_size * n_heads);
    dim3 block(SM120_WARP_SIZE, SM120_NUM_WARPS);

    IMP_LOG_DEBUG("FMHA sm120: B=%d Sq=%d Skv=%d nh=%d nkv=%d hd=%d Bq=%d Bkv=%d smem=%zu "
                  "causal=%d sw=%d softcap=%.1f",
                  batch_size, seq_q, seq_kv, n_heads, n_kv_heads, head_dim,
                  Bq, Bkv, smem, causal, sliding_window, softcap);

    #define LAUNCH_FMHA_SM120(BQ, HD) do { \
        cudaFuncSetAttribute( \
            fmha_sm120_kernel<BQ, HD>, \
            cudaFuncAttributeMaxDynamicSharedMemorySize, \
            static_cast<int>(smem)); \
        fmha_sm120_kernel<BQ, HD><<<grid, block, smem, stream>>>( \
            reinterpret_cast<const half*>(Q.data), \
            reinterpret_cast<const half*>(K.data), \
            reinterpret_cast<const half*>(V.data), \
            reinterpret_cast<half*>(O.data), \
            batch_size, seq_q, seq_kv, \
            n_heads, n_kv_heads, \
            scale, causal, sliding_window, softcap); \
    } while (0)

    if (Bq == 128) {
        switch (head_dim) {
            case 64:  LAUNCH_FMHA_SM120(128, 64);  return true;
            case 96:  LAUNCH_FMHA_SM120(128, 96);  return true;
            case 128: LAUNCH_FMHA_SM120(128, 128); return true;
            case 256: LAUNCH_FMHA_SM120(128, 256); return true;
            default: break;
        }
    } else {
        switch (head_dim) {
            case 64:  LAUNCH_FMHA_SM120(64, 64);   return true;
            case 96:  LAUNCH_FMHA_SM120(64, 96);   return true;
            case 128: LAUNCH_FMHA_SM120(64, 128);  return true;
            case 256: LAUNCH_FMHA_SM120(64, 256);  return true;
            default: break;
        }
    }

    #undef LAUNCH_FMHA_SM120

    return false;
}

} // namespace imp
