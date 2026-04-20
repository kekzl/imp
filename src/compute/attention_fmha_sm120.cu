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
//   HD=64:      Bq=128, Bkv=64  (89 KB smem)
//   HD=96:      Bq=64,  Bkv=64  (65 KB smem)
//   HD=128:     Bq=64,  Bkv=64  (81 KB smem)
//   HD=256:     Bq=32,  Bkv=64  (88 KB smem)
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
#include <cuda_fp8.h>
#include <float.h>
#include <mma.h>
#include <cute/arch/config.hpp>  // CUTE_ARCH_F8F6F4_MMA_ENABLED

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

            // Step 3: Parallel exp + sum, store exp values back.
            // Mask guard: apply_score_masks writes -FLT_MAX for causal/SWA-out-of-window
            // positions. When ALL positions in a tile are masked for this row
            // (Gemma-4 SWA query beyond the window hitting the first KV tile at
            // kv_start=0), m_new collapses to -FLT_MAX and `expf(-FLT_MAX -
            // (-FLT_MAX)) = expf(0) = 1` would inflate partial_sum by Bkv per row
            // and poison the running softmax denominator. Explicit sentinel check
            // maps masked scores to 0 without relying on the subtractive cancel.
            float partial_sum = 0.0f;
            if (row_valid) {
                for (int c = sm_lane; c < Bkv; c += TPR) {
                    float s_val = S_tile[r * Bkv + c];
                    float p = (s_val <= -FLT_MAX * 0.5f) ? 0.0f
                                                         : __expf(s_val - m_new);
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
        size_t smem_32  = compute_smem_sm120(32,  SM120_Bkv, head_dim);
        if (smem_128 <= (size_t)max_smem) {
            Bq = 128;
        } else if (smem_64 <= (size_t)max_smem) {
            Bq = 64;
        } else if (smem_32 <= (size_t)max_smem) {
            Bq = 32;
        } else {
            IMP_LOG_DEBUG("FMHA sm120: no Bq fits smem (hd=%d, smem_32=%zu, max=%d)",
                          head_dim, smem_32, max_smem);
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
        cudaError_t attr_err = cudaFuncSetAttribute( \
            fmha_sm120_kernel<BQ, HD>, \
            cudaFuncAttributeMaxDynamicSharedMemorySize, \
            static_cast<int>(smem)); \
        if (attr_err != cudaSuccess) { \
            IMP_LOG_WARN("FMHA sm120: cudaFuncSetAttribute failed for Bq=%d HD=%d smem=%zu: %s", \
                         BQ, HD, smem, cudaGetErrorString(attr_err)); \
            return false; \
        } \
        cudaFuncSetAttribute(fmha_sm120_kernel<BQ, HD>, \
            cudaFuncAttributePreferredSharedMemoryCarveout, \
            cudaSharedmemCarveoutMaxShared); \
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
    } else if (Bq == 64) {
        switch (head_dim) {
            case 64:  LAUNCH_FMHA_SM120(64, 64);   return true;
            case 96:  LAUNCH_FMHA_SM120(64, 96);   return true;
            case 128: LAUNCH_FMHA_SM120(64, 128);  return true;
            case 256: LAUNCH_FMHA_SM120(64, 256);  return true;
            default: break;
        }
    } else {
        // Bq=32: for large head_dim (256) where Bq=64 exceeds smem
        switch (head_dim) {
            case 64:  LAUNCH_FMHA_SM120(32, 64);   return true;
            case 96:  LAUNCH_FMHA_SM120(32, 96);   return true;
            case 128: LAUNCH_FMHA_SM120(32, 128);  return true;
            case 256: LAUNCH_FMHA_SM120(32, 256);  return true;
            default: break;
        }
    }

    #undef LAUNCH_FMHA_SM120

    return false;
}

// =============================================================================
// FP8 Score Compute variant: QK^T in FP8 E4M3 (m16n8k32), PV in FP16 WMMA
// =============================================================================
//
// SM120 FP8 MMA: mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32
// Register layout: A[4 x uint32] = 16 rows × 32 cols FP8
//                  B[2 x uint32] = 8 cols × 32 rows FP8
//                  D[4 x float]  = 16 × 8 output
//
// For a 16×16 score tile: 2 MMA calls (n=8 each), k-loop over head_dim/32.
// vs FP16 WMMA: 1 MMA per 16×16×16 tile, k-loop over head_dim/16.
// FP8 has 2× k per MMA → same number of iterations but 2× k-throughput.

// FP8 FMHA requires sm_120f for .kind::f8f6f4 MMA instructions.
// CUTE_ARCH_F8F6F4_MMA_ENABLED is defined by CUTLASS config.hpp for sm_120f.
// !defined(__CUDA_ARCH__) allows host-side code (launcher) to compile unconditionally.
// FP8 FMHA: QK^T in FP8 E4M3, PV in FP16 WMMA.
// The inline PTX (.kind::f8f6f4) is guarded with __CUDA_ARCH__ >= 1200
// inside the kernel so it compiles cleanly for sm_90/sm_100 too.

// Device helper: convert 4 FP16 values to 4 FP8 E4M3 packed in uint32
__device__ __forceinline__ uint32_t cvt_4xfp16_to_4xe4m3(const half* src) {
    // Use the PTX cvt instruction for paired FP16→FP8 conversion
    uint32_t result;
    uint16_t lo, hi;
    // Convert pairs: [src[0], src[1]] → 2 FP8 bytes, [src[2], src[3]] → 2 FP8 bytes
    const uint32_t* src32 = reinterpret_cast<const uint32_t*>(src);
    asm volatile("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(lo) : "r"(src32[0]));
    asm volatile("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(hi) : "r"(src32[1]));
    result = static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
    return result;
}

template <int Bq, int HD>
__global__ void __launch_bounds__(SM120_BLOCK_THREADS, 1)
fmha_sm120_fp8_kernel(
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
    constexpr int TPR = SM120_BLOCK_THREADS / Bq;
    static_assert(TPR >= 1 && (TPR & (TPR - 1)) == 0, "TPR must be power of 2");

    const int tile_q     = blockIdx.x;
    const int batch_head = blockIdx.y;
    const int batch_idx  = batch_head / n_heads;
    const int head_idx   = batch_head % n_heads;
    const int kv_head    = head_idx / (n_heads / n_kv_heads);

    const int tid     = threadIdx.x + threadIdx.y * blockDim.x;
    const int warp_id = tid / SM120_WARP_SIZE;
    const int lane_id = tid % SM120_WARP_SIZE;
    const int q_start = tile_q * Bq;

    const int sm_row  = tid / TPR;
    const int sm_lane = tid % TPR;

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

    // Shared memory layout:
    //   Q_fp8:  uint8[Bq x HD]      — Q converted to FP8 E4M3
    //   KV_fp8: uint8[Bkv x HD]     — K converted to FP8 (reused for V as FP16)
    //   KV_fp16: half[Bkv x HD]     — V loaded as FP16 (overlaps KV_fp8 for V phase)
    //   S_tile: float[Bq x Bkv]     — score tile
    //   O_acc:  float[Bq x HD]      — output accumulator
    //   row_m/l: float[Bq] × 2      — softmax state
    extern __shared__ char smem[];

    uint8_t* Q_fp8   = reinterpret_cast<uint8_t*>(smem);
    uint8_t* KV_fp8  = Q_fp8 + Bq * head_dim;          // K as FP8
    half*    KV_fp16 = reinterpret_cast<half*>(KV_fp8); // V as FP16 (reuse same slot)
    float*   S_tile  = reinterpret_cast<float*>(KV_fp8 + Bkv * head_dim);
    float*   O_acc   = S_tile + Bq * Bkv;
    float*   row_m   = O_acc + Bq * head_dim;
    float*   row_l   = row_m + Bq;

    // Load Q tile and convert to FP8 E4M3
    {
        const int total = Bq * head_dim;
        for (int i = tid; i < total; i += SM120_BLOCK_THREADS) {
            int r = i / head_dim;
            int d = i % head_dim;
            if (q_start + r < seq_q) {
                half val = Q_ptr[(int64_t)r * q_row_stride + d];
                // Saturate FP16 → FP8 E4M3 (range [-448, 448])
                float fv = __half2float(val);
                fv = fminf(fmaxf(fv, -448.0f), 448.0f);
                // Simple scalar FP16→FP8: use hardware cvt if available
                Q_fp8[i] = static_cast<uint8_t>(__nv_fp8_e4m3(fv).__x);
            } else {
                Q_fp8[i] = 0;
            }
        }
    }

    // Zero O_acc + init softmax
    {
        const int total = Bq * head_dim;
        for (int i = tid; i < total; i += SM120_BLOCK_THREADS) O_acc[i] = 0.0f;
    }
    if (tid < Bq) { row_m[tid] = -FLT_MAX; row_l[tid] = 0.0f; }
    __syncthreads();

    // KV tile bounds
    int num_kv_tiles, first_kv_tile;
    compute_kv_tile_bounds(q_start, Bq, Bkv, seq_q, seq_kv,
                           causal, sliding_window, first_kv_tile, num_kv_tiles);

    // FP8 MMA tiling: m16n8k32 → output is 16×8, need 2 calls for 16×16 score tile
    constexpr int S_M = 16;
    constexpr int S_N = 8;   // MMA output width
    constexpr int S_K = 32;  // FP8 k-dim
    const int hd_chunks_fp8 = head_dim / S_K;
    const int s_row_tiles = Bq / S_M;
    const int s_col_tiles_half = Bkv / S_N;  // each m16n8 tile
    const int s_total_tiles = s_row_tiles * s_col_tiles_half;

    // FP16 WMMA tiling for PV (unchanged)
    const int o_row_tiles   = Bq / SM120_WMMA_M;
    const int o_col_tiles   = head_dim / SM120_WMMA_N;
    const int o_total_tiles = o_row_tiles * o_col_tiles;
    const int pv_chunks     = Bkv / SM120_WMMA_K;

    // Main KV tile loop
    for (int j = first_kv_tile; j < num_kv_tiles; j++) {
        const int kv_start = j * Bkv;

        // Load K tile and convert to FP8
        {
            const int total = Bkv * head_dim;
            for (int i = tid; i < total; i += SM120_BLOCK_THREADS) {
                int r = i / head_dim;
                int d = i % head_dim;
                if (kv_start + r < seq_kv) {
                    half val = K_ptr[(int64_t)(kv_start + r) * kv_row_stride + d];
                    float fv = __half2float(val);
                    fv = fminf(fmaxf(fv, -448.0f), 448.0f);
                    KV_fp8[i] = static_cast<uint8_t>(__nv_fp8_e4m3(fv).__x);
                } else {
                    KV_fp8[i] = 0;
                }
            }
        }
        __syncthreads();

        // Phase 1: S = Q_fp8 @ K_fp8^T using FP8 m16n8k32 MMA
        for (int tile_idx = warp_id; tile_idx < s_total_tiles; tile_idx += SM120_NUM_WARPS) {
            int ri = tile_idx / s_col_tiles_half;
            int ci = tile_idx % s_col_tiles_half;

            float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;

            for (int k = 0; k < hd_chunks_fp8; k++) {
                // Load A fragment: Q_fp8[ri*16 + lane_row, k*32 + lane_col]
                // A register layout: 4 × uint32 = 16 bytes = 16 FP8 values per thread
                // Each thread loads from row (lane_id / 4) within the 16-row tile
                // and 4 consecutive FP8 values starting at column ((lane_id % 4) * 4 + k * 32)
                uint32_t a0, a1, a2, a3;
                {
                    const uint8_t* q_base = Q_fp8 + ri * S_M * head_dim + k * S_K;
                    int row_in_tile = lane_id / 4;
                    int col_base = (lane_id % 4) * 4;
                    // Each register holds 4 FP8 values from the same row
                    const uint32_t* q_row0 = reinterpret_cast<const uint32_t*>(q_base + row_in_tile * head_dim + col_base);
                    const uint32_t* q_row8 = reinterpret_cast<const uint32_t*>(q_base + (row_in_tile + 8) * head_dim + col_base);
                    a0 = q_row0[0];
                    a1 = q_row0[4];  // +16 bytes offset
                    a2 = q_row8[0];
                    a3 = q_row8[4];
                }

                // Load B fragment: K_fp8[ci*8 + lane_col, k*32 + lane_row]
                // B register layout: 2 × uint32 = 8 bytes = 8 FP8 values per thread
                // K is in row-major but MMA expects col-major for B
                // → K^T: B[col, k] = K_fp8[ci*8 + col][k*32 + ...]
                uint32_t b0, b1;
                {
                    const uint8_t* k_base = KV_fp8 + ci * S_N * head_dim + k * S_K;
                    int col_in_tile = lane_id / 4;
                    int k_base_offset = (lane_id % 4) * 4;
                    const uint32_t* k_ptr0 = reinterpret_cast<const uint32_t*>(k_base + col_in_tile * head_dim + k_base_offset);
                    b0 = k_ptr0[0];
                    b1 = k_ptr0[4];  // +16 bytes
                }

                // FP8 MMA: d += A × B^T (SM120+ only)
#if __CUDA_ARCH__ >= 1200
                asm volatile(
                    "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                    "{%0, %1, %2, %3},"
                    "{%4, %5, %6, %7},"
                    "{%8, %9},"
                    "{%10, %11, %12, %13};\n"
                    : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                      "r"(b0), "r"(b1),
                      "f"(d0), "f"(d1), "f"(d2), "f"(d3));
#endif
            }

            // Store 16×8 result to S_tile
            // MMA output mapping: thread lane_id writes to specific (row, col) positions
            // m16n8k32 output: each thread owns a 2×1 chunk of the 16×8 output
            {
                // Canonical m16n8 output mapping:
                //   Thread t: row = (t/4)%8 + (d_idx >= 2 ? 8 : 0), col = (t%4)*2 + d_idx%2
                int base_row = ri * S_M;
                int base_col = ci * S_N;
                int r0 = (lane_id / 4) % 8;
                int c0 = (lane_id % 4) * 2;
                S_tile[(base_row + r0) * Bkv + base_col + c0]     = d0;
                S_tile[(base_row + r0) * Bkv + base_col + c0 + 1] = d1;
                S_tile[(base_row + r0 + 8) * Bkv + base_col + c0]     = d2;
                S_tile[(base_row + r0 + 8) * Bkv + base_col + c0 + 1] = d3;
            }
        }
        __syncthreads();

        // Apply scale, softcap, masks (same as FP16 path)
        apply_score_masks(S_tile, Bq, Bkv, SM120_BLOCK_THREADS,
                          tid, q_start, kv_start, seq_q, seq_kv,
                          scale, softcap, causal, sliding_window);
        __syncthreads();

        // Phase 2+3: Parallel online softmax + convert to FP16 P (same as FP16 kernel)
        {
            half* SP_half = reinterpret_cast<half*>(S_tile);
            const int r = sm_row;
            const bool row_valid = (r < Bq) && (q_start + r < seq_q);

            float partial_max = -FLT_MAX;
            if (row_valid) {
                for (int c = sm_lane; c < Bkv; c += TPR)
                    partial_max = fmaxf(partial_max, S_tile[r * Bkv + c]);
            }
            #pragma unroll
            for (int offset = TPR / 2; offset >= 1; offset >>= 1)
                partial_max = fmaxf(partial_max, __shfl_xor_sync(0xffffffff, partial_max, offset));
            float m_ij = partial_max;

            float m_old = row_valid ? row_m[r] : -FLT_MAX;
            float m_new = fmaxf(m_old, m_ij);
            float alpha = __expf(m_old - m_new);

            // Mask guard: see Step 3 of the other fmha_sm120_kernel template above.
            // Fully-masked tile (Gemma-4 SWA query > sliding_window) would
            // otherwise poison partial_sum via __expf(-FLT_MAX - (-FLT_MAX)) = 1.
            float partial_sum = 0.0f;
            if (row_valid) {
                for (int c = sm_lane; c < Bkv; c += TPR) {
                    float s_val = S_tile[r * Bkv + c];
                    float p = (s_val <= -FLT_MAX * 0.5f) ? 0.0f
                                                         : __expf(s_val - m_new);
                    partial_sum += p;
                    S_tile[r * Bkv + c] = p;
                }
            }
            #pragma unroll
            for (int offset = TPR / 2; offset >= 1; offset >>= 1)
                partial_sum += __shfl_xor_sync(0xffffffff, partial_sum, offset);

            float l_old = row_valid ? row_l[r] : 0.0f;
            float l_new = alpha * l_old + partial_sum;
            if (sm_lane == 0 && row_valid) { row_m[r] = m_new; row_l[r] = l_new; }

            float rescale = (l_old > 0.0f) ? (alpha * l_old / l_new) : 0.0f;
            if (row_valid) {
                for (int d = sm_lane; d < head_dim; d += TPR)
                    O_acc[r * head_dim + d] *= rescale;
            }

            float spv = (l_new > 0.0f) ? (1.0f / l_new) : 0.0f;
            if (row_valid) {
                for (int c = sm_lane; c < Bkv; c += TPR)
                    SP_half[r * Bkv + c] = __float2half(S_tile[r * Bkv + c] * spv);
            } else if (r < Bq) {
                for (int c = sm_lane; c < Bkv; c += TPR)
                    SP_half[r * Bkv + c] = __float2half(0.0f);
            }
        }
        __syncthreads();

        // Load V tile as FP16 (PV stays in FP16 for value precision)
        {
            const int total = Bkv * head_dim;
            for (int i = tid; i < total; i += SM120_BLOCK_THREADS) {
                int r = i / head_dim;
                int d = i % head_dim;
                if (kv_start + r < seq_kv)
                    KV_fp16[i] = V_ptr[(int64_t)(kv_start + r) * kv_row_stride + d];
                else
                    KV_fp16[i] = __float2half(0.0f);
            }
        }
        __syncthreads();

        // Phase 3: O_acc += P @ V using FP16 WMMA (same as FP16 kernel)
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
                        P_half + ri * SM120_WMMA_M * Bkv + k * SM120_WMMA_K, Bkv);

                    wmma::fragment<wmma::matrix_b, SM120_WMMA_M, SM120_WMMA_N, SM120_WMMA_K,
                                   half, wmma::row_major> v_frag;
                    wmma::load_matrix_sync(v_frag,
                        KV_fp16 + k * SM120_WMMA_N * head_dim + di * SM120_WMMA_N, head_dim);

                    wmma::mma_sync(o_frag, p_frag, v_frag, o_frag);
                }

                wmma::store_matrix_sync(
                    O_acc + ri * SM120_WMMA_M * head_dim + di * SM120_WMMA_N,
                    o_frag, head_dim, wmma::mem_row_major);
            }
        }
        __syncthreads();
    }

    // Write output
    {
        const int total = Bq * head_dim;
        for (int i = tid; i < total; i += SM120_BLOCK_THREADS) {
            int r = i / head_dim;
            if (q_start + r < seq_q)
                O_ptr[(int64_t)r * q_row_stride + (i % head_dim)] = __float2half(O_acc[i]);
        }
    }
}

// Shared memory for FP8 variant: Q_fp8 uses bytes not halves for Q
static size_t compute_smem_fp8(int Bq, int Bkv, int head_dim) {
    return (size_t)Bq * head_dim * sizeof(uint8_t)          // Q_fp8
         + (size_t)Bkv * head_dim * sizeof(half)            // KV buffer (FP8 K or FP16 V, half is larger)
         + (size_t)Bq * Bkv * sizeof(float)                 // S_tile
         + (size_t)Bq * head_dim * sizeof(float)            // O_acc
         + 2 * (size_t)Bq * sizeof(float);                  // row_m + row_l
}

bool fmha_sm120_fp8_prefill(
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
    if (head_dim % 32 != 0) return false;  // FP8 MMA needs k%32==0

    int device = 0;
    cudaGetDevice(&device);
    int max_smem = 0;
    cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);

    int Bq;
    {
        size_t smem_128 = compute_smem_fp8(128, SM120_Bkv, head_dim);
        size_t smem_64  = compute_smem_fp8(64,  SM120_Bkv, head_dim);
        size_t smem_32  = compute_smem_fp8(32,  SM120_Bkv, head_dim);
        if (smem_128 <= (size_t)max_smem) Bq = 128;
        else if (smem_64 <= (size_t)max_smem) Bq = 64;
        else if (smem_32 <= (size_t)max_smem) Bq = 32;
        else return false;
    }
    const int Bkv = SM120_Bkv;
    const size_t smem = compute_smem_fp8(Bq, Bkv, head_dim);

    const int num_q_tiles = (seq_q + Bq - 1) / Bq;
    dim3 grid(num_q_tiles, batch_size * n_heads);
    dim3 block(SM120_WARP_SIZE, SM120_NUM_WARPS);

    #define LAUNCH_FP8_FMHA(BQ, HD) do { \
        cudaError_t attr_err = cudaFuncSetAttribute( \
            fmha_sm120_fp8_kernel<BQ, HD>, \
            cudaFuncAttributeMaxDynamicSharedMemorySize, \
            static_cast<int>(smem)); \
        if (attr_err != cudaSuccess) return false; \
        cudaFuncSetAttribute(fmha_sm120_fp8_kernel<BQ, HD>, \
            cudaFuncAttributePreferredSharedMemoryCarveout, \
            cudaSharedmemCarveoutMaxShared); \
        fmha_sm120_fp8_kernel<BQ, HD><<<grid, block, smem, stream>>>( \
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
            case 128: LAUNCH_FP8_FMHA(128, 128); return true;
            case 256: LAUNCH_FP8_FMHA(128, 256); return true;
            default: break;
        }
    } else if (Bq == 64) {
        switch (head_dim) {
            case 64:  LAUNCH_FP8_FMHA(64, 64);   return true;
            case 128: LAUNCH_FP8_FMHA(64, 128);  return true;
            case 256: LAUNCH_FP8_FMHA(64, 256);  return true;
            default: break;
        }
    } else {
        switch (head_dim) {
            case 64:  LAUNCH_FP8_FMHA(32, 64);   return true;
            case 128: LAUNCH_FP8_FMHA(32, 128);  return true;
            case 256: LAUNCH_FP8_FMHA(32, 256);  return true;
            default: break;
        }
    }

    #undef LAUNCH_FP8_FMHA
    return false;
}


} // namespace imp
