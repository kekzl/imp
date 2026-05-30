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

static constexpr int SM120_WARP_SIZE = 32;
static constexpr int SM120_NUM_WARPS = 8;
static constexpr int SM120_BLOCK_THREADS = SM120_WARP_SIZE * SM120_NUM_WARPS;  // 256
static constexpr int SM120_Bkv = 64;                                           // KV tile size (columns)

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
__global__ void __launch_bounds__(SM120_BLOCK_THREADS, 2) fmha_sm120_kernel(
    const half* __restrict__ Q, const half* __restrict__ K, const half* __restrict__ V, half* __restrict__ O,
    int batch_size, int seq_q, int seq_kv, int n_heads, int n_kv_heads, float scale, bool causal,
    int sliding_window, float softcap, int q_offset) {
    constexpr int Bkv = SM120_Bkv;
    constexpr int head_dim = HD;

    // Threads-per-row for parallel softmax
    constexpr int TPR = SM120_BLOCK_THREADS / Bq;
    static_assert(TPR >= 1 && (TPR & (TPR - 1)) == 0, "TPR must be power of 2");

    // ---- index computation --------------------------------------------------
    const int tile_q = blockIdx.x;
    const int batch_head = blockIdx.y;
    const int batch_idx = batch_head / n_heads;
    const int head_idx = batch_head % n_heads;
    const int kv_head = head_idx / (n_heads / n_kv_heads);

    const int tid = threadIdx.x + threadIdx.y * blockDim.x;
    const int warp_id = tid / SM120_WARP_SIZE;
    const int q_start = tile_q * Bq;

    // Parallel softmax: which row and lane within row
    const int sm_row = tid / TPR;
    const int sm_lane = tid % TPR;

    // Global memory strides (row-major [batch, seq, heads, head_dim])
    const int64_t q_row_stride = (int64_t)n_heads * head_dim;
    const int64_t kv_row_stride = (int64_t)n_kv_heads * head_dim;

    const half* Q_ptr = Q + (int64_t)batch_idx * seq_q * q_row_stride + (int64_t)q_start * q_row_stride +
                        (int64_t)head_idx * head_dim;
    const half* K_ptr = K + (int64_t)batch_idx * seq_kv * kv_row_stride + (int64_t)kv_head * head_dim;
    const half* V_ptr = V + (int64_t)batch_idx * seq_kv * kv_row_stride + (int64_t)kv_head * head_dim;
    half* O_ptr = O + (int64_t)batch_idx * seq_q * q_row_stride + (int64_t)q_start * q_row_stride +
                  (int64_t)head_idx * head_dim;

    // ---- shared memory layout -----------------------------------------------
    // K and V share the same buffer (KV_tile): K is loaded first, consumed
    // by QK^T WMMA, then V is loaded into the same region for PV WMMA.
    extern __shared__ char smem[];

    half* Q_tile = reinterpret_cast<half*>(smem);
    half* KV_tile = Q_tile + Bq * head_dim;  // shared K/V buffer
    float* S_tile = reinterpret_cast<float*>(KV_tile + Bkv * head_dim);
    float* O_acc = S_tile + Bq * Bkv;
    float* row_m = O_acc + Bq * head_dim;
    float* row_l = row_m + Bq;

    // ---- load Q tile (vectorized float4 = 8 halves per iter) ---------------
    {
        const int total_vec8 = (Bq * head_dim) / 8;
        for (int vi = tid; vi < total_vec8; vi += SM120_BLOCK_THREADS) {
            int i = vi * 8;
            int r = i / head_dim;
            int d = i % head_dim;
            float4* dst = reinterpret_cast<float4*>(&Q_tile[i]);
            if (q_start + r < seq_q) {
                const float4* src = reinterpret_cast<const float4*>(&Q_ptr[(int64_t)r * q_row_stride + d]);
                *dst = *src;
            } else {
                *dst = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }
    }

    // ---- zero output accumulator + init running softmax state ---------------
    {
        // float4 = 4 FP32 zeros per store. Bq*HD is always a multiple of 4
        // (HD ∈ {64,96,128,256}, Bq ∈ {32,64,128}).
        const int total_vec4 = (Bq * head_dim) / 4;
        const float4 zero = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        for (int vi = tid; vi < total_vec4; vi += SM120_BLOCK_THREADS) {
            reinterpret_cast<float4*>(O_acc)[vi] = zero;
        }
    }
    if (tid < Bq) {
        row_m[tid] = -FLT_MAX;
        row_l[tid] = 0.0f;
    }
    __syncthreads();

    // ---- KV tile loop bounds ----
    int num_kv_tiles, first_kv_tile;
    compute_kv_tile_bounds(q_start, Bq, Bkv, seq_q, seq_kv, causal, sliding_window, first_kv_tile,
                           num_kv_tiles, q_offset);

    // Derived WMMA tiling constants
    const int hd_chunks = head_dim / SM120_WMMA_K;
    const int s_row_tiles = Bq / SM120_WMMA_M;
    const int s_col_tiles = Bkv / SM120_WMMA_N;
    const int s_total_tiles = s_row_tiles * s_col_tiles;
    const int o_row_tiles = Bq / SM120_WMMA_M;
    const int o_col_tiles = head_dim / SM120_WMMA_N;
    const int o_total_tiles = o_row_tiles * o_col_tiles;
    const int pv_chunks = Bkv / SM120_WMMA_K;

    // ================================================================
    // Main loop over KV tiles (Sawtooth: alternate scan direction per Q tile for L2 locality)
    // ================================================================
    const bool sawtooth_reverse = (blockIdx.x % 2 == 1);
    const int n_kv_iters = num_kv_tiles - first_kv_tile;
    for (int iter = 0; iter < n_kv_iters; iter++) {
        const int j = sawtooth_reverse ? (num_kv_tiles - 1 - iter) : (first_kv_tile + iter);
        const int kv_start = j * Bkv;

        // ---- Load K tile (vectorized float4 = 8 halves per iter) ----
        {
            const int total_vec8 = (Bkv * head_dim) / 8;
            for (int vi = tid; vi < total_vec8; vi += SM120_BLOCK_THREADS) {
                int i = vi * 8;
                int r = i / head_dim;
                int d = i % head_dim;
                float4* dst = reinterpret_cast<float4*>(&KV_tile[i]);
                if (kv_start + r < seq_kv) {
                    const float4* src = reinterpret_cast<const float4*>(
                        &K_ptr[(int64_t)(kv_start + r) * kv_row_stride + d]);
                    *dst = *src;
                } else {
                    *dst = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
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
                wmma::fragment<wmma::matrix_a, SM120_WMMA_M, SM120_WMMA_N, SM120_WMMA_K, half,
                               wmma::row_major>
                    a_frag;
                wmma::load_matrix_sync(a_frag, Q_tile + ri * SM120_WMMA_M * head_dim + k * SM120_WMMA_K,
                                       head_dim);

                wmma::fragment<wmma::matrix_b, SM120_WMMA_M, SM120_WMMA_N, SM120_WMMA_K, half,
                               wmma::col_major>
                    b_frag;
                wmma::load_matrix_sync(b_frag, KV_tile + ci * SM120_WMMA_N * head_dim + k * SM120_WMMA_K,
                                       head_dim);

                wmma::mma_sync(acc, a_frag, b_frag, acc);
            }

            wmma::store_matrix_sync(S_tile + ri * SM120_WMMA_M * Bkv + ci * SM120_WMMA_N, acc, Bkv,
                                    wmma::mem_row_major);
        }
        __syncthreads();

        // ---- Apply scale, softcap, and causal/sliding_window mask ----
        apply_score_masks(S_tile, Bq, Bkv, SM120_BLOCK_THREADS, tid, q_start, kv_start, seq_q, seq_kv, scale,
                          softcap, causal, sliding_window, q_offset);
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
                    float p = (s_val <= -FLT_MAX * 0.5f) ? 0.0f : __expf(s_val - m_new);
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

        // ---- Load V tile (vectorized: float4 = 8 halves per iter) ----
        {
            // All supported head_dims (64, 96, 128, 256) are multiples of 8,
            // so float4 loads are always aligned and in-bounds per row.
            const int total_vec8 = (Bkv * head_dim) / 8;
            for (int vi = tid; vi < total_vec8; vi += SM120_BLOCK_THREADS) {
                int i = vi * 8;
                int r = i / head_dim;
                int d = i % head_dim;
                float4* dst = reinterpret_cast<float4*>(&KV_tile[i]);
                if (kv_start + r < seq_kv) {
                    const float4* src = reinterpret_cast<const float4*>(
                        &V_ptr[(int64_t)(kv_start + r) * kv_row_stride + d]);
                    *dst = *src;
                } else {
                    *dst = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
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
                wmma::load_matrix_sync(o_frag, O_acc + ri * SM120_WMMA_M * head_dim + di * SM120_WMMA_N,
                                       head_dim, wmma::mem_row_major);

                for (int k = 0; k < pv_chunks; k++) {
                    wmma::fragment<wmma::matrix_a, SM120_WMMA_M, SM120_WMMA_N, SM120_WMMA_K, half,
                                   wmma::row_major>
                        p_frag;
                    wmma::load_matrix_sync(p_frag, P_half + ri * SM120_WMMA_M * Bkv + k * SM120_WMMA_K, Bkv);

                    wmma::fragment<wmma::matrix_b, SM120_WMMA_M, SM120_WMMA_N, SM120_WMMA_K, half,
                                   wmma::row_major>
                        v_frag;
                    wmma::load_matrix_sync(v_frag, KV_tile + k * SM120_WMMA_N * head_dim + di * SM120_WMMA_N,
                                           head_dim);

                    wmma::mma_sync(o_frag, p_frag, v_frag, o_frag);
                }

                wmma::store_matrix_sync(O_acc + ri * SM120_WMMA_M * head_dim + di * SM120_WMMA_N, o_frag,
                                        head_dim, wmma::mem_row_major);
            }
        }
        __syncthreads();
    }

    // ---- write final output to global memory (vectorized: 4 FP32 → 4 FP16 per iter) ----
    {
        const int total_vec4 = (Bq * head_dim) / 4;
        for (int vi = tid; vi < total_vec4; vi += SM120_BLOCK_THREADS) {
            int i = vi * 4;
            int r = i / head_dim;
            if (q_start + r >= seq_q)
                continue;
            // 4 FP32 → 2 half2 via __float22half2_rn → store as float (= 4 halves packed)
            float4 v = reinterpret_cast<const float4*>(O_acc)[vi];
            half2 lo = __float22half2_rn(make_float2(v.x, v.y));
            half2 hi = __float22half2_rn(make_float2(v.z, v.w));
            uint2 packed;
            packed.x = *reinterpret_cast<const uint32_t*>(&lo);
            packed.y = *reinterpret_cast<const uint32_t*>(&hi);
            *reinterpret_cast<uint2*>(&O_ptr[(int64_t)r * q_row_stride + (i % head_dim)]) = packed;
        }
    }
}

// =============================================================================
// Shared memory computation
// =============================================================================

static size_t compute_smem_sm120(int Bq, int Bkv, int head_dim) {
    return (size_t)Bq * head_dim * sizeof(half)     // Q_tile
           + (size_t)Bkv * head_dim * sizeof(half)  // KV_tile (shared K/V buffer)
           + (size_t)Bq * Bkv * sizeof(float)       // S_tile (float scores / half P overlay)
           + (size_t)Bq * head_dim * sizeof(float)  // O_acc
           + 2 * (size_t)Bq * sizeof(float);        // row_m + row_l
}

// =============================================================================
// Host launcher
// =============================================================================

bool fmha_sm120_prefill(const Tensor& Q, const Tensor& K, const Tensor& V, Tensor& O, float scale,
                        bool causal, int sliding_window, float softcap, cudaStream_t stream,
                        int q_offset) {
    if (Q.qtype != QType::F16)
        return false;

    const int batch_size = static_cast<int>(Q.shape[0]);
    const int seq_q = static_cast<int>(Q.shape[1]);
    const int n_heads = static_cast<int>(Q.shape[2]);
    const int head_dim = static_cast<int>(Q.shape[3]);
    const int seq_kv = static_cast<int>(K.shape[1]);
    const int n_kv_heads = static_cast<int>(K.shape[2]);

    if (n_kv_heads == 0 || n_heads % n_kv_heads != 0)
        return false;
    if (seq_q == 0 || seq_kv == 0)
        return false;
    if (head_dim % SM120_WMMA_K != 0)
        return false;

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
        size_t smem_64 = compute_smem_sm120(64, SM120_Bkv, head_dim);
        size_t smem_32 = compute_smem_sm120(32, SM120_Bkv, head_dim);
        size_t occ2_cap = static_cast<size_t>(max_smem) / 2;
        if (smem_128 <= occ2_cap) {
            Bq = 128;
        } else if (smem_64 <= occ2_cap) {
            Bq = 64;
        } else if (smem_32 <= occ2_cap) {
            Bq = 32;
        } else if (smem_32 <= (size_t)max_smem) {
            Bq = 32;
        } else {
            IMP_LOG_DEBUG("FMHA sm120: no Bq fits smem (hd=%d, smem_32=%zu, max=%d)", head_dim, smem_32,
                          max_smem);
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

    IMP_LOG_DEBUG(
        "FMHA sm120: B=%d Sq=%d Skv=%d nh=%d nkv=%d hd=%d Bq=%d Bkv=%d smem=%zu "
        "causal=%d sw=%d softcap=%.1f",
        batch_size, seq_q, seq_kv, n_heads, n_kv_heads, head_dim, Bq, Bkv, smem, causal, sliding_window,
        softcap);

#define LAUNCH_FMHA_SM120(BQ, HD)                                                                         \
    do {                                                                                                  \
        cudaError_t attr_err = cudaFuncSetAttribute(fmha_sm120_kernel<BQ, HD>,                            \
                                                    cudaFuncAttributeMaxDynamicSharedMemorySize,          \
                                                    static_cast<int>(smem));                              \
        if (attr_err != cudaSuccess) {                                                                    \
            IMP_LOG_WARN("FMHA sm120: cudaFuncSetAttribute failed for Bq=%d HD=%d smem=%zu: %s", BQ, HD,  \
                         smem, cudaGetErrorString(attr_err));                                             \
            return false;                                                                                 \
        }                                                                                                 \
        cudaFuncSetAttribute(fmha_sm120_kernel<BQ, HD>, cudaFuncAttributePreferredSharedMemoryCarveout,   \
                             cudaSharedmemCarveoutMaxShared);                                             \
        fmha_sm120_kernel<BQ, HD>                                                                         \
            <<<grid, block, smem, stream>>>(reinterpret_cast<const half*>(Q.data),                        \
                                            reinterpret_cast<const half*>(K.data),                        \
                                            reinterpret_cast<const half*>(V.data),                        \
                                            reinterpret_cast<half*>(O.data), batch_size, seq_q, seq_kv,   \
                                            n_heads, n_kv_heads, scale, causal, sliding_window, softcap,  \
                                            q_offset);                                                    \
    } while (0)

    if (Bq == 128) {
        switch (head_dim) {
            case 64:
                LAUNCH_FMHA_SM120(128, 64);
                return true;
            case 96:
                LAUNCH_FMHA_SM120(128, 96);
                return true;
            case 128:
                LAUNCH_FMHA_SM120(128, 128);
                return true;
            case 256:
                LAUNCH_FMHA_SM120(128, 256);
                return true;
            default:
                break;
        }
    } else if (Bq == 64) {
        switch (head_dim) {
            case 64:
                LAUNCH_FMHA_SM120(64, 64);
                return true;
            case 96:
                LAUNCH_FMHA_SM120(64, 96);
                return true;
            case 128:
                LAUNCH_FMHA_SM120(64, 128);
                return true;
            case 256:
                LAUNCH_FMHA_SM120(64, 256);
                return true;
            default:
                break;
        }
    } else {
        // Bq=32: for large head_dim (256) where Bq=64 exceeds smem
        switch (head_dim) {
            case 64:
                LAUNCH_FMHA_SM120(32, 64);
                return true;
            case 96:
                LAUNCH_FMHA_SM120(32, 96);
                return true;
            case 128:
                LAUNCH_FMHA_SM120(32, 128);
                return true;
            case 256:
                LAUNCH_FMHA_SM120(32, 256);
                return true;
            default:
                break;
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
__global__ void __launch_bounds__(SM120_BLOCK_THREADS, 1) fmha_sm120_fp8_kernel(
    const half* __restrict__ Q, const half* __restrict__ K, const half* __restrict__ V, half* __restrict__ O,
    int batch_size, int seq_q, int seq_kv, int n_heads, int n_kv_heads, float scale, bool causal,
    int sliding_window, float softcap, int q_offset) {
    constexpr int Bkv = SM120_Bkv;
    constexpr int head_dim = HD;
    constexpr int TPR = SM120_BLOCK_THREADS / Bq;
    static_assert(TPR >= 1 && (TPR & (TPR - 1)) == 0, "TPR must be power of 2");

    const int tile_q = blockIdx.x;
    const int batch_head = blockIdx.y;
    const int batch_idx = batch_head / n_heads;
    const int head_idx = batch_head % n_heads;
    const int kv_head = head_idx / (n_heads / n_kv_heads);

    const int tid = threadIdx.x + threadIdx.y * blockDim.x;
    const int warp_id = tid / SM120_WARP_SIZE;
    const int lane_id = tid % SM120_WARP_SIZE;
    const int q_start = tile_q * Bq;

    const int sm_row = tid / TPR;
    const int sm_lane = tid % TPR;

    const int64_t q_row_stride = (int64_t)n_heads * head_dim;
    const int64_t kv_row_stride = (int64_t)n_kv_heads * head_dim;

    const half* Q_ptr = Q + (int64_t)batch_idx * seq_q * q_row_stride + (int64_t)q_start * q_row_stride +
                        (int64_t)head_idx * head_dim;
    const half* K_ptr = K + (int64_t)batch_idx * seq_kv * kv_row_stride + (int64_t)kv_head * head_dim;
    const half* V_ptr = V + (int64_t)batch_idx * seq_kv * kv_row_stride + (int64_t)kv_head * head_dim;
    half* O_ptr = O + (int64_t)batch_idx * seq_q * q_row_stride + (int64_t)q_start * q_row_stride +
                  (int64_t)head_idx * head_dim;

    // Shared memory layout:
    //   Q_fp8:  uint8[Bq x HD]      — Q converted to FP8 E4M3
    //   KV_fp8: uint8[Bkv x HD]     — K converted to FP8 (reused for V as FP16)
    //   KV_fp16: half[Bkv x HD]     — V loaded as FP16 (overlaps KV_fp8 for V phase)
    //   S_tile: float[Bq x Bkv]     — score tile
    //   O_acc:  float[Bq x HD]      — output accumulator
    //   row_m/l: float[Bq] × 2      — softmax state
    extern __shared__ char smem[];

    uint8_t* Q_fp8 = reinterpret_cast<uint8_t*>(smem);
    uint8_t* KV_fp8 = Q_fp8 + Bq * head_dim;          // K as FP8 (first half of KV region)
    half* KV_fp16 = reinterpret_cast<half*>(KV_fp8);  // V as FP16 (reuses full KV region)
    // S_tile must live AFTER the full V-as-half region, not after just the
    // FP8 K region. V writes Bkv * head_dim halves = 2 * Bkv * head_dim bytes,
    // so advancing only Bkv * head_dim bytes (as the code originally did)
    // places S_tile inside the V area — V row Bkv/2+ overwrites P values
    // and poisons the PV MMA output with garbage/NaN.
    float* S_tile = reinterpret_cast<float*>(KV_fp8 + Bkv * head_dim * sizeof(half));
    float* O_acc = S_tile + Bq * Bkv;
    float* row_m = O_acc + Bq * head_dim;
    float* row_l = row_m + Bq;

    // Load Q tile and convert to FP8 E4M3 (vectorized: 4 halves → 4 FP8 bytes per cvt pair).
    // HW cvt.rn.satfinite.e4m3x2.f16x2 already clamps to ±448 — no manual saturate.
    {
        const int total_vec4 = (Bq * head_dim) / 4;
        for (int vi = tid; vi < total_vec4; vi += SM120_BLOCK_THREADS) {
            int i = vi * 4;
            int r = i / head_dim;
            int d = i % head_dim;
            // Out-of-range rows zero-fill as u32 (4 bytes)
            if (q_start + r >= seq_q) {
                reinterpret_cast<uint32_t*>(Q_fp8)[vi] = 0;
                continue;
            }
            // 4 consecutive halves are always within head_dim (HD % 4 == 0 for Bq multiples).
            const half* src = &Q_ptr[(int64_t)r * q_row_stride + d];
            reinterpret_cast<uint32_t*>(Q_fp8)[vi] = cvt_4xfp16_to_4xe4m3(src);
        }
        // Scalar tail if Bq*head_dim isn't a multiple of 4 (shouldn't happen on
        // supported HDs: 64,96,128,256 × Bq divisible configs all hit the vec4 fast path).
    }

    // Zero O_acc (vectorized float4 = 4 FP32 zeros/iter) + init softmax
    {
        const int total_vec4 = (Bq * head_dim) / 4;
        const float4 zero = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        for (int vi = tid; vi < total_vec4; vi += SM120_BLOCK_THREADS) {
            reinterpret_cast<float4*>(O_acc)[vi] = zero;
        }
    }
    if (tid < Bq) {
        row_m[tid] = -FLT_MAX;
        row_l[tid] = 0.0f;
    }
    __syncthreads();

    // KV tile bounds
    int num_kv_tiles, first_kv_tile;
    compute_kv_tile_bounds(q_start, Bq, Bkv, seq_q, seq_kv, causal, sliding_window, first_kv_tile,
                           num_kv_tiles, q_offset);

    // FP8 MMA tiling: m16n8k32 → output is 16×8, need 2 calls for 16×16 score tile
    constexpr int S_M = 16;
    constexpr int S_N = 8;   // MMA output width
    constexpr int S_K = 32;  // FP8 k-dim
    const int hd_chunks_fp8 = head_dim / S_K;
    const int s_row_tiles = Bq / S_M;
    const int s_col_tiles_half = Bkv / S_N;  // each m16n8 tile
    const int s_total_tiles = s_row_tiles * s_col_tiles_half;

    // FP16 WMMA tiling for PV (unchanged)
    const int o_row_tiles = Bq / SM120_WMMA_M;
    const int o_col_tiles = head_dim / SM120_WMMA_N;
    const int o_total_tiles = o_row_tiles * o_col_tiles;
    const int pv_chunks = Bkv / SM120_WMMA_K;

    // Main KV tile loop
    for (int j = first_kv_tile; j < num_kv_tiles; j++) {
        const int kv_start = j * Bkv;

        // Load K tile and convert to FP8 (vectorized: 4 halves → 4 FP8 bytes).
        {
            const int total_vec4 = (Bkv * head_dim) / 4;
            for (int vi = tid; vi < total_vec4; vi += SM120_BLOCK_THREADS) {
                int i = vi * 4;
                int r = i / head_dim;
                int d = i % head_dim;
                if (kv_start + r >= seq_kv) {
                    reinterpret_cast<uint32_t*>(KV_fp8)[vi] = 0;
                    continue;
                }
                const half* src = &K_ptr[(int64_t)(kv_start + r) * kv_row_stride + d];
                reinterpret_cast<uint32_t*>(KV_fp8)[vi] = cvt_4xfp16_to_4xe4m3(src);
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
                    const uint32_t* q_row0 = reinterpret_cast<const uint32_t*>(
                        q_base + row_in_tile * head_dim + col_base);
                    const uint32_t* q_row8 = reinterpret_cast<const uint32_t*>(
                        q_base + (row_in_tile + 8) * head_dim + col_base);
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
                    const uint32_t* k_ptr0 = reinterpret_cast<const uint32_t*>(
                        k_base + col_in_tile * head_dim + k_base_offset);
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
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(d0), "f"(d1), "f"(d2),
                      "f"(d3));
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
                S_tile[(base_row + r0) * Bkv + base_col + c0] = d0;
                S_tile[(base_row + r0) * Bkv + base_col + c0 + 1] = d1;
                S_tile[(base_row + r0 + 8) * Bkv + base_col + c0] = d2;
                S_tile[(base_row + r0 + 8) * Bkv + base_col + c0 + 1] = d3;
            }
        }
        __syncthreads();

        // Apply scale, softcap, masks (same as FP16 path)
        apply_score_masks(S_tile, Bq, Bkv, SM120_BLOCK_THREADS, tid, q_start, kv_start, seq_q, seq_kv, scale,
                          softcap, causal, sliding_window, q_offset);
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
                    float p = (s_val <= -FLT_MAX * 0.5f) ? 0.0f : __expf(s_val - m_new);
                    partial_sum += p;
                    S_tile[r * Bkv + c] = p;
                }
            }
#pragma unroll
            for (int offset = TPR / 2; offset >= 1; offset >>= 1)
                partial_sum += __shfl_xor_sync(0xffffffff, partial_sum, offset);

            float l_old = row_valid ? row_l[r] : 0.0f;
            float l_new = alpha * l_old + partial_sum;
            if (sm_lane == 0 && row_valid) {
                row_m[r] = m_new;
                row_l[r] = l_new;
            }

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

        // Load V tile as FP16 (vectorized float4 = 8 halves per iter)
        {
            const int total_vec8 = (Bkv * head_dim) / 8;
            for (int vi = tid; vi < total_vec8; vi += SM120_BLOCK_THREADS) {
                int i = vi * 8;
                int r = i / head_dim;
                int d = i % head_dim;
                float4* dst = reinterpret_cast<float4*>(&KV_fp16[i]);
                if (kv_start + r < seq_kv) {
                    const float4* src = reinterpret_cast<const float4*>(
                        &V_ptr[(int64_t)(kv_start + r) * kv_row_stride + d]);
                    *dst = *src;
                } else {
                    *dst = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                }
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
                wmma::load_matrix_sync(o_frag, O_acc + ri * SM120_WMMA_M * head_dim + di * SM120_WMMA_N,
                                       head_dim, wmma::mem_row_major);

                for (int k = 0; k < pv_chunks; k++) {
                    wmma::fragment<wmma::matrix_a, SM120_WMMA_M, SM120_WMMA_N, SM120_WMMA_K, half,
                                   wmma::row_major>
                        p_frag;
                    wmma::load_matrix_sync(p_frag, P_half + ri * SM120_WMMA_M * Bkv + k * SM120_WMMA_K, Bkv);

                    wmma::fragment<wmma::matrix_b, SM120_WMMA_M, SM120_WMMA_N, SM120_WMMA_K, half,
                                   wmma::row_major>
                        v_frag;
                    wmma::load_matrix_sync(v_frag, KV_fp16 + k * SM120_WMMA_N * head_dim + di * SM120_WMMA_N,
                                           head_dim);

                    wmma::mma_sync(o_frag, p_frag, v_frag, o_frag);
                }

                wmma::store_matrix_sync(O_acc + ri * SM120_WMMA_M * head_dim + di * SM120_WMMA_N, o_frag,
                                        head_dim, wmma::mem_row_major);
            }
        }
        __syncthreads();
    }

    // Write output (vectorized: 4 FP32 → 4 FP16 per iter)
    {
        const int total_vec4 = (Bq * head_dim) / 4;
        for (int vi = tid; vi < total_vec4; vi += SM120_BLOCK_THREADS) {
            int i = vi * 4;
            int r = i / head_dim;
            if (q_start + r >= seq_q)
                continue;
            float4 v = reinterpret_cast<const float4*>(O_acc)[vi];
            half2 lo = __float22half2_rn(make_float2(v.x, v.y));
            half2 hi = __float22half2_rn(make_float2(v.z, v.w));
            uint2 packed;
            packed.x = *reinterpret_cast<const uint32_t*>(&lo);
            packed.y = *reinterpret_cast<const uint32_t*>(&hi);
            *reinterpret_cast<uint2*>(&O_ptr[(int64_t)r * q_row_stride + (i % head_dim)]) = packed;
        }
    }
}

// Shared memory for FP8 variant: Q_fp8 uses bytes not halves for Q
static size_t compute_smem_fp8(int Bq, int Bkv, int head_dim) {
    return (size_t)Bq * head_dim * sizeof(uint8_t)  // Q_fp8
           + (size_t)Bkv * head_dim * sizeof(half)  // KV buffer (FP8 K or FP16 V, half is larger)
           + (size_t)Bq * Bkv * sizeof(float)       // S_tile
           + (size_t)Bq * head_dim * sizeof(float)  // O_acc
           + 2 * (size_t)Bq * sizeof(float);        // row_m + row_l
}

bool fmha_sm120_fp8_prefill(const Tensor& Q, const Tensor& K, const Tensor& V, Tensor& O, float scale,
                            bool causal, int sliding_window, float softcap, cudaStream_t stream,
                            int q_offset) {
    if (Q.qtype != QType::F16)
        return false;

    const int batch_size = static_cast<int>(Q.shape[0]);
    const int seq_q = static_cast<int>(Q.shape[1]);
    const int n_heads = static_cast<int>(Q.shape[2]);
    const int head_dim = static_cast<int>(Q.shape[3]);
    const int seq_kv = static_cast<int>(K.shape[1]);
    const int n_kv_heads = static_cast<int>(K.shape[2]);

    if (n_kv_heads == 0 || n_heads % n_kv_heads != 0)
        return false;
    if (seq_q == 0 || seq_kv == 0)
        return false;
    if (head_dim % 32 != 0)
        return false;  // FP8 MMA needs k%32==0

    int device = 0;
    cudaGetDevice(&device);
    int max_smem = 0;
    cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);

    int Bq;
    {
        size_t smem_128 = compute_smem_fp8(128, SM120_Bkv, head_dim);
        size_t smem_64 = compute_smem_fp8(64, SM120_Bkv, head_dim);
        size_t smem_32 = compute_smem_fp8(32, SM120_Bkv, head_dim);
        if (smem_128 <= (size_t)max_smem)
            Bq = 128;
        else if (smem_64 <= (size_t)max_smem)
            Bq = 64;
        else if (smem_32 <= (size_t)max_smem)
            Bq = 32;
        else
            return false;
    }
    const int Bkv = SM120_Bkv;
    const size_t smem = compute_smem_fp8(Bq, Bkv, head_dim);

    const int num_q_tiles = (seq_q + Bq - 1) / Bq;
    dim3 grid(num_q_tiles, batch_size * n_heads);
    dim3 block(SM120_WARP_SIZE, SM120_NUM_WARPS);

#define LAUNCH_FP8_FMHA(BQ, HD)                                                                             \
    do {                                                                                                    \
        cudaError_t attr_err = cudaFuncSetAttribute(fmha_sm120_fp8_kernel<BQ, HD>,                          \
                                                    cudaFuncAttributeMaxDynamicSharedMemorySize,            \
                                                    static_cast<int>(smem));                                \
        if (attr_err != cudaSuccess)                                                                        \
            return false;                                                                                   \
        cudaFuncSetAttribute(fmha_sm120_fp8_kernel<BQ, HD>, cudaFuncAttributePreferredSharedMemoryCarveout, \
                             cudaSharedmemCarveoutMaxShared);                                               \
        fmha_sm120_fp8_kernel<BQ, HD>                                                                       \
            <<<grid, block, smem, stream>>>(reinterpret_cast<const half*>(Q.data),                          \
                                            reinterpret_cast<const half*>(K.data),                          \
                                            reinterpret_cast<const half*>(V.data),                          \
                                            reinterpret_cast<half*>(O.data), batch_size, seq_q, seq_kv,     \
                                            n_heads, n_kv_heads, scale, causal, sliding_window, softcap,    \
                                            q_offset);                                                      \
    } while (0)

    if (Bq == 128) {
        switch (head_dim) {
            case 128:
                LAUNCH_FP8_FMHA(128, 128);
                return true;
            case 256:
                LAUNCH_FP8_FMHA(128, 256);
                return true;
            default:
                break;
        }
    } else if (Bq == 64) {
        switch (head_dim) {
            case 64:
                LAUNCH_FP8_FMHA(64, 64);
                return true;
            case 128:
                LAUNCH_FP8_FMHA(64, 128);
                return true;
            case 256:
                LAUNCH_FP8_FMHA(64, 256);
                return true;
            default:
                break;
        }
    } else {
        switch (head_dim) {
            case 64:
                LAUNCH_FP8_FMHA(32, 64);
                return true;
            case 128:
                LAUNCH_FP8_FMHA(32, 128);
                return true;
            case 256:
                LAUNCH_FP8_FMHA(32, 256);
                return true;
            default:
                break;
        }
    }

#undef LAUNCH_FP8_FMHA
    return false;
}

// =============================================================================
// FA2 register-resident kernel ("echtes FA")
// =============================================================================
//
// True FlashAttention-2 for sm_120. Unlike fmha_sm120_fp8_kernel above (which
// materializes S, P and O in shared memory and round-trips them every KV tile
// behind 4 __syncthreads → barrier-bound, ncu: 14.5% compute / 75.7% L1/TEX),
// this keeps the score tile S, the softmax weights P and the output accumulator
// O entirely in REGISTERS. Only K (fp8) and V (f16) are staged in smem.
//
// Work mapping: 8 warps × 16 query rows = Bq=128 rows/block. Each warp owns its
// 16 rows and runs an INDEPENDENT online softmax — no cross-warp reduction, no
// softmax smem. KV processed in Bkv=64 tiles.
//
// The layout trick that makes it transpose-free: the m16n8 accumulator output
// of QK (fragment layout: thread t holds rows {t/4, t/4+8} × cols {(t%4)*2,+1})
// is byte-identical to the m16n8k16 A-operand layout of the PV MMA. So after the
// in-register softmax, two adjacent 16×8 S tiles (Bkv cols [16m,16m+8) and
// [16m+8,16m+16)) assemble directly into the 16×16 P A-fragment for K-group m
// of P@V — no movmatrix, no smem.
//
//   per KV tile: load K+V → smem → __syncthreads → QK(reg) → softmax(reg) →
//                PV(reg) → __syncthreads.  (2 barriers/tile, 0 S/P/O round-trips)
//
// cp.async double-buffer: K/V for tile j+1 are prefetched (raw f16, cp.async)
// into the alternate smem slot while tile j computes — the KV-load latency that
// previously serialized behind __syncthreads now overlaps the prior tile's
// QK/softmax/PV. K is converted f16→fp8 inline at QK operand-fetch (spare ALU on
// this barrier-bound kernel). Targets the 23% SM-util / barrier-bound ceiling.

__device__ __forceinline__ uint32_t pack2_f2h(float a, float b) {
    __half2 h = __floats2half2_rn(a, b);
    uint32_t r;
    memcpy(&r, &h, 4);
    return r;
}
__device__ __forceinline__ uint32_t pack2_hh(half a, half b) {
    __half2 h = __halves2half2(a, b);
    uint32_t r;
    memcpy(&r, &h, 4);
    return r;
}
__device__ __forceinline__ float quad_max(float v) {
    v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFF, v, 1));
    v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFF, v, 2));
    return v;
}
__device__ __forceinline__ float quad_sum(float v) {
    v += __shfl_xor_sync(0xFFFFFFFF, v, 1);
    v += __shfl_xor_sync(0xFFFFFFFF, v, 2);
    return v;
}

// cp.async primitives (cp_async_cg_16 / cp_async_commit / cp_async_wait_group)
// come from compute/attention_paged_common.cuh (included above).

// Prefetch one Bkv×HD KV tile (raw f16, no conversion) into double-buffer slots
// via cp.async. K stays f16 in smem and is converted to fp8 inline at QK-MMA
// operand-fetch time (the kernel is barrier-bound, not compute-bound — spare ALU).
// Bank-conflict padding (ncu 2026-05-30: this FA2 kernel was mio_throttle-bound —
// ~75% of smem load wavefronts were bank-conflict replays because a head_dim=128
// row stride is exactly 32 banks (128 B), so every row aliases onto the same banks).
// Padding the smem row strides breaks the aliasing:
//   Q_fp8 (bytes): +16 → stride 144 B (36 words). The four uint32 QK reads each
//                  become a perfect 0..31 bank permutation → conflict-free.
//   K/V (halfs):   +8  → stride 136 (16 B, preserves cp.async 16-B alignment) →
//                  cuts the wide cvt/V smem reads from ~8-way down to ~2-way.
// The pad columns are never written or read; they only shift row addresses.
constexpr int FA2_Q_PAD = 16;   // extra bytes per Q_fp8 row
constexpr int FA2_KV_PAD = 8;   // extra halfs per K/V row (16 B, cp.async-aligned)

// Out-of-range rows: K left stale (masked out in the score step), V zero-filled
// (P=0 for those cols, but 0*NaN=NaN would poison O, so V must be finite).
template <int HD>
__device__ __forceinline__ void prefetch_kv_tile(half* K_dst, half* V_dst, const half* K_ptr,
                                                  const half* V_ptr, int kv_start, int seq_kv,
                                                  int64_t kv_row_stride, int tid) {
    constexpr int Bkv = 64;
    constexpr int VEC = 8;                       // 8 halfs = 16 B per cp.async
    constexpr int vecs_per_row = HD / VEC;       // 16
    constexpr int total_vecs = Bkv * vecs_per_row;
    constexpr int KVS = HD + FA2_KV_PAD;         // padded smem row stride (halfs)
#pragma unroll
    for (int vi = tid; vi < total_vecs; vi += SM120_BLOCK_THREADS) {
        int r = vi / vecs_per_row;
        int c = (vi % vecs_per_row) * VEC;
        if (kv_start + r < seq_kv) {
            const half* ks = K_ptr + (int64_t)(kv_start + r) * kv_row_stride + c;
            const half* vs = V_ptr + (int64_t)(kv_start + r) * kv_row_stride + c;
            cp_async_cg_16(&K_dst[r * KVS + c], ks);
            cp_async_cg_16(&V_dst[r * KVS + c], vs);
        } else {
            *reinterpret_cast<uint4*>(&V_dst[r * KVS + c]) = make_uint4(0, 0, 0, 0);
        }
    }
}

template <int HD>
__global__ void fmha_sm120_fa2_kernel(const half* __restrict__ Q, const half* __restrict__ K,
                                      const half* __restrict__ V, half* __restrict__ O, int batch_size,
                                      int seq_q, int seq_kv, int n_heads, int n_kv_heads, float scale,
                                      bool causal, int sliding_window, float softcap, int q_offset) {
    constexpr int Bq = 128;
    constexpr int Bkv = 64;
    constexpr int head_dim = HD;
    constexpr int N_S = Bkv / 8;    // QK N-tiles (8-col groups) per row tile = 8
    constexpr int N_O = HD / 8;     // PV N-tiles (8 HD-col groups)
    constexpr int N_KG = Bkv / 16;  // PV K-groups (16 Bkv-col groups) = 4
    constexpr int KC = HD / 32;     // QK k-chunks (32 each)

    const int tile_q = blockIdx.x;
    const int batch_head = blockIdx.y;
    const int batch_idx = batch_head / n_heads;
    const int head_idx = batch_head % n_heads;
    const int kv_head = head_idx / (n_heads / n_kv_heads);

    const int lane = threadIdx.x;       // 0..31
    const int warp_id = threadIdx.y;    // 0..7  → this warp's row tile
    const int tid = lane + warp_id * 32;
    const int q_start = tile_q * Bq;
    const int row_lo = warp_id * 16 + lane / 4;  // local row within Bq for d0/d1
    const int rl = lane / 4;                      // row in 16-tile (lo)
    const int cl = (lane % 4) * 2;                // col base in 8/16-tile

    const int64_t q_row_stride = (int64_t)n_heads * head_dim;
    const int64_t kv_row_stride = (int64_t)n_kv_heads * head_dim;
    const half* Q_ptr = Q + (int64_t)batch_idx * seq_q * q_row_stride + (int64_t)q_start * q_row_stride +
                        (int64_t)head_idx * head_dim;
    const half* K_ptr = K + (int64_t)batch_idx * seq_kv * kv_row_stride + (int64_t)kv_head * head_dim;
    const half* V_ptr = V + (int64_t)batch_idx * seq_kv * kv_row_stride + (int64_t)kv_head * head_dim;
    half* O_ptr = O + (int64_t)batch_idx * seq_q * q_row_stride + (int64_t)q_start * q_row_stride +
                  (int64_t)head_idx * head_dim;

    // Padded smem row strides (see FA2_Q_PAD / FA2_KV_PAD — bank-conflict relief).
    constexpr int QSTRIDE = head_dim + FA2_Q_PAD;   // Q_fp8 row stride in BYTES
    constexpr int KVSTRIDE = head_dim + FA2_KV_PAD;  // K/V row stride in HALFS

    extern __shared__ char smem[];
    uint8_t* Q_fp8 = reinterpret_cast<uint8_t*>(smem);
    half* K_buf = reinterpret_cast<half*>(Q_fp8 + Bq * QSTRIDE);  // [2][Bkv*KVSTRIDE] f16
    half* V_buf = K_buf + 2 * Bkv * KVSTRIDE;                     // [2][Bkv*KVSTRIDE] f16

    // ---- load Q → fp8 once (4 halves → 4 e4m3 per thread) ----
    {
        const int total_vec4 = (Bq * head_dim) / 4;
        for (int vi = tid; vi < total_vec4; vi += SM120_BLOCK_THREADS) {
            int i = vi * 4;
            int r = i / head_dim;
            int d = i % head_dim;  // multiple of 4 → uint32-aligned into the padded row
            uint32_t* dst = reinterpret_cast<uint32_t*>(Q_fp8 + r * QSTRIDE + d);
            if (q_start + r >= seq_q) {
                *dst = 0;
            } else {
                *dst = cvt_4xfp16_to_4xe4m3(&Q_ptr[(int64_t)r * q_row_stride + d]);
            }
        }
    }

    // ---- per-warp register state ----
    float O_frag[N_O][4];
#pragma unroll
    for (int hn = 0; hn < N_O; hn++)
        O_frag[hn][0] = O_frag[hn][1] = O_frag[hn][2] = O_frag[hn][3] = 0.0f;
    float mA = -FLT_MAX, mB = -FLT_MAX, lA = 0.0f, lB = 0.0f;

    int num_kv_tiles, first_kv_tile;
    compute_kv_tile_bounds(q_start, Bq, Bkv, seq_q, seq_kv, causal, sliding_window, first_kv_tile,
                           num_kv_tiles, q_offset);
    // prologue: kick off the first KV tile's load into buffer slot 0
    if (first_kv_tile < num_kv_tiles)
        prefetch_kv_tile<head_dim>(K_buf, V_buf, K_ptr, V_ptr, first_kv_tile * Bkv, seq_kv, kv_row_stride,
                                   tid);
    cp_async_commit();
    __syncthreads();  // Q_fp8 (produced above) visible before QK reads it

    for (int j = first_kv_tile; j < num_kv_tiles; j++) {
        const int slot = (j - first_kv_tile) & 1;
        const int kv_start = j * Bkv;

        // prefetch tile j+1 into the alternate slot, overlapping this tile's compute
        if (j + 1 < num_kv_tiles) {
            const int nslot = slot ^ 1;
            prefetch_kv_tile<head_dim>(K_buf + nslot * Bkv * KVSTRIDE, V_buf + nslot * Bkv * KVSTRIDE,
                                       K_ptr, V_ptr, (j + 1) * Bkv, seq_kv, kv_row_stride, tid);
            cp_async_commit();
            cp_async_wait_group<1>();  // this tile (slot) landed; tile j+1 still in flight
        } else {
            cp_async_wait_group<0>();
        }
        __syncthreads();  // this tile's K/V fully landed for all threads

        const half* K_cur = K_buf + slot * Bkv * KVSTRIDE;
        const half* V_cur = V_buf + slot * Bkv * KVSTRIDE;

        // ---- QK: S[n] = Q(warp rows) @ K[n-tile]^T, fp8 m16n8k32 (K cvt f16→fp8 inline) ----
        float S[N_S][4];
#pragma unroll
        for (int n = 0; n < N_S; n++) {
            float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
#pragma unroll
            for (int k = 0; k < KC; k++) {
                const uint8_t* qb = Q_fp8 + warp_id * 16 * QSTRIDE + k * 32;
                uint32_t a0 = *reinterpret_cast<const uint32_t*>(qb + rl * QSTRIDE + cl * 2);
                uint32_t a1 = *reinterpret_cast<const uint32_t*>(qb + rl * QSTRIDE + cl * 2 + 16);
                uint32_t a2 = *reinterpret_cast<const uint32_t*>(qb + (rl + 8) * QSTRIDE + cl * 2);
                uint32_t a3 = *reinterpret_cast<const uint32_t*>(qb + (rl + 8) * QSTRIDE + cl * 2 + 16);
                const half* kb = K_cur + n * 8 * KVSTRIDE + k * 32;
                uint32_t b0 = cvt_4xfp16_to_4xe4m3(kb + rl * KVSTRIDE + cl * 2);
                uint32_t b1 = cvt_4xfp16_to_4xe4m3(kb + rl * KVSTRIDE + cl * 2 + 16);
#if __CUDA_ARCH__ >= 1200
                asm volatile(
                    "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                    : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(d0), "f"(d1), "f"(d2),
                      "f"(d3));
#endif
            }
            S[n][0] = d0;
            S[n][1] = d1;
            S[n][2] = d2;
            S[n][3] = d3;
        }

        // ---- scale + softcap + causal/SWA mask (per register) ----
#pragma unroll
        for (int n = 0; n < N_S; n++) {
            int colb = kv_start + n * 8 + cl;
            int gqA = q_offset + q_start + warp_id * 16 + rl;
            int gqB = gqA + 8;
            int lqA = q_start + warp_id * 16 + rl;
#pragma unroll
            for (int e = 0; e < 4; e++) {
                int row16 = (e < 2) ? rl : (rl + 8);
                int col = colb + (e & 1);
                int gq = (e < 2) ? gqA : gqB;
                int lq = lqA + ((e < 2) ? 0 : 8);
                float v = S[n][e];
                if (lq < seq_q && col < seq_kv) {
                    v *= scale;
                    if (softcap > 0.0f)
                        v = softcap * tanhf(v / softcap);
                    if (causal && gq < col)
                        v = -FLT_MAX;
                    if (sliding_window > 0 && (gq - col) >= sliding_window)
                        v = -FLT_MAX;
                } else {
                    v = -FLT_MAX;
                }
                S[n][e] = v;
                (void)row16;
            }
        }

        // ---- online softmax (register, quad-shuffle across the 4 lanes/row) ----
        float mlA = -FLT_MAX, mlB = -FLT_MAX;
#pragma unroll
        for (int n = 0; n < N_S; n++) {
            mlA = fmaxf(mlA, fmaxf(S[n][0], S[n][1]));
            mlB = fmaxf(mlB, fmaxf(S[n][2], S[n][3]));
        }
        float mijA = quad_max(mlA), mijB = quad_max(mlB);
        float mnA = fmaxf(mA, mijA), mnB = fmaxf(mB, mijB);
        float alphaA = __expf(mA - mnA), alphaB = __expf(mB - mnB);

        float psA = 0.f, psB = 0.f;
#pragma unroll
        for (int n = 0; n < N_S; n++) {
            float p0 = (S[n][0] <= -FLT_MAX * 0.5f) ? 0.f : __expf(S[n][0] - mnA);
            float p1 = (S[n][1] <= -FLT_MAX * 0.5f) ? 0.f : __expf(S[n][1] - mnA);
            float p2 = (S[n][2] <= -FLT_MAX * 0.5f) ? 0.f : __expf(S[n][2] - mnB);
            float p3 = (S[n][3] <= -FLT_MAX * 0.5f) ? 0.f : __expf(S[n][3] - mnB);
            S[n][0] = p0;
            S[n][1] = p1;
            S[n][2] = p2;
            S[n][3] = p3;
            psA += p0 + p1;
            psB += p2 + p3;
        }
        psA = quad_sum(psA);
        psB = quad_sum(psB);
        lA = alphaA * lA + psA;
        lB = alphaB * lB + psB;
        mA = mnA;
        mB = mnB;
        // rescale O accumulator (rows A=o0,o1 ; B=o2,o3)
#pragma unroll
        for (int hn = 0; hn < N_O; hn++) {
            O_frag[hn][0] *= alphaA;
            O_frag[hn][1] *= alphaA;
            O_frag[hn][2] *= alphaB;
            O_frag[hn][3] *= alphaB;
        }

        // ---- PV: O += P @ V, m16n8k16 f16 ----
#pragma unroll
        for (int m = 0; m < N_KG; m++) {
            uint32_t ra0 = pack2_f2h(S[2 * m][0], S[2 * m][1]);
            uint32_t ra1 = pack2_f2h(S[2 * m][2], S[2 * m][3]);
            uint32_t ra2 = pack2_f2h(S[2 * m + 1][0], S[2 * m + 1][1]);
            uint32_t ra3 = pack2_f2h(S[2 * m + 1][2], S[2 * m + 1][3]);
            int kr0 = m * 16 + cl;  // Bkv row = (lane%4)*2
#pragma unroll
            for (int hn = 0; hn < N_O; hn++) {
                int ncol = hn * 8 + rl;  // HD col
                uint32_t rb0 = pack2_hh(V_cur[(kr0) * KVSTRIDE + ncol], V_cur[(kr0 + 1) * KVSTRIDE + ncol]);
                uint32_t rb1 = pack2_hh(V_cur[(kr0 + 8) * KVSTRIDE + ncol], V_cur[(kr0 + 9) * KVSTRIDE + ncol]);
                float o0 = O_frag[hn][0], o1 = O_frag[hn][1], o2 = O_frag[hn][2], o3 = O_frag[hn][3];
#if __CUDA_ARCH__ >= 1200
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                    : "=f"(o0), "=f"(o1), "=f"(o2), "=f"(o3)
                    : "r"(ra0), "r"(ra1), "r"(ra2), "r"(ra3), "r"(rb0), "r"(rb1), "f"(o0), "f"(o1), "f"(o2),
                      "f"(o3));
#endif
                O_frag[hn][0] = o0;
                O_frag[hn][1] = o1;
                O_frag[hn][2] = o2;
                O_frag[hn][3] = o3;
            }
        }
        __syncthreads();
    }

    // ---- normalize by row sum and write O ----
    float invA = (lA > 0.f) ? (1.0f / lA) : 0.f;
    float invB = (lB > 0.f) ? (1.0f / lB) : 0.f;
    int rowA = warp_id * 16 + rl;       // local row (O_ptr already at q_start)
    int rowB = rowA + 8;
#pragma unroll
    for (int hn = 0; hn < N_O; hn++) {
        int col = hn * 8 + cl;
        if (q_start + rowA < seq_q) {
            O_ptr[(int64_t)rowA * q_row_stride + col] = __float2half(O_frag[hn][0] * invA);
            O_ptr[(int64_t)rowA * q_row_stride + col + 1] = __float2half(O_frag[hn][1] * invA);
        }
        if (q_start + rowB < seq_q) {
            O_ptr[(int64_t)rowB * q_row_stride + col] = __float2half(O_frag[hn][2] * invB);
            O_ptr[(int64_t)rowB * q_row_stride + col + 1] = __float2half(O_frag[hn][3] * invB);
        }
    }
}

static size_t compute_smem_fa2(int head_dim) {
    constexpr int Bq = 128, Bkv = 64;
    const size_t qstride = head_dim + FA2_Q_PAD;    // bytes (bank-conflict pad)
    const size_t kvstride = head_dim + FA2_KV_PAD;  // halfs (bank-conflict pad)
    return (size_t)Bq * qstride * sizeof(uint8_t)        // Q_fp8 (padded)
           + (size_t)2 * Bkv * kvstride * sizeof(half)    // K_buf[2] f16 (double-buffer, padded)
           + (size_t)2 * Bkv * kvstride * sizeof(half);   // V_buf[2] f16 (double-buffer, padded)
}

bool fmha_sm120_fa2_prefill(const Tensor& Q, const Tensor& K, const Tensor& V, Tensor& O, float scale,
                            bool causal, int sliding_window, float softcap, cudaStream_t stream,
                            int q_offset) {
    if (Q.qtype != QType::F16)
        return false;
    const int batch_size = static_cast<int>(Q.shape[0]);
    const int seq_q = static_cast<int>(Q.shape[1]);
    const int n_heads = static_cast<int>(Q.shape[2]);
    const int head_dim = static_cast<int>(Q.shape[3]);
    const int seq_kv = static_cast<int>(K.shape[1]);
    const int n_kv_heads = static_cast<int>(K.shape[2]);

    if (n_kv_heads == 0 || n_heads % n_kv_heads != 0)
        return false;
    if (seq_q == 0 || seq_kv == 0)
        return false;
    if (head_dim != 128)  // first cut: HD=128 only
        return false;

    int device = 0;
    cudaGetDevice(&device);
    int max_smem = 0;
    cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    const size_t smem = compute_smem_fa2(head_dim);
    if (smem > (size_t)max_smem)
        return false;

    constexpr int Bq = 128;
    const int num_q_tiles = (seq_q + Bq - 1) / Bq;
    dim3 grid(num_q_tiles, batch_size * n_heads);
    dim3 block(SM120_WARP_SIZE, SM120_NUM_WARPS);

    auto kern = fmha_sm120_fa2_kernel<128>;
    cudaError_t aerr =
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));
    if (aerr != cudaSuccess)
        return false;

    static bool logged_once = false;
    if (!logged_once) {
        logged_once = true;
        IMP_LOG_INFO("FMHA FA2 register-resident kernel ACTIVE (hd=128, smem=%zu B, seq_q=%d seq_kv=%d)", smem,
                     seq_q, seq_kv);
    }
    kern<<<grid, block, smem, stream>>>(reinterpret_cast<const half*>(Q.data),
                                        reinterpret_cast<const half*>(K.data),
                                        reinterpret_cast<const half*>(V.data),
                                        reinterpret_cast<half*>(O.data), batch_size, seq_q, seq_kv, n_heads,
                                        n_kv_heads, scale, causal, sliding_window, softcap, q_offset);
    return true;
}

}  // namespace imp
