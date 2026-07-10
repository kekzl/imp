// =============================================================================
// attention_fmha_sm120.cu -- Native sm_120 (Consumer Blackwell) Flash Attention 2
// =============================================================================
//
// Flash Attention 2 prefill kernel built on WMMA HMMA fragments
// (mma.sync.m16n8k16.f16 for QK^T / PV; the FP8 variant uses
// mma.sync.m16n8k32.e4m3 for QK^T). This is NOT wgmma/tcgen05/TMEM -- those are
// Hopper- and datacenter-Blackwell-only (sm_90+/sm_100) and do not exist on
// sm_120a (see attention_fmha_sm120.h). Capabilities over a stock FMHA:
//
//   - Supports sliding window
//   - Supports softcap + causal + sliding window combined
//   - Online softmax (running row max/sum), no global materialization of S
//
// Thread organization: 256 threads (8 warps of 32).
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
#include "core/cuda_static_reset.h"
#include "core/logging.h"
#include "runtime/process_diag.h"
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
// On sm_120a these WMMA ops lower to HMMA (mma.sync); there is no wgmma on
// this target. The explicit WMMA tiling gives us control over the softmax
// fusion.
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

            // Step 6: Fused softmax normalize + float->half conversion.
            // SP_half aliases S_tile COMPACTLY: half row r lives in the bytes
            // of float row r/2, so in-place stores clobber float scores other
            // threads have not read yet (row 2r+1's halves land in row r's
            // cols >= Bkv/2 — deterministic even intra-warp — and padding
            // rows skip Steps 1-5 and race ahead zero-filling valid rows'
            // scores, worst at short seq_q; issue #528). Stage this thread's
            // halves in registers, barrier, then store.
            constexpr int CPT = Bkv / TPR;  // columns per thread
            float spv = (l_new > 0.0f) ? (1.0f / l_new) : 0.0f;
            half hbuf[CPT];
#pragma unroll
            for (int i = 0; i < CPT; i++) {
                int c = sm_lane + i * TPR;
                hbuf[i] = __float2half(row_valid ? S_tile[r * Bkv + c] * spv : 0.0f);
            }
            __syncthreads();  // all float reads of S_tile complete before any half write
#pragma unroll
            for (int i = 0; i < CPT; i++)
                SP_half[r * Bkv + sm_lane + i * TPR] = hbuf[i];
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
                        bool causal, int sliding_window, float softcap, cudaStream_t stream, int q_offset) {
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

#define LAUNCH_FMHA_SM120(BQ, HD)                                                                        \
    do {                                                                                                 \
        cudaError_t attr_err = cudaFuncSetAttribute(fmha_sm120_kernel<BQ, HD>,                           \
                                                    cudaFuncAttributeMaxDynamicSharedMemorySize,         \
                                                    static_cast<int>(smem));                             \
        if (attr_err != cudaSuccess) {                                                                   \
            IMP_LOG_WARN("FMHA sm120: cudaFuncSetAttribute failed for Bq=%d HD=%d smem=%zu: %s", BQ, HD, \
                         smem, cudaGetErrorString(attr_err));                                            \
            return false;                                                                                \
        }                                                                                                \
        cudaFuncSetAttribute(fmha_sm120_kernel<BQ, HD>, cudaFuncAttributePreferredSharedMemoryCarveout,  \
                             cudaSharedmemCarveoutMaxShared);                                            \
        fmha_sm120_kernel<BQ, HD><<<grid, block, smem, stream>>>(                                        \
            reinterpret_cast<const half*>(Q.data), reinterpret_cast<const half*>(K.data),                \
            reinterpret_cast<const half*>(V.data), reinterpret_cast<half*>(O.data), batch_size, seq_q,   \
            seq_kv, n_heads, n_kv_heads, scale, causal, sliding_window, softcap, q_offset);              \
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

// Scaled variant (#680 fp8-QK campaign): multiply by 1/s before the convert so
// the operand uses the full e4m3 dynamic range. The raw (unscaled) conversion
// is the #511 quality cliff — ~10% relative score error on real activations.
__device__ __forceinline__ uint32_t cvt_4xfp16_to_4xe4m3_scaled(const half* src, __half2 inv_s) {
    uint32_t result;
    uint16_t lo, hi;
    const __half2* s2 = reinterpret_cast<const __half2*>(src);
    __half2 a = __hmul2(s2[0], inv_s);
    __half2 b = __hmul2(s2[1], inv_s);
    asm volatile("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(lo)
                 : "r"(*reinterpret_cast<uint32_t*>(&a)));
    asm volatile("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(hi)
                 : "r"(*reinterpret_cast<uint32_t*>(&b)));
    result = static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
    return result;
}

// Grid-stride |x| max over a contiguous fp16 buffer (Q or gathered K of one
// chunk — small next to the attention itself). Caller zeroes `amax` first.
__global__ void fa2_amax_fp16_kernel(const half* __restrict__ x, int64_t n, float* __restrict__ amax) {
    float m = 0.f;
    for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < n;
         i += (int64_t)gridDim.x * blockDim.x)
        m = fmaxf(m, fabsf(__half2float(x[i])));
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFF, m, off));
    if ((threadIdx.x & 31) == 0)
        atomicMax(reinterpret_cast<unsigned int*>(amax), __float_as_uint(m));
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

            // In-place float→half compaction: stage in registers + barrier
            // (SP_half row r aliases the bytes of float row r/2 — issue #528,
            // see the f16 kernel above).
            constexpr int CPT = Bkv / TPR;
            float spv = (l_new > 0.0f) ? (1.0f / l_new) : 0.0f;
            half hbuf[CPT];
#pragma unroll
            for (int i = 0; i < CPT; i++) {
                int c = sm_lane + i * TPR;
                hbuf[i] = __float2half(row_valid ? S_tile[r * Bkv + c] * spv : 0.0f);
            }
            __syncthreads();  // all float reads of S_tile complete before any half write
#pragma unroll
            for (int i = 0; i < CPT; i++)
                SP_half[r * Bkv + sm_lane + i * TPR] = hbuf[i];
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
        fmha_sm120_fp8_kernel<BQ, HD><<<grid, block, smem, stream>>>(                                       \
            reinterpret_cast<const half*>(Q.data), reinterpret_cast<const half*>(K.data),                   \
            reinterpret_cast<const half*>(V.data), reinterpret_cast<half*>(O.data), batch_size, seq_q,      \
            seq_kv, n_heads, n_kv_heads, scale, causal, sliding_window, softcap, q_offset);                 \
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
// ldmatrix.x4: one warp instruction loads four 8x8 b16 tiles from smem —
// replaces 4-6 scalar LDS per MMA operand fetch (this kernel was LSU-
// instruction-bound: ncu 2026-06-07 measured 3.4 shared-LDS per tensor op;
// SASS showed 256 LDS.U16 per KV tile for the V fragments alone). Each group
// of 8 lanes supplies the row addresses of one tile (rows stay 16-B aligned
// thanks to the FA2_KV_PAD stride). The non-trans form delivers fragments in
// mma A/B register order for row-major sources; .trans transposes each 8x8
// during delivery — exactly the V[kv][hd] -> B[k][n] fragment turn the PV MMA
// needs, killing the strided 2-byte V loads AND their pack2 PRMTs.
__device__ __forceinline__ void ldsm_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3,
                                        const half* smem_row) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
    uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(smem_row));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(a));
#else
    r0 = r1 = r2 = r3 = 0;
    (void)smem_row;
#endif
}
__device__ __forceinline__ void ldsm_x4_trans(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3,
                                              const half* smem_row) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
    uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(smem_row));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(a));
#else
    r0 = r1 = r2 = r3 = 0;
    (void)smem_row;
#endif
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
constexpr int FA2_Q_PAD = 16;  // extra bytes per Q_fp8 row
constexpr int FA2_KV_PAD = 8;  // extra halfs per K/V row (16 B, cp.async-aligned)

// Out-of-range rows: K left stale (masked out in the score step), V zero-filled
// (P=0 for those cols, but 0*NaN=NaN would poison O, so V must be finite).
template <int HD, int BKV = 64>
__device__ __forceinline__ void prefetch_kv_tile(half* K_dst, half* V_dst, const half* K_ptr,
                                                 const half* V_ptr, int kv_start, int seq_kv,
                                                 int64_t kv_row_stride, int tid, int nthreads) {
    constexpr int Bkv = BKV;
    constexpr int VEC = 8;                  // 8 halfs = 16 B per cp.async
    constexpr int vecs_per_row = HD / VEC;  // 16
    constexpr int total_vecs = Bkv * vecs_per_row;
    constexpr int KVS = HD + FA2_KV_PAD;  // padded smem row stride (halfs)
    for (int vi = tid; vi < total_vecs; vi += nthreads) {
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

// Split K/V loaders for the TWOSLOT pipeline: K and V of a tile load in
// different phases (V under QK compute, next K under PV compute), so each
// needs its own issue point. Same OOR policy as the combined loader.
template <int HD, int BKV = 64>
__device__ __forceinline__ void prefetch_k_tile(half* K_dst, const half* K_ptr, int kv_start, int seq_kv,
                                                int64_t kv_row_stride, int tid, int nthreads) {
    constexpr int VEC = 8;
    constexpr int vecs_per_row = HD / VEC;
    constexpr int total_vecs = BKV * vecs_per_row;
    constexpr int KVS = HD + FA2_KV_PAD;
    for (int vi = tid; vi < total_vecs; vi += nthreads) {
        int r = vi / vecs_per_row;
        int c = (vi % vecs_per_row) * VEC;
        if (kv_start + r < seq_kv)
            cp_async_cg_16(&K_dst[r * KVS + c], K_ptr + (int64_t)(kv_start + r) * kv_row_stride + c);
    }
}

template <int HD, int BKV = 64>
__device__ __forceinline__ void prefetch_v_tile(half* V_dst, const half* V_ptr, int kv_start, int seq_kv,
                                                int64_t kv_row_stride, int tid, int nthreads) {
    constexpr int VEC = 8;
    constexpr int vecs_per_row = HD / VEC;
    constexpr int total_vecs = BKV * vecs_per_row;
    constexpr int KVS = HD + FA2_KV_PAD;
    for (int vi = tid; vi < total_vecs; vi += nthreads) {
        int r = vi / vecs_per_row;
        int c = (vi % vecs_per_row) * VEC;
        if (kv_start + r < seq_kv) {
            cp_async_cg_16(&V_dst[r * KVS + c], V_ptr + (int64_t)(kv_start + r) * kv_row_stride + c);
        } else {
            *reinterpret_cast<uint4*>(&V_dst[r * KVS + c]) = make_uint4(0, 0, 0, 0);
        }
    }
}

// FP16QK=false: Q staged in smem as e4m3, K converted f16->fp8 inline, QK via
// mma.m16n8k32.e4m3 (2x score throughput; quality validated ONLY at long
// context — e4m3 score noise compounds across layers at short seq, #511/#512).
// FP16QK=true: Q staged as f16, K read from f16 smem directly, QK via
// mma.m16n8k16.f16 — half the score throughput, full fp16 accuracy. This is
// the short-sequence variant that replaces the materialized cuBLAS+softmax
// path below fmha_prefill_threshold without the e4m3 quality risk.
// Softmax, masking and PV are shared — one grammar of truth for the math.
// BKV=32 halves the KV double-buffer (~70 KB → ~35 KB smem) so 2 CTAs/SM fit —
// the occupancy "smem surgery" lever from #597 for the grid-underfill band.
// TWOSLOT (#597 second cut): keeps the FULL Bkv=64 tile at the same ~35 KB by
// replacing the K/V double-buffer with a two-slot rotation — one K slot, one
// V slot. K and V of a tile load in different phases (V_j under QK_j's MMAs,
// K_{j+1} under PV_j's), so the cp.async overlap survives with half the smem:
// 2 CTAs/SM AND half the per-tile online-softmax/barrier overhead of BKV=32.
// PVF16: f16-accumulate the PV MMA and keep the O accumulator as packed half2
// (#667 follow-up to the QK f16acc). The online-softmax O rows are convex
// combinations of V rows (P weights in [0,1], normalized by lA/lB at the end),
// so the accumulator magnitude is bounded by max|V| — range is safe; the
// rescale-and-add rounding is what the PPL gate has to clear. Cuts the PV MMAs
// from the 1/4-rate f32-acc class to full rate AND halves the O-fragment
// register footprint (N_O*4 floats -> N_O*2 b32), the dominant per-thread
// register cost of the Bq=128 band. Requires FP16QK.
// FP8SCALED (#680): amax-scaled e4m3 conversion for the fp8-QK path. Q and K
// convert as x*(448/amax) (full e4m3 range, no saturation/mantissa cliff) and
// the score scale absorbs (amax_q*amax_k/448^2). d_amax = device floats
// {amax_q, amax_k} produced by fa2_amax_fp16_kernel just before launch.
template <int Bq, int HD, bool FP16QK = false, bool F16ACC = false, int BKV = 64, bool TWOSLOT = false,
          bool PVF16 = false, bool FP8SCALED = false>
__global__ void fmha_sm120_fa2_kernel(const half* __restrict__ Q, const half* __restrict__ K,
                                      const half* __restrict__ V, half* __restrict__ O, int batch_size,
                                      int seq_q, int seq_kv, int n_heads, int n_kv_heads, float scale,
                                      bool causal, int sliding_window, float softcap, int q_offset,
                                      const float* __restrict__ d_amax = nullptr,
                                      const int* __restrict__ d_kv_len = nullptr) {
    constexpr int Bkv = BKV;
    static_assert(BKV % 16 == 0 && (BKV / 8) % 2 == 0, "QK n-pair loop and PV K-groups need BKV % 16 == 0");
    constexpr int head_dim = HD;
    // Each warp owns one 16-row tile (mma m16) → warps = Bq/16, threads = warps*32.
    // Bq=128 → 8 warps/256 thr (large seq, max latency-hiding); Bq=64 → 4 warps/128 thr
    // (small seq: 2× more q-tiles → more CTAs → fills the 170 SMs instead of one short wave).
    constexpr int NWARPS = Bq / 16;
    constexpr int NTHREADS = NWARPS * 32;
    constexpr int N_S = Bkv / 8;    // QK N-tiles (8-col groups) per row tile = 8
    constexpr int N_O = HD / 8;     // PV N-tiles (8 HD-col groups)
    constexpr int N_KG = Bkv / 16;  // PV K-groups (16 Bkv-col groups) = 4
    constexpr int KC = HD / 32;     // QK k-chunks (32 each)

    const int tile_q = blockIdx.x;
    const int batch_head = blockIdx.y;
    const int batch_idx = batch_head / n_heads;
    const int head_idx = batch_head % n_heads;
    const int kv_head = head_idx / (n_heads / n_kv_heads);

    // Graph-captured verify (#847): the real KV length lives on device and the
    // baked seq_kv is only the buffer capacity (single-sequence chunks, so the
    // batch stride it feeds is inert). q_offset follows from the chunked
    // continuation invariant seq_kv == q_offset + seq_q.
    if (d_kv_len != nullptr) {
        seq_kv = __ldg(d_kv_len);
        q_offset = seq_kv - seq_q;
    }

    const int lane = threadIdx.x;     // 0..31
    const int warp_id = threadIdx.y;  // 0..7  → this warp's row tile
    const int tid = lane + warp_id * 32;
    const int q_start = tile_q * Bq;
    const int row_lo = warp_id * 16 + lane / 4;  // local row within Bq for d0/d1
    const int rl = lane / 4;                     // row in 16-tile (lo)
    const int cl = (lane % 4) * 2;               // col base in 8/16-tile

    const int64_t q_row_stride = (int64_t)n_heads * head_dim;
    const int64_t kv_row_stride = (int64_t)n_kv_heads * head_dim;
    const half* Q_ptr = Q + (int64_t)batch_idx * seq_q * q_row_stride + (int64_t)q_start * q_row_stride +
                        (int64_t)head_idx * head_dim;
    const half* K_ptr = K + (int64_t)batch_idx * seq_kv * kv_row_stride + (int64_t)kv_head * head_dim;
    const half* V_ptr = V + (int64_t)batch_idx * seq_kv * kv_row_stride + (int64_t)kv_head * head_dim;
    half* O_ptr = O + (int64_t)batch_idx * seq_q * q_row_stride + (int64_t)q_start * q_row_stride +
                  (int64_t)head_idx * head_dim;

    // Padded smem row strides (see FA2_Q_PAD / FA2_KV_PAD — bank-conflict relief).
    // fp16 mode keeps Q in registers (no smem tile at all); fp8 mode stages
    // e4m3 bytes with the byte-stride layout.
    constexpr int QSTRIDE = head_dim + FA2_Q_PAD;    // bytes (fp8 mode only)
    constexpr int KVSTRIDE = head_dim + FA2_KV_PAD;  // K/V row stride in HALFS
    constexpr size_t Q_SMEM_BYTES = FP16QK ? 0 : (size_t)Bq * QSTRIDE * sizeof(uint8_t);

    extern __shared__ char smem[];
    uint8_t* Q_fp8 = reinterpret_cast<uint8_t*>(smem);           // fp8 mode view
    // Double-buffer mode: K_buf/V_buf are [2][Bkv*KVSTRIDE]. TWOSLOT mode:
    // one slot each — half the footprint, rotation handled by load phases.
    constexpr int KV_SLOTS = TWOSLOT ? 1 : 2;
    half* K_buf = reinterpret_cast<half*>(smem + Q_SMEM_BYTES);  // [KV_SLOTS][Bkv*KVSTRIDE] f16
    half* V_buf = K_buf + KV_SLOTS * Bkv * KVSTRIDE;             // [KV_SLOTS][Bkv*KVSTRIDE] f16

    // ---- load Q once: fp8 mode converts 4 halves → 4 e4m3 into smem; fp16
    // mode skips smem entirely — each lane pulls its loop-invariant
    // A-fragment words straight from global into registers below (Q is read
    // exactly once per CTA, and dropping the staging tile frees enough smem
    // for the 8-warp Bq=128 config: the kernel is latency-bound at 4 warps,
    // 1 warp/scheduler). ----
    // FP8SCALED operand scales (uniform per-thread loads; amax==0 -> identity).
    float fp8_sq = 1.0f, fp8_sk = 1.0f;
    __half2 fp8_inv_sq2 = __float2half2_rn(1.0f);
    __half2 fp8_inv_sk2 = __float2half2_rn(1.0f);
    if constexpr (FP8SCALED) {
        const float aq = d_amax ? d_amax[0] : 0.0f;
        const float ak = d_amax ? d_amax[1] : 0.0f;
        if (aq > 0.0f) {
            fp8_sq = aq / 448.0f;
            fp8_inv_sq2 = __float2half2_rn(448.0f / aq);
        }
        if (ak > 0.0f) {
            fp8_sk = ak / 448.0f;
            fp8_inv_sk2 = __float2half2_rn(448.0f / ak);
        }
    }
    (void)fp8_sq;
    (void)fp8_sk;
    if constexpr (!FP16QK) {
        const int total_vec4 = (Bq * head_dim) / 4;
        for (int vi = tid; vi < total_vec4; vi += NTHREADS) {
            int i = vi * 4;
            int r = i / head_dim;
            int d = i % head_dim;  // multiple of 4 → uint32-aligned into the padded row
            uint32_t* dst = reinterpret_cast<uint32_t*>(Q_fp8 + r * QSTRIDE + d);
            if (q_start + r >= seq_q) {
                *dst = 0;
            } else if constexpr (FP8SCALED) {
                *dst = cvt_4xfp16_to_4xe4m3_scaled(&Q_ptr[(int64_t)r * q_row_stride + d], fp8_inv_sq2);
            } else {
                *dst = cvt_4xfp16_to_4xe4m3(&Q_ptr[(int64_t)r * q_row_stride + d]);
            }
        }
    }

    // ---- per-warp register state ----
    // PVF16: O held as packed half2 — reg [0] = (row rl, cols cl,cl+1),
    // reg [1] = (row rl+8, same cols), matching the m16n8k16 f16-D layout.
    static_assert(!PVF16 || FP16QK, "PVF16 requires the fp16-qk path");
    float O_frag[PVF16 ? 1 : N_O][4];
    uint32_t O_h2[PVF16 ? N_O : 1][2];
    if constexpr (PVF16) {
#pragma unroll
        for (int hn = 0; hn < N_O; hn++)
            O_h2[hn][0] = O_h2[hn][1] = 0u;
    } else {
#pragma unroll
        for (int hn = 0; hn < N_O; hn++)
            O_frag[hn][0] = O_frag[hn][1] = O_frag[hn][2] = O_frag[hn][3] = 0.0f;
    }
    float mA = -FLT_MAX, mB = -FLT_MAX, lA = 0.0f, lB = 0.0f;

    int num_kv_tiles, first_kv_tile;
    compute_kv_tile_bounds(q_start, Bq, Bkv, seq_q, seq_kv, causal, sliding_window, first_kv_tile,
                           num_kv_tiles, q_offset);
    // prologue: kick off the first KV tile's load into buffer slot 0.
    // TWOSLOT: K only — V_0 issues at the top of the first loop iteration so
    // its load overlaps QK_0's MMAs.
    if (first_kv_tile < num_kv_tiles) {
        if constexpr (TWOSLOT)
            prefetch_k_tile<head_dim, Bkv>(K_buf, K_ptr, first_kv_tile * Bkv, seq_kv, kv_row_stride, tid,
                                           NTHREADS);
        else
            prefetch_kv_tile<head_dim, Bkv>(K_buf, V_buf, K_ptr, V_ptr, first_kv_tile * Bkv, seq_kv,
                                            kv_row_stride, tid, NTHREADS);
    }
    cp_async_commit();
    __syncthreads();  // Q_fp8 (produced above) visible before QK reads it

    // fp16 mode: the Q A-fragments are loop-invariant — load them once,
    // straight from global memory into registers (no smem staging; rows past
    // seq_q are zero-filled like the old staging loop did). Register layout
    // matches the a0..a3 the QK MMA consumed: a0=(rl,cl) a1=(rl+8,cl)
    // a2=(rl,cl+8) a3=(rl+8,cl+8).
    uint32_t a_frag[FP16QK ? (HD / 16) : 1][4];
    if constexpr (FP16QK) {
        const int r0 = warp_id * 16 + rl;
        const int r1 = r0 + 8;
        const bool v0 = (q_start + r0) < seq_q;
        const bool v1 = (q_start + r1) < seq_q;
        const half* q0 = Q_ptr + (int64_t)r0 * q_row_stride;
        const half* q1 = Q_ptr + (int64_t)r1 * q_row_stride;
#pragma unroll
        for (int k = 0; k < HD / 16; k++) {
            const int d = k * 16 + cl;
            a_frag[k][0] = v0 ? *reinterpret_cast<const uint32_t*>(q0 + d) : 0u;
            a_frag[k][1] = v1 ? *reinterpret_cast<const uint32_t*>(q1 + d) : 0u;
            a_frag[k][2] = v0 ? *reinterpret_cast<const uint32_t*>(q0 + d + 8) : 0u;
            a_frag[k][3] = v1 ? *reinterpret_cast<const uint32_t*>(q1 + d + 8) : 0u;
        }
    }

    for (int j = first_kv_tile; j < num_kv_tiles; j++) {
        const int slot = TWOSLOT ? 0 : ((j - first_kv_tile) & 1);
        const int kv_start = j * Bkv;

        if constexpr (TWOSLOT) {
            // Issue V_j now — it loads while QK_j's MMAs run below. Then wait
            // for K_j (the older commit group; FIFO completion) before QK.
            prefetch_v_tile<head_dim, Bkv>(V_buf, V_ptr, kv_start, seq_kv, kv_row_stride, tid, NTHREADS);
            cp_async_commit();
            cp_async_wait_group<1>();  // K_j landed; V_j still in flight
            __syncthreads();           // K_j visible to all threads
        } else {
            // prefetch tile j+1 into the alternate slot, overlapping this tile's compute
            if (j + 1 < num_kv_tiles) {
                const int nslot = slot ^ 1;
                prefetch_kv_tile<head_dim, Bkv>(K_buf + nslot * Bkv * KVSTRIDE,
                                                V_buf + nslot * Bkv * KVSTRIDE, K_ptr, V_ptr, (j + 1) * Bkv,
                                                seq_kv, kv_row_stride, tid, NTHREADS);
                cp_async_commit();
                cp_async_wait_group<1>();  // this tile (slot) landed; tile j+1 still in flight
            } else {
                cp_async_wait_group<0>();
            }
            __syncthreads();  // this tile's K/V fully landed for all threads
        }

        const half* K_cur = K_buf + slot * Bkv * KVSTRIDE;
        const half* V_cur = V_buf + slot * Bkv * KVSTRIDE;

        // ---- QK: S[n] = Q(warp rows) @ K[n-tile]^T ----
        // fp8 mode: m16n8k32.e4m3 (K cvt f16→fp8 inline). fp16 mode: m16n8k16.f16.
        float S[N_S][4];
        if constexpr (FP16QK) {
            // n-tiles processed in pairs: one ldmatrix.x4 fetches the K
            // B-fragments (b0,b1) of BOTH tiles — lanes 0-7 / 8-15 address
            // {rows n*8+0..7, k-lo} / {same rows, k-hi}, lanes 16-31 the same
            // for tile n+1. A comes from the hoisted loop-invariant a_frag.
#pragma unroll
            for (int n = 0; n < N_S; n += 2) {
                const int b_row = (n + (lane >> 4)) * 8 + (lane & 7);
                const int b_khalf = ((lane >> 3) & 1) << 3;
                const half* krow = K_cur + b_row * KVSTRIDE + b_khalf;
                // software pipeline: fetch k+1's B fragments while k's MMAs
                // run — at 1-2 warps per scheduler the LDSM→HMMA dependency
                // latency is otherwise exposed (short_scoreboard-bound).
                uint32_t b0, b1, c0, c1;
                ldsm_x4(b0, b1, c0, c1, krow);
                if constexpr (F16ACC) {
                    // f16-accumulate QK^T (#597, opt-in attention.fa2_f16acc).
                    // GeForce sm_120 runs f16-src/f32-acc HMMA at 1/4 rate (253
                    // of 838 TFLOPS, #606); f16-acc lifts the score MMA to the
                    // full-rate class (+3-4% pp2048/pp4096 NVFP4, +0.37% PPL on
                    // Qwen3-14B). Scores are softmaxed immediately so the
                    // reduced accumulate precision is low-risk. Accumulators are
                    // 2 packed-half2 registers (also halves the score reg foot-
                    // print). PV stays f32-acc (online O sum needs the range).
                    uint32_t dA[2] = {0u, 0u};
                    uint32_t dB[2] = {0u, 0u};
#pragma unroll
                    for (int k = 0; k < HD / 16; k++) {
                        uint32_t nb0, nb1, nc0, nc1;
                        if (k + 1 < HD / 16)
                            ldsm_x4(nb0, nb1, nc0, nc1, krow + (k + 1) * 16);
#if __CUDA_ARCH__ >= 1200
                        asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                            "{%0,%1},{%2,%3,%4,%5},{%6,%7},{%0,%1};\n"
                            : "+r"(dA[0]), "+r"(dA[1])
                            : "r"(a_frag[k][0]), "r"(a_frag[k][1]), "r"(a_frag[k][2]),
                              "r"(a_frag[k][3]), "r"(b0), "r"(b1));
                        asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                            "{%0,%1},{%2,%3,%4,%5},{%6,%7},{%0,%1};\n"
                            : "+r"(dB[0]), "+r"(dB[1])
                            : "r"(a_frag[k][0]), "r"(a_frag[k][1]), "r"(a_frag[k][2]),
                              "r"(a_frag[k][3]), "r"(c0), "r"(c1));
#endif
                        if (k + 1 < HD / 16) {
                            b0 = nb0;
                            b1 = nb1;
                            c0 = nc0;
                            c1 = nc1;
                        }
                    }
                    const half2 a01 = *reinterpret_cast<const half2*>(&dA[0]);
                    const half2 a23 = *reinterpret_cast<const half2*>(&dA[1]);
                    const half2 b01 = *reinterpret_cast<const half2*>(&dB[0]);
                    const half2 b23 = *reinterpret_cast<const half2*>(&dB[1]);
                    S[n][0] = __low2float(a01);
                    S[n][1] = __high2float(a01);
                    S[n][2] = __low2float(a23);
                    S[n][3] = __high2float(a23);
                    S[n + 1][0] = __low2float(b01);
                    S[n + 1][1] = __high2float(b01);
                    S[n + 1][2] = __low2float(b23);
                    S[n + 1][3] = __high2float(b23);
                } else {
                    float dA[4] = {0.f, 0.f, 0.f, 0.f};
                    float dB[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
                    for (int k = 0; k < HD / 16; k++) {
                        uint32_t nb0, nb1, nc0, nc1;
                        if (k + 1 < HD / 16)
                            ldsm_x4(nb0, nb1, nc0, nc1, krow + (k + 1) * 16);
#if __CUDA_ARCH__ >= 1200
                        asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                            "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                            : "=f"(dA[0]), "=f"(dA[1]), "=f"(dA[2]), "=f"(dA[3])
                            : "r"(a_frag[k][0]), "r"(a_frag[k][1]), "r"(a_frag[k][2]),
                              "r"(a_frag[k][3]), "r"(b0), "r"(b1), "f"(dA[0]), "f"(dA[1]), "f"(dA[2]),
                              "f"(dA[3]));
                        asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                            "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                            : "=f"(dB[0]), "=f"(dB[1]), "=f"(dB[2]), "=f"(dB[3])
                            : "r"(a_frag[k][0]), "r"(a_frag[k][1]), "r"(a_frag[k][2]),
                              "r"(a_frag[k][3]), "r"(c0), "r"(c1), "f"(dB[0]), "f"(dB[1]), "f"(dB[2]),
                              "f"(dB[3]));
#endif
                        if (k + 1 < HD / 16) {
                            b0 = nb0;
                            b1 = nb1;
                            c0 = nc0;
                            c1 = nc1;
                        }
                    }
#pragma unroll
                    for (int e = 0; e < 4; e++) {
                        S[n][e] = dA[e];
                        S[n + 1][e] = dB[e];
                    }
                }
            }
        } else {
#pragma unroll
        for (int n = 0; n < N_S; n++) {
            float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
            {
#pragma unroll
                for (int k = 0; k < KC; k++) {
                    const uint8_t* qb = Q_fp8 + warp_id * 16 * QSTRIDE + k * 32;
                    uint32_t a0 = *reinterpret_cast<const uint32_t*>(qb + rl * QSTRIDE + cl * 2);
                    uint32_t a1 = *reinterpret_cast<const uint32_t*>(qb + rl * QSTRIDE + cl * 2 + 16);
                    uint32_t a2 = *reinterpret_cast<const uint32_t*>(qb + (rl + 8) * QSTRIDE + cl * 2);
                    uint32_t a3 = *reinterpret_cast<const uint32_t*>(qb + (rl + 8) * QSTRIDE + cl * 2 + 16);
                    const half* kb = K_cur + n * 8 * KVSTRIDE + k * 32;
                    uint32_t b0, b1;
                    if constexpr (FP8SCALED) {
                        b0 = cvt_4xfp16_to_4xe4m3_scaled(kb + rl * KVSTRIDE + cl * 2, fp8_inv_sk2);
                        b1 = cvt_4xfp16_to_4xe4m3_scaled(kb + rl * KVSTRIDE + cl * 2 + 16, fp8_inv_sk2);
                    } else {
                        b0 = cvt_4xfp16_to_4xe4m3(kb + rl * KVSTRIDE + cl * 2);
                        b1 = cvt_4xfp16_to_4xe4m3(kb + rl * KVSTRIDE + cl * 2 + 16);
                    }
#if __CUDA_ARCH__ >= 1200
                    asm volatile(
                        "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(d0), "f"(d1), "f"(d2),
                          "f"(d3));
#endif
                }
            }
            S[n][0] = d0;
            S[n][1] = d1;
            S[n][2] = d2;
            S[n][3] = d3;
        }
        }

        if constexpr (TWOSLOT) {
            // QK_j is done with K_buf. Wait for V_j (needed by PV below), and
            // barrier so every warp's QK reads retired — then K_buf is free
            // and K_{j+1} can stream in under the softmax + PV phase.
            cp_async_wait_group<0>();
            __syncthreads();
            if (j + 1 < num_kv_tiles) {
                prefetch_k_tile<head_dim, Bkv>(K_buf, K_ptr, (j + 1) * Bkv, seq_kv, kv_row_stride, tid,
                                               NTHREADS);
                cp_async_commit();
            }
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
                    if constexpr (FP8SCALED)
                        v *= fp8_sq * fp8_sk;
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
        if constexpr (PVF16) {
            const __half2 hA = __float2half2_rn(alphaA);
            const __half2 hB = __float2half2_rn(alphaB);
#pragma unroll
            for (int hn = 0; hn < N_O; hn++) {
                __half2& r0 = *reinterpret_cast<__half2*>(&O_h2[hn][0]);
                __half2& r1 = *reinterpret_cast<__half2*>(&O_h2[hn][1]);
                r0 = __hmul2(r0, hA);
                r1 = __hmul2(r1, hB);
            }
        } else {
#pragma unroll
            for (int hn = 0; hn < N_O; hn++) {
                O_frag[hn][0] *= alphaA;
                O_frag[hn][1] *= alphaA;
                O_frag[hn][2] *= alphaB;
                O_frag[hn][3] *= alphaB;
            }
        }

        // ---- PV: O += P @ V, m16n8k16 f16 ----
        // V B-fragments via ldmatrix.x4.trans over hn-pairs: lanes 0-7 / 8-15
        // address V rows m*16+0..7 / +8..15 at HD col hn*8, lanes 16-31 the
        // same rows at col (hn+1)*8. .trans turns each row-major 8x8 V block
        // into the col-major B fragment (r0/r1 = rb0/rb1 of hn, r2/r3 of
        // hn+1) — was 4 strided LDS.U16 + 2 PRMT packs per MMA.
#pragma unroll
        for (int m = 0; m < N_KG; m++) {
            uint32_t ra0 = pack2_f2h(S[2 * m][0], S[2 * m][1]);
            uint32_t ra1 = pack2_f2h(S[2 * m][2], S[2 * m][3]);
            uint32_t ra2 = pack2_f2h(S[2 * m + 1][0], S[2 * m + 1][1]);
            uint32_t ra3 = pack2_f2h(S[2 * m + 1][2], S[2 * m + 1][3]);
            const half* vrow = V_cur + (m * 16 + (lane & 15)) * KVSTRIDE + ((lane >> 4) << 3);
            // software pipeline: V fragments for hn+2 fetched under hn's MMAs
            uint32_t rb0, rb1, rc0, rc1;
            ldsm_x4_trans(rb0, rb1, rc0, rc1, vrow);
#pragma unroll
            for (int hn = 0; hn < N_O; hn += 2) {
                uint32_t sb0, sb1, sc0, sc1;
                if (hn + 2 < N_O)
                    ldsm_x4_trans(sb0, sb1, sc0, sc1, vrow + (hn + 2) * 8);
                if constexpr (PVF16) {
#if __CUDA_ARCH__ >= 1200
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                        "{%0,%1},{%2,%3,%4,%5},{%6,%7},{%0,%1};\n"
                        : "+r"(O_h2[hn][0]), "+r"(O_h2[hn][1])
                        : "r"(ra0), "r"(ra1), "r"(ra2), "r"(ra3), "r"(rb0), "r"(rb1));
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                        "{%0,%1},{%2,%3,%4,%5},{%6,%7},{%0,%1};\n"
                        : "+r"(O_h2[hn + 1][0]), "+r"(O_h2[hn + 1][1])
                        : "r"(ra0), "r"(ra1), "r"(ra2), "r"(ra3), "r"(rc0), "r"(rc1));
#endif
                } else {
                float o0 = O_frag[hn][0], o1 = O_frag[hn][1], o2 = O_frag[hn][2], o3 = O_frag[hn][3];
                float p0 = O_frag[hn + 1][0], p1 = O_frag[hn + 1][1], p2 = O_frag[hn + 1][2],
                      p3 = O_frag[hn + 1][3];
#if __CUDA_ARCH__ >= 1200
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                    : "=f"(o0), "=f"(o1), "=f"(o2), "=f"(o3)
                    : "r"(ra0), "r"(ra1), "r"(ra2), "r"(ra3), "r"(rb0), "r"(rb1), "f"(o0), "f"(o1), "f"(o2),
                      "f"(o3));
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                    : "=f"(p0), "=f"(p1), "=f"(p2), "=f"(p3)
                    : "r"(ra0), "r"(ra1), "r"(ra2), "r"(ra3), "r"(rc0), "r"(rc1), "f"(p0), "f"(p1), "f"(p2),
                      "f"(p3));
#endif
                O_frag[hn][0] = o0;
                O_frag[hn][1] = o1;
                O_frag[hn][2] = o2;
                O_frag[hn][3] = o3;
                O_frag[hn + 1][0] = p0;
                O_frag[hn + 1][1] = p1;
                O_frag[hn + 1][2] = p2;
                O_frag[hn + 1][3] = p3;
                }
                if (hn + 2 < N_O) {
                    rb0 = sb0;
                    rb1 = sb1;
                    rc0 = sc0;
                    rc1 = sc1;
                }
            }
        }
        __syncthreads();
    }

    // ---- normalize by row sum and write O ----
    float invA = (lA > 0.f) ? (1.0f / lA) : 0.f;
    float invB = (lB > 0.f) ? (1.0f / lB) : 0.f;
    int rowA = warp_id * 16 + rl;  // local row (O_ptr already at q_start)
    int rowB = rowA + 8;
#pragma unroll
    for (int hn = 0; hn < N_O; hn++) {
        int col = hn * 8 + cl;
        float oA0, oA1, oB0, oB1;
        if constexpr (PVF16) {
            const float2 fA = __half22float2(*reinterpret_cast<const __half2*>(&O_h2[hn][0]));
            const float2 fB = __half22float2(*reinterpret_cast<const __half2*>(&O_h2[hn][1]));
            oA0 = fA.x, oA1 = fA.y, oB0 = fB.x, oB1 = fB.y;
        } else {
            oA0 = O_frag[hn][0], oA1 = O_frag[hn][1], oB0 = O_frag[hn][2], oB1 = O_frag[hn][3];
        }
        if (q_start + rowA < seq_q) {
            O_ptr[(int64_t)rowA * q_row_stride + col] = __float2half(oA0 * invA);
            O_ptr[(int64_t)rowA * q_row_stride + col + 1] = __float2half(oA1 * invA);
        }
        if (q_start + rowB < seq_q) {
            O_ptr[(int64_t)rowB * q_row_stride + col] = __float2half(oB0 * invB);
            O_ptr[(int64_t)rowB * q_row_stride + col + 1] = __float2half(oB1 * invB);
        }
    }
}

static size_t compute_smem_fa2(int Bq, int head_dim, bool fp16_qk, int Bkv, bool twoslot = false) {
    const size_t kvstride = head_dim + FA2_KV_PAD;  // halfs (bank-conflict pad)
    // Q tile: fp8 mode stages e4m3 bytes (FA2_Q_PAD); fp16 mode keeps Q in
    // registers (loop-invariant A-fragments loaded once from global) — no
    // smem tile at all, which is what lets Bq=128 (8 warps) fit.
    const size_t q_bytes = fp16_qk ? 0 : (size_t)Bq * (head_dim + FA2_Q_PAD) * sizeof(uint8_t);
    const size_t slots = twoslot ? 1 : 2;  // TWOSLOT: one K + one V slot
    return q_bytes + slots * Bkv * kvstride * sizeof(half)  // K_buf f16 (padded)
           + slots * Bkv * kvstride * sizeof(half);         // V_buf f16 (padded)
}

// FP8SCALED per-chunk operand amax buffer (persistent 2-float device buffer,
// lazily created below; file-scope so the reset hook can free it).
static float* s_d_amax = nullptr;

// Pre-cudaDeviceReset hook (see core/cuda_static_reset.h).
void fmha_sm120_reset_static_cuda_state() {
    if (s_d_amax) {
        (void)cudaFree(s_d_amax);
        s_d_amax = nullptr;
    }
}

bool fmha_sm120_fa2_prefill(const Tensor& Q, const Tensor& K, const Tensor& V, Tensor& O, float scale,
                            bool causal, int sliding_window, float softcap, cudaStream_t stream, int q_offset,
                            bool fp16_qk, const int* d_kv_len) {
    if (Q.qtype != QType::F16)
        return false;
    if (d_kv_len != nullptr && !fp16_qk)
        return false;  // device-length replay is wired for the f16-QK path only
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
    // HD=128 is the tuned mainline. HD=256 (Qwen3.6 hybrids, gemma-class) is a
    // stage-1 port behind attention.fa2_hd256 (default ON since #932): fp16-qk only,
    // fixed Bq=64/Bkv=64/TWOSLOT (double-buffer at HD=256 needs 135 KB smem >
    // the 99 KB opt-in; TWOSLOT fits at 67.6 KB). Register pressure doubles
    // with HD (a_frag + O accumulator scale linearly), so the HD=256 instances
    // ride the f16acc/pv_f16 variants where possible — see the launch table.
    if (head_dim != 128 && !(head_dim == 256 && fp16_qk && imp::process_diag_fa2_hd256()))
        return false;

    int device = 0;
    cudaGetDevice(&device);
    int max_smem = 0;
    cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);

    // Bq/Bkv selection: grid = (q_tiles × batch×heads) and per-SM residency must
    // together fill the SMs. Three bands (fp16qk path; #597 occupancy surgery):
    //  - blocks_128 ≥ sm_count: Bq=128/Bkv=64 — long-ctx config, grid fills at
    //    1 CTA/SM with 8 warps of latency-hiding and the deepest cp.async overlap.
    //  - sm_count/2 ≤ blocks_128 < sm_count: the profiled 0.75-wave underfill band
    //    (chunked prefill: ~4 q-tiles × ~40 heads < 170 SMs). Bq=64 doubles the
    //    grid and Bkv=32 halves the KV double-buffer (~70 KB → ~35 KB) so 2 CTAs/SM
    //    become resident: same 8 warps/SM, but every SM gets work (~2 waves instead
    //    of 0.75) and barriers split into two independent 4-warp scopes. Bq=128
    //    can't use the freed smem — 2 CTAs × 8 warps × ~175 regs exceeds the
    //    64K-register file, so the 2-CTA config requires the 4-warp CTA.
    //  - blocks_128 < sm_count/2: even the Bq=64 grid stays below the SM count, so
    //    2-CTA residency never materializes — keep Bkv=64 (deeper cp.async overlap,
    //    half the per-tile softmax/barrier overhead).
    const long blocks_128 = (long)((seq_q + 127) / 128) * batch_size * n_heads;

    // f16-acc QK^T (#597) only applies to the fp16_qk path (the fp8 path keeps
    // its f32 accumulate). Opt-in: +3-4% pp2048/pp4096 NVFP4 for +0.37% PPL.
    const bool f16acc = fp16_qk && imp::process_diag_fa2_f16acc();
    // PV f16-accumulate rides on the f16acc path (full-rate PV MMA + halved
    // O-fragment registers); attention.fa2_pv_f16acc, default on.
    const bool pv_f16 = f16acc && imp::process_diag_fa2_pv_f16acc();
    int Bq, Bkv;
    bool twoslot = false;
    bool fp8_scaled = false;
    decltype(&fmha_sm120_fa2_kernel<128, 128>) kern;
    if (head_dim == 256) {
        // Stage-1 HD=256 port (fp16-qk only, gated above): one configuration.
        // Bq=64 → 4 warps; TWOSLOT keeps the full Bkv=64 tile at 67.6 KB smem
        // (double-buffer would need 135 KB). Per-thread registers ~2× the
        // HD=128 profile (a_frag 32→64 regs, O f32 64→128 / pv-f16 32→64), so
        // pv_f16 is strongly preferred; the f32-acc variant exists for A/B
        // but is expected to spill.
        Bq = 64, Bkv = 64, twoslot = true;
        kern = pv_f16   ? fmha_sm120_fa2_kernel<64, 256, true, true, 64, true, true>
               : f16acc ? fmha_sm120_fa2_kernel<64, 256, true, true, 64, true>
                        : fmha_sm120_fa2_kernel<64, 256, true, false, 64, true>;
    } else if (fp16_qk) {
        if (blocks_128 >= (long)sm_count) {
            Bq = 128, Bkv = 64;
            kern = pv_f16   ? fmha_sm120_fa2_kernel<128, 128, true, true, 64, false, true>
                   : f16acc ? fmha_sm120_fa2_kernel<128, 128, true, true, 64>
                            : fmha_sm120_fa2_kernel<128, 128, true, false, 64>;
        } else if (blocks_128 >= (long)(sm_count / 2)) {
            // Underfill band (#597): Bq=64 doubles the grid and TWOSLOT halves
            // the KV smem (~70 KB → ~35 KB) at the FULL Bkv=64 tile, so 2
            // CTAs/SM become resident. Supersedes the Bkv=32 double-buffer
            // (same residency, but half the online-softmax rescales and fewer
            // barriers per KV row).
            Bq = 64, Bkv = 64, twoslot = true;
            kern = pv_f16   ? fmha_sm120_fa2_kernel<64, 128, true, true, 64, true, true>
                   : f16acc ? fmha_sm120_fa2_kernel<64, 128, true, true, 64, true>
                            : fmha_sm120_fa2_kernel<64, 128, true, false, 64, true>;
        } else {
            Bq = 64, Bkv = 64;
            kern = pv_f16   ? fmha_sm120_fa2_kernel<64, 128, true, true, 64, false, true>
                   : f16acc ? fmha_sm120_fa2_kernel<64, 128, true, true, 64>
                            : fmha_sm120_fa2_kernel<64, 128, true, false, 64>;
        }
    } else {
        const bool use_bq64 = blocks_128 < (long)sm_count;
        Bq = use_bq64 ? 64 : 128;
        Bkv = 64;
        // #680 fp8-QK campaign: amax-scaled e4m3 conversion (the raw variant
        // is the #511 quality cliff and stays the default for A/B).
        if (imp::process_diag_fp8_qk_scaled()) {
            fp8_scaled = true;
            kern = use_bq64 ? fmha_sm120_fa2_kernel<64, 128, false, false, 64, false, false, true>
                            : fmha_sm120_fa2_kernel<128, 128, false, false, 64, false, false, true>;
        } else {
            kern = use_bq64 ? fmha_sm120_fa2_kernel<64, 128> : fmha_sm120_fa2_kernel<128, 128>;
        }
    }

    const size_t smem = compute_smem_fa2(Bq, head_dim, fp16_qk, Bkv, twoslot);
    if (smem > (size_t)max_smem)
        return false;

    const int num_q_tiles = (seq_q + Bq - 1) / Bq;
    dim3 grid(num_q_tiles, batch_size * n_heads);
    dim3 block(SM120_WARP_SIZE, Bq / 16);  // warps = Bq/16
    cudaError_t aerr = cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            static_cast<int>(smem));
    if (aerr != cudaSuccess)
        return false;

    static bool logged_once = false;
    if (!logged_once) {
        logged_once = true;
        IMP_LOG_INFO(
            "FMHA FA2 register-resident kernel ACTIVE (hd=%d, Bq=%d, Bkv=%d, qk=%s, qk_acc=%s, "
            "pv_acc=%s, kv_buf=%s, smem=%zu B, seq_q=%d seq_kv=%d)",
            head_dim, Bq, Bkv, fp16_qk ? "f16" : "e4m3", f16acc ? "f16" : "f32", pv_f16 ? "f16" : "f32",
            twoslot ? "twoslot" : "dbuf", smem, seq_q, seq_kv);
    }
    // FP8SCALED: per-chunk operand amaxes for Q and the gathered K (two tiny
    // grid-stride passes; s_d_amax is the file-scope persistent buffer above).
    if (fp8_scaled) {
        if (!s_d_amax && cudaMalloc(&s_d_amax, 2 * sizeof(float)) != cudaSuccess) {
            s_d_amax = nullptr;
            fp8_scaled = false;  // fall back to raw conversion this call
        }
        if (s_d_amax) {
            cudaMemsetAsync(s_d_amax, 0, 2 * sizeof(float), stream);
            const int64_t qn = (int64_t)batch_size * seq_q * n_heads * head_dim;
            const int64_t kn = (int64_t)batch_size * seq_kv * n_kv_heads * head_dim;
            fa2_amax_fp16_kernel<<<128, 256, 0, stream>>>(reinterpret_cast<const half*>(Q.data), qn,
                                                          s_d_amax);
            fa2_amax_fp16_kernel<<<128, 256, 0, stream>>>(reinterpret_cast<const half*>(K.data), kn,
                                                          s_d_amax + 1);
        }
    }
    kern<<<grid, block, smem, stream>>>(reinterpret_cast<const half*>(Q.data),
                                        reinterpret_cast<const half*>(K.data),
                                        reinterpret_cast<const half*>(V.data),
                                        reinterpret_cast<half*>(O.data), batch_size, seq_q, seq_kv, n_heads,
                                        n_kv_heads, scale, causal, sliding_window, softcap, q_offset,
                                        fp8_scaled ? s_d_amax : nullptr, d_kv_len);
    return true;
}

}  // namespace imp
