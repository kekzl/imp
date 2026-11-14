// FA2 prefill on WMMA HMMA fragments (mma.sync.m16n8k16.f16 QK^T/PV).
// NOT wgmma/tcgen05/TMEM (sm_90a, sm_100+ only, unavailable on sm_120a). 256 threads (8 warps).
// Bq/Bkv by head_dim (99KB smem budget): HD64 Bq128/Bkv64 (~89KB); HD96 Bq64/Bkv64 (~65KB);
// HD128 Bq64/Bkv64 (~81KB); HD256 Bq32/Bkv64 (~88KB).
// Shared memory: Q_tile half[Bq,HD], KV_tile half[Bkv,HD] (K then reused for V), S_tile
// float[Bq,Bkv] (aliased half P), O_acc float[Bq,HD], row_m/row_l float[Bq].

#include "compute/attention_fmha_sm120.h"
#include "compute/attention_paged_common.cuh"
#include "compute/fmha_sm120_tile_select.h"
#include "core/cuda_static_reset.h"
#include "core/device_info.h"
#include "core/logging.h"
#include "core/process_diag.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <float.h>
#include <iterator>  // std::size
#include <utility>   // std::index_sequence
#include <mma.h>

using namespace nvcuda;

namespace imp {

// =============================================================================
// Constants
// =============================================================================

static constexpr int SM120_WARP_SIZE = 32;
static constexpr int SM120_NUM_WARPS = 8;
static constexpr int SM120_BLOCK_THREADS = SM120_WARP_SIZE * SM120_NUM_WARPS;  // 256
static constexpr int SM120_Bkv = 64;                                           // KV tile size (columns)

// WMMA m16n16k16 building block. On sm_120a these lower to HMMA (mma.sync); no wgmma on this
// target. Explicit tiling gives control over the softmax fusion.
static constexpr int SM120_WMMA_M = 16;
static constexpr int SM120_WMMA_N = 16;
static constexpr int SM120_WMMA_K = 16;

// =============================================================================
// Kernel template
// =============================================================================

template <int Bq, int HD, bool FixedKvOrder = false>
__global__ void __launch_bounds__(SM120_BLOCK_THREADS, 2) fmha_sm120_kernel(
    const half* __restrict__ Q, const half* __restrict__ K, const half* __restrict__ V, half* __restrict__ O,
    int batch_size, int seq_q, int seq_kv, int n_heads, int n_kv_heads, float scale, bool causal,
    int sliding_window, float softcap, int q_offset, const half* __restrict__ sinks) {
    // KV tile columns: hd=512 (Gemma-4 global / Qwen3.5-27B) uses a narrow Bkv so Q/KV/O_acc fits
    // the 99KB smem opt-in (Bq=16,Bkv=32,HD=512 needs ~82KB). Bkv=32 halves KV-tile iterations and
    // engages 2 warps in the QK WMMA vs Bkv=16, at acceptable FP16-class accuracy cost. This is the
    // O(n) fallback for long-context S-matrix overflow where cuBLAS can't run.
    constexpr int Bkv = (HD >= 512) ? 32 : SM120_Bkv;
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

    const int tid = threadIdx.x + (threadIdx.y * blockDim.x);
    const int warp_id = tid / SM120_WARP_SIZE;
    const int q_start = tile_q * Bq;

    // Parallel softmax: which row and lane within row
    const int sm_row = tid / TPR;
    const int sm_lane = tid % TPR;

    // Global memory strides (row-major [batch, seq, heads, head_dim])
    const int64_t q_row_stride = (int64_t)n_heads * head_dim;
    const int64_t kv_row_stride = (int64_t)n_kv_heads * head_dim;

    const half* Q_ptr = Q + ((int64_t)batch_idx * seq_q * q_row_stride) + ((int64_t)q_start * q_row_stride) +
                        ((int64_t)head_idx * head_dim);
    const half* K_ptr = K + ((int64_t)batch_idx * seq_kv * kv_row_stride) + ((int64_t)kv_head * head_dim);
    const half* V_ptr = V + ((int64_t)batch_idx * seq_kv * kv_row_stride) + ((int64_t)kv_head * head_dim);
    half* O_ptr = O + ((int64_t)batch_idx * seq_q * q_row_stride) + ((int64_t)q_start * q_row_stride) +
                  ((int64_t)head_idx * head_dim);

    // ---- shared memory layout -----------------------------------------------
    // K and V share the same buffer (KV_tile): K is loaded first, consumed
    // by QK^T WMMA, then V is loaded into the same region for PV WMMA.
    extern __shared__ char smem[];

    half* Q_tile = reinterpret_cast<half*>(smem);
    half* KV_tile = Q_tile + static_cast<ptrdiff_t>(Bq * head_dim);  // shared K/V buffer
    float* S_tile = reinterpret_cast<float*>(KV_tile + static_cast<ptrdiff_t>(Bkv * head_dim));
    float* O_acc = S_tile + static_cast<ptrdiff_t>(Bq * Bkv);
    float* row_m = O_acc + static_cast<ptrdiff_t>(Bq * head_dim);
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
                const float4* src = reinterpret_cast<const float4*>(&Q_ptr[((int64_t)r * q_row_stride) + d]);
                *dst = *src;
            } else {
                *dst = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }
    }

    // ---- zero output accumulator + init running softmax state ---------------
    {
        // float4 = 4 FP32 zeros per store. Bq*HD is always a multiple of 4
        // (HD ∈ {64,96,128,256,512}, Bq ∈ {16,32,64,128}).
        const int total_vec4 = (Bq * head_dim) / 4;
        const float4 zero = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        for (int vi = tid; vi < total_vec4; vi += SM120_BLOCK_THREADS) {
            reinterpret_cast<float4*>(O_acc)[vi] = zero;
        }
    }
    if (tid < Bq) {
        if (sinks != nullptr) {
            // gpt-oss learned sink (#547/#992): virtual extra logit column, no V contribution. Seeding
            // online-softmax state with it (m=sink, l=exp(sink-sink)=1) keeps it correctly rescaled by
            // every subsequent l *= exp(m_old-m_new), matching the cuBLAS softmax max/denominator join.
            row_m[tid] = __half2float(sinks[head_idx]);
            row_l[tid] = 1.0f;
        } else {
            row_m[tid] = -FLT_MAX;
            row_l[tid] = 0.0f;
        }
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
    // FixedKvOrder: every row scans KV tiles 0..n, so its bits do not depend on its Q tile (#2167).
    const bool sawtooth_reverse = (blockIdx.x % 2 == 1) & !FixedKvOrder;
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
                        &K_ptr[((int64_t)(kv_start + r) * kv_row_stride) + d]);
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
                wmma::load_matrix_sync(a_frag,
                                       Q_tile + (static_cast<ptrdiff_t>(ri) * SM120_WMMA_M * head_dim) +
                                           static_cast<ptrdiff_t>(k * SM120_WMMA_K),
                                       head_dim);

                wmma::fragment<wmma::matrix_b, SM120_WMMA_M, SM120_WMMA_N, SM120_WMMA_K, half,
                               wmma::col_major>
                    b_frag;
                wmma::load_matrix_sync(b_frag,
                                       KV_tile + (static_cast<ptrdiff_t>(ci) * SM120_WMMA_N * head_dim) +
                                           static_cast<ptrdiff_t>(k * SM120_WMMA_K),
                                       head_dim);

                wmma::mma_sync(acc, a_frag, b_frag, acc);
            }

            wmma::store_matrix_sync(S_tile + (static_cast<ptrdiff_t>(ri) * SM120_WMMA_M * Bkv) +
                                        static_cast<ptrdiff_t>(ci * SM120_WMMA_N),
                                    acc, Bkv, wmma::mem_row_major);
        }
        __syncthreads();

        // ---- Apply scale, softcap, and causal/sliding_window mask ----
        apply_score_masks(S_tile, Bq, Bkv, SM120_BLOCK_THREADS, tid, q_start, kv_start, seq_q, seq_kv, scale,
                          softcap, causal, sliding_window, q_offset);
        __syncthreads();

        // Phase 2+3: parallel online softmax + rescale O + SP->half (all threads participate, TPR
        // threads/row cooperate via warp shuffle). SP_float/SP_half alias the same shared memory.
        {
            half* SP_half = reinterpret_cast<half*>(S_tile);
            const int r = sm_row;
            const bool row_valid = (r < Bq) && (q_start + r < seq_q);

            // Step 1: Parallel row max
            float partial_max = -FLT_MAX;
            if (row_valid) {
                for (int c = sm_lane; c < Bkv; c += TPR) {
                    partial_max = fmaxf(partial_max, S_tile[(r * Bkv) + c]);
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

            // Step 3: parallel exp+sum. Mask guard: when a whole tile is masked (SWA query beyond window at
            // kv_start=0), m_new collapses to -FLT_MAX and expf(0)=1 would inflate partial_sum by Bkv per
            // row.
            // Explicit sentinel maps masked scores to 0.
            float partial_sum = 0.0f;
            if (row_valid) {
                for (int c = sm_lane; c < Bkv; c += TPR) {
                    float s_val = S_tile[(r * Bkv) + c];
                    float p = (s_val <= -FLT_MAX * 0.5f) ? 0.0f : __expf(s_val - m_new);
                    partial_sum += p;
                    S_tile[(r * Bkv) + c] = p;
                }
            }
#pragma unroll
            for (int offset = TPR / 2; offset >= 1; offset >>= 1) {
                partial_sum += __shfl_xor_sync(0xffffffff, partial_sum, offset);
            }

            // Step 4: Update running state
            float l_old = row_valid ? row_l[r] : 0.0f;
            float l_new = (alpha * l_old) + partial_sum;
            if (sm_lane == 0 && row_valid) {
                row_m[r] = m_new;
                row_l[r] = l_new;
            }

            // Step 5: Rescale O_acc. IEEE division: a tile fully masked for this row gives l/l == 1
            // exactly (the approximate one does not), so a row's bits do not depend on its Q tile.
            float rescale = (l_old > 0.0f) ? __fdiv_rn(alpha * l_old, l_new) : 0.0f;
            if (row_valid) {
                for (int d = sm_lane; d < head_dim; d += TPR) {
                    O_acc[(r * head_dim) + d] *= rescale;
                }
            }

            // Step 6: fused softmax normalize + float->half, in place. SP_half aliases S_tile compactly
            // (half row r in float row r/2's bytes); unsynced stores clobber unread float scores (#528,
            // worst at short seq_q). Stage in registers, barrier, then store.
            constexpr int CPT = Bkv / TPR;  // columns per thread
            float spv = (l_new > 0.0f) ? (1.0f / l_new) : 0.0f;
            half hbuf[CPT];
#pragma unroll
            for (int i = 0; i < CPT; i++) {
                int c = sm_lane + (i * TPR);
                hbuf[i] = __float2half(row_valid ? S_tile[(r * Bkv) + c] * spv : 0.0f);
            }
            __syncthreads();  // all float reads of S_tile complete before any half write
#pragma unroll
            for (int i = 0; i < CPT; i++)
                SP_half[(r * Bkv) + sm_lane + (i * TPR)] = hbuf[i];
        }
        __syncthreads();

        // ---- Load V tile (vectorized: float4 = 8 halves per iter) ----
        {
            // All supported head_dims (64, 96, 128, 256, 512) are multiples of
            // 8, so float4 loads are always aligned and in-bounds per row.
            const int total_vec8 = (Bkv * head_dim) / 8;
            for (int vi = tid; vi < total_vec8; vi += SM120_BLOCK_THREADS) {
                int i = vi * 8;
                int r = i / head_dim;
                int d = i % head_dim;
                float4* dst = reinterpret_cast<float4*>(&KV_tile[i]);
                if (kv_start + r < seq_kv) {
                    const float4* src = reinterpret_cast<const float4*>(
                        &V_ptr[((int64_t)(kv_start + r) * kv_row_stride) + d]);
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
                wmma::load_matrix_sync(o_frag,
                                       O_acc + (static_cast<ptrdiff_t>(ri) * SM120_WMMA_M * head_dim) +
                                           static_cast<ptrdiff_t>(di * SM120_WMMA_N),
                                       head_dim, wmma::mem_row_major);

                for (int k = 0; k < pv_chunks; k++) {
                    wmma::fragment<wmma::matrix_a, SM120_WMMA_M, SM120_WMMA_N, SM120_WMMA_K, half,
                                   wmma::row_major>
                        p_frag;
                    wmma::load_matrix_sync(p_frag,
                                           P_half + (static_cast<ptrdiff_t>(ri) * SM120_WMMA_M * Bkv) +
                                               static_cast<ptrdiff_t>(k * SM120_WMMA_K),
                                           Bkv);

                    wmma::fragment<wmma::matrix_b, SM120_WMMA_M, SM120_WMMA_N, SM120_WMMA_K, half,
                                   wmma::row_major>
                        v_frag;
                    wmma::load_matrix_sync(v_frag,
                                           KV_tile + (static_cast<ptrdiff_t>(k) * SM120_WMMA_N * head_dim) +
                                               static_cast<ptrdiff_t>(di * SM120_WMMA_N),
                                           head_dim);

                    wmma::mma_sync(o_frag, p_frag, v_frag, o_frag);
                }

                wmma::store_matrix_sync(O_acc + (static_cast<ptrdiff_t>(ri) * SM120_WMMA_M * head_dim) +
                                            static_cast<ptrdiff_t>(di * SM120_WMMA_N),
                                        o_frag, head_dim, wmma::mem_row_major);
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
            *reinterpret_cast<uint2*>(&O_ptr[((int64_t)r * q_row_stride) + (i % head_dim)]) = packed;
        }
    }
}

// =============================================================================
// Shared memory computation
// =============================================================================

// sm120_fmha_smem_bytes (fmha_sm120_tile_select.h): Q + shared K/V tile + S + O_acc + row_m/row_l.

// =============================================================================
// Host launcher
// =============================================================================

bool fmha_sm120_prefill(const Tensor& Q, const Tensor& K, const Tensor& V, Tensor& O, float scale,
                        bool causal, int sliding_window, float softcap, cudaStream_t stream, int q_offset,
                        const half* sinks, bool fixed_kv_order) {
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
    const int max_smem = device_attr_or_0(cudaDevAttrMaxSharedMemoryPerBlockOptin);  // 0: declines below

    // KV tile columns — must match the kernel's compile-time `Bkv` (which
    // derives from HD): hd=512 uses Bkv=32 (~82 KB at Bq=16, fits the 99 KB
    // opt-in; +40% at long context — see the kernel-side comment).
    const int Bkv = sm120_fmha_bkv(head_dim);
    static_assert(sm120_fmha_bkv(128) == SM120_Bkv, "tile-select Bkv must match the kernel");

    // Bq: largest instanced tile at occupancy 2 (smem <= max/2), else occupancy 1 (#2243).
    const int Bq = sm120_fmha_select_bq(head_dim, static_cast<size_t>(max_smem));
    if (Bq == 0) {
        IMP_LOG_DEBUG("FMHA sm120: no instanced Bq fits smem (hd=%d, max=%d)", head_dim, max_smem);
        return false;
    }

    const size_t smem = sm120_fmha_smem_bytes(Bq, Bkv, head_dim);
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

#define LAUNCH_FMHA_SM120_FN(KFN, BQ, HD)                                                                \
    do {                                                                                                 \
        cudaError_t attr_err = cudaFuncSetAttribute(KFN, cudaFuncAttributeMaxDynamicSharedMemorySize,    \
                                                    static_cast<int>(smem));                             \
        if (attr_err != cudaSuccess) {                                                                   \
            IMP_LOG_WARN("FMHA sm120: cudaFuncSetAttribute failed for Bq=%d HD=%d smem=%zu: %s", BQ, HD, \
                         smem, cudaGetErrorString(attr_err));                                            \
            return false;                                                                                \
        }                                                                                                \
        IMP_CUDA_CHECK_LOG(cudaFuncSetAttribute(KFN, cudaFuncAttributePreferredSharedMemoryCarveout,     \
                                                cudaSharedmemCarveoutMaxShared));                        \
        (KFN)<<<grid, block, smem, stream>>>(reinterpret_cast<const half*>(Q.data),                      \
                                           reinterpret_cast<const half*>(K.data),                        \
                                           reinterpret_cast<const half*>(V.data),                        \
                                           reinterpret_cast<half*>(O.data), batch_size, seq_q, seq_kv,   \
                                           n_heads, n_kv_heads, scale, causal, sliding_window, softcap,  \
                                           q_offset, sinks);                                             \
        IMP_CUDA_CHECK_LAUNCH();                                                                         \
    } while (0)
#define LAUNCH_FMHA_SM120(BQ, HD) LAUNCH_FMHA_SM120_FN((fmha_sm120_kernel<BQ, HD, false>), BQ, HD)

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
    } else if (Bq == 32) {
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
    } else {
        // Bq=16: only head_dim=512 (Gemma-4 global layers) lands here — the
        // narrow Bkv=16 tile keeps SMEM under the 99 KB opt-in. Smaller head
        // dims always fit at Bq>=32, so no other case is instantiated here.
        switch (head_dim) {
            case 512: {
                // fixed_kv_order is instanced for hd=512 only: its one caller (#2167).
                using Hd512Kernel = decltype(&fmha_sm120_kernel<16, 512, false>);
                const Hd512Kernel kernels_hd512[2] = {&fmha_sm120_kernel<16, 512, false>,
                                                      &fmha_sm120_kernel<16, 512, true>};
                LAUNCH_FMHA_SM120_FN(kernels_hd512[static_cast<int>(fixed_kv_order)], 16, 512);
                return true;
            }
            default:
                break;
        }
    }

#undef LAUNCH_FMHA_SM120
#undef LAUNCH_FMHA_SM120_FN

    return false;
}

// FA2 register-resident kernel ("echtes FA"): keeps S/P/O entirely in registers. Only K/V (f16)
// stage in smem, Q stays in registers. 8 warps x 16 rows = Bq=128; each warp runs an independent
// online softmax (no cross-warp reduction). KV tiles Bkv=64.
// Transpose-free PV: the QK m16n8 accumulator layout (thread t: rows{t/4,t/4+8} x
// cols{(t%4)*2,+1}) is byte-identical to the m16n8k16 PV A-operand layout, so two adjacent
// 16x8 S tiles assemble directly into the 16x16 P A-fragment - no movmatrix, no smem.
// Per KV tile: load K+V->smem->sync->QK(reg)->softmax(reg)->PV(reg)->sync (2 barriers/tile).
// cp.async double-buffers K/V for tile j+1 while tile j computes; K converts f16->fp8 inline
// at QK operand-fetch (spare ALU on this barrier-bound kernel).

__device__ __forceinline__ uint32_t pack2_f2h(float a, float b) {
    const __half2 h = __floats2half2_rn(a, b);
    return static_cast<uint32_t>(__half_as_ushort(__low2half(h))) |
           (static_cast<uint32_t>(__half_as_ushort(__high2half(h))) << 16);
}
// ldsm_x4: one warp instruction loads four 8x8 b16 tiles from smem, replacing several scalar
// LDS per MMA operand fetch. Non-trans form delivers fragments in MMA A/B register order;
// .trans transposes each 8x8 during delivery - the V[kv][hd]->B[k][n] turn PV needs, killing
// strided 2-byte V loads and pack2 PRMTs.
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

// Prefetch Bkv x HD KV tile (raw f16) via cp.async into double-buffer slots. Bank-conflict
// padding: head_dim=128 row stride is exactly 32 banks, aliasing every row onto the same banks.
//   K/V (halfs): +8B -> stride 136B (preserves 16B cp.async alignment), cuts wide reads to ~2-way.
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
            const half* ks = K_ptr + ((int64_t)(kv_start + r) * kv_row_stride) + c;
            const half* vs = V_ptr + ((int64_t)(kv_start + r) * kv_row_stride) + c;
            cp_async_cg_16(&K_dst[(r * KVS) + c], ks);
            cp_async_cg_16(&V_dst[(r * KVS) + c], vs);
        } else {
            *reinterpret_cast<uint4*>(&V_dst[(r * KVS) + c]) = make_uint4(0, 0, 0, 0);
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
            cp_async_cg_16(&K_dst[(r * KVS) + c], K_ptr + ((int64_t)(kv_start + r) * kv_row_stride) + c);
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
            cp_async_cg_16(&V_dst[(r * KVS) + c], V_ptr + ((int64_t)(kv_start + r) * kv_row_stride) + c);
        } else {
            *reinterpret_cast<uint4*>(&V_dst[(r * KVS) + c]) = make_uint4(0, 0, 0, 0);
        }
    }
}

// QK via mma.m16n8k16.f16, full accuracy: the short-sequence path replacing materialized
// cuBLAS+softmax below fmha_prefill_threshold (the e4m3 QK variant compounded noise, #511/#512).
// TWOSLOT (#597): one K slot + one V slot (not a double-buffer) keeps the full Bkv=64 tile at
// ~35KB smem, seating 2 CTAs/SM with half BKV=32's per-tile barrier overhead.
// PVF16 (#667): f16-accumulate PV, O as packed half2 - safe since online-softmax O is a convex
// combination of V rows (weights in [0,1]); halves PV MMA cost and O register footprint.
template <int Bq, int HD, bool F16ACC = false, int BKV = 64, bool TWOSLOT = false, bool PVF16 = false>
__device__ __forceinline__ void fmha_sm120_fa2_body(const half* __restrict__ Q, const half* __restrict__ K,
                                                    const half* __restrict__ V, half* __restrict__ O,
                                                    int batch_size, int seq_q, int seq_kv, int n_heads,
                                                    int n_kv_heads, float scale, bool causal,
                                                    int sliding_window, float softcap, int q_offset,
                                                    const int* __restrict__ d_kv_len, bool heavy_first,
                                                    const half* __restrict__ sinks) {
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

    // Causal: q-tile t attends (t+1)*Bq/Bkv KV tiles, so CTAs of one head differ up to 16x in work.
    // blockIdx.x is the fastest scheduling index; handing out heavy tiles first (fa2_heavy_first)
    // leaves light ones for the wave tail. Rows never depend on CTA order.
    const int tile_q = (causal && heavy_first) ? static_cast<int>(gridDim.x) - 1 - static_cast<int>(blockIdx.x)
                                               : static_cast<int>(blockIdx.x);
    const int batch_head = blockIdx.y;
    const int batch_idx = batch_head / n_heads;
    const int head_idx = batch_head % n_heads;
    const int kv_head = head_idx / (n_heads / n_kv_heads);

    // Graph-captured verify (#847): real KV length lives on device; baked seq_kv is only buffer
    // capacity. q_offset follows from the chunked continuation invariant seq_kv == q_offset+seq_q.
    if (d_kv_len != nullptr) {
        seq_kv = __ldg(d_kv_len);
        q_offset = seq_kv - seq_q;
    }

    const int lane = threadIdx.x;     // 0..31
    const int warp_id = threadIdx.y;  // 0..7  → this warp's row tile
    const int tid = lane + (warp_id * 32);
    const int q_start = tile_q * Bq;
    const int row_lo = (warp_id * 16) + (lane / 4);  // local row within Bq for d0/d1
    const int rl = lane / 4;                     // row in 16-tile (lo)
    const int cl = (lane % 4) * 2;               // col base in 8/16-tile

    const int64_t q_row_stride = (int64_t)n_heads * head_dim;
    const int64_t kv_row_stride = (int64_t)n_kv_heads * head_dim;
    const half* Q_ptr = Q + ((int64_t)batch_idx * seq_q * q_row_stride) + ((int64_t)q_start * q_row_stride) +
                        ((int64_t)head_idx * head_dim);
    const half* K_ptr = K + ((int64_t)batch_idx * seq_kv * kv_row_stride) + ((int64_t)kv_head * head_dim);
    const half* V_ptr = V + ((int64_t)batch_idx * seq_kv * kv_row_stride) + ((int64_t)kv_head * head_dim);
    half* O_ptr = O + ((int64_t)batch_idx * seq_q * q_row_stride) + ((int64_t)q_start * q_row_stride) +
                  ((int64_t)head_idx * head_dim);

    // Padded smem row stride (see FA2_KV_PAD, bank-conflict relief). Q stays in registers: no smem
    // tile, which frees enough smem for the 8-warp Bq=128 config.
    constexpr int KVSTRIDE = head_dim + FA2_KV_PAD;  // K/V row stride in HALFS

    extern __shared__ char smem[];
    // Double-buffer mode: K_buf/V_buf are [2][Bkv*KVSTRIDE]. TWOSLOT mode:
    // one slot each — half the footprint, rotation handled by load phases.
    constexpr int KV_SLOTS = TWOSLOT ? 1 : 2;
    half* K_buf = reinterpret_cast<half*>(smem);                              // [KV_SLOTS][Bkv*KVSTRIDE] f16
    half* V_buf = K_buf + static_cast<ptrdiff_t>(KV_SLOTS * Bkv * KVSTRIDE);  // [KV_SLOTS][Bkv*KVSTRIDE] f16

    // ---- per-warp register state ----
    // PVF16: O held as packed half2 — reg [0] = (row rl, cols cl,cl+1),
    // reg [1] = (row rl+8, same cols), matching the m16n8k16 f16-D layout.
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
    __syncthreads();

    // Q A-fragments are loop-invariant, loaded once straight from global into registers
    // (rows past seq_q zero-filled). Layout matches the QK MMA's a0..a3: a0=(rl,cl) a1=(rl+8,cl)
    // a2=(rl,cl+8) a3=(rl+8,cl+8).
    uint32_t a_frag[HD / 16][4];
    {
        const int r0 = (warp_id * 16) + rl;
        const int r1 = r0 + 8;
        const bool v0 = (q_start + r0) < seq_q;
        const bool v1 = (q_start + r1) < seq_q;
        const half* q0 = Q_ptr + ((int64_t)r0 * q_row_stride);
        const half* q1 = Q_ptr + ((int64_t)r1 * q_row_stride);
#pragma unroll
        for (int k = 0; k < HD / 16; k++) {
            const int d = (k * 16) + cl;
            a_frag[k][0] = v0 ? *reinterpret_cast<const uint32_t*>(q0 + d) : 0u;
            a_frag[k][1] = v1 ? *reinterpret_cast<const uint32_t*>(q1 + d) : 0u;
            a_frag[k][2] = v0 ? *reinterpret_cast<const uint32_t*>(q0 + d + 8) : 0u;
            a_frag[k][3] = v1 ? *reinterpret_cast<const uint32_t*>(q1 + d + 8) : 0u;
        }
    }
    // Scores stay raw dot products; softmax applies the scale inside one FMA feeding __expf so the
    // per-element v*=scale pass is avoided. softcap works in the natural domain and writes the
    // capped score back in raw units, keeping this FMA the single scale point.
    const float s_eff = scale;
    const float inv_scale = 1.0f / scale;
    if (sinks != nullptr) {
        // gpt-oss learned sink: extra logit column without a V row. Seed m with it in raw-score units
        // (softmax multiplies by s_eff) and l = exp(0) = 1, as the WMMA FMHA does (#547/#992).
        mA = mB = __half2float(sinks[head_idx]) / s_eff;
        lA = lB = 1.0f;
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
                prefetch_kv_tile<head_dim, Bkv>(K_buf + static_cast<ptrdiff_t>(nslot * Bkv * KVSTRIDE),
                                                V_buf + static_cast<ptrdiff_t>(nslot * Bkv * KVSTRIDE), K_ptr,
                                                V_ptr, (j + 1) * Bkv, seq_kv, kv_row_stride, tid, NTHREADS);
                cp_async_commit();
                cp_async_wait_group<1>();  // this tile (slot) landed; tile j+1 still in flight
            } else {
                cp_async_wait_group<0>();
            }
            __syncthreads();  // this tile's K/V fully landed for all threads
        }

        const half* K_cur = K_buf + static_cast<ptrdiff_t>(slot * Bkv * KVSTRIDE);
        const half* V_cur = V_buf + static_cast<ptrdiff_t>(slot * Bkv * KVSTRIDE);

        // ---- QK: S[n] = Q(warp rows) @ K[n-tile]^T ----
        // m16n8k16.f16.
        float S[N_S][4];
        {
            // n-tiles processed in pairs: one ldmatrix.x4 fetches K B-fragments (b0,b1) of both tiles -
            // lanes 0-7/8-15 address {rows n*8+0..7,k-lo}/{same rows,k-hi}, lanes 16-31 the same for tile
            // n+1. A comes from the hoisted loop-invariant a_frag.
#pragma unroll
            for (int n = 0; n < N_S; n += 2) {
                const int b_row = ((n + (lane >> 4)) * 8) + (lane & 7);
                const int b_khalf = ((lane >> 3) & 1) << 3;
                const half* krow = K_cur + static_cast<ptrdiff_t>(b_row * KVSTRIDE) + b_khalf;
                // software pipeline: fetch k+1's B fragments while k's MMAs
                // run — at 1-2 warps per scheduler the LDSM→HMMA dependency
                // latency is otherwise exposed (short_scoreboard-bound).
                uint32_t b0, b1, c0, c1;
                ldsm_x4(b0, b1, c0, c1, krow);
                if constexpr (F16ACC) {
                    // f16-accumulate QK^T (#597, opt-in fa2_f16acc): sm_120 runs f16-src/f32-acc HMMA at 1/4
                    // rate
                    // (#606); f16-acc lifts the score MMA to full-rate. Scores are softmaxed immediately so
                    // reduced
                    // accumulate precision is low-risk. Accumulators are 2 packed-half2 regs. PV stays
                    // f32-acc
                    // (online O sum needs the range).
                    uint32_t dA[2] = {0u, 0u};
                    uint32_t dB[2] = {0u, 0u};
#pragma unroll
                    for (int k = 0; k < HD / 16; k++) {
                        uint32_t nb0, nb1, nc0, nc1;
                        if (k + 1 < HD / 16)
                            ldsm_x4(nb0, nb1, nc0, nc1, krow + static_cast<ptrdiff_t>((k + 1) * 16));
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
                            ldsm_x4(nb0, nb1, nc0, nc1, krow + static_cast<ptrdiff_t>((k + 1) * 16));
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

        // softcap + causal/SWA mask (per register): a CTA-uniform interior tile (no causal/window edge
        // inside it) skips the compares/selects; only diagonal/edge tiles pay them.
        const int gq_lo = q_offset + q_start;
        const int gq_hi = q_offset + q_start + Bq - 1;
        const bool tile_masked = (kv_start + Bkv > seq_kv) || (q_start + Bq > seq_q) ||
                                 (causal && (kv_start + Bkv - 1) > gq_lo) ||
                                 (sliding_window > 0 && (gq_hi - kv_start) >= sliding_window);
        if (!tile_masked) {
            if (softcap > 0.0f) {
#pragma unroll
                for (int n = 0; n < N_S; n++)
#pragma unroll
                    for (int e = 0; e < 4; e++)
                        S[n][e] = softcap * tanhf(S[n][e] * scale / softcap) * inv_scale;
            }
        } else {
#pragma unroll
            for (int n = 0; n < N_S; n++) {
                int colb = kv_start + (n * 8) + cl;
                int gqA = q_offset + q_start + (warp_id * 16) + rl;
                int gqB = gqA + 8;
                int lqA = q_start + (warp_id * 16) + rl;
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    int col = colb + (e & 1);
                    int gq = (e < 2) ? gqA : gqB;
                    int lq = lqA + ((e < 2) ? 0 : 8);
                    float v = S[n][e];
                    if (lq < seq_q && col < seq_kv) {
                        if (softcap > 0.0f) {
                            float v_nat = v * scale;
                            v = softcap * tanhf(v_nat / softcap) * inv_scale;
                        }
                        if (causal && gq < col)
                            v = -FLT_MAX;
                        if (sliding_window > 0 && (gq - col) >= sliding_window)
                            v = -FLT_MAX;
                    } else {
                        v = -FLT_MAX;
                    }
                    S[n][e] = v;
                }
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
        // p = exp(scale*(s-m)) as one FMA feeding __expf. An exp2 form (scale*log2e folded in) was
        // faster but changed PPL; this form keeps the old numerics and drops a separate v*=scale pass.
        // Interior tiles need no -FLT_MAX guard: every score finite, so is the row max.
        const float alphaA = __expf((mA - mnA) * s_eff), alphaB = __expf((mB - mnB) * s_eff);
        const float mlA2 = -mnA * s_eff, mlB2 = -mnB * s_eff;

        float psA = 0.f, psB = 0.f;
        if (!tile_masked) {
#pragma unroll
            for (int n = 0; n < N_S; n++) {
                float p0 = __expf(fmaf(S[n][0], s_eff, mlA2));
                float p1 = __expf(fmaf(S[n][1], s_eff, mlA2));
                float p2 = __expf(fmaf(S[n][2], s_eff, mlB2));
                float p3 = __expf(fmaf(S[n][3], s_eff, mlB2));
                S[n][0] = p0;
                S[n][1] = p1;
                S[n][2] = p2;
                S[n][3] = p3;
                psA += p0 + p1;
                psB += p2 + p3;
            }
        } else {
#pragma unroll
            for (int n = 0; n < N_S; n++) {
                float p0 = (S[n][0] <= -FLT_MAX * 0.5f) ? 0.f : __expf(fmaf(S[n][0], s_eff, mlA2));
                float p1 = (S[n][1] <= -FLT_MAX * 0.5f) ? 0.f : __expf(fmaf(S[n][1], s_eff, mlA2));
                float p2 = (S[n][2] <= -FLT_MAX * 0.5f) ? 0.f : __expf(fmaf(S[n][2], s_eff, mlB2));
                float p3 = (S[n][3] <= -FLT_MAX * 0.5f) ? 0.f : __expf(fmaf(S[n][3], s_eff, mlB2));
                S[n][0] = p0;
                S[n][1] = p1;
                S[n][2] = p2;
                S[n][3] = p3;
                psA += p0 + p1;
                psB += p2 + p3;
            }
        }
        psA = quad_sum(psA);
        psB = quad_sum(psB);
        lA = (alphaA * lA) + psA;
        lB = (alphaB * lB) + psB;
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

        // PV: O += P@V, m16n8k16 f16. V B-fragments via ldmatrix.x4.trans over hn-pairs: lanes 0-7/8-15
        // address V rows m*16+0..7/+8..15 at col hn*8, lanes 16-31 the same rows at col (hn+1)*8.
        // .trans turns row-major 8x8 V into the col-major B fragment.
#pragma unroll
        for (int m = 0; m < N_KG; m++) {
            uint32_t ra0 = pack2_f2h(S[2 * m][0], S[2 * m][1]);
            uint32_t ra1 = pack2_f2h(S[2 * m][2], S[2 * m][3]);
            uint32_t ra2 = pack2_f2h(S[(2 * m) + 1][0], S[(2 * m) + 1][1]);
            uint32_t ra3 = pack2_f2h(S[(2 * m) + 1][2], S[(2 * m) + 1][3]);
            const half* vrow = V_cur + static_cast<ptrdiff_t>(((m * 16) + (lane & 15)) * KVSTRIDE) +
                               ((lane >> 4) << 3);
            // software pipeline: V fragments for hn+2 fetched under hn's MMAs
            uint32_t rb0, rb1, rc0, rc1;
            ldsm_x4_trans(rb0, rb1, rc0, rc1, vrow);
#pragma unroll
            for (int hn = 0; hn < N_O; hn += 2) {
                uint32_t sb0, sb1, sc0, sc1;
                if (hn + 2 < N_O)
                    ldsm_x4_trans(sb0, sb1, sc0, sc1, vrow + static_cast<ptrdiff_t>((hn + 2) * 8));
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
    int rowA = (warp_id * 16) + rl;  // local row (O_ptr already at q_start)
    int rowB = rowA + 8;
#pragma unroll
    for (int hn = 0; hn < N_O; hn++) {
        int col = (hn * 8) + cl;
        float oA0, oA1, oB0, oB1;
        if constexpr (PVF16) {
            const float2 fA = __half22float2(*reinterpret_cast<const __half2*>(&O_h2[hn][0]));
            const float2 fB = __half22float2(*reinterpret_cast<const __half2*>(&O_h2[hn][1]));
            oA0 = fA.x, oA1 = fA.y, oB0 = fB.x, oB1 = fB.y;
        } else {
            oA0 = O_frag[hn][0], oA1 = O_frag[hn][1], oB0 = O_frag[hn][2], oB1 = O_frag[hn][3];
        }
        if (q_start + rowA < seq_q) {
            O_ptr[((int64_t)rowA * q_row_stride) + col] = __float2half(oA0 * invA);
            O_ptr[((int64_t)rowA * q_row_stride) + col + 1] = __float2half(oA1 * invA);
        }
        if (q_start + rowB < seq_q) {
            O_ptr[((int64_t)rowB * q_row_stride) + col] = __float2half(oB0 * invB);
            O_ptr[((int64_t)rowB * q_row_stride) + col + 1] = __float2half(oB1 * invB);
        }
    }
}

template <int Bq, int HD, bool F16ACC = false, int BKV = 64, bool TWOSLOT = false, bool PVF16 = false>
__global__ void fmha_sm120_fa2_kernel(const half* __restrict__ Q, const half* __restrict__ K,
                                      const half* __restrict__ V, half* __restrict__ O, int batch_size,
                                      int seq_q, int seq_kv, int n_heads, int n_kv_heads, float scale,
                                      bool causal, int sliding_window, float softcap, int q_offset,
                                      const int* __restrict__ d_kv_len = nullptr, bool heavy_first = true,
                                      const half* __restrict__ sinks = nullptr) {
    fmha_sm120_fa2_body<Bq, HD, F16ACC, BKV, TWOSLOT, PVF16>(Q, K, V, O, batch_size, seq_q, seq_kv, n_heads,
                                                             n_kv_heads, scale, causal, sliding_window,
                                                             softcap, q_offset, d_kv_len, heavy_first, sinks);
}

// Dense FA2 (process_diag_fa2_dense_2cta): the Bq=128 TWOSLOT instance at two CTAs per SM.
template <int Bq, int HD, bool F16ACC, int BKV, bool TWOSLOT, bool PVF16>
__global__ void __launch_bounds__(Bq / 16 * 32, 2) fmha_sm120_fa2_kernel_2cta(
    const half* __restrict__ Q, const half* __restrict__ K, const half* __restrict__ V, half* __restrict__ O,
    int batch_size, int seq_q, int seq_kv, int n_heads, int n_kv_heads, float scale, bool causal,
    int sliding_window, float softcap, int q_offset, const int* __restrict__ d_kv_len = nullptr,
    bool heavy_first = true, const half* __restrict__ sinks = nullptr) {
    fmha_sm120_fa2_body<Bq, HD, F16ACC, BKV, TWOSLOT, PVF16>(Q, K, V, O, batch_size, seq_q, seq_kv, n_heads,
                                                             n_kv_heads, scale, causal, sliding_window,
                                                             softcap, q_offset, d_kv_len, heavy_first, sinks);
}

// Q stays in registers (loop-invariant A-fragments loaded once from global): no smem tile, which
// is what lets Bq=128 (8 warps) fit.
static size_t compute_smem_fa2(int head_dim, int Bkv, bool twoslot = false) {
    const size_t kvstride = head_dim + FA2_KV_PAD;  // halfs (bank-conflict pad)
    const size_t slots = twoslot ? 1 : 2;           // TWOSLOT: one K + one V slot
    return (slots * Bkv * kvstride * sizeof(half))  // K_buf f16 (padded)
           + (slots * Bkv * kvstride * sizeof(half));  // V_buf f16 (padded)
}

// hd=64: the hd=128 bands, f32 accumulators (f16acc/pv_f16 flags do not apply), Bkv=64.
// Dense 2-CTA: 2 x 36864 B dbuf smem fits one SM; the bound caps 163 -> 128 regs.
static decltype(&fmha_sm120_fa2_kernel<128, 128>) fa2_hd64_kernel(long blocks_128, int sm_count, int& Bq,
                                                                  bool& twoslot) {
    Bq = blocks_128 >= (long)sm_count ? 128 : 64;
    twoslot = Bq == 64 && blocks_128 >= (long)(sm_count / 2);
    if (Bq == 128)
        return imp::process_diag_fa2_dense_2cta()
                   ? fmha_sm120_fa2_kernel_2cta<128, 64, false, 64, false, false>
                   : fmha_sm120_fa2_kernel<128, 64, false, 64>;
    return twoslot ? fmha_sm120_fa2_kernel<64, 64, false, 64, true>
                   : fmha_sm120_fa2_kernel<64, 64, false, 64>;
}

bool fmha_sm120_fa2_prefill(const Tensor& Q, const Tensor& K, const Tensor& V, Tensor& O, float scale,
                            bool causal, int sliding_window, float softcap, cudaStream_t stream, int q_offset,
                            const int* d_kv_len, const half* sinks) {
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
    // HD=128 is the tuned mainline. HD=256 (Qwen3.6 hybrids, gemma-class) is a stage-1 port, default
    // on since #932: fixed Bq=64/Bkv=64/TWOSLOT (a double buffer at HD=256 needs 135KB smem > the 99KB
    // opt-in; TWOSLOT fits at 67.6KB). HD=64 (gpt-oss): f32 accumulators for QK and PV. gpt-oss has no
    // QK norm: f16-accumulated raw logits moved its PPL 308.48 (cuBLAS) -> 267.28.
    if (head_dim != 128 && head_dim != 64 && !(head_dim == 256 && imp::process_diag_fa2_hd256()))
        return false;

    const int max_smem = device_attr_or_0(cudaDevAttrMaxSharedMemoryPerBlockOptin);  // 0: declines below
    const int sm_count = device_sm_count_or_0();                                     // 0: tile bands only

    // Bq/Bkv bands (#597 occupancy surgery), grid = q_tiles x batch*heads:
    //   blocks_128 >= sm_count: Bq=128/Bkv=64, 1 CTA/SM, 8 warps latency-hiding, deepest cp.async.
    //   sm_count/2 <= blocks_128 < sm_count (underfill): Bq=64 doubles the grid, Bkv=32 halves the
    //     KV double-buffer (~70KB->~35KB) for 2 CTAs/SM; Bq=128 can't fit (2x8 warps x ~175 regs
    //     exceeds the register file).
    //   blocks_128 < sm_count/2: stays below SM count even at Bq=64; keep Bkv=64 for deeper overlap.
    const long blocks_128 = (long)((seq_q + 127) / 128) * batch_size * n_heads;

    // f16-acc QK^T (#597): +3-4% pp2048/pp4096 NVFP4 for +0.37% PPL.
    bool f16acc = imp::process_diag_fa2_f16acc();
    // PV f16-accumulate rides on the f16acc path (full-rate PV MMA + halved
    // O-fragment registers), default on.
    bool pv_f16 = f16acc && imp::process_diag_fa2_pv_f16acc();
    int Bq, Bkv;
    bool twoslot = false;
    decltype(&fmha_sm120_fa2_kernel<128, 128>) kern;
    if (head_dim == 256) {
        // Stage-1 HD=256 port: Bq=64 (4 warps), TWOSLOT keeps the full Bkv=64 tile at 67.6KB
        // (double-buffer needs 135KB). Registers ~2x the HD=128 profile, so pv_f16 is preferred;
        // f32-acc variant exists for A/B but is expected to spill.
        Bq = 64, Bkv = 64, twoslot = true;
        kern = pv_f16   ? fmha_sm120_fa2_kernel<64, 256, true, 64, true, true>
               : f16acc ? fmha_sm120_fa2_kernel<64, 256, true, 64, true>
                        : fmha_sm120_fa2_kernel<64, 256, false, 64, true>;
    } else if (head_dim == 64) {
        f16acc = pv_f16 = false;
        Bkv = 64;
        kern = fa2_hd64_kernel(blocks_128, sm_count, Bq, twoslot);
    } else if (blocks_128 >= (long)sm_count) {
        Bq = 128, Bkv = 64;
        // Dense 2-CTA: TWOSLOT at Bq=128 (35 KB smem) with the kernel pinned to <= 128 registers,
        // so two 8-warp CTAs share an SM.
        const bool two_cta = pv_f16 && imp::process_diag_fa2_dense_2cta();
        twoslot = two_cta;
        kern = two_cta  ? fmha_sm120_fa2_kernel_2cta<128, 128, true, 64, true, true>
               : pv_f16 ? fmha_sm120_fa2_kernel<128, 128, true, 64, false, true>
               : f16acc ? fmha_sm120_fa2_kernel<128, 128, true, 64>
                        : fmha_sm120_fa2_kernel<128, 128, false, 64>;
    } else if (blocks_128 >= (long)(sm_count / 2)) {
        // Underfill band (#597): Bq=64 doubles the grid and TWOSLOT halves the KV smem (~70KB->~35KB)
        // at the full Bkv=64 tile, seating 2 CTAs/SM. Supersedes the Bkv=32 double-buffer at HD=128
        // (same residency, fewer barriers).
        Bq = 64, Bkv = 64, twoslot = true;
        kern = pv_f16   ? fmha_sm120_fa2_kernel<64, 128, true, 64, true, true>
               : f16acc ? fmha_sm120_fa2_kernel<64, 128, true, 64, true>
                        : fmha_sm120_fa2_kernel<64, 128, false, 64, true>;
    } else {
        Bq = 64, Bkv = 64;
        kern = pv_f16   ? fmha_sm120_fa2_kernel<64, 128, true, 64, false, true>
               : f16acc ? fmha_sm120_fa2_kernel<64, 128, true, 64>
                        : fmha_sm120_fa2_kernel<64, 128, false, 64>;
    }

    const size_t smem = compute_smem_fa2(head_dim, Bkv, twoslot);
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
            "FMHA FA2 register-resident kernel ACTIVE (hd=%d, Bq=%d, Bkv=%d, qk=f16, qk_acc=%s, "
            "pv_acc=%s, kv_buf=%s, smem=%zu B, seq_q=%d seq_kv=%d)",
            head_dim, Bq, Bkv, f16acc ? "f16" : "f32", pv_f16 ? "f16" : "f32", twoslot ? "twoslot" : "dbuf",
            smem, seq_q, seq_kv);
    }
    kern<<<grid, block, smem, stream>>>(reinterpret_cast<const half*>(Q.data),
                                        reinterpret_cast<const half*>(K.data),
                                        reinterpret_cast<const half*>(V.data),
                                        reinterpret_cast<half*>(O.data), batch_size, seq_q, seq_kv, n_heads,
                                        n_kv_heads, scale, causal, sliding_window, softcap, q_offset,
                                        d_kv_len, imp::process_diag_fa2_heavy_first(), sinks);
    IMP_CUDA_CHECK_LAUNCH();
    return true;
}

}  // namespace imp
