// =============================================================================
// attention_fmha_sm120_cluster.cu -- FMHA prefill with cluster K/V broadcast
// =============================================================================
//
// M5 Slice 2 (review/phase5_synthesis.md §2.2 — perf/m5-cluster-launch-fmha):
// the legacy per-(Q-head × batch × q-tile) FP16 kernel in
// attention_fmha_sm120.cu loads K and V tiles from global memory once per Q
// head. On GQA configs (n_q_per_kv ∈ {2, 4, 8}) this re-loads identical KV
// data n_q_per_kv× across sibling Q heads of the same KV head.
//
// This file launches a sibling kernel under cudaLaunchKernelEx with a
// cluster dimension of n_q_per_kv. Blocks sharing a KV head form a cluster:
//
//   - block.rank() == 0 owns the K- and V-tile loads into its own shared
//     memory.
//   - sibling blocks read K / V via DSMEM (cluster.map_shared_rank(...,0))
//     established by the cluster.sync() barriers that bracket each load.
//   - each block keeps its own Q tile, S tile, O accumulator and per-row
//     softmax state (independent — softmax does not cross Q heads).
//
// Net effect on KV bandwidth: 1 / n_q_per_kv (4× — 8×) of the legacy global
// reads. Per-block shared-memory footprint grows by one extra Bkv·HD halfs
// for the split K / V buffers (vs the legacy single KV_tile slot).
//
// PR 2.1 dispatches Bq = 64 for every HD ∈ {64, 96, 128, 256}. Bq = 128 is
// known-good for HD ∈ {96, 128, 256} but produces wrong output specifically
// for HD = 64 (suspected register-pressure / occupancy interaction —
// see select_cluster_Bq). PR 2.3 tile-tuning revisits Bq selection.
//
// All scheduling decisions match the M5 Slice 1 helper
// (runtime/cluster_launch.h): GPC-spread policy, power-of-2 cluster check.
//
// The new kernel is bit-identical to fmha_sm120_kernel<Bq, HD> within
// FP16/FP32 reordering tolerance — see tests/test_attention_fmha_sm120.cu
// ClusterPath* cases.
// =============================================================================

#include "compute/attention_fmha_sm120.h"
#include "compute/attention_paged_common.cuh"
#include "core/logging.h"
#include "runtime/cluster_launch.h"
#include "runtime/config.h"
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <float.h>
#include <mma.h>

using namespace nvcuda;

namespace imp {

namespace {

constexpr int CL_WARP_SIZE = 32;
constexpr int CL_NUM_WARPS = 8;
constexpr int CL_BLOCK_THREADS = CL_WARP_SIZE * CL_NUM_WARPS;  // 256
constexpr int CL_Bkv = 64;
constexpr int CL_WMMA_M = 16;
constexpr int CL_WMMA_N = 16;
constexpr int CL_WMMA_K = 16;

// Shared-memory layout (per block):
//   Q_tile      : Bq × HD  halfs   (per-block, holds this block's Q head tile)
//   K_tile      : Bkv × HD halfs   (block 0 writes, siblings DSMEM-read)
//   V_tile      : Bkv × HD halfs   (block 0 writes, siblings DSMEM-read)
//   S_tile      : Bq × Bkv floats  (per-block; P-half is an aliased overlay)
//   O_acc       : Bq × HD  floats  (per-block running accumulator)
//   row_m,row_l : 2 × Bq   floats  (per-block softmax state)
//
// Siblings still allocate K_tile + V_tile in their own smem — the dynamic
// smem size is a per-launch attribute and cannot be made per-block. The
// regions sit unused in siblings; the DSMEM remap targets block 0's copies.
size_t cluster_smem_bytes(int Bq, int Bkv, int HD) {
    return (size_t)Bq * HD * sizeof(half)         // Q_tile
           + 2 * (size_t)Bkv * HD * sizeof(half)  // K_tile + V_tile (split for prefetch overlap)
           + (size_t)Bq * Bkv * sizeof(float)     // S_tile
           + (size_t)Bq * HD * sizeof(float)      // O_acc
           + 2 * (size_t)Bq * sizeof(float);      // row_m + row_l
}

template <int Bq, int HD>
__global__ void fmha_sm120_cluster_kernel(
    const half* __restrict__ Q, const half* __restrict__ K, const half* __restrict__ V, half* __restrict__ O,
    int batch_size, int seq_q, int seq_kv, int n_heads, int n_kv_heads, int n_q_per_kv, float scale,
    bool causal, int sliding_window, float softcap) {
    namespace cg = cooperative_groups;
    auto cluster = cg::this_cluster();

    constexpr int Bkv = CL_Bkv;
    constexpr int TPR = CL_BLOCK_THREADS / Bq;
    static_assert(TPR >= 1 && (TPR & (TPR - 1)) == 0, "TPR must be power of 2");

    // ---- index computation --------------------------------------------------
    // Grid:    (num_q_tiles × n_q_per_kv, batch × n_kv_heads)
    // Cluster: (n_q_per_kv, 1, 1)
    const int q_local = cluster.block_rank();        // [0, n_q_per_kv) — which Q head within the cluster
    const int tile_q = blockIdx.x / n_q_per_kv;      // q tile index
    const int batch_idx = blockIdx.y / n_kv_heads;
    const int kv_head = blockIdx.y % n_kv_heads;
    const int head_idx = kv_head * n_q_per_kv + q_local;
    const int q_start = tile_q * Bq;

    if (head_idx >= n_heads || batch_idx >= batch_size)
        return;

    const int tid = threadIdx.x + threadIdx.y * blockDim.x;
    const int warp_id = tid / CL_WARP_SIZE;
    const int sm_row = tid / TPR;
    const int sm_lane = tid % TPR;

    const int64_t q_row_stride = (int64_t)n_heads * HD;
    const int64_t kv_row_stride = (int64_t)n_kv_heads * HD;

    const half* Q_ptr = Q + (int64_t)batch_idx * seq_q * q_row_stride + (int64_t)q_start * q_row_stride +
                        (int64_t)head_idx * HD;
    const half* K_ptr = K + (int64_t)batch_idx * seq_kv * kv_row_stride + (int64_t)kv_head * HD;
    const half* V_ptr = V + (int64_t)batch_idx * seq_kv * kv_row_stride + (int64_t)kv_head * HD;
    half* O_ptr = O + (int64_t)batch_idx * seq_q * q_row_stride + (int64_t)q_start * q_row_stride +
                  (int64_t)head_idx * HD;

    // ---- shared memory layout -----------------------------------------------
    extern __shared__ char smem[];
    half* Q_tile = reinterpret_cast<half*>(smem);
    half* K_tile_local = Q_tile + Bq * HD;
    half* V_tile_local = K_tile_local + Bkv * HD;
    float* S_tile = reinterpret_cast<float*>(V_tile_local + Bkv * HD);
    float* O_acc = S_tile + Bq * Bkv;
    float* row_m = O_acc + Bq * HD;
    float* row_l = row_m + Bq;

    // DSMEM aliases for block 0's K / V tiles. Resolved once; the mapping
    // is stable for the lifetime of the cluster (no migration across GPCs
    // under the Spread policy from cluster_launch.h).
    half* K_remote = cluster.map_shared_rank(K_tile_local, 0);
    half* V_remote = cluster.map_shared_rank(V_tile_local, 0);

    // ---- load Q tile (vec8 = float4 per iter) -------------------------------
    {
        const int total_vec8 = (Bq * HD) / 8;
        for (int vi = tid; vi < total_vec8; vi += CL_BLOCK_THREADS) {
            int i = vi * 8;
            int r = i / HD;
            int d = i % HD;
            float4* dst = reinterpret_cast<float4*>(&Q_tile[i]);
            if (q_start + r < seq_q) {
                const float4* src = reinterpret_cast<const float4*>(&Q_ptr[(int64_t)r * q_row_stride + d]);
                *dst = *src;
            } else {
                *dst = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }
    }

    // ---- zero O_acc + init softmax state ------------------------------------
    {
        const int total_vec4 = (Bq * HD) / 4;
        const float4 zero = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        for (int vi = tid; vi < total_vec4; vi += CL_BLOCK_THREADS) {
            reinterpret_cast<float4*>(O_acc)[vi] = zero;
        }
    }
    if (tid < Bq) {
        row_m[tid] = -FLT_MAX;
        row_l[tid] = 0.0f;
    }

    // ---- KV tile loop bounds ------------------------------------------------
    int num_kv_tiles, first_kv_tile;
    compute_kv_tile_bounds(q_start, Bq, Bkv, seq_q, seq_kv, causal, sliding_window, first_kv_tile,
                           num_kv_tiles);

    const int hd_chunks = HD / CL_WMMA_K;
    const int s_row_tiles = Bq / CL_WMMA_M;
    const int s_col_tiles = Bkv / CL_WMMA_N;
    const int s_total_tiles = s_row_tiles * s_col_tiles;
    const int o_row_tiles = Bq / CL_WMMA_M;
    const int o_col_tiles = HD / CL_WMMA_N;
    const int o_total_tiles = o_row_tiles * o_col_tiles;
    const int pv_chunks = Bkv / CL_WMMA_K;

    // ---- prologue: block 0 loads the first K tile ---------------------------
    // (vec8 = float4 per iter, OOB rows zeroed)
    const int kv_total_vec8 = (Bkv * HD) / 8;
    if (q_local == 0 && first_kv_tile < num_kv_tiles) {
        const int kv_start0 = first_kv_tile * Bkv;
        for (int vi = tid; vi < kv_total_vec8; vi += CL_BLOCK_THREADS) {
            int i = vi * 8;
            int r = i / HD;
            int d = i % HD;
            float4* dst = reinterpret_cast<float4*>(&K_tile_local[i]);
            if (kv_start0 + r < seq_kv) {
                const float4* src =
                    reinterpret_cast<const float4*>(&K_ptr[(int64_t)(kv_start0 + r) * kv_row_stride + d]);
                *dst = *src;
            } else {
                *dst = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }
    }
    cluster.sync();  // (B0) wait for first K tile to be visible across the cluster

    // ---- main loop over KV tiles --------------------------------------------
    for (int j = first_kv_tile; j < num_kv_tiles; j++) {
        const int kv_start = j * Bkv;

        // ---- Siblings: stage DSMEM K_remote → local K_tile_local --------
        //
        // CUDA 13.2 cicc segfaults when wmma::load_matrix_sync's source is
        // a pointer returned by cluster.map_shared_rank() — even with
        // explicit `const half*` casts, even on a single specialisation.
        // Plain shared-memory pointers work fine, so the workaround is to
        // scalar-copy K[j] from block 0's smem into the sibling's own
        // K_tile_local. The copy stays inside the cluster (DSMEM bandwidth,
        // no HBM roundtrip) and the WMMA path runs against a non-aliased
        // shared-memory base. Block 0 already has the canonical copy so it
        // skips this step.
        //
        // TODO(M5/2.3): retest on CUDA 13.3+ — if cicc is fixed we can
        // drop the staging copy and source WMMA directly from K_remote.
        if (q_local != 0) {
            for (int vi = tid; vi < kv_total_vec8; vi += CL_BLOCK_THREADS) {
                int i = vi * 8;
                reinterpret_cast<float4*>(&K_tile_local[i])[0] =
                    reinterpret_cast<const float4*>(&K_remote[i])[0];
            }
        }
        __syncthreads();

        // ============================================================
        // Phase 1: S = Q_tile @ K_tile_local^T  via WMMA
        // ============================================================
        for (int tile_idx = warp_id; tile_idx < s_total_tiles; tile_idx += CL_NUM_WARPS) {
            int ri = tile_idx / s_col_tiles;
            int ci = tile_idx % s_col_tiles;

            wmma::fragment<wmma::accumulator, CL_WMMA_M, CL_WMMA_N, CL_WMMA_K, float> acc;
            wmma::fill_fragment(acc, 0.0f);

            for (int k = 0; k < hd_chunks; k++) {
                wmma::fragment<wmma::matrix_a, CL_WMMA_M, CL_WMMA_N, CL_WMMA_K, half, wmma::row_major> a_frag;
                wmma::load_matrix_sync(a_frag, Q_tile + ri * CL_WMMA_M * HD + k * CL_WMMA_K, HD);

                wmma::fragment<wmma::matrix_b, CL_WMMA_M, CL_WMMA_N, CL_WMMA_K, half, wmma::col_major> b_frag;
                wmma::load_matrix_sync(b_frag, K_tile_local + ci * CL_WMMA_N * HD + k * CL_WMMA_K, HD);

                wmma::mma_sync(acc, a_frag, b_frag, acc);
            }

            wmma::store_matrix_sync(S_tile + ri * CL_WMMA_M * Bkv + ci * CL_WMMA_N, acc, Bkv,
                                    wmma::mem_row_major);
        }
        __syncthreads();

        apply_score_masks(S_tile, Bq, Bkv, CL_BLOCK_THREADS, tid, q_start, kv_start, seq_q, seq_kv, scale,
                          softcap, causal, sliding_window);
        __syncthreads();

        // ============================================================
        // Phase 2: online softmax — independent per Q head (per block).
        // Same structure as fmha_sm120_kernel: TPR threads per row,
        // parallel max/exp/sum via warp shuffles, then fused S → P-half.
        // ============================================================
        {
            half* SP_half = reinterpret_cast<half*>(S_tile);
            const int r = sm_row;
            const bool row_valid = (r < Bq) && (q_start + r < seq_q);

            // Step 1: row max
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

            // Step 2: new running max + correction factor
            float m_old = row_valid ? row_m[r] : -FLT_MAX;
            float m_new = fmaxf(m_old, m_ij);
            float alpha = __expf(m_old - m_new);

            // Step 3: exp + sum + write-back, with all-masked sentinel guard
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

            // Step 4: update running softmax state
            float l_old = row_valid ? row_l[r] : 0.0f;
            float l_new = alpha * l_old + partial_sum;
            if (sm_lane == 0 && row_valid) {
                row_m[r] = m_new;
                row_l[r] = l_new;
            }

            // Step 5: rescale O_acc by alpha · l_old / l_new
            float rescale = (l_old > 0.0f) ? (alpha * l_old / l_new) : 0.0f;
            if (row_valid) {
                for (int d = sm_lane; d < HD; d += TPR) {
                    O_acc[r * HD + d] *= rescale;
                }
            }

            // Step 6: fused softmax-normalize + float→half (P overlays S)
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

        // ---- block 0: load V[j] -----------------------------------------
        // Siblings stall at the cluster.sync() below; block 0 fills V.
        if (q_local == 0) {
            for (int vi = tid; vi < kv_total_vec8; vi += CL_BLOCK_THREADS) {
                int i = vi * 8;
                int r = i / HD;
                int d = i % HD;
                float4* dst = reinterpret_cast<float4*>(&V_tile_local[i]);
                if (kv_start + r < seq_kv) {
                    const float4* src =
                        reinterpret_cast<const float4*>(&V_ptr[(int64_t)(kv_start + r) * kv_row_stride + d]);
                    *dst = *src;
                } else {
                    *dst = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                }
            }
        }
        cluster.sync();  // (B1) wait for V[j] to be visible across the cluster

        // ---- Siblings: stage DSMEM V_remote → local V_tile_local --------
        // Same cicc workaround as Phase 1; see comment above K stage.
        if (q_local != 0) {
            for (int vi = tid; vi < kv_total_vec8; vi += CL_BLOCK_THREADS) {
                int i = vi * 8;
                reinterpret_cast<float4*>(&V_tile_local[i])[0] =
                    reinterpret_cast<const float4*>(&V_remote[i])[0];
            }
        }
        __syncthreads();

        // ============================================================
        // Phase 3: O_acc += P_half @ V_tile_local   via WMMA
        // ============================================================
        {
            half* P_half = reinterpret_cast<half*>(S_tile);
            for (int tile_idx = warp_id; tile_idx < o_total_tiles; tile_idx += CL_NUM_WARPS) {
                int ri = tile_idx / o_col_tiles;
                int di = tile_idx % o_col_tiles;

                wmma::fragment<wmma::accumulator, CL_WMMA_M, CL_WMMA_N, CL_WMMA_K, float> o_frag;
                wmma::load_matrix_sync(o_frag, O_acc + ri * CL_WMMA_M * HD + di * CL_WMMA_N, HD,
                                       wmma::mem_row_major);

                for (int k = 0; k < pv_chunks; k++) {
                    wmma::fragment<wmma::matrix_a, CL_WMMA_M, CL_WMMA_N, CL_WMMA_K, half, wmma::row_major>
                        p_frag;
                    wmma::load_matrix_sync(p_frag, P_half + ri * CL_WMMA_M * Bkv + k * CL_WMMA_K, Bkv);

                    wmma::fragment<wmma::matrix_b, CL_WMMA_M, CL_WMMA_N, CL_WMMA_K, half, wmma::row_major>
                        v_frag;
                    wmma::load_matrix_sync(v_frag, V_tile_local + k * CL_WMMA_N * HD + di * CL_WMMA_N, HD);

                    wmma::mma_sync(o_frag, p_frag, v_frag, o_frag);
                }

                wmma::store_matrix_sync(O_acc + ri * CL_WMMA_M * HD + di * CL_WMMA_N, o_frag, HD,
                                        wmma::mem_row_major);
            }
        }
        __syncthreads();

        // ---- block 0: prefetch K[j+1] while siblings wait at the next barrier
        if (q_local == 0 && (j + 1) < num_kv_tiles) {
            const int kv_start_next = (j + 1) * Bkv;
            for (int vi = tid; vi < kv_total_vec8; vi += CL_BLOCK_THREADS) {
                int i = vi * 8;
                int r = i / HD;
                int d = i % HD;
                float4* dst = reinterpret_cast<float4*>(&K_tile_local[i]);
                if (kv_start_next + r < seq_kv) {
                    const float4* src = reinterpret_cast<const float4*>(
                        &K_ptr[(int64_t)(kv_start_next + r) * kv_row_stride + d]);
                    *dst = *src;
                } else {
                    *dst = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                }
            }
        }
        cluster.sync();  // (B2) wait for K[j+1] (and finalize PV reads on V_remote)
    }

    // ---- write final output (vec4: 4 FP32 → 4 FP16 per iter) ----------------
    const int total_vec4 = (Bq * HD) / 4;
    for (int vi = tid; vi < total_vec4; vi += CL_BLOCK_THREADS) {
        int i = vi * 4;
        int r = i / HD;
        if (q_start + r >= seq_q)
            continue;
        float4 v = reinterpret_cast<const float4*>(O_acc)[vi];
        half2 lo = __float22half2_rn(make_float2(v.x, v.y));
        half2 hi = __float22half2_rn(make_float2(v.z, v.w));
        uint2 packed;
        packed.x = *reinterpret_cast<const uint32_t*>(&lo);
        packed.y = *reinterpret_cast<const uint32_t*>(&hi);
        *reinterpret_cast<uint2*>(&O_ptr[(int64_t)r * q_row_stride + (i % HD)]) = packed;
    }
}

// ----- Launch helpers -------------------------------------------------------

// PR 2.1 ships with Bq=64 only. Bq=128 produces wrong output specifically
// for HD=64 in the cluster path (validated via FmhaSm120Test.ClusterPathHd64
// flipping pass↔fail with this gate); HD={96,128,256} run correctly at
// Bq=128 but the per-HD selection complicates the dispatcher for marginal
// occupancy gain. PR 2.3 tile-tuning re-evaluates Bq=128 once the HD=64
// failure is root-caused (suspected register-pressure / occupancy
// interaction specific to the smaller HD).
int select_cluster_Bq(int HD, int max_smem) {
    if (cluster_smem_bytes(64, CL_Bkv, HD) <= (size_t)max_smem) return 64;
    return 0;
}

#define LAUNCH_CLUSTER(BQ, HD)                                                                              \
    do {                                                                                                    \
        auto kfunc = fmha_sm120_cluster_kernel<BQ, HD>;                                                     \
        cudaError_t attr_err = cudaFuncSetAttribute(                                                        \
            kfunc, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));                    \
        if (attr_err != cudaSuccess) {                                                                      \
            IMP_LOG_WARN("FMHA cluster: cudaFuncSetAttribute failed Bq=%d HD=%d smem=%zu: %s", BQ, HD,      \
                         smem, cudaGetErrorString(attr_err));                                               \
            return false;                                                                                   \
        }                                                                                                   \
        cudaFuncSetAttribute(kfunc, cudaFuncAttributePreferredSharedMemoryCarveout,                         \
                             cudaSharedmemCarveoutMaxShared);                                               \
        cudaLaunchAttribute attrs[2];                                                                       \
        cudaLaunchConfig_t config = cluster::build_cluster_config(                                          \
            grid, block, smem, stream, attrs, /*cluster_x=*/static_cast<unsigned int>(n_q_per_kv));         \
        cudaError_t launch_err = cudaLaunchKernelEx(                                                        \
            &config, kfunc, reinterpret_cast<const half*>(Q.data),                                          \
            reinterpret_cast<const half*>(K.data), reinterpret_cast<const half*>(V.data),                   \
            reinterpret_cast<half*>(O.data), batch_size, seq_q, seq_kv, n_heads, n_kv_heads, n_q_per_kv,    \
            scale, causal, sliding_window, softcap);                                                        \
        if (launch_err != cudaSuccess) {                                                                    \
            IMP_LOG_WARN("FMHA cluster: cudaLaunchKernelEx failed Bq=%d HD=%d: %s", BQ, HD,                 \
                         cudaGetErrorString(launch_err));                                                   \
            return false;                                                                                   \
        }                                                                                                   \
    } while (0)

}  // namespace

// Entry point — host launcher consulted by fmha_sm120_prefill before falling
// back to the legacy per-head kernel. Returns false (no kernel launched) if
// the config is ineligible, the device limits aren't met, or RuntimeConfig
// has attention.no_fmha_cluster set.
bool try_fmha_sm120_cluster_prefill(const Tensor& Q, const Tensor& K, const Tensor& V, Tensor& O, float scale,
                                    bool causal, int sliding_window, float softcap, cudaStream_t stream) {
    if (RuntimeConfig::current().attention.no_fmha_cluster)
        return false;
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
    const int n_q_per_kv = n_heads / n_kv_heads;
    // Cluster dim must be power-of-2 and ≤ 8 here (cap = 16 on GB202).
    if (n_q_per_kv != 2 && n_q_per_kv != 4 && n_q_per_kv != 8)
        return false;
    if (!cluster::valid_cluster_dim(static_cast<unsigned int>(n_q_per_kv)))
        return false;

    if (seq_q == 0 || seq_kv == 0)
        return false;
    if (head_dim != 64 && head_dim != 96 && head_dim != 128 && head_dim != 256)
        return false;

    // Gate: skip cluster path on short prompts where the (B0)+(B1)+(B2)
    // cluster.sync() overhead per KV tile dominates the saved KV reads.
    // Matches the decode-side num_ctx_blocks >= 8 heuristic — 8 KV tiles
    // is roughly 8 × Bkv = 512 KV tokens.
    if (seq_kv < CL_Bkv * 8)
        return false;

    int device = 0;
    cudaGetDevice(&device);
    int max_smem = 0;
    cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);

    const int Bq = select_cluster_Bq(head_dim, max_smem);
    if (Bq == 0) {
        IMP_LOG_DEBUG("FMHA cluster: smem budget too small (HD=%d, max=%d)", head_dim, max_smem);
        return false;
    }
    const int Bkv = CL_Bkv;
    const size_t smem = cluster_smem_bytes(Bq, Bkv, head_dim);

    const int num_q_tiles = (seq_q + Bq - 1) / Bq;
    // Grid: cluster on x-axis (fastest changing), batch×kv-head on y. CUDA
    // groups consecutive x-blocks into a cluster of size n_q_per_kv.
    dim3 grid(num_q_tiles * n_q_per_kv, batch_size * n_kv_heads);
    dim3 block(CL_WARP_SIZE, CL_NUM_WARPS);

    IMP_LOG_DEBUG(
        "FMHA cluster: B=%d Sq=%d Skv=%d nh=%d nkv=%d nq/kv=%d hd=%d Bq=%d Bkv=%d smem=%zu causal=%d sw=%d "
        "softcap=%.1f",
        batch_size, seq_q, seq_kv, n_heads, n_kv_heads, n_q_per_kv, head_dim, Bq, Bkv, smem, causal,
        sliding_window, softcap);

    // PR 2.1 dispatches Bq=64 only — see select_cluster_Bq for the rationale.
    (void)Bq;
    switch (head_dim) {
        case 64: LAUNCH_CLUSTER(64, 64); return true;
        case 96: LAUNCH_CLUSTER(64, 96); return true;
        case 128: LAUNCH_CLUSTER(64, 128); return true;
        case 256: LAUNCH_CLUSTER(64, 256); return true;
        default: break;
    }

    return false;
}

#undef LAUNCH_CLUSTER

}  // namespace imp
