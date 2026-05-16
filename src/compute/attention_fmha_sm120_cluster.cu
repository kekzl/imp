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
// Bq selection (post-A/B investigation 2026-05-17): Bq = 128 for HD ∈
// {96, 128, 256}, Bq = 64 for HD = 64. The HD = 64 carve-out is a
// wrong-output bug (see select_cluster_Bq). FP8 variant additionally
// drops Bq = 128 for HD = 256 (extra K_fp8 slot puts smem over the 228
// KiB optin limit) so HD = 256 stays on Bq = 64 in FP8.
//
// All scheduling decisions match the M5 Slice 1 helper
// (runtime/cluster_launch.h): GPC-spread policy, power-of-2 cluster check.
//
// The new kernel is bit-identical to fmha_sm120_kernel<Bq, HD> within
// FP16/FP32 reordering tolerance — see tests/test_attention_fmha_sm120.cu
// ClusterPath* cases.
//
// !! DISABLED BY DEFAULT (2026-05-17) !!
// RuntimeConfig.attention.no_fmha_cluster defaults to true after an A/B
// sweep on the four production NVFP4 MoE models showed cluster losing
// up to -22 % on HD=256 (Qwen3.6-35B pp=2048; Gemma-4-26B pp=512). HD=128
// GQA=8 wins +6-11 % at pp=512 but is negative at pp=2048. The Spread
// scheduling policy caps concurrent clusters at the GPC count (12 on
// RTX 5090) — fewer concurrent blocks than the legacy per-(head, tile)
// kernel. The DSMEM bandwidth saving doesn't compensate.
//
// Cluster code retained for opt-in via `--set attention.no_fmha_cluster=false`
// and for future tuning passes (relaxed Spread policy, fewer cluster.sync
// barriers, smaller cluster_x via head fan-in subdivision, …). See
// m5_slice2_cluster_refuted_2026_05_17.md memo.
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

// Bq selection: prefer Bq=128 for HD ∈ {96, 128, 256} (halves the per-attn
// kernel-launch count vs Bq=64; cluster vs legacy A/B on NVFP4 MoE models
// showed cluster losing 6-20 % at pp=512 with Bq=64 in part because legacy
// dispatches HD=128 at Bq=128). HD=64 stays on Bq=64 because Bq=128 +
// HD=64 produces wrong output in the cluster kernel (see
// FmhaSm120Test.ClusterPathHd64 — bisected 2026-05-16, suspected
// register-pressure / occupancy interaction unique to the smaller HD).
int select_cluster_Bq(int HD, int max_smem) {
    if (HD != 64 && cluster_smem_bytes(128, CL_Bkv, HD) <= (size_t)max_smem) return 128;
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

// =========================================================================
// FP8 cluster kernel (Slice 2.2)
// =========================================================================
//
// Mirrors the FP16 cluster kernel above but routes QK^T through SM120 FP8
// E4M3 m16n8k32 MMA (same inline-PTX path as fmha_sm120_fp8_kernel in
// attention_fmha_sm120.cu). DSMEM still carries FP16 K so the cluster
// staging copy doesn't have to assume FP8 quantization happened correctly
// in block 0; each block converts its locally-staged FP16 K → FP8 once
// per KV tile (matching the per-block cost of the legacy FP8 kernel).
//
// Smem layout per block (vs the FP16 cluster):
//   Q_fp8         : Bq  × HD  uint8  — Q tile converted to E4M3 once at start
//   K_fp16_local  : Bkv × HD  half   — block 0 canonical, siblings stage from
//                                      K_remote each iter
//   V_fp16_local  : Bkv × HD  half   — block 0 canonical, siblings stage from
//                                      V_remote each iter
//   K_fp8_local   : Bkv × HD  uint8  — per-block, written by the local
//                                      FP16→FP8 conversion before Phase 1
//   S_tile        : Bq  × Bkv float  — score buffer (P_half overlay)
//   O_acc         : Bq  × HD  float  — output accumulator
//   row_m,row_l   : 2 × Bq    float
//
// We keep K_fp16_local *and* K_fp8_local distinct (no aliasing) so the
// per-block FP8 conversion can happen in lockstep across all blocks (block
// 0 isn't a special case for the convert step — it reads its own canonical
// K_fp16_local). The smem overhead (Bkv·HD bytes for K_fp8) is well below
// the optin limit at Bq=64 for every supported HD.

__device__ __forceinline__ uint32_t cluster_cvt_4xfp16_to_4xe4m3(const half* src) {
    uint16_t lo, hi;
    const uint32_t* src32 = reinterpret_cast<const uint32_t*>(src);
    asm volatile("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(lo) : "r"(src32[0]));
    asm volatile("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(hi) : "r"(src32[1]));
    return static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
}

size_t cluster_fp8_smem_bytes(int Bq, int Bkv, int HD) {
    return (size_t)Bq * HD * sizeof(uint8_t)        // Q_fp8
           + 2 * (size_t)Bkv * HD * sizeof(half)    // K_fp16_local + V_fp16_local
           + (size_t)Bkv * HD * sizeof(uint8_t)     // K_fp8_local
           + (size_t)Bq * Bkv * sizeof(float)       // S_tile
           + (size_t)Bq * HD * sizeof(float)        // O_acc
           + 2 * (size_t)Bq * sizeof(float);        // row_m + row_l
}

template <int Bq, int HD>
__global__ void fmha_sm120_fp8_cluster_kernel(
    const half* __restrict__ Q, const half* __restrict__ K, const half* __restrict__ V, half* __restrict__ O,
    int batch_size, int seq_q, int seq_kv, int n_heads, int n_kv_heads, int n_q_per_kv, float scale,
    bool causal, int sliding_window, float softcap) {
    namespace cg = cooperative_groups;
    auto cluster = cg::this_cluster();

    constexpr int Bkv = CL_Bkv;
    constexpr int TPR = CL_BLOCK_THREADS / Bq;
    static_assert(TPR >= 1 && (TPR & (TPR - 1)) == 0, "TPR must be power of 2");
    static_assert(HD % 32 == 0, "FP8 MMA requires HD % 32 == 0");

    const int q_local = cluster.block_rank();
    const int tile_q = blockIdx.x / n_q_per_kv;
    const int batch_idx = blockIdx.y / n_kv_heads;
    const int kv_head = blockIdx.y % n_kv_heads;
    const int head_idx = kv_head * n_q_per_kv + q_local;
    const int q_start = tile_q * Bq;

    if (head_idx >= n_heads || batch_idx >= batch_size)
        return;

    const int tid = threadIdx.x + threadIdx.y * blockDim.x;
    const int warp_id = tid / CL_WARP_SIZE;
    const int lane_id = tid % CL_WARP_SIZE;
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

    extern __shared__ char smem[];
    uint8_t* Q_fp8 = reinterpret_cast<uint8_t*>(smem);
    half* K_fp16_local = reinterpret_cast<half*>(Q_fp8 + Bq * HD);
    half* V_fp16_local = K_fp16_local + Bkv * HD;
    uint8_t* K_fp8_local = reinterpret_cast<uint8_t*>(V_fp16_local + Bkv * HD);
    float* S_tile = reinterpret_cast<float*>(K_fp8_local + Bkv * HD);
    float* O_acc = S_tile + Bq * Bkv;
    float* row_m = O_acc + Bq * HD;
    float* row_l = row_m + Bq;

    half* K_remote = cluster.map_shared_rank(K_fp16_local, 0);
    half* V_remote = cluster.map_shared_rank(V_fp16_local, 0);

    // ---- Load + convert Q tile (FP16 → FP8 E4M3, vec4 = 4 halves per cvt pair)
    {
        const int total_vec4 = (Bq * HD) / 4;
        for (int vi = tid; vi < total_vec4; vi += CL_BLOCK_THREADS) {
            int i = vi * 4;
            int r = i / HD;
            int d = i % HD;
            if (q_start + r >= seq_q) {
                reinterpret_cast<uint32_t*>(Q_fp8)[vi] = 0;
                continue;
            }
            const half* src = &Q_ptr[(int64_t)r * q_row_stride + d];
            reinterpret_cast<uint32_t*>(Q_fp8)[vi] = cluster_cvt_4xfp16_to_4xe4m3(src);
        }
    }

    // ---- Zero O_acc + init softmax state
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

    // ---- KV tile bounds
    int num_kv_tiles, first_kv_tile;
    compute_kv_tile_bounds(q_start, Bq, Bkv, seq_q, seq_kv, causal, sliding_window, first_kv_tile,
                           num_kv_tiles);

    // FP8 MMA constants
    constexpr int S_M = 16;
    constexpr int S_N = 8;
    constexpr int S_K = 32;
    const int hd_chunks_fp8 = HD / S_K;
    const int s_row_tiles = Bq / S_M;
    const int s_col_tiles_half = Bkv / S_N;
    const int s_total_tiles = s_row_tiles * s_col_tiles_half;

    // FP16 WMMA constants for PV
    const int o_row_tiles = Bq / CL_WMMA_M;
    const int o_col_tiles = HD / CL_WMMA_N;
    const int o_total_tiles = o_row_tiles * o_col_tiles;
    const int pv_chunks = Bkv / CL_WMMA_K;

    const int kv_total_vec8 = (Bkv * HD) / 8;

    // ---- Prologue: block 0 loads first K tile as FP16
    if (q_local == 0 && first_kv_tile < num_kv_tiles) {
        const int kv_start0 = first_kv_tile * Bkv;
        for (int vi = tid; vi < kv_total_vec8; vi += CL_BLOCK_THREADS) {
            int i = vi * 8;
            int r = i / HD;
            int d = i % HD;
            float4* dst = reinterpret_cast<float4*>(&K_fp16_local[i]);
            if (kv_start0 + r < seq_kv) {
                const float4* src =
                    reinterpret_cast<const float4*>(&K_ptr[(int64_t)(kv_start0 + r) * kv_row_stride + d]);
                *dst = *src;
            } else {
                *dst = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }
    }
    cluster.sync();

    for (int j = first_kv_tile; j < num_kv_tiles; j++) {
        const int kv_start = j * Bkv;

        // ---- Siblings: stage K_remote (FP16) → my K_fp16_local
        if (q_local != 0) {
            for (int vi = tid; vi < kv_total_vec8; vi += CL_BLOCK_THREADS) {
                int i = vi * 8;
                reinterpret_cast<float4*>(&K_fp16_local[i])[0] =
                    reinterpret_cast<const float4*>(&K_remote[i])[0];
            }
        }
        __syncthreads();

        // ---- All blocks: convert K_fp16_local → K_fp8_local (vec4 per cvt pair)
        {
            const int total_vec4 = (Bkv * HD) / 4;
            for (int vi = tid; vi < total_vec4; vi += CL_BLOCK_THREADS) {
                int i = vi * 4;
                reinterpret_cast<uint32_t*>(K_fp8_local)[vi] =
                    cluster_cvt_4xfp16_to_4xe4m3(&K_fp16_local[i]);
            }
        }
        __syncthreads();

        // ============================================================
        // Phase 1: S = Q_fp8 @ K_fp8_local^T via FP8 m16n8k32 MMA
        // ============================================================
        for (int tile_idx = warp_id; tile_idx < s_total_tiles; tile_idx += CL_NUM_WARPS) {
            int ri = tile_idx / s_col_tiles_half;
            int ci = tile_idx % s_col_tiles_half;

            float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;

            for (int k = 0; k < hd_chunks_fp8; k++) {
                uint32_t a0, a1, a2, a3;
                {
                    const uint8_t* q_base = Q_fp8 + ri * S_M * HD + k * S_K;
                    int row_in_tile = lane_id / 4;
                    int col_base = (lane_id % 4) * 4;
                    const uint32_t* q_row0 = reinterpret_cast<const uint32_t*>(
                        q_base + row_in_tile * HD + col_base);
                    const uint32_t* q_row8 = reinterpret_cast<const uint32_t*>(
                        q_base + (row_in_tile + 8) * HD + col_base);
                    a0 = q_row0[0];
                    a1 = q_row0[4];
                    a2 = q_row8[0];
                    a3 = q_row8[4];
                }

                uint32_t b0, b1;
                {
                    const uint8_t* k_base = K_fp8_local + ci * S_N * HD + k * S_K;
                    int col_in_tile = lane_id / 4;
                    int k_base_offset = (lane_id % 4) * 4;
                    const uint32_t* k_ptr0 = reinterpret_cast<const uint32_t*>(
                        k_base + col_in_tile * HD + k_base_offset);
                    b0 = k_ptr0[0];
                    b1 = k_ptr0[4];
                }

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

            // Write 16×8 result to S_tile using canonical m16n8 mapping
            {
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

        apply_score_masks(S_tile, Bq, Bkv, CL_BLOCK_THREADS, tid, q_start, kv_start, seq_q, seq_kv, scale,
                          softcap, causal, sliding_window);
        __syncthreads();

        // ============================================================
        // Phase 2: online softmax (per-block; identical structure to FP16 cluster)
        // ============================================================
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
                for (int d = sm_lane; d < HD; d += TPR)
                    O_acc[r * HD + d] *= rescale;
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

        // ---- Block 0: load V[j] (FP16) → V_fp16_local
        if (q_local == 0) {
            for (int vi = tid; vi < kv_total_vec8; vi += CL_BLOCK_THREADS) {
                int i = vi * 8;
                int r = i / HD;
                int d = i % HD;
                float4* dst = reinterpret_cast<float4*>(&V_fp16_local[i]);
                if (kv_start + r < seq_kv) {
                    const float4* src =
                        reinterpret_cast<const float4*>(&V_ptr[(int64_t)(kv_start + r) * kv_row_stride + d]);
                    *dst = *src;
                } else {
                    *dst = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                }
            }
        }
        cluster.sync();

        // ---- Siblings: stage V_remote → V_fp16_local
        if (q_local != 0) {
            for (int vi = tid; vi < kv_total_vec8; vi += CL_BLOCK_THREADS) {
                int i = vi * 8;
                reinterpret_cast<float4*>(&V_fp16_local[i])[0] =
                    reinterpret_cast<const float4*>(&V_remote[i])[0];
            }
        }
        __syncthreads();

        // ============================================================
        // Phase 3: O_acc += P @ V_fp16_local via FP16 WMMA
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
                    wmma::load_matrix_sync(v_frag, V_fp16_local + k * CL_WMMA_N * HD + di * CL_WMMA_N, HD);

                    wmma::mma_sync(o_frag, p_frag, v_frag, o_frag);
                }

                wmma::store_matrix_sync(O_acc + ri * CL_WMMA_M * HD + di * CL_WMMA_N, o_frag, HD,
                                        wmma::mem_row_major);
            }
        }
        __syncthreads();

        // ---- Block 0: prefetch K[j+1] (FP16) → K_fp16_local
        if (q_local == 0 && (j + 1) < num_kv_tiles) {
            const int kv_start_next = (j + 1) * Bkv;
            for (int vi = tid; vi < kv_total_vec8; vi += CL_BLOCK_THREADS) {
                int i = vi * 8;
                int r = i / HD;
                int d = i % HD;
                float4* dst = reinterpret_cast<float4*>(&K_fp16_local[i]);
                if (kv_start_next + r < seq_kv) {
                    const float4* src = reinterpret_cast<const float4*>(
                        &K_ptr[(int64_t)(kv_start_next + r) * kv_row_stride + d]);
                    *dst = *src;
                } else {
                    *dst = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                }
            }
        }
        cluster.sync();
    }

    // ---- Write final output (vec4 = 4 FP32 → 4 FP16 per iter)
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

int select_cluster_fp8_Bq(int HD, int max_smem) {
    // Same Bq policy as the FP16 cluster, but: HD=256 at Bq=128 doesn't fit
    // the FP8 layout (extra Bkv·HD K_fp8 slot) → falls back to Bq=64.
    // HD=64 stays on Bq=64 by the same wrong-output carve-out.
    if (HD != 64 && cluster_fp8_smem_bytes(128, CL_Bkv, HD) <= (size_t)max_smem) return 128;
    if (cluster_fp8_smem_bytes(64, CL_Bkv, HD) <= (size_t)max_smem) return 64;
    return 0;
}

#define LAUNCH_FP8_CLUSTER(BQ, HD)                                                                          \
    do {                                                                                                    \
        auto kfunc = fmha_sm120_fp8_cluster_kernel<BQ, HD>;                                                 \
        cudaError_t attr_err = cudaFuncSetAttribute(                                                        \
            kfunc, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));                    \
        if (attr_err != cudaSuccess) {                                                                      \
            IMP_LOG_WARN("FMHA FP8 cluster: cudaFuncSetAttribute failed Bq=%d HD=%d smem=%zu: %s", BQ, HD,  \
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
            IMP_LOG_WARN("FMHA FP8 cluster: cudaLaunchKernelEx failed Bq=%d HD=%d: %s", BQ, HD,             \
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

    // Bq=128 for HD ∈ {96, 128, 256}, Bq=64 for HD=64. See select_cluster_Bq.
    if (Bq == 128) {
        switch (head_dim) {
            case 96: LAUNCH_CLUSTER(128, 96); return true;
            case 128: LAUNCH_CLUSTER(128, 128); return true;
            case 256: LAUNCH_CLUSTER(128, 256); return true;
            default: break;
        }
    }
    switch (head_dim) {
        case 64: LAUNCH_CLUSTER(64, 64); return true;
        case 96: LAUNCH_CLUSTER(64, 96); return true;
        case 128: LAUNCH_CLUSTER(64, 128); return true;
        case 256: LAUNCH_CLUSTER(64, 256); return true;
        default: break;
    }

    return false;
}

// Sibling of try_fmha_sm120_cluster_prefill for the FP8 score-compute
// variant — consulted by fmha_sm120_fp8_prefill before the legacy
// per-head FP8 kernel. Same gate, same opt-out flag, same Bq=64 policy.
bool try_fmha_sm120_fp8_cluster_prefill(const Tensor& Q, const Tensor& K, const Tensor& V, Tensor& O,
                                        float scale, bool causal, int sliding_window, float softcap,
                                        cudaStream_t stream) {
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
    if (n_q_per_kv != 2 && n_q_per_kv != 4 && n_q_per_kv != 8)
        return false;
    if (!cluster::valid_cluster_dim(static_cast<unsigned int>(n_q_per_kv)))
        return false;

    if (seq_q == 0 || seq_kv == 0)
        return false;
    if (head_dim % 32 != 0)
        return false;  // FP8 MMA m16n8k32 requires HD % 32 == 0
    if (head_dim != 64 && head_dim != 96 && head_dim != 128 && head_dim != 256)
        return false;
    if (head_dim == 96)
        return false;  // 96 % 32 != 0 — FP8 MMA can't dispatch

    if (seq_kv < CL_Bkv * 8)
        return false;

    int device = 0;
    cudaGetDevice(&device);
    int max_smem = 0;
    cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);

    const int Bq = select_cluster_fp8_Bq(head_dim, max_smem);
    if (Bq == 0) {
        IMP_LOG_DEBUG("FMHA FP8 cluster: smem budget too small (HD=%d, max=%d)", head_dim, max_smem);
        return false;
    }
    const int Bkv = CL_Bkv;
    const size_t smem = cluster_fp8_smem_bytes(Bq, Bkv, head_dim);

    const int num_q_tiles = (seq_q + Bq - 1) / Bq;
    dim3 grid(num_q_tiles * n_q_per_kv, batch_size * n_kv_heads);
    dim3 block(CL_WARP_SIZE, CL_NUM_WARPS);

    IMP_LOG_DEBUG(
        "FMHA FP8 cluster: B=%d Sq=%d Skv=%d nh=%d nkv=%d nq/kv=%d hd=%d Bq=%d Bkv=%d smem=%zu causal=%d "
        "sw=%d softcap=%.1f",
        batch_size, seq_q, seq_kv, n_heads, n_kv_heads, n_q_per_kv, head_dim, Bq, Bkv, smem, causal,
        sliding_window, softcap);

    // Bq=128 only for HD=128 in the FP8 path (HD=64 carve-out; HD=256 smem).
    if (Bq == 128 && head_dim == 128) {
        LAUNCH_FP8_CLUSTER(128, 128);
        return true;
    }
    switch (head_dim) {
        case 64: LAUNCH_FP8_CLUSTER(64, 64); return true;
        case 128: LAUNCH_FP8_CLUSTER(64, 128); return true;
        case 256: LAUNCH_FP8_CLUSTER(64, 256); return true;
        default: break;
    }

    return false;
}

#undef LAUNCH_CLUSTER
#undef LAUNCH_FP8_CLUSTER

}  // namespace imp
