#include "compute/moe_routing.h"
#include "compute/moe_routing_internal.cuh"
#include "compute/warp_reduce.cuh"
#include "core/logging.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cfloat>
#include <cstdio>
#include <cstring>

namespace imp {

// Top-k gating: one block per token, processes gate_logits[n_experts], softmax, top-k
// select, normalize weights.
// expert_indices[token*top_k+j] = j-th selected expert; expert_weights[...] = its normalized weight.

__global__ void topk_gating_kernel(const float* __restrict__ gate_logits, int n_experts, int top_k,
                                   int32_t* __restrict__ expert_indices, float* __restrict__ expert_weights,
                                   bool use_sigmoid, bool normalize_weights,
                                   const half* __restrict__ score_bias) {
    const int token = blockIdx.x;
    const int tid = threadIdx.x;
    const float* logits = gate_logits + static_cast<int64_t>(token) * n_experts;

    constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
    __shared__ float s_warp[NUM_WARPS];
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    extern __shared__ char smem_raw[];
    float* s_probs = reinterpret_cast<float*>(smem_raw);
    // When using score_bias, we need a second array for selection scores
    // s_probs holds unbiased scores (used for weight values)
    // s_sel_probs holds biased scores (used for top-k selection)
    float* s_sel_probs = s_probs + (score_bias ? n_experts : 0);
    float* s_topk_val = s_probs + (score_bias ? 2 * n_experts : n_experts);
    int32_t* s_topk_idx = reinterpret_cast<int32_t*>(s_topk_val + top_k);

    if (use_sigmoid) {
        // --- Sigmoid gating (Nemotron-H): prob_i = sigmoid(logit_i) ---
        for (int i = tid; i < n_experts; i += blockDim.x) {
            float p = 1.0f / (1.0f + expf(-logits[i]));
            s_probs[i] = p;
            if (score_bias) {
                // Bias is added to sigmoid outputs for selection only
                s_sel_probs[i] = p + __half2float(score_bias[i]);
            }
        }
    } else {
        // --- Softmax gating (Mixtral, DeepSeek, etc.) ---
        // Step 1: Find max for numerical stability
        float local_max = -FLT_MAX;
        for (int i = tid; i < n_experts; i += blockDim.x) {
            local_max = fmaxf(local_max, logits[i]);
        }
        local_max = warp_reduce_max(local_max);

        if (lane_id == 0)
            s_warp[warp_id] = local_max;
        __syncthreads();
        if (warp_id == 0) {
            float val = (lane_id < NUM_WARPS) ? s_warp[lane_id] : -FLT_MAX;
            val = warp_reduce_max(val);
            if (lane_id == 0)
                s_warp[0] = val;
        }
        __syncthreads();
        float gmax = s_warp[0];
        // Every warp must read the max before warp 0 overwrites slot 0 with its partial sum:
        // without this barrier a late warp took a sum as the max and top-k picked wrong experts.
        __syncthreads();

        // Step 2: Compute exp and sum for softmax
        float local_sum = 0.0f;
        for (int i = tid; i < n_experts; i += blockDim.x) {
            local_sum += expf(logits[i] - gmax);
        }
        local_sum = warp_reduce_sum(local_sum);

        if (lane_id == 0)
            s_warp[warp_id] = local_sum;
        __syncthreads();
        if (warp_id == 0) {
            float val = (lane_id < NUM_WARPS) ? s_warp[lane_id] : 0.0f;
            val = warp_reduce_sum(val);
            if (lane_id == 0)
                s_warp[0] = val;
        }
        __syncthreads();
        float inv_sum = 1.0f / s_warp[0];

        // Step 3: Compute softmax probabilities
        for (int i = tid; i < n_experts; i += blockDim.x) {
            s_probs[i] = expf(logits[i] - gmax) * inv_sum;
        }
    }
    __syncthreads();

    // Parallel top-k: block-wide argmax reduction, each iteration finds global max, records,
    // masks it out. With score_bias: select on biased scores (s_sel_probs) but weight values
    // use UNBIASED scores (s_probs).
    {
        // Shared memory for argmax reduction across warps
        __shared__ float s_warp_max[NUM_WARPS];
        __shared__ int s_warp_argmax[NUM_WARPS];

        const float* sel = score_bias ? s_sel_probs : s_probs;

        // Strided ownership tid + j*BLOCK_SIZE (moe_routing_internal.cuh): one expert per thread
        // capped the search at 255 on the 512-expert Qwen3.8-Flash-Next router.
        float my_vals[kTopkSlotsPerThread];
        topk_load_slots(sel, n_experts, tid, my_vals);

        for (int k = 0; k < top_k; k++) {
            float wmax;
            int widx;
            topk_owned_argmax(my_vals, tid, wmax, widx);
#pragma unroll
            for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
                float other_val = __shfl_xor_sync(0xFFFFFFFF, wmax, off);
                int other_idx = __shfl_xor_sync(0xFFFFFFFF, widx, off);
                if (other_val > wmax) {
                    wmax = other_val;
                    widx = other_idx;
                }
            }

            // Write per-warp results
            if (lane_id == 0) {
                s_warp_max[warp_id] = wmax;
                s_warp_argmax[warp_id] = widx;
            }
            __syncthreads();

            // First warp reduces across all warps
            if (warp_id == 0) {
                float v = (lane_id < NUM_WARPS) ? s_warp_max[lane_id] : -FLT_MAX;
                int ix = (lane_id < NUM_WARPS) ? s_warp_argmax[lane_id] : -1;
#pragma unroll
                for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
                    float ov = __shfl_xor_sync(0xFFFFFFFF, v, off);
                    int oi = __shfl_xor_sync(0xFFFFFFFF, ix, off);
                    if (ov > v) {
                        v = ov;
                        ix = oi;
                    }
                }
                if (lane_id == 0) {
                    s_topk_idx[k] = ix;
                    // Store UNBIASED weight for the selected expert
                    s_topk_val[k] = score_bias ? s_probs[ix] : v;
                }
            }
            __syncthreads();

            // Mask out the selected expert so it won't be picked again
            topk_clear_slot(my_vals, s_topk_idx[k], tid);
        }

        // Thread 0: normalize weights and write output
        if (tid == 0) {
            float multiplier = 1.0f;
            if (normalize_weights) {
                float norm = 0.0f;
                for (int j = 0; j < top_k; ++j)
                    norm += s_topk_val[j];
                multiplier = (norm > 0.0f) ? (1.0f / norm) : 1.0f;
            }
            int base = token * top_k;
            for (int j = 0; j < top_k; ++j) {
                expert_indices[base + j] = s_topk_idx[j];
                expert_weights[base + j] = s_topk_val[j] * multiplier;
            }
        }
    }
}

// Fused gate GEMV + softmax/sigmoid + top-k (decode M=1): avoids intermediate FP32 logits
// buffer + 1 kernel launch. 1 block x 256 threads (8 warps); each warp computes
// ceil(n_experts/8) dot products into shared logits, then all threads cooperate on
// softmax/sigmoid + top-k (same algorithm as topk_gating_kernel).

__global__ void gemv_gate_topk_fused_kernel(const half* __restrict__ W_gate,  // [n_experts, d_model] FP16
                                            const half* __restrict__ x,       // [d_model] FP16 input
                                            int n_experts, int d_model, int top_k,
                                            int32_t* __restrict__ expert_indices,  // [top_k] output
                                            float* __restrict__ expert_weights,    // [top_k] output
                                            bool use_sigmoid, bool normalize_weights,
                                            const half* __restrict__ score_bias) {
    const int tid = threadIdx.x;
    constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    // Shared memory layout:
    //   s_logits[n_experts]       — gate logit / probability values
    //   s_sel_probs[n_experts]    — biased selection probs (only if score_bias)
    //   s_topk_val[top_k]        — selected top-k values
    //   s_topk_idx[top_k]        — selected top-k indices
    //   s_warp_max[NUM_WARPS]    — warp reduction scratch
    //   s_warp_argmax[NUM_WARPS] — warp reduction scratch
    extern __shared__ char smem_raw[];
    float* s_logits = reinterpret_cast<float*>(smem_raw);
    float* s_sel_probs = s_logits + (score_bias ? n_experts : 0);
    float* s_topk_val = s_logits + (score_bias ? 2 * n_experts : n_experts);
    int32_t* s_topk_idx = reinterpret_cast<int32_t*>(s_topk_val + top_k);
    float* s_warp_scratch = reinterpret_cast<float*>(s_topk_idx + top_k);
    int* s_warp_argmax = reinterpret_cast<int*>(s_warp_scratch + NUM_WARPS);

    // ---- Phase 1: Gate GEMV — compute dot(W_gate[e], x) for all experts ----
    const int K2 = d_model / 2;
    const half2* x2 = reinterpret_cast<const half2*>(x);

    for (int e = warp_id; e < n_experts; e += NUM_WARPS) {
        const half2* W2 = reinterpret_cast<const half2*>(W_gate + static_cast<size_t>(e) * d_model);
        float sum = 0.0f;

        for (int i = lane_id; i < K2; i += WARP_SIZE) {
            half2 w = W2[i];
            half2 v = x2[i];
            sum += __half2float(w.x) * __half2float(v.x);
            sum += __half2float(w.y) * __half2float(v.y);
        }

// Warp shuffle reduction
#pragma unroll
        for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
            sum += __shfl_down_sync(0xFFFFFFFF, sum, off);

        if (lane_id == 0)
            s_logits[e] = sum;
    }
    __syncthreads();

    // ---- Phase 2: Softmax or sigmoid ----
    if (use_sigmoid) {
        for (int i = tid; i < n_experts; i += blockDim.x) {
            float p = 1.0f / (1.0f + expf(-s_logits[i]));
            s_logits[i] = p;  // overwrite logit with prob
            if (score_bias) {
                s_sel_probs[i] = p + __half2float(score_bias[i]);
            }
        }
    } else {
        // Softmax: find max
        float local_max = -FLT_MAX;
        for (int i = tid; i < n_experts; i += blockDim.x) {
            local_max = fmaxf(local_max, s_logits[i]);
        }
        local_max = warp_reduce_max(local_max);
        if (lane_id == 0)
            s_warp_scratch[warp_id] = local_max;
        __syncthreads();
        if (warp_id == 0) {
            float val = (lane_id < NUM_WARPS) ? s_warp_scratch[lane_id] : -FLT_MAX;
            val = warp_reduce_max(val);
            if (lane_id == 0)
                s_warp_scratch[0] = val;
        }
        __syncthreads();
        float gmax = s_warp_scratch[0];
        // Same barrier as topk_gating_kernel: slot 0 is rewritten with a partial sum below.
        __syncthreads();

        // Compute exp and sum
        float local_sum = 0.0f;
        for (int i = tid; i < n_experts; i += blockDim.x) {
            local_sum += expf(s_logits[i] - gmax);
        }
        local_sum = warp_reduce_sum(local_sum);
        if (lane_id == 0)
            s_warp_scratch[warp_id] = local_sum;
        __syncthreads();
        if (warp_id == 0) {
            float val = (lane_id < NUM_WARPS) ? s_warp_scratch[lane_id] : 0.0f;
            val = warp_reduce_sum(val);
            if (lane_id == 0)
                s_warp_scratch[0] = val;
        }
        __syncthreads();
        float inv_sum = 1.0f / s_warp_scratch[0];

        // Normalize
        for (int i = tid; i < n_experts; i += blockDim.x) {
            s_logits[i] = expf(s_logits[i] - gmax) * inv_sum;
        }
    }
    __syncthreads();

    // ---- Phase 3: Top-k selection (same algorithm as topk_gating_kernel) ----
    {
        const float* sel = score_bias ? s_sel_probs : s_logits;
        float my_vals[kTopkSlotsPerThread];
        topk_load_slots(sel, n_experts, tid, my_vals);

        for (int k = 0; k < top_k; k++) {
            float wmax;
            int widx;
            topk_owned_argmax(my_vals, tid, wmax, widx);
#pragma unroll
            for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
                float other_val = __shfl_xor_sync(0xFFFFFFFF, wmax, off);
                int other_idx = __shfl_xor_sync(0xFFFFFFFF, widx, off);
                if (other_val > wmax) {
                    wmax = other_val;
                    widx = other_idx;
                }
            }

            if (lane_id == 0) {
                s_warp_scratch[warp_id] = wmax;
                s_warp_argmax[warp_id] = widx;
            }
            __syncthreads();

            if (warp_id == 0) {
                float v = (lane_id < NUM_WARPS) ? s_warp_scratch[lane_id] : -FLT_MAX;
                int ix = (lane_id < NUM_WARPS) ? s_warp_argmax[lane_id] : -1;
#pragma unroll
                for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
                    float ov = __shfl_xor_sync(0xFFFFFFFF, v, off);
                    int oi = __shfl_xor_sync(0xFFFFFFFF, ix, off);
                    if (ov > v) {
                        v = ov;
                        ix = oi;
                    }
                }
                if (lane_id == 0) {
                    s_topk_idx[k] = ix;
                    s_topk_val[k] = score_bias ? s_logits[ix] : v;
                }
            }
            __syncthreads();

            topk_clear_slot(my_vals, s_topk_idx[k], tid);
        }

        // Thread 0: normalize weights and write output
        if (tid == 0) {
            float multiplier = 1.0f;
            if (normalize_weights) {
                float norm = 0.0f;
                for (int j = 0; j < top_k; ++j)
                    norm += s_topk_val[j];
                multiplier = (norm > 0.0f) ? (1.0f / norm) : 1.0f;
            }
            for (int j = 0; j < top_k; ++j) {
                expert_indices[j] = s_topk_idx[j];
                expert_weights[j] = s_topk_val[j] * multiplier;
            }
        }
    }
}

// Multi-CTA stable permute (#2465): flat idx -> offsets[e] + rank among earlier flat idx of expert e,
// the serial-walk layout of #1546. CTA t owns tile t: pass 1 writes per-tile histograms, pass 2
// scans them per CTA and ranks inside each warp's sub-range with __match_any_sync.
static constexpr int kPermuteWarps = BLOCK_SIZE / WARP_SIZE;
static constexpr int kPermuteMinTile = 1024;
static constexpr int kPermuteMaxTiles = 128;
// Scatter smem = ((kPermuteWarps + 2) * E + kPermuteWarps) * 4 B <= 48 KiB default -> E <= 1228.
static constexpr int kPermuteMaxExperts = 1024;

// Tile = multiple of BLOCK_SIZE (each warp sub-range a multiple of 32), at most kPermuteMaxTiles tiles.
static int permute_tile_elems(int total) {
    int tile = (total + kPermuteMaxTiles - 1) / kPermuteMaxTiles;
    tile = (tile + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
    return tile < kPermuteMinTile ? kPermuteMinTile : tile;
}

static __global__ void __launch_bounds__(BLOCK_SIZE) moe_permute_tile_hist_kernel(
    const int32_t* __restrict__ expert_indices, int total, int n_experts, int tile,
    int32_t* __restrict__ tile_counts) {
    extern __shared__ int32_t s_hist[];
    for (int e = threadIdx.x; e < n_experts; e += blockDim.x)
        s_hist[e] = 0;
    __syncthreads();
    const int beg = static_cast<int>(blockIdx.x) * tile;
    const int end = min(beg + tile, total);
    for (int i = beg + static_cast<int>(threadIdx.x); i < end; i += blockDim.x)
        atomicAdd(&s_hist[expert_indices[i]], 1);
    __syncthreads();
    int32_t* out = tile_counts + (static_cast<ptrdiff_t>(blockIdx.x) * n_experts);
    for (int e = threadIdx.x; e < n_experts; e += blockDim.x)
        out[e] = s_hist[e];
}

static __global__ void __launch_bounds__(BLOCK_SIZE) moe_permute_scatter_kernel(
    const int32_t* __restrict__ expert_indices, int total, int top_k, int n_experts, int tile,
    const int32_t* __restrict__ tile_counts, int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ sorted_flat_idx, int32_t* __restrict__ expert_offsets,
    int32_t* __restrict__ token_to_expanded) {
    extern __shared__ int32_t smem[];
    int32_t* s_warp = smem;                                                        // [kPermuteWarps][E]
    int32_t* s_base = smem + (static_cast<ptrdiff_t>(kPermuteWarps) * n_experts);  // [E]
    int32_t* s_pre = s_base + n_experts;                                           // [E]
    int32_t* s_wsum = s_pre + n_experts;                                           // [kPermuteWarps]
    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid % WARP_SIZE;
    const int warp = tid / WARP_SIZE;
    const int t = static_cast<int>(blockIdx.x);
    const int n_tiles = static_cast<int>(gridDim.x);

    for (int i = tid; i < kPermuteWarps * n_experts; i += BLOCK_SIZE)
        s_warp[i] = 0;
    // Bucket size over all tiles, and the rows earlier tiles put in each bucket.
    for (int e = tid; e < n_experts; e += BLOCK_SIZE) {
        int tot = 0;
        int pre = 0;
        for (int u = 0; u < n_tiles; ++u) {
            const int c = tile_counts[(static_cast<ptrdiff_t>(u) * n_experts) + e];
            tot += c;
            pre += (u < t) ? c : 0;
        }
        s_base[e] = tot;
        s_pre[e] = pre;
    }
    __syncthreads();

    // Per-warp counts over the warp's contiguous sub-range (counts are order-independent).
    const int sub = tile / kPermuteWarps;
    const int wbeg = (t * tile) + (warp * sub);
    const int wend = min(wbeg + sub, total);
    for (int i = wbeg + lane; i < wend; i += WARP_SIZE)
        atomicAdd(&s_warp[(warp * n_experts) + expert_indices[i]], 1);

    // Exclusive scan of bucket sizes: thread owns experts [e0, e1), warp shuffle + warp sums.
    const int per = (n_experts + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int e0 = min(tid * per, n_experts);
    const int e1 = min(e0 + per, n_experts);
    int local = 0;
    for (int e = e0; e < e1; ++e)
        local += s_base[e];
    int incl = local;
#pragma unroll
    for (int off = 1; off < WARP_SIZE; off <<= 1) {
        const int v = __shfl_up_sync(0xffffffffu, incl, off);
        if (lane >= off)
            incl += v;
    }
    if (lane == WARP_SIZE - 1)
        s_wsum[warp] = incl;
    __syncthreads();
    int run = incl - local;
    for (int w = 0; w < warp; ++w)
        run += s_wsum[w];
    for (int e = e0; e < e1; ++e) {
        const int c = s_base[e];
        s_base[e] = run;
        if (t == 0)
            expert_offsets[e] = run;
        run += c;
    }
    if (t == 0 && tid == BLOCK_SIZE - 1)
        expert_offsets[n_experts] = run;
    __syncthreads();

    // Each warp's first slot per bucket: bucket start + earlier tiles + earlier warps of this tile.
    for (int e = tid; e < n_experts; e += BLOCK_SIZE) {
        int r = s_base[e] + s_pre[e];
        for (int w = 0; w < kPermuteWarps; ++w) {
            const int c = s_warp[(w * n_experts) + e];
            s_warp[(w * n_experts) + e] = r;
            r += c;
        }
    }
    __syncthreads();

    // Warp walks its sub-range 32 at a time: slot = warp cursor + rank among equal lower lanes.
    int32_t* cursor = s_warp + (static_cast<ptrdiff_t>(warp) * n_experts);
    const unsigned lower = (1u << lane) - 1u;
    for (int b = wbeg; b < wend; b += WARP_SIZE) {
        const int idx = b + lane;
        const bool valid = idx < wend;
        const int expert = valid ? expert_indices[idx] : -1;
        const unsigned peers = __match_any_sync(0xffffffffu, expert);
        const int rank = __popc(peers & lower);
        if (valid) {
            const int dest = cursor[expert] + rank;
            sorted_token_ids[dest] = idx / top_k;
            sorted_flat_idx[dest] = idx;
            if (token_to_expanded)
                token_to_expanded[idx] = dest;
        }
        __syncwarp();
        if (valid && rank == 0)
            cursor[expert] += __popc(peers);
        __syncwarp();
    }
}

// Single-CTA stable permute, same layout as the multi-CTA pair: only for n_experts >
// kPermuteMaxExperts (scatter smem over 48 KiB) or no scratch.

__global__ void __launch_bounds__(256) moe_fused_permute_deterministic_kernel(
    const int32_t* __restrict__ expert_indices, int n_tokens, int top_k, int n_experts,
    int32_t* __restrict__ sorted_token_ids, int32_t* __restrict__ sorted_flat_idx,
    int32_t* __restrict__ expert_offsets, int32_t* __restrict__ token_to_expanded) {
    extern __shared__ int32_t smem[];
    int32_t* s_counts = smem;
    int32_t* s_write_pos = smem + n_experts;

    const int tid = threadIdx.x;
    const int total = n_tokens * top_k;

    // Phase 1: Zero counts
    for (int i = tid; i < n_experts; i += blockDim.x)
        s_counts[i] = 0;
    __syncthreads();

    // Phase 2: Count tokens per expert (atomics in shared memory — counts are
    // order-independent, so this stays parallel).
    for (int i = tid; i < total; i += blockDim.x) {
        int expert = expert_indices[i];
        atomicAdd(&s_counts[expert], 1);
    }
    __syncthreads();

    // Phase 3: Exclusive scan + initialize write positions (thread 0).
    if (tid == 0) {
        int32_t running = 0;
        for (int i = 0; i < n_experts; i++) {
            expert_offsets[i] = running;
            s_write_pos[i] = 0;
            running += s_counts[i];
        }
        expert_offsets[n_experts] = running;
    }
    __syncthreads();

    // Deterministic scatter: a token's slot within its expert bucket = its rank among earlier
    // flat indices routed to the same expert (#1546), making layout independent of scheduling.
    // One blockDim-sized chunk at a time, chunks in index order, threads counting only lower
    // thread ids -> identical layout to a full single-thread walk.
    // #2218 bounded: 2 * n_experts < 49152 B / 4 = 12288 (smem bytes moe_routing.cu:602,720, no opt-in:
    // launch fails above the 48 KiB default dynamic smem)
    int32_t* s_chunk = smem + static_cast<ptrdiff_t>(2 * n_experts);  // [blockDim.x]
    const int block_n = static_cast<int>(blockDim.x);
    for (int base = 0; base < total; base += block_n) {
        const int idx = base + tid;
        const int expert = (idx < total) ? expert_indices[idx] : -1;
        s_chunk[tid] = expert;
        __syncthreads();

        if (idx < total) {
            int rank = 0;
            for (int j = 0; j < tid; j++)
                rank += (s_chunk[j] == expert) ? 1 : 0;
            const int dest = expert_offsets[expert] + s_write_pos[expert] + rank;
            sorted_token_ids[dest] = idx / top_k;
            sorted_flat_idx[dest] = idx;
            if (token_to_expanded)
                token_to_expanded[idx] = dest;
        }
        __syncthreads();

        // Advance the write positions by what this chunk consumed, so the next
        // chunk starts where this one ended.
        const int chunk_n = (total - base) < block_n ? (total - base) : block_n;
        for (int e = tid; e < n_experts; e += block_n) {
            int c = 0;
            for (int j = 0; j < chunk_n; j++)
                c += (s_chunk[j] == e) ? 1 : 0;
            s_write_pos[e] += c;
        }
        __syncthreads();
    }
}

size_t moe_permute_scratch_ints(int n_experts) {
    return n_experts > kPermuteMaxExperts ? 0 : static_cast<size_t>(kPermuteMaxTiles) * n_experts;
}

void moe_permute(const int32_t* expert_indices, int n_tokens, int top_k, int n_experts,
                 int32_t* sorted_token_ids, int32_t* sorted_flat_idx, int32_t* expert_offsets,
                 int32_t* token_to_expanded, int32_t* scratch, cudaStream_t stream) {
    const int total = n_tokens * top_k;
    if (n_experts > kPermuteMaxExperts || scratch == nullptr) {
        const size_t smem = ((static_cast<size_t>(n_experts) * 2) + BLOCK_SIZE) * sizeof(int32_t);
        moe_fused_permute_deterministic_kernel<<<1, BLOCK_SIZE, smem, stream>>>(
            expert_indices, n_tokens, top_k, n_experts, sorted_token_ids, sorted_flat_idx, expert_offsets,
            token_to_expanded);
        IMP_CUDA_CHECK_LAUNCH();
        return;
    }
    const int tile = permute_tile_elems(total);
    const int n_tiles = total > 0 ? (total + tile - 1) / tile : 1;
    const size_t smem_hist = static_cast<size_t>(n_experts) * sizeof(int32_t);
    moe_permute_tile_hist_kernel<<<n_tiles, BLOCK_SIZE, smem_hist, stream>>>(expert_indices, total, n_experts,
                                                                             tile, scratch);
    IMP_CUDA_CHECK_LAUNCH();
    const size_t smem_scatter = ((static_cast<size_t>(kPermuteWarps + 2) * n_experts) + kPermuteWarps) *
                                sizeof(int32_t);
    moe_permute_scatter_kernel<<<n_tiles, BLOCK_SIZE, smem_scatter, stream>>>(expert_indices, total, top_k,
                                                                              n_experts, tile, scratch,
                                                                              sorted_token_ids,
                                                                              sorted_flat_idx, expert_offsets,
                                                                              token_to_expanded);
    IMP_CUDA_CHECK_LAUNCH();
}

// ============================================================================
// Helper to set up a Tensor descriptor
// ============================================================================

static Tensor make_tensor_1d(void* data, QType dtype, int64_t size, bool on_device) {
    Tensor t;
    t.data = data;
    t.qtype = dtype;
    t.ndim = 1;
    t.shape[0] = size;
    t.shape[1] = 0;
    t.shape[2] = 0;
    t.shape[3] = 0;
    t.stride[0] = 1;
    t.stride[1] = 0;
    t.stride[2] = 0;
    t.stride[3] = 0;
    t.on_device = on_device;
    return t;
}

static Tensor make_tensor_2d(void* data, QType dtype, int64_t d0, int64_t d1, bool on_device) {
    Tensor t;
    t.data = data;
    t.qtype = dtype;
    t.ndim = 2;
    t.shape[0] = d0;
    t.shape[1] = d1;
    t.shape[2] = 0;
    t.shape[3] = 0;
    t.stride[0] = d1;
    t.stride[1] = 1;
    t.stride[2] = 0;
    t.stride[3] = 0;
    t.on_device = on_device;
    return t;
}

// ============================================================================
// Public API: moe_topk_gating
// ============================================================================

void moe_topk_gating(const Tensor& gate_logits, int top_k, MoeRoutingResult& result, cudaStream_t stream,
                     bool use_sigmoid, bool normalize_weights, const void* score_bias) {
    const int n_tokens = static_cast<int>(gate_logits.shape[0]);
    const int n_experts = static_cast<int>(gate_logits.shape[1]);
    const float* d_logits = static_cast<const float*>(gate_logits.data);
    const int total_assignments = n_tokens * top_k;

    // ---- Allocate result tensors ----

    auto check_alloc = [](cudaError_t err, const char* name) -> bool {
        if (err != cudaSuccess) {
            IMP_LOG_ERROR("moe_topk_gating: cudaMalloc failed for %s: %s", name, cudaGetErrorString(err));
            return false;
        }
        return true;
    };

    // expert_indices: [n_tokens, top_k] int32
    int32_t* d_expert_indices = nullptr;
    if (!check_alloc(cudaMalloc(&d_expert_indices, static_cast<size_t>(total_assignments) * sizeof(int32_t)),
                     "expert_indices"))
        return;

    // expert_weights: [n_tokens, top_k] float
    float* d_expert_weights = nullptr;
    if (!check_alloc(cudaMalloc(&d_expert_weights, static_cast<size_t>(total_assignments) * sizeof(float)),
                     "expert_weights")) {
        IMP_CUDA_CHECK_LOG(cudaFree(d_expert_indices));
        return;
    }

    // sorted_token_ids: [total_assignments] int32
    // We allocate 2x to hold a parallel sorted_flat_idx array right after.
    int32_t* d_sorted_token_ids = nullptr;
    if (!check_alloc(cudaMalloc(&d_sorted_token_ids,
                                static_cast<size_t>(total_assignments) * 2 * sizeof(int32_t)),
                     "sorted_token_ids")) {
        IMP_CUDA_CHECK_LOG(cudaFree(d_expert_indices));
        IMP_CUDA_CHECK_LOG(cudaFree(d_expert_weights));
        return;
    }
    int32_t* d_sorted_flat_idx = d_sorted_token_ids + total_assignments;

    // expert_offsets: [n_experts + 1] int32, then the permute's tile histograms (freed with it).
    int32_t* d_expert_offsets = nullptr;
    const size_t offsets_ints = static_cast<size_t>(n_experts) + 1 + moe_permute_scratch_ints(n_experts);
    if (!check_alloc(cudaMalloc(&d_expert_offsets, offsets_ints * sizeof(int32_t)), "expert_offsets")) {
        IMP_CUDA_CHECK_LOG(cudaFree(d_expert_indices));
        IMP_CUDA_CHECK_LOG(cudaFree(d_expert_weights));
        IMP_CUDA_CHECK_LOG(cudaFree(d_sorted_token_ids));
        return;
    }

    // ---- Kernel 1: Softmax + top-k selection per token ----
    int probs_arrays = score_bias ? 2 : 1;
    size_t smem_gating = static_cast<size_t>(n_experts) * probs_arrays * sizeof(float) +
                         static_cast<size_t>(top_k) * sizeof(float) +
                         static_cast<size_t>(top_k) * sizeof(int32_t);

    topk_gating_kernel<<<n_tokens, BLOCK_SIZE, smem_gating, stream>>>(d_logits, n_experts, top_k,
                                                                      d_expert_indices, d_expert_weights,
                                                                      use_sigmoid, normalize_weights,
                                                                      static_cast<const half*>(score_bias));
    IMP_CUDA_CHECK_LAUNCH();

    // ---- Stable count + scan + scatter ----
    int32_t* d_permute_scratch = moe_permute_scratch_ints(n_experts) ? d_expert_offsets + n_experts + 1
                                                                     : nullptr;
    moe_permute(d_expert_indices, n_tokens, top_k, n_experts, d_sorted_token_ids, d_sorted_flat_idx,
                d_expert_offsets, nullptr, d_permute_scratch, stream);

    // ---- Fill result struct ----
    result.expert_indices = make_tensor_2d(d_expert_indices, QType::INT32, n_tokens, top_k, true);
    result.expert_weights = make_tensor_2d(d_expert_weights, QType::F32, n_tokens, top_k, true);
    // sorted_token_ids: we expose the full allocation (includes flat_idx)
    // but the tensor shape only covers the token IDs part.
    result.sorted_token_ids = make_tensor_1d(d_sorted_token_ids, QType::INT32, total_assignments, true);
    result.expert_offsets = make_tensor_1d(d_expert_offsets, QType::INT32, n_experts + 1, true);
}

// ============================================================================
// MoeRoutingBuffers -- pre-allocated pool
// ============================================================================

MoeRoutingBuffers::~MoeRoutingBuffers() { free(); }

void MoeRoutingBuffers::allocate(int max_tok, int max_exp, int top_k_val) {
    free();
    max_tokens = max_tok;
    max_experts = max_exp;
    top_k = top_k_val;

    int total_assignments = max_tokens * top_k;
    auto align256 = [](size_t x) -> size_t { return (x + 255) & ~size_t(255); };

    size_t indices_sz = align256(static_cast<size_t>(total_assignments) * sizeof(int32_t));
    size_t weights_sz = align256(static_cast<size_t>(total_assignments) * sizeof(float));
    size_t sorted_sz = align256(static_cast<size_t>(total_assignments) * 2 * sizeof(int32_t));
    size_t offsets_sz = align256(static_cast<size_t>(max_experts + 1) * sizeof(int32_t));
    size_t scratch_sz = align256(moe_permute_scratch_ints(max_experts) * sizeof(int32_t));
    size_t t2e_sz = align256(static_cast<size_t>(total_assignments) * sizeof(int32_t));

    pool_size = indices_sz + weights_sz + sorted_sz + offsets_sz + scratch_sz + t2e_sz;
    cudaError_t err = cudaMalloc(&pool, pool_size);
    if (err != cudaSuccess) {
        pool = nullptr;
        pool_size = 0;
        return;
    }

    char* ptr = static_cast<char*>(pool);
    expert_indices = reinterpret_cast<int32_t*>(ptr);
    ptr += indices_sz;
    expert_weights = reinterpret_cast<float*>(ptr);
    ptr += weights_sz;
    sorted_token_ids = reinterpret_cast<int32_t*>(ptr);
    ptr += sorted_sz;
    expert_offsets = reinterpret_cast<int32_t*>(ptr);
    ptr += offsets_sz;
    permute_scratch = scratch_sz ? reinterpret_cast<int32_t*>(ptr) : nullptr;
    ptr += scratch_sz;
    token_to_expanded = reinterpret_cast<int32_t*>(ptr);
    ptr += t2e_sz;
}

void MoeRoutingBuffers::free() {
    if (pool) {
        IMP_CUDA_CHECK_LOG(cudaFree(pool));
        pool = nullptr;
    }
    pool_size = 0;
    expert_indices = nullptr;
    expert_weights = nullptr;
    sorted_token_ids = nullptr;
    expert_offsets = nullptr;
    permute_scratch = nullptr;
    token_to_expanded = nullptr;
}

// ============================================================================
// moe_topk_gating with pre-allocated buffers
// ============================================================================

void moe_topk_gating(const Tensor& gate_logits, int top_k, MoeRoutingBuffers& buffers,
                     MoeRoutingResult& result, cudaStream_t stream, bool use_sigmoid, bool normalize_weights,
                     const void* score_bias, bool skip_sorting) {
    const int n_tokens = static_cast<int>(gate_logits.shape[0]);
    const int n_experts = static_cast<int>(gate_logits.shape[1]);
    const float* d_logits = static_cast<const float*>(gate_logits.data);
    const int total_assignments = n_tokens * top_k;

    int32_t* d_expert_indices = buffers.expert_indices;
    float* d_expert_weights = buffers.expert_weights;
    int32_t* d_sorted_token_ids = buffers.sorted_token_ids;
    int32_t* d_expert_offsets = buffers.expert_offsets;

    // Kernel 1: Softmax + top-k selection per token
    int probs_arrays = score_bias ? 2 : 1;
    size_t smem_gating = static_cast<size_t>(n_experts) * probs_arrays * sizeof(float) +
                         static_cast<size_t>(top_k) * sizeof(float) +
                         static_cast<size_t>(top_k) * sizeof(int32_t);

    topk_gating_kernel<<<n_tokens, BLOCK_SIZE, smem_gating, stream>>>(d_logits, n_experts, top_k,
                                                                      d_expert_indices, d_expert_weights,
                                                                      use_sigmoid, normalize_weights,
                                                                      static_cast<const half*>(score_bias));
    IMP_CUDA_CHECK_LAUNCH();

    if (!skip_sorting) {
        int32_t* d_sorted_flat_idx = d_sorted_token_ids + total_assignments;
        // Scratch is sized for buffers.max_experts; a wider router takes the single-CTA kernel.
        int32_t* scratch = n_experts <= buffers.max_experts ? buffers.permute_scratch : nullptr;
        moe_permute(d_expert_indices, n_tokens, top_k, n_experts, d_sorted_token_ids, d_sorted_flat_idx,
                    d_expert_offsets, buffers.token_to_expanded, scratch, stream);
    }

    // Fill result struct (no ownership -- memory belongs to buffers)
    result.owns_memory = false;
    result.token_to_expanded = buffers.token_to_expanded;
    result.expert_indices = make_tensor_2d(d_expert_indices, QType::INT32, n_tokens, top_k, true);
    result.expert_weights = make_tensor_2d(d_expert_weights, QType::F32, n_tokens, top_k, true);
    result.sorted_token_ids = make_tensor_1d(d_sorted_token_ids, QType::INT32, total_assignments, true);
    result.expert_offsets = make_tensor_1d(d_expert_offsets, QType::INT32, n_experts + 1, true);
}

// ============================================================================
// Fused gate GEMV + topk routing launcher
// ============================================================================

void moe_gate_topk_fused(const void* W_gate, const void* x, int n_experts, int d_model, int top_k,
                         MoeRoutingBuffers& buffers, MoeRoutingResult& result, cudaStream_t stream,
                         bool use_sigmoid, bool normalize_weights, const void* score_bias) {
    // Shared memory: logits + optional sel_probs + topk_val + topk_idx
    //                + warp scratch (float[8] + int[8])
    constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
    int probs_arrays = score_bias ? 2 : 1;
    size_t smem = static_cast<size_t>(n_experts) * probs_arrays * sizeof(float) +
                  static_cast<size_t>(top_k) * sizeof(float) + static_cast<size_t>(top_k) * sizeof(int32_t) +
                  NUM_WARPS * sizeof(float) + NUM_WARPS * sizeof(int);

    gemv_gate_topk_fused_kernel<<<1, BLOCK_SIZE, smem, stream>>>(static_cast<const half*>(W_gate),
                                                                 static_cast<const half*>(x), n_experts,
                                                                 d_model, top_k, buffers.expert_indices,
                                                                 buffers.expert_weights, use_sigmoid,
                                                                 normalize_weights,
                                                                 static_cast<const half*>(score_bias));
    IMP_CUDA_CHECK_LAUNCH();

    // Fill result struct (no ownership — memory belongs to buffers)
    result.owns_memory = false;
    result.expert_indices = make_tensor_2d(buffers.expert_indices, QType::INT32, 1, top_k, true);
    result.expert_weights = make_tensor_2d(buffers.expert_weights, QType::F32, 1, top_k, true);
    result.sorted_token_ids = make_tensor_1d(buffers.sorted_token_ids, QType::INT32, top_k, true);
    result.expert_offsets = make_tensor_1d(buffers.expert_offsets, QType::INT32, n_experts + 1, true);
}

}  // namespace imp
