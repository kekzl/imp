#include "compute/moe_routing.h"
#include "compute/moe_routing_internal.cuh"
#include "core/logging.h"
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace imp {

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

static __global__ void __launch_bounds__(256) moe_fused_permute_deterministic_kernel(
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

}  // namespace imp
