// Token-tiled FP8 split-K decode attention (head_dim=128, block_size % 16 == 0).
//
// The per-token pipeline kernel (attention_paged_fp8.cu) serializes on a
// global->smem round-trip every token: V[t] and K[t+1] are committed in one
// cp.async group, so the PV accumulate waits on the K[t+1] fetch each
// iteration. At 16k ctx that chain leaves the kernel latency-bound at ~10%
// DRAM while attention is ~70% of decode wall-clock (PERF_LOG 06-18).
//
// This variant loads one whole KV page (16 tokens x 128 B K + V) per warp
// with bulk cp.async (double-buffered across the warp's page iterations) and
// computes 16 QK dots in parallel from smem — lane j handles token j&15,
// dim-half j>>4 — followed by one tile-wise online-softmax update. The
// per-token serial chain (smem wait -> 5-shuffle reduce -> exp) collapses to
// one max/exp step per 16 tokens, and the only exposed global latency is the
// first page per warp.
//
// Only head_dim==128 && block_size % 16 == 0 dispatches here (the FP8-KV
// flagship shapes); everything else stays on the pipeline kernel. Warps
// iterate 16-token chunks (a page is 1+ chunks; chunks never straddle pages).

#include "compute/attention_paged.h"
#include "compute/attention_paged_common.cuh"
#include "core/logging.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <float.h>

namespace imp {

namespace {

constexpr int kTileHeadDim = 128;
constexpr int kTileTokens = 16;  // tokens per smem tile (block_size % 16 == 0 dispatch-guarded)
// Row stride 144 B keeps 16 B alignment for cp.async while breaking the
// 128 B == all-32-banks aliasing (4-way LDS conflicts remain in the dot
// phase; measured secondary to the removed per-token latency chain).
constexpr int kTileRowStride = kTileHeadDim + 16;
constexpr int kTileBytes = kTileTokens * kTileRowStride;              // 2304 B
constexpr int kWarpSmemBytes = 4 * kTileBytes;                        // K/V x 2 buffers
constexpr int kBlockSmemBytes = NUM_WARPS * kWarpSmemBytes;           // 73728 B
constexpr int kGqaStages = 2;  // GQA prefetch ring depth (measured: deeper does not help)

// HW pair conversion: 2 packed E4M3 bytes -> f16x2 (exact; e4m3 c f16).
__device__ __forceinline__ half2 fp8x2_to_half2(uint32_t two_bytes) {
    uint32_t r;
    asm("{ .reg .b16 lo; cvt.u16.u32 lo, %1; cvt.rn.f16x2.e4m3x2 %0, lo; }" : "=r"(r) : "r"(two_bytes));
    half2 h;
    memcpy(&h, &r, 4);
    return h;
}

// Issue the bulk cp.async of one 16-token KV chunk (K + V) into a warp's tile
// buffers. 16 rows x 128 B per tile; each lane copies four 16 B pieces per
// tile. All 16 slots are always loaded (chunks never straddle a page and the
// physical page is fully allocated); invalid tokens are masked in compute.
// k_src/v_src point at the chunk's first token slot for this kv head.
__device__ __forceinline__ void tile_load(uint8_t* k_tile, uint8_t* v_tile, const uint8_t* k_src,
                                          const uint8_t* v_src, int kv_slot_stride, int lane) {
#pragma unroll
    for (int i = 0; i < 4; i++) {
        int piece = lane + 32 * i;  // 128 pieces: row = piece/8, 16B col = piece%8
        int row = piece >> 3;
        int col = (piece & 7) * 16;
        cp_async_ca_16(k_tile + row * kTileRowStride + col, k_src + row * kv_slot_stride + col);
        cp_async_ca_16(v_tile + row * kTileRowStride + col, v_src + row * kv_slot_stride + col);
    }
}

// Block-cooperative variant of tile_load for the GQA kernel: all blockDim.x
// threads split the 256 16 B pieces (128 K + 128 V) of one chunk.
__device__ __forceinline__ void tile_load_block(uint8_t* k_tile, uint8_t* v_tile, const uint8_t* k_src,
                                                const uint8_t* v_src, int kv_slot_stride, int tid,
                                                int nthreads) {
    for (int piece = tid; piece < 256; piece += nthreads) {
        int idx = piece & 127;
        int row = idx >> 3;
        int col = (idx & 7) * 16;
        if (piece < 128)
            cp_async_ca_16(k_tile + row * kTileRowStride + col, k_src + row * kv_slot_stride + col);
        else
            cp_async_ca_16(v_tile + row * kTileRowStride + col, v_src + row * kv_slot_stride + col);
    }
}

}  // namespace

template <int HEAD_DIM>
__global__ void paged_attention_splitk_fp8_tile_kernel(
    const half* __restrict__ Q, const uint8_t* __restrict__ K_cache, const uint8_t* __restrict__ V_cache,
    float* __restrict__ partial_out, const int* __restrict__ block_tables,
    const int* __restrict__ context_lens, int batch_size, int n_heads, int n_kv_heads, int block_size,
    float scale, float kv_scale, int max_num_blocks, int num_splits, int sliding_window, float softcap) {
    static_assert(HEAD_DIM == kTileHeadDim, "tile kernel is head_dim=128 only");
    constexpr int ELEMS = HEAD_DIM / WARP_SIZE;

    const int batch_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int split_idx = blockIdx.z;
    const int kv_head = head_idx / (n_heads / n_kv_heads);

    const int ctx_len = context_lens[batch_idx];
    if (ctx_len <= 0)
        return;

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int lane_offset = lane_id * ELEMS;

    int effective_start = 0;
    if (sliding_window > 0 && ctx_len > sliding_window)
        effective_start = ctx_len - sliding_window;
    const int first_block = effective_start / block_size;
    const int num_ctx_blocks = (ctx_len + block_size - 1) / block_size;
    const int total_blocks = num_ctx_blocks - first_block;

    int blocks_per_split = (total_blocks + num_splits - 1) / num_splits;
    int split_start = first_block + split_idx * blocks_per_split;
    int split_end = split_start + blocks_per_split;
    if (split_end > num_ctx_blocks)
        split_end = num_ctx_blocks;

    if (split_start >= split_end) {
        write_empty_split_sentinel<HEAD_DIM>(partial_out, batch_idx, n_heads, head_idx, num_splits, split_idx,
                                             lane_offset);
        return;
    }

    const int* bt = block_tables + (int64_t)batch_idx * max_num_blocks;
    const int kv_block_stride = block_size * n_kv_heads * HEAD_DIM;
    const int kv_slot_stride = n_kv_heads * HEAD_DIM;
    const int kv_head_off = kv_head * HEAD_DIM;
    const float fused_scale = scale * kv_scale;

    extern __shared__ char smem_tile_fp8[];
    uint8_t* my_smem = reinterpret_cast<uint8_t*>(smem_tile_fp8) + warp_id * kWarpSmemBytes;
    uint8_t* k_tiles[2] = {my_smem, my_smem + 2 * kTileBytes};
    uint8_t* v_tiles[2] = {my_smem + kTileBytes, my_smem + 3 * kTileBytes};

    // Dot-phase mapping: lane = (dim_half << 4) | token.
    const int my_tok = lane_id & 15;
    const int my_half = lane_id >> 4;

    // Q slice for this lane's dim half: 64 dims as 32 half2 (exact convert per FMA).
    half2 q_reg[HEAD_DIM / 4];
    {
        const uint4* q4 = reinterpret_cast<const uint4*>(Q + (int64_t)batch_idx * n_heads * HEAD_DIM +
                                                         (int64_t)head_idx * HEAD_DIM + my_half * (HEAD_DIM / 2));
#pragma unroll
        for (int i = 0; i < HEAD_DIM / 16; i++) {
            uint4 v = q4[i];
            memcpy(&q_reg[i * 4], &v, 16);
        }
    }

    float m_w = -FLT_MAX;
    float l_w = 0.0f;
    float o_reg[ELEMS];
#pragma unroll
    for (int i = 0; i < ELEMS; i++)
        o_reg[i] = 0.0f;

    // Warps iterate 16-token chunks; a page holds block_size/16 of them.
    const int chunks_per_page = block_size / kTileTokens;
    const int chunk_begin = split_start * chunks_per_page;
    const int chunk_end = split_end * chunks_per_page;

    auto chunk_src = [&](int ch, const uint8_t* __restrict__ cache) {
        const int page = ch / chunks_per_page;
        const int slot0 = (ch - page * chunks_per_page) * kTileTokens;
        return cache + (int64_t)bt[page] * kv_block_stride + slot0 * kv_slot_stride + kv_head_off;
    };

    int chunk = chunk_begin + warp_id;
    int cur = 0;
    if (chunk < chunk_end) {
        tile_load(k_tiles[0], v_tiles[0], chunk_src(chunk, K_cache), chunk_src(chunk, V_cache),
                  kv_slot_stride, lane_id);
        cp_async_commit();
    }

    for (; chunk < chunk_end; chunk += NUM_WARPS) {
        const int next_chunk = chunk + NUM_WARPS;
        if (next_chunk < chunk_end) {
            tile_load(k_tiles[1 - cur], v_tiles[1 - cur], chunk_src(next_chunk, K_cache),
                      chunk_src(next_chunk, V_cache), kv_slot_stride, lane_id);
            cp_async_commit();
            cp_async_wait_group<1>();
        } else {
            cp_async_wait_group<0>();
        }
        __syncwarp();

        // Valid token range in this chunk.
        int tok_start = chunk * kTileTokens;
        int n_toks = min(ctx_len - tok_start, kTileTokens);
        int first_tok = max(effective_start - tok_start, 0);
        const bool valid = (my_tok >= first_tok) && (my_tok < n_toks);

        // ---- Dot phase: lane (token, dim-half) computes a 64-dim partial ----
        const uint8_t* k_row = k_tiles[cur] + my_tok * kTileRowStride + my_half * (HEAD_DIM / 2);
        float part = 0.0f;
#pragma unroll
        for (int i = 0; i < HEAD_DIM / 8; i++) {
            uint32_t packed = reinterpret_cast<const uint32_t*>(k_row)[i];
            float2 k01 = __half22float2(fp8x2_to_half2(packed));
            float2 k23 = __half22float2(fp8x2_to_half2(packed >> 16));
            float2 q01 = __half22float2(q_reg[i * 2]);
            float2 q23 = __half22float2(q_reg[i * 2 + 1]);
            part = __fmaf_rn(q01.x, k01.x, part);
            part = __fmaf_rn(q01.y, k01.y, part);
            part = __fmaf_rn(q23.x, k23.x, part);
            part = __fmaf_rn(q23.y, k23.y, part);
        }
        float dot = part + __shfl_xor_sync(0xFFFFFFFF, part, 16);
        dot *= fused_scale;
        dot = apply_softcap(dot, softcap);
        if (!valid)
            dot = -FLT_MAX;

        // ---- Tile-wise online softmax (warp-uniform state update) ----
        float m_tile = dot;
#pragma unroll
        for (int off = 8; off > 0; off >>= 1)
            m_tile = fmaxf(m_tile, __shfl_xor_sync(0xFFFFFFFF, m_tile, off));
        const float m_new = fmaxf(m_w, m_tile);
        const float rescale = expf(m_w - m_new);

        float p = valid ? expf(dot - m_new) : 0.0f;
        float l_tile = (my_half == 0) ? p : 0.0f;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            l_tile += __shfl_xor_sync(0xFFFFFFFF, l_tile, off);
        l_w = l_w * rescale + l_tile;
        m_w = m_new;

        // ---- PV phase: lane owns dims [lane*4, lane*4+4) ----
#pragma unroll
        for (int i = 0; i < ELEMS; i++)
            o_reg[i] *= rescale;
        const uint32_t* v_words = reinterpret_cast<const uint32_t*>(v_tiles[cur]);
        constexpr int kRowWords = kTileRowStride / 4;
#pragma unroll
        for (int j = 0; j < kTileTokens; j++) {
            float pj = __shfl_sync(0xFFFFFFFF, p, j) * kv_scale;
            if (pj == 0.0f)
                continue;
            uint32_t packed = v_words[j * kRowWords + lane_id];
            float2 v01 = __half22float2(fp8x2_to_half2(packed));
            float2 v23 = __half22float2(fp8x2_to_half2(packed >> 16));
            o_reg[0] = __fmaf_rn(pj, v01.x, o_reg[0]);
            o_reg[1] = __fmaf_rn(pj, v01.y, o_reg[1]);
            o_reg[2] = __fmaf_rn(pj, v23.x, o_reg[2]);
            o_reg[3] = __fmaf_rn(pj, v23.y, o_reg[3]);
        }

        // A lane racing into the next iteration would cp.async into the buffer
        // other lanes are still reading (independent thread scheduling).
        __syncwarp();
        cur = 1 - cur;
    }

    // o_reg is exp-weight-summed (unnormalized); the shared cross-warp reduce
    // expects the running-normalized form (o / l).
    if (l_w > 0.0f) {
        const float inv_l = 1.0f / l_w;
#pragma unroll
        for (int i = 0; i < ELEMS; i++)
            o_reg[i] *= inv_l;
    }

    __syncthreads();
    crosswarp_reduce_splitk<HEAD_DIM>(reinterpret_cast<float*>(smem_tile_fp8), m_w, l_w, o_reg, warp_id,
                                      lane_id, lane_offset, partial_out, batch_idx, n_heads, head_idx,
                                      num_splits, split_idx);
}

// ---------------------------------------------------------------------------
// GQA-batched variant: grid.y = n_kv_heads; one block computes ALL
// G = n_heads/n_kv_heads Q heads of its KV head (warp w = Q head w), sharing
// each KV tile in block-local smem. The per-head tile kernel reads every KV
// byte G times through L2 (one block per Q head); this variant reads it once,
// cutting L2 traffic /G on the L2-bound long-ctx path. Smem drops from 73 KB
// (8 private double-buffered warp pipelines) to 18 KB (one shared
// kGqaStages-deep pipeline), so blocks are multi-resident per SM and latency
// hiding moves from warps-within-block to blocks-per-SM. Chunks advance
// block-serially (2 __syncthreads per 16-token chunk); parallelism is
// recovered via a higher split count chosen by the launcher's geometry-aware
// heuristic.
//
// Each warp holds a complete split result for a distinct head, so there is no
// cross-warp reduce: warps write (m, l, O_unnormalized-at-m) partials directly,
// which is exactly the format paged_attention_reduce_kernel merges.
template <int HEAD_DIM>
__global__ void paged_attention_splitk_fp8_tile_gqa_kernel(
    const half* __restrict__ Q, const uint8_t* __restrict__ K_cache, const uint8_t* __restrict__ V_cache,
    float* __restrict__ partial_out, const int* __restrict__ block_tables,
    const int* __restrict__ context_lens, int batch_size, int n_heads, int n_kv_heads, int block_size,
    float scale, float kv_scale, int max_num_blocks, int num_splits, int sliding_window, float softcap) {
    static_assert(HEAD_DIM == kTileHeadDim, "tile kernel is head_dim=128 only");
    constexpr int ELEMS = HEAD_DIM / WARP_SIZE;

    const int batch_idx = blockIdx.x;
    const int kv_head = blockIdx.y;
    const int split_idx = blockIdx.z;
    const int warp_id = threadIdx.x / WARP_SIZE;  // blockDim.x == 32 * G, warp w = Q head w
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int head_idx = kv_head * (n_heads / n_kv_heads) + warp_id;

    const int ctx_len = context_lens[batch_idx];
    if (ctx_len <= 0)
        return;

    int effective_start = 0;
    if (sliding_window > 0 && ctx_len > sliding_window)
        effective_start = ctx_len - sliding_window;
    const int first_block = effective_start / block_size;
    const int num_ctx_blocks = (ctx_len + block_size - 1) / block_size;
    const int total_blocks = num_ctx_blocks - first_block;

    int blocks_per_split = (total_blocks + num_splits - 1) / num_splits;
    int split_start = first_block + split_idx * blocks_per_split;
    int split_end = split_start + blocks_per_split;
    if (split_end > num_ctx_blocks)
        split_end = num_ctx_blocks;

    constexpr int partial_stride = 2 + HEAD_DIM;
    float* out = partial_out +
                 (int64_t)((batch_idx * n_heads + head_idx) * num_splits + split_idx) * partial_stride;

    if (split_start >= split_end) {
        // Per-warp sentinel (each warp owns a distinct head's partial).
        if (lane_id == 0) {
            out[0] = -FLT_MAX;
            out[1] = 0.0f;
        }
#pragma unroll
        for (int i = 0; i < ELEMS; i++)
            out[2 + lane_id * ELEMS + i] = 0.0f;
        return;
    }

    const int* bt = block_tables + (int64_t)batch_idx * max_num_blocks;
    const int kv_block_stride = block_size * n_kv_heads * HEAD_DIM;
    const int kv_slot_stride = n_kv_heads * HEAD_DIM;
    const int kv_head_off = kv_head * HEAD_DIM;
    const float fused_scale = scale * kv_scale;

    __shared__ uint8_t smem_gqa[kGqaStages * 2 * kTileBytes];  // per stage: (K tile, V tile)
    uint8_t* k_tiles[kGqaStages];
    uint8_t* v_tiles[kGqaStages];
#pragma unroll
    for (int i = 0; i < kGqaStages; i++) {
        k_tiles[i] = smem_gqa + i * 2 * kTileBytes;
        v_tiles[i] = smem_gqa + i * 2 * kTileBytes + kTileBytes;
    }

    // Dot-phase mapping: lane = (dim_half << 4) | token.
    const int my_tok = lane_id & 15;
    const int my_half = lane_id >> 4;

    // Q slice for this warp's head and this lane's dim half (see tile kernel).
    half2 q_reg[HEAD_DIM / 4];
    {
        const uint4* q4 = reinterpret_cast<const uint4*>(Q + (int64_t)batch_idx * n_heads * HEAD_DIM +
                                                         (int64_t)head_idx * HEAD_DIM + my_half * (HEAD_DIM / 2));
#pragma unroll
        for (int i = 0; i < HEAD_DIM / 16; i++) {
            uint4 v = q4[i];
            memcpy(&q_reg[i * 4], &v, 16);
        }
    }

    float m_w = -FLT_MAX;
    float l_w = 0.0f;
    float o_reg[ELEMS];
#pragma unroll
    for (int i = 0; i < ELEMS; i++)
        o_reg[i] = 0.0f;

    const int chunks_per_page = block_size / kTileTokens;
    const int chunk_begin = split_start * chunks_per_page;
    const int chunk_end = split_end * chunks_per_page;

    auto chunk_src = [&](int ch, const uint8_t* __restrict__ cache) {
        const int page = ch / chunks_per_page;
        const int slot0 = (ch - page * chunks_per_page) * kTileTokens;
        return cache + (int64_t)bt[page] * kv_block_stride + slot0 * kv_slot_stride + kv_head_off;
    };

    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    // kStages-deep prefetch ring: the block-serial chunk walk exposes one
    // global->smem latency per chunk with a 2-deep buffer (the per-head tile
    // kernel hides it behind 8 independent warp pipelines; here all warps step
    // the same chunk). Every iteration commits exactly ONE cp.async group
    // (empty groups past the tail keep the count aligned), so
    // wait_group<kStages-1> always completes the current chunk's load.
    for (int i = 0; i < kGqaStages - 1; i++) {
        if (chunk_begin + i < chunk_end)
            tile_load_block(k_tiles[i], v_tiles[i], chunk_src(chunk_begin + i, K_cache),
                            chunk_src(chunk_begin + i, V_cache), kv_slot_stride, tid, nthreads);
        cp_async_commit();
    }

    int cur = 0;
    for (int chunk = chunk_begin; chunk < chunk_end; chunk++) {
        const int pf = chunk + kGqaStages - 1;
        if (pf < chunk_end) {
            const int slot = (cur + kGqaStages - 1) % kGqaStages;
            tile_load_block(k_tiles[slot], v_tiles[slot], chunk_src(pf, K_cache), chunk_src(pf, V_cache),
                            kv_slot_stride, tid, nthreads);
        }
        cp_async_commit();
        cp_async_wait_group<kGqaStages - 1>();
        __syncthreads();

        // Valid token range in this chunk.
        int tok_start = chunk * kTileTokens;
        int n_toks = min(ctx_len - tok_start, kTileTokens);
        int first_tok = max(effective_start - tok_start, 0);
        const bool valid = (my_tok >= first_tok) && (my_tok < n_toks);

        // ---- Dot phase (identical to the per-head tile kernel) ----
        const uint8_t* k_row = k_tiles[cur] + my_tok * kTileRowStride + my_half * (HEAD_DIM / 2);
        float part = 0.0f;
#pragma unroll
        for (int i = 0; i < HEAD_DIM / 8; i++) {
            uint32_t packed = reinterpret_cast<const uint32_t*>(k_row)[i];
            float2 k01 = __half22float2(fp8x2_to_half2(packed));
            float2 k23 = __half22float2(fp8x2_to_half2(packed >> 16));
            float2 q01 = __half22float2(q_reg[i * 2]);
            float2 q23 = __half22float2(q_reg[i * 2 + 1]);
            part = __fmaf_rn(q01.x, k01.x, part);
            part = __fmaf_rn(q01.y, k01.y, part);
            part = __fmaf_rn(q23.x, k23.x, part);
            part = __fmaf_rn(q23.y, k23.y, part);
        }
        float dot = part + __shfl_xor_sync(0xFFFFFFFF, part, 16);
        dot *= fused_scale;
        dot = apply_softcap(dot, softcap);
        if (!valid)
            dot = -FLT_MAX;

        // ---- Tile-wise online softmax (warp-uniform state update) ----
        float m_tile = dot;
#pragma unroll
        for (int off = 8; off > 0; off >>= 1)
            m_tile = fmaxf(m_tile, __shfl_xor_sync(0xFFFFFFFF, m_tile, off));
        const float m_new = fmaxf(m_w, m_tile);
        const float rescale = expf(m_w - m_new);

        float p = valid ? expf(dot - m_new) : 0.0f;
        float l_tile = (my_half == 0) ? p : 0.0f;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            l_tile += __shfl_xor_sync(0xFFFFFFFF, l_tile, off);
        l_w = l_w * rescale + l_tile;
        m_w = m_new;

        // ---- PV phase: lane owns dims [lane*4, lane*4+4) ----
#pragma unroll
        for (int i = 0; i < ELEMS; i++)
            o_reg[i] *= rescale;
        const uint32_t* v_words = reinterpret_cast<const uint32_t*>(v_tiles[cur]);
        constexpr int kRowWords = kTileRowStride / 4;
#pragma unroll
        for (int j = 0; j < kTileTokens; j++) {
            float pj = __shfl_sync(0xFFFFFFFF, p, j) * kv_scale;
            if (pj == 0.0f)
                continue;
            uint32_t packed = v_words[j * kRowWords + lane_id];
            float2 v01 = __half22float2(fp8x2_to_half2(packed));
            float2 v23 = __half22float2(fp8x2_to_half2(packed >> 16));
            o_reg[0] = __fmaf_rn(pj, v01.x, o_reg[0]);
            o_reg[1] = __fmaf_rn(pj, v01.y, o_reg[1]);
            o_reg[2] = __fmaf_rn(pj, v23.x, o_reg[2]);
            o_reg[3] = __fmaf_rn(pj, v23.y, o_reg[3]);
        }

        // All warps must be done reading this buffer before a later iteration
        // prefetches into it.
        __syncthreads();
        cur = (cur + 1) % kGqaStages;
    }

    // Direct per-warp partial write: (m, l, O unnormalized at m) — o_reg is
    // already the exp-weighted sum, which is the partial format the split-K
    // reduce kernel expects (no /l + *l round-trip, no cross-warp merge).
    if (lane_id == 0) {
        out[0] = m_w;
        out[1] = l_w;
    }
#pragma unroll
    for (int i = 0; i < ELEMS; i++)
        out[2 + lane_id * ELEMS + i] = o_reg[i];
}

bool paged_attention_splitk_fp8_tile_supported(int head_dim, int block_size) {
    return head_dim == kTileHeadDim && block_size >= kTileTokens && block_size % kTileTokens == 0;
}

void paged_attention_splitk_fp8_tile_launch(const half* Q, const uint8_t* K_cache, const uint8_t* V_cache,
                                            float* partial_out, const int* block_tables,
                                            const int* context_lens, int batch_size, int n_heads,
                                            int n_kv_heads, int block_size, float scale, float kv_scale,
                                            int max_num_blocks, int num_splits, int sliding_window,
                                            float softcap, cudaStream_t stream) {
    static bool smem_opted_in = [] {
        cudaError_t err = cudaFuncSetAttribute(paged_attention_splitk_fp8_tile_kernel<kTileHeadDim>,
                                               cudaFuncAttributeMaxDynamicSharedMemorySize, kBlockSmemBytes);
        if (err != cudaSuccess)
            IMP_LOG_ERROR("fp8 tile attention smem opt-in failed: %s", cudaGetErrorString(err));
        return err == cudaSuccess;
    }();
    (void)smem_opted_in;

    dim3 grid(batch_size, n_heads, num_splits);
    dim3 block(BLOCK_THREADS);
    paged_attention_splitk_fp8_tile_kernel<kTileHeadDim><<<grid, block, kBlockSmemBytes, stream>>>(
        Q, K_cache, V_cache, partial_out, block_tables, context_lens, batch_size, n_heads, n_kv_heads,
        block_size, scale, kv_scale, max_num_blocks, num_splits, sliding_window, softcap);
    IMP_CUDA_CHECK_LAUNCH();
}

// Split count for the GQA variant's geometry: with grid.y = n_kv_heads (G x
// fewer head-blocks than the per-head kernel) and multi-resident blocks,
// target 2*SMs concurrent blocks, capped by the page count and the split-K
// scratch size. Measured @16k (Coder-30B, batch*n_kv_heads = 4): e2e decode is
// flat for s in [85..128] and drops below (kernel underfill) as well as above
// (the reduce kernel's partial traffic scales with s) — 2*SMs sits at the
// plateau's cheap end.
int paged_attention_splitk_fp8_tile_gqa_splits(int batch_size, int n_heads, int n_kv_heads, int head_dim,
                                               int block_size, int max_context_len) {
    const int sms = kpar_n_sms();
    const int num_ctx_blocks = (max_context_len + block_size - 1) / block_size;
    const int bh_kv = batch_size * n_kv_heads;
    int s = (2 * sms + bh_kv - 1) / bh_kv;
    s = min(s, num_ctx_blocks);
    void* sk_ptr = nullptr;
    size_t sk_size = 0;
    paged_attention_get_splitk_scratch(&sk_ptr, &sk_size);
    const size_t per_split = (size_t)batch_size * n_heads * (2 + head_dim) * sizeof(float);
    s = min(s, (int)(sk_size / per_split));
    return max(s, 1);
}

bool paged_attention_splitk_fp8_tile_gqa_supported(int head_dim, int block_size, int n_heads,
                                                   int n_kv_heads) {
    if (!paged_attention_splitk_fp8_tile_supported(head_dim, block_size))
        return false;
    if (n_kv_heads <= 0 || n_heads % n_kv_heads != 0)
        return false;
    const int g = n_heads / n_kv_heads;
    return g >= 2 && g <= NUM_WARPS;  // warp w = Q head w; one warp group per KV head
}

void paged_attention_splitk_fp8_tile_gqa_launch(const half* Q, const uint8_t* K_cache,
                                                const uint8_t* V_cache, float* partial_out,
                                                const int* block_tables, const int* context_lens,
                                                int batch_size, int n_heads, int n_kv_heads, int block_size,
                                                float scale, float kv_scale, int max_num_blocks,
                                                int num_splits, int sliding_window, float softcap,
                                                cudaStream_t stream) {
    const int g = n_heads / n_kv_heads;
    dim3 grid(batch_size, n_kv_heads, num_splits);
    dim3 block(WARP_SIZE * g);  // static smem (9 KB) — no opt-in needed
    paged_attention_splitk_fp8_tile_gqa_kernel<kTileHeadDim><<<grid, block, 0, stream>>>(
        Q, K_cache, V_cache, partial_out, block_tables, context_lens, batch_size, n_heads, n_kv_heads,
        block_size, scale, kv_scale, max_num_blocks, num_splits, sliding_window, softcap);
    IMP_CUDA_CHECK_LAUNCH();
}

}  // namespace imp
