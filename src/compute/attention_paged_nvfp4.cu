#include "compute/attention_paged.h"
#include "compute/attention_paged_common.cuh"
#include "compute/attention.h"
#include "quant/turboquant_fp4.cuh"
#include "core/logging.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <float.h>

namespace imp {

// ---------------------------------------------------------------------------
// NVFP4 Paged Attention Decode
//
// KV cache stores 2 FP4 (E2M1) values per byte (low nibble = even, high = odd).
// Per-token-head-group_of_16 UE4M3 (FP8 E4M3) scales stored separately.
//
// Layout per cache block:
//   K_cache : [block, t_in_block, kv_head, head_dim/2]    uint8_t  (packed FP4)
//   V_cache : same shape as K_cache
//   K_scales: [block, t_in_block, kv_head, head_dim/16]   uint8_t  (UE4M3)
//   V_scales: same shape as K_scales
//
// Dequant: val = e2m1_decode(nibble) * ue4m3_decode(scale_byte)
//
// Each lane covers ELEMS = HEAD_DIM/32 contiguous elems. For all imp head_dims
// (64/128/256/512), ELEMS ≤ 16 so a lane covers a slice of exactly ONE
// 16-element scale group. lane scale index = lane_offset / 16.
// ---------------------------------------------------------------------------

// UE4M3 byte → float (standard FP8 E4M3, sign always 0 in NVFP4 scale role).
__device__ __forceinline__ float ue4m3_decode(uint8_t bits) {
    __nv_fp8_e4m3 v;
    memcpy(&v, &bits, 1);
    return static_cast<float>(v);
}

// Scale dtype tag for the NVFP4 paged attention kernels. Default E4M3 is the
// existing NVFP4 path; UE8M0 reserves the second template arm for the
// upcoming MXFP4-KV variant. This Slice 1 commit only adds the template parameter; the
// UE8M0 branch is implemented in Slice 2.
enum class ScaleDtype : int { E4M3 = 0, UE8M0 = 1 };

// Decode one packed scale byte into a float, dispatching on SCALE_DTYPE at
// compile time. E4M3 path uses the existing ue4m3_decode helper. UE8M0 path
// uses tq_fp4_ue8m0_to_float from src/quant/turboquant_fp4.cuh.
template <ScaleDtype S>
__device__ __forceinline__ float decode_kv_scale(uint8_t bits);

template <>
__device__ __forceinline__ float decode_kv_scale<ScaleDtype::E4M3>(uint8_t bits) {
    return ue4m3_decode(bits);
}

template <>
__device__ __forceinline__ float decode_kv_scale<ScaleDtype::UE8M0>(uint8_t bits) {
    return tq_fp4_ue8m0_to_float(bits);
}

// Decode one packed FP4 byte (low nibble = .x, high nibble = .y) → half2.
// Single PTX `cvt.rn.f16x2.e2m1x2` (sm_120+, CUDA 13.2+). Replaces the prior
// per-nibble 8-entry magnitude LUT + sign branch (~12 ops/byte → 1).
__device__ __forceinline__ half2 fp4_byte_to_half2(uint32_t byte_val) {
    uint32_t fp16x2;
    asm("{ .reg .b8 t; cvt.u8.u32 t, %1; cvt.rn.f16x2.e2m1x2 %0, t; }"
        : "=r"(fp16x2) : "r"(byte_val));
    return *reinterpret_cast<half2*>(&fp16x2);
}

// Load a lane's PACK packed-FP4 bytes in as few global loads as the alignment
// allows. Every stride in the KV layout is a multiple of HEAD_DIM/2 and the
// lane's byte offset is lane_id * PACK, so the pointer carries PACK-byte
// alignment: HD=256 loads 4 bytes as one word instead of four LDG.E.U8.
// __ldg, not __ldcs: the GQA-tile refutation (#1785) established that the
// cross-head re-reads of this cache are L2 hits worth keeping.
template <int PACK>
__device__ __forceinline__ void load_packed_fp4(const uint8_t* __restrict__ src, uint8_t (&out)[PACK]) {
    if constexpr (PACK % 4 == 0) {
        const uint32_t* p = reinterpret_cast<const uint32_t*>(src);
#pragma unroll
        for (int i = 0; i < PACK / 4; i++) {
            uint32_t w = __ldg(&p[i]);
#pragma unroll
            for (int b = 0; b < 4; b++)
                out[4 * i + b] = static_cast<uint8_t>((w >> (8 * b)) & 0xFF);
        }
    } else if constexpr (PACK % 2 == 0) {
        const ushort* p = reinterpret_cast<const ushort*>(src);
#pragma unroll
        for (int i = 0; i < PACK / 2; i++) {
            ushort w = __ldg(&p[i]);
            out[2 * i] = static_cast<uint8_t>(w & 0xFF);
            out[2 * i + 1] = static_cast<uint8_t>(w >> 8);
        }
    } else {
#pragma unroll
        for (int i = 0; i < PACK; i++)
            out[i] = __ldg(&src[i]);
    }
}

// ---------------------------------------------------------------------------
// Non-Split-K NVFP4 decode kernel
// ---------------------------------------------------------------------------

template <int HEAD_DIM, ScaleDtype SCALE_DTYPE = ScaleDtype::E4M3>
__global__ void paged_attention_decode_nvfp4_kernel(
    const half* __restrict__ Q,
    const uint8_t* __restrict__ K_cache,   // packed FP4 pairs
    const uint8_t* __restrict__ V_cache,   // packed FP4 pairs
    const uint8_t* __restrict__ K_scales,  // UE4M3 per group
    const uint8_t* __restrict__ V_scales,  // UE4M3 per group
    half* __restrict__ O, const int* __restrict__ block_tables, const int* __restrict__ context_lens,
    int batch_size, int n_heads, int n_kv_heads, int block_size, float scale, int max_context_len,
    int max_num_blocks, int sliding_window, float softcap, const half* __restrict__ attn_sinks) {
    static_assert(HEAD_DIM % WARP_SIZE == 0, "HEAD_DIM must be divisible by WARP_SIZE");
    static_assert(HEAD_DIM % 16 == 0, "HEAD_DIM must be divisible by 16 (NVFP4 group size)");
    constexpr int ELEMS = HEAD_DIM / WARP_SIZE;

    const int batch_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int kv_head = head_idx / (n_heads / n_kv_heads);

    const int ctx_len = context_lens[batch_idx];
    if (ctx_len <= 0)
        return;

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int lane_offset = lane_id * ELEMS;
    const int lane_group = lane_offset / 16;  // scale group covering this lane's elems

    // Q into registers
    const half* Q_ptr = Q + (int64_t)batch_idx * n_heads * HEAD_DIM + (int64_t)head_idx * HEAD_DIM;
    float q_reg[ELEMS];
    {
        const half2* Q_ptr2 = reinterpret_cast<const half2*>(Q_ptr + lane_offset);
#pragma unroll
        for (int i = 0; i < ELEMS / 2; i++) {
            half2 h2 = Q_ptr2[i];
            q_reg[2 * i] = __half2float(h2.x);
            q_reg[2 * i + 1] = __half2float(h2.y);
        }
    }

    const int* bt = block_tables + (int64_t)batch_idx * max_num_blocks;
    const int kv_head_bytes = HEAD_DIM / 2;
    const int kv_block_stride = block_size * n_kv_heads * kv_head_bytes;
    const int kv_slot_stride = n_kv_heads * kv_head_bytes;
    const int sc_groups = HEAD_DIM / 16;
    const int sc_block_stride = block_size * n_kv_heads * sc_groups;
    const int sc_slot_stride = n_kv_heads * sc_groups;

    int effective_start = 0;
    if (sliding_window > 0 && ctx_len > sliding_window)
        effective_start = ctx_len - sliding_window;
    const int first_block = effective_start / block_size;
    const int num_ctx_blocks = (ctx_len + block_size - 1) / block_size;

    float m_w = -FLT_MAX;
    float l_w = 0.0f;
    float o_reg[ELEMS];
#pragma unroll
    for (int i = 0; i < ELEMS; i++)
        o_reg[i] = 0.0f;

    for (int blk = first_block + warp_id; blk < num_ctx_blocks; blk += NUM_WARPS) {
        int phys_block = bt[blk];
        // StreamingLLM eviction leaves -1 sentinels in the table; a negative
        // physical block would be an OOB KV read. The FP16 twin has carried
        // this since #963 and the quantised ones did not (#1678): host-side
        // eviction keeps the window range valid, so this is defense-in-depth -
        // future range drift degrades to a skipped block instead of an illegal
        // access or silent garbage.
        if (phys_block < 0)
            continue;
        const uint8_t* K_block = K_cache + (int64_t)phys_block * kv_block_stride;
        const uint8_t* V_block = V_cache + (int64_t)phys_block * kv_block_stride;
        const uint8_t* K_sc_block = K_scales + (int64_t)phys_block * sc_block_stride;
        const uint8_t* V_sc_block = V_scales + (int64_t)phys_block * sc_block_stride;

        int tok_start = blk * block_size;
        int tok_end = tok_start + block_size;
        if (tok_end > ctx_len)
            tok_end = ctx_len;

        int first_tok = 0;
        if (tok_start < effective_start)
            first_tok = effective_start - tok_start;

        for (int t = first_tok; t < (tok_end - tok_start); t++) {
            const uint8_t* K_tok = K_block + t * kv_slot_stride + kv_head * kv_head_bytes;
            const uint8_t* V_tok = V_block + t * kv_slot_stride + kv_head * kv_head_bytes;

            // Per-lane scale (one group covers all ELEMS for this lane)
            float k_scale = decode_kv_scale<SCALE_DTYPE>(
                K_sc_block[t * sc_slot_stride + kv_head * sc_groups + lane_group]);
            float v_scale = decode_kv_scale<SCALE_DTYPE>(
                V_sc_block[t * sc_slot_stride + kv_head * sc_groups + lane_group]);
            const half2 k_scale_h2 = __float2half2_rn(k_scale);
            const half2 v_scale_h2 = __float2half2_rn(v_scale);

            // V is loaded here, not after the softmax: its bytes do not depend
            // on the weight, so issuing both loads up front puts the warp
            // reduction and the two expf of online_softmax_step between issue
            // and use. This is not the 2026-05 smem pipeline (see note below) -
            // no prefetch across tokens, no shared memory, no extra registers
            // beyond one PACK-byte buffer.
            uint8_t k_bytes[ELEMS / 2];
            uint8_t v_bytes[ELEMS / 2];
            load_packed_fp4<ELEMS / 2>(K_tok + lane_offset / 2, k_bytes);
            load_packed_fp4<ELEMS / 2>(V_tok + lane_offset / 2, v_bytes);

            // Q.K dot — HW FP4 decode (cvt.rn.f16x2.e2m1x2) + half2 scale fold
            float dot = 0.0f;
#pragma unroll
            for (int i = 0; i < ELEMS / 2; i++) {
                half2 kh2 = fp4_byte_to_half2(k_bytes[i]);
                kh2 = __hmul2(kh2, k_scale_h2);
                float2 kf = __half22float2(kh2);
                dot = __fmaf_rn(q_reg[2 * i], kf.x, dot);
                dot = __fmaf_rn(q_reg[2 * i + 1], kf.y, dot);
            }
            dot = warp_reduce_sum(dot);
            dot *= scale;
            dot = apply_softcap(dot, softcap);

            float rescale, w_new;
            online_softmax_step(dot, m_w, l_w, rescale, w_new);

#pragma unroll
            for (int i = 0; i < ELEMS / 2; i++) {
                half2 vh2 = fp4_byte_to_half2(v_bytes[i]);
                vh2 = __hmul2(vh2, v_scale_h2);
                float2 vf = __half22float2(vh2);
                o_reg[2 * i] = __fmaf_rn(w_new, vf.x, rescale * o_reg[2 * i]);
                o_reg[2 * i + 1] = __fmaf_rn(w_new, vf.y, rescale * o_reg[2 * i + 1]);
            }
        }
    }

    extern __shared__ char smem_nvfp4[];
    crosswarp_reduce_and_write<HEAD_DIM>(reinterpret_cast<float*>(smem_nvfp4), m_w, l_w, o_reg, warp_id,
                                         lane_id, lane_offset, O, batch_idx, n_heads, head_idx, attn_sinks);
}

// ---------------------------------------------------------------------------
// Split-K NVFP4 decode kernel
// ---------------------------------------------------------------------------

template <int HEAD_DIM, ScaleDtype SCALE_DTYPE = ScaleDtype::E4M3>
__global__ void paged_attention_splitk_nvfp4_kernel(
    const half* __restrict__ Q, const uint8_t* __restrict__ K_cache, const uint8_t* __restrict__ V_cache,
    const uint8_t* __restrict__ K_scales, const uint8_t* __restrict__ V_scales,
    float* __restrict__ partial_out, const int* __restrict__ block_tables,
    const int* __restrict__ context_lens, int batch_size, int n_heads, int n_kv_heads, int block_size,
    float scale, int max_num_blocks, int num_splits, int sliding_window, float softcap) {
    static_assert(HEAD_DIM % WARP_SIZE == 0, "HEAD_DIM must be divisible by WARP_SIZE");
    static_assert(HEAD_DIM % 16 == 0, "HEAD_DIM must be divisible by 16 (NVFP4 group size)");
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
    const int lane_group = lane_offset / 16;

    const half* Q_ptr = Q + (int64_t)batch_idx * n_heads * HEAD_DIM + (int64_t)head_idx * HEAD_DIM;
    float q_reg[ELEMS];
    {
        const half2* Q_ptr2 = reinterpret_cast<const half2*>(Q_ptr + lane_offset);
#pragma unroll
        for (int i = 0; i < ELEMS / 2; i++) {
            half2 h2 = Q_ptr2[i];
            q_reg[2 * i] = __half2float(h2.x);
            q_reg[2 * i + 1] = __half2float(h2.y);
        }
    }

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
    const int kv_head_bytes = HEAD_DIM / 2;
    const int kv_block_stride = block_size * n_kv_heads * kv_head_bytes;
    const int kv_slot_stride = n_kv_heads * kv_head_bytes;
    const int sc_groups = HEAD_DIM / 16;
    const int sc_block_stride = block_size * n_kv_heads * sc_groups;
    const int sc_slot_stride = n_kv_heads * sc_groups;

    float m_w = -FLT_MAX;
    float l_w = 0.0f;
    float o_reg[ELEMS];
#pragma unroll
    for (int i = 0; i < ELEMS; i++)
        o_reg[i] = 0.0f;

    for (int blk = split_start + warp_id; blk < split_end; blk += NUM_WARPS) {
        int phys_block = bt[blk];
        // StreamingLLM eviction leaves -1 sentinels in the table; a negative
        // physical block would be an OOB KV read. The FP16 twin has carried
        // this since #963 and the quantised ones did not (#1678): host-side
        // eviction keeps the window range valid, so this is defense-in-depth -
        // future range drift degrades to a skipped block instead of an illegal
        // access or silent garbage.
        if (phys_block < 0)
            continue;
        const uint8_t* K_block = K_cache + (int64_t)phys_block * kv_block_stride;
        const uint8_t* V_block = V_cache + (int64_t)phys_block * kv_block_stride;
        const uint8_t* K_sc_block = K_scales + (int64_t)phys_block * sc_block_stride;
        const uint8_t* V_sc_block = V_scales + (int64_t)phys_block * sc_block_stride;

        int tok_start = blk * block_size;
        int tok_end = tok_start + block_size;
        if (tok_end > ctx_len)
            tok_end = ctx_len;

        int first_tok = 0;
        if (tok_start < effective_start)
            first_tok = effective_start - tok_start;

        for (int t = first_tok; t < (tok_end - tok_start); t++) {
            const uint8_t* K_tok = K_block + t * kv_slot_stride + kv_head * kv_head_bytes;
            const uint8_t* V_tok = V_block + t * kv_slot_stride + kv_head * kv_head_bytes;

            float k_scale = decode_kv_scale<SCALE_DTYPE>(
                K_sc_block[t * sc_slot_stride + kv_head * sc_groups + lane_group]);
            float v_scale = decode_kv_scale<SCALE_DTYPE>(
                V_sc_block[t * sc_slot_stride + kv_head * sc_groups + lane_group]);
            const half2 k_scale_h2 = __float2half2_rn(k_scale);
            const half2 v_scale_h2 = __float2half2_rn(v_scale);

            // Both loads issued before the reduction - see the twin above.
            uint8_t k_bytes[ELEMS / 2];
            uint8_t v_bytes[ELEMS / 2];
            load_packed_fp4<ELEMS / 2>(K_tok + lane_offset / 2, k_bytes);
            load_packed_fp4<ELEMS / 2>(V_tok + lane_offset / 2, v_bytes);

            float dot = 0.0f;
#pragma unroll
            for (int i = 0; i < ELEMS / 2; i++) {
                half2 kh2 = fp4_byte_to_half2(k_bytes[i]);
                kh2 = __hmul2(kh2, k_scale_h2);
                float2 kf = __half22float2(kh2);
                dot = __fmaf_rn(q_reg[2 * i], kf.x, dot);
                dot = __fmaf_rn(q_reg[2 * i + 1], kf.y, dot);
            }
            dot = warp_reduce_sum(dot);
            dot *= scale;
            dot = apply_softcap(dot, softcap);

            float rescale, w_new;
            online_softmax_step(dot, m_w, l_w, rescale, w_new);

#pragma unroll
            for (int i = 0; i < ELEMS / 2; i++) {
                half2 vh2 = fp4_byte_to_half2(v_bytes[i]);
                vh2 = __hmul2(vh2, v_scale_h2);
                float2 vf = __half22float2(vh2);
                o_reg[2 * i] = __fmaf_rn(w_new, vf.x, rescale * o_reg[2 * i]);
                o_reg[2 * i + 1] = __fmaf_rn(w_new, vf.y, rescale * o_reg[2 * i + 1]);
            }
        }
    }

    extern __shared__ char smem_sk_nvfp4[];
    crosswarp_reduce_splitk<HEAD_DIM>(reinterpret_cast<float*>(smem_sk_nvfp4), m_w, l_w, o_reg, warp_id,
                                      lane_id, lane_offset, partial_out, batch_idx, n_heads, head_idx,
                                      num_splits, split_idx);
}

// Note: a pipelined splitk variant (double-buffered K + V via smem, mirroring
// `paged_attention_splitk_int4_pipeline_kernel`) was tested 2026-05-08 and
// regressed Qwen3-8B Q8 + NVFP4 KV decode by ~3% (147.0 → 142.7 tok/s,
// 5 reps). Once the inner loop became HW-FP4-cvt-bound (PR landed earlier
// today) there was no longer enough work between issuing K[t+1] and using
// K[t] for the prefetch to hide global-memory latency. The int4 pattern
// works because INT4 dequant is heavier (8-entry LUT + sign branch). Don't
// re-attempt without first profiling to confirm the kernel is back to
// memory-bound (e.g. once block_size grows past 16 or HD>512 lands).
//
// That verdict still stands and it is not the one above. The cost was never
// the number of BYTES in flight, it was the number of LOAD INSTRUCTIONS: at
// HD=256 this kernel issued 8 separate LDG.E.U8 per token because a uint8_t*
// carries no provable alignment. `load_packed_fp4` reads the same bytes as
// one word per operand, and both operands are issued before the reduction
// (2026-08-30, #1817): 64.0 → 74.1 tok/s on Qwen3.8-27B-NVFP4 at 77k context,
// 20 → 2 LDG.E.U8 in SASS, 56 registers and zero spills before and after.

// ---------------------------------------------------------------------------
// Host launcher
// ---------------------------------------------------------------------------

void paged_attention_decode_nvfp4(const Tensor& Q, const Tensor& K_cache, const Tensor& V_cache, Tensor& O,
                                  const uint8_t* K_scales, const uint8_t* V_scales, const int* block_tables,
                                  const int* context_lens, int block_size, float scale, int max_context_len,
                                  int sliding_window, float softcap, cudaStream_t stream,
                                  int max_blocks_per_seq, int n_sinks, const void* attn_sinks) {
    (void)n_sinks;  // StreamingLLM not yet wired; LEARNED sinks are (#1345)
    const half* sinks_h = reinterpret_cast<const half*>(attn_sinks);
    const int batch_size = static_cast<int>(Q.shape[0]);
    const int n_heads = static_cast<int>(Q.shape[2]);
    const int head_dim = static_cast<int>(Q.shape[3]);
    const int n_kv_heads = static_cast<int>(K_cache.shape[2]);

    const int max_num_blocks = (max_blocks_per_seq > 0) ? max_blocks_per_seq
                                                        : (max_context_len + block_size - 1) / block_size;

    size_t smem_bytes = NUM_WARPS * sizeof(float) + NUM_WARPS * sizeof(float) +
                        NUM_WARPS * head_dim * sizeof(float);

    void* scratch_ptr = nullptr;
    int num_splits = compute_splitk_splits(batch_size, n_heads, head_dim, max_context_len, block_size,
                                           &scratch_ptr);

    if (num_splits > 1) {
        float* partial = static_cast<float*>(scratch_ptr);
        dim3 grid1(batch_size, n_heads, num_splits);
        dim3 block1(BLOCK_THREADS);

#define LAUNCH_SPLITK_NVFP4(HD)                                                                            \
    paged_attention_splitk_nvfp4_kernel<HD>                                                                \
        <<<grid1, block1, smem_bytes, stream>>>(reinterpret_cast<const half*>(Q.data),                     \
                                                reinterpret_cast<const uint8_t*>(K_cache.data),            \
                                                reinterpret_cast<const uint8_t*>(V_cache.data), K_scales,  \
                                                V_scales, partial, block_tables, context_lens, batch_size, \
                                                n_heads, n_kv_heads, block_size, scale, max_num_blocks,    \
                                                num_splits, sliding_window, softcap);                      \
    IMP_CUDA_CHECK_LAUNCH()

        switch (head_dim) {
            case 64:
                LAUNCH_SPLITK_NVFP4(64);
                break;
            case 128:
                LAUNCH_SPLITK_NVFP4(128);
                break;
            case 256:
                LAUNCH_SPLITK_NVFP4(256);
                break;
            case 512:
                LAUNCH_SPLITK_NVFP4(512);
                break;
            default:
                // #1674: logged and returned, leaving O unwritten.
                paged_attention_unsupported_head_dim("paged_attention_decode_nvfp4 splitk", head_dim);
        }
#undef LAUNCH_SPLITK_NVFP4

        paged_attention_launch_reduce(partial, reinterpret_cast<half*>(O.data), batch_size, n_heads, head_dim,
                                      num_splits, stream, sinks_h);
    } else {
        dim3 grid(batch_size, n_heads);
        dim3 block(BLOCK_THREADS);

#define LAUNCH_NVFP4(HD)                                                                                     \
    paged_attention_decode_nvfp4_kernel<HD><<<grid, block, smem_bytes, stream>>>(                            \
        reinterpret_cast<const half*>(Q.data), reinterpret_cast<const uint8_t*>(K_cache.data),               \
        reinterpret_cast<const uint8_t*>(V_cache.data), K_scales, V_scales, reinterpret_cast<half*>(O.data), \
        block_tables, context_lens, batch_size, n_heads, n_kv_heads, block_size, scale, max_context_len,     \
        max_num_blocks, sliding_window, softcap, sinks_h);                                                   \
    IMP_CUDA_CHECK_LAUNCH()

        switch (head_dim) {
            case 64:
                LAUNCH_NVFP4(64);
                break;
            case 128:
                LAUNCH_NVFP4(128);
                break;
            case 256:
                LAUNCH_NVFP4(256);
                break;
            case 512:
                LAUNCH_NVFP4(512);
                break;
            default:
                paged_attention_unsupported_head_dim("paged_attention_decode_nvfp4", head_dim);
        }
#undef LAUNCH_NVFP4
    }
}

// ---------------------------------------------------------------------------
// MXFP4-KV launcher — same kernel as NVFP4 but with UE8M0 scale decode.
// Pool layout and scale grouping are identical to NVFP4 (per design memo
// §3.1.2); only the scale byte semantics differ (UE8M0 vs E4M3).
// ---------------------------------------------------------------------------

void paged_attention_decode_mxfp4_kv(const Tensor& Q, const Tensor& K_cache, const Tensor& V_cache, Tensor& O,
                                     const uint8_t* K_scales, const uint8_t* V_scales,
                                     const int* block_tables, const int* context_lens, int block_size,
                                     float scale, int max_context_len, int sliding_window, float softcap,
                                     cudaStream_t stream, int max_blocks_per_seq, int n_sinks,
                                     const void* attn_sinks) {
    (void)n_sinks;  // StreamingLLM not yet wired; LEARNED sinks are (#1345)
    const half* sinks_h = reinterpret_cast<const half*>(attn_sinks);
    const int batch_size = static_cast<int>(Q.shape[0]);
    const int n_heads = static_cast<int>(Q.shape[2]);
    const int head_dim = static_cast<int>(Q.shape[3]);
    const int n_kv_heads = static_cast<int>(K_cache.shape[2]);

    const int max_num_blocks = (max_blocks_per_seq > 0) ? max_blocks_per_seq
                                                        : (max_context_len + block_size - 1) / block_size;

    size_t smem_bytes = NUM_WARPS * sizeof(float) + NUM_WARPS * sizeof(float) +
                        NUM_WARPS * head_dim * sizeof(float);

    void* scratch_ptr = nullptr;
    int num_splits = compute_splitk_splits(batch_size, n_heads, head_dim, max_context_len, block_size,
                                           &scratch_ptr);

    if (num_splits > 1) {
        float* partial = static_cast<float*>(scratch_ptr);
        dim3 grid1(batch_size, n_heads, num_splits);
        dim3 block1(BLOCK_THREADS);

#define LAUNCH_SPLITK_MXFP4KV(HD)                                                                          \
    paged_attention_splitk_nvfp4_kernel<HD, ScaleDtype::UE8M0>                                             \
        <<<grid1, block1, smem_bytes, stream>>>(reinterpret_cast<const half*>(Q.data),                     \
                                                reinterpret_cast<const uint8_t*>(K_cache.data),            \
                                                reinterpret_cast<const uint8_t*>(V_cache.data), K_scales,  \
                                                V_scales, partial, block_tables, context_lens, batch_size, \
                                                n_heads, n_kv_heads, block_size, scale, max_num_blocks,    \
                                                num_splits, sliding_window, softcap);                      \
    IMP_CUDA_CHECK_LAUNCH()

        switch (head_dim) {
            case 64:
                LAUNCH_SPLITK_MXFP4KV(64);
                break;
            case 128:
                LAUNCH_SPLITK_MXFP4KV(128);
                break;
            case 256:
                LAUNCH_SPLITK_MXFP4KV(256);
                break;
            case 512:
                LAUNCH_SPLITK_MXFP4KV(512);
                break;
            default:
                // #1674: logged and returned, leaving O unwritten.
                paged_attention_unsupported_head_dim("paged_attention_decode_mxfp4_kv splitk", head_dim);
        }
#undef LAUNCH_SPLITK_MXFP4KV

        paged_attention_launch_reduce(partial, reinterpret_cast<half*>(O.data), batch_size, n_heads, head_dim,
                                      num_splits, stream, sinks_h);
    } else {
        dim3 grid(batch_size, n_heads);
        dim3 block(BLOCK_THREADS);

#define LAUNCH_MXFP4KV(HD)                                                                                   \
    paged_attention_decode_nvfp4_kernel<HD, ScaleDtype::UE8M0><<<grid, block, smem_bytes, stream>>>(         \
        reinterpret_cast<const half*>(Q.data), reinterpret_cast<const uint8_t*>(K_cache.data),               \
        reinterpret_cast<const uint8_t*>(V_cache.data), K_scales, V_scales, reinterpret_cast<half*>(O.data), \
        block_tables, context_lens, batch_size, n_heads, n_kv_heads, block_size, scale, max_context_len,     \
        max_num_blocks, sliding_window, softcap, sinks_h);                                                   \
    IMP_CUDA_CHECK_LAUNCH()

        switch (head_dim) {
            case 64:
                LAUNCH_MXFP4KV(64);
                break;
            case 128:
                LAUNCH_MXFP4KV(128);
                break;
            case 256:
                LAUNCH_MXFP4KV(256);
                break;
            case 512:
                LAUNCH_MXFP4KV(512);
                break;
            default:
                paged_attention_unsupported_head_dim("paged_attention_decode_mxfp4_kv", head_dim);
        }
#undef LAUNCH_MXFP4KV
    }
}

}  // namespace imp
