#include "compute/attention_paged.h"
#include "compute/attention_paged_common.cuh"
#include "compute/attention.h"
#include "core/logging.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <mma.h>
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

// Decode one packed FP4 byte (low nibble = .x, high nibble = .y) → half2.
// Single PTX `cvt.rn.f16x2.e2m1x2` (sm_120+, CUDA 13.2+). Replaces the prior
// per-nibble 8-entry magnitude LUT + sign branch (~12 ops/byte → 1).
__device__ __forceinline__ half2 fp4_byte_to_half2(uint32_t byte_val) {
    uint32_t fp16x2;
    asm("{ .reg .b8 t; cvt.u8.u32 t, %1; cvt.rn.f16x2.e2m1x2 %0, t; }"
        : "=r"(fp16x2) : "r"(byte_val));
    return *reinterpret_cast<half2*>(&fp16x2);
}

// ---------------------------------------------------------------------------
// Non-Split-K NVFP4 decode kernel
// ---------------------------------------------------------------------------

template <int HEAD_DIM>
__global__ void paged_attention_decode_nvfp4_tc_kernel(
    const half* __restrict__ Q,
    const uint8_t* __restrict__ K_cache,    // packed FP4 pairs
    const uint8_t* __restrict__ V_cache,    // packed FP4 pairs
    const uint8_t* __restrict__ K_scales,   // UE4M3 per group
    const uint8_t* __restrict__ V_scales,   // UE4M3 per group
    half* __restrict__ O, const int* __restrict__ block_tables,
    const int* __restrict__ context_lens, int batch_size, int n_heads, int n_kv_heads, int block_size,
    float scale, int max_context_len, int max_num_blocks, int sliding_window, float softcap) {
    static_assert(HEAD_DIM % WARP_SIZE == 0, "HEAD_DIM must be divisible by WARP_SIZE");
    static_assert(HEAD_DIM % 16 == 0, "HEAD_DIM must be divisible by 16 (NVFP4 group size)");
    constexpr int ELEMS = HEAD_DIM / WARP_SIZE;

    // Per-warp WMMA scratch: sQ[16][16] halves + sK[16][16] halves +
    // sFV[16][16] floats = 2048 bytes/warp. Total: NUM_WARPS * 2048 = 16 KiB.
    // Comes BEFORE the crosswarp_reduce smem region.
    extern __shared__ char tc_smem_raw[];
    __half* tc_smem = reinterpret_cast<__half*>(tc_smem_raw);
    constexpr int WARP_TC_HALVES = 16 * 16 + 16 * 16 + 16 * 16 * 2;  // sQ + sK + sFV (float in halves)

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

        // ---------------------------------------------------------------
        // BitDecoding TC dispatch: WMMA Q.K^T over all valid tokens of this
        // page block at once. block_size <= 16 maps cleanly to a 16x16 WMMA
        // tile (m=16 rows of replicated Q × n=16 token columns, k=16
        // head-dim chunks). V accumulation remains per-token scalar in
        // Phase 1 (Phase 2 will TC the PV path).
        // ---------------------------------------------------------------
        constexpr int K_TILES = HEAD_DIM / 16;

        // Per-warp WMMA scratch
        __half* sQ_w = tc_smem + warp_id * WARP_TC_HALVES;
        __half* sK_w = sQ_w + 16 * 16;
        float*  sFV_w = reinterpret_cast<float*>(sK_w + 16 * 16);  // FP32 V WMMA accum store

        using namespace nvcuda;
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, 16, 16, 16, __half> c_frag;
        wmma::fill_fragment(c_frag, __float2half(0.0f));

        const int n_toks = tok_end - tok_start;

#pragma unroll
        for (int k_tile = 0; k_tile < K_TILES; k_tile++) {
            const int hd_off = k_tile * 16;

            // Replicate Q[hd_off..hd_off+16] across all 16 rows of sQ_w.
            // 16 threads suffice; spread over the warp's 32 lanes.
            for (int i = lane_id; i < 16 * 16; i += WARP_SIZE) {
                int col = i % 16;
                sQ_w[i] = Q_ptr[hd_off + col];
            }

            // Dequant 16 tokens × 16 hd-chunk into sK_w[token, hd_local].
            // Out-of-range tokens (past tok_end OR before first_tok) get 0
            // so their dot product is 0 → softmax weight 0 after scaling.
            for (int i = lane_id; i < 16 * 16; i += WARP_SIZE) {
                int t = i / 16;
                int hd_local = i % 16;
                int hd_global = hd_off + hd_local;
                if (t >= first_tok && t < n_toks) {
                    const uint8_t* K_tok = K_block + t * kv_slot_stride + kv_head * kv_head_bytes;
                    uint32_t b = K_tok[hd_global / 2];
                    half2 hh = fp4_byte_to_half2(b);
                    half v = (hd_global & 1) ? hh.y : hh.x;
                    float scale_k = ue4m3_decode(
                        K_sc_block[t * sc_slot_stride + kv_head * sc_groups + (hd_global / 16)]);
                    sK_w[i] = __float2half(__half2float(v) * scale_k);
                } else {
                    sK_w[i] = __float2half(0.0f);
                }
            }
            __syncwarp();

            wmma::load_matrix_sync(a_frag, sQ_w, 16);
            // Row-major sK_w loaded with col_major declaration → effectively
            // transposed: B[hd, tok] = sK_w[tok][hd].
            wmma::load_matrix_sync(b_frag, sK_w, 16);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            __syncwarp();
        }

        // Store accumulator back to sK_w. Row 0 holds the 16 token dots.
        wmma::store_matrix_sync(sK_w, c_frag, 16, wmma::mem_row_major);
        __syncwarp();

        // ---------------------------------------------------------------
        // Phase 2 BISECT STEP A: block-softmax + SCALAR V accum.
        //
        // Critical: Phase 1's online_softmax_step (in attention_paged_common.cuh)
        // produces NORMALIZED rescale + w_new (divided by l_new at each step),
        // so o_reg is always normalized = (running sum exp(d-m_w) V) / l_w.
        // crosswarp_reduce_and_write assumes warp_o is normalized — it computes
        //   weight = exp(m_w - global_max) * l_w
        //   o_val += weight * warp_o[w]
        //   o_val /= global_l
        // For this to give the correct global attention, warp_o must be the
        // l_w-divided normalized form. Block-softmax must preserve that
        // invariant.
        // ---------------------------------------------------------------

        // Pass 1: read all 16 dots, find local max
        float dots_scaled[16];
        float m_local = -FLT_MAX;
#pragma unroll
        for (int t = 0; t < 16; t++) {
            float d = -FLT_MAX;
            if (t >= first_tok && t < n_toks) {
                d = __half2float(sK_w[t]) * scale;
                d = apply_softcap(d, softcap);
                m_local = fmaxf(m_local, d);
            }
            dots_scaled[t] = d;
        }

        float m_new = fmaxf(m_w, m_local);
        float exp_diff = (m_w == -FLT_MAX) ? 0.0f : __expf(m_w - m_new);

        // Pass 2: per-token un-normalized weights + l_local
        float weights[16];
        float l_local = 0.0f;
#pragma unroll
        for (int t = 0; t < 16; t++) {
            float w = (dots_scaled[t] > -FLT_MAX) ? __expf(dots_scaled[t] - m_new) : 0.0f;
            weights[t] = w;
            l_local += w;
        }

        float l_new = exp_diff * l_w + l_local;
        // Normalized rescale: (exp_diff * l_w_old) / l_new — same role as
        // Phase 1's online_softmax_step.rescale, applied once per block.
        float rescale_norm = (l_new > 0.0f) ? (exp_diff * l_w / l_new) : 0.0f;
        float l_inv = (l_new > 0.0f) ? (1.0f / l_new) : 0.0f;

        m_w = m_new;
        l_w = l_new;

        // Apply normalized rescale to existing o_reg
#pragma unroll
        for (int i = 0; i < ELEMS; i++) o_reg[i] *= rescale_norm;

        // ---------------------------------------------------------------
        // WMMA V accum: D[m=16, n=16] = A[m, k]=normalized_weights[k] × B[k=tok, n=hd_local]
        // (B as row_major reads sK_w[tok*16+hd] = V[tok][hd] directly — no
        //  transpose needed; different from QK phase's col_major B which
        //  intentionally transposes K.)
        //
        // Per-lane scatter into o_reg: each lane owns ELEMS contiguous
        // absolute hd offsets within ONE 16-element chunk, so exactly one
        // kt iteration's contributions land in this lane's o_reg.
        // ---------------------------------------------------------------
        constexpr int LANES_PER_CHUNK = (16 / ELEMS) > 0 ? (16 / ELEMS) : 1;
        const int my_chunk = lane_id / LANES_PER_CHUNK;
        const int my_offset_in_chunk = (lane_id % LANES_PER_CHUNK) * ELEMS;

        // Replicate normalized weights into sQ_w[16, 16]: A[m, k] = w_norm[k]
        for (int i = lane_id; i < 16 * 16; i += WARP_SIZE) {
            int col = i % 16;
            sQ_w[i] = __float2half(weights[col] * l_inv);
        }
        __syncwarp();

        // FP32 accumulator for V — sums over 16 weighted tokens accumulate
        // enough error in FP16 to drift output after a few decode steps.
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> v_frag;
        // V phase B operand is row_major (token in k_dim, hd in n_dim).
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b_frag_v;

#pragma unroll
        for (int kt = 0; kt < K_TILES; kt++) {
            const int hd_off = kt * 16;

            // Dequant V[16 tokens × 16 hd_chunk] into sK_w[token, hd_local]
            for (int i = lane_id; i < 16 * 16; i += WARP_SIZE) {
                int t = i / 16;
                int hd_local = i % 16;
                int hd_global = hd_off + hd_local;
                if (t >= first_tok && t < n_toks) {
                    const uint8_t* V_tok = V_block + t * kv_slot_stride + kv_head * kv_head_bytes;
                    uint32_t b = V_tok[hd_global / 2];
                    half2 hh = fp4_byte_to_half2(b);
                    half v = (hd_global & 1) ? hh.y : hh.x;
                    float scale_v = ue4m3_decode(
                        V_sc_block[t * sc_slot_stride + kv_head * sc_groups + (hd_global / 16)]);
                    sK_w[i] = __float2half(__half2float(v) * scale_v);
                } else {
                    sK_w[i] = __float2half(0.0f);
                }
            }
            __syncwarp();

            wmma::fill_fragment(v_frag, 0.0f);
            wmma::load_matrix_sync(a_frag, sQ_w, 16);
            wmma::load_matrix_sync(b_frag_v, sK_w, 16);
            wmma::mma_sync(v_frag, a_frag, b_frag_v, v_frag);

            // Store v_frag (FP32 accumulator) into sFV_w. Row 0 = 16 contributions.
            wmma::store_matrix_sync(sFV_w, v_frag, 16, wmma::mem_row_major);
            __syncwarp();

            // Per-lane scatter: only lanes whose owned chunk matches kt accumulate.
            if (my_chunk == kt) {
#pragma unroll
                for (int e = 0; e < ELEMS; e++) {
                    o_reg[e] += sFV_w[my_offset_in_chunk + e];
                }
            }
            __syncwarp();
        }
    }

    // crosswarp reduce smem starts AFTER the per-warp TC scratch region
    // (NUM_WARPS * WARP_TC_HALVES halves = NUM_WARPS * 1024 bytes).
    char* crosswarp_smem = tc_smem_raw + NUM_WARPS * WARP_TC_HALVES * sizeof(__half);
    crosswarp_reduce_and_write<HEAD_DIM>(reinterpret_cast<float*>(crosswarp_smem), m_w, l_w, o_reg, warp_id,
                                         lane_id, lane_offset, O, batch_idx, n_heads, head_idx);
}

// ---------------------------------------------------------------------------
// Split-K NVFP4 decode kernel
// ---------------------------------------------------------------------------

template <int HEAD_DIM>
__global__ void paged_attention_splitk_nvfp4_tc_kernel(
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

            float k_scale = ue4m3_decode(
                K_sc_block[t * sc_slot_stride + kv_head * sc_groups + lane_group]);
            float v_scale = ue4m3_decode(
                V_sc_block[t * sc_slot_stride + kv_head * sc_groups + lane_group]);
            const half2 k_scale_h2 = __float2half2_rn(k_scale);
            const half2 v_scale_h2 = __float2half2_rn(v_scale);

            float dot = 0.0f;
            {
                const uint8_t* k_bytes = K_tok + lane_offset / 2;
#pragma unroll
                for (int i = 0; i < ELEMS / 2; i++) {
                    half2 kh2 = fp4_byte_to_half2(k_bytes[i]);
                    kh2 = __hmul2(kh2, k_scale_h2);
                    float2 kf = __half22float2(kh2);
                    dot = __fmaf_rn(q_reg[2 * i], kf.x, dot);
                    dot = __fmaf_rn(q_reg[2 * i + 1], kf.y, dot);
                }
            }
            dot = warp_reduce_sum(dot);
            dot *= scale;
            dot = apply_softcap(dot, softcap);

            float rescale, w_new;
            online_softmax_step(dot, m_w, l_w, rescale, w_new);

            {
                const uint8_t* v_bytes = V_tok + lane_offset / 2;
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

// ---------------------------------------------------------------------------
// Host launcher
// ---------------------------------------------------------------------------

void paged_attention_decode_nvfp4_tc(const Tensor& Q, const Tensor& K_cache, const Tensor& V_cache, Tensor& O,
                                  const uint8_t* K_scales, const uint8_t* V_scales, const int* block_tables,
                                  const int* context_lens, int block_size, float scale, int max_context_len,
                                  int sliding_window, float softcap, cudaStream_t stream,
                                  int max_blocks_per_seq, int n_sinks) {
    (void)n_sinks;  // streaming not yet wired
    const int batch_size = static_cast<int>(Q.shape[0]);
    const int n_heads = static_cast<int>(Q.shape[2]);
    const int head_dim = static_cast<int>(Q.shape[3]);
    const int n_kv_heads = static_cast<int>(K_cache.shape[2]);

    const int max_num_blocks = (max_blocks_per_seq > 0) ? max_blocks_per_seq
                                                        : (max_context_len + block_size - 1) / block_size;

    // BitDecoding TC variant adds NUM_WARPS * 2048 bytes of WMMA scratch
    // (16x16 sQ halves + 16x16 sK halves + 16x16 sFV floats per warp).
    // Total: 8 warps × 2048 = 16 KiB. Comes BEFORE the crosswarp_reduce smem.
    constexpr size_t TC_SCRATCH_PER_WARP =
        (16 * 16) * sizeof(__half) + (16 * 16) * sizeof(__half) + (16 * 16) * sizeof(float);
    size_t smem_bytes = NUM_WARPS * TC_SCRATCH_PER_WARP +
                        NUM_WARPS * sizeof(float) + NUM_WARPS * sizeof(float) +
                        NUM_WARPS * head_dim * sizeof(float);

    void* scratch_ptr = nullptr;
    int num_splits = compute_splitk_splits(batch_size, n_heads, head_dim, max_context_len, block_size,
                                           &scratch_ptr);

    if (num_splits > 1) {
        float* partial = static_cast<float*>(scratch_ptr);
        dim3 grid1(batch_size, n_heads, num_splits);
        dim3 block1(BLOCK_THREADS);

#define LAUNCH_SPLITK_NVFP4(HD)                                                                            \
    paged_attention_splitk_nvfp4_tc_kernel<HD>                                                                \
        <<<grid1, block1, smem_bytes, stream>>>(reinterpret_cast<const half*>(Q.data),                     \
                                                reinterpret_cast<const uint8_t*>(K_cache.data),            \
                                                reinterpret_cast<const uint8_t*>(V_cache.data), K_scales,  \
                                                V_scales, partial, block_tables, context_lens, batch_size, \
                                                n_heads, n_kv_heads, block_size, scale, max_num_blocks,    \
                                                num_splits, sliding_window, softcap)

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
                IMP_LOG_ERROR("paged_attention_decode_nvfp4_tc splitk: unsupported head_dim %d", head_dim);
                return;
        }
#undef LAUNCH_SPLITK_NVFP4

        paged_attention_launch_reduce(partial, reinterpret_cast<half*>(O.data), batch_size, n_heads, head_dim,
                                      num_splits, stream);
    } else {
        dim3 grid(batch_size, n_heads);
        dim3 block(BLOCK_THREADS);

#define LAUNCH_NVFP4(HD)                                                                                     \
    paged_attention_decode_nvfp4_tc_kernel<HD><<<grid, block, smem_bytes, stream>>>(                            \
        reinterpret_cast<const half*>(Q.data), reinterpret_cast<const uint8_t*>(K_cache.data),               \
        reinterpret_cast<const uint8_t*>(V_cache.data), K_scales, V_scales, reinterpret_cast<half*>(O.data), \
        block_tables, context_lens, batch_size, n_heads, n_kv_heads, block_size, scale, max_context_len,     \
        max_num_blocks, sliding_window, softcap)

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
                IMP_LOG_ERROR("paged_attention_decode_nvfp4_tc: unsupported head_dim %d", head_dim);
                return;
        }
#undef LAUNCH_NVFP4
    }
}

}  // namespace imp
