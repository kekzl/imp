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

// Kernel uses no __launch_bounds__: with the dots/weights moved to shared
// mem (warp-shfl reduction, see block-softmax section) the spill is gone
// (cuobjdump STACK:0 across HD ∈ {64,128,256,512}); the compiler picks the
// best occupancy/register trade-off automatically.
template <int HEAD_DIM>
__global__ void paged_attention_decode_nvfp4_tc_kernel(
    const half* __restrict__ Q,
    const uint8_t* __restrict__ K_cache,    // packed FP4 pairs
    const uint8_t* __restrict__ V_cache,    // packed FP4 pairs
    const uint8_t* __restrict__ K_scales,   // UE4M3 per group
    const uint8_t* __restrict__ V_scales,   // UE4M3 per group
    half* __restrict__ O, const int* __restrict__ block_tables,
    const int* __restrict__ context_lens, int batch_size, int n_heads, int n_kv_heads, int block_size,
    float scale, int max_context_len, int max_num_blocks, int sliding_window, float softcap,
    // Phase 3b residual args.
    // Single-seq form (batch_size==1): K_residual / V_residual point at the
    //   (seq, layer) slice; residual_count_scalar / residual_write_idx_scalar
    //   carry per-seq state. Layout [residual_n_tokens, n_kv_heads, head_dim].
    // Multi-seq form (batch_size>=1): K_residual_base / V_residual_base point at
    //   slot 0's (K|V) data for this layer; residual_seq_stride_elems is the FP16
    //   stride between slots; d_residual_seq_slots/_counts/_write_idxes are device
    //   arrays of length batch_size, indexed by blockIdx.x.
    // Multi-seq is selected when d_residual_seq_slots != nullptr.
    const half* __restrict__ K_residual, const half* __restrict__ V_residual,
    int residual_count_scalar, int residual_n_tokens, int residual_write_idx_scalar,
    const half* __restrict__ K_residual_base, const half* __restrict__ V_residual_base,
    int residual_seq_stride_elems, const int* __restrict__ d_residual_seq_slots,
    const int* __restrict__ d_residual_counts, const int* __restrict__ d_residual_write_idxes) {
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

    // Phase 3b: pick residual form (single-seq scalar vs multi-seq array)
    // and resolve per-batch K/V residual pointer + count + write_idx.
    const half* K_res_ptr = nullptr;
    const half* V_res_ptr = nullptr;
    int residual_count = 0;
    int residual_write_idx = 0;
    if (d_residual_seq_slots != nullptr && K_residual_base != nullptr && residual_n_tokens > 0) {
        // Multi-seq array form
        int seq_slot = d_residual_seq_slots[batch_idx];
        int rc = (d_residual_counts != nullptr) ? d_residual_counts[batch_idx] : 0;
        int rw = (d_residual_write_idxes != nullptr) ? d_residual_write_idxes[batch_idx] : 0;
        if (seq_slot >= 0 && rc > 0) {
            K_res_ptr = K_residual_base + (int64_t)seq_slot * residual_seq_stride_elems;
            V_res_ptr = V_residual_base + (int64_t)seq_slot * residual_seq_stride_elems;
            residual_count = rc;
            residual_write_idx = rw;
        }
    } else if (K_residual != nullptr && V_residual != nullptr && residual_count_scalar > 0 &&
               residual_n_tokens > 0) {
        // Single-seq scalar form
        K_res_ptr = K_residual;
        V_res_ptr = V_residual;
        residual_count = residual_count_scalar;
        residual_write_idx = residual_write_idx_scalar;
    }

    // Split absolute KV range into paged-tail-clipped + residual.
    // residual_have: residual is enabled for this batch AND has at least one
    // entry within both ctx_len and the [effective_start, ctx_len) window.
    const bool residual_have = (K_res_ptr != nullptr) && (V_res_ptr != nullptr) &&
                               (residual_count > 0);
    int residual_active_count = 0;
    int residual_first_abs = ctx_len;  // start of residual chronological range
    int residual_skip = 0;             // chronologically-stale residual entries to skip
    if (residual_have) {
        int res_have = residual_count;
        if (res_have > ctx_len) res_have = ctx_len;
        int res_chrono_first = ctx_len - res_have;
        residual_first_abs = (effective_start > res_chrono_first) ? effective_start : res_chrono_first;
        residual_active_count = ctx_len - residual_first_abs;
        residual_skip = res_have - residual_active_count;
        if (residual_active_count <= 0) {
            residual_active_count = 0;  // window excludes all residual entries
        }
    }
    const int paged_end_token = ctx_len - residual_active_count;
    const int first_block = effective_start / block_size;
    const int num_paged_blocks = (paged_end_token + block_size - 1) / block_size;

    float m_w = -FLT_MAX;
    float l_w = 0.0f;
    float o_reg[ELEMS];
#pragma unroll
    for (int i = 0; i < ELEMS; i++)
        o_reg[i] = 0.0f;

    for (int blk = first_block + warp_id; blk < num_paged_blocks; blk += NUM_WARPS) {
        int phys_block = bt[blk];
        const uint8_t* K_block = K_cache + (int64_t)phys_block * kv_block_stride;
        const uint8_t* V_block = V_cache + (int64_t)phys_block * kv_block_stride;
        const uint8_t* K_sc_block = K_scales + (int64_t)phys_block * sc_block_stride;
        const uint8_t* V_sc_block = V_scales + (int64_t)phys_block * sc_block_stride;

        int tok_start = blk * block_size;
        int tok_end = tok_start + block_size;
        if (tok_end > paged_end_token)
            tok_end = paged_end_token;

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

        // Block-softmax with shared-mem dots/weights to avoid the per-thread
        // float[16] arrays (forced a 64-byte stack spill into DRAM-backed
        // local memory at HD>=128, dominated decode tg/s).
        //
        // Lanes 0..15 each compute one entry; full 32-lane warp-shfl reduce
        // for m_local / l_local so EVERY lane sees the same scalar (the
        // sQ_w fill below has lanes 16..31 also reading weights[col], so
        // they need consistent l_inv).
        float* dots_smem_p = reinterpret_cast<float*>(sK_w) + 16 * 16 / 2;  // alias unused half region
        // Use the front of sFV_w for dots/weights — sFV_w is per-warp (1024B
        // floats = 256 entries) and only used after this for V WMMA store.
        float* dots_smem = sFV_w;
        float* weights_smem = sFV_w + 16;

        const bool t_active_p = (lane_id < 16) && (lane_id >= first_tok) && (lane_id < n_toks);
        float my_dot_p = -FLT_MAX;
        if (t_active_p) {
            my_dot_p = __half2float(sK_w[lane_id]) * scale;
            my_dot_p = apply_softcap(my_dot_p, softcap);
        }
        if (lane_id < 16) dots_smem[lane_id] = my_dot_p;
        __syncwarp();

        float m_local = my_dot_p;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            m_local = fmaxf(m_local, __shfl_xor_sync(0xffffffff, m_local, off));
        }

        float m_new = fmaxf(m_w, m_local);
        float exp_diff = (m_w == -FLT_MAX) ? 0.0f : __expf(m_w - m_new);

        float my_weight_p = 0.0f;
        if (t_active_p && my_dot_p > -FLT_MAX) {
            my_weight_p = __expf(my_dot_p - m_new);
        }
        if (lane_id < 16) weights_smem[lane_id] = my_weight_p;
        __syncwarp();

        float l_local = my_weight_p;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            l_local += __shfl_xor_sync(0xffffffff, l_local, off);
        }
        (void)dots_smem_p;  // alias not used; keep dots in sFV_w for clarity

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

        // Replicate normalized weights into sQ_w[16, 16]: A[m, k] = w_norm[k].
        // Reads weights_smem[col] (not a per-thread array) — bank-conflict-free
        // since 32 lanes broadcast-read 16 distinct addresses.
        for (int i = lane_id; i < 16 * 16; i += WARP_SIZE) {
            int col = i % 16;
            sQ_w[i] = __float2half(weights_smem[col] * l_inv);
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

    // ------------------------------------------------------------------
    // Phase 3b: residual FP16 pass over the newest `residual_active_count`
    // tokens. Same WMMA QK + block-softmax + WMMA V structure as the paged
    // loop, but reads K/V directly as FP16 from the residual ring (no FP4
    // dequant, no UE4M3 fold). Tiles distribute round-robin across warps;
    // each warp's m_w/l_w/o_reg evolves independently and the cross-warp
    // reduce later integrates them correctly.
    //
    // Ring slot for chronological position i in the active range:
    //   slot = (residual_write_idx + residual_n_tokens - residual_count
    //          + residual_skip + i) % residual_n_tokens
    // ------------------------------------------------------------------
    if (residual_active_count > 0) {
        const int kv_head_stride_res = HEAD_DIM;            // FP16 elems per (slot, kv_head)
        const int slot_stride_res = n_kv_heads * HEAD_DIM;  // FP16 elems per slot

        // Per-warp scratch (same layout as paged path)
        __half* sQ_r = tc_smem + warp_id * WARP_TC_HALVES;
        __half* sK_r = sQ_r + 16 * 16;
        float*  sFV_r = reinterpret_cast<float*>(sK_r + 16 * 16);

        constexpr int K_TILES_R = HEAD_DIM / 16;

        const int n_tiles_r = (residual_active_count + 15) / 16;
        const int slot_base = (residual_write_idx + residual_n_tokens - residual_count + residual_skip)
                              % residual_n_tokens;

        // Fragments declared INSIDE the loop so they don't allocate registers
        // for warps that skip the loop entirely (register pressure cuts kernel
        // occupancy and slows the paged loop, which is the dominant cost at
        // long context — verified via A/B at ctx=1024 vs 4096).
        for (int tile = warp_id; tile < n_tiles_r; tile += NUM_WARPS) {
            using namespace nvcuda;
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b_frag_v;
            wmma::fragment<wmma::accumulator, 16, 16, 16, __half> c_frag;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> v_frag;

            const int tile_first = tile * 16;
            int tile_count = residual_active_count - tile_first;
            if (tile_count > 16) tile_count = 16;

            wmma::fill_fragment(c_frag, __float2half(0.0f));

#pragma unroll
            for (int k_tile = 0; k_tile < K_TILES_R; k_tile++) {
                const int hd_off = k_tile * 16;

                // Replicate Q[hd_off..hd_off+16] across all 16 rows of sQ_r.
                for (int i = lane_id; i < 16 * 16; i += WARP_SIZE) {
                    int col = i % 16;
                    sQ_r[i] = Q_ptr[hd_off + col];
                }

                // Load FP16 K from residual ring. Out-of-tile slots → 0.
                for (int i = lane_id; i < 16 * 16; i += WARP_SIZE) {
                    int t = i / 16;
                    int hd_local = i % 16;
                    int hd_global = hd_off + hd_local;
                    if (t < tile_count) {
                        int slot = (slot_base + tile_first + t) % residual_n_tokens;
                        sK_r[i] = K_res_ptr[(int64_t)slot * slot_stride_res +
                                            kv_head * kv_head_stride_res + hd_global];
                    } else {
                        sK_r[i] = __float2half(0.0f);
                    }
                }
                __syncwarp();

                wmma::load_matrix_sync(a_frag, sQ_r, 16);
                wmma::load_matrix_sync(b_frag, sK_r, 16);
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
                __syncwarp();
            }

            wmma::store_matrix_sync(sK_r, c_frag, 16, wmma::mem_row_major);
            __syncwarp();

            // Block-softmax (mirrors paged loop's normalized-rescale invariant).
            //
            // dots and weights live in shared memory rather than per-thread
            // register arrays. Per-thread `float[16]` forced a 64-byte stack
            // spill (cuobjdump STACK:64 at HD>=128) that turned the residual
            // pass into a 3× tg/s regression on the long-context decode path
            // — every spilled access is a DRAM round-trip. With shared-mem
            // tables, lanes 0..15 each compute one entry, the warp shfl-
            // reduces for m_local / l_local, and the V-phase A-operand fill
            // reads from shared memory instead of registers. Reuses the front
            // 32 floats of sFV_r (which is 16×16=256 floats so plenty of room).
            float* dots_smem = sFV_r;             // [16] floats per warp
            float* weights_smem = sFV_r + 16;     // [16] floats per warp

            const bool t_active = (lane_id < 16) && (lane_id < tile_count);
            float my_dot = -FLT_MAX;
            if (t_active) {
                my_dot = __half2float(sK_r[lane_id]) * scale;
                my_dot = apply_softcap(my_dot, softcap);
            }
            if (lane_id < 16) dots_smem[lane_id] = my_dot;
            __syncwarp();

            // Full 32-lane warp reduce so EVERY lane (including 16..31, which
            // have my_dot = -FLT_MAX) sees the same m_local. Skipping the
            // off=16 step would leave lanes 16..31 with m_local=-FLT_MAX,
            // diverging the subsequent m_new / exp_diff / l_inv computation
            // and corrupting the per-lane sQ_r weight fill (each lane writes
            // some rows of the A operand; inconsistent l_inv → garbage WMMA).
            float m_local = my_dot;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                m_local = fmaxf(m_local, __shfl_xor_sync(0xffffffff, m_local, off));
            }

            float m_new = fmaxf(m_w, m_local);
            float exp_diff = (m_w == -FLT_MAX) ? 0.0f : __expf(m_w - m_new);

            float my_weight = 0.0f;
            if (t_active && my_dot > -FLT_MAX) {
                my_weight = __expf(my_dot - m_new);
            }
            if (lane_id < 16) weights_smem[lane_id] = my_weight;
            __syncwarp();

            float l_local = my_weight;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                l_local += __shfl_xor_sync(0xffffffff, l_local, off);
            }

            float l_new = exp_diff * l_w + l_local;
            float rescale_norm = (l_new > 0.0f) ? (exp_diff * l_w / l_new) : 0.0f;
            float l_inv = (l_new > 0.0f) ? (1.0f / l_new) : 0.0f;

            m_w = m_new;
            l_w = l_new;

#pragma unroll
            for (int i = 0; i < ELEMS; i++) o_reg[i] *= rescale_norm;

            // Replicate normalized weights into sQ_r (A operand). Each lane
            // reads its column's weight from shared mem — bank-conflict-free
            // since 32 lanes broadcast-read 16 distinct addresses.
            for (int i = lane_id; i < 16 * 16; i += WARP_SIZE) {
                int col = i % 16;
                sQ_r[i] = __float2half(weights_smem[col] * l_inv);
            }
            __syncwarp();

            constexpr int LANES_PER_CHUNK_R = (16 / ELEMS) > 0 ? (16 / ELEMS) : 1;
            const int my_chunk_r = lane_id / LANES_PER_CHUNK_R;
            const int my_offset_in_chunk_r = (lane_id % LANES_PER_CHUNK_R) * ELEMS;

#pragma unroll
            for (int kt = 0; kt < K_TILES_R; kt++) {
                const int hd_off = kt * 16;

                // Load FP16 V from residual ring.
                for (int i = lane_id; i < 16 * 16; i += WARP_SIZE) {
                    int t = i / 16;
                    int hd_local = i % 16;
                    int hd_global = hd_off + hd_local;
                    if (t < tile_count) {
                        int slot = (slot_base + tile_first + t) % residual_n_tokens;
                        sK_r[i] = V_res_ptr[(int64_t)slot * slot_stride_res +
                                            kv_head * kv_head_stride_res + hd_global];
                    } else {
                        sK_r[i] = __float2half(0.0f);
                    }
                }
                __syncwarp();

                wmma::fill_fragment(v_frag, 0.0f);
                wmma::load_matrix_sync(a_frag, sQ_r, 16);
                wmma::load_matrix_sync(b_frag_v, sK_r, 16);
                wmma::mma_sync(v_frag, a_frag, b_frag_v, v_frag);
                wmma::store_matrix_sync(sFV_r, v_frag, 16, wmma::mem_row_major);
                __syncwarp();

                if (my_chunk_r == kt) {
#pragma unroll
                    for (int e = 0; e < ELEMS; e++) {
                        o_reg[e] += sFV_r[my_offset_in_chunk_r + e];
                    }
                }
                __syncwarp();
            }
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

// ---------------------------------------------------------------------------
// Phase 3b residual + reduce kernel.
//
// Replaces `paged_attention_reduce_kernel` for the residual path. Reads the
// per-split paged partials from `partial_out` (written by the existing splitk
// paged kernel), processes the FP16 residual ring tokens, and emits the final
// merged O.
//
// Why a fused kernel: profiling (nsys, 2026-05-09) showed the original Phase 3b
// design — embed the residual pass into the non-splitk decode kernel — forced
// the launcher onto the slow non-splitk path (414 µs/call vs splitk at 33
// µs/call, 12× regression). Letting splitk run normally and folding residual
// into the reduce kernel preserves the splitk parallelism while still matching
// the residual-FP16 contribution.
//
// Math (write-through residual: paged contains all tokens, residual contains
// the same tail tokens at higher precision):
//   m_paged   = max over paged splits of m_s
//   l_paged   = sum_s exp(m_s - m_paged) * l_s
//   O_paged_unnorm[d] = sum_s exp(m_s - m_paged) * partial_out[s, 2+d]
//                     = sum over all paged tokens of exp(d_t - m_paged) V_t
//   m_res, l_res, O_res_unnorm[d]: same accumulation over residual tokens.
//   m_global  = max(m_paged, m_res)
//   l_global  = exp(m_paged - m_global) * l_paged + exp(m_res - m_global) * l_res
//   O_final[d] = (exp(m_paged - m_global) * O_paged_unnorm[d] +
//                 exp(m_res - m_global) * O_res_unnorm[d]) / l_global
//
// One block per (batch, head). NUM_WARPS warps cooperatively process the
// residual tokens (round-robin); thread 0 reduces paged partials.
// ---------------------------------------------------------------------------
template <int HEAD_DIM>
__global__ void paged_attention_residual_reduce_kernel(
    const float* __restrict__ partial_out,           // [b, h, num_paged_splits, 2+HD]
    const half* __restrict__ Q,                      // [b, 1, n_heads, hd]
    half* __restrict__ O,                            // [b, 1, n_heads, hd]
    const int* __restrict__ context_lens,            // [b]
    int n_heads, int n_kv_heads, int num_paged_splits,
    float scale, int sliding_window, float softcap,
    int residual_n_tokens,
    // Single-seq scalar form (active when d_residual_seq_slots == nullptr):
    const half* __restrict__ K_residual,
    const half* __restrict__ V_residual,
    int residual_count_scalar, int residual_write_idx_scalar,
    // Multi-seq array form:
    const half* __restrict__ K_residual_base,
    const half* __restrict__ V_residual_base,
    int residual_seq_stride_elems,
    const int* __restrict__ d_residual_seq_slots,
    const int* __restrict__ d_residual_counts,
    const int* __restrict__ d_residual_write_idxes) {
    static_assert(HEAD_DIM % WARP_SIZE == 0, "HEAD_DIM must be divisible by WARP_SIZE");
    constexpr int ELEMS = HEAD_DIM / WARP_SIZE;
    constexpr int partial_stride = 2 + HEAD_DIM;

    const int batch_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int kv_head = head_idx / (n_heads / n_kv_heads);

    const int ctx_len = context_lens[batch_idx];
    if (ctx_len <= 0) return;

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int lane_offset = lane_id * ELEMS;

    // Resolve residual data (single-seq scalar OR multi-seq array form).
    const half* K_res = nullptr;
    const half* V_res = nullptr;
    int rc = 0, rwi = 0;
    if (d_residual_seq_slots != nullptr && K_residual_base != nullptr && residual_n_tokens > 0) {
        int slot = d_residual_seq_slots[batch_idx];
        int rc_b = (d_residual_counts != nullptr) ? d_residual_counts[batch_idx] : 0;
        int rw_b = (d_residual_write_idxes != nullptr) ? d_residual_write_idxes[batch_idx] : 0;
        if (slot >= 0 && rc_b > 0) {
            K_res = K_residual_base + (int64_t)slot * residual_seq_stride_elems;
            V_res = V_residual_base + (int64_t)slot * residual_seq_stride_elems;
            rc = rc_b;
            rwi = rw_b;
        }
    } else if (K_residual != nullptr && V_residual != nullptr && residual_count_scalar > 0 &&
               residual_n_tokens > 0) {
        K_res = K_residual;
        V_res = V_residual;
        rc = residual_count_scalar;
        rwi = residual_write_idx_scalar;
    }

    int effective_start = 0;
    if (sliding_window > 0 && ctx_len > sliding_window)
        effective_start = ctx_len - sliding_window;
    int res_active = 0;
    int res_skip = 0;
    if (K_res != nullptr) {
        int rh = (rc > ctx_len) ? ctx_len : rc;
        int rcf = ctx_len - rh;
        int rfa = (effective_start > rcf) ? effective_start : rcf;
        res_active = ctx_len - rfa;
        res_skip = rh - res_active;
        if (res_active < 0) res_active = 0;
    }

    // Step 1: compute (m_paged, l_paged) from per-split partials. Single-thread
    // reduction over a small (≤ 32) split count — same pattern as the standard
    // reduce kernel.
    const float* paged_base = partial_out +
        (int64_t)((batch_idx * n_heads + head_idx) * num_paged_splits) * partial_stride;

    __shared__ float s_m_paged;
    __shared__ float s_l_paged;
    __shared__ float s_m_res;
    __shared__ float s_l_res;

    if (threadIdx.x == 0) {
        float gmax = -FLT_MAX;
        for (int s = 0; s < num_paged_splits; s++) {
            gmax = fmaxf(gmax, paged_base[s * partial_stride]);
        }
        s_m_paged = gmax;
        float gl = 0.0f;
        for (int s = 0; s < num_paged_splits; s++) {
            float m = paged_base[s * partial_stride];
            float l = paged_base[s * partial_stride + 1];
            gl += expf(m - gmax) * l;
        }
        s_l_paged = gl;
    }
    __syncthreads();

    const float m_paged = s_m_paged;
    const float l_paged = s_l_paged;

    // Step 2: process residual tokens. Round-robin distribution across warps;
    // each warp tracks its own (m_w, l_w, o_reg) state.
    extern __shared__ char smem_red[];
    float* warp_max = reinterpret_cast<float*>(smem_red);   // [NUM_WARPS]
    float* warp_l   = warp_max + NUM_WARPS;                  // [NUM_WARPS]
    float* warp_o   = warp_l + NUM_WARPS;                    // [NUM_WARPS * HEAD_DIM]

    if (res_active > 0) {
        const half* Q_ptr = Q + (int64_t)batch_idx * n_heads * HEAD_DIM + (int64_t)head_idx * HEAD_DIM;
        float q_reg[ELEMS];
        const half2* Q_ptr2 = reinterpret_cast<const half2*>(Q_ptr + lane_offset);
#pragma unroll
        for (int i = 0; i < ELEMS / 2; i++) {
            half2 h2 = Q_ptr2[i];
            q_reg[2 * i] = __half2float(h2.x);
            q_reg[2 * i + 1] = __half2float(h2.y);
        }

        const int slot_stride = n_kv_heads * HEAD_DIM;
        const int sb = (rwi + residual_n_tokens - rc + res_skip) % residual_n_tokens;

        float m_w = -FLT_MAX;
        float l_w = 0.0f;
        float o_reg[ELEMS];
#pragma unroll
        for (int i = 0; i < ELEMS; i++) o_reg[i] = 0.0f;

        for (int t = warp_id; t < res_active; t += NUM_WARPS) {
            int slot = (sb + t) % residual_n_tokens;
            const half* K_tok = K_res + (int64_t)slot * slot_stride + kv_head * HEAD_DIM;
            const half* V_tok = V_res + (int64_t)slot * slot_stride + kv_head * HEAD_DIM;

            float dot = 0.0f;
            const half2* k_h2 = reinterpret_cast<const half2*>(K_tok + lane_offset);
#pragma unroll
            for (int i = 0; i < ELEMS / 2; i++) {
                half2 kh = k_h2[i];
                float2 kf = __half22float2(kh);
                dot = __fmaf_rn(q_reg[2 * i], kf.x, dot);
                dot = __fmaf_rn(q_reg[2 * i + 1], kf.y, dot);
            }
            dot = warp_reduce_sum(dot);
            dot *= scale;
            dot = apply_softcap(dot, softcap);

            float rescale, w_new;
            online_softmax_step(dot, m_w, l_w, rescale, w_new);

            const half2* v_h2 = reinterpret_cast<const half2*>(V_tok + lane_offset);
#pragma unroll
            for (int i = 0; i < ELEMS / 2; i++) {
                half2 vh = v_h2[i];
                float2 vf = __half22float2(vh);
                o_reg[2 * i] = __fmaf_rn(w_new, vf.x, rescale * o_reg[2 * i]);
                o_reg[2 * i + 1] = __fmaf_rn(w_new, vf.y, rescale * o_reg[2 * i + 1]);
            }
        }

        // Stash per-warp (m, l, o) into smem.
        if (lane_id == 0) {
            warp_max[warp_id] = m_w;
            warp_l[warp_id] = l_w;
        }
#pragma unroll
        for (int i = 0; i < ELEMS; i++)
            warp_o[warp_id * HEAD_DIM + lane_offset + i] = o_reg[i];
        __syncthreads();

        if (threadIdx.x == 0) {
            float m_global = -FLT_MAX;
            for (int w = 0; w < NUM_WARPS; w++) m_global = fmaxf(m_global, warp_max[w]);
            float l_global = 0.0f;
            for (int w = 0; w < NUM_WARPS; w++)
                l_global += expf(warp_max[w] - m_global) * warp_l[w];
            s_m_res = m_global;
            s_l_res = l_global;
        }
    } else {
        if (threadIdx.x == 0) {
            s_m_res = -FLT_MAX;
            s_l_res = 0.0f;
        }
    }
    __syncthreads();

    const float m_res = s_m_res;
    const float l_res = s_l_res;

    // Step 3: combined merge (paged + residual) per dim, parallelized across threads.
    float m_global, w_paged, w_res, inv_l;
    if (res_active > 0) {
        m_global = fmaxf(m_paged, m_res);
        w_paged = expf(m_paged - m_global);
        w_res = expf(m_res - m_global);
        float l_global = w_paged * l_paged + w_res * l_res;
        inv_l = (l_global > 0.0f) ? (1.0f / l_global) : 0.0f;
    } else {
        m_global = m_paged;
        w_paged = 1.0f;
        w_res = 0.0f;
        inv_l = (l_paged > 0.0f) ? (1.0f / l_paged) : 0.0f;
    }

    for (int d = threadIdx.x; d < HEAD_DIM; d += blockDim.x) {
        // Aggregate paged partials at m_paged basis: sum_s exp(m_s - m_paged) * partial[s, 2+d]
        float o_paged_unnorm = 0.0f;
        for (int s = 0; s < num_paged_splits; s++) {
            float m_s = paged_base[s * partial_stride];
            float weight_s = expf(m_s - m_paged);
            o_paged_unnorm += weight_s * paged_base[s * partial_stride + 2 + d];
        }

        // Aggregate residual partials at m_res basis: sum_w exp(m_w - m_res) * l_w * warp_o[w, d]
        float o_res_unnorm = 0.0f;
        if (res_active > 0) {
            for (int w = 0; w < NUM_WARPS; w++) {
                float weight_w = expf(warp_max[w] - m_res) * warp_l[w];
                o_res_unnorm += weight_w * warp_o[w * HEAD_DIM + d];
            }
        }

        float final_o = inv_l * (w_paged * o_paged_unnorm + w_res * o_res_unnorm);
        int out_idx = batch_idx * n_heads * HEAD_DIM + head_idx * HEAD_DIM + d;
        O[out_idx] = __float2half(final_o);
    }
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
                                  int max_blocks_per_seq, int n_sinks,
                                  const half* K_residual, const half* V_residual,
                                  int residual_count, int residual_n_tokens, int residual_write_idx,
                                  const half* K_residual_base, const half* V_residual_base,
                                  int residual_seq_stride_elems, const int* d_residual_seq_slots,
                                  const int* d_residual_counts, const int* d_residual_write_idxes) {
    (void)n_sinks;  // streaming not yet wired
    const int batch_size = static_cast<int>(Q.shape[0]);
    const int n_heads = static_cast<int>(Q.shape[2]);
    const int head_dim = static_cast<int>(Q.shape[3]);
    const int n_kv_heads = static_cast<int>(K_cache.shape[2]);

    const int max_num_blocks = (max_blocks_per_seq > 0) ? max_blocks_per_seq
                                                        : (max_context_len + block_size - 1) / block_size;

    // Phase 3b: residual is not wired into the split-K scaffold.  Force
    // non-split path when EITHER form is active. Multi-seq form is selected
    // by d_residual_seq_slots != nullptr; single-seq scalar form requires
    // K_residual && residual_count > 0 && batch_size == 1.
    const bool residual_active_multiseq =
        (d_residual_seq_slots != nullptr) && (K_residual_base != nullptr) &&
        (V_residual_base != nullptr) && (residual_n_tokens > 0);
    const bool residual_active_scalar =
        (K_residual != nullptr) && (V_residual != nullptr) &&
        (residual_count > 0) && (residual_n_tokens > 0) && (batch_size == 1) &&
        !residual_active_multiseq;
    const bool residual_active = residual_active_multiseq || residual_active_scalar;

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

    // Residual path: ALWAYS use splitk + residual_reduce_kernel even if
    // compute_splitk_splits returned 1. Embedding the residual pass into the
    // non-splitk decode kernel was the previous design and forced a 12× slow
    // path (414 µs/call vs splitk at 33 µs/call) per nsys profile (2026-05-09).
    // The splitk paged kernel runs unmodified (writes per-split partials);
    // residual_reduce_kernel folds in the FP16 residual contribution as part
    // of the reduce. Requires scratch_ptr to be allocated — caller (engine
    // workspace) provides this when NVFP4 KV is enabled.
    if (residual_active && scratch_ptr == nullptr) {
        IMP_LOG_WARN("paged_attention_decode_nvfp4_tc: residual active but no splitk "
                     "scratch — falling back to non-splitk path (slow)");
    }
    const bool use_residual_reduce = residual_active && scratch_ptr != nullptr;
    if (use_residual_reduce && num_splits < 1) num_splits = 1;

    if (num_splits > 1 || use_residual_reduce) {
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

        if (use_residual_reduce) {
            // Combined paged-reduce + residual-merge. Smem layout:
            //   warp_max[NUM_WARPS] + warp_l[NUM_WARPS] + warp_o[NUM_WARPS * HEAD_DIM] floats.
            const size_t reduce_smem =
                NUM_WARPS * sizeof(float) + NUM_WARPS * sizeof(float) +
                NUM_WARPS * head_dim * sizeof(float);
            dim3 grid_red(batch_size, n_heads);
            dim3 block_red(BLOCK_THREADS);

#define LAUNCH_RESIDUAL_REDUCE(HD)                                                                            \
    paged_attention_residual_reduce_kernel<HD><<<grid_red, block_red, reduce_smem, stream>>>(                 \
        partial, reinterpret_cast<const half*>(Q.data), reinterpret_cast<half*>(O.data),                      \
        context_lens, n_heads, n_kv_heads, num_splits, scale, sliding_window, softcap,                        \
        residual_n_tokens,                                                                                    \
        residual_active_scalar ? K_residual : nullptr,                                                        \
        residual_active_scalar ? V_residual : nullptr,                                                        \
        residual_active_scalar ? residual_count : 0,                                                          \
        residual_active_scalar ? residual_write_idx : 0,                                                      \
        residual_active_multiseq ? K_residual_base : nullptr,                                                 \
        residual_active_multiseq ? V_residual_base : nullptr,                                                 \
        residual_active_multiseq ? residual_seq_stride_elems : 0,                                             \
        residual_active_multiseq ? d_residual_seq_slots : nullptr,                                            \
        residual_active_multiseq ? d_residual_counts : nullptr,                                               \
        residual_active_multiseq ? d_residual_write_idxes : nullptr)
            switch (head_dim) {
                case 64:  LAUNCH_RESIDUAL_REDUCE(64);  break;
                case 128: LAUNCH_RESIDUAL_REDUCE(128); break;
                case 256: LAUNCH_RESIDUAL_REDUCE(256); break;
                case 512: LAUNCH_RESIDUAL_REDUCE(512); break;
                default:
                    IMP_LOG_ERROR("paged_attention_residual_reduce: unsupported head_dim %d", head_dim);
                    return;
            }
#undef LAUNCH_RESIDUAL_REDUCE
        } else {
            paged_attention_launch_reduce(partial, reinterpret_cast<half*>(O.data), batch_size, n_heads,
                                          head_dim, num_splits, stream);
        }
    } else {
        dim3 grid(batch_size, n_heads);
        dim3 block(BLOCK_THREADS);

#define LAUNCH_NVFP4(HD)                                                                                       \
    paged_attention_decode_nvfp4_tc_kernel<HD><<<grid, block, smem_bytes, stream>>>(                              \
        reinterpret_cast<const half*>(Q.data), reinterpret_cast<const uint8_t*>(K_cache.data),                 \
        reinterpret_cast<const uint8_t*>(V_cache.data), K_scales, V_scales, reinterpret_cast<half*>(O.data),   \
        block_tables, context_lens, batch_size, n_heads, n_kv_heads, block_size, scale, max_context_len,       \
        max_num_blocks, sliding_window, softcap,                                                               \
        residual_active_scalar ? K_residual : nullptr,                                                         \
        residual_active_scalar ? V_residual : nullptr,                                                         \
        residual_active_scalar ? residual_count : 0,                                                           \
        residual_active ? residual_n_tokens : 0,                                                               \
        residual_active_scalar ? residual_write_idx : 0,                                                       \
        residual_active_multiseq ? K_residual_base : nullptr,                                                  \
        residual_active_multiseq ? V_residual_base : nullptr,                                                  \
        residual_active_multiseq ? residual_seq_stride_elems : 0,                                              \
        residual_active_multiseq ? d_residual_seq_slots : nullptr,                                             \
        residual_active_multiseq ? d_residual_counts : nullptr,                                                \
        residual_active_multiseq ? d_residual_write_idxes : nullptr)

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
