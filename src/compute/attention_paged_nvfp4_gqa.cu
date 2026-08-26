// GQA-aware NVFP4 paged decode attention.
//
// The per-Q-head NVFP4 decode kernel (attention_paged_nvfp4.cu) launches one
// block per (seq, Q head): with GQA every Q head of a group re-reads AND
// re-dequantizes the same KV bytes — 6x on Qwen3.8's 24Q/4KV geometry, which
// put the kernel at ~8% of its per-launch DRAM floor while being 8.9% of the
// 32-stream serving profile (2026-08-26). The FP16 twin has had the fix since
// its GQA kernel and FP8 has the tile-GQA sibling; NVFP4 was the one
// quantized dtype without the variant.
//
// Structure follows paged_attention_gqa_kernel (attention_paged.cu) verbatim
// where the dtype allows: one block per (seq, kv_head); the KV block is
// dequantized ONCE into a double-buffered shared FP16 tile (scales folded in
// during the load, so the compute loop is scale-free); every Q head of the
// group computes from that tile with `warps_per_q` warps and its own online
// softmax; a cross-warp shared-memory reduction combines the per-warp
// partials per Q head. Learned sinks join max + denominator in the reduction
// exactly as in the FP16 kernel (#547). StreamingLLM sinks are not wired,
// matching the per-Q-head NVFP4 kernel ((void)n_sinks there).
//
// Non-split-K regime only by construction: the launcher dispatches here only
// when compute_splitk_splits() said 1 — which is every batched-serving shape
// (batch x n_heads >= 2 x SMs), exactly where the 6x re-read costs the most.
// Split shapes keep the per-Q-head split-K path unchanged.

#include "compute/attention_paged.h"
#include "compute/attention_paged_common.cuh"
#include "core/logging.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <float.h>

namespace imp {

namespace {

// Max Q heads per KV head served (same bound as the FP16 GQA kernel).
constexpr int kMaxQPerKv = 16;

// HW pair conversion: 1 packed FP4 byte (E2M1 x2) -> f16x2. Same PTX as the
// per-Q-head kernel (attention_paged_nvfp4.cu).
__device__ __forceinline__ half2 gqa_fp4_byte_to_half2(uint32_t byte_val) {
    uint32_t fp16x2;
    asm("{ .reg .b8 t; cvt.u8.u32 t, %1; cvt.rn.f16x2.e2m1x2 %0, t; }" : "=r"(fp16x2) : "r"(byte_val));
    return *reinterpret_cast<half2*>(&fp16x2);
}

__device__ __forceinline__ float gqa_ue4m3_decode(uint8_t bits) {
    __nv_fp8_e4m3 v;
    memcpy(&v, &bits, 1);
    return static_cast<float>(v);
}

// Dequantize one KV block's slice for `kv_head` into a shared FP16 tile,
// scales folded in. Byte-linear thread mapping: consecutive threads read
// consecutive packed bytes (coalesced), each byte expands to two halfs.
//   k_src/v_src: first token slot of this block, at this kv_head's column.
//   k_sc/v_sc:   matching scale bytes ([slot, group] with the strides below).
__device__ __forceinline__ void gqa_dequant_tile(half* s_k, half* s_v, const uint8_t* k_src,
                                                 const uint8_t* v_src, const uint8_t* k_sc,
                                                 const uint8_t* v_sc, int n_slots, int head_dim,
                                                 int kv_slot_bytes, int sc_slot_stride) {
    const int row_bytes = head_dim / 2;
    const int total_bytes = n_slots * row_bytes;
    for (int idx = threadIdx.x; idx < total_bytes; idx += blockDim.x) {
        const int slot = idx / row_bytes;
        const int pair = idx % row_bytes;          // byte within the row
        const int group = (pair * 2) / 16;         // 16-elem scale group
        const uint8_t kb = k_src[slot * kv_slot_bytes + pair];
        const uint8_t vb = v_src[slot * kv_slot_bytes + pair];
        const float ks = gqa_ue4m3_decode(k_sc[slot * sc_slot_stride + group]);
        const float vs = gqa_ue4m3_decode(v_sc[slot * sc_slot_stride + group]);
        half2 kh = __hmul2(gqa_fp4_byte_to_half2(kb), __float2half2_rn(ks));
        half2 vh = __hmul2(gqa_fp4_byte_to_half2(vb), __float2half2_rn(vs));
        const int out = slot * head_dim + pair * 2;
        s_k[out] = kh.x;
        s_k[out + 1] = kh.y;
        s_v[out] = vh.x;
        s_v[out + 1] = vh.y;
    }
}

__global__ void __launch_bounds__(1024) paged_attention_gqa_nvfp4_kernel(
    const half* __restrict__ Q, const uint8_t* __restrict__ K_cache, const uint8_t* __restrict__ V_cache,
    const uint8_t* __restrict__ K_scales, const uint8_t* __restrict__ V_scales, half* __restrict__ O,
    const int* __restrict__ block_tables, const int* __restrict__ context_lens, int batch_size, int n_heads,
    int n_kv_heads, int head_dim, int block_size, float scale, int max_context_len, int max_num_blocks,
    int n_q_per_kv, int warps_per_q, int sliding_window, float softcap,
    const half* __restrict__ attn_sinks) {
    const int batch_idx = blockIdx.x;
    const int kv_head = blockIdx.y;

    const int ctx_len = context_lens[batch_idx];
    if (ctx_len <= 0)
        return;

    const int total_warps = blockDim.x / WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    const int q_local = warp_id / warps_per_q;
    const int warp_in_q = warp_id % warps_per_q;
    const int head_idx = kv_head * n_q_per_kv + q_local;
    const bool active = (q_local < n_q_per_kv);

    const int elems_per_thread = (head_dim + WARP_SIZE - 1) / WARP_SIZE;

    // Q into registers (max head_dim 512 -> 16 elems/lane).
    float q_reg[16];
    if (active) {
        const half* Q_ptr = Q + (int64_t)batch_idx * n_heads * head_dim + (int64_t)head_idx * head_dim;
        for (int i = 0; i < elems_per_thread; i++) {
            int d = lane_id + i * WARP_SIZE;
            q_reg[i] = (d < head_dim) ? __half2float(Q_ptr[d]) : 0.0f;
        }
    } else {
        for (int i = 0; i < elems_per_thread; i++)
            q_reg[i] = 0.0f;
    }

    const int* bt = block_tables + (int64_t)batch_idx * max_num_blocks;
    const int kv_head_bytes = head_dim / 2;
    const int kv_slot_bytes = n_kv_heads * kv_head_bytes;             // packed bytes per token slot
    const int64_t kv_block_stride = (int64_t)block_size * kv_slot_bytes;
    const int sc_groups = head_dim / 16;
    const int sc_slot_stride = n_kv_heads * sc_groups;
    const int64_t sc_block_stride = (int64_t)block_size * sc_slot_stride;

    float m_w = -FLT_MAX;
    float l_w = 0.0f;
    float o_reg[16];
    for (int i = 0; i < elems_per_thread; i++)
        o_reg[i] = 0.0f;

    // Double-buffered dequantized KV tile: [buf][K|V], each block_size*head_dim halfs.
    extern __shared__ __align__(32) char smem_gqa_nvfp4[];
    half* s_kv_h = reinterpret_cast<half*>(smem_gqa_nvfp4);
    const int tile_elems = block_size * head_dim;

    // Sliding window only (StreamingLLM sinks not wired on the NVFP4 path,
    // same as the per-Q-head kernel).
    const ContextRange range = compute_context_range(ctx_len, block_size, sliding_window, /*n_sinks=*/0);
    const int first_block = range.first_block;
    const int num_ctx_blocks = range.num_ctx_blocks;

    int buf = 0;
    if (first_block < num_ctx_blocks) {
        const int phys = bt[first_block];
        if (phys >= 0) {
            gqa_dequant_tile(s_kv_h, s_kv_h + tile_elems,
                             K_cache + phys * kv_block_stride + kv_head * kv_head_bytes,
                             V_cache + phys * kv_block_stride + kv_head * kv_head_bytes,
                             K_scales + phys * sc_block_stride + kv_head * sc_groups,
                             V_scales + phys * sc_block_stride + kv_head * sc_groups, block_size, head_dim,
                             kv_slot_bytes, sc_slot_stride);
        }
    }
    __syncthreads();

    for (int blk = first_block; blk < num_ctx_blocks; blk = next_valid_block(range, blk)) {
        const half* s_k_cur = s_kv_h + buf * 2 * tile_elems;
        const half* s_v_cur = s_k_cur + tile_elems;
        const bool cur_valid = bt[blk] >= 0;  // -1 = StreamingLLM eviction hole (#1678 class)

        const int next_buf = 1 - buf;
        const int next_blk = next_valid_block(range, blk);
        if (next_blk < num_ctx_blocks) {
            const int next_phys = bt[next_blk];
            if (next_phys >= 0) {
                half* s_k_next = s_kv_h + next_buf * 2 * tile_elems;
                gqa_dequant_tile(s_k_next, s_k_next + tile_elems,
                                 K_cache + next_phys * kv_block_stride + kv_head * kv_head_bytes,
                                 V_cache + next_phys * kv_block_stride + kv_head * kv_head_bytes,
                                 K_scales + next_phys * sc_block_stride + kv_head * sc_groups,
                                 V_scales + next_phys * sc_block_stride + kv_head * sc_groups, block_size,
                                 head_dim, kv_slot_bytes, sc_slot_stride);
            }
        }

        int first_tok = 0, last_tok = 0;
        const bool have_toks = block_token_range(range, blk, block_size, ctx_len, first_tok, last_tok);

        if (active && have_toks && cur_valid) {
            for (int ti = warp_in_q + first_tok; ti < last_tok; ti += warps_per_q) {
                float dot = 0.0f;
                for (int i = 0; i < elems_per_thread; i++) {
                    int d = lane_id + i * WARP_SIZE;
                    if (d < head_dim)
                        dot += q_reg[i] * __half2float(s_k_cur[ti * head_dim + d]);
                }
                dot = warp_reduce_sum(dot);
                dot *= scale;
                dot = apply_softcap(dot, softcap);

                float rescale, w_new;
                online_softmax_step(dot, m_w, l_w, rescale, w_new);

                for (int i = 0; i < elems_per_thread; i++) {
                    int d = lane_id + i * WARP_SIZE;
                    float v_val = (d < head_dim) ? __half2float(s_v_cur[ti * head_dim + d]) : 0.0f;
                    o_reg[i] = rescale * o_reg[i] + w_new * v_val;
                }
            }
        }

        __syncthreads();  // both the next-tile dequant and this compute must finish
        buf = next_buf;
    }

    if (!active)
        return;

    // Cross-warp reduction per Q head (same layout as the FP16 GQA kernel;
    // reuses the tile smem, safe after the final __syncthreads above).
    float* red_max = reinterpret_cast<float*>(smem_gqa_nvfp4);
    float* red_l = red_max + total_warps;
    float* red_o = red_l + total_warps;

    if (lane_id == 0) {
        red_max[warp_id] = m_w;
        red_l[warp_id] = l_w;
    }
    for (int i = 0; i < elems_per_thread; i++) {
        int d = lane_id + i * WARP_SIZE;
        if (d < head_dim)
            red_o[warp_id * head_dim + d] = o_reg[i];
    }
    __syncthreads();

    if (warp_in_q == 0) {
        const int base_w = q_local * warps_per_q;

        float global_max = -FLT_MAX;
        for (int w = 0; w < warps_per_q; w++)
            global_max = fmaxf(global_max, red_max[base_w + w]);
        if (attn_sinks)
            global_max = fmaxf(global_max, __half2float(attn_sinks[head_idx]));

        float global_l = 0.0f;
        for (int w = 0; w < warps_per_q; w++)
            global_l += expf(red_max[base_w + w] - global_max) * red_l[base_w + w];
        if (attn_sinks)
            global_l += expf(__half2float(attn_sinks[head_idx]) - global_max);

        for (int i = 0; i < elems_per_thread; i++) {
            int d = lane_id + i * WARP_SIZE;
            if (d < head_dim) {
                float o_val = 0.0f;
                for (int w = 0; w < warps_per_q; w++) {
                    float weight = expf(red_max[base_w + w] - global_max) * red_l[base_w + w];
                    o_val += weight * red_o[(base_w + w) * head_dim + d];
                }
                if (global_l > 0.0f)
                    o_val /= global_l;
                int out_idx = batch_idx * n_heads * head_dim + head_idx * head_dim + d;
                stcs_half(&O[out_idx], __float2half(o_val));
            }
        }
    }
}

}  // namespace

bool paged_attention_gqa_nvfp4_supported(int head_dim, int n_heads, int n_kv_heads) {
    if (head_dim > 512 || (head_dim % WARP_SIZE) != 0 || (head_dim % 16) != 0)
        return false;
    if (n_kv_heads <= 0 || (n_heads % n_kv_heads) != 0)
        return false;
    const int n_q_per_kv = n_heads / n_kv_heads;
    return n_q_per_kv >= 2 && n_q_per_kv <= kMaxQPerKv;
}

bool paged_attention_gqa_nvfp4_launch(const half* Q, const uint8_t* K_cache, const uint8_t* V_cache,
                                      const uint8_t* K_scales, const uint8_t* V_scales, half* O,
                                      const int* block_tables, const int* context_lens, int batch_size,
                                      int n_heads, int n_kv_heads, int head_dim, int block_size, float scale,
                                      int max_context_len, int max_num_blocks, int sliding_window,
                                      float softcap, const half* attn_sinks, cudaStream_t stream) {
    const int n_q_per_kv = n_heads / n_kv_heads;
    const int warps_per_q = (n_q_per_kv <= 8) ? 4 : 2;
    const int total_warps = n_q_per_kv * warps_per_q;
    const int threads = total_warps * WARP_SIZE;

    const size_t kv_tile_bytes = 4u * block_size * head_dim * sizeof(half);
    const size_t red_bytes =
        total_warps * sizeof(float) * 2 + (size_t)total_warps * head_dim * sizeof(float);
    const size_t smem = (kv_tile_bytes > red_bytes) ? kv_tile_bytes : red_bytes;

    // Opt-in above the 48 KiB static cap, re-armed for the largest value seen
    // (same rationale as the FP16 GQA launcher: one non-templated function
    // shared across models — a one-shot guard would freeze the first cap).
    if (smem > 48 * 1024) {
        static size_t s_optin = 0;
        if (smem > s_optin) {
            cudaError_t fa = cudaFuncSetAttribute(paged_attention_gqa_nvfp4_kernel,
                                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                  static_cast<int>(smem));
            if (fa == cudaSuccess) {
                s_optin = smem;
            } else {
                IMP_LOG_WARN("paged_attention_gqa_nvfp4: smem opt-in (%zu B) failed: %s", smem,
                             cudaGetErrorString(fa));
                (void)cudaGetLastError();
                return false;  // caller falls back to the per-Q-head kernel
            }
        }
    }

    dim3 grid(batch_size, n_kv_heads);
    dim3 block(threads);
    paged_attention_gqa_nvfp4_kernel<<<grid, block, smem, stream>>>(
        Q, K_cache, V_cache, K_scales, V_scales, O, block_tables, context_lens, batch_size, n_heads,
        n_kv_heads, head_dim, block_size, scale, max_context_len, max_num_blocks, n_q_per_kv, warps_per_q,
        sliding_window, softcap, attn_sinks);
    IMP_CUDA_CHECK_LAUNCH();
    return true;
}

}  // namespace imp
