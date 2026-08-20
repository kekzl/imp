// =============================================================================
// mtp_forward.cu — Multi-Token-Predictor draft step (Phase 2 scaffolding)
// =============================================================================
// See header for status. This TU implements the REDUCED forward path:
//   emb_norm  = RMSNorm(tok_emb[prev_token_id], pre_fc_norm_embedding)
//   h_norm    = RMSNorm(d_h_prev,               pre_fc_norm_hidden)
//   fc_in     = concat(emb_norm, h_norm)         // [2*hidden_dim]
//   fc_out    = fc @ fc_in                       // [hidden_dim]
//   // TRANSFORMER BLOCK SKIPPED (Phase 2.2 future work)
//   h_final   = RMSNorm(fc_out, final_norm)
//   logits    = lm_head @ h_final                // [vocab]
//   token     = argmax(logits)
// =============================================================================

#include "compute/warp_reduce.cuh"
#include "compute/mtp_forward.h"
#include "compute/activation.h"     // swiglu, shared_expert_gate_scale
#include "compute/embedding.h"      // embedding_lookup (handles quantized tables)
#include "compute/gemm.h"
#include "quant/nvfp4_gemm.h"       // gemv_nvfp4_kpar_fp32 (NVFP4 chain lm_head)
#include "compute/layernorm.h"
#include "compute/moe_routing.h"
// relu_sqr_inplace — activation for the non-gated (Nemotron) MTP experts.
#include "compute/ssm.h"
#include "compute/rope.h"           // qknorm_rope_fused
#include "compute/rope_yarn.cuh"    // rope_yarn (shared YaRN device math)
#include "core/logging.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdlib>
#include <cstdio>
#include <cstring>

namespace imp {

// ---------------------------------------------------------------------------
// Tiny kernels
// ---------------------------------------------------------------------------

// Concatenate two [hidden_dim] FP16 vectors into [2*hidden_dim].
// out[0..hidden_dim-1]   = a
// out[hidden_dim..2hd-1] = b
__global__ void mtp_concat_kernel(const __half* __restrict__ a, const __half* __restrict__ b,
                                   __half* __restrict__ out, int hidden_dim) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= 2 * hidden_dim) return;
    out[t] = (t < hidden_dim) ? a[t] : b[t - hidden_dim];
}

__device__ __forceinline__ float mtp_logit_to_float(__half v) { return __half2float(v); }
__device__ __forceinline__ float mtp_logit_to_float(float v)  { return v; }

// Argmax over an FP16/FP32 vector [vocab_size]. Single CTA; uses shared-memory
// reduction. Writes the argmax index to *out_idx as int32. Caller must ensure
// vocab_size fits one block (i.e., we strip-mine over vocab_size).
template <typename T>
__global__ void mtp_argmax_kernel(const T* __restrict__ logits, int vocab_size,
                                   int* __restrict__ out_idx) {
    constexpr int kThreads = 256;
    __shared__ float s_val[kThreads];
    __shared__ int   s_idx[kThreads];

    int tid = threadIdx.x;
    float best_val = -1.0e38f;
    int   best_idx = 0;
    for (int i = tid; i < vocab_size; i += kThreads) {
        float v = mtp_logit_to_float(logits[i]);
        if (v > best_val) {
            best_val = v;
            best_idx = i;
        }
    }
    s_val[tid] = best_val;
    s_idx[tid] = best_idx;
    __syncthreads();
    for (int off = kThreads / 2; off > 0; off >>= 1) {
        if (tid < off) {
            if (s_val[tid + off] > s_val[tid]) {
                s_val[tid] = s_val[tid + off];
                s_idx[tid] = s_idx[tid + off];
            }
        }
        __syncthreads();
    }
    if (tid == 0) *out_idx = s_idx[0];
}

// Top-W over an FP16/FP32 vector [vocab_size]. Single CTA; top_w (≤ kMtpMaxTopW)
// sequential argmax passes, masking previously selected indices. Writes the W
// descending-logit indices to out_idx[0..top_w). W is tiny relative to the
// lm_head GEMM that produced the logits, so the extra passes are cheap.
template <typename T>
__global__ void mtp_topk_kernel(const T* __restrict__ logits, int vocab_size,
                                int top_w, int* __restrict__ out_idx) {
    constexpr int kThreads = 256;
    __shared__ float s_val[kThreads];
    __shared__ int   s_idx[kThreads];
    __shared__ int   s_found[kMtpMaxTopW];

    int tid = threadIdx.x;
    for (int w = 0; w < top_w; ++w) {
        float best_val = -1.0e38f;
        int   best_idx = 0;
        for (int i = tid; i < vocab_size; i += kThreads) {
            bool taken = false;
            for (int f = 0; f < w; ++f) {
                if (s_found[f] == i) { taken = true; break; }
            }
            if (taken) continue;
            float v = mtp_logit_to_float(logits[i]);
            if (v > best_val) { best_val = v; best_idx = i; }
        }
        s_val[tid] = best_val;
        s_idx[tid] = best_idx;
        __syncthreads();
        for (int off = kThreads / 2; off > 0; off >>= 1) {
            if (tid < off) {
                if (s_val[tid + off] > s_val[tid]) {
                    s_val[tid] = s_val[tid + off];
                    s_idx[tid] = s_idx[tid + off];
                }
            }
            __syncthreads();
        }
        if (tid == 0) {
            s_found[w]  = s_idx[0];
            out_idx[w]  = s_idx[0];
        }
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// MoE residual + shared-expert combine kernel
// ---------------------------------------------------------------------------
// fc_out[i] += moe_out[i] + shared_out[i]   for i in [0, hidden_dim)
// moe_out already contains the residual-added MoE output (residual was fed
// through moe_weighted_sum_residual). We need to add shared_out which has
// already been scaled by the sigmoid gate (via shared_expert_gate_scale).
__global__ void mtp_add_shared_kernel(__half* __restrict__ fc_out,
                                       const __half* __restrict__ shared_out,
                                       int hidden_dim) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= hidden_dim) return;
    float v = __half2float(fc_out[t]) + __half2float(shared_out[t]);
    fc_out[t] = __float2half(v);
}

// ---------------------------------------------------------------------------
// Gated attention output kernel (Phase 2.2.Attn MVP, M=1, no KV history)
// ---------------------------------------------------------------------------
// For Qwen3.6 MTP's attn_output_gate=True attention:
//   q_proj outputs [num_heads * 2 * head_dim], interleaved per-head as
//   [head_0_q (head_dim), head_0_gate (head_dim), head_1_q (...), ...].
//   The "q" half feeds Q@K dot-product attention; the "gate" half is
//   silu'd and elementwise multiplied with the attention output.
//
// For M=1 with no MTP KV history, the attention softmax over a single
// token is identically 1, so attention_out_per_head = V (broadcast from
// num_kv_heads to num_heads via GQA). Final per-head output is
// silu(gate) * V_broadcast.
//
// This kernel computes: out[h, d] = silu(gate[h, d]) * v[h / group_size, d]
// where group_size = num_heads / num_kv_heads.
//
// q_full layout: [num_heads, 2, head_dim] interpreted as
//   q_full[h, 0, d] = q[h, d]      (first head_dim per head)
//   q_full[h, 1, d] = gate[h, d]   (second head_dim per head)
__global__ void mtp_gated_v_broadcast_kernel(
    const __half* __restrict__ q_full,    // [num_heads, 2 * head_dim]
    const __half* __restrict__ v,         // [num_kv_heads, head_dim]
    __half* __restrict__ out,             // [num_heads * head_dim]
    int num_heads, int num_kv_heads, int head_dim) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_heads * head_dim;
    if (t >= total) return;
    int h = t / head_dim;
    int d = t % head_dim;
    int kv_h = h * num_kv_heads / num_heads;  // GQA broadcast

    // gate is the SECOND head_dim slice of q_full[h]
    int gate_idx = h * (2 * head_dim) + head_dim + d;
    float g = __half2float(q_full[gate_idx]);
    // silu(g) = g * sigmoid(g)
    // Qwen3-Next attn_output_gate: attn_out *= sigmoid(gate) (NOT silu).
    // Ref: vLLM Qwen3NextAttention.forward — attn_output * torch.sigmoid(gate).
    float gate_act = 1.0f / (1.0f + expf(-g));

    float v_val = __half2float(v[kv_h * head_dim + d]);
    out[t] = __float2half(gate_act * v_val);
}

// Elementwise add: fc_out += attn_residual
__global__ void mtp_add_kernel(__half* __restrict__ fc_out,
                                const __half* __restrict__ attn_residual,
                                int hidden_dim) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= hidden_dim) return;
    float v = __half2float(fc_out[t]) + __half2float(attn_residual[t]);
    fc_out[t] = __float2half(v);
}

// ---------------------------------------------------------------------------
// MTP KV-cache append + softmax attention scan (Phase 2.2.Attn+KV)
// ---------------------------------------------------------------------------
// Append k[h], v[h] (one per kv-head) to the cache at position `pos`, then
// run softmax attention over positions [0, pos+1). One CTA per Q head. Q
// attends to its corresponding KV head (GQA: q_head h → kv_head h * NKV/NH).
//
// Q layout: q_full[h, 0..head_dim) is the "q" half (first head_dim of each
//           head's 2*head_dim slice). The "gate" half is q_full[h, head_dim..)
//           and is applied AFTER the attention via silu(gate)*attn_out.
// K cache layout: [seq_len, num_kv_heads, head_dim] row-major.
// V cache layout: same.
//
// For decode (M=1): threads in a CTA cooperatively compute Q·K dot products
// for all cached positions, do a numerically-stable softmax, then a weighted
// sum of V. seq_len up to a few thousand fits in shared mem with FP32 scores.
//
// NOTE: this version does NOT apply RoPE. Without RoPE, attention scores
// reflect only the CONTENT similarity between query and past keys — still
// useful for drafting (the content has positional information baked in via
// the upstream main-model hidden states) but theoretically less precise.
// RoPE is documented as a follow-on improvement.
__global__ void mtp_attn_kv_scan_kernel(
    const __half* __restrict__ q_attn,   // [num_heads, head_dim] — Q with qk-norm + RoPE applied
    const __half* __restrict__ k_cache,  // [seq_len_cap, num_kv_heads, head_dim] — RoPE pre-applied
    const __half* __restrict__ v_cache,  // [seq_len_cap, num_kv_heads, head_dim]
    __half* __restrict__ out,            // [num_heads, head_dim]
    int seq_len, int num_heads, int num_kv_heads, int head_dim,
    int max_seq_len, float scale) {
    int h = blockIdx.x;
    if (h >= num_heads) return;
    int tid = threadIdx.x;
    int gqa = num_heads / num_kv_heads;
    int kv_h = h / gqa;

    extern __shared__ float s_scores[];  // sized: seq_len * sizeof(float)

    // Q row for this head: contiguous [head_dim] from the rotated Q buffer.
    const __half* q_row = q_attn + h * head_dim;

    // ---- (1) Q · K for each cached t, accumulate into shared mem ----
    // One thread per t (with strip-mining if seq_len > blockDim.x).
    float max_score = -1.0e30f;
    for (int t = tid; t < seq_len; t += blockDim.x) {
        const __half* k_row = k_cache + (static_cast<int64_t>(t) * num_kv_heads + kv_h) * head_dim;
        float acc = 0.0f;
        // Inner dot product along head_dim — let one thread do the whole thing
        // (head_dim=256 is small enough for serial accumulation per-t).
        for (int d = 0; d < head_dim; ++d) {
            acc += __half2float(q_row[d]) * __half2float(k_row[d]);
        }
        float scaled = acc * scale;
        s_scores[t] = scaled;
        if (scaled > max_score) max_score = scaled;
    }

    // Reduce max across block via shared mem (small block: kBlock = 256 typical).
    __shared__ float s_block_max;
    if (tid == 0) s_block_max = -1.0e30f;
    __syncthreads();
    atomic_max_float(&s_block_max, max_score);
    __syncthreads();
    float gmax = s_block_max;

    // ---- (2) Numerically-stable softmax denominator ----
    __shared__ float s_block_sum;
    if (tid == 0) s_block_sum = 0.0f;
    __syncthreads();
    float local_sum = 0.0f;
    for (int t = tid; t < seq_len; t += blockDim.x) {
        float e = expf(s_scores[t] - gmax);
        s_scores[t] = e;
        local_sum += e;
    }
    atomicAdd(&s_block_sum, local_sum);
    __syncthreads();
    float denom = s_block_sum;
    float inv_denom = (denom > 0.0f) ? (1.0f / denom) : 0.0f;

    // ---- (3) Weighted sum: out[h, d] = Σ_t (e_t / denom) * V[t, kv_h, d] ----
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int t = 0; t < seq_len; ++t) {
            const __half* v_row = v_cache + (static_cast<int64_t>(t) * num_kv_heads + kv_h) * head_dim;
            acc += s_scores[t] * inv_denom * __half2float(v_row[d]);
        }
        out[h * head_dim + d] = __float2half(acc);
    }
}

// Apply silu(gate) elementwise to attn_out in-place. gate is the second
// head_dim slice of q_full[h]. Used after the attention scan.
__global__ void mtp_gate_attn_out_kernel(
    __half* __restrict__ attn_out,            // [num_heads, head_dim] in/out
    const __half* __restrict__ q_full,        // [num_heads, 2*head_dim]
    int num_heads, int head_dim) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_heads * head_dim;
    if (t >= total) return;
    int h = t / head_dim;
    int d = t % head_dim;
    int gate_idx = h * (2 * head_dim) + head_dim + d;
    float g = __half2float(q_full[gate_idx]);
    // Qwen3-Next attn_output_gate: attn_out *= sigmoid(gate) (NOT silu).
    // Ref: vLLM Qwen3NextAttention.forward — attn_output * torch.sigmoid(gate).
    float gate_act = 1.0f / (1.0f + expf(-g));
    float v = __half2float(attn_out[t]);
    attn_out[t] = __float2half(gate_act * v);
}

// Append k_row (one step's k_proj output, shape [num_kv_heads, head_dim])
// into k_cache[pos, :, :]. Same for V.
__global__ void mtp_kv_append_kernel(
    const __half* __restrict__ k_step,   // [num_kv_heads * head_dim]
    const __half* __restrict__ v_step,
    __half* __restrict__ k_cache,        // [max_seq, num_kv_heads, head_dim]
    __half* __restrict__ v_cache,
    int pos, int num_kv_heads, int head_dim) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_kv_heads * head_dim;
    if (t >= total) return;
    int64_t off = (static_cast<int64_t>(pos) * num_kv_heads * head_dim) + t;
    k_cache[off] = k_step[t];
    v_cache[off] = v_step[t];
}

// ---------------------------------------------------------------------------
// MTP mrope (Multi-RoPE) — Qwen3-VL-style RoPE with section split
// ---------------------------------------------------------------------------
// For Qwen3.6 mrope_section = [11, 11, 10] (half-counts) means the rope_dim/2
// frequency pairs are split:
//   pair k ∈ [0, 11):  section 0 (T, temporal)  → uses positions[0]
//   pair k ∈ [11, 22): section 1 (H, height)    → uses positions[1]
//   pair k ∈ [22, 32): section 2 (W, width)     → uses positions[2]
//
// For text-only tokens positions[0]=positions[1]=positions[2]=mtp_pos,
// so mrope mathematically reduces to standard partial-rope. The kernel
// is written generically to support multimodal positions in the future.
//
// NeoX style: pair k rotates (x[k], x[k+rope_dim/2]).
// Frequency: inv_freq[k] = theta^(-2k/rope_dim), shared across sections.
// Rotation: (x0, x1) → (x0*cos - x1*sin, x0*sin + x1*cos)
//
// One CTA per head. Threads handle pairs in strided fashion. Untouched dims
// ([rope_dim, head_dim)) are unchanged.
template <bool IsKv>
__global__ void mtp_mrope_kernel(
    __half* __restrict__ x,           // Q: [n_heads, head_dim], K: [n_kv_heads, head_dim]
    int n_heads, int head_dim, int rope_dim, float theta,
    int sec0, int sec1, int sec2,    // mrope_section half-counts (sec0+sec1+sec2 == rope_dim/2)
    int pos_t, int pos_h, int pos_w,
    // RoPE scaling — mirrors rope.cu's rope_forward_kernel so the draft head
    // rotates identically to the verifier (issue #897). inv_scaling = 1/freq_scale;
    // ext_factor > 0 engages YaRN blending (corr_dim_0/1, attn_factor=mscale).
    float inv_scaling, float ext_factor, float attn_factor, float corr_dim_0, float corr_dim_1) {
    int h = blockIdx.x;
    if (h >= n_heads) return;
    __half* row = x + static_cast<int64_t>(h) * head_dim;
    int pairs = rope_dim / 2;
    int s01 = sec0;
    int s12 = sec0 + sec1;
    for (int k = threadIdx.x; k < pairs; k += blockDim.x) {
        // Determine which section this pair belongs to.
        int pos;
        if      (k < s01) pos = pos_t;
        else if (k < s12) pos = pos_h;
        else              pos = pos_w;
        // cos/sin with the same YaRN / linear-scaling math as the main forward.
        float c, s;
        if (ext_factor != 0.0f) {
            // YaRN mode: per-dimension frequency blending.
            float theta_extrap =
                static_cast<float>(pos) / powf(theta, static_cast<float>(2 * k) / static_cast<float>(rope_dim));
            rope_yarn(theta_extrap, inv_scaling, corr_dim_0, corr_dim_1, 2 * k, ext_factor, attn_factor, c, s);
        } else {
            // Linear mode: frequency = theta^(-2k/rope_dim), scaled by inv_scaling.
            float freq = inv_scaling / powf(theta, static_cast<float>(2 * k) / static_cast<float>(rope_dim));
            float angle = static_cast<float>(pos) * freq;
            c = __cosf(angle);
            s = __sinf(angle);
        }
        // NeoX-style pair: (x[k], x[k + rope_dim/2])
        int i0 = k;
        int i1 = k + pairs;
        float x0 = __half2float(row[i0]);
        float x1 = __half2float(row[i1]);
        row[i0] = __float2half(x0 * c - x1 * s);
        row[i1] = __float2half(x0 * s + x1 * c);
    }
    // (void) IsKv — currently unused but reserved for any future per-head
    // GQA/MQA divergences. Both Q and K paths share the same rotation math.
    if (false) (void)IsKv;
}

// Host wrapper: apply the YaRN-aware mrope rotation to a single MTP step's
// Q [n_heads, head_dim] and K [n_kv_heads, head_dim] in place. Text-only, so
// the three mrope position components collapse to `pos`. Shared by the draft
// step and the rope-parity unit test (issue #897).
void mtp_apply_mrope(void* d_q, int n_heads, void* d_k, int n_kv_heads, int head_dim, int rope_dim,
                     float theta, int sec0, int sec1, int sec2, int pos, float inv_scaling,
                     float ext_factor, float attn_factor, float corr_dim_0, float corr_dim_1,
                     cudaStream_t stream) {
    if (rope_dim <= 0 || sec0 + sec1 + sec2 != rope_dim / 2) return;
    const int kBlock = 128;
    if (n_heads > 0 && d_q) {
        mtp_mrope_kernel<false><<<n_heads, kBlock, 0, stream>>>(
            static_cast<__half*>(d_q), n_heads, head_dim, rope_dim, theta, sec0, sec1, sec2,
            pos, pos, pos, inv_scaling, ext_factor, attn_factor, corr_dim_0, corr_dim_1);
        IMP_CUDA_CHECK_LAUNCH();
    }
    if (n_kv_heads > 0 && d_k) {
        mtp_mrope_kernel<true><<<n_kv_heads, kBlock, 0, stream>>>(
            static_cast<__half*>(d_k), n_kv_heads, head_dim, rope_dim, theta, sec0, sec1, sec2,
            pos, pos, pos, inv_scaling, ext_factor, attn_factor, corr_dim_0, corr_dim_1);
        IMP_CUDA_CHECK_LAUNCH();
    }
}

// ---------------------------------------------------------------------------
// Workspace alloc/free
// ---------------------------------------------------------------------------
bool mtp_workspace_allocate(MtpDraftWorkspace& ws, int hidden_dim, int vocab_size,
                            int n_experts, int top_k, int expert_d_ff, int shared_d_ff,
                            int num_heads, int num_kv_heads, int head_dim,
                            int max_seq_len) {
    if (hidden_dim <= 0 || vocab_size <= 0) return false;
    auto alloc = [](void** p, size_t bytes) {
        return cudaMalloc(p, bytes) == cudaSuccess;
    };
    bool ok = true;
    // Phase 2.1 buffers (always allocated)
    ok &= alloc(&ws.d_emb_norm,   hidden_dim * sizeof(__half));
    ok &= alloc(&ws.d_h_norm,     hidden_dim * sizeof(__half));
    ok &= alloc(&ws.d_fc_in,      2 * hidden_dim * sizeof(__half));
    ok &= alloc(&ws.d_fc_out,     hidden_dim * sizeof(__half));
    ok &= alloc(&ws.d_h_final,    hidden_dim * sizeof(__half));
    ok &= alloc(&ws.d_logits,     vocab_size * sizeof(__half));
    ok &= alloc(&ws.d_logits_f32, vocab_size * sizeof(float));
    ok &= alloc(reinterpret_cast<void**>(&ws.d_topk), kMtpMaxTopW * sizeof(int));
    ok &= alloc(reinterpret_cast<void**>(&ws.d_chain_tokens), kMtpMaxChainK * sizeof(int32_t));
    ok &= alloc(reinterpret_cast<void**>(&ws.d_argmax), sizeof(int));
    ok &= alloc(reinterpret_cast<void**>(&ws.d_tok), sizeof(int32_t));

    ws.hidden_dim   = hidden_dim;
    ws.n_experts    = n_experts;
    ws.top_k        = top_k;
    ws.expert_d_ff  = expert_d_ff;
    ws.shared_d_ff  = shared_d_ff;
    ws.num_heads    = num_heads;
    ws.num_kv_heads = num_kv_heads;
    ws.head_dim     = head_dim;

    // Phase 2.2.Attn buffers (only if attention dims > 0)
    if (ok && num_heads > 0 && head_dim > 0) {
        ok &= alloc(&ws.d_input_norm,    hidden_dim * sizeof(__half));
        ok &= alloc(&ws.d_q_full,        2 * num_heads * head_dim * sizeof(__half));
        ok &= alloc(&ws.d_q_attn,        num_heads * head_dim * sizeof(__half));
        if (num_kv_heads > 0) {
            ok &= alloc(&ws.d_k_proj,    num_kv_heads * head_dim * sizeof(__half));
            ok &= alloc(&ws.d_v_proj,    num_kv_heads * head_dim * sizeof(__half));
        }
        ok &= alloc(&ws.d_attn_out,      num_heads * head_dim * sizeof(__half));
        ok &= alloc(&ws.d_attn_residual, hidden_dim * sizeof(__half));
        // Device int for RoPE position (single int)
        ok &= (cudaMalloc(reinterpret_cast<void**>(&ws.d_mtp_position), sizeof(int)) == cudaSuccess);
    }
    // Phase 2.2.Attn+KV buffers (cap max_seq_len)
    if (ok && num_heads > 0 && head_dim > 0 && num_kv_heads > 0 && max_seq_len > 0) {
        size_t kv_bytes = static_cast<size_t>(max_seq_len) * num_kv_heads * head_dim * sizeof(__half);
        ok &= alloc(&ws.d_k_cache, kv_bytes);
        ok &= alloc(&ws.d_v_cache, kv_bytes);
        ws.max_seq_len = max_seq_len;
        ws.mtp_pos = 0;
    }

    // Phase 2.2 MoE buffers (only if n_experts > 0)
    const bool has_moe = n_experts > 0 && top_k > 0 && expert_d_ff > 0;
    if (ok && has_moe) {
        ok &= alloc(&ws.d_expert_gate_up,  2 * expert_d_ff * sizeof(__half));
        ok &= alloc(&ws.d_expert_act,      expert_d_ff * sizeof(__half));
        ok &= alloc(&ws.d_expert_outputs,  top_k * hidden_dim * sizeof(__half));

        // Routing pool (max 1 token for M=1 decode).
        ws.routing_buf.allocate(/*max_tokens=*/1, /*max_experts=*/n_experts, /*top_k=*/top_k);

        // Pinned host buffers for D2H of routing decision.
        if (ok) {
            ws.h_expert_indices = PinnedBuffer::acquire(cuda_host_pinned_allocator(),
                                                       top_k * sizeof(int));
            ws.h_expert_weights = PinnedBuffer::acquire(cuda_host_pinned_allocator(),
                                                       top_k * sizeof(float));
            ok &= !ws.h_expert_indices.empty() && !ws.h_expert_weights.empty();
        }
    }
    // MLP scratch shared by both variants: the MoE shared expert AND the
    // dense MTP MLP (Qwen3.6 dense checkpoints embed a plain SwiGLU MLP,
    // mapped onto the shared_expert fields — no router, no sigmoid gate).
    if (ok && (has_moe || shared_d_ff > 0)) {
        ok &= alloc(&ws.d_post_norm, hidden_dim * sizeof(__half));
        ok &= alloc(&ws.d_moe_out,   hidden_dim * sizeof(__half));
    }
    if (ok && shared_d_ff > 0) {
        ok &= alloc(&ws.d_shared_gate, shared_d_ff * sizeof(__half));
        ok &= alloc(&ws.d_shared_up,   shared_d_ff * sizeof(__half));
        ok &= alloc(&ws.d_shared_act,  shared_d_ff * sizeof(__half));
        ok &= alloc(&ws.d_shared_out,  hidden_dim * sizeof(__half));
    }

    if (!ok) mtp_workspace_free(ws);
    return ok;
}

void mtp_workspace_free(MtpDraftWorkspace& ws) {
    auto frfn = [](void*& p) {
        if (p) { cudaFree(p); p = nullptr; }
    };
    frfn(ws.d_emb_norm);
    frfn(ws.d_h_norm);
    frfn(ws.d_fc_in);
    frfn(ws.d_fc_out);
    frfn(ws.d_h_final);
    frfn(ws.d_logits);
    frfn(ws.d_logits_f32);
    if (ws.d_topk) { cudaFree(ws.d_topk); ws.d_topk = nullptr; }
    if (ws.d_chain_tokens) { cudaFree(ws.d_chain_tokens); ws.d_chain_tokens = nullptr; }
    if (ws.d_argmax) { cudaFree(ws.d_argmax); ws.d_argmax = nullptr; }
    if (ws.d_tok) {
        cudaFree(ws.d_tok);
        ws.d_tok = nullptr;
    }
    frfn(ws.d_post_norm);
    frfn(ws.d_expert_gate_up);
    frfn(ws.d_expert_act);
    frfn(ws.d_expert_outputs);
    frfn(ws.d_moe_out);
    frfn(ws.d_shared_gate);
    frfn(ws.d_shared_up);
    frfn(ws.d_shared_act);
    frfn(ws.d_shared_out);
    ws.routing_buf.free();
    ws.h_expert_indices.reset();
    ws.h_expert_weights.reset();
    frfn(ws.d_input_norm);
    frfn(ws.d_q_full);
    frfn(ws.d_q_attn);
    frfn(ws.d_k_proj);
    frfn(ws.d_v_proj);
    frfn(ws.d_attn_out);
    frfn(ws.d_attn_residual);
    if (ws.d_mtp_position) { cudaFree(ws.d_mtp_position); ws.d_mtp_position = nullptr; }
    frfn(ws.d_k_cache);
    frfn(ws.d_v_cache);
    ws.mtp_pos = 0;
    ws.max_seq_len = 0;
    ws.hidden_dim = ws.n_experts = ws.top_k = ws.expert_d_ff = ws.shared_d_ff = 0;
    ws.num_heads = ws.num_kv_heads = ws.head_dim = 0;
}

// ---------------------------------------------------------------------------
// Draft step
// ---------------------------------------------------------------------------
bool mtp_draft_step(int prev_token_id, const void* d_h_prev,
                    const MtpHead& mtp,
                    const Tensor& main_tok_emb,
                    const Tensor& main_lm_head,
                    MtpDraftWorkspace& ws,
                    int hidden_dim, int vocab_size,
                    int* out_token_id,
                    cudaStream_t stream,
                    int* out_topk_ids, int top_w,
                    const NvFP4QuantResult* lm_head_nvfp4,
                    const int32_t* d_prev_token,
                    int32_t* d_out_token) {
    if (!mtp.loaded) {
        IMP_LOG_ERROR("mtp_draft_step: MTP head not loaded");
        return false;
    }
    if (!d_h_prev) return false;  // out_token_id == nullptr → feed-only step
    if (!ws.d_emb_norm || !ws.d_h_norm || !ws.d_fc_in || !ws.d_fc_out ||
        !ws.d_h_final || !ws.d_logits) {
        IMP_LOG_ERROR("mtp_draft_step: workspace not allocated");
        return false;
    }
    if (main_tok_emb.data == nullptr || main_lm_head.data == nullptr) {
        IMP_LOG_ERROR("mtp_draft_step: main embedding or lm_head not on GPU");
        return false;
    }
    if (d_prev_token == nullptr && (prev_token_id < 0 || prev_token_id >= vocab_size)) {
        IMP_LOG_ERROR("mtp_draft_step: token_id %d out of range [0,%d)",
                      prev_token_id, vocab_size);
        return false;
    }

    // Step 1: embedding lookup for prev_token_id.
    // CRITICAL: the main model's embedding table is NVFP4-quantized on
    // Qwen3.6-NVFP4 (lm_head is the only ignored module). Reading it as
    // raw FP16 produces garbage — every "embedding" decoded to the same
    // bit pattern, locking MTP predictions to a single token regardless
    // of input. imp::embedding_lookup handles the qtype dispatch.
    if (d_prev_token != nullptr) {
        // Device-chain input: the previous step's argmax already lives on
        // device — no upload, no scratch.
        int64_t out_shape[2] = {1, hidden_dim};
        Tensor  out_view(ws.d_fc_in, QType::F16, 2, out_shape, /*on_device=*/true);
        imp::embedding_lookup(main_tok_emb, d_prev_token, /*n_tokens=*/1, out_view,
                              main_tok_emb.qtype, stream);
    } else {
        // Upload prev_token_id to the workspace's persistent token-id slot so
        // embedding_lookup can dispatch with the correct signature. (The
        // graph-friendly _from_device overload also exists if needed.)
        // Persistent rather than per-step: see the ws.d_tok comment in the
        // header. Allocated once by mtp_workspace_allocate.
        if (ws.d_tok == nullptr) {
            IMP_LOG_ERROR("mtp_draft_step: token-id scratch not allocated");
            return false;
        }
        int32_t h_tok = static_cast<int32_t>(prev_token_id);
        cudaMemcpyAsync(ws.d_tok, &h_tok, sizeof(int32_t), cudaMemcpyHostToDevice, stream);
        int64_t out_shape[2] = {1, hidden_dim};
        Tensor  out_view(ws.d_fc_in, QType::F16, 2, out_shape, /*on_device=*/true);
        imp::embedding_lookup(main_tok_emb, ws.d_tok, /*n_tokens=*/1, out_view, main_tok_emb.qtype, stream);
    }

    // Step 2: emb_norm = RMSNorm(emb, pre_fc_norm_embedding)
    // imp::rmsnorm dispatcher reads x.shape[0]=rows + x.shape[1]=d_model and
    // EARLY-RETURNS when d_model==0. A 1D Tensor [hidden_dim] would be
    // misinterpreted as rows=hidden_dim, d_model=0 — no work would be done
    // and the output buffer would keep its uninitialized contents.
    // → MUST use 2D shape [1, hidden_dim].
    int64_t shape_2d[2]  = {1, hidden_dim};
    Tensor emb_view(ws.d_fc_in,   QType::F16, 2, shape_2d, /*on_device=*/true);
    Tensor h_view  (const_cast<void*>(d_h_prev), QType::F16, 2, shape_2d, true);
    Tensor emb_n   (ws.d_emb_norm, QType::F16, 2, shape_2d, true);
    Tensor h_n     (ws.d_h_norm,   QType::F16, 2, shape_2d, true);
    imp::rmsnorm(emb_view, mtp.pre_fc_norm_embedding, emb_n, 1e-6f, stream);
    imp::rmsnorm(h_view,   mtp.pre_fc_norm_hidden,    h_n,   1e-6f, stream);

    // Step 3: concat(emb_n, h_n) into d_fc_in (overwrites the temp emb storage).
    {
        int block = 256;
        int grid  = (2 * hidden_dim + block - 1) / block;
        mtp_concat_kernel<<<grid, block, 0, stream>>>(
            static_cast<const __half*>(ws.d_emb_norm),
            static_cast<const __half*>(ws.d_h_norm),
            static_cast<__half*>(ws.d_fc_in),
            hidden_dim);
        IMP_CUDA_CHECK_LAUNCH();
    }

    // Step 4: fc_out = fc @ fc_in  ([hidden_dim, 2*hidden_dim] x [2*hidden_dim] = [hidden_dim])
    {
        int64_t fc_in_shape[2]  = {1, 2 * hidden_dim};
        int64_t fc_out_shape[2] = {1, hidden_dim};
        Tensor fc_in_view (ws.d_fc_in,  QType::F16, 2, fc_in_shape,  true);
        Tensor fc_out_view(ws.d_fc_out, QType::F16, 2, fc_out_shape, true);
        imp::gemm(fc_in_view, mtp.fc, fc_out_view, 1.0f, 0.0f, stream);
    }

    // Step 5: transformer block.
    //
    // 5.A — Attention (Phase 2.2.Attn MVP): Qwen3.6 MTP uses
    //   attn_output_gate=True (per upstream vllm `Qwen3NextAttention`):
    //   q_proj outputs [num_heads, 2*head_dim] per-token, split per-head
    //   into (q, gate). Standard GQA attention produces out[h] of head_dim,
    //   then out *= silu(gate) before o_proj reduces to hidden_dim.
    //
    //   This MVP handles the M=1 first-draft case (no MTP KV history yet):
    //   the softmax over a single token reduces to identity, so
    //   attn_out[h] = V[h // GQA_group] (broadcast). The gate-output
    //   multiplication still fires correctly. K is computed but unused.
    //
    //   K>=1 draft steps would attend over prior MTP K cache entries — a
    //   full KV cache + attention kernel is future work (Phase 2.2.Attn+KV).
    if (ws.num_heads > 0 && ws.head_dim > 0 &&
        mtp.input_layernorm.data && mtp.q_proj.data && mtp.k_proj.data &&
        mtp.v_proj.data && mtp.o_proj.data) {
        const int hd  = hidden_dim;
        const int nh  = ws.num_heads;
        const int nkv = ws.num_kv_heads;
        const int hdh = ws.head_dim;

        // 5.A.1 — input_layernorm(fc_out) → d_input_norm
        {
            int64_t hd1[2] = {1, hd};
            Tensor fc_out_view (ws.d_fc_out,    QType::F16, 2, hd1, true);
            Tensor in_view     (ws.d_input_norm,QType::F16, 2, hd1, true);
            imp::rmsnorm(fc_out_view, mtp.input_layernorm, in_view, 1e-6f, stream);
        }
        // 5.A.2 — Q (full, including gate): q_proj @ d_input_norm → [2 * nh * hdh]
        {
            int64_t in_shape[2]  = {1, hd};
            int64_t out_shape[2] = {1, 2 * nh * hdh};
            Tensor in_view (ws.d_input_norm, QType::F16, 2, in_shape,  true);
            Tensor out_view(ws.d_q_full,     QType::F16, 2, out_shape, true);
            imp::gemm(in_view, mtp.q_proj, out_view, 1.0f, 0.0f, stream);
        }
        // 5.A.3 — K, V: k_proj/v_proj @ d_input_norm → [nkv * hdh] each
        if (ws.d_k_proj && nkv > 0) {
            int64_t in_shape[2]  = {1, hd};
            int64_t out_shape[2] = {1, nkv * hdh};
            Tensor in_view (ws.d_input_norm, QType::F16, 2, in_shape,  true);
            Tensor out_view(ws.d_k_proj,     QType::F16, 2, out_shape, true);
            imp::gemm(in_view, mtp.k_proj, out_view, 1.0f, 0.0f, stream);
        }
        if (ws.d_v_proj && nkv > 0) {
            int64_t in_shape[2]  = {1, hd};
            int64_t out_shape[2] = {1, nkv * hdh};
            Tensor in_view (ws.d_input_norm, QType::F16, 2, in_shape,  true);
            Tensor out_view(ws.d_v_proj,     QType::F16, 2, out_shape, true);
            imp::gemm(in_view, mtp.v_proj, out_view, 1.0f, 0.0f, stream);
        }

        // 5.A.4 — Attention path:
        //   - With KV cache present + max_seq capacity remaining: extract Q
        //     from q_full per-head, apply fused qk-norm+RoPE on (Q,K),
        //     append rotated K + V to cache, run softmax attention scan over
        //     positions [0, mtp_pos+1), apply silu(gate) elementwise.
        //   - Else (cache absent or full): fall back to M=1 broadcast MVP.
        bool use_kv_scan = (ws.d_k_cache != nullptr && ws.d_v_cache != nullptr &&
                            ws.max_seq_len > 0 && ws.mtp_pos < ws.max_seq_len);
        if (use_kv_scan) {
            // 5.A.4.pre — Extract Q (without gate) from q_full[h, 0..head_dim).
            // q_full layout per head: [q (head_dim), gate (head_dim)] when the
            // head is attn_output_gate=True (Qwen3.6). Nemotron has no gate
            // half, so its q_full is already the contiguous Q buffer and the
            // strided copy would interleave garbage from the next head.
            const size_t q_src_pitch = static_cast<size_t>(mtp.attn_output_gate ? 2 * hdh : hdh) *
                                       sizeof(__half);
            cudaMemcpy2DAsync(
                /*dst=*/ws.d_q_attn,
                /*dpitch=*/static_cast<size_t>(hdh) * sizeof(__half),
                /*src=*/ws.d_q_full,
                /*spitch=*/q_src_pitch,
                /*width=*/static_cast<size_t>(hdh) * sizeof(__half),
                /*height=*/static_cast<size_t>(nh), cudaMemcpyDeviceToDevice, stream);
            // 5.A.4.qknorm — Per-head RMSNorm on Q and K (Qwen3-style).
            // Reshape to [n_heads, head_dim] and apply rmsnorm with arch_norm_offset
            // for Qwen3.5/3.6's gamma=1+W convention. Independent of RoPE so
            // we can ship qk-norm without committing to standard partial-rope.
            if (mtp.q_norm.data) {
                int64_t q_shape[2] = {nh, hdh};
                Tensor q_view(ws.d_q_attn, QType::F16, 2, q_shape, /*on_device=*/true);
                imp::rmsnorm(q_view, mtp.q_norm, q_view, ws.rms_norm_eps, stream,
                             ws.arch_norm_offset);
            }
            if (mtp.k_norm.data) {
                int64_t k_shape[2] = {nkv, hdh};
                Tensor k_view(ws.d_k_proj, QType::F16, 2, k_shape, /*on_device=*/true);
                imp::rmsnorm(k_view, mtp.k_norm, k_view, ws.rms_norm_eps, stream,
                             ws.arch_norm_offset);
            }
            // 5.A.4.rope — mrope-aware Q/K rotation. For text-only tokens
            // the 3 mrope position components are all equal to mtp_pos,
            // reducing to standard partial-rope mathematically. The kernel
            // is structured to support distinct T/H/W positions for future
            // multimodal token handling. NeoX-style pairing only.
            // Skipped entirely on a NoPE head (Nemotron-H): its main-model
            // attention layers carry no position either — the Mamba layers do.
            // Rotating here would put the draft in a different frame from the
            // model it drafts for, which costs accept rate, not correctness.
            if (mtp.attn_rope && ws.rope_dim > 0 &&
                ws.mrope_sec0 + ws.mrope_sec1 + ws.mrope_sec2 == ws.rope_dim / 2) {
                // RoPE-scaling params mirrored from the main forward (issue #897):
                // inv_scaling = 1/freq_scale; ext_factor>0 → YaRN. Defaults leave
                // the base (unscaled) rope unchanged. Text-only → single position.
                mtp_apply_mrope(ws.d_q_attn, nh, ws.d_k_proj, nkv, hdh, ws.rope_dim, ws.rope_theta,
                                ws.mrope_sec0, ws.mrope_sec1, ws.mrope_sec2, ws.mtp_pos,
                                1.0f / ws.rope_freq_scale, ws.yarn_ext_factor, ws.yarn_attn_factor,
                                ws.yarn_corr_dim_0, ws.yarn_corr_dim_1, stream);
            }
            const int pos = ws.mtp_pos;
            // 5.A.4.a — append k_step, v_step into cache at pos
            {
                int block = 256;
                int grid  = (nkv * hdh + block - 1) / block;
                mtp_kv_append_kernel<<<grid, block, 0, stream>>>(
                    static_cast<const __half*>(ws.d_k_proj),
                    static_cast<const __half*>(ws.d_v_proj),
                    static_cast<__half*>(ws.d_k_cache),
                    static_cast<__half*>(ws.d_v_cache),
                    pos, nkv, hdh);
                IMP_CUDA_CHECK_LAUNCH();
            }
            // 5.A.4.b — softmax attention scan over [0, pos+1)
            //   shared mem: seq_len * sizeof(float). Cap with the kernel's
            //   single-block design — at decode max_seq_len ~16K this is
            //   16K × 4 = 64 KiB, which fits sm_120's per-SM shared-mem budget.
            //   Use opt-in dynamic shared mem.
            {
                const int seq_len = pos + 1;
                const int kBlock = 256;
                const size_t shmem_bytes = static_cast<size_t>(seq_len) * sizeof(float);
                const float scale = 1.0f / sqrtf(static_cast<float>(hdh));
                mtp_attn_kv_scan_kernel<<<nh, kBlock, shmem_bytes, stream>>>(
                    static_cast<const __half*>(ws.d_q_attn),
                    static_cast<const __half*>(ws.d_k_cache),
                    static_cast<const __half*>(ws.d_v_cache),
                    static_cast<__half*>(ws.d_attn_out),
                    seq_len, nh, nkv, hdh, ws.max_seq_len, scale);
                IMP_CUDA_CHECK_LAUNCH();
            }
            // 5.A.4.c — silu(gate) * attn_out (in-place). Only when the head
            // actually has a gate half; without one this would multiply the
            // output by a sigmoid of Q itself.
            if (mtp.attn_output_gate) {
                int block = 256;
                int grid  = (nh * hdh + block - 1) / block;
                mtp_gate_attn_out_kernel<<<grid, block, 0, stream>>>(
                    static_cast<__half*>(ws.d_attn_out),
                    static_cast<const __half*>(ws.d_q_full),
                    nh, hdh);
                IMP_CUDA_CHECK_LAUNCH();
            }
            ws.mtp_pos = pos + 1;
        } else if (mtp.attn_output_gate) {
            // MVP fallback: silu(gate) * V_broadcast
            int block = 256;
            int grid  = (nh * hdh + block - 1) / block;
            mtp_gated_v_broadcast_kernel<<<grid, block, 0, stream>>>(
                static_cast<const __half*>(ws.d_q_full),
                static_cast<const __half*>(ws.d_v_proj),
                static_cast<__half*>(ws.d_attn_out),
                nh, nkv, hdh);
            IMP_CUDA_CHECK_LAUNCH();
        } else {
            // Same fallback without a gate: softmax over one token is identity,
            // so attn_out[h] is just V broadcast across the GQA group.
            const int group = (nkv > 0) ? (nh / nkv) : 1;
            for (int h = 0; h < nh; ++h) {
                const int kv = (group > 0) ? (h / group) : 0;
                cudaMemcpyAsync(static_cast<__half*>(ws.d_attn_out) + static_cast<size_t>(h) * hdh,
                                static_cast<const __half*>(ws.d_v_proj) + static_cast<size_t>(kv) * hdh,
                                static_cast<size_t>(hdh) * sizeof(__half), cudaMemcpyDeviceToDevice, stream);
            }
        }
        // 5.A.5 — o_proj @ d_attn_out → d_attn_residual
        {
            int64_t in_shape[2]  = {1, nh * hdh};
            int64_t out_shape[2] = {1, hd};
            Tensor in_view (ws.d_attn_out,      QType::F16, 2, in_shape,  true);
            Tensor out_view(ws.d_attn_residual, QType::F16, 2, out_shape, true);
            imp::gemm(in_view, mtp.o_proj, out_view, 1.0f, 0.0f, stream);
        }
        // 5.A.6 — residual: fc_out += attn_residual
        {
            int block = 256;
            int grid  = (hd + block - 1) / block;
            mtp_add_kernel<<<grid, block, 0, stream>>>(
                static_cast<__half*>(ws.d_fc_out),
                static_cast<const __half*>(ws.d_attn_residual),
                hd);
            IMP_CUDA_CHECK_LAUNCH();
        }
        // (K is computed for shape symmetry but unused in the M=1, no-history MVP.)
        (void)mtp.k_proj;
        (void)mtp.q_norm;
        (void)mtp.k_norm;
    }

    // 5.B — MLP block. Two checkpoint variants:
    //   MoE (Qwen3.6-35B sidecar): 256-expert top-8 MoE + shared expert with
    //     sigmoid gating (imp::moe_gate_topk_fused / swiglu /
    //     shared_expert_gate_scale primitives).
    //   Dense (Qwen3.6-27B embedded head): a plain SwiGLU MLP — loaded onto
    //     the shared_expert fields, no router, no sigmoid gate. Runs as
    //     "residual + shared path" with the expert stage skipped.
    const bool mtp_has_moe = ws.n_experts > 0 && ws.top_k > 0 && ws.expert_d_ff > 0 &&
                             mtp.router.data != nullptr;
    const bool mtp_has_dense_mlp = !mtp_has_moe && ws.shared_d_ff > 0 &&
                                   mtp.shared_expert_gate_proj.data != nullptr &&
                                   mtp.shared_expert_up_proj.data != nullptr &&
                                   mtp.shared_expert_down_proj.data != nullptr;
    if (mtp_has_moe || mtp_has_dense_mlp) {
        const int hd = hidden_dim;
        const int d_ff_e = ws.expert_d_ff;
        const int d_ff_s = ws.shared_d_ff;
        const int top_k  = ws.top_k;
        const int ne     = ws.n_experts;

        // 5.B.1 — post_attention_layernorm(fc_out) → d_post_norm
        {
            int64_t hd1[2] = {1, hd};
            Tensor fc_out_view (ws.d_fc_out,   QType::F16, 2, hd1, true);
            Tensor pn_view     (ws.d_post_norm,QType::F16, 2, hd1, true);
            imp::rmsnorm(fc_out_view, mtp.post_attention_layernorm, pn_view, 1e-6f, stream);
        }

        if (mtp_has_moe) {
        // 5.B.2 — Router + top-k. moe_gate_topk_fused: router @ post_norm,
        //         softmax, top-k. Writes into ws.routing_buf.
        MoeRoutingResult routing{};
        // Nemotron adds a DeepSeek-style additive bias to the router logits
        // before top-k (`e_score_correction_bias`). Null on the Qwen layout, so
        // that path is unchanged.
        imp::moe_gate_topk_fused(mtp.router.data, ws.d_post_norm, ne, hd, top_k, ws.routing_buf, routing,
                                 stream,
                                 /*use_sigmoid=*/false, /*normalize_weights=*/true,
                                 /*score_bias=*/mtp.router_score_bias.data);

        // 5.B.3 — D2H copy of expert indices + weights so the host loop can
        //         dispatch per-expert GEMVs.
        //
        // Device-side path (Nemotron layout, experts restacked at upload): the
        // GEMV takes the expert id from device memory, so nothing about the
        // routing has to reach the host. This is the difference between a draft
        // that can be captured and one that cannot — the host round trip below
        // costs a full pipeline stall per draft token.
        const bool device_side_experts = mtp.experts_up_stacked.data != nullptr &&
                                         mtp.experts_down_stacked.data != nullptr;
        if (device_side_experts) {
            const int32_t* d_idx = static_cast<const int32_t*>(ws.routing_buf.expert_indices);
            const size_t up_stride = static_cast<size_t>(d_ff_e) * hd;
            const size_t dn_stride = static_cast<size_t>(hd) * d_ff_e;
            // up: y[k] = W_up[e_k] @ post_norm  (shared input → x_stride 0)
            imp::gemv_f16_moe_decode(mtp.experts_up_stacked.data, d_idx,
                                     static_cast<const __half*>(ws.d_post_norm),
                                     static_cast<__half*>(ws.d_expert_gate_up), d_ff_e, hd, up_stride,
                                     /*x_stride=*/0, top_k, stream);
            // act = relu(up)^2 over all top_k slots at once.
            {
                int64_t act_shape[2] = {top_k, d_ff_e};
                Tensor act_t(ws.d_expert_gate_up, QType::F16, 2, act_shape, true);
                imp::relu_sqr_inplace(act_t, stream);
            }
            // down: y[k] = W_down[e_k] @ act[k]  (per-expert input → x_stride d_ff_e)
            imp::gemv_f16_moe_decode(mtp.experts_down_stacked.data, d_idx,
                                     static_cast<const __half*>(ws.d_expert_gate_up),
                                     static_cast<__half*>(ws.d_expert_outputs), hd, d_ff_e, dn_stride,
                                     /*x_stride=*/d_ff_e, top_k, stream);
        } else {
            cudaMemcpyAsync(ws.h_expert_indices.as<int>(), ws.routing_buf.expert_indices, top_k * sizeof(int),
                            cudaMemcpyDeviceToHost, stream);
            cudaMemcpyAsync(ws.h_expert_weights.as<float>(), ws.routing_buf.expert_weights,
                            top_k * sizeof(float), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);

            // 5.B.4 — For each chosen expert: GEMV gate_up_packed[e] @ post_norm,
            //         swiglu, GEMV down_packed[e] @ act, store into d_expert_outputs[k].
            //
            // Layout of packed tensors:
            //   experts_gate_up_packed shape: [ne, 2*d_ff_e, hd]   FP16
            //   experts_down_packed   shape: [ne,   hd,    d_ff_e] FP16
            const size_t gu_per_expert_bytes = static_cast<size_t>(2) * d_ff_e * hd * sizeof(__half);
            const size_t dn_per_expert_bytes = static_cast<size_t>(hd) * d_ff_e * sizeof(__half);

            int64_t gu_shape[2] = {2 * d_ff_e, hd};
            int64_t dn_shape[2] = {hd, d_ff_e};

            for (int k = 0; k < top_k; ++k) {
                int e_idx = ws.h_expert_indices.as<int>()[k];
                if (e_idx < 0 || e_idx >= ne) {
                    IMP_LOG_WARN("mtp MoE: invalid expert index %d (top_k=%d)", e_idx, k);
                    continue;
                }

                // Two layouts. Qwen: one packed [ne, 2*d_ff_e, hd] stack, addressed
                // by offset, gate+up → SwiGLU. Nemotron: per-expert 2-D tensors and
                // no gate half at all — squared ReLU, exactly like its main FFN.
                const bool non_gated = mtp.experts_non_gated;
                Tensor gu_view, dn_view;
                if (non_gated) {
                    gu_view = mtp.experts_up[static_cast<size_t>(e_idx)];
                    dn_view = mtp.experts_down[static_cast<size_t>(e_idx)];
                } else {
                    char* gu_base = static_cast<char*>(mtp.experts_gate_up_packed.data) +
                                    static_cast<size_t>(e_idx) * gu_per_expert_bytes;
                    char* dn_base = static_cast<char*>(mtp.experts_down_packed.data) +
                                    static_cast<size_t>(e_idx) * dn_per_expert_bytes;
                    gu_view = Tensor(gu_base, QType::F16, 2, gu_shape, true);
                    dn_view = Tensor(dn_base, QType::F16, 2, dn_shape, true);
                }

                // gate_up = gu_view @ post_norm  (width 2*d_ff_e gated, d_ff_e not)
                {
                    int64_t in_shape[2] = {1, hd};
                    int64_t out_shape[2] = {1, non_gated ? d_ff_e : 2 * d_ff_e};
                    Tensor in_view(ws.d_post_norm, QType::F16, 2, in_shape, true);
                    Tensor out_view(ws.d_expert_gate_up, QType::F16, 2, out_shape, true);
                    imp::gemm(in_view, gu_view, out_view, 1.0f, 0.0f, stream);
                }
                if (non_gated) {
                    // act = relu(up)^2, in place — then copied to the act buffer the
                    // down projection reads, keeping the two paths' plumbing identical.
                    int64_t act_shape[2] = {1, d_ff_e};
                    Tensor up_t(ws.d_expert_gate_up, QType::F16, 2, act_shape, true);
                    imp::relu_sqr_inplace(up_t, stream);
                    cudaMemcpyAsync(ws.d_expert_act, ws.d_expert_gate_up,
                                    static_cast<size_t>(d_ff_e) * sizeof(__half), cudaMemcpyDeviceToDevice,
                                    stream);
                } else {
                    // swiglu: gate = first half, up = second half → act = silu(gate)*up
                    int64_t half_shape[2] = {1, d_ff_e};
                    Tensor gate_view(ws.d_expert_gate_up, QType::F16, 2, half_shape, true);
                    Tensor up_view(static_cast<char*>(ws.d_expert_gate_up) + d_ff_e * sizeof(__half),
                                   QType::F16, 2, half_shape, true);
                    Tensor act_view(ws.d_expert_act, QType::F16, 2, half_shape, true);
                    imp::swiglu(gate_view, up_view, act_view, stream);
                }
                // down = dn_view @ act → write directly into d_expert_outputs[k * hd]
                {
                    int64_t in_shape[2] = {1, d_ff_e};
                    int64_t out_shape[2] = {1, hd};
                    Tensor in_view(ws.d_expert_act, QType::F16, 2, in_shape, true);
                    __half* out_base = static_cast<__half*>(ws.d_expert_outputs) + k * hd;
                    Tensor out_view(out_base, QType::F16, 2, out_shape, true);
                    imp::gemm(in_view, dn_view, out_view, 1.0f, 0.0f, stream);
                }
            }
        }  // device_side_experts ? ... : host-loop

        // 5.B.5 — Weighted sum + residual: moe_out = fc_out + Σ_k w[k]*expert_outputs[k]
        // Reads expert_weights straight from device memory, so it needs no host
        // copy either — true for both paths above.
        imp::moe_weighted_sum_residual(
            /*expert_outputs=*/ws.d_expert_outputs,
            /*expert_weights=*/ws.routing_buf.expert_weights,
            /*residual=*/      ws.d_fc_out,
            /*output=*/        ws.d_moe_out,
            /*d_model=*/       hd,
            /*top_k=*/         top_k,
            stream);
        } else {
            // Dense variant: no experts — the accumulator starts as the pure
            // residual and the "shared" path below IS the MLP.
            cudaMemcpyAsync(ws.d_moe_out, ws.d_fc_out, hd * sizeof(__half),
                            cudaMemcpyDeviceToDevice, stream);
        }

        // 5.B.6 — Shared expert / dense MLP: silu(gate_proj·x) * (up_proj·x),
        //         optionally scaled by sigmoid(shared_expert_gate · x) (MoE
        //         checkpoints only), added to moe_out (which already includes
        //         the attention residual).
        // The Nemotron shared expert is non-gated: up_proj + down_proj only, so
        // gate_proj being null must not disable it the way it does for a Qwen
        // dense-MLP head.
        const bool shared_non_gated = mtp.experts_non_gated;
        if (d_ff_s > 0 && (mtp.shared_expert_gate_proj.data || shared_non_gated) &&
            mtp.shared_expert_up_proj.data && mtp.shared_expert_down_proj.data) {
            // shared_gate = shared_expert_gate_proj @ post_norm  → [d_ff_s]
            if (!shared_non_gated) {
                int64_t in_shape[2]  = {1, hd};
                int64_t out_shape[2] = {1, d_ff_s};
                Tensor in_view (ws.d_post_norm,   QType::F16, 2, in_shape,  true);
                Tensor out_view(ws.d_shared_gate, QType::F16, 2, out_shape, true);
                imp::gemm(in_view, mtp.shared_expert_gate_proj, out_view, 1.0f, 0.0f, stream);
            }
            // shared_up = shared_expert_up_proj @ post_norm  → [d_ff_s]
            {
                int64_t in_shape[2]  = {1, hd};
                int64_t out_shape[2] = {1, d_ff_s};
                Tensor in_view (ws.d_post_norm, QType::F16, 2, in_shape,  true);
                Tensor out_view(ws.d_shared_up, QType::F16, 2, out_shape, true);
                imp::gemm(in_view, mtp.shared_expert_up_proj, out_view, 1.0f, 0.0f, stream);
            }
            // shared_act = silu(shared_gate) * shared_up, or relu(up)^2 when the
            // checkpoint has no gate half.
            {
                int64_t s_shape[2] = {1, d_ff_s};
                if (shared_non_gated) {
                    Tensor up_t(ws.d_shared_up, QType::F16, 2, s_shape, true);
                    imp::relu_sqr_inplace(up_t, stream);
                    cudaMemcpyAsync(ws.d_shared_act, ws.d_shared_up,
                                    static_cast<size_t>(d_ff_s) * sizeof(__half), cudaMemcpyDeviceToDevice,
                                    stream);
                } else {
                    Tensor gate_view(ws.d_shared_gate, QType::F16, 2, s_shape, true);
                    Tensor up_view(ws.d_shared_up, QType::F16, 2, s_shape, true);
                    Tensor act_view(ws.d_shared_act, QType::F16, 2, s_shape, true);
                    imp::swiglu(gate_view, up_view, act_view, stream);
                }
            }
            // shared_out = shared_expert_down_proj @ shared_act  → [hd]
            {
                int64_t in_shape[2]  = {1, d_ff_s};
                int64_t out_shape[2] = {1, hd};
                Tensor in_view (ws.d_shared_act, QType::F16, 2, in_shape,  true);
                Tensor out_view(ws.d_shared_out, QType::F16, 2, out_shape, true);
                imp::gemm(in_view, mtp.shared_expert_down_proj, out_view, 1.0f, 0.0f, stream);
            }
            // Apply sigmoid(shared_expert_gate · post_norm) scalar to shared_out
            // in-place via the existing fused kernel. MoE checkpoints only —
            // the dense-MLP variant has no gate tensor and is unscaled.
            if (mtp.shared_expert_gate.data != nullptr) {
                imp::shared_expert_gate_scale(
                    /*x=*/ ws.d_post_norm,
                    /*W=*/ mtp.shared_expert_gate.data,
                    /*y_inout=*/ ws.d_shared_out,
                    /*n=*/ 1,
                    /*d_model=*/ hd,
                    /*d=*/ hd,
                    stream);
            }

            // moe_out += shared_out → write back into d_fc_out for downstream
            {
                int block = 256;
                int grid  = (hd + block - 1) / block;
                mtp_add_shared_kernel<<<grid, block, 0, stream>>>(
                    static_cast<__half*>(ws.d_moe_out),
                    static_cast<const __half*>(ws.d_shared_out),
                    hd);
                IMP_CUDA_CHECK_LAUNCH();
            }
        }
        // Copy d_moe_out → d_fc_out (overwrite) so downstream RMSNorm reads the
        // post-transformer hidden state. moe_weighted_sum_residual already
        // added fc_out as residual into d_moe_out; the shared-expert addition
        // above (if present) updates d_moe_out in place.
        cudaMemcpyAsync(ws.d_fc_out, ws.d_moe_out, hd * sizeof(__half),
                        cudaMemcpyDeviceToDevice, stream);
    }
    // else: legacy reduced forward (Phase 2.1 behavior) — d_fc_out unchanged.

    // Step 6: h_final = RMSNorm(fc_out, final_norm)
    {
        int64_t hd1_shape[2] = {1, hidden_dim};
        Tensor fc_out_view (ws.d_fc_out,  QType::F16, 2, hd1_shape, true);
        Tensor h_final_view(ws.d_h_final, QType::F16, 2, hd1_shape, true);
        imp::rmsnorm(fc_out_view, mtp.final_norm, h_final_view, 1e-6f, stream);
    }

    // Feed-only step (prefill / verify catch-up): the KV append above is the
    // whole point — skip the lm_head GEMV, argmax and stream sync.
    if (out_token_id == nullptr && d_out_token == nullptr)
        return true;

    // Step 7: logits = lm_head @ h_final. When an NVFP4 decode-cache view of
    // the lm_head is available, use it — the full-vocab weight read dominates
    // per-draft cost (~2.5 GB FP16 on Qwen3.6-27B's 248k vocab; NVFP4 reads
    // ~4x less). Draft-only precision: verification stays lossless.
    const bool nvfp4_lm = (lm_head_nvfp4 != nullptr && ws.d_logits_f32 != nullptr);
    if (nvfp4_lm) {
        gemv_nvfp4_kpar_fp32(*lm_head_nvfp4, static_cast<const half*>(ws.d_h_final),
                             static_cast<float*>(ws.d_logits_f32), vocab_size, hidden_dim,
                             stream);
    } else {
        int64_t h_final_shape[2] = {1, hidden_dim};
        int64_t logits_shape[2]  = {1, vocab_size};
        Tensor h_final_view(ws.d_h_final, QType::F16, 2, h_final_shape, true);
        Tensor logits_view (ws.d_logits,  QType::F16, 2, logits_shape,  true);
        imp::gemm(h_final_view, main_lm_head, logits_view, 1.0f, 0.0f, stream);
    }

    // Step 8 (device chain): argmax straight into the caller's device slot —
    // no D2H, no sync; the caller drains the chain in one copy at the end.
    if (d_out_token != nullptr) {
        if (nvfp4_lm) {
            mtp_argmax_kernel<<<1, 256, 0, stream>>>(
                static_cast<const float*>(ws.d_logits_f32), vocab_size, d_out_token);
            IMP_CUDA_CHECK_LAUNCH();
        } else {
            mtp_argmax_kernel<<<1, 256, 0, stream>>>(
                static_cast<const __half*>(ws.d_logits), vocab_size, d_out_token);
            IMP_CUDA_CHECK_LAUNCH();
        }
        return true;
    }

    // Step 8: argmax (or top-W) → device int → D2H.
    const bool want_topk = (out_topk_ids != nullptr && top_w > 0);
    if (want_topk) {
        // Top-W path (Stage 0 tree-ceiling probe): reuse the pre-allocated
        // ws.d_topk buffer. out_token_id is set to the argmax (top-0).
        const int w = std::min(top_w, kMtpMaxTopW);
        if (ws.d_topk == nullptr) {
            IMP_LOG_ERROR("mtp_draft_step: top-W requested but ws.d_topk not allocated");
            return false;
        }
        if (nvfp4_lm) {
            mtp_topk_kernel<<<1, 256, 0, stream>>>(
                static_cast<const float*>(ws.d_logits_f32), vocab_size, w, ws.d_topk);
            IMP_CUDA_CHECK_LAUNCH();
        } else {
            mtp_topk_kernel<<<1, 256, 0, stream>>>(
                static_cast<const __half*>(ws.d_logits), vocab_size, w, ws.d_topk);
            IMP_CUDA_CHECK_LAUNCH();
        }
        if (cudaMemcpyAsync(out_topk_ids, ws.d_topk, w * sizeof(int),
                            cudaMemcpyDeviceToHost, stream) != cudaSuccess)
            return false;
        cudaStreamSynchronize(stream);
        *out_token_id = out_topk_ids[0];
        return true;
    }

    // Host path: persistent argmax scratch (ws.d_argmax) — a per-draft
    // cudaMallocAsync/cudaFreeAsync pair costs host time on the chain.
    int* d_idx = ws.d_argmax;
    bool owned_idx = false;
    if (d_idx == nullptr) {
        if (cudaMallocAsync(&d_idx, sizeof(int), stream) != cudaSuccess) {
            IMP_LOG_ERROR("mtp_draft_step: argmax scratch alloc failed");
            return false;
        }
        owned_idx = true;
    }
    if (nvfp4_lm) {
        mtp_argmax_kernel<<<1, 256, 0, stream>>>(
            static_cast<const float*>(ws.d_logits_f32), vocab_size, d_idx);
        IMP_CUDA_CHECK_LAUNCH();
    } else {
        mtp_argmax_kernel<<<1, 256, 0, stream>>>(
            static_cast<const __half*>(ws.d_logits), vocab_size, d_idx);
        IMP_CUDA_CHECK_LAUNCH();
    }
    if (cudaMemcpyAsync(out_token_id, d_idx, sizeof(int),
                        cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
        if (owned_idx) cudaFreeAsync(d_idx, stream);
        return false;
    }
    if (owned_idx) cudaFreeAsync(d_idx, stream);
    cudaStreamSynchronize(stream);
    return true;
}

}  // namespace imp
