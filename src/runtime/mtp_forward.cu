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

#include "runtime/mtp_forward.h"
#include "compute/activation.h"     // swiglu, shared_expert_gate_scale
#include "compute/gemm.h"
#include "compute/layernorm.h"
#include "compute/moe_routing.h"
#include "core/logging.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdlib>
#include <cstdio>

namespace imp {

// ---------------------------------------------------------------------------
// Tiny kernels
// ---------------------------------------------------------------------------

// Gather one row of an FP16 embedding matrix [vocab, hidden] into a flat output
// [hidden]. One CTA, hidden_dim threads (rounded up).
__global__ void mtp_emb_gather_kernel(int token_id, const __half* __restrict__ emb,
                                       __half* __restrict__ out, int hidden_dim) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= hidden_dim) return;
    out[t] = emb[static_cast<int64_t>(token_id) * hidden_dim + t];
}

// Concatenate two [hidden_dim] FP16 vectors into [2*hidden_dim].
// out[0..hidden_dim-1]   = a
// out[hidden_dim..2hd-1] = b
__global__ void mtp_concat_kernel(const __half* __restrict__ a, const __half* __restrict__ b,
                                   __half* __restrict__ out, int hidden_dim) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= 2 * hidden_dim) return;
    out[t] = (t < hidden_dim) ? a[t] : b[t - hidden_dim];
}

// Argmax over an FP16 vector [vocab_size]. Single CTA; uses shared-memory
// reduction. Writes the argmax index to *out_idx as int32. Caller must ensure
// vocab_size fits one block (i.e., we strip-mine over vocab_size).
__global__ void mtp_argmax_kernel(const __half* __restrict__ logits, int vocab_size,
                                   int* __restrict__ out_idx) {
    constexpr int kThreads = 256;
    __shared__ float s_val[kThreads];
    __shared__ int   s_idx[kThreads];

    int tid = threadIdx.x;
    float best_val = -1.0e38f;
    int   best_idx = 0;
    for (int i = tid; i < vocab_size; i += kThreads) {
        float v = __half2float(logits[i]);
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
    float silu_g = g * (1.0f / (1.0f + expf(-g)));

    float v_val = __half2float(v[kv_h * head_dim + d]);
    out[t] = __float2half(silu_g * v_val);
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
// Workspace alloc/free
// ---------------------------------------------------------------------------
bool mtp_workspace_allocate(MtpDraftWorkspace& ws, int hidden_dim, int vocab_size,
                            int n_experts, int top_k, int expert_d_ff, int shared_d_ff,
                            int num_heads, int num_kv_heads, int head_dim) {
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
        if (num_kv_heads > 0) {
            ok &= alloc(&ws.d_k_proj,    num_kv_heads * head_dim * sizeof(__half));
            ok &= alloc(&ws.d_v_proj,    num_kv_heads * head_dim * sizeof(__half));
        }
        ok &= alloc(&ws.d_attn_out,      num_heads * head_dim * sizeof(__half));
        ok &= alloc(&ws.d_attn_residual, hidden_dim * sizeof(__half));
    }

    // Phase 2.2 MoE buffers (only if n_experts > 0)
    if (ok && n_experts > 0 && top_k > 0 && expert_d_ff > 0) {
        ok &= alloc(&ws.d_post_norm,       hidden_dim * sizeof(__half));
        ok &= alloc(&ws.d_router_logits,   n_experts * sizeof(__half));
        ok &= alloc(&ws.d_expert_gate_up,  2 * expert_d_ff * sizeof(__half));
        ok &= alloc(&ws.d_expert_act,      expert_d_ff * sizeof(__half));
        ok &= alloc(&ws.d_expert_outputs,  top_k * hidden_dim * sizeof(__half));
        ok &= alloc(&ws.d_moe_out,         hidden_dim * sizeof(__half));

        if (shared_d_ff > 0) {
            ok &= alloc(&ws.d_shared_gate, shared_d_ff * sizeof(__half));
            ok &= alloc(&ws.d_shared_up,   shared_d_ff * sizeof(__half));
            ok &= alloc(&ws.d_shared_act,  shared_d_ff * sizeof(__half));
            ok &= alloc(&ws.d_shared_out,  hidden_dim * sizeof(__half));
        }

        // Routing pool (max 1 token for M=1 decode).
        ws.routing_buf.allocate(/*max_tokens=*/1, /*max_experts=*/n_experts, /*top_k=*/top_k);

        // Pinned host buffers for D2H of routing decision.
        if (ok) {
            ok &= (cudaHostAlloc(reinterpret_cast<void**>(&ws.h_expert_indices),
                                  top_k * sizeof(int), cudaHostAllocDefault) == cudaSuccess);
            ok &= (cudaHostAlloc(reinterpret_cast<void**>(&ws.h_expert_weights),
                                  top_k * sizeof(float), cudaHostAllocDefault) == cudaSuccess);
        }
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
    frfn(ws.d_post_norm);
    frfn(ws.d_router_logits);
    frfn(ws.d_expert_gate_up);
    frfn(ws.d_expert_act);
    frfn(ws.d_expert_outputs);
    frfn(ws.d_moe_out);
    frfn(ws.d_shared_gate);
    frfn(ws.d_shared_up);
    frfn(ws.d_shared_act);
    frfn(ws.d_shared_out);
    ws.routing_buf.free();
    if (ws.h_expert_indices) { cudaFreeHost(ws.h_expert_indices); ws.h_expert_indices = nullptr; }
    if (ws.h_expert_weights) { cudaFreeHost(ws.h_expert_weights); ws.h_expert_weights = nullptr; }
    frfn(ws.d_input_norm);
    frfn(ws.d_q_full);
    frfn(ws.d_k_proj);
    frfn(ws.d_v_proj);
    frfn(ws.d_attn_out);
    frfn(ws.d_attn_residual);
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
                    cudaStream_t stream) {
    if (!mtp.loaded) {
        IMP_LOG_ERROR("mtp_draft_step: MTP head not loaded");
        return false;
    }
    if (!d_h_prev || !out_token_id) return false;
    if (!ws.d_emb_norm || !ws.d_h_norm || !ws.d_fc_in || !ws.d_fc_out ||
        !ws.d_h_final || !ws.d_logits) {
        IMP_LOG_ERROR("mtp_draft_step: workspace not allocated");
        return false;
    }
    if (main_tok_emb.data == nullptr || main_lm_head.data == nullptr) {
        IMP_LOG_ERROR("mtp_draft_step: main embedding or lm_head not on GPU");
        return false;
    }
    if (prev_token_id < 0 || prev_token_id >= vocab_size) {
        IMP_LOG_ERROR("mtp_draft_step: token_id %d out of range [0,%d)",
                      prev_token_id, vocab_size);
        return false;
    }

    // Step 1: gather embedding for prev_token_id into d_fc_in's first hidden_dim slot.
    // We'll overwrite with normalized result via rmsnorm next.
    {
        int block = 256;
        int grid  = (hidden_dim + block - 1) / block;
        mtp_emb_gather_kernel<<<grid, block, 0, stream>>>(
            prev_token_id,
            static_cast<const __half*>(main_tok_emb.data),
            static_cast<__half*>(ws.d_fc_in),  // reuse as temp emb storage
            hidden_dim);
    }

    // Step 2: emb_norm = RMSNorm(emb, pre_fc_norm_embedding)
    // Build [1, hidden_dim] FP16 Tensor views around our raw pointers.
    int64_t hd_shape[1]  = {hidden_dim};
    Tensor emb_view(ws.d_fc_in,   QType::F16, 1, hd_shape, /*on_device=*/true);
    Tensor h_view  (const_cast<void*>(d_h_prev), QType::F16, 1, hd_shape, true);
    Tensor emb_n   (ws.d_emb_norm, QType::F16, 1, hd_shape, true);
    Tensor h_n     (ws.d_h_norm,   QType::F16, 1, hd_shape, true);
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
            int64_t hd1[1] = {hd};
            Tensor fc_out_view (ws.d_fc_out,    QType::F16, 1, hd1, true);
            Tensor in_view     (ws.d_input_norm,QType::F16, 1, hd1, true);
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
        // 5.A.3 — V: v_proj @ d_input_norm → [nkv * hdh]
        if (ws.d_v_proj && nkv > 0) {
            int64_t in_shape[2]  = {1, hd};
            int64_t out_shape[2] = {1, nkv * hdh};
            Tensor in_view (ws.d_input_norm, QType::F16, 2, in_shape,  true);
            Tensor out_view(ws.d_v_proj,     QType::F16, 2, out_shape, true);
            imp::gemm(in_view, mtp.v_proj, out_view, 1.0f, 0.0f, stream);
        }
        // 5.A.4 — Gated attention (M=1, no history): out[h, d] = silu(gate[h, d]) * V[h/gqa, d]
        {
            int block = 256;
            int grid  = (nh * hdh + block - 1) / block;
            mtp_gated_v_broadcast_kernel<<<grid, block, 0, stream>>>(
                static_cast<const __half*>(ws.d_q_full),
                static_cast<const __half*>(ws.d_v_proj),
                static_cast<__half*>(ws.d_attn_out),
                nh, nkv, hdh);
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
        }
        // (K is computed for shape symmetry but unused in the M=1, no-history MVP.)
        (void)mtp.k_proj;
        (void)mtp.q_norm;
        (void)mtp.k_norm;
    }

    // 5.B — MoE (Phase 2.2.MoE, this commit): full 256-expert top-8 MoE
    //   forward + shared-expert with sigmoid gating. Uses the existing
    //   imp::moe_gate_topk_fused / swiglu / shared_expert_gate_scale
    //   primitives.
    if (ws.n_experts > 0 && ws.top_k > 0 && ws.expert_d_ff > 0 &&
        mtp.router.data != nullptr) {
        const int hd = hidden_dim;
        const int d_ff_e = ws.expert_d_ff;
        const int d_ff_s = ws.shared_d_ff;
        const int top_k  = ws.top_k;
        const int ne     = ws.n_experts;

        // 5.B.1 — post_attention_layernorm(fc_out) → d_post_norm
        {
            int64_t hd1[1] = {hd};
            Tensor fc_out_view (ws.d_fc_out,   QType::F16, 1, hd1, true);
            Tensor pn_view     (ws.d_post_norm,QType::F16, 1, hd1, true);
            imp::rmsnorm(fc_out_view, mtp.post_attention_layernorm, pn_view, 1e-6f, stream);
        }

        // 5.B.2 — Router + top-k. moe_gate_topk_fused: router @ post_norm,
        //         softmax, top-k. Writes into ws.routing_buf.
        MoeRoutingResult routing{};
        imp::moe_gate_topk_fused(mtp.router.data, ws.d_post_norm, ne, hd, top_k,
                                  ws.routing_buf, routing, stream,
                                  /*use_sigmoid=*/false, /*normalize_weights=*/true);

        // 5.B.3 — D2H copy of expert indices + weights so the host loop can
        //         dispatch per-expert GEMVs. This is non-graph-safe but
        //         drafts run outside graph capture for now.
        cudaMemcpyAsync(ws.h_expert_indices, ws.routing_buf.expert_indices,
                        top_k * sizeof(int), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(ws.h_expert_weights, ws.routing_buf.expert_weights,
                        top_k * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        // 5.B.4 — For each chosen expert: GEMV gate_up_packed[e] @ post_norm,
        //         swiglu, GEMV down_packed[e] @ act, store into d_expert_outputs[k].
        //
        // Layout of packed tensors:
        //   experts_gate_up_packed shape: [ne, 2*d_ff_e, hd]   FP16
        //   experts_down_packed   shape: [ne,   hd,    d_ff_e] FP16
        const size_t gu_per_expert_bytes  = static_cast<size_t>(2) * d_ff_e * hd * sizeof(__half);
        const size_t dn_per_expert_bytes  = static_cast<size_t>(hd) * d_ff_e * sizeof(__half);

        int64_t gu_shape[2] = {2 * d_ff_e, hd};
        int64_t dn_shape[2] = {hd, d_ff_e};

        for (int k = 0; k < top_k; ++k) {
            int   e_idx = ws.h_expert_indices[k];
            if (e_idx < 0 || e_idx >= ne) {
                IMP_LOG_WARN("mtp MoE: invalid expert index %d (top_k=%d)", e_idx, k);
                continue;
            }

            // Build view tensors into the packed buffers for this expert.
            char* gu_base = static_cast<char*>(mtp.experts_gate_up_packed.data)
                            + static_cast<size_t>(e_idx) * gu_per_expert_bytes;
            char* dn_base = static_cast<char*>(mtp.experts_down_packed.data)
                            + static_cast<size_t>(e_idx) * dn_per_expert_bytes;
            Tensor gu_view(gu_base, QType::F16, 2, gu_shape, true);
            Tensor dn_view(dn_base, QType::F16, 2, dn_shape, true);

            // gate_up = gu_view @ post_norm
            {
                int64_t in_shape[2]  = {1, hd};
                int64_t out_shape[2] = {1, 2 * d_ff_e};
                Tensor in_view (ws.d_post_norm,      QType::F16, 2, in_shape,  true);
                Tensor out_view(ws.d_expert_gate_up, QType::F16, 2, out_shape, true);
                imp::gemm(in_view, gu_view, out_view, 1.0f, 0.0f, stream);
            }
            // swiglu: gate = first half, up = second half → act = silu(gate)*up
            {
                int64_t half_shape[2] = {1, d_ff_e};
                Tensor gate_view(ws.d_expert_gate_up,
                                  QType::F16, 2, half_shape, true);
                Tensor up_view(  static_cast<char*>(ws.d_expert_gate_up) + d_ff_e * sizeof(__half),
                                  QType::F16, 2, half_shape, true);
                Tensor act_view(ws.d_expert_act, QType::F16, 2, half_shape, true);
                imp::swiglu(gate_view, up_view, act_view, stream);
            }
            // down = dn_view @ act → write directly into d_expert_outputs[k * hd]
            {
                int64_t in_shape[2]  = {1, d_ff_e};
                int64_t out_shape[2] = {1, hd};
                Tensor in_view (ws.d_expert_act, QType::F16, 2, in_shape, true);
                __half* out_base = static_cast<__half*>(ws.d_expert_outputs) + k * hd;
                Tensor out_view(out_base, QType::F16, 2, out_shape, true);
                imp::gemm(in_view, dn_view, out_view, 1.0f, 0.0f, stream);
            }
        }

        // 5.B.5 — Weighted sum + residual: moe_out = fc_out + Σ_k w[k]*expert_outputs[k]
        imp::moe_weighted_sum_residual(
            /*expert_outputs=*/ws.d_expert_outputs,
            /*expert_weights=*/ws.routing_buf.expert_weights,
            /*residual=*/      ws.d_fc_out,
            /*output=*/        ws.d_moe_out,
            /*d_model=*/       hd,
            /*top_k=*/         top_k,
            stream);

        // 5.B.6 — Shared expert (if present): silu(gate_proj·x) * (up_proj·x),
        //         scale by sigmoid(shared_expert_gate_inp · x), then add to
        //         moe_out (which already includes the attention residual).
        if (d_ff_s > 0 && mtp.shared_expert_gate_proj.data && mtp.shared_expert_up_proj.data &&
            mtp.shared_expert_down_proj.data && mtp.shared_expert_gate.data) {
            // shared_gate = shared_expert_gate_proj @ post_norm  → [d_ff_s]
            {
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
            // shared_act = silu(shared_gate) * shared_up
            {
                int64_t s_shape[2] = {1, d_ff_s};
                Tensor gate_view(ws.d_shared_gate, QType::F16, 2, s_shape, true);
                Tensor up_view  (ws.d_shared_up,   QType::F16, 2, s_shape, true);
                Tensor act_view (ws.d_shared_act,  QType::F16, 2, s_shape, true);
                imp::swiglu(gate_view, up_view, act_view, stream);
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
            // in-place via the existing fused kernel.
            imp::shared_expert_gate_scale(
                /*x=*/ ws.d_post_norm,
                /*W=*/ mtp.shared_expert_gate.data,
                /*y_inout=*/ ws.d_shared_out,
                /*n=*/ 1,
                /*d_model=*/ hd,
                /*d=*/ hd,
                stream);

            // moe_out += shared_out → write back into d_fc_out for downstream
            {
                int block = 256;
                int grid  = (hd + block - 1) / block;
                mtp_add_shared_kernel<<<grid, block, 0, stream>>>(
                    static_cast<__half*>(ws.d_moe_out),
                    static_cast<const __half*>(ws.d_shared_out),
                    hd);
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
        int64_t hd1_shape[1] = {hidden_dim};
        Tensor fc_out_view (ws.d_fc_out,  QType::F16, 1, hd1_shape, true);
        Tensor h_final_view(ws.d_h_final, QType::F16, 1, hd1_shape, true);
        imp::rmsnorm(fc_out_view, mtp.final_norm, h_final_view, 1e-6f, stream);
    }

    // Step 7: logits = lm_head @ h_final
    {
        int64_t h_final_shape[2] = {1, hidden_dim};
        int64_t logits_shape[2]  = {1, vocab_size};
        Tensor h_final_view(ws.d_h_final, QType::F16, 2, h_final_shape, true);
        Tensor logits_view (ws.d_logits,  QType::F16, 2, logits_shape,  true);
        imp::gemm(h_final_view, main_lm_head, logits_view, 1.0f, 0.0f, stream);
    }

    // Step 8: argmax → device int → D2H to out_token_id.
    int* d_idx = nullptr;
    if (cudaMallocAsync(&d_idx, sizeof(int), stream) != cudaSuccess) {
        IMP_LOG_ERROR("mtp_draft_step: argmax scratch alloc failed");
        return false;
    }
    mtp_argmax_kernel<<<1, 256, 0, stream>>>(
        static_cast<const __half*>(ws.d_logits), vocab_size, d_idx);
    if (cudaMemcpyAsync(out_token_id, d_idx, sizeof(int),
                        cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
        cudaFreeAsync(d_idx, stream);
        return false;
    }
    cudaFreeAsync(d_idx, stream);
    cudaStreamSynchronize(stream);
    return true;
}

}  // namespace imp
