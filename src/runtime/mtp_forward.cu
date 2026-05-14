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
#include "compute/layernorm.h"
#include "compute/gemm.h"
#include "core/logging.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdlib>

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
// Workspace alloc/free
// ---------------------------------------------------------------------------
bool mtp_workspace_allocate(MtpDraftWorkspace& ws, int hidden_dim, int vocab_size) {
    if (hidden_dim <= 0 || vocab_size <= 0) return false;
    auto alloc = [](void** p, size_t bytes) {
        return cudaMalloc(p, bytes) == cudaSuccess;
    };
    bool ok = true;
    ok &= alloc(&ws.d_emb_norm,   hidden_dim * sizeof(__half));
    ok &= alloc(&ws.d_h_norm,     hidden_dim * sizeof(__half));
    ok &= alloc(&ws.d_fc_in,      2 * hidden_dim * sizeof(__half));
    ok &= alloc(&ws.d_fc_out,     hidden_dim * sizeof(__half));
    ok &= alloc(&ws.d_h_final,    hidden_dim * sizeof(__half));
    ok &= alloc(&ws.d_logits,     vocab_size * sizeof(__half));
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
}

// ---------------------------------------------------------------------------
// Draft step
// ---------------------------------------------------------------------------
bool mtp_draft_step(int prev_token_id, const void* d_h_prev,
                    const MtpHead& mtp,
                    const Tensor& main_tok_emb,
                    const Tensor& main_lm_head,
                    const MtpDraftWorkspace& ws,
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

    // Step 5 (PHASE 2.2 PLACEHOLDER): transformer block skipped.
    //   Real impl: input_layernorm → self_attn (q/k/v_proj + GQA + o_proj) +
    //   residual → post_attention_layernorm → MoE (router + 256 experts +
    //   shared expert) + residual.
    //   For Phase 2.1: passthrough — copy d_fc_out into d_fc_out (no-op).

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
