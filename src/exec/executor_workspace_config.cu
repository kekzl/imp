#include "exec/executor.h"
#include "exec/workspace.h"
#include "exec/executor_kernels.h"
#include "exec/executor_helpers.h"
#include "core/logging.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>

namespace imp {

// ---------------------------------------------------------------------------
// Shared workspace configuration (pure pointer arithmetic, no allocation)
// ---------------------------------------------------------------------------

void GraphExecutor::configure_attn_workspace(int max_tokens) {
    const auto& cfg = model_->config();
    int d = cfg.d_model;
    int nh = cfg.n_heads;
    int nkv = cfg.n_kv_heads;
    int hd = cfg.head_dim > 0 ? cfg.head_dim : (d / nh);
    size_t es = dtype_size(compute_dtype_);

    char* ptr = static_cast<char*>(ws_.shared());

    q_ = make_workspace_tensor(ptr, compute_dtype_, max_tokens, nh * hd,
                               align256(static_cast<size_t>(max_tokens) * nh * hd * es));
    // K and V are contiguous (no alignment gap) to enable strided batched GEMM.
    // v_.data == k_.data + kv_raw exactly, so output_stride = kv_raw / es.
    // MLA: mla_assemble_kv materialises K/V for all n_heads (not just n_kv_heads=1),
    // so size the workspace to n_heads * head_dim to avoid overflow.
    {
        int kv_cols = cfg.is_mla() ? (nh * hd) : (nkv * hd);
        size_t kv_raw = static_cast<size_t>(max_tokens) * kv_cols * es;
        int64_t kv_shape[2] = {static_cast<int64_t>(max_tokens), static_cast<int64_t>(kv_cols)};
        k_ = Tensor(ptr, compute_dtype_, 2, kv_shape, true);
        v_ = Tensor(ptr + kv_raw, compute_dtype_, 2, kv_shape, true);
        ptr += align256(2 * kv_raw);
    }
    attn_out_ = make_workspace_tensor(ptr, compute_dtype_, max_tokens, nh * hd,
                                      align256(static_cast<size_t>(max_tokens) * nh * hd * es));
    proj_out_ = make_workspace_tensor(ptr, compute_dtype_, max_tokens, d,
                                      align256(static_cast<size_t>(max_tokens) * d * es));
}

void GraphExecutor::configure_ffn_workspace(int max_tokens) {
    const auto& cfg = model_->config();
    int d = cfg.d_model;
    int ff = cfg.d_ff;
    size_t es = dtype_size(compute_dtype_);

    char* ptr = static_cast<char*>(ws_.shared());

    gate_out_ = make_workspace_tensor(ptr, compute_dtype_, max_tokens, ff,
                                      align256(static_cast<size_t>(max_tokens) * ff * es));
    up_out_ = make_workspace_tensor(ptr, compute_dtype_, max_tokens, ff,
                                    align256(static_cast<size_t>(max_tokens) * ff * es));
    swiglu_out_ = make_workspace_tensor(ptr, compute_dtype_, max_tokens, ff,
                                        align256(static_cast<size_t>(max_tokens) * ff * es));
    ffn_out_ = make_workspace_tensor(ptr, compute_dtype_, max_tokens, d,
                                     align256(static_cast<size_t>(max_tokens) * d * es));
}

void GraphExecutor::configure_moe_workspace(int max_tokens) {
    const auto& cfg = model_->config();
    int d = cfg.d_model;
    int ne = cfg.n_experts;
    int top_k = cfg.n_experts_active;
    int eff = max_expert_eff_;
    size_t es = dtype_size(compute_dtype_);
    int expanded = max_tokens * top_k;

    char* ptr = static_cast<char*>(ws_.shared());

    // gate_logits: FP32
    moe_.gate_logits = make_workspace_tensor(ptr, QType::F32, max_tokens, ne,
                                             align256(static_cast<size_t>(max_tokens) * ne * sizeof(float)));

    moe_.gathered = make_workspace_tensor(ptr, compute_dtype_, expanded, d,
                                          align256(static_cast<size_t>(expanded) * d * es));
    moe_.expert_gate = make_workspace_tensor(ptr, compute_dtype_, expanded, eff,
                                             align256(static_cast<size_t>(expanded) * eff * es));
    moe_.expert_up = make_workspace_tensor(ptr, compute_dtype_, expanded, eff,
                                           align256(static_cast<size_t>(expanded) * eff * es));
    moe_.expert_swiglu = make_workspace_tensor(ptr, compute_dtype_, expanded, eff,
                                               align256(static_cast<size_t>(expanded) * eff * es));
    moe_.expert_down = make_workspace_tensor(ptr, compute_dtype_, expanded, d,
                                             align256(static_cast<size_t>(expanded) * d * es));
    moe_.scatter_out = make_workspace_tensor(ptr, QType::F32, max_tokens, d,
                                             align256(static_cast<size_t>(max_tokens) * d * sizeof(float)));
}

void GraphExecutor::configure_ssm_workspace(int max_tokens) {
    const auto& cfg = model_->config();
    int d = cfg.d_model;
    int inner = cfg.ssm_inner_size;
    int n_heads = cfg.ssm_dt_rank;
    int conv_channels = cfg.ssm_conv_channels();
    int ssm_in_dim = inner + conv_channels + n_heads;
    size_t es = dtype_size(compute_dtype_);

    char* ptr = static_cast<char*>(ws_.shared());

    // GDN layers need FP32 intermediate (4 bytes/elem) for numerical precision.
    // Non-GDN SSM layers only need FP16 (es bytes/elem).
    size_t proj_elem_size = has_gdn_ ? sizeof(float) : es;
    ssm_proj_buf_ = make_workspace_tensor(ptr, compute_dtype_, max_tokens, ssm_in_dim,
                                          align256(static_cast<size_t>(max_tokens) * ssm_in_dim *
                                                   proj_elem_size));
    ssm_xBC_buf_ = make_workspace_tensor(ptr, compute_dtype_, max_tokens, conv_channels,
                                         align256(static_cast<size_t>(max_tokens) * conv_channels * es));
    ssm_y_buf_ = make_workspace_tensor(ptr, compute_dtype_, max_tokens, inner,
                                       align256(static_cast<size_t>(max_tokens) * inner * es));
    ssm_z_buf_ = make_workspace_tensor(ptr, compute_dtype_, max_tokens, inner,
                                       align256(static_cast<size_t>(max_tokens) * inner * es));
    ssm_out_buf_ = make_workspace_tensor(ptr, compute_dtype_, max_tokens, d,
                                         align256(static_cast<size_t>(max_tokens) * d * es));
    // GDN layers store BOTH alpha and beta projections in ssm_dt_buf_ (sequentially).
    // Allocate 2x n_heads to fit both. Non-GDN SSM only uses 1x (dt projection).
    size_t dt_multiplier = has_gdn_ ? 2 : 1;
    ssm_dt_buf_ = make_workspace_tensor(ptr, compute_dtype_, max_tokens, n_heads * dt_multiplier,
                                        align256(static_cast<size_t>(max_tokens) * n_heads * dt_multiplier *
                                                 es));
    // Output buffer for the fused GDN input projection (4-way: ssm_in + gdn_gate +
    // gdn_alpha + gdn_beta concatenated along N). Only sized on has_gdn_ models.
    int fused_total_out = conv_channels + inner + 2 * n_heads;
    size_t fused_bytes =
        has_gdn_ ? align256(static_cast<size_t>(max_tokens) * fused_total_out * es) : align256(0);
    gdn_fused_proj_buf_ =
        make_workspace_tensor(ptr, compute_dtype_, max_tokens, fused_total_out, fused_bytes);
}

bool Workspace::resize_workspace(int new_max_tokens, cudaStream_t stream) {
    // Resize targets the PREFILL shared arena. While the decode workspace is
    // active (slot 1), shared_workspace_ aliases the fixed-size decode buffer
    // — growing through that alias cudaFreeAsync's the decode buffer and
    // leaves decode_shared_workspace_ dangling; the next use_workspace(1)
    // re-installs the freed pointer and decode kernels write into freed
    // memory (#948: server wedged with an illegal memory access whenever a
    // chunked prefill followed a batch=1 decode, which leaves slot 1 active).
    // Restore the prefill workspace first, exactly like the decode path does
    // before its own resize (step_decode_forward).
    if (active_workspace_ == 1)
        use_workspace(0);
    if (new_max_tokens == shared_workspace_max_tokens_ || new_max_tokens <= 0)
        return true;
    if (new_max_tokens > *max_tokens_)
        new_max_tokens = *max_tokens_;  // never exceed init-time max

    // Recompute shared sizes for the new token count
    int saved_max = *max_tokens_;
    *max_tokens_ = new_max_tokens;
    compute_shared_sizes(new_max_tokens);
    *max_tokens_ = saved_max;

    size_t new_shared = std::max({attn_shared_size_, ffn_shared_size_, moe_shared_size_, ssm_shared_size_});
    if (new_shared == 0)
        return true;

    if (new_shared > shared_workspace_size_) {
        // Only reallocate when growing — reuse existing buffer if large enough.
        // This avoids expensive cudaMallocAsync/cudaFreeAsync on every batch size change.
        if (shared_workspace_) {
            IMP_CUDA_CHECK_LOG(cudaFreeAsync(shared_workspace_, stream));
        }
        generation_++;  // arena moves — captured verify graphs must invalidate
        cudaError_t err = cudaMallocAsync(&shared_workspace_, new_shared, stream);
        if (err != cudaSuccess) {
            IMP_LOG_ERROR("Failed to resize shared workspace to %zu bytes: %s", new_shared,
                          cudaGetErrorString(err));
            shared_workspace_ = nullptr;
            shared_workspace_size_ = 0;
            return false;
        }
        shared_workspace_size_ = new_shared;
    }
    shared_workspace_max_tokens_ = new_max_tokens;
    return true;
}

// ---------------------------------------------------------------------------
// Dual workspace for concurrent prefill/decode overlap
// ---------------------------------------------------------------------------

bool Workspace::allocate_decode_workspace(cudaStream_t stream, int max_batch) {
    if (decode_workspace_)
        return true;  // already allocated
    if (max_batch <= 0)
        max_batch = 1;

    const auto& cfg = model_->config();
    int dm = cfg.d_model;
    decode_max_batch_ = max_batch;

    // Persistent workspace for max_batch tokens: hidden + residual + norm_out + logits
    size_t persistent = static_cast<size_t>(dm) * sizeof(half) * 3 *
                            max_batch  // hidden + residual + norm_out
                        + static_cast<size_t>(cfg.vocab_size) * sizeof(float) * max_batch;  // logits
    if ((*fp32_accum_buf_))
        persistent += static_cast<size_t>(dm) * sizeof(float) * max_batch;  // fp32_hidden

    decode_workspace_ = vram_alloc(vram_alloc_, persistent, "decode_workspace");
    if (!decode_workspace_) {
        IMP_LOG_ERROR("Failed to allocate decode workspace");
        return false;
    }
    decode_persistent_size_ = persistent;

    // Shared workspace for max_batch tokens
    int saved = *max_tokens_;
    *max_tokens_ = max_batch;
    compute_shared_sizes(max_batch);
    *max_tokens_ = saved;

    size_t shared = std::max({attn_shared_size_, ffn_shared_size_, moe_shared_size_, ssm_shared_size_});
    if (shared > 0) {
        decode_shared_workspace_ = vram_alloc(vram_alloc_, shared, "decode_shared_workspace");
        if (!decode_shared_workspace_) {
            IMP_LOG_ERROR("Failed to allocate decode shared workspace");
            vram_free(vram_alloc_, decode_workspace_);
            decode_workspace_ = nullptr;
            return false;
        }
    }
    decode_shared_size_ = shared;

    // Recompute sizes for original max_tokens
    compute_shared_sizes(*max_tokens_);

    IMP_LOG_INFO("Decode overlap workspace: %.2f MiB for max_batch=%d (persistent=%.1f KB, shared=%.1f KB)",
                 (persistent + shared) / (1024.0 * 1024.0), max_batch, persistent / 1024.0, shared / 1024.0);
    return true;
}

void Workspace::use_workspace(int slot) {
    if (slot == active_workspace_)
        return;

    const auto& cfg = model_->config();
    int dm = cfg.d_model;

    if (slot == 1 && decode_workspace_) {
        // Save prefill workspace
        saved_prefill_ws_.persistent = persistent_workspace_;
        saved_prefill_ws_.persistent_size = persistent_workspace_size_;
        saved_prefill_ws_.shared = shared_workspace_;
        saved_prefill_ws_.shared_size = shared_workspace_size_;
        saved_prefill_ws_.shared_max_tokens = shared_workspace_max_tokens_;
        saved_prefill_ws_.hidden = (*hidden_);
        saved_prefill_ws_.residual = (*residual_);
        saved_prefill_ws_.norm_out = (*norm_out_);
        saved_prefill_ws_.logits = (*logits_);
        saved_prefill_ws_.fp32_accum = (*fp32_accum_buf_);
        saved_prefill_ws_.fp32_hidden = (*fp32_hidden_);

        // Switch to decode workspace
        persistent_workspace_ = decode_workspace_;
        persistent_workspace_size_ = decode_persistent_size_;
        shared_workspace_ = decode_shared_workspace_;
        shared_workspace_size_ = decode_shared_size_;
        shared_workspace_max_tokens_ = decode_max_batch_;

        // Set up tensor views into decode workspace (sized for decode_max_batch_)
        int mb = decode_max_batch_;
        char* p = static_cast<char*>(decode_workspace_);
        int64_t shape_mb[2] = {mb, dm};
        (*hidden_) = Tensor(p, QType::F16, 2, shape_mb, true);
        p += static_cast<size_t>(dm) * sizeof(half) * mb;
        (*residual_) = Tensor(p, QType::F16, 2, shape_mb, true);
        p += static_cast<size_t>(dm) * sizeof(half) * mb;
        (*norm_out_) = Tensor(p, QType::F16, 2, shape_mb, true);
        p += static_cast<size_t>(dm) * sizeof(half) * mb;
        int64_t shape_logits[2] = {mb, cfg.vocab_size};
        (*logits_) = Tensor(p, QType::F32, 2, shape_logits, true);
        p += static_cast<size_t>(cfg.vocab_size) * sizeof(float) * mb;
        if (saved_prefill_ws_.fp32_accum) {
            (*fp32_accum_buf_) = p;
            int64_t shape_fp32[2] = {mb, dm};
            (*fp32_hidden_) = Tensor(p, QType::F32, 2, shape_fp32, true);
        }

        active_workspace_ = 1;
    } else if (slot == 0) {
        // Restore prefill workspace
        persistent_workspace_ = saved_prefill_ws_.persistent;
        persistent_workspace_size_ = saved_prefill_ws_.persistent_size;
        shared_workspace_ = saved_prefill_ws_.shared;
        shared_workspace_size_ = saved_prefill_ws_.shared_size;
        shared_workspace_max_tokens_ = saved_prefill_ws_.shared_max_tokens;
        (*hidden_) = saved_prefill_ws_.hidden;
        (*residual_) = saved_prefill_ws_.residual;
        (*norm_out_) = saved_prefill_ws_.norm_out;
        (*logits_) = saved_prefill_ws_.logits;
        (*fp32_accum_buf_) = saved_prefill_ws_.fp32_accum;
        (*fp32_hidden_) = saved_prefill_ws_.fp32_hidden;
        active_workspace_ = 0;
    }
}

bool GraphExecutor::layer_has_attention(int layer) const { return model_->layer(layer).wq.data != nullptr; }

bool GraphExecutor::layer_has_ssm(int layer) const { return model_->layer(layer).ssm_in.data != nullptr; }

bool GraphExecutor::layer_has_gdn(int layer) const { return model_->layer(layer).gdn_gate.data != nullptr; }

bool GraphExecutor::layer_has_moe(int layer) const {
    const auto& ly = model_->layer(layer);
    return ly.moe_gate.data != nullptr;
}

bool GraphExecutor::layer_has_dense_ffn(int layer) const {
    const auto& ly = model_->layer(layer);
    return ly.w_up.data != nullptr && ly.moe_gate.data == nullptr;
}

Tensor GraphExecutor::view_tokens(const Tensor& buf, int n_tokens) const {
    // buf is always [max_tokens_, cols] from allocate_buffers.
    // Return a [n_tokens, cols] view.
    return slice_rows(buf, n_tokens);
}

void GraphExecutor::ensure_logits_pinned(int total_floats) {
    if (h_logits_pinned_ && h_logits_pinned_size_ >= total_floats)
        return;
    if (h_logits_pinned_)
        IMP_CUDA_CHECK_LOG(cudaFreeHost(h_logits_pinned_));
    IMP_CUDA_CHECK_LOG(cudaHostAlloc(&h_logits_pinned_, total_floats * sizeof(float), cudaHostAllocDefault));
    h_logits_pinned_size_ = total_floats;
}

}  // namespace imp
