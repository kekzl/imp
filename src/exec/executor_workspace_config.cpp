#include "exec/executor.h"
#include "exec/workspace_sizes.h"
#include "exec/workspace.h"
#include "exec/executor_kernels.h"
#include "exec/executor_helpers.h"
#include "core/logging.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <stdexcept>
#include <string>

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

    q_ = make_workspace_tensor(ptr, compute_dtype_, max_tokens, static_cast<int64_t>(nh) * hd,
                               align256(static_cast<size_t>(max_tokens) * nh * hd * es));
    // K and V are contiguous (no alignment gap) for strided batched GEMM: v_.data == k_.data
    // + kv_raw exactly, so output_stride = kv_raw / es.
    // MLA: mla_assemble_kv materializes K/V for all n_heads (not just n_kv_heads=1); size the
    // workspace to n_heads * head_dim to avoid overflow.
    {
        int kv_cols = cfg.is_mla() ? (nh * hd) : (nkv * hd);
        size_t kv_raw = static_cast<size_t>(max_tokens) * kv_cols * es;
        int64_t kv_shape[2] = {static_cast<int64_t>(max_tokens), static_cast<int64_t>(kv_cols)};
        k_ = Tensor(ptr, compute_dtype_, 2, kv_shape, true);
        v_ = Tensor(ptr + kv_raw, compute_dtype_, 2, kv_shape, true);
        ptr += align256(2 * kv_raw);
    }
    attn_out_ = make_workspace_tensor(ptr, compute_dtype_, max_tokens, static_cast<int64_t>(nh) * hd,
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
    int ssm_in_dim = cfg.ssm_proj_dim();
    size_t es = dtype_size(compute_dtype_);

    char* ptr = static_cast<char*>(ws_.shared());

    // GDN layers need FP32 intermediate (4 bytes/elem) for numerical precision.
    // Non-GDN SSM layers only need FP16 (es bytes/elem).
    size_t proj_elem_size = has_gdn_ ? sizeof(float) : es;
    ssm_proj_buf_ = make_workspace_tensor(ptr, compute_dtype_, max_tokens, ssm_in_dim,
                                          exec_ssm_proj_bytes(max_tokens, ssm_in_dim, proj_elem_size));
    ssm_xBC_buf_ = make_workspace_tensor(ptr, compute_dtype_, max_tokens, conv_channels,
                                         align256(static_cast<size_t>(max_tokens) * conv_channels * es));
    ssm_y_buf_ = make_workspace_tensor(ptr, compute_dtype_, max_tokens, inner,
                                       align256(static_cast<size_t>(max_tokens) * inner * es));
    // The attention output gate borrows this buffer, so it is sized for the
    // wider of the two tenants (exec_ssm_z_cols) — not for `inner` alone.
    const int z_cols = ssm_z_cols_ > 0 ? ssm_z_cols_ : inner;
    ssm_z_buf_ = make_workspace_tensor(ptr, compute_dtype_, max_tokens, z_cols,
                                       align256(static_cast<size_t>(max_tokens) * z_cols * es));
    ssm_out_buf_ = make_workspace_tensor(ptr, compute_dtype_, max_tokens, d,
                                         align256(static_cast<size_t>(max_tokens) * d * es));
    // GDN layers store BOTH alpha and beta projections in ssm_dt_buf_ (sequentially).
    // Allocate 2x n_heads to fit both. Non-GDN SSM only uses 1x (dt projection).
    size_t dt_multiplier = has_gdn_ ? 2 : 1;
    ssm_dt_buf_ = make_workspace_tensor(ptr, compute_dtype_, max_tokens, n_heads * dt_multiplier,
                                        has_gdn_ ? exec_gdn_dt_bytes(max_tokens, n_heads, es)
                                                 : align256(static_cast<size_t>(max_tokens) * n_heads * es));
    // Output buffer for the fused GDN input projection (4-way: ssm_in + gdn_gate +
    // gdn_alpha + gdn_beta concatenated along N). Only sized on has_gdn_ models.
    int fused_total_out = conv_channels + inner + (2 * n_heads);
    size_t fused_bytes =
        has_gdn_ ? align256(static_cast<size_t>(max_tokens) * fused_total_out * es) : align256(0);
    gdn_fused_proj_buf_ =
        make_workspace_tensor(ptr, compute_dtype_, max_tokens, fused_total_out, fused_bytes);
}

void Workspace::resize_workspace(int new_max_tokens, cudaStream_t stream) {
    if (!try_resize_workspace(new_max_tokens, stream))
        throw std::runtime_error("shared workspace: grow to " + std::to_string(new_max_tokens) +
                                 " rows failed, the forward would overrun the old buffer");
}

bool Workspace::try_resize_workspace(int new_max_tokens, cudaStream_t stream) {
    // Grow branch routes through vram_alloc/vram_free (not stream-ordered), so nothing here
    // needs `stream`. Parameter stays: eight call sites pass it, matching the stream-ordered
    // variant this branch would take if it ever goes live.
    (void)stream;
    if (new_max_tokens == shared_workspace_max_tokens_ || new_max_tokens <= 0)
        return true;
    new_max_tokens = std::min(new_max_tokens, *max_tokens_);  // never exceed init-time max

    // Phase sizes at a token count; *max_tokens_ reads as that count while they are computed.
    const auto size_phases_at = [this](int rows) {
        const int saved_max = *max_tokens_;
        *max_tokens_ = rows;
        compute_shared_sizes(rows);
        *max_tokens_ = saved_max;
    };
    size_phases_at(new_max_tokens);

    size_t new_shared = std::max({attn_shared_size_, ffn_shared_size_, moe_shared_size_, ssm_shared_size_});
    if (new_shared == 0)
        return true;

    if (new_shared > shared_workspace_size_) {
        // Only reallocate when growing, through the SAME allocator that owns the buffer
        // (allocate_shared_workspace/free_workspace via vram_alloc/vram_free). Mixing
        // cudaFreeAsync with cudaMallocAsync here would repeat AUDIT B10's invalid pair.
        // The branch is unreachable today (allocate_shared_workspace sizes at max_tokens_, this
        // clamps to it, every phase term monotone in T); fixed rather than deleted because that
        // argument depends on surrounding code, not this branch.
        const auto take = [this](size_t b) {
            return DeviceOwned<void>::allocate(vram_alloc_, b, "shared_workspace");
        };
        if (!grow_keeping_old(shared_workspace_, shared_workspace_size_, new_shared, take)) {
            IMP_LOG_ERROR("Failed to resize shared workspace to %zu bytes", new_shared);
            size_phases_at(shared_workspace_max_tokens_);  // back to the kept rows
            return false;
        }
        bump_generation();  // arena moved: every captured graph recaptures
    }
    shared_workspace_max_tokens_ = new_max_tokens;
    return true;
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
    // Return a [n_tokens, cols] view. The view may be a writer: a small-M producer stamp on buf retires.
    smallm_xq_tag_.touch(buf.data);
    return slice_rows(buf, n_tokens);
}

void GraphExecutor::ensure_logits_pinned(int total_floats) {
    if (!h_logits_pinned_.empty() && h_logits_pinned_size_ >= total_floats)
        return;
    // Move-assignment releases the previous buffer, so the grow path has no
    // free of its own to get wrong (T5b, memory/host_pinned.h).
    h_logits_pinned_ = PinnedBuffer::acquire(cuda_host_pinned_allocator(),
                                             static_cast<size_t>(total_floats) * sizeof(float));
    h_logits_pinned_size_ = h_logits_pinned_.empty() ? 0 : total_floats;
}

}  // namespace imp
