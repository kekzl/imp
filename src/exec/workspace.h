#pragma once

#include "core/tensor.h"
#include "core/qtype.h"
#include <cuda_runtime.h>
#include <cstddef>

namespace imp {

class Model;
class VRAMAllocator;
struct MoEWorkspace;

// The shared/persistent/decode scratch arena, extracted from GraphExecutor (D2,
// component 3). Owns the forward-pass scratch arena (shared + persistent
// workspace, the per-phase sizes, and the decode/prefill workspace swap state);
// the cross-cutting auxiliary-buffer hub (qscratch/moe/attn) stays on
// GraphExecutor.
//
// The arena BUFFERS (shared/persistent/decode pointers + sizes + swap state)
// are owned here. The activation/phase TENSORS that the arena carves and the
// hot path reads (hidden_/q_/moe_/ssm_*/fp32_accum_buf_/…) stay GraphExecutor
// members — the moved methods write them through pointers set in init(),
// mirroring how QuantPipeline writes the caller-owned caches. The bodies are
// otherwise byte-identical (sizing logic unchanged), so the move is
// behaviour-neutral.
//
// See docs/superpowers/specs/2026-06-08-workspace-component-design.md.
class Workspace {
public:
    // Build context, set once from GraphExecutor::init (mirrors QuantPipeline's
    // pointer-context). The pointer args are to LIVE GraphExecutor members so
    // the moved methods read/write them exactly as before (e.g. has_gdn_ is
    // still false at the first compute_shared_sizes() and true afterward).
    void init(const Model& model, VRAMAllocator& alloc, QType compute_dtype, int* max_tokens, bool use_pdl,
              MoEWorkspace& moe,
              // model-feature flags (read for phase sizing)
              const bool* has_moe, const bool* has_ssm, const bool* has_gdn, const bool* has_dense_ffn,
              const int* max_expert_eff, const int* max_logit_tokens,
              // persistent activation tensors (written by allocate/use)
              Tensor* hidden, Tensor* residual, Tensor* norm_out, Tensor* logits, void** fp32_accum_buf,
              Tensor* fp32_hidden,
              // attention phase tensors
              Tensor* q, Tensor* k, Tensor* v, Tensor* attn_out, Tensor* proj_out,
              // dense FFN phase tensors
              Tensor* gate_out, Tensor* up_out, Tensor* swiglu_out, Tensor* ffn_out,
              // SSM phase tensors
              Tensor* ssm_proj_buf, Tensor* ssm_xBC_buf, Tensor* ssm_y_buf, Tensor* ssm_z_buf,
              Tensor* ssm_out_buf, Tensor* ssm_dt_buf, Tensor* gdn_fused_proj_buf) {
        model_ = &model;
        vram_alloc_ = &alloc;
        compute_dtype_ = compute_dtype;
        max_tokens_ = max_tokens;
        use_pdl_ = use_pdl;
        moe_ = &moe;
        has_moe_ = has_moe;
        has_ssm_ = has_ssm;
        has_gdn_ = has_gdn;
        has_dense_ffn_ = has_dense_ffn;
        max_expert_eff_ = max_expert_eff;
        max_logit_tokens_ = max_logit_tokens;
        hidden_ = hidden;
        residual_ = residual;
        norm_out_ = norm_out;
        logits_ = logits;
        fp32_accum_buf_ = fp32_accum_buf;
        fp32_hidden_ = fp32_hidden;
        q_ = q;
        k_ = k;
        v_ = v;
        attn_out_ = attn_out;
        proj_out_ = proj_out;
        gate_out_ = gate_out;
        up_out_ = up_out;
        swiglu_out_ = swiglu_out;
        ffn_out_ = ffn_out;
        ssm_proj_buf_ = ssm_proj_buf;
        ssm_xBC_buf_ = ssm_xBC_buf;
        ssm_y_buf_ = ssm_y_buf;
        ssm_z_buf_ = ssm_z_buf;
        ssm_out_buf_ = ssm_out_buf;
        ssm_dt_buf_ = ssm_dt_buf;
        gdn_fused_proj_buf_ = gdn_fused_proj_buf;
    }

    // --- zero-overhead inline accessors (hot path reads these) ---
    void* shared() const { return shared_workspace_; }
    void* persistent() const { return persistent_workspace_; }
    size_t shared_size() const { return shared_workspace_size_; }
    int shared_max_tokens() const { return shared_workspace_max_tokens_; }
    int active() const { return active_workspace_; }
    bool has_decode_workspace() const { return decode_workspace_ != nullptr; }

    // --- moved lifecycle methods ---
    [[nodiscard]] bool allocate_persistent_workspace(int max_tokens);
    [[nodiscard]] bool allocate_shared_workspace(int max_tokens);
    bool allocate_decode_workspace(cudaStream_t stream, int max_batch = 1);
    void use_workspace(int slot);
    [[nodiscard]] bool resize_workspace(int new_max_tokens, cudaStream_t stream);
    void compute_shared_sizes(int max_tokens);
    size_t workspace_estimate() const;
    void configure_attn_workspace(int max_tokens);
    void configure_ffn_workspace(int max_tokens);
    void configure_moe_workspace(int max_tokens);
    void configure_ssm_workspace(int max_tokens);

private:
    // --- build context (pointers to LIVE GraphExecutor state) ---
    const Model* model_ = nullptr;
    VRAMAllocator* vram_alloc_ = nullptr;
    QType compute_dtype_ = QType::F16;
    int* max_tokens_ = nullptr;  // GraphExecutor::max_tokens_ (read + transient save/restore)
    bool use_pdl_ = false;
    MoEWorkspace* moe_ = nullptr;

    const bool* has_moe_ = nullptr;
    const bool* has_ssm_ = nullptr;
    const bool* has_gdn_ = nullptr;
    const bool* has_dense_ffn_ = nullptr;
    const int* max_expert_eff_ = nullptr;
    const int* max_logit_tokens_ = nullptr;

    // Persistent activation tensors (owned by GraphExecutor; written here).
    Tensor* hidden_ = nullptr;
    Tensor* residual_ = nullptr;
    Tensor* norm_out_ = nullptr;
    Tensor* logits_ = nullptr;
    void** fp32_accum_buf_ = nullptr;
    Tensor* fp32_hidden_ = nullptr;

    // Attention phase tensors.
    Tensor* q_ = nullptr;
    Tensor* k_ = nullptr;
    Tensor* v_ = nullptr;
    Tensor* attn_out_ = nullptr;
    Tensor* proj_out_ = nullptr;

    // Dense FFN phase tensors.
    Tensor* gate_out_ = nullptr;
    Tensor* up_out_ = nullptr;
    Tensor* swiglu_out_ = nullptr;
    Tensor* ffn_out_ = nullptr;

    // SSM phase tensors.
    Tensor* ssm_proj_buf_ = nullptr;
    Tensor* ssm_xBC_buf_ = nullptr;
    Tensor* ssm_y_buf_ = nullptr;
    Tensor* ssm_z_buf_ = nullptr;
    Tensor* ssm_out_buf_ = nullptr;
    Tensor* ssm_dt_buf_ = nullptr;
    Tensor* gdn_fused_proj_buf_ = nullptr;

    // --- owned: persistent GPU workspace (always valid, not reconfigured) ---
    void* persistent_workspace_ = nullptr;
    size_t persistent_workspace_size_ = 0;

    // --- owned: shared GPU workspace (reconfigured per layer phase) ---
    void* shared_workspace_ = nullptr;
    size_t shared_workspace_size_ = 0;
    int shared_workspace_max_tokens_ = 0;

    // Pre-computed phase sizes (for max_tokens_).
    size_t attn_shared_size_ = 0;
    size_t ffn_shared_size_ = 0;
    size_t moe_shared_size_ = 0;
    size_t ssm_shared_size_ = 0;

    // --- owned: dual workspace for concurrent prefill/decode overlap ---
    void* decode_workspace_ = nullptr;         // persistent buf for decode
    void* decode_shared_workspace_ = nullptr;  // shared buf for decode
    size_t decode_persistent_size_ = 0;
    size_t decode_shared_size_ = 0;
    int decode_max_batch_ = 1;  // max decode batch size this workspace supports
    int active_workspace_ = 0;

    // Saved prefill workspace pointers (restored when switching back).
    struct SavedWorkspace {
        void* persistent;
        size_t persistent_size;
        void* shared;
        size_t shared_size;
        int shared_max_tokens;
        Tensor hidden, residual, norm_out, logits;
        void* fp32_accum;
        Tensor fp32_hidden;
    };
    SavedWorkspace saved_prefill_ws_;
};

}  // namespace imp
