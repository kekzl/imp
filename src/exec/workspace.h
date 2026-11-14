#pragma once

#include "core/tensor.h"
#include "core/qtype.h"
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace imp {

class Model;
class VRAMAllocator;
struct MoEWorkspace;

// Shared/persistent scratch arena, extracted from GraphExecutor (D2, component
// 3): owns the forward-pass scratch arena (buffers + sizes).
// The cross-cutting auxiliary-buffer hub (qscratch/moe/attn) stays on GraphExecutor.
// PER-PHASE activation tensors are still carved by GraphExecutor's
// configure_*_workspace methods (pure slicing, they read the buffer via ws_.shared());
// only the six PERSISTENT tensors (hidden_/residual_/norm_out_/logits_/
// fp32_accum_buf_/fp32_hidden_) are written here, mirroring how QuantPipeline writes the
// caller-owned caches. Sizing logic is unchanged; the move is behaviour-neutral.
class Workspace {
public:
    // Build context, set once from GraphExecutor::init (mirrors QuantPipeline's
    // pointer-context). Pointer args point to LIVE GraphExecutor members so the moved
    // methods read/write them exactly as before (e.g. has_gdn_ is false at the first
    // compute_shared_sizes() call and true afterward).
    void init(const Model& model, VRAMAllocator* alloc, QType compute_dtype, int* max_tokens,
              MoEWorkspace& moe,
              // model-feature flags (read for phase sizing)
              const bool* has_moe, const bool* has_ssm, const bool* has_gdn, const bool* has_dense_ffn,
              const int* max_expert_eff, const int* max_logit_tokens,
              // persistent activation tensors (carved by allocate_persistent; the four
              // per-phase configure_* carvers are back on GraphExecutor, so the 16 phase
              // tensors no longer pass through here)
              Tensor* hidden, Tensor* residual, Tensor* norm_out, Tensor* logits, void** fp32_accum_buf,
              Tensor* fp32_hidden) {
        model_ = &model;
        vram_alloc_ = alloc;
        compute_dtype_ = compute_dtype;
        max_tokens_ = max_tokens;
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
    }

    // --- zero-overhead inline accessors (hot path reads these) ---
    void* shared() const { return shared_workspace_; }
    void* persistent() const { return persistent_workspace_; }
    size_t shared_size() const { return shared_workspace_size_; }
    // Persistent + shared bytes currently held; resident before the KV plan reads free VRAM (#2365).
    size_t allocated_bytes() const { return persistent_workspace_size_ + shared_workspace_size_; }
    int shared_max_tokens() const { return shared_workspace_max_tokens_; }
    // Bumped whenever shared_workspace_ is (re)allocated or freed — consumers
    // holding raw pointers into the arena (captured verify graphs, #847) use
    // this to detect that their baked addresses died.
    uint64_t generation() const { return generation_; }

    // --- moved lifecycle methods ---
    [[nodiscard]] bool allocate_persistent_workspace(int max_tokens);
    [[nodiscard]] bool allocate_shared_workspace(int max_tokens);
    [[nodiscard]] bool resize_workspace(int new_max_tokens, cudaStream_t stream);
    void compute_shared_sizes(int max_tokens);
    // include_attn_scores: charge the cuBLAS S-matrix term (capped 256 MiB, non-MoE only).
    // Pass false when FA2 serves all prefill: the allocator skips the buffer there (#932),
    // so the estimate must not hold phantom headroom for it (#943).
    size_t workspace_estimate(bool include_attn_scores) const;

    // Frees the owned shared + persistent workspace buffers (called from
    // GraphExecutor::free_buffers).
    void free_buffers();

private:
    // --- build context (pointers to LIVE GraphExecutor state) ---
    const Model* model_ = nullptr;
    VRAMAllocator* vram_alloc_ = nullptr;
    QType compute_dtype_ = QType::F16;
    int* max_tokens_ = nullptr;  // GraphExecutor::max_tokens_ (read + transient save/restore)
    MoEWorkspace* moe_ = nullptr;

    const bool* has_moe_ = nullptr;
    const bool* has_ssm_ = nullptr;
    const bool* has_gdn_ = nullptr;
    const bool* has_dense_ffn_ = nullptr;
    const int* max_expert_eff_ = nullptr;
    const int* max_logit_tokens_ = nullptr;

    // Persistent activation tensors owned by GraphExecutor, carved by
    // allocate_persistent_workspace, which stays here. The 16 per-phase activation tensors moved out with the
    // configure_*_workspace carvers (back on GraphExecutor).
    Tensor* hidden_ = nullptr;
    Tensor* residual_ = nullptr;
    Tensor* norm_out_ = nullptr;
    Tensor* logits_ = nullptr;
    void** fp32_accum_buf_ = nullptr;
    Tensor* fp32_hidden_ = nullptr;

    // --- owned: persistent GPU workspace (always valid, not reconfigured) ---
    void* persistent_workspace_ = nullptr;
    size_t persistent_workspace_size_ = 0;

    // --- owned: shared GPU workspace (reconfigured per layer phase) ---
    void* shared_workspace_ = nullptr;
    size_t shared_workspace_size_ = 0;
    int shared_workspace_max_tokens_ = 0;
    uint64_t generation_ = 1;  // see generation()

    // Pre-computed phase sizes (for max_tokens_).
    size_t attn_shared_size_ = 0;
    size_t ffn_shared_size_ = 0;
    size_t moe_shared_size_ = 0;
    size_t ssm_shared_size_ = 0;
};

}  // namespace imp
