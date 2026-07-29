#include "exec/workspace_sizes.h"

#include "model/model.h"

#include <algorithm>

namespace imp {

namespace {

// Logical (unpacked) K of a weight tensor. NVFP4 stores two e2m1 values per
// byte, so shape[...] is the PACKED byte count and the dequant target is twice
// as wide — the same distinction allocate_nvfp4_dequant_workspace makes.
struct Dims {
    int64_t n = 0;
    int64_t k = 0;
    bool valid() const { return n > 0 && k > 0; }
};

Dims weight_dims(const Tensor& t) {
    if (!t.data || t.ndim < 2)
        return {};
    const bool packed = (t.qtype == QType::NVFP4);
    if (t.ndim >= 3)  // [n_experts, N, K] — one expert is the dequant unit
        return {t.shape[1], t.shape[2] * (packed ? 2 : 1)};
    return {t.shape[0], t.shape[1] * (packed ? 2 : 1)};
}

// The tensor set the MMVQ / NVFP4-dequant paths can see. Deliberately the same
// list compute_native_cache_demand() scans (runtime/vram_budget.cpp), so the
// two cannot drift apart on a model where one of them matters.
template <class F>
void for_each_weight(const Model& model, F&& f) {
    f(model.output_proj());
    const int n = model.n_layers();
    for (int i = 0; i < n; ++i) {
        const auto& L = model.layer(i);
        f(L.wq); f(L.wk); f(L.wv); f(L.wo);
        f(L.w_gate); f(L.w_up); f(L.w_down);
        f(L.w_gate_shared); f(L.w_up_shared); f(L.w_down_shared);
        f(L.ssm_in); f(L.ssm_out); f(L.gdn_gate);
        if (L.expert_gate_packed.data)
            f(L.expert_gate_packed);
        else if (!L.expert_w_gate.empty())
            f(L.expert_w_gate[0]);
        if (L.expert_up_packed.data)
            f(L.expert_up_packed);
        else if (!L.expert_w_up.empty())
            f(L.expert_w_up[0]);
        if (L.expert_down_packed.data)
            f(L.expert_down_packed);
        else if (!L.expert_w_down.empty())
            f(L.expert_w_down[0]);
    }
}

}  // namespace

int exec_max_tokens(const Model& model, int max_seq_len) {
    const auto& cfg = model.config();
    int effective = (max_seq_len > 0) ? max_seq_len : cfg.max_seq_len;
    int t = std::min(effective, 4096);
    if (t <= 0)
        t = 4096;
    // AS-BUILT condition, not the intended one — executor_workspace.cu reads
    // has_gdn_ before it is assigned, so this fires for SSM+MoE only. Matching
    // the intent here would under-reserve by 2x on those models (AUDIT B18).
    const auto& prof = model.profile();
    if (prof.is_ssm && prof.is_moe)
        t = std::min(t, 2048);
    return t;
}

int exec_max_weight_k(const Model& model) {
    int64_t max_k = 0;
    for_each_weight(model, [&](const Tensor& t) {
        const Dims d = weight_dims(t);
        if (d.valid())
            max_k = std::max(max_k, d.k);
    });
    return static_cast<int>(max_k);
}

ExecT2Demand exec_t2_demand(const Model& model, int max_seq_len) {
    ExecT2Demand out;
    const int t = exec_max_tokens(model, max_seq_len);
    const int max_k = exec_max_weight_k(model);
    if (t <= 0 || max_k <= 0)
        return out;

    // gemm_scratch.cu: per_call = max_tokens * ceil(K/32) * 36, need = 2x.
    out.mmvq_scratch = static_cast<size_t>(t) * ((static_cast<size_t>(max_k) + 31) / 32) * 36 * 2;

    // executor_workspace_buffers.cu: the largest dequant target at or below the
    // 512 MiB cap. Targets above the cap are served by the uncapturable path,
    // so they must NOT raise the reservation.
    constexpr size_t kCap = 512ULL * 1024 * 1024;
    size_t covered = 0;
    auto consider = [&](int64_t n, int64_t k) {
        if (n <= 0 || k <= 0)
            return;
        const size_t bytes = static_cast<size_t>(n) * static_cast<size_t>(k) * 2;
        if (bytes <= kCap)
            covered = std::max(covered, bytes);
    };
    for_each_weight(model, [&](const Tensor& tw) {
        const Dims d = weight_dims(tw);
        consider(d.n, d.k);
    });

    // Config-derived shapes, because this runs BEFORE the weight upload and
    // some checkpoints do not carry their final layout yet. gpt-oss is the
    // case that forced this: its experts arrive as
    // expert_gate_up_packed_blocks, a 4D U8 [ne, 2*d_ff, K/32, 16] slot that
    // the upload consumes, so the tensor scan sees nothing resembling the
    // dequant target. The real target is one expert's FUSED gate_up —
    // 2*expert_d_ff x d_model — which is exactly the 31.64 MiB the workspace
    // asked for and was refused (AUDIT B23). Deriving it from the config
    // instead of the tensors makes the reservation independent of when it runs.
    const auto& cfg = model.config();
    if (cfg.n_experts > 0) {
        const int64_t eff = cfg.expert_d_ff > 0 ? cfg.expert_d_ff : cfg.d_ff;
        consider(2 * eff, cfg.d_model);  // fused gate_up
        consider(eff, cfg.d_model);      // gate / up / down individually
        consider(cfg.d_model, eff);
    }
    if (cfg.d_ff > 0) {
        consider(2 * static_cast<int64_t>(cfg.d_ff), cfg.d_model);
        consider(cfg.d_ff, cfg.d_model);
        consider(cfg.d_model, cfg.d_ff);
    }

    out.nvfp4_dequant = covered;
    return out;
}

}  // namespace imp
