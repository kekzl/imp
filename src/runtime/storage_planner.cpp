#include "runtime/storage_planner.h"
#include "model/tensor_kind_table.h"
#include "model/model.h"
#include "model/model_config.h"
#include "core/tensor.h"

#include <algorithm>

namespace imp {

namespace {

int64_t bytes_for_tier(int64_t rows, int64_t cols, StorageTier tier) {
    int64_t n = rows * cols;
    switch (tier) {
        case StorageTier::FP32:
            return n * 4;
        case StorageTier::FP16:
            return n * 2;
        case StorageTier::FP8:
            return n + 4;  // packed bytes + per-tensor scale
        case StorageTier::NVFP4:
            return n / 2 + n / 16;  // packed FP4 + micro-scales
        case StorageTier::CUTLASS_NVFP4:
            return n / 2 + n / 16;
        case StorageTier::MXFP4:
            return n / 2 + n / 32;
        case StorageTier::Undefined:
            return 0;
    }
    return 0;
}

// Pick the initial (best allowable) tier for a tensor given its (source-qtype-
// refined) capabilities and the hints. Hints can only push toward a tier that
// the refined capabilities still list as supported — that's how the Q4_K-
// source W_GATE case correctly stays FP16 even when `prefer_nvfp4_decode=true`.
StorageTier pick_initial_tier(TensorKind kind, const KindCapabilities& cap, const PlanHints& hints) {
    // dual_path hint: attention projections prefer FP8; FFN prefer NVFP4.
    if (hints.dual_path_attn_fp8_ffn_nvfp4) {
        const bool is_attn_proj = (kind == TensorKind::WQ || kind == TensorKind::WK ||
                                   kind == TensorKind::WV || kind == TensorKind::WO);
        const bool is_ffn_proj = (kind == TensorKind::W_GATE || kind == TensorKind::W_UP ||
                                  kind == TensorKind::W_DOWN || kind == TensorKind::EXPERT_GATE ||
                                  kind == TensorKind::EXPERT_UP || kind == TensorKind::EXPERT_DOWN);
        if (is_attn_proj && mask_contains(cap.supported, StorageTier::FP8))
            return StorageTier::FP8;
        if (is_ffn_proj && mask_contains(cap.supported, StorageTier::NVFP4))
            return StorageTier::NVFP4;
    }

    // prefer_nvfp4_decode: pick NVFP4 only if the (refined) capabilities still
    // list it. For Q4_K sources `effective_capabilities` stripped NVFP4, so the
    // hint silently falls through to required_floor (FP16). That's the
    // structural fix for the 2026-05-24 Q4_K coverage-gap bug.
    if (hints.prefer_nvfp4_decode && mask_contains(cap.supported, StorageTier::NVFP4))
        return StorageTier::NVFP4;

    return cap.required_floor;
}

// Return the next-smaller (more compressed) supported tier after `current`,
// never going below `floor`. Returns `current` if no such tier exists.
// StorageTier enum order: FP32=1, FP16=2, FP8=3, NVFP4=4, CUTLASS_NVFP4=5, MXFP4=6
// Higher integer = more compressed, so "downgrade" = increase enum value.
StorageTier downgrade_one(StorageTier current, StorageTier floor, const KindCapabilities& cap) {
    for (int s = static_cast<int>(current) + 1; s <= static_cast<int>(StorageTier::MXFP4); ++s) {
        auto candidate = static_cast<StorageTier>(s);
        if (!mask_contains(cap.supported, candidate))
            continue;
        // Only downgrade if the candidate is at or below the floor in compression.
        // floor is the *required* minimum quality, i.e. the least compressed tier
        // we must stay at. Since higher integer = more compressed, the floor
        // constraint means candidate >= floor (we can go more compressed than floor,
        // not less). We never need to enforce a ceiling here — downgrade always
        // moves toward more compression.
        (void)floor;  // floor enforced by the caller (skip if tier==required_floor)
        return candidate;
    }
    return current;
}

// Explicit kind overrides t.kind, which is UNKNOWN after weight_upload.cu
// creates fresh Tensor descriptors. The planner uses the field position
// (L.wq → WQ, L.wk → WK, …) rather than the stored kind so that Phase 5
// plan-driven allocation works correctly even before kind preservation is
// added to every upload code path.
//
// `t.qtype` IS preserved across weight_upload, so the planner uses it for
// source-qtype-aware capability refinement via `effective_capabilities`.
void add_tensor(const Tensor& t, TensorKind kind, StoragePlan& plan, TensorID& next_id, size_t& total,
                const PlanHints& hints) {
    if (!t.data)
        return;
    if (kind == TensorKind::UNKNOWN)
        return;  // skip unclassified tensors

    const auto cap = effective_capabilities(kind, t.qtype);
    StorageTier tier = pick_initial_tier(kind, cap, hints);
    // Clamp to supported: if pick_initial_tier returned something unsupported,
    // fall back to required_floor.
    if (!mask_contains(cap.supported, tier))
        tier = cap.required_floor;

    int64_t rows = (t.ndim > 0 ? t.shape[0] : 1);
    int64_t cols = (t.ndim > 1 ? t.shape[1] : 1);
    int64_t bytes = bytes_for_tier(rows, cols, tier);

    plan.entries.push_back({next_id++, kind, t.qtype, tier, bytes, rows, cols, t.data, false});
    total += static_cast<size_t>(bytes);
}

}  // namespace

StoragePlan plan_storage(const Model& model, const ModelConfig& cfg, const PlanHints& hints) {
    StoragePlan plan;
    TensorID next_id = 0;
    size_t total = 0;

    int n_layers = cfg.n_layers;
    // If the model has more layers than cfg.n_layers, iterate over all of them.
    // In synthetic test models, layers_ is populated directly so we use the
    // larger of the two.
    if (model.n_layers() > n_layers)
        n_layers = model.n_layers();

    for (int i = 0; i < n_layers; ++i) {
        const auto& L = model.layer(i);
        add_tensor(L.wq, TensorKind::WQ, plan, next_id, total, hints);
        add_tensor(L.wk, TensorKind::WK, plan, next_id, total, hints);
        add_tensor(L.wv, TensorKind::WV, plan, next_id, total, hints);
        add_tensor(L.wo, TensorKind::WO, plan, next_id, total, hints);
        // MLA (DeepSeek-V2/V3) latent projections. add_tensor skips null tensors,
        // so non-MLA models (empty kv_a/kv_b fields) are no-ops here.
        add_tensor(L.kv_a_proj, TensorKind::KV_A_PROJ, plan, next_id, total, hints);
        add_tensor(L.kv_a_layernorm, TensorKind::KV_A_NORM, plan, next_id, total, hints);
        add_tensor(L.kv_b_proj, TensorKind::KV_B_PROJ, plan, next_id, total, hints);
        add_tensor(L.w_gate, TensorKind::W_GATE, plan, next_id, total, hints);
        add_tensor(L.w_up, TensorKind::W_UP, plan, next_id, total, hints);
        add_tensor(L.w_down, TensorKind::W_DOWN, plan, next_id, total, hints);
        // Shared-expert FFN (Nemotron / DeepSeek / Qwen3.5-MoE). Same kinds as
        // the regular FFN projections — capabilities and tier choice mirror.
        add_tensor(L.w_gate_shared, TensorKind::W_GATE, plan, next_id, total, hints);
        add_tensor(L.w_up_shared, TensorKind::W_UP, plan, next_id, total, hints);
        add_tensor(L.w_down_shared, TensorKind::W_DOWN, plan, next_id, total, hints);
        add_tensor(L.ssm_in, TensorKind::SSM_IN, plan, next_id, total, hints);
        add_tensor(L.ssm_out, TensorKind::SSM_OUT, plan, next_id, total, hints);
        // gdn_gate is intentionally NOT enumerated for overlay caching: it is
        // consumed only by the specialized GDN scan kernel (gdn_kernel.cu) via
        // the raw `L.gdn_gate.data` pointer, never through `gemm_dispatch`. An
        // overlay copy would burn VRAM with no consumer. The diagnostic in
        // pre_dequant_weights would otherwise (correctly) flag a 24-handle
        // "gap" on every GDN model — see commit 3c7803a for the discovery and
        // PR #43 for the per-kind gap diagnostic that surfaced this.
        for (const auto& e : L.expert_w_gate)
            add_tensor(e, TensorKind::EXPERT_GATE, plan, next_id, total, hints);
        for (const auto& e : L.expert_w_up)
            add_tensor(e, TensorKind::EXPERT_UP, plan, next_id, total, hints);
        for (const auto& e : L.expert_w_down)
            add_tensor(e, TensorKind::EXPERT_DOWN, plan, next_id, total, hints);
    }

    // Top-level (model-global) tensors. Embeddings and LM head have their own
    // tier choices (LM head is NVFP4-prequant on Qwen3-Coder-30B-FP4) and must
    // not be omitted from the plan — the future PlanExecutor owns their GPU
    // storage allocation too.
    add_tensor(model.token_embedding(), TensorKind::TOK_EMBED, plan, next_id, total, hints);
    add_tensor(model.output_proj(), TensorKind::LM_HEAD, plan, next_id, total, hints);

    // Budget satisfaction: iteratively downgrade the entry with the highest
    // bytes-saved potential until we fit or everything is at required_floor.
    // Uses effective_capabilities(kind, source_qtype) so Q4_K-source tensors
    // can't be downgraded to NVFP4 (no compression win, possible quality risk).
    if (hints.vram_budget_bytes > 0 && total > hints.vram_budget_bytes) {
        bool progress = true;
        while (total > hints.vram_budget_bytes && progress) {
            progress = false;
            // Find the entry with the most bytes that can still be downgraded.
            size_t best_idx = plan.entries.size();
            int64_t best_savings = 0;
            for (size_t idx = 0; idx < plan.entries.size(); ++idx) {
                auto& e = plan.entries[idx];
                const auto cap = effective_capabilities(e.kind, e.source_qtype);
                if (e.tier == cap.required_floor)
                    continue;
                StorageTier next = downgrade_one(e.tier, cap.required_floor, cap);
                if (next == e.tier)
                    continue;
                int64_t new_bytes = bytes_for_tier(e.rows, e.cols, next);
                int64_t savings = e.bytes - new_bytes;
                if (savings > best_savings) {
                    best_savings = savings;
                    best_idx = idx;
                }
            }
            if (best_idx < plan.entries.size() && best_savings > 0) {
                auto& e = plan.entries[best_idx];
                const auto cap = effective_capabilities(e.kind, e.source_qtype);
                StorageTier next = downgrade_one(e.tier, cap.required_floor, cap);
                int64_t new_bytes = bytes_for_tier(e.rows, e.cols, next);
                total -= static_cast<size_t>(e.bytes - new_bytes);
                e.bytes = new_bytes;
                e.tier = next;
                progress = true;
            }
        }
    }

    plan.projected_vram_bytes = total;
    if (hints.vram_budget_bytes > 0 && total > hints.vram_budget_bytes) {
        plan.failed = true;
        plan.failure_reason = "vram budget insufficient even at required_floor tiers";
    }
    return plan;
}

void StoragePlan::build_index_() const {
    by_src_.clear();
    by_src_.reserve(entries.size());
    for (const auto& e : entries) {
        if (e.source_data)
            by_src_[e.source_data] = &e;
    }
    index_built_ = true;
}

const StoragePlan::Entry* StoragePlan::entry_of(const void* src) const {
    if (!src)
        return nullptr;
    if (!index_built_)
        build_index_();
    auto it = by_src_.find(src);
    return it == by_src_.end() ? nullptr : it->second;
}

StorageTier StoragePlan::tier_of(const void* src) const {
    const Entry* e = entry_of(src);
    return e ? e->tier : StorageTier::Undefined;
}

}  // namespace imp
