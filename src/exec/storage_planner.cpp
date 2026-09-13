#include "exec/storage_planner.h"
#include "model/tensor_kind_table.h"
#include "model/model.h"
#include "model/model_config.h"
#include "core/tensor.h"

#include <algorithm>
#include <utility>

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

// The plan must budget the INCREMENTAL cost of reaching a tier, not its full footprint.
// A native-NVFP4 source already holds the packed nibbles + micro-scales on device (Phase
// 0b registers zero-copy); only the Phase 3b CUTLASS SfAtom repack (~n/16) allocates.
// Pricing the full tier bytes projected phantom demand on native checkpoints, making a
// real insufficiency indistinguishable from the normal case (#1765).
int64_t incremental_bytes_for_tier(int64_t rows, int64_t cols, StorageTier tier, QType source) {
    if (source == QType::NVFP4) {
        if (tier == StorageTier::NVFP4)
            return 0;  // decode cache borrows the resident source storage
        if (tier == StorageTier::CUTLASS_NVFP4)
            return rows * cols / 16;  // SfAtom sidecar; nibbles stay shared
    }
    // An F16/F32-source tensor at the FP16 tier is the resident upload itself: Phase 1 only
    // builds FP16 cache copies for dequantable (or native-FP8) sources, so nothing new is
    // allocated here. Without this the token embedding alone kept the budget check failing
    // on native checkpoints after the NVFP4 entries were priced honestly.
    if ((source == QType::F16 || source == QType::BF16 || source == QType::F32) &&
        tier == StorageTier::FP16)
        return 0;
    return bytes_for_tier(rows, cols, tier);
}

// Pick the initial (best allowable) tier for a tensor given its (source-qtype-refined)
// capabilities and hints. Hints can only push toward a tier the refined capabilities
// still list as supported: that's how a Q4_K-source W_GATE correctly stays FP16 even
// with prefer_nvfp4_decode=true.
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

    // prefer_nvfp4_decode: pick NVFP4 only if the refined capabilities still list it. For
    // Q4_K sources effective_capabilities strips NVFP4, so the hint falls through to
    // required_floor (FP16). Structural fix for the Q4_K coverage-gap bug.
    if (hints.prefer_nvfp4_decode && mask_contains(cap.supported, StorageTier::NVFP4))
        return StorageTier::NVFP4;

    return cap.required_floor;
}

// Return the next-smaller (more compressed) supported tier after current, never below
// floor; returns current if none exists. StorageTier order: FP32=1, FP16=2, FP8=3,
// NVFP4=4, CUTLASS_NVFP4=5, MXFP4=6 - higher integer = more compressed, so "downgrade"
// means increasing the enum value.
StorageTier downgrade_one(StorageTier current, StorageTier floor, const KindCapabilities& cap) {
    for (int s = std::to_underlying(current) + 1; s <= std::to_underlying(StorageTier::MXFP4); ++s) {
        auto candidate = static_cast<StorageTier>(s);
        if (!mask_contains(cap.supported, candidate))
            continue;
        // Only downgrade if the candidate is at or below the floor in compression. floor is the
        // required minimum quality (least-compressed tier we must stay at); since higher
        // integer = more compressed, downgrade always moves toward more compression, so no
        // ceiling check is needed here.
        (void)floor;  // floor enforced by the caller (skip if tier==required_floor)
        return candidate;
    }
    return current;
}

// Explicit kind overrides t.kind, which is UNKNOWN after weight_upload.cu creates fresh
// Tensor descriptors. The planner uses field position (L.wq -> WQ, etc.) rather than the
// stored kind, so Phase 5 plan-driven allocation works even before kind preservation is
// added to every upload path. t.qtype IS preserved, so the planner uses it for
// source-qtype-aware capability refinement (effective_capabilities).
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
    int64_t bytes = incremental_bytes_for_tier(rows, cols, tier, t.qtype);

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
        // gdn_gate is intentionally NOT enumerated for overlay caching: it's consumed only by
        // the specialized GDN scan kernel via the raw L.gdn_gate.data pointer, never through
        // gemm_dispatch. An overlay copy would burn VRAM with no consumer (see PR #43's
        // per-kind gap diagnostic).
        for (const auto& e : L.expert_w_gate)
            add_tensor(e, TensorKind::EXPERT_GATE, plan, next_id, total, hints);
        for (const auto& e : L.expert_w_up)
            add_tensor(e, TensorKind::EXPERT_UP, plan, next_id, total, hints);
        for (const auto& e : L.expert_w_down)
            add_tensor(e, TensorKind::EXPERT_DOWN, plan, next_id, total, hints);
    }

    // Top-level (model-global) tensors: embeddings and LM head have their own tier choices
    // and must not be omitted from the plan; the future PlanExecutor owns their GPU storage
    // allocation too.
    add_tensor(model.token_embedding(), TensorKind::TOK_EMBED, plan, next_id, total, hints);
    add_tensor(model.output_proj(), TensorKind::LM_HEAD, plan, next_id, total, hints);

    // Budget satisfaction: iteratively downgrade the entry with the highest bytes-saved
    // potential until the total fits or everything is at required_floor. Uses
    // effective_capabilities(kind, source_qtype) so Q4_K-source tensors can't be downgraded
    // to NVFP4 (no compression win, possible quality risk).
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
                int64_t new_bytes = incremental_bytes_for_tier(e.rows, e.cols, next, e.source_qtype);
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
                int64_t new_bytes = incremental_bytes_for_tier(e.rows, e.cols, next, e.source_qtype);
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

}  // namespace imp
