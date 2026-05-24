// Pre-dequant Phase 4: tensor registry.
// Walks the model's WeightMap and registers each tensor's role +
// runtime location in the GraphExecutor's tensor table. Also builds
// per-layer NVFP4 device-args caches for the MoE prefill fast path.
//
// Extracted from executor_pre_dequant.cu in Phase 3 of the architecture
// refactor roadmap. See pre_dequant_internal.h for shared helpers.

#include "exec/executor.h"
#include "exec/pre_dequant_internal.h"
#include "core/logging.h"
#include "runtime/storage_planner.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdlib>
#include <string>
#include <vector>

using imp::pre_dequant_internal::borrow_payload_from_wcache;
using imp::pre_dequant_internal::infer_tier_from_wcache;

namespace imp {

void GraphExecutor::pre_dequant_phase4_tensor_registry_(
    const ModelConfig& cfg, cudaStream_t stream) {
    (void)stream;  // unused but kept for signature consistency
    // Build WeightRegistry from wcache_ contents (phase-2 shim).
    registry_.clear();
    // Explicit kind overrides t.kind which is UNKNOWN after weight_upload.cu
    // creates fresh Tensor descriptors (TensorKind is not preserved through
    // the upload code paths). Phase 5 plan-driven allocation requires kind to
    // be correct, so we pass it explicitly from the field position.
    auto register_tensor = [&](const Tensor& t, TensorKind kind) -> TensorID {
        if (!t.data)
            return kInvalidTensorID;
        StorageTier tier = infer_tier_from_wcache(wcache_, t.data);
        if (tier == StorageTier::Undefined)
            return kInvalidTensorID;
        TensorID id = registry_.reserve(kind, t.shape[0], t.ndim > 1 ? t.shape[1] : 1);
        auto& h = registry_.handle(id);
        h.primary_tier = tier;
        // Phase 5 PR #1 Commit 5.1.3.a: stash the ORIGINAL GGUF pointer + qtype
        // so weight_dispatch can fall back to the small-M dp4a path on the
        // original quant when primary_tier is an overlay (FP16/NVFP4/etc).
        // Always borrowed — never freed by registry.
        h.source_data = t.data;
        h.source_qtype = t.qtype;
        borrow_payload_from_wcache(h, wcache_, t.data);
        return id;
    };

    for (int i = 0; i < cfg.n_layers; ++i) {
        // const_cast: model_ is const Model* but the *_id fields are metadata
        // stamped exactly once here during load — safe to mutate.
        auto& L = const_cast<Model*>(model_)->layer(i);
        L.wq_id = register_tensor(L.wq, TensorKind::WQ);
        L.wk_id = register_tensor(L.wk, TensorKind::WK);
        L.wv_id = register_tensor(L.wv, TensorKind::WV);
        L.wo_id = register_tensor(L.wo, TensorKind::WO);
        L.w_gate_id = register_tensor(L.w_gate, TensorKind::W_GATE);
        L.w_up_id = register_tensor(L.w_up, TensorKind::W_UP);
        L.w_down_id = register_tensor(L.w_down, TensorKind::W_DOWN);
        // Shared-expert FFN — matches StoragePlanner enumeration from PR #38.
        L.w_gate_shared_id = register_tensor(L.w_gate_shared, TensorKind::W_GATE);
        L.w_up_shared_id = register_tensor(L.w_up_shared, TensorKind::W_UP);
        L.w_down_shared_id = register_tensor(L.w_down_shared, TensorKind::W_DOWN);
        L.ssm_in_id = register_tensor(L.ssm_in, TensorKind::SSM_IN);
        L.ssm_out_id = register_tensor(L.ssm_out, TensorKind::SSM_OUT);
        L.gdn_gate_id = register_tensor(L.gdn_gate, TensorKind::GDN_GATE);

        // Per-expert TensorIDs (Task 3.4)
        const int ne_layer = static_cast<int>(L.expert_w_gate.size());
        const int ne_up = static_cast<int>(L.expert_w_up.size());
        const int ne_down = static_cast<int>(L.expert_w_down.size());
        L.expert_gate_ids.assign(ne_layer, kInvalidTensorID);
        L.expert_up_ids.assign(ne_up, kInvalidTensorID);
        L.expert_down_ids.assign(ne_down, kInvalidTensorID);
        for (int e = 0; e < ne_layer; ++e)
            L.expert_gate_ids[e] = register_tensor(L.expert_w_gate[e], TensorKind::EXPERT_GATE);
        for (int e = 0; e < ne_up; ++e)
            L.expert_up_ids[e] = register_tensor(L.expert_w_up[e], TensorKind::EXPERT_UP);
        for (int e = 0; e < ne_down; ++e)
            L.expert_down_ids[e] = register_tensor(L.expert_w_down[e], TensorKind::EXPERT_DOWN);
        L.moe_gate_id = register_tensor(L.moe_gate, TensorKind::ROUTER);
        L.shared_expert_gate_id = register_tensor(L.shared_expert_gate_inp, TensorKind::SHARED_EXPERT_GATE);

        // Borrow nvfp4_moe pointers for packed 3D expert NVFP4 cache (Task 3.4)
        {
            auto it = wcache_.nvfp4_moe.find(L.expert_gate_packed.data);
            L.nvfp4_moe_gate_ptr = (it != wcache_.nvfp4_moe.end()) ? &it->second : nullptr;
        }
        {
            auto it = wcache_.nvfp4_moe.find(L.expert_up_packed.data);
            L.nvfp4_moe_up_ptr = (it != wcache_.nvfp4_moe.end()) ? &it->second : nullptr;
        }
        {
            auto it = wcache_.nvfp4_moe.find(L.expert_down_packed.data);
            L.nvfp4_moe_down_ptr = (it != wcache_.nvfp4_moe.end()) ? &it->second : nullptr;
        }
        // Borrow fp16 pointers for packed expert tensors (Task 3.4)
        {
            auto it = wcache_.fp16.find(L.expert_gate_packed.data);
            L.fp16_packed_gate_cache = (it != wcache_.fp16.end()) ? &it->second : nullptr;
        }
        {
            auto it = wcache_.fp16.find(L.expert_up_packed.data);
            L.fp16_packed_up_cache = (it != wcache_.fp16.end()) ? &it->second : nullptr;
        }
        {
            auto it = wcache_.fp16.find(L.expert_down_packed.data);
            L.fp16_packed_down_cache = (it != wcache_.fp16.end()) ? &it->second : nullptr;
        }
    }
    // Register model-level (non-layer) tensors.
    const_cast<Model*>(model_)->out_proj_id = register_tensor(model_->output_proj(), TensorKind::LM_HEAD);
    const_cast<Model*>(model_)->tok_emb_id = register_tensor(model_->token_embedding(),
                                                             TensorKind::TOK_EMBED);

    // Register fused KV / gate+up overlays. Layer-keyed (not pointer-keyed)
    // because a fused tensor is built fresh — the source pointers (wk, wv)
    // are the *unfused* weights and don't appear in any per-tensor wcache_ map.
    //
    // Ownership transfer (Phase 4.2): the registry handle takes ownership of
    // the GPU pointer. `h.owned_bytes` is set to the allocation size so the
    // registry destructor (`free_owned_storage`) will free it. The wcache_
    // map entry is erased after transfer so that the workspace cleanup's
    // wcache_.fused_kv loop becomes a no-op — no double-free.
    auto register_fused = [&](TensorKind kind, const Tensor& t) -> TensorID {
        if (!t.data)
            return kInvalidTensorID;
        TensorID id = registry_.reserve(kind, t.shape[0], t.ndim > 1 ? t.shape[1] : 1);
        auto& h = registry_.handle(id);
        h.primary_tier = StorageTier::FP16;
        h.payload.fp16.data = static_cast<half*>(t.data);
        h.owned_bytes = static_cast<int64_t>(t.nbytes());
        return id;
    };
    for (int i = 0; i < cfg.n_layers; ++i) {
        auto& L = const_cast<Model*>(model_)->layer(i);
        if (auto it = wcache_.fused_kv.find(i); it != wcache_.fused_kv.end()) {
            L.fused_kv_id = register_fused(TensorKind::FUSED_KV, it->second);
        }
        if (auto it = wcache_.fused_gate_up.find(i); it != wcache_.fused_gate_up.end()) {
            L.fused_gate_up_id = register_fused(TensorKind::FUSED_GATE_UP, it->second);
        }
    }
    // Transfer storage ownership: clear the wcache_.fused_kv / fused_gate_up
    // maps so the legacy cleanup loops in executor_workspace_buffers.cu find
    // them empty. The underlying pointers live on in the registry handles
    // and are freed by `registry_.free_owned_storage()` in workspace cleanup.
    wcache_.fused_kv.clear();
    wcache_.fused_gate_up.clear();

    IMP_LOG_INFO("WeightRegistry populated with %zu handles (phase-2 shim)", registry_.size());

    // Phase 4 (Option C) overlay diagnostic: report ideal vs actual overlay
    // population. The plan enumerates every quantize-able tensor at its
    // preferred tier ("ideal overlay"). The registry tracks tensors actually
    // cached by the runtime ("actual overlay"). Native GGUF blocks (Q4_K_M,
    // Q5_K_M, Q6_K, Q8_0, MXFP4) stay as mmap'd `Model::gpu_allocations_`
    // and are dequantized per kernel call — they bypass the overlay layer
    // entirely, so the diff between plan and registry is informational, not
    // an error.
    {
        StoragePlan ideal_plan = plan_storage(*model_, cfg, hints_);
        size_t plan_overlay = 0;
        size_t plan_fp16 = 0, plan_fp8 = 0, plan_nvfp4 = 0;
        size_t plan_cutlass_nvfp4 = 0, plan_mxfp4 = 0, plan_fp32 = 0;
        for (const auto& e : ideal_plan.entries) {
            switch (e.tier) {
                case StorageTier::FP16:
                    ++plan_fp16;
                    ++plan_overlay;
                    break;
                case StorageTier::FP8:
                    ++plan_fp8;
                    ++plan_overlay;
                    break;
                case StorageTier::NVFP4:
                    ++plan_nvfp4;
                    ++plan_overlay;
                    break;
                case StorageTier::CUTLASS_NVFP4:
                    ++plan_cutlass_nvfp4;
                    ++plan_overlay;
                    break;
                case StorageTier::MXFP4:
                    ++plan_mxfp4;
                    ++plan_overlay;
                    break;
                case StorageTier::FP32:
                    ++plan_fp32;
                    break;
                case StorageTier::Undefined:
                    break;
            }
        }
        size_t registry_count = registry_.size();
        IMP_LOG_INFO(
            "Phase-4 overlay: registry=%zu cached / plan-ideal=%zu "
            "(uncached %zu remain as native GGUF blocks)",
            registry_count, plan_overlay, plan_overlay > registry_count ? plan_overlay - registry_count : 0);

        // When there is a registry/plan gap, surface the by-kind delta so the
        // missing TensorKinds are immediately visible. Helps when adding a new
        // model that has tensor kinds the runtime caches but plan_storage
        // doesn't yet enumerate (or vice versa).
        if (registry_count < plan_overlay) {
            int plan_per_kind[static_cast<int>(TensorKind::_COUNT)] = {0};
            int registry_per_kind[static_cast<int>(TensorKind::_COUNT)] = {0};
            for (const auto& e : ideal_plan.entries) {
                bool overlay = (e.tier == StorageTier::FP16 || e.tier == StorageTier::FP8 ||
                                e.tier == StorageTier::NVFP4 || e.tier == StorageTier::CUTLASS_NVFP4 ||
                                e.tier == StorageTier::MXFP4);
                if (overlay)
                    ++plan_per_kind[static_cast<int>(e.kind)];
            }
            for (TensorID id = 0; id < static_cast<TensorID>(registry_.size()); ++id) {
                ++registry_per_kind[static_cast<int>(registry_.handle(id).kind)];
            }
            for (int k = 0; k < static_cast<int>(TensorKind::_COUNT); ++k) {
                int diff = plan_per_kind[k] - registry_per_kind[k];
                if (diff > 0) {
                    IMP_LOG_INFO("Phase-4 gap by kind: %s plan=%d registry=%d (uncached=%d)",
                                 tensor_kind_name(static_cast<TensorKind>(k)), plan_per_kind[k],
                                 registry_per_kind[k], diff);
                }
            }
        }
        IMP_LOG_INFO(
            "Phase-4 plan-ideal tiers: fp16=%zu fp8=%zu nvfp4=%zu "
            "cutlass_nvfp4=%zu mxfp4=%zu fp32=%zu",
            plan_fp16, plan_fp8, plan_nvfp4, plan_cutlass_nvfp4, plan_mxfp4, plan_fp32);
        IMP_LOG_INFO(
            "Phase-4 wcache actual: fp16=%zu fp8=%zu nvfp4=%zu "
            "cutlass_nvfp4=%zu cutlass_mxfp4=%zu nvfp4_moe=%zu "
            "fused_kv=%zu fused_gate_up=%zu",
            wcache_.fp16.size(), wcache_.fp8.size(), wcache_.nvfp4.size(), wcache_.cutlass_nvfp4.size(),
            wcache_.cutlass_mxfp4.size(), wcache_.nvfp4_moe.size(), wcache_.fused_kv.size(),
            wcache_.fused_gate_up.size());
        // Native layer counterpart to the overlay diagnostic: tensors uploaded
        // as their on-disk format and dispatched through qtype-specific kernels
        // (no tier choice, no cascade-bug class). gpu_allocations_ tracks every
        // GPU pointer the Model owns — Q4_K_M / Q5_K_M / Q6_K / Q8_0 / MXFP4
        // blocks, norms, embeddings, scratch buffers. Together with the overlay
        // counts above this gives the full Option-C two-layer storage picture.
        IMP_LOG_INFO(
            "Phase-4 native: %zu Model::gpu_allocations_ pointers "
            "(GGUF blocks + norms + scratch — bypass the overlay layer)",
            model_->gpu_allocations_.size());

        // Phase 5 PR #1 Commit 5.1.4.a: log how many bytes of original GGUF
        // are REDUNDANT — fully covered by the overlay tier such that the
        // dispatch never reads `source_data`. Diagnostic-only here; actual
        // freeing (5.1.4.b) requires dispatch-site safety guards.
        //
        // Qualifying tiers: NVFP4 / CUTLASS_NVFP4 / FP8 / MXFP4 (decode-fast
        // kernels exist for the overlay). FP16-cached weights still need the
        // original for dp4a decode (per 5.1.3.c) and are not counted.
        {
            size_t droppable_count = 0;
            size_t droppable_bytes = 0;
            for (TensorID id = 0; id < static_cast<TensorID>(registry_.size()); ++id) {
                const auto& h = registry_.handle(id);
                if (!h.can_drop_source())
                    continue;
                int64_t cols = h.shape[1] > 0 ? h.shape[1] : 1;
                size_t row_bytes = qtype_row_bytes(h.source_qtype, cols);
                size_t bytes = row_bytes * static_cast<size_t>(h.shape[0]);
                droppable_bytes += bytes;
                ++droppable_count;
            }
            IMP_LOG_INFO(
                "Phase-4 drop-source diagnostic: %zu handles, %.2f MiB of original "
                "GGUF could be freed (overlay tier covers prefill + decode). "
                "Actual freeing deferred to Commit 5.1.4.b.",
                droppable_count, droppable_bytes / (1024.0 * 1024.0));
        }
    }

    // -----------------------------------------------------------------------
    // Phase 3c-full Step 3: pre-cache per-layer NVFP4 device-args ptr arrays.
    // -----------------------------------------------------------------------
    // The CUTLASS 3.x device-args dispatch (Phase 3c-full Step 2b, opt-in via
    // IMP_NVFP4_DEVICE_ARGS=1) consumes per-expert weight pointers as
    // device-resident arrays. Per-call host iteration + 3× cudaMemcpyAsync
    // (~3 KiB total) was the residual overhead blocking full CUDA-graph
    // capture of the MoE prefill. Build the caches once here while the
    // handle payloads are guaranteed populated; the forward path then uses
    // the device pointers directly.
    //
    // Conditions: model is MoE (ne > 0) and at least one layer has all three
    // projections backed by CUTLASS NVFP4 handles (post-Phase-3 setup).
    {
        const int ne = cfg.n_experts;
        const int n_layers = cfg.n_layers;
        if (ne > 0) {
        moe_.per_layer_da_cache.assign(n_layers, MoEWorkspace::PerLayerNvfp4DeviceArgsCache{});

        std::vector<const void*> h_B_ptrs(ne), h_SFB_ptrs(ne);
        std::vector<float>       h_alpha(ne);
        bool any_built = false;

        auto build_proj =
            [&](const std::vector<TensorID>& ids, const void**& d_B,
                const void**& d_SFB, float*& d_alpha) -> bool {
                if (static_cast<int>(ids.size()) != ne)
                    return false;
                for (int e = 0; e < ne; ++e) {
                    if (ids[e] == kInvalidTensorID)
                        return false;
                    const auto& h = registry_.handle(ids[e]);
                    if (!h.payload.cutlass_nvfp4.weight ||
                        !h.payload.cutlass_nvfp4.sf)
                        return false;
                    h_B_ptrs[e]   = h.payload.cutlass_nvfp4.weight;
                    h_SFB_ptrs[e] = h.payload.cutlass_nvfp4.sf;
                    h_alpha[e]    = h.payload.cutlass_nvfp4.global_scale
                                        ? *h.payload.cutlass_nvfp4.global_scale
                                        : 1.0f;
                }
                cudaError_t err;
                err = cudaMalloc(&d_B,   ne * sizeof(const void*)); if (err != cudaSuccess) return false;
                err = cudaMalloc(&d_SFB, ne * sizeof(const void*)); if (err != cudaSuccess) return false;
                err = cudaMalloc(&d_alpha, ne * sizeof(float));     if (err != cudaSuccess) return false;
                cudaMemcpy(const_cast<void**>(d_B),   h_B_ptrs.data(),
                           ne * sizeof(const void*), cudaMemcpyHostToDevice);
                cudaMemcpy(const_cast<void**>(d_SFB), h_SFB_ptrs.data(),
                           ne * sizeof(const void*), cudaMemcpyHostToDevice);
                cudaMemcpy(d_alpha, h_alpha.data(),
                           ne * sizeof(float),       cudaMemcpyHostToDevice);
                return true;
            };

        int eligible_layers = 0;  // layers that have any MoE expert ptrs
        int built_layers = 0;
        int host_resident_layers = 0;  // intentionally not built (force_host or budget offload)
        std::vector<int> failed_layers;
        for (int li = 0; li < n_layers; ++li) {
            const auto& L = model_->layer(li);
            auto& c = moe_.per_layer_da_cache[li];
            const bool moe_layer = !L.expert_up_ids.empty() || !L.expert_down_ids.empty() ||
                                   !L.expert_gate_ids.empty();
            if (!moe_layer) {
                // Pure dense layer in a hybrid model (e.g. attention-only layer
                // alongside MoE layers). Not eligible for the da_cache; skip
                // without counting against the must-populate gate.
                continue;
            }
            // Host-resident layers (host-offload / force_host_experts) by
            // design have no CUTLASS NVFP4 weight payload — the per-layer
            // fallback dispatch is the intended path. Don't count these as
            // QW8 build failures. Detect via either packed-tensor (GGUF Path A)
            // or per-expert tensor (SafeTensors Path B) staying on host.
            const bool packed_host = (L.expert_up_packed.data && !L.expert_up_packed.on_device);
            bool per_expert_host = false;
            if (!packed_host && !L.expert_w_up.empty()) {
                // Any per-expert weight on host => layer is host-resident.
                for (const auto& w : L.expert_w_up) {
                    if (w.data && !w.on_device) { per_expert_host = true; break; }
                }
            }
            if (packed_host || per_expert_host) {
                ++host_resident_layers;
                continue;
            }
            ++eligible_layers;
            bool g_ok = !L.expert_gate_ids.empty() &&
                        build_proj(L.expert_gate_ids, c.d_gate_B_ptrs,
                                   c.d_gate_SFB_ptrs, c.d_gate_alpha);
            bool u_ok = build_proj(L.expert_up_ids, c.d_up_B_ptrs,
                                   c.d_up_SFB_ptrs, c.d_up_alpha);
            bool d_ok = build_proj(L.expert_down_ids, c.d_down_B_ptrs,
                                   c.d_down_SFB_ptrs, c.d_down_alpha);
            // For non-gated experts (e.g. Gemma-4 SwiGLU absorbing W_gate),
            // expert_gate_ids may be empty by design — accept up+down only.
            c.ready = (g_ok || L.expert_gate_ids.empty()) && u_ok && d_ok;
            if (c.ready) {
                ++built_layers;
                any_built = true;
            } else {
                failed_layers.push_back(li);
            }
        }
        // QW8 from review/phase5_synthesis.md §2.1: hard-fail (not log-INFO)
        // when the NVFP4 da_cache populates <100% of MoE-eligible layers.
        // Partial coverage means the per-layer fallback fires for the missing
        // layers and decode silently regresses ~5× on Qwen3-Coder / Gemma-4
        // NVFP4 (per moe_prefill_graphs_plan_2026_05_10 + cuda_graphs_moe_works).
        // A partial build is almost always a load-time symptom of a
        // mismatched expert layout or a budget that fell short of needed
        // device allocations — the right response is to fail loud at init
        // rather than ship the user a slow build.
        // Only abort on *partial* coverage of device-resident MoE layers
        // (the genuine "silent 5× regression" case). If nothing built at
        // all, this model isn't going through the NVFP4 MoE da_cache path
        // at runtime — log INFO and continue (covers --no-nvfp4 on GGUF
        // MoE, Q4_K_M / Q6_K MoE without prequant scales, and synthetic
        // force_host_experts spikes).
        if (eligible_layers > 0 && built_layers > 0 && built_layers < eligible_layers) {
            std::string failed_str;
            for (size_t i = 0; i < failed_layers.size() && i < 16; ++i) {
                if (!failed_str.empty()) failed_str += ", ";
                failed_str += std::to_string(failed_layers[i]);
            }
            if (failed_layers.size() > 16) failed_str += ", …";
            IMP_LOG_FATAL(
                "NVFP4 da_cache: only %d/%d MoE-eligible layers populated. "
                "Failing layers: [%s]. Partial coverage forces the per-layer "
                "fallback dispatch (~5× slower decode on NVFP4 MoE models). "
                "Likely cause: missing expert_*_ids, invalid CutlassNvFP4 "
                "weight handles, or cudaMalloc failure for the per-layer "
                "ptr arrays. Aborting before the engine silently ships a "
                "slow build.",
                built_layers, eligible_layers, failed_str.c_str());
            std::abort();
        }
        if (any_built) {
            IMP_LOG_INFO(
                "Pre-cached per-layer NVFP4 device-args ptr arrays for "
                "%d/%d layers × 3 projections × %d experts (~%.1f KiB)",
                built_layers, eligible_layers, ne,
                (built_layers * 3.0 * ne * (2 * sizeof(void*) + sizeof(float))) /
                    1024.0);
        }
        if (host_resident_layers > 0) {
            IMP_LOG_INFO(
                "NVFP4 da_cache: skipped %d host-resident MoE layer(s) "
                "(per-layer fallback / H2D staging is the intended path).",
                host_resident_layers);
        }
        }  // ne > 0
    }
}

}  // namespace imp
