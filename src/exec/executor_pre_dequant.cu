// Pre-dequant orchestrator.
// Dispatches the six pre-dequant phases (0/0b/1/2/3/3c/4) to their
// extracted translation units. Each phase lives in src/exec/pre_dequant_phase*.cu.
//
// Adding a new phase: write one new src/exec/pre_dequant_phase*.cu file,
// add it to CMakeLists.txt IMP_EXEC_SOURCES, declare the method on
// QuantPipeline in quant_pipeline.h, and call it from QuantPipeline::build()
// below.

#include "exec/executor.h"
#include "memory/mem_account.h"
#include "memory/vram_query.h"
#include "exec/pre_dequant_internal.h"
#include "exec/quant_pipeline.h"
#include "core/logging.h"
#include "runtime/storage_planner.h"

#include <cuda_runtime.h>
#include <algorithm>

namespace imp {

// Delegate: the init-time quantization pipeline lives in QuantPipeline. The
// engine call site (engine_kv_cache_init.cpp) is unchanged. The four long-lived
// caches + moe_ stay owned by GraphExecutor and are filled by reference; the
// forward hot path reads them exactly as before (byte-identical).
void GraphExecutor::pre_dequant_weights(cudaStream_t stream, const VRAMBudget& budget) {
    if (!initialized_ || !model_)
        return;
    quant_pipeline_.build(*model_, runtime_config(), *vram_alloc_, budget, stream,
                          wcache_, qscratch_, registry_, hints_, moe_, max_tokens_);
}

void QuantPipeline::build(const Model& model, const RuntimeConfig& rcfg, VRAMAllocator& alloc,
                          const VRAMBudget& budget, cudaStream_t stream, WeightCaches& wcache,
                          QuantScratch& qscratch, WeightRegistry& registry, PlanHints& hints,
                          MoEWorkspace& moe, int max_tokens) {
    model_ = &model; runtime_config_ = &rcfg; vram_alloc_ = &alloc;
    wcache_ = &wcache; qscratch_ = &qscratch; registry_ = &registry;
    hints_ = &hints; moe_ = &moe; max_tokens_ = max_tokens;
    // `budget` and `stream` are threaded explicitly through every phase call
    // below (not stored as members).
    // Skip all weight caching for debugging numerical precision issues

    const auto& cfg = model_->config();

    // Compute effective cache budget from free VRAM minus reserve.
    // This preserves the existing per-phase budget tracking while the VRAMBudget
    // struct controls strategy-level decisions (which phases to skip).
    size_t free_vram = 0, total_vram = 0;
    vram_budget_mem_get_info(&free_vram, &total_vram);
    // Reserve headroom to avoid shared/system memory fallback on WSL2 (not
    // visible via nvidia-smi) — canonical floor in vram_query.h.
    size_t min_reserve = std::max(budget.reserve_bytes, vram_reserve_floor(total_vram));
    // Deduct NVFP4 decode cache (Phase 3, not yet allocated) from the EARLY
    // phases' budget so Phase 1's FP16 cache doesn't overcommit VRAM on large
    // dense models (Gemma-3-12B Q4_K_M: 12.3 GiB FP16 + 1.4 GiB NVFP4 + 6.1 GiB
    // KV → IMA). KV cache is already allocated before Phase 1 so free_vram
    // already reflects it. Deducting the reservation from the SHARED budget
    // charged it to Phase 3 too — see split_pre_dequant_budget (#1100).
    const PreDequantBudget budgets = split_pre_dequant_budget(free_vram, min_reserve,
                                                              budget.nvfp4_cache_bytes);
    size_t remaining_budget = budgets.shared;
    size_t early_budget = budgets.early;
    const size_t early_budget_start = early_budget;

    // --- Phase 0: Promote NVFP4 pre-quantized weights to Tensor sidecars ---
    // (body extracted to pre_dequant_phase0_promote_nvfp4_sidecars_)
    pre_dequant_phase0_promote_nvfp4_sidecars_(cfg, stream);

    // --- Phase 0b: register prequant-promoted NVFP4 weights in CUTLASS cache ---
    // (body extracted to pre_dequant_phase0b_register_cutlass_nvfp4_)
    pre_dequant_phase0b_register_cutlass_nvfp4_(cfg, stream);

    // Stage 1 (one-tier-truth): build the budget-constrained StoragePlan once
    // and hold it for the model's lifetime. Built AFTER Phase 0 so prequant-
    // NVFP4 weights already carry QType::NVFP4 (Phase 0 stamps it) — building
    // earlier mis-tiered native-NVFP4 weights as FP16/FP8. Phase 1 reads its
    // FP16-tier decision from the plan; a plan-vs-actual parity diagnostic
    // runs in Phase 4. The per-phase cache SIZING stays with the heuristic
    // budget (a separate UNCONSTRAINED plan feeds the #875 weight-cache
    // reserve in vram_budget.cpp — see the comment there).
    hints_->vram_budget_bytes = remaining_budget;
    storage_plan_ = plan_storage(*model_, cfg, *hints_);
    apply_arch_rules_(storage_plan_, cfg);
    if (storage_plan_.failed) {
        IMP_LOG_WARN("StoragePlanner: plan failed — %s", storage_plan_.failure_reason.c_str());
    } else {
        IMP_LOG_INFO("StoragePlanner: %zu entries, projected VRAM %.2f MiB",
                     storage_plan_.entries.size(),
                     storage_plan_.projected_vram_bytes / (1024.0 * 1024.0));
    }

    // --- Phase 1: FP16 weight cache + fused KV + fused gate+up (extracted) ---
    pre_dequant_phase1_fp16_cache_(cfg, budget, early_budget, stream);

    // --- Phase 2: FP8 cache for uncached weights (extracted) ---
    pre_dequant_phase2_fp8_cache_(cfg, budget, early_budget, stream);

    // Phases 1/2 decrement their own budget as they allocate. Charge exactly
    // what they spent against the shared budget — the untouched NVFP4
    // reservation stays with Phase 3 instead of being lost to it.
    pre_dequant_internal::deduct_budget(remaining_budget, early_budget_start - early_budget);

    // --- Phase 2b: FP8 decode sidecar for native-precision GDN/SSM
    // projections (gemm.fp8_ssm_proj) ---
    pre_dequant_phase2b_fp8_ssm_sidecar_(cfg, stream);

    // --- Phase 3: NVFP4 decode weight cache + 3b CUTLASS + 3c-native (extracted) ---
    pre_dequant_phase3_nvfp4_decode_(cfg, budget, remaining_budget, stream);


    // --- Phase 3c (standalone): Native MXFP4 GGUF when NVFP4 decode is disabled (extracted) ---
    pre_dequant_phase3c_standalone_mxfp4_(cfg, stream);

    // --- Phase 4: tensor registry + overlay diagnostic + NVFP4 device-args (extracted) ---
    pre_dequant_phase4_tensor_registry_(cfg, stream);

    // --- Phase 4b: free GGUF sources no path reads anymore ---
    pre_dequant_phase4b_drop_redundant_sources_(cfg, stream);

    // MemAccount per-pool attribution (vram_audit diagnostic): the caches
    // above allocate via raw cudaMallocAsync, invisible to VRAMAllocator, so
    // note the build totals here — one note per cache family.
    MemAccount::instance().note("WEIGHT_CACHE_FP16",
                                static_cast<std::ptrdiff_t>(wcache_->fp16_bytes));
    // Minus the SSM sidecar: it is the one FP8 cache that goes through
    // VRAMAllocator, which names its own charges now, so including it here
    // counted it twice.
    MemAccount::instance().note("WEIGHT_CACHE_FP8",
                                static_cast<std::ptrdiff_t>(wcache_->fp8_bytes - wcache_->fp8_sidecar_bytes));
    MemAccount::instance().note(
        "WEIGHT_CACHE_NVFP4",
        static_cast<std::ptrdiff_t>(wcache_->nvfp4_bytes + wcache_->nvfp4_moe_bytes));
    MemAccount::instance().note(
        "WEIGHT_CACHE_CUTLASS_SF",
        static_cast<std::ptrdiff_t>(wcache_->cutlass_nvfp4_bytes + wcache_->cutlass_mxfp4_bytes));
}

// NVFP4 view of the LM head for the MTP draft chain — mirrors the decode-path
// LM-head dispatch (executor_forward.cu): secondary wcache_.nvfp4 entry first,
// then the native-NVFP4 registry tier; FP8 takes precedence and disables it.
bool GraphExecutor::lm_head_nvfp4_view(NvFP4QuantResult& out) const {
    if (!model_)
        return false;
    const Tensor& lm = model_->output_proj();
    if (!lm.data)
        return false;
    if (wcache_.fp8.count(lm.data))
        return false;
    auto it = wcache_.nvfp4.find(lm.data);
    if (it != wcache_.nvfp4.end()) {
        out = it->second;
        out.owned = false;  // borrows the decode-cache storage
        return true;
    }
    if (model_->out_proj_id == kInvalidTensorID)
        return false;
    const WeightHandle& h = registry_.handle(model_->out_proj_id);
    if (h.primary_tier != StorageTier::NVFP4 || h.payload.nvfp4.data == nullptr)
        return false;
    out.packed_data  = h.payload.nvfp4.data;
    out.micro_scales = h.payload.nvfp4.block_scales;
    out.tensor_scale = (h.payload.nvfp4.tensor_scale != nullptr)
                           ? *h.payload.nvfp4.tensor_scale
                           : 1.0f;
    out.N = model_->config().vocab_size;
    out.K = model_->config().d_model;
    out.owned = false;
    return true;
}

// Fold the scattered arch-specific overlay rules into one pass over the plan so
// the plan reproduces what the legacy builders do (the precondition for making
// builders plan-driven without changing behaviour). Driven by the Phase-4
// plan/actual parity diagnostic: a rule is added here only where parity shows
// the plan and the legacy path disagree for a real (non-budget) reason.
void QuantPipeline::apply_arch_rules_(StoragePlan& plan, const ModelConfig& cfg) const {
    // FP8 prefill unavailable (sm_120 cuBLAS, and disabled for gemma/GDN): the
    // FP8-floor kinds (WK/WV/QKV_FUSED) the plan picked from the kind table fall
    // back to FP16 at build time. Encode that so the plan matches.
    if (!wcache_->use_fp8) {
        for (auto& e : plan.entries)
            if (e.tier == StorageTier::FP8)
                e.tier = StorageTier::FP16;
    }

    // LM head → NVFP4 per the #982 net rule (nvfp4_lm_head_enabled): the kind
    // table caps LM_HEAD at FP16, so the plan can't pick NVFP4 itself, but the
    // legacy path quantizes the lm_head to an NVFP4 decode cache. This plan
    // site only serves NATIVE (BF16/F16) heads (+8-16% decode, +2.2% PPL —
    // owner-accepted, GOAL-listed); quantized GGUF heads route through the
    // phase-3 collector, which applies the size/arch-gated auto rule.
    // GDN/SSM hybrids keep the FP16 lm_head unless nvfp4_lm_head_gdn.
    if (pre_dequant_internal::nvfp4_lm_head_enabled(runtime_config(), /*quantized_source=*/false,
                                                    model_->profile().is_dense, cfg.d_model)) {
        bool is_gdn = false;
        for (int i = 0; i < cfg.n_layers; i++) {
            const auto& L = model_->layer(i);
            if (L.ssm_in.data || L.ssm_out.data || L.gdn_gate.data) {
                is_gdn = true;
                break;
            }
        }
        if (!is_gdn || runtime_config().gemm.nvfp4_lm_head_gdn) {
            for (auto& e : plan.entries)
                if (e.kind == TensorKind::LM_HEAD)
                    e.tier = StorageTier::NVFP4;
        }
    }

    // gemma-3: the NVFP4 decode cache must be built FROM an FP16 companion copy,
    // not from scratch — from-scratch corrupts gemma-3 decode (first step emits
    // token 0 / <pad>, then IMA). The legacy Phase 1 keeps these FP16-cached;
    // encode that as an fp16_companion flag on every NVFP4-tier entry so the
    // plan-driven Phase 1 (Stage 1.3) preserves the same backing copy.
    if (model_->profile().is_gemma3) {
        for (auto& e : plan.entries) {
            if (e.tier == StorageTier::NVFP4)
                e.fp16_companion = true;
        }
    }
}

}  // namespace imp
