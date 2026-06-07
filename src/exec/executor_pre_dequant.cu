// Pre-dequant orchestrator.
// Dispatches the six pre-dequant phases (0/0b/1/2/3/3c/4) to their
// extracted translation units. Each phase lives in src/exec/pre_dequant_phase*.cu.
//
// Adding a new phase: write one new src/exec/pre_dequant_phase*.cu file,
// add it to CMakeLists.txt IMP_EXEC_SOURCES, declare the method on
// GraphExecutor in executor.h, and call it from pre_dequant_weights() below.

#include "exec/executor.h"
#include "core/logging.h"
#include "runtime/storage_planner.h"

#include <cuda_runtime.h>
#include <algorithm>

namespace imp {

void GraphExecutor::pre_dequant_weights(cudaStream_t stream, const VRAMBudget& budget) {
    if (!initialized_ || !model_)
        return;
    // Skip all weight caching for debugging numerical precision issues

    const auto& cfg = model_->config();

    // Compute effective cache budget from free VRAM minus reserve.
    // This preserves the existing per-phase budget tracking while the VRAMBudget
    // struct controls strategy-level decisions (which phases to skip).
    size_t free_vram = 0, total_vram = 0;
    IMP_CUDA_CHECK_LOG(cudaMemGetInfo(&free_vram, &total_vram));
    // Reserve at least 10% of total VRAM as headroom to avoid shared/system
    // memory fallback on WSL2 (not visible via nvidia-smi).
    size_t min_reserve = std::max(budget.reserve_bytes, total_vram / 10);
    // Deduct NVFP4 decode cache (Phase 3, not yet allocated) from the budget
    // so Phase 1's FP16 cache doesn't overcommit VRAM on large dense models
    // (Gemma-3-12B Q4_K_M: 12.3 GiB FP16 + 1.4 GiB NVFP4 + 6.1 GiB KV → IMA).
    // KV cache is already allocated before Phase 1 so free_vram already reflects it.
    size_t total_reserve = min_reserve + budget.nvfp4_cache_bytes;
    size_t remaining_budget = (free_vram > total_reserve) ? (free_vram - total_reserve) : 0;

    // --- Phase 0: Promote NVFP4 pre-quantized weights to Tensor sidecars ---
    // (body extracted to pre_dequant_phase0_promote_nvfp4_sidecars_)
    pre_dequant_phase0_promote_nvfp4_sidecars_(cfg, stream);

    // --- Phase 0b: register prequant-promoted NVFP4 weights in CUTLASS cache ---
    // (body extracted to pre_dequant_phase0b_register_cutlass_nvfp4_)
    pre_dequant_phase0b_register_cutlass_nvfp4_(cfg, stream);

    // Stage 1 (one-tier-truth): build the StoragePlan once and hold it for the
    // model's lifetime. The pre-dequant phases are being migrated to read their
    // overlay-tier decision from `storage_plan_` (plan_tier_of) instead of
    // scattered nvfp4_beneficial/plan_routes_to_fp16 checks. Built AFTER Phase 0
    // so prequant-NVFP4 weights already carry QType::NVFP4 (Phase 0 stamps it) —
    // building earlier mis-tiered native-NVFP4 weights as FP16/FP8. A plan-vs-
    // actual parity diagnostic runs in Phase 4 to surface any mismatch before a
    // builder is switched. The plan does not drive allocation yet.
    hints_.vram_budget_bytes = remaining_budget;
    storage_plan_ = plan_storage(*model_, cfg, hints_);
    apply_arch_rules_(storage_plan_, cfg);
    if (storage_plan_.failed) {
        IMP_LOG_WARN("StoragePlanner: plan failed — %s", storage_plan_.failure_reason.c_str());
    } else {
        IMP_LOG_INFO("StoragePlanner: %zu entries, projected VRAM %.2f MiB",
                     storage_plan_.entries.size(),
                     storage_plan_.projected_vram_bytes / (1024.0 * 1024.0));
    }

    // --- Phase 1: FP16 weight cache + fused KV + fused gate+up (extracted) ---
    pre_dequant_phase1_fp16_cache_(cfg, budget, remaining_budget, stream);

    // --- Phase 2: FP8 cache for uncached weights (extracted) ---
    pre_dequant_phase2_fp8_cache_(cfg, budget, remaining_budget, stream);

    // --- Phase 3: NVFP4 decode weight cache + 3b CUTLASS + 3c-native (extracted) ---
    pre_dequant_phase3_nvfp4_decode_(cfg, budget, remaining_budget, stream);


    // --- Phase 3c (standalone): Native MXFP4 GGUF when NVFP4 decode is disabled (extracted) ---
    pre_dequant_phase3c_standalone_mxfp4_(cfg, stream);

    // --- Phase 4: tensor registry + overlay diagnostic + NVFP4 device-args (extracted) ---
    pre_dequant_phase4_tensor_registry_(cfg, stream);

    // --- Phase 4b: mark redundant sources as dropped (5.1.4.b — bisect mode) ---
    pre_dequant_phase4b_drop_redundant_sources_(cfg, stream);
}

// Fold the scattered arch-specific overlay rules into one pass over the plan so
// the plan reproduces what the legacy builders do (the precondition for making
// builders plan-driven without changing behaviour). Driven by the Phase-4
// plan/actual parity diagnostic: a rule is added here only where parity shows
// the plan and the legacy path disagree for a real (non-budget) reason.
void GraphExecutor::apply_arch_rules_(StoragePlan& plan, const ModelConfig& cfg) const {
    // FP8 prefill unavailable (sm_120 cuBLAS, and disabled for gemma/GDN): the
    // FP8-floor kinds (WK/WV/QKV_FUSED) the plan picked from the kind table fall
    // back to FP16 at build time. Encode that so the plan matches.
    if (!wcache_.use_fp8) {
        for (auto& e : plan.entries)
            if (e.tier == StorageTier::FP8)
                e.tier = StorageTier::FP16;
    }

    // LM head → NVFP4 when nvfp4_lm_head (default on): the kind table caps
    // LM_HEAD at FP16, so the plan can't pick NVFP4 itself, but the legacy path
    // quantizes the BF16/Q*_K lm_head to an NVFP4 decode cache (+8-16% decode,
    // +2.2% PPL — owner-accepted). GDN/SSM-hybrid models keep the FP16 lm_head
    // unless nvfp4_lm_head_gdn (mirrors nvfp4_decode_cache_fp16_lm_head_).
    if (runtime_config().gemm.nvfp4_lm_head) {
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
    if (cfg.arch == ModelArch::GEMMA3) {
        for (auto& e : plan.entries) {
            if (e.tier == StorageTier::NVFP4)
                e.fp16_companion = true;
        }
    }
}

}  // namespace imp
