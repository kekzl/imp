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
    size_t remaining_budget = (free_vram > min_reserve) ? (free_vram - min_reserve) : 0;

    // Phase 4.2: run StoragePlanner for diagnostic purposes.
    // The plan output is NOT used to drive allocation yet — the existing legacy code
    // path still decides what to allocate. Log discrepancies between the plan and
    // the legacy decisions so we can catch bugs before Phase 4.4+ flips to
    // plan-driven allocation. Actual storage ownership flip happens in Phase 5.
    {
        hints_.vram_budget_bytes = remaining_budget;
        StoragePlan diag_plan = plan_storage(*model_, cfg, hints_);
        if (diag_plan.failed) {
            IMP_LOG_WARN("StoragePlanner (diagnostic): plan failed — %s", diag_plan.failure_reason.c_str());
        } else {
            IMP_LOG_INFO("StoragePlanner (diagnostic): %zu entries, projected VRAM %.2f MiB",
                         diag_plan.entries.size(), diag_plan.projected_vram_bytes / (1024.0 * 1024.0));
        }
    }

    // --- Phase 0: Promote NVFP4 pre-quantized weights to Tensor sidecars ---
    // (body extracted to pre_dequant_phase0_promote_nvfp4_sidecars_)
    pre_dequant_phase0_promote_nvfp4_sidecars_(cfg, stream);

    // --- Phase 0b: register prequant-promoted NVFP4 weights in CUTLASS cache ---
    // (body extracted to pre_dequant_phase0b_register_cutlass_nvfp4_)
    pre_dequant_phase0b_register_cutlass_nvfp4_(cfg, stream);

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
}

}  // namespace imp
