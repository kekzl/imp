#pragma once

#include "core/tensor.h"            // Tensor, QType (used by Nvfp4DecodeContext + signatures)
#include "runtime/storage_planner.h" // StoragePlan, PlanHints, StorageTier
#include "exec/weight_handle.h"      // WeightRegistry
#include "runtime/config.h"          // RuntimeConfig (runtime_config() accessor)
#include <cuda_runtime.h>
#include <cstddef>
#include <vector>
#include <unordered_set>

namespace imp {

class Model;
class VRAMAllocator;
struct ModelConfig;
struct VRAMBudget;     // defined in exec/executor.h
struct WeightCaches;   // defined in exec/executor.h
struct QuantScratch;
struct MoEWorkspace;

// Per-call state for QuantPipeline::pre_dequant_phase3_nvfp4_decode_().
// Bundles the locals shared between sub-phase helpers — exclusion set,
// candidate entries, mode string, MoE-cache accumulators.
struct Nvfp4DecodeContext {
    // One entry per weight that may receive NVFP4 quantization. Populated by
    // the collect phase; consumed by the mode-1 / mode-2 / second-pass phases.
    struct Entry {
        const void* orig_ptr{};
        Tensor weight;
        QType qtype{};
        bool from_scratch{};
    };

    std::unordered_set<const void*> exclude_ptrs;
    std::vector<Entry> entries;
    const char* mode_str = "";          // "additive" or "only", set early
    size_t moe_budget = 0;              // initialised before the MoE-cache phase
    int    nvfp4_moe_count = 0;         // populated by the MoE-cache phase
    size_t nvfp4_moe_total = 0;
    size_t nvfp4_moe_ms_freed = 0;      // duplicated per-expert micro-scales freed (borrow path)
    // Shared VRAM safety reserve for mode 2 paths (dense incremental,
    // CUTLASS NVFP4, MoE expert caching). Computed once at the top of
    // pre_dequant_phase3_nvfp4_decode_() from the model's actual attention
    // layout — replaces the previous `total_mem / 10` heuristic which
    // reserved 3.2 GiB on a 32 GiB 5090 and starved the dense NVFP4 cache
    // (20 of 281 tensors uncached on Qwen3-14B Q6_K → −20% decode tok/s).
    size_t safety_reserve = 0;
};

// Init-time weight-quantization pipeline, extracted from GraphExecutor (D2).
// Runs once via build(); fills the four long-lived caches (owned by the caller)
// and owns only the build-only StoragePlan + decode context. The forward hot
// path reads the caches unchanged (byte-identical). See the QuantPipeline
// design memo (archived: docs/archive/README.md).
class QuantPipeline {
public:
    // Runs the full init-time quantization pipeline once. Populates the four
    // long-lived caches (owned by the caller) from the model's weights; owns the
    // transient StoragePlan + decode context internally.
    void build(const Model& model, const RuntimeConfig& rcfg, VRAMAllocator& alloc,
               const VRAMBudget& budget, cudaStream_t stream, WeightCaches& wcache,
               QuantScratch& qscratch, WeightRegistry& registry, PlanHints& hints,
               MoEWorkspace& moe, int max_tokens);

private:
    // Build context (set at the top of build(); the phase methods read these
    // exactly as they read the same-named GraphExecutor members today).
    const Model* model_ = nullptr;
    VRAMAllocator* vram_alloc_ = nullptr;
    const RuntimeConfig* runtime_config_ = nullptr;
    WeightCaches* wcache_ = nullptr;
    QuantScratch* qscratch_ = nullptr;
    WeightRegistry* registry_ = nullptr;
    PlanHints* hints_ = nullptr;
    MoEWorkspace* moe_ = nullptr;
    int max_tokens_ = 0;   // workspace max token count (build-time scratch sizing)

    // Accessor mirroring GraphExecutor::runtime_config() so the moved phase
    // methods read the config exactly as before. Set in build() from the
    // owning GraphExecutor's already-validated config.
    // DELIBERATE DUPLICATION (behaviour-neutral verbatim move): keeps the ~12
    // runtime_config() call sites in the moved phases byte-identical. When the
    // next GraphExecutor component (MoeRunner/Workspace) needs a 3rd copy, hoist
    // this to a shared free helper in runtime/config.h instead.
    const RuntimeConfig& runtime_config() const noexcept {
        static const RuntimeConfig kDefault;
        return runtime_config_ ? *runtime_config_ : kDefault;
    }

    // Owned build-only state.
    StoragePlan storage_plan_;
    // Planned overlay tier for a source pointer, or Undefined if the pointer is
    // not in the plan (native GGUF blocks bypass the overlay layer).
    StorageTier plan_tier_of(const void* src) const { return storage_plan_.tier_of(src); }

    // --- moved phase / helper declarations (verbatim from executor.h) ---
    // Stage 1.2: fold the scattered arch-specific overlay rules into one pass
    // over the freshly-built plan, so the plan matches the legacy builders.
    void apply_arch_rules_(StoragePlan& plan, const ModelConfig& cfg) const;

    void pre_dequant_phase0_promote_nvfp4_sidecars_(const ModelConfig& cfg, cudaStream_t stream);
    void pre_dequant_phase0b_register_cutlass_nvfp4_(const ModelConfig& cfg, cudaStream_t stream);
    void pre_dequant_phase1_fp16_cache_(const ModelConfig& cfg, const VRAMBudget& budget,
                                        size_t& remaining_budget, cudaStream_t stream);
    void pre_dequant_phase2_fp8_cache_(const ModelConfig& cfg, const VRAMBudget& budget,
                                       size_t& remaining_budget, cudaStream_t stream);
    void pre_dequant_phase3_nvfp4_decode_(const ModelConfig& cfg, const VRAMBudget& budget,
                                          size_t& remaining_budget, cudaStream_t stream);
    // Sub-phase helpers for pre_dequant_phase3_nvfp4_decode_. Each operates
    // on a shared Nvfp4DecodeContext; the orchestrator above calls them in
    // sequence, mirroring the legacy monolithic body.
    void nvfp4_decode_collect_candidates_(const ModelConfig& cfg, Nvfp4DecodeContext& dctx);
    void nvfp4_decode_cache_fp16_lm_head_(const ModelConfig& cfg, cudaStream_t stream);
    void nvfp4_decode_cache_fp16_projections_(const ModelConfig& cfg, cudaStream_t stream);
    void nvfp4_decode_quantize_mode2_(cudaStream_t stream, Nvfp4DecodeContext& dctx);
    void nvfp4_decode_quantize_mode1_(size_t& remaining_budget, cudaStream_t stream,
                                      Nvfp4DecodeContext& dctx);
    void nvfp4_decode_free_fp16_and_migrate_fp8_(size_t& remaining_budget, cudaStream_t stream,
                                                 Nvfp4DecodeContext& dctx);
    void nvfp4_decode_second_pass_(const VRAMBudget& budget, cudaStream_t stream,
                                   Nvfp4DecodeContext& dctx);
    void nvfp4_decode_convert_cutlass_(const ModelConfig& cfg, size_t& remaining_budget,
                                       cudaStream_t stream);
    void nvfp4_decode_convert_mxfp4_and_native_(const ModelConfig& cfg, cudaStream_t stream);
    void nvfp4_decode_mxfp4_fp16_fallback_(const ModelConfig& cfg, cudaStream_t stream);
    void nvfp4_decode_cache_moe_experts_(const ModelConfig& cfg, size_t& remaining_budget,
                                         cudaStream_t stream, Nvfp4DecodeContext& dctx);
    bool cache_moe_native_nvfp4_(Tensor& packed, std::vector<Tensor>& experts, cudaStream_t stream,
                                 Nvfp4DecodeContext& dctx, bool& moe_budget_exhausted,
                                 size_t& moe_logical_avail);
    void gpt_oss_convert_moe_experts_(const ModelConfig& cfg, Nvfp4DecodeContext& dctx);
    void pre_dequant_phase3c_standalone_mxfp4_(const ModelConfig& cfg, cudaStream_t stream);
    void pre_dequant_phase4_tensor_registry_(const ModelConfig& cfg, cudaStream_t stream);
    void pre_dequant_phase4b_drop_redundant_sources_(const ModelConfig& cfg, cudaStream_t stream);
};

}  // namespace imp
