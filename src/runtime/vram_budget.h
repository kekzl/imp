#pragma once

#include "model/model.h"
#include "memory/kv_cache.h"
#include <cstddef>

namespace imp {

struct EngineConfig;

// Mandatory decode-cache demand of a native-NVFP4 (is_nvfp4_prequant) checkpoint.
// All-or-nothing for CUDA-graph decode: one MoE layer missing its nvfp4_moe entry falls
// to the host-args legacy path, which aborts capture (26-40 tok/s vs ~250 captured).
struct NativeCacheDemand {
    // Persistent CUTLASS SfAtom SF slab (phase 3b) across ALL registered NVFP4 weights
    // (dense projections, experts, GDN/SSM projections, LM head). Sized with
    // cutlass_nvfp4_sf_size() plus 256-byte per-entry alignment (pre_dequant_phase3_cutlass.cpp).
    size_t sf_bytes = 0;
    // Largest transient per-(layer,proj) contiguous MoE expert copy (phase 3-moe copy
    // branch: packed + micro-scales + tensor-scales). The zero-copy borrow branch needs
    // ~none of this: it is an upper bound.
    size_t moe_slab_bytes = 0;
    size_t total() const { return sf_bytes + moe_slab_bytes; }
};

// Scan the model's projection/expert tensor shapes and compute the mandatory decode-cache
// demand. Reads only shapes + data-non-null, so it works pre-upload (resolver time, host
// tensors) and post-upload alike. Returns all-zero for non-prequant models.
NativeCacheDemand compute_native_cache_demand(const Model& model);

// NVFP4 bytes phase 3-moe builds for a GGUF model's device-resident experts (whole layers only).
// 0 for NVFP4-prequant checkpoints (their native cache is mandatory_moe_bytes).
size_t gguf_moe_nvfp4_cache_bytes(const Model& model, bool decode_all, int nvfp4_decode_mode);

struct VRAMBudget;
// VRAM a mode-2 weight cache built from live free VRAM must leave free for what lands after it:
// the planned KV pool + 256 MiB workspace, the allocator headroom KV growth keeps, the library
// claim, the IMMA planes and the SSM slab. 0 when no plan applied (kv_plan_bytes == 0).
size_t planned_post_cache_reserve(const VRAMBudget& budget, size_t total_vram);

struct PlanResult;
// GGUF only (no-op for NVFP4-prequant or a rejected plan): sets kv_plan_bytes to the plan's KV
// pool and weight_cache_grant_bytes to optional_caches minus the allocator headroom the plan's
// slack against `distributable` leaves uncovered.
void apply_plan_to_weight_caches(VRAMBudget& budget, const PlanResult& plan, size_t distributable,
                                 bool nvfp4_prequant);

// VRAM budget for weight cache allocation (computed by Engine::plan_vram_budget).
struct VRAMBudget {
    enum Strategy { FP8_PREFILL_NVFP4_DECODE, NVFP4_DECODE_ONLY, FP16_ONLY };
    Strategy strategy = FP16_ONLY;
    size_t kv_cache_bytes = 0;
    size_t fp8_cache_bytes = 0;  // 0 for sub-8-bit models
    size_t nvfp4_cache_bytes = 0;
    size_t reserve_bytes = 1024ULL * 1024 * 1024;  // 1 GiB safety
    int kv_max_blocks = 0;
    // Value of kv_max_blocks before the min_kv_tokens rescue floor (vram_budget.cpp) was
    // applied. Equal to kv_max_blocks when no floor applied. Kept to distinguish computed
    // sizing from floor-inflated reports in engine_kv_cache_init.cpp divergence logs (#1747).
    int kv_blocks_pre_floor = 0;
    // For re-sizing at a measured library reserve (#2436): the reserve before the library floor,
    // and kv_max_blocks before the post-weight fit clamp (0 = that clamp did not bind).
    size_t reserve_other_bytes = 0;
    int kv_blocks_before_fit_clamp = 0;
    // SWA-aware sizing (kv_cache.swa_sizing): capacity of the dedicated
    // sliding-window block group (0 = feature off). Sized batch-shaped:
    // ceil(swa_live_tokens / block_size) + 1 blocks per sequence slot.
    int swa_max_blocks = 0;
    bool nvfp4_second_pass = false;  // true → re-run NVFP4 after FP16-Free
    // Guaranteed byte-floors for mandatory native-NVFP4 decode caches (0 for non-prequant
    // models). Phase 3b / phase 3-moe floor their live-free-derived budgets at these values:
    // a lagging cudaMemGetInfo cannot starve a cache the plan already granted (AUDIT B62).
    size_t mandatory_sf_bytes = 0;
    size_t mandatory_moe_bytes = 0;
    // Total weight-cache demand this pass charged against post-weight headroom
    // (nvfp4 estimate + cutlass_sf estimate, absorbing native-NVFP4 and planner-driven
    // reserves). Exposed so plan_memory() compares against the SAME figure (A7 step 2b).
    size_t weight_cache_estimate_bytes = 0;
    // TRANSIENT slice of the estimate above: init-time headroom phase-0..3 builders re-derive
    // (max(total/10, floor) + margin), free again once the KV pool is sized.
    // Live pass needs it in the estimate; the shadow plan must NOT charge it against KV (#1765).
    size_t weight_cache_transient_bytes = 0;
    // Batch-shaped SSM/GDN state footprint charged as overhead (0 on
    // non-recurrent models). Same reason as above.
    size_t ssm_footprint_bytes = 0;
    // s8 + (alpha, beta) planes mmq_q8_imma caches per prefilled Q8_0 weight
    // (0 when none, or IMMA prefill paths off). Charged as overhead here AND handed to
    // mmq_q8_imma_set_plane_budget(): uncharged, the spill victim was a per-start lottery (#1899).
    size_t imma_plane_bytes = 0;
    // Library claim of the first forward (measured or the cold-start constant): absent from live
    // free VRAM until then, so phases sizing from cudaMemGetInfo must charge it.
    size_t library_reserve_bytes = 0;
    // plan_memory()'s optional_caches: what the mode-2 NVFP4 caches (dense + GGUF MoE) may take
    // in total. Set after the plan; SIZE_MAX when no plan applies (live-free sizing only).
    size_t weight_cache_grant_bytes = static_cast<size_t>(-1);
    // plan_memory()'s KV pool (global + SWA + per-block metadata); 0 when no plan applies.
    size_t kv_plan_bytes = 0;
};

// Pure computation: plan VRAM allocation split between KV cache, FP8 prefill
// cache, and NVFP4 decode cache based on model characteristics and config.
// No GPU allocation, just arithmetic.
//
// swa_live_tokens / n_swa_layers (kv_cache.swa_sizing, both 0 = off): sliding-window
// layers are charged a fixed per-sequence live span of swa_live_tokens (window + slack
// + burst/chunk peak) instead of max_seq_len; only global layers scale with context.
//
// KV is charged the full measured native-NVFP4 decode-cache demand; phase 3 is floored
// at it via mandatory_sf_bytes / mandatory_moe_bytes, a planned guarantee (AUDIT B62).
//
// native_demand: precomputed NativeCacheDemand (Engine's cached scan); nullptr computes
// it locally (tests, standalone).
// ssm_reserved_slots: recurrent-state slots past max_batch_size the multi-candidate
// speculative verify reserves (SSMState::init n_reserved); priced with the pool.
// q8_imma_prefill / moe_imma_prefill: gemm.q8_imma_enabled and gemm.moe_imma_prefill;
// decide whether IMMA prefill planes (imma_plane_bytes) are charged. The pass cannot
// read RuntimeConfig itself.
VRAMBudget compute_vram_budget(const Model& model, const EngineConfig& config, int n_kv_layers, int head_dim,
                               size_t free_vram, int swa_live_tokens = 0, int n_swa_layers = 0,
                               const NativeCacheDemand* native_demand = nullptr, int ssm_reserved_slots = 0,
                               bool q8_imma_prefill = true, bool moe_imma_prefill = true);

// Split of the post-reserve VRAM budget across the pre-dequant phases.
struct PreDequantBudget {
    // Free VRAM minus the safety reserve: the ceiling for the whole cache
    // build. Phase 3 (NVFP4 decode cache) spends what Phases 1/2 leave.
    size_t shared = 0;
    // Phases 1/2 (FP16 + FP8 caches): `shared` minus the NVFP4 decode cache's
    // reservation. That cache is planned but not yet allocated, so free_vram
    // does not show it and the early phases would otherwise overcommit it.
    size_t early = 0;
};

// Pure arithmetic. The NVFP4 reservation is withheld from Phases 1/2 ONLY: charging it to
// the shared budget too double-charges Phase 3 (the KV pool is already allocated before the
// cache build, so its bytes are gone from free_vram once, then charged again there) (#1100).
inline PreDequantBudget split_pre_dequant_budget(size_t free_vram, size_t reserve_bytes,
                                                 size_t nvfp4_reservation_bytes) {
    PreDequantBudget b;
    b.shared = (free_vram > reserve_bytes) ? (free_vram - reserve_bytes) : 0;
    b.early = (b.shared > nvfp4_reservation_bytes) ? (b.shared - nvfp4_reservation_bytes) : 0;
    return b;
}

// Bytes of one paged KV block for a single layer, K+V combined (2x), packing- and scale-aware.
// Single source for every KV-size estimate (#942): raw dtype_size() returns 0 for NVFP4/MXFP4_KV,
// counts INT4 at 1 byte/elem, and omits the per-token or per-16-element-group scale overhead.
size_t kv_block_bytes_per_layer(QType kv_dtype, int block_size, int n_kv_heads, int head_dim);

// Mean of kv_block_bytes_per_layer over the layers, rounded up, as the per-layer pool lays them
// out (head_dim_per_layer set, every layer attention, no SWA sizing, not INT8/INT4); else the max
// shape. Gemma-4-26B F16: 3.4375 MiB per block over 30 layers vs 7.5 MiB at the max shape.
size_t kv_block_bytes_layer_mean(const ModelConfig& mcfg, QType kv_dtype, int block_size, int head_dim,
                                 bool swa_sizing);

// attention.sparse_topk_tokens metadata gate: nullptr = the key min/max pool is built, else the
// reason it is refused. One rule for the plan charge and the enable gate (#2360).
const char* sparse_minmax_refusal(QType kv_dtype, bool mla_absorb, bool token_recycling,
                                  bool persistent_prefix_cache);

}  // namespace imp
