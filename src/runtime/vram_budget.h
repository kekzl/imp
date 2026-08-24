#pragma once

#include "model/model.h"
#include "memory/kv_cache.h"
#include <cstddef>

namespace imp {

struct EngineConfig;  // forward declaration

// Mandatory decode-cache demand of a native-NVFP4 (is_nvfp4_prequant)
// checkpoint. These caches are all-or-nothing for CUDA-graph decode: a
// single MoE layer without its nvfp4_moe entry falls to the host-args
// legacy path, which throws under capture and aborts the WHOLE decode
// graph (26-40 tok/s per-step instead of ~250 captured on Qwen3.6-35B).
struct NativeCacheDemand {
    // Persistent CUTLASS SfAtom SF slab (phase 3b) across ALL registered
    // NVFP4 weights — dense projections, experts, GDN/SSM projections,
    // LM head. Sized with cutlass_nvfp4_sf_size() + the slab's 256-byte
    // per-entry alignment, mirroring pre_dequant_phase3_cutlass.cu.
    size_t sf_bytes = 0;
    // Largest transient per-(layer,proj) contiguous MoE expert copy
    // (phase 3-moe copy branch: packed + micro-scales + tensor-scales).
    // The zero-copy borrow branch needs ~none of this — it is an upper bound.
    size_t moe_slab_bytes = 0;
    size_t total() const { return sf_bytes + moe_slab_bytes; }
};

// Scan the model's projection/expert tensor shapes and compute the
// mandatory decode-cache demand. Reads only shapes + data-non-null, so it
// works pre-upload (resolver time, host tensors) and post-upload alike.
// Returns all-zero for non-prequant models.
NativeCacheDemand compute_native_cache_demand(const Model& model);

// VRAM budget for weight cache allocation (computed by Engine::plan_vram_budget).
// Replaces ad-hoc "remaining_budget" with per-phase caps computed upfront.
struct VRAMBudget {
    enum Strategy { FP8_PREFILL_NVFP4_DECODE, NVFP4_DECODE_ONLY, FP16_ONLY };
    Strategy strategy = FP16_ONLY;
    size_t kv_cache_bytes = 0;
    size_t fp8_cache_bytes = 0;  // 0 for sub-8-bit models
    size_t nvfp4_cache_bytes = 0;
    size_t reserve_bytes = 1024ULL * 1024 * 1024;  // 1 GiB safety
    int kv_max_blocks = 0;
    // What the sizing arrived at BEFORE the min_kv_tokens rescue floor raised
    // it (vram_budget.cpp, "raising KV from N to M blocks"). Kept because the
    // divergence log in engine_kv_cache_init.cpp printed the floored figure
    // under "live pass would have said", so a start that hit the floor
    // reported a lower bound of the configuration as if it were a reading
    // (#1747). Equal to kv_max_blocks when no floor was applied.
    int kv_blocks_pre_floor = 0;
    // SWA-aware sizing (kv_cache.swa_sizing): capacity of the dedicated
    // sliding-window block group (0 = feature off). Sized batch-shaped:
    // ceil(swa_live_tokens / block_size) + 1 blocks per sequence slot.
    int swa_max_blocks = 0;
    bool nvfp4_second_pass = false;  // true → re-run NVFP4 after FP16-Free
    // Guaranteed byte-floors for the mandatory native-NVFP4 decode caches
    // (0 for non-prequant models). Phase 3b / phase 3-moe floor their live-
    // free-derived mode-2 budgets at these values so a lagging
    // cudaMemGetInfo (async frees are reclaimed late on this driver) cannot
    // starve a cache the PLAN has already granted. Unconditional since the
    // balloon was removed (AUDIT B62): the guarantee is the floor itself, not
    // bytes some caller pre-held.
    size_t mandatory_sf_bytes = 0;
    size_t mandatory_moe_bytes = 0;
    // Total weight-cache demand this pass charged against the post-weight
    // headroom (nvfp4 estimate + the cutlass_sf estimate, which absorbs the
    // native-NVFP4 and planner-driven reserves). Already computed internally;
    // exposed so plan_memory() can be fed the SAME demand figure and the two
    // allocation policies compared like for like (A7 step 2b).
    size_t weight_cache_estimate_bytes = 0;
    // Batch-shaped SSM/GDN state footprint charged as overhead (0 on
    // non-recurrent models). Same reason as above.
    size_t ssm_footprint_bytes = 0;
};

// Pure computation: plan VRAM allocation split between KV cache, FP8 prefill
// cache, and NVFP4 decode cache based on model characteristics and config.
// No GPU allocation — just arithmetic.
//
// swa_live_tokens / n_swa_layers (kv_cache.swa_sizing, both 0 = off):
// sliding-window layers are charged a fixed per-sequence live span of
// swa_live_tokens (window + slack + burst/chunk peak) instead of
// max_seq_len; only the remaining global layers scale with context.
//
// The mandatory native-NVFP4 decode caches used to be physically pre-reserved
// by an Engine-held balloon, and this took its size so KV was charged only for
// the uncovered remainder. The balloon is gone (AUDIT B62): KV is now charged
// the full measured demand, and phase 3 is floored at it through
// mandatory_sf_bytes / mandatory_moe_bytes — a planned guarantee instead of
// hidden bytes.
//
// native_demand: precomputed NativeCacheDemand (Engine's cached scan) to
// skip the tensor rescan; nullptr computes it locally (tests, standalone).
VRAMBudget compute_vram_budget(const Model& model, const EngineConfig& config, int n_kv_layers, int head_dim,
                               size_t free_vram, int swa_live_tokens = 0, int n_swa_layers = 0,
                               const NativeCacheDemand* native_demand = nullptr);

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

// Pure arithmetic. The NVFP4 reservation is withheld from Phases 1/2 ONLY.
// Charging it to the shared budget as well charged it to Phase 3 — the phase
// that *is* the reservation — so the decode cache paid for itself twice: the
// KV pool is allocated before the cache build, so its bytes are already gone
// from free_vram, and every one of them then came out of the decode cache a
// second time. On Qwen3-14B-Q6_K at the server's full-context default
// (max_seq_len 40960 → a 5.9 GiB KV pool) that left 100/280 tensors cached
// instead of 278/280, dropping decode 38% while ~11 GiB sat free (#1100).
inline PreDequantBudget split_pre_dequant_budget(size_t free_vram, size_t reserve_bytes,
                                                 size_t nvfp4_reservation_bytes) {
    PreDequantBudget b;
    b.shared = (free_vram > reserve_bytes) ? (free_vram - reserve_bytes) : 0;
    b.early = (b.shared > nvfp4_reservation_bytes) ? (b.shared - nvfp4_reservation_bytes) : 0;
    return b;
}

// Bytes of one paged KV block for a single layer, K+V combined (2x),
// packing- and scale-aware. Single source for every KV-size estimate
// (#942): raw dtype_size() returns 0 for NVFP4/MXFP4_KV (zeroing the
// estimate), counts INT4 at 1 byte/elem (2x the packed size), and knows
// nothing about the per-token (INT8/INT4) or per-16-element-group
// (NVFP4/MXFP4_KV) scale overhead the cache actually stores.
size_t kv_block_bytes_per_layer(QType kv_dtype, int block_size, int n_kv_heads, int head_dim);

}  // namespace imp
