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
    // The zero-copy borrow branch needs ~none of this — it is an upper
    // bound, held only until the balloon is released before phase 3.
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
    // SWA-aware sizing (kv_cache.swa_sizing): capacity of the dedicated
    // sliding-window block group (0 = feature off). Sized batch-shaped:
    // ceil(swa_live_tokens / block_size) + 1 blocks per sequence slot.
    int swa_max_blocks = 0;
    bool nvfp4_second_pass = false;  // true → re-run NVFP4 after FP16-Free
    // Guaranteed byte-floors for the mandatory native-NVFP4 decode caches
    // (0 for non-prequant models). Phase 3b / phase 3-moe floor their live-
    // free-derived mode-2 budgets at these values so a lagging
    // cudaMemGetInfo (async frees are reclaimed late on this driver) can't
    // starve a cache whose room was physically reserved via the balloon.
    size_t mandatory_sf_bytes = 0;
    size_t mandatory_moe_bytes = 0;
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
// mandatory_cache_prealloc: bytes already physically reserved for the
// mandatory native-NVFP4 decode caches (Engine's balloon, held while this
// runs — free_vram excludes it). The prequant reserve then only charges KV
// for the UNCOVERED remainder, preventing a double-count.
VRAMBudget compute_vram_budget(const Model& model, const EngineConfig& config, int n_kv_layers, int head_dim,
                               size_t free_vram, int swa_live_tokens = 0, int n_swa_layers = 0,
                               size_t mandatory_cache_prealloc = 0);

// Bytes of one paged KV block for a single layer, K+V combined (2x),
// packing- and scale-aware. Single source for every KV-size estimate
// (#942): raw dtype_size() returns 0 for NVFP4/MXFP4_KV (zeroing the
// estimate), counts INT4 at 1 byte/elem (2x the packed size), and knows
// nothing about the per-token (INT8/INT4) or per-16-element-group
// (NVFP4/MXFP4_KV) scale overhead the cache actually stores.
size_t kv_block_bytes_per_layer(QType kv_dtype, int block_size, int n_kv_heads, int head_dim);

}  // namespace imp
