#pragma once

// Phase-2b FP8 E4M3 decode sidecar: which full-precision weights get a per-row-scale copy that the
// M = 1 decode GEMV reads (gemv_fp8_rowscale), and the bytes it charges. Pure, so the unit lane pins
// the route selection (gemm.fp8_attn_proj, gemm.fp8_ffn_proj) and the VRAM charge without a GPU.

#include "model/model_config.h"

#include <array>
#include <cstddef>
#include <string>
#include <unordered_set>
#include <vector>

namespace imp {

struct Fp8SidecarScope {
    bool attn_qo = false;  // q / o projections
    bool attn_kv = false;  // k / v projections
    bool ffn = false;      // dense FFN: w_gate / w_up / w_down and the shared-expert trio (gemm.fp8_ffn_proj)
};

// attn_proj: "auto" (profile decides), "qo", "on", "off". ffn_proj: "auto" (profile decides: Gemma-4 NVFP4),
// "on", "off".
[[nodiscard]] inline Fp8SidecarScope fp8_sidecar_scope(const std::string& attn_proj, bool profile_attn_full,
                                                       const std::string& ffn_proj, bool profile_ffn) {
    const bool full = attn_proj == "on" || (attn_proj == "auto" && profile_attn_full);
    const bool ffn = ffn_proj == "on" || (ffn_proj == "auto" && profile_ffn);
    return {.attn_qo = full || attn_proj == "qo", .attn_kv = full, .ffn = ffn};
}

// Candidates of one layer, nullptr = not a target. SSM rows are always candidates: at decode the fused
// GDN input pack replaces ssm_in where it exists; without a pack ssm_in and gdn_gate run separately.
[[nodiscard]] inline std::array<const Tensor*, 13> fp8_sidecar_candidates(const TransformerLayer& L,
                                                                          const Fp8SidecarScope& s) {
    const bool pack = L.gdn_input_packed.data != nullptr;
    return {pack ? &L.gdn_input_packed : &L.ssm_in,
            pack ? nullptr : &L.gdn_gate,
            &L.ssm_out,
            s.attn_qo ? &L.wq : nullptr,
            s.attn_kv ? &L.wk : nullptr,
            s.attn_kv ? &L.wv : nullptr,
            s.attn_qo ? &L.wo : nullptr,
            s.ffn ? &L.w_gate : nullptr,
            s.ffn ? &L.w_up : nullptr,
            s.ffn ? &L.w_down : nullptr,
            s.ffn ? &L.w_gate_shared : nullptr,
            s.ffn ? &L.w_up_shared : nullptr,
            s.ffn ? &L.w_down_shared : nullptr};
}

// F16 device residents only (BF16 converted at upload; the calibration kernel reads __half), K % 16 == 0.
[[nodiscard]] inline bool fp8_sidecar_eligible(const Tensor& w) {
    return w.data != nullptr && w.on_device && w.qtype == QType::F16 && w.ndim == 2 && w.shape[0] > 0 &&
           w.shape[1] > 0 && (w.shape[1] % 16) == 0;
}

struct Fp8SidecarEntry {
    const void* ptr;
    size_t n_elems;
    int rows;
    int cols;
};

// bytes: the FP8 slab (1 byte per element, VRAMAllocator tag "fp8_ssm_sidecar"); rows: fp32 scales.
struct Fp8SidecarPlan {
    std::vector<Fp8SidecarEntry> entries;
    size_t bytes = 0;
    size_t rows = 0;
};

// cached(ptr): the weight already has an FP8 / NVFP4 entry and needs no sidecar. A shared pointer is
// planned once.
template <class LayerAt, class Cached>
[[nodiscard]] Fp8SidecarPlan plan_fp8_sidecar(int n_layers, const LayerAt& layer_at, const Fp8SidecarScope& s,
                                              const Cached& cached) {
    Fp8SidecarPlan plan;
    std::unordered_set<const void*> seen;
    for (int i = 0; i < n_layers; ++i) {
        for (const Tensor* w : fp8_sidecar_candidates(layer_at(i), s)) {
            if (w == nullptr || !fp8_sidecar_eligible(*w) || cached(w->data) || !seen.insert(w->data).second)
                continue;
            const size_t n = static_cast<size_t>(w->shape[0]) * static_cast<size_t>(w->shape[1]);
            plan.entries.push_back(
                {w->data, n, static_cast<int>(w->shape[0]), static_cast<int>(w->shape[1])});
            plan.bytes += n;
            plan.rows += static_cast<size_t>(w->shape[0]);
        }
    }
    return plan;
}

}  // namespace imp
