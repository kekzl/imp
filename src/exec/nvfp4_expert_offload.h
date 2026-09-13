#pragma once

#include "core/tensor.h"

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

namespace imp {

// Host-resident NVFP4 MoE experts: the LRU slot pool works for GGUF (#1370) because one
// expert is one contiguous byte range. NVFP4 is TWO ranges (packed FP4 + FP8 E4M3
// micro-scales); the kernels address them with separate bases/strides:
//   W  = packed_data  + idx * expert_stride_packed
//   MS = micro_scales + idx * expert_stride_ms
// Concatenating both into one slot (packed_data=pool, micro_scales=pool+packed_off(),
// both strides=slot_bytes()) resolves both to the same slot with no kernel changes.
// tensor_scales does NOT fall out this way: kernels read tensor_scales[idx] per-expert,
// so it needs a per-slot device mirror, rewritten whenever a slot's occupant changes.

// Packed weights load through uint2 (nvfp4_gemm_internal.cuh, gemv_nvfp4_row): slot base
// and packed stride must be 8-byte aligned. Micro-scales need no alignment of their own
// but sit behind the packed block, so padding keeps the NEXT slot's packed block aligned.
// 16 covers both with room.
inline constexpr size_t kNvFP4SlotAlign = 16;

inline constexpr size_t nvfp4_align_up(size_t v, size_t a) { return (v + a - 1) / a * a; }

// Byte layout of one cache slot holding a single NVFP4 expert.
struct NvFP4SlotLayout {
    size_t packed_bytes = 0;  // N * K/2, the FP4 weights
    size_t ms_bytes = 0;      // N * K/16, the FP8 E4M3 micro-scales

    // Offset of the micro-scale block within the slot.
    constexpr size_t packed_off() const { return nvfp4_align_up(packed_bytes, kNvFP4SlotAlign); }
    // Total slot size, itself aligned so slot i+1 starts aligned too.
    constexpr size_t slot_bytes() const {
        return nvfp4_align_up(packed_off() + ms_bytes, kNvFP4SlotAlign);
    }
};

// Layout for an [N, K] expert. K must be a multiple of 16 (the micro-block
// size the kernels hard-code); returns an empty layout otherwise, which every
// caller treats as "this path does not apply".
inline NvFP4SlotLayout nvfp4_slot_layout(int64_t N, int64_t K) {
    NvFP4SlotLayout l;
    if (N <= 0 || K <= 0 || K % 16 != 0)
        return l;
    l.packed_bytes = static_cast<size_t>(N) * static_cast<size_t>(K / 2);
    l.ms_bytes = static_cast<size_t>(N) * static_cast<size_t>(K / 16);
    return l;
}

// Single definition of whether a projection's experts can be served from the slot pool:
// NVFP4-promoted, host-resident, carrying micro-scales, shaped like every other expert in
// the projection. #1384 and #1403 were both a predicate standing in front of the check
// meant to catch the problem.
inline bool nvfp4_host_experts_servable(const std::vector<Tensor>& experts) {
    if (experts.empty() || !experts[0].data)
        return false;
    const Tensor& e0 = experts[0];
    if (e0.on_device || e0.qtype != QType::NVFP4 || !e0.scales || e0.ndim != 2)
        return false;
    if (nvfp4_slot_layout(e0.shape[0], e0.shape[1] * 2).slot_bytes() == 0)
        return false;
    for (const Tensor& e : experts) {
        if (!e.data || !e.scales || e.on_device || e.qtype != QType::NVFP4)
            return false;
        if (e.ndim != 2 || e.shape[0] != e0.shape[0] || e.shape[1] != e0.shape[1])
            return false;
    }
    return true;
}

// One projection's experts staged contiguously on device in the layout the NVFP4 GEMMs
// expect: packed + e*packed_stride, ms + e*ms_stride. Empty packed = not staged (e.g. no
// gate on a non-gated model); caller falls back per expert.
struct StagedProj {
    const char* packed = nullptr;
    const char* ms = nullptr;
    size_t packed_stride = 0;
    size_t ms_stride = 0;
    int n_experts = 0;
    // SfAtom scales + per-expert pointer arrays were built for this
    // projection, so it can take the CUTLASS device-args prefill.
    bool cutlass_ready = false;

    bool valid() const { return packed != nullptr && ms != nullptr && n_experts > 0; }
    bool covers(int expert) const { return valid() && expert >= 0 && expert < n_experts; }
};

// nvfp4_scratch_ key naming one per-expert MoE weight: "L{layer}.expert_w_{gate,up,down}.
// {expert}". Dense weights ("L5.wq", "out_proj") deliberately do not parse: no host path,
// so promoting one would label bytes NVFP4 that nothing can serve. One parser because
// Phase 0 and both scale-upload paths ask the same question; parsing twice lets them drift.
struct NvFP4ExpertKey {
    enum class Kind { Gate, Up, Down };
    bool valid = false;
    int layer = -1;
    int expert = -1;
    Kind kind = Kind::Gate;
};

inline NvFP4ExpertKey parse_expert_key(std::string_view key) {
    NvFP4ExpertKey out;
    if (key.size() < 2 || key[0] != 'L')
        return out;
    const size_t dot = key.find('.', 1);
    if (dot == std::string_view::npos || dot == 1)
        return out;

    int layer = 0;
    for (size_t i = 1; i < dot; ++i) {
        if (key[i] < '0' || key[i] > '9')
            return out;
        layer = layer * 10 + (key[i] - '0');
    }

    std::string_view rest = key.substr(dot + 1);
    if (rest.rfind("expert_w_", 0) != 0)
        return out;
    rest.remove_prefix(9);  // "expert_w_"

    NvFP4ExpertKey::Kind kind;
    if (rest.rfind("gate.", 0) == 0) {
        kind = NvFP4ExpertKey::Kind::Gate;
        rest.remove_prefix(5);
    } else if (rest.rfind("up.", 0) == 0) {
        kind = NvFP4ExpertKey::Kind::Up;
        rest.remove_prefix(3);
    } else if (rest.rfind("down.", 0) == 0) {
        kind = NvFP4ExpertKey::Kind::Down;
        rest.remove_prefix(5);
    } else {
        return out;
    }

    if (rest.empty())
        return out;
    int expert = 0;
    for (char c : rest) {
        if (c < '0' || c > '9')
            return out;
        expert = expert * 10 + (c - '0');
    }

    out.valid = true;
    out.layer = layer;
    out.expert = expert;
    out.kind = kind;
    return out;
}

inline bool is_expert_key(std::string_view key) { return parse_expert_key(key).valid; }

}  // namespace imp
