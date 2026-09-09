#pragma once

// One formula for the recurrent (SSM/GDN) state footprint, and the message an
// operator reads when the pool does not fit.
//
// The formula lived twice and the two copies disagreed. `runtime/vram_budget.cpp`
// charged `conv_channels * (conv_kernel - 1) * 4` with no alignment; the
// allocator in `memory/ssm_state.cu` took
// `align256(conv_channels * conv_kernel * 4) + align256(h bytes)` per layer.
// On Qwen3.8-27B-NVFP4 (48 GDN layers, conv_kernel 4, 64 slots) that is 4968 MiB
// planned against 5088 MiB allocated: the plan was 120 MiB short of the
// allocation it was supposed to bound, in the direction that oversubscribes the
// card (docs/internals/MEMORY.md D14).
//
// The allocator's shape is the one that wins, because it is the one that
// reaches cudaMalloc. Header-only and free of CUDA so the CPU lane pins it.

#include "core/qtype.h"

#include <cstddef>
#include <format>
#include <string>

namespace imp {

// The geometry `SSMState::init` is called with, as plain scalars.
struct SsmStateGeometry {
    int n_ssm_layers = 0;   // GDN/Mamba layers only, NOT the model's layer count
    int conv_channels = 0;  // ModelConfig::ssm_conv_channels()
    int conv_kernel = 0;
    int n_heads = 0;     // ModelConfig::ssm_dt_rank
    int head_dim = 0;    // ssm_inner_size / n_heads
    int state_size = 0;  // ModelConfig::ssm_state_size
    QType h_dtype = QType::F32;
};

// Every sub-allocation of the slab is 256-byte aligned, so the pool is bigger
// than the raw tensor bytes and the plan has to charge the padded figure.
inline size_t ssm_align256(size_t bytes) { return (bytes + 255) & ~static_cast<size_t>(255); }

// conv state: always FP32 (small, needs the precision), one window of
// conv_kernel taps. The full kernel width, not kernel-1: the slab holds the
// tap the next step writes.
inline size_t ssm_conv_bytes_per_layer(const SsmStateGeometry& g) {
    return ssm_align256(static_cast<size_t>(g.conv_channels) * static_cast<size_t>(g.conv_kernel) *
                        sizeof(float));
}

// recurrent h state, in the configured storage dtype (F32 or F16).
inline size_t ssm_h_bytes_per_layer(const SsmStateGeometry& g) {
    return ssm_align256(static_cast<size_t>(g.n_heads) * static_cast<size_t>(g.head_dim) *
                        static_cast<size_t>(g.state_size) * qtype_elem_bytes(g.h_dtype));
}

inline size_t ssm_bytes_per_layer(const SsmStateGeometry& g) {
    return ssm_conv_bytes_per_layer(g) + ssm_h_bytes_per_layer(g);
}

// One scheduler slot: every GDN layer's conv + h state, contiguous.
inline size_t ssm_bytes_per_slot(const SsmStateGeometry& g) {
    return ssm_bytes_per_layer(g) * static_cast<size_t>(g.n_ssm_layers > 0 ? g.n_ssm_layers : 0);
}

// The whole pool: the scheduler's live slots plus the slots the
// multi-candidate speculative verify reserves past them.
inline size_t ssm_pool_bytes(const SsmStateGeometry& g, int slots, int reserved_slots) {
    const int n = (slots > 0 ? slots : 0) + (reserved_slots > 0 ? reserved_slots : 0);
    return ssm_bytes_per_slot(g) * static_cast<size_t>(n);
}

// What the operator is told when the pool is refused. Pure, so the CPU lane can
// pin the four things that make the refusal actionable: which pool, how big,
// how many slots it was for, and the knob that shrinks it. The old message was
// `Failed to allocate SSM state pool (%zu bytes)` - a byte count with no slot
// count, no free figure and no lever.
inline std::string ssm_pool_failure_message(size_t bytes, int slots, int reserved_slots, int n_ssm_layers,
                                            size_t free_bytes) {
    constexpr double kMiB = 1024.0 * 1024.0;
    const int total_slots = (slots > 0 ? slots : 0) + (reserved_slots > 0 ? reserved_slots : 0);
    const double per_slot = total_slots > 0 ? static_cast<double>(bytes) / total_slots / kMiB : 0.0;
    return std::format(
        "SSM/GDN state pool refused: {:.0f} MiB for {} slots ({} live + {} reserved) x {} layers, "
        "{:.1f} MiB per slot, {:.0f} MiB free. Lower runtime.max_batch_size (each slot costs "
        "{:.1f} MiB) or free VRAM; a hybrid model cannot run without recurrent state.",
        bytes / kMiB, total_slots, slots > 0 ? slots : 0, reserved_slots > 0 ? reserved_slots : 0,
        n_ssm_layers, per_slot, free_bytes / kMiB, per_slot);
}

}  // namespace imp
