#pragma once
// Inkling2-Loop per-slot recurrent slab (B-111 S3): a sequence slot's conv state and the K/V rings of its
// sliding applications at fixed offsets within one slot stride, every slot in one allocation
// (exec/inkling_slab.h), as SSMState holds GDN/Mamba slots. CUDA-free (tests/test_inkling_slab_layout.cpp).

#include "model/inkling_config.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace imp {

// Rows of one forward segment, and sliding K/V kept behind the window: how far a rewind may go back
// (InklingForward::kSeg, kSlack).
inline constexpr int kInklingSeg = 1024;
inline constexpr int kInklingSlack = 5120;

struct InklingSlabLayout {
    static constexpr size_t kAlign = 256;
    static constexpr size_t kNone = SIZE_MAX;
    int ring = 0;              // positions of a sliding application's K/V ring
    size_t kv_row_halves = 0;  // n_kv_heads * head_dim: one ring row of K or of V
    // Conv state of one application: k [3, KD], v [3, KD], attn [3, H], mlp [3, H] fp32.
    size_t conv_app_floats = 0;
    size_t conv_bytes = 0;  // the n_apps conv states, contiguous at offset 0
    // Per application: byte offset of its K / V ring, kNone = full attention.
    std::vector<size_t> k_off, v_off;
    size_t slot_bytes = 0;  // one slot, a multiple of kAlign: the stride between slots

    [[nodiscard]] size_t ring_bytes() const { return static_cast<size_t>(ring) * kv_row_halves * 2; }
    [[nodiscard]] size_t total_bytes(int slots) const {
        return slots > 0 ? slot_bytes * static_cast<size_t>(slots) : 0;
    }
    // Byte offset of slot s in a slab of slots slots, kNone outside it.
    [[nodiscard]] size_t slot_offset(int s, int slots) const {
        return s >= 0 && s < slots ? slot_bytes * static_cast<size_t>(s) : kNone;
    }
};

[[nodiscard]] inline size_t inkling_slab_align(size_t b) {
    return (b + InklingSlabLayout::kAlign - 1) / InklingSlabLayout::kAlign * InklingSlabLayout::kAlign;
}

// max_tokens: rows of one forward call; the ring keeps window + one segment + kInklingSlack positions.
[[nodiscard]] inline InklingSlabLayout inkling_slab_layout(const InklingConfig& c, int max_tokens) {
    InklingSlabLayout l;
    const size_t KD = static_cast<size_t>(c.n_kv_heads) * c.head_dim, H = static_cast<size_t>(c.hidden);
    l.ring = c.window + std::min(std::max(max_tokens, 0), kInklingSeg) + kInklingSlack;
    l.kv_row_halves = KD;
    l.conv_app_floats = 3 * (2 * KD + 2 * H);
    l.conv_bytes = inkling_slab_align(l.conv_app_floats * static_cast<size_t>(c.n_apps()) * sizeof(float));
    size_t off = l.conv_bytes;
    const size_t rb = inkling_slab_align(l.ring_bytes());
    for (int a = 0; a < c.n_apps(); ++a) {
        const size_t layer = static_cast<size_t>(c.app_layer(a));
        const bool global = layer < c.global.size() && c.global[layer];
        l.k_off.push_back(global ? InklingSlabLayout::kNone : off);
        off += global ? 0 : rb;
        l.v_off.push_back(global ? InklingSlabLayout::kNone : off);
        off += global ? 0 : rb;
    }
    l.slot_bytes = inkling_slab_align(off);
    return l;
}

}  // namespace imp
