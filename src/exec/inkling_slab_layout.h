#pragma once
// Inkling2-Loop per-slot recurrent slab (B-111 S3): a sequence slot's conv state and the K/V rings of its
// sliding applications at fixed offsets within one slot stride, every slot in one allocation
// (exec/inkling_slab.h), as SSMState holds GDN/Mamba slots. After the slots, one lookahead region per
// lookahead slot (S3b). CUDA-free (tests/test_inkling_slab_layout.cpp).

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
// Rows of the largest captured call = anchor + drafts (InklingForward::kGraphRows).
inline constexpr int kInklingGraphRows = 8;
// Slots with a draft context: 4 with one each reached 31983 MiB on the 32 GB card (397 MiB at 32768).
inline constexpr int kInklingLookaheadSlots = 2;
inline constexpr size_t kInklingSlabNone = SIZE_MAX;

// Rows of a draft local layer's K/V ring (InklingDraft keeps 2 x this): window + block + cap + back.
[[nodiscard]] inline int inkling_draft_ring(const InklingDraftConfig& d, int cap, int back) {
    return d.window + d.block + cap + std::max(back, 0);
}

// One lookahead slot's region: the forward's lookahead planes, the target logits of its stored call and
// the draft's local K/V rings. Byte offsets from the region base, each 256-aligned.
struct InklingLookLayout {
    int taps = 0, seg = 0, vocab = 0, draft_ring = 0;  // what the region was planned for
    size_t feat = 0, conv_pre = 0, conv_tmp = 0, conv_log = 0, look_h = 0, ids = 0, logits = 0;
    std::vector<size_t> dk, dv;  // per draft layer: its K / V ring, kInklingSlabNone = the pool layer
    size_t bytes = 0;            // a multiple of 256, 0 = no lookahead region
};

// A lookahead region of a slab: layout l at base; empty = none.
struct InklingLookRegion {
    const InklingLookLayout* l = nullptr;
    std::byte* base = nullptr;
    explicit operator bool() const { return l && base; }
    template <class T>
    [[nodiscard]] T* at(size_t off) const {
        return off == kInklingSlabNone ? nullptr : reinterpret_cast<T*>(base + off);
    }
};

struct InklingSlabLayout {
    static constexpr size_t kAlign = 256;
    static constexpr size_t kNone = kInklingSlabNone;
    int ring = 0;              // positions of a sliding application's K/V ring
    size_t kv_row_halves = 0;  // n_kv_heads * head_dim: one ring row of K or of V
    // Conv state of one application: k [3, KD], v [3, KD], attn [3, H], mlp [3, H] fp32.
    size_t conv_app_floats = 0;
    size_t conv_bytes = 0;  // the n_apps conv states, contiguous at offset 0
    // Per application: byte offset of its K / V ring, kNone = full attention.
    std::vector<size_t> k_off, v_off;
    size_t slot_bytes = 0;   // one slot, a multiple of kAlign: the stride between slots
    InklingLookLayout look;  // bytes 0 = no lookahead planned

    [[nodiscard]] size_t ring_bytes() const { return static_cast<size_t>(ring) * kv_row_halves * 2; }
    // Lookahead regions a slab of slots slots holds when it holds them at all.
    [[nodiscard]] int look_slots(int slots) const {
        return look.bytes > 0 ? std::clamp(slots, 0, kInklingLookaheadSlots) : 0;
    }
    [[nodiscard]] size_t total_bytes(int slots, int looks = 0) const {
        return slots > 0 ? slot_bytes * static_cast<size_t>(slots) + look.bytes * std::max(looks, 0) : 0;
    }
    // Byte offset of slot s in a slab of slots slots, kNone outside it.
    [[nodiscard]] size_t slot_offset(int s, int slots) const {
        return s >= 0 && s < slots ? slot_bytes * static_cast<size_t>(s) : kNone;
    }
    // Byte offset of slot s's lookahead region (after every slot), kNone when s holds none.
    [[nodiscard]] size_t look_offset(int s, int slots, int looks) const {
        return s >= 0 && s < looks && s < slots && look.bytes > 0
                   ? total_bytes(slots) + look.bytes * static_cast<size_t>(s)
                   : kNone;
    }
};

[[nodiscard]] inline size_t inkling_slab_align(size_t b) {
    return (b + InklingSlabLayout::kAlign - 1) / InklingSlabLayout::kAlign * InklingSlabLayout::kAlign;
}

// Cross-slot prefix snapshot (B-111 S4): conv state at a checkpoint position pos, then per sliding
// application window rows of K and window rows of V, row j = position pos - window + j (a row reads keys back
// to pos - window + 1), then the same for every local layer of the lookahead draft (its window). The same
// bytes for every slot: a slot resumes another slot's sequence, target and draft, from it.
struct InklingSnapLayout {
    size_t conv_bytes = 0;  // InklingSlabLayout::conv_bytes
    size_t app_bytes = 0;   // K and V rows of one sliding application
    int window = 0, sliding = 0;
    size_t draft_off = 0, draft_row = 0, draft_bytes = 0;  // draft part: draft_local layers, 0 = none
    int draft_window = 0, draft_local = 0;
    size_t bytes = 0;  // one entry
};

[[nodiscard]] inline InklingSnapLayout inkling_snap_layout(const InklingSlabLayout& sl, int window,
                                                           const InklingDraftConfig* d = nullptr) {
    InklingSnapLayout s;
    s.window = std::clamp(window, 0, std::max(sl.ring, 0));
    s.conv_bytes = sl.conv_bytes;
    s.app_bytes = 2 * static_cast<size_t>(s.window) * sl.kv_row_halves * 2;
    for (size_t k : sl.k_off)
        s.sliding += k != InklingSlabLayout::kNone ? 1 : 0;
    s.draft_off = s.conv_bytes + s.app_bytes * static_cast<size_t>(s.sliding);
    if (d) {
        for (bool local : d->local)
            s.draft_local += local ? 1 : 0;
        s.draft_window = std::max(d->window, 0);
        s.draft_row = static_cast<size_t>(d->kv_heads) * d->head_dim * 2;
        s.draft_bytes = 2 * static_cast<size_t>(s.draft_local) * static_cast<size_t>(s.draft_window) *
                        s.draft_row;
    }
    s.bytes = s.draft_off + s.draft_bytes;
    return s;
}

// One copy between a slot (byte offset from its base) and a snapshot entry.
struct InklingSnapCopy {
    size_t slot_off = 0, entry_off = 0, bytes = 0;
};

// Rows of positions max(0, pos - window) .. pos - 1 of one ring at ring_off (rows of row bytes, position q at
// row q % ring; ring 0 = row q), entry row j = position pos - window + j from entry_off: two copies where the
// window wraps the ring. Appends to out.
inline void inkling_ring_window_copies(size_t ring_off, int ring, size_t row, int window, int pos,
                                       size_t entry_off, std::vector<InklingSnapCopy>& out) {
    for (int q = std::max(pos - window, 0); q < pos && window > 0;) {
        const int r = ring > 0 ? q % ring : q, n = ring > 0 ? std::min(pos - q, ring - r) : pos - q;
        out.push_back({ring_off + static_cast<size_t>(r) * row,
                       entry_off + static_cast<size_t>(q - (pos - window)) * row,
                       static_cast<size_t>(n) * row});
        q += n;
    }
}

// The target's ring copies of the window before pos: K, then V of every sliding application.
inline void inkling_snap_ring_copies(const InklingSlabLayout& sl, int window, int pos,
                                     std::vector<InklingSnapCopy>& out) {
    out.clear();
    const InklingSnapLayout s = inkling_snap_layout(sl, window);
    const size_t row = sl.kv_row_halves * 2;
    size_t app = s.conv_bytes;
    for (size_t a = 0; a < sl.k_off.size() && pos > 0; ++a) {
        if (sl.k_off[a] == InklingSlabLayout::kNone)
            continue;
        inkling_ring_window_copies(sl.k_off[a], sl.ring, row, s.window, pos, app, out);
        inkling_ring_window_copies(sl.v_off[a], sl.ring, row, s.window, pos, app + s.app_bytes / 2, out);
        app += s.app_bytes;
    }
}

// Conv-state checkpoints a slot keeps (kGrid apart): a rewind reaches back ring - window positions and a
// segment stores its checkpoints before the high-water mark moves (seg of margin), plus one either end.
[[nodiscard]] inline int inkling_checkpoint_slots(const InklingSlabLayout& sl, int window, int seg,
                                                  int grid) {
    return grid > 0 ? (std::max(sl.ring - window, 0) + std::max(seg, 0)) / grid + 2 : 0;
}

// The lookahead region of the target c (conv_app_floats per application) with draft d: feature planes
// [taps][seg][H] fp32, conv state before the call, a scratch copy, the per-row conv log, the call's states,
// token ids, logits [kInklingGraphRows][vocab] fp32, then K and V of every local draft layer.
[[nodiscard]] inline InklingLookLayout inkling_look_layout(const InklingConfig& c, size_t conv_app_floats,
                                                           int seg, const InklingDraftConfig& d, int vocab) {
    InklingLookLayout l;
    if (d.taps.size() < 2 || vocab < 1)
        return l;
    const size_t H = static_cast<size_t>(c.hidden), R = kInklingGraphRows;
    const size_t n = conv_app_floats * static_cast<size_t>(c.n_apps());
    l.taps = static_cast<int>(d.taps.size());
    l.seg = seg;
    l.vocab = vocab;
    l.draft_ring = inkling_draft_ring(d, seg, kInklingSeg + kInklingSlack);
    size_t off = 0;
    const auto put = [&off](size_t bytes) {
        const size_t at = off;
        off += inkling_slab_align(bytes);
        return at;
    };
    l.feat = put(d.taps.size() * static_cast<size_t>(seg) * H * sizeof(float));
    l.conv_pre = put(n * sizeof(float));
    l.conv_tmp = put(n * sizeof(float));
    l.conv_log = put(n / 3 * R * sizeof(float));
    l.look_h = put(H * R * 2);
    l.ids = put(R * sizeof(int32_t));
    l.logits = put(R * static_cast<size_t>(vocab) * sizeof(float));
    const size_t row = static_cast<size_t>(d.kv_heads) * d.head_dim * 2;  // fp16 K or V of one position
    const size_t ring = 2 * static_cast<size_t>(l.draft_ring) * row;
    for (int i = 0; i < d.layers; ++i) {
        const bool local = static_cast<size_t>(i) < d.local.size() && d.local[static_cast<size_t>(i)];
        l.dk.push_back(local ? put(ring) : kInklingSlabNone);
        l.dv.push_back(local ? put(ring) : kInklingSlabNone);
    }
    l.bytes = off;
    return l;
}

// max_tokens: rows of one forward call; the ring keeps window + one segment + kInklingSlack positions.
// draft (null = no lookahead): plans a lookahead region with vocab logits per row.
[[nodiscard]] inline InklingSlabLayout inkling_slab_layout(const InklingConfig& c, int max_tokens,
                                                           const InklingDraftConfig* draft = nullptr,
                                                           int vocab = 0) {
    InklingSlabLayout l;
    const size_t KD = static_cast<size_t>(c.n_kv_heads) * c.head_dim, H = static_cast<size_t>(c.hidden);
    const int seg = std::min(std::max(max_tokens, 0), kInklingSeg);
    l.ring = c.window + seg + kInklingSlack;
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
    if (draft)
        l.look = inkling_look_layout(c, l.conv_app_floats, seg, *draft, vocab);
    return l;
}

}  // namespace imp
