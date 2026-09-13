#pragma once

// Fused-projection scale split provenance. A checkpoint may ship q|k|v as one qkv_proj or
// gate|up as one gate_up_proj; weight_map.cpp splits the DATA per-slot and SLICES the
// [rows, K/16] scale plane by output-row range, but siblings share one
// weight_global_scale (the only per-tensor number they have in common).
// Two distinct facts:
//   fused:           weight came from one checkpoint tensor (says nothing about scale
//                     layout; each scratch entry gets its own cudaMallocAsync).
//   spans_one_plane: the loader's fix-up arm wrote sibling pointers as offsets into the
//                     base's plane (the only case a row offset is meaningful).
// Conflating them let an unrelated promotion failure point a sibling's micro-scales into
// the base's plane with the base's global scale (Qwen3.8-27B: w_up read 5.57 MB past
// w_gate's plane, logged as a normal split).

#include <cstdint>
#include <cstring>
#include <string>

namespace imp {

// What the fix-up arm knows before it writes a sibling's scale pointer.
struct FusedSplitRequest {
    // Set by the ONLY producer of a real split: weight_map.cpp's `qkv_proj` /
    // `gate_up_proj` branches, recorded per layer. False means the checkpoint
    // carried separate tensors and there is nothing to repair.
    bool provenance = false;
    int64_t base_rows = 0;      // rows of the sibling that owns the scale plane
    int64_t sib_rows = 0;       // rows of each sibling being repaired
    int n_sibs = 0;             // qkv repairs k and v (2), gate_up repairs up (1)
    int64_t base_k_packed = 0;  // shape[1] of the base weight (K/2 bytes)
    int64_t sib_k_packed = 0;   // shape[1] of a repaired sibling
    int64_t plane_rows = 0;     // rows of the base's weight_scale tensor, 0 = unknown
};

// Rows the fused scale plane must cover for the split to be in bounds.
inline int64_t fused_split_needed_rows(const FusedSplitRequest& r) {
    return r.base_rows + static_cast<int64_t>(r.n_sibs) * r.sib_rows;
}

// True when the fix-up arm may fire. Declines are not errors: a checkpoint with separate
// tensors declines every layer and serves normally. On today's producers the arm cannot
// fire in either direction (weight_map already slices per sibling): kept as the repair
// path for a fused layout whose scale plane is NOT sliced per sibling, and to keep the
// wrong repair impossible.
inline bool fused_split_eligible(const FusedSplitRequest& r, std::string* why_not) {
    auto no = [&](const char* m) {
        if (why_not)
            *why_not = m;
        return false;
    };
    if (!r.provenance)
        return no("no fused-projection split on this layer (weight_map saw separate tensors)");
    if (r.n_sibs < 1 || r.base_rows <= 0 || r.sib_rows <= 0)
        return no("degenerate row counts");
    if (r.base_k_packed <= 0 || r.base_k_packed != r.sib_k_packed)
        return no("siblings disagree about K, so no single scale plane describes both");
    if (r.plane_rows <= 0)
        return no("the fused scale plane's row count is unknown");
    if (r.plane_rows < fused_split_needed_rows(r))
        return no(
            "the fused scale plane covers only the base's rows, so it was already sliced per "
            "sibling and there is nothing to repair");
    return true;
}

// One sibling after promotion, as the assertion sees it.
struct MergedScaleMember {
    const void* scales = nullptr;
    float tensor_scale = 0.0f;
    int64_t rows = 0;
    // Rows of THIS member's own weight_scale tensor, read back from the promote
    // scratch. 0 = the member has no scratch entry of its own, which is what the
    // repaired siblings of a `spans_one_plane` group look like.
    int64_t plane_rows = 0;
};

// A {wq,wk,wv} or {w_gate,w_up} group of one layer. Member 0 owns the plane in
// the `spans_one_plane` case.
struct MergedScaleGroup {
    int layer = -1;
    const char* what = "";
    int count = 0;
    MergedScaleMember m[3];
    bool fused = false;            // the weight came from one checkpoint tensor
    bool spans_one_plane = false;  // the fix-up arm wrote m[1..]'s pointers into m[0]'s plane
    int64_t scale_row_bytes = 0;   // K_packed / 8
};

// Load-time assertion: false + err when the group's scales can't be explained by the
// recorded provenance. Non-fused groups get the one check that can't false-positive
// (two distinct allocations never share an address); offsets are checked only when the
// fix-up arm actually wrote them, since a tidy allocator placing two planes adjacently
// must not be refused.
inline bool merged_scale_group_ok(const MergedScaleGroup& g, std::string* err) {
    auto fail = [&](const std::string& m) {
        if (err)
            *err = "layer " + std::to_string(g.layer) + " " + g.what + ": " + m;
        return false;
    };
    for (int i = 1; i < g.count; ++i)
        for (int j = 0; j < i; ++j)
            if (g.m[i].scales != nullptr && g.m[i].scales == g.m[j].scales)
                return fail(
                    "two siblings share the same scale pointer, so one of them reads the "
                    "other's micro-scales row for row");
    if (!g.fused)
        return true;

    for (int i = 1; i < g.count; ++i) {
        // Bit equality, not a tolerance: the siblings were divided by ONE
        // weight_global_scale, so any difference means one of them was promoted
        // against a scale the checkpoint never gave it.
        if (std::memcmp(&g.m[i].tensor_scale, &g.m[0].tensor_scale, sizeof(float)) != 0)
            return fail("sibling " + std::to_string(i) + " carries tensor_scale " +
                        std::to_string(g.m[i].tensor_scale) + " but the fused tensor's is " +
                        std::to_string(g.m[0].tensor_scale));
    }

    if (!g.spans_one_plane) {
        // Sliced layout: every sibling owns a plane of exactly its own rows. Too small under-runs
        // into the next allocation; too large means the slice never happened and the sibling
        // reads its neighbours' rows as its own.
        for (int i = 0; i < g.count; ++i) {
            if (g.m[i].plane_rows > 0 && g.m[i].plane_rows != g.m[i].rows)
                return fail("sibling " + std::to_string(i) + " has " + std::to_string(g.m[i].rows) +
                            " rows but a scale plane of " + std::to_string(g.m[i].plane_rows));
        }
        return true;
    }

    const char* base = static_cast<const char*>(g.m[0].scales);
    if (base == nullptr)
        return fail("split from one fused tensor but the base carries no scales");
    if (g.scale_row_bytes <= 0)
        return fail("scale row stride is not known");
    int64_t off_rows = g.m[0].rows;
    for (int i = 1; i < g.count; ++i) {
        const MergedScaleMember& s = g.m[i];
        if (s.scales == nullptr)
            return fail("sibling " + std::to_string(i) + " lost its scales after the split");
        const char* want = base + off_rows * g.scale_row_bytes;
        if (static_cast<const char*>(s.scales) != want)
            return fail("sibling " + std::to_string(i) + " points " +
                        std::to_string(static_cast<const char*>(s.scales) - base) +
                        " bytes into the plane, the row offset says " +
                        std::to_string(off_rows * g.scale_row_bytes));
        off_rows += s.rows;
    }
    if (g.m[0].plane_rows > 0 && off_rows > g.m[0].plane_rows)
        return fail("the split addresses " + std::to_string(off_rows) + " rows of a plane holding " +
                    std::to_string(g.m[0].plane_rows));
    return true;
}

}  // namespace imp
