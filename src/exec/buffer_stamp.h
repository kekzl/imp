#pragma once

#include <cstdint>

namespace imp {

// A producer launch wrote `out` (n rows, for the norm weight `key`) so the next block can skip its own
// launch. view_tokens touches every buffer it hands out: a view of `out` retires the stamp (it may be a
// writer), except the consumer block's entry view, which carries it (D-6: no forget call per writer).
template <class Payload>
struct BufferStamp {
    const void* out = nullptr;
    const void* key = nullptr;
    int n = 0;
    Payload payload{};
    uint64_t gen = 0;               // view_gen at the stamp; 0 = no stamp
    mutable uint64_t view_gen = 1;  // +1 per touch of out, so a stamp never reads 0

    void stamp(const void* o, const void* k, int rows, Payload p = {}) {
        out = o;
        key = k;
        n = rows;
        payload = p;
        gen = view_gen;
    }
    [[nodiscard]] bool live() const { return out != nullptr && gen == view_gen; }
    [[nodiscard]] bool matches(const void* o, const void* k, int rows) const {
        return live() && out == o && key == k && n == rows;
    }
    void touch(const void* base) const {
        if (base != nullptr && base == out)
            ++view_gen;
    }
    // After the consumer block's entry view: a stamp live before that view stays live.
    void carry(bool was_live) {
        if (was_live)
            gen = view_gen;
    }
    void clear() {
        out = nullptr;
        key = nullptr;
        n = 0;
        payload = {};
        gen = 0;
    }
};

// The prenorm producer's extras: its small-M activation stamp (0 = it quantized nothing) and the
// MoE router input it wrote in the same launch (null = none).
struct PrenormPayload {
    uint64_t xq_gen = 0;
    const void* router = nullptr;
};

}  // namespace imp
