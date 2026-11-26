#pragma once

#include <cstdint>

namespace imp {

// The small-M A4 scratch holds quantize(src[0..m, 0..k), scale). A producer stamp (fused norm or
// activation + quantize) lets a GEMM on src skip its quantize until src is touched; view_tokens
// touches every buffer it hands out, a view may be a writer (D-6: no forget call per writer).
struct SmallMXqTag {
    const void* src = nullptr;
    int m = 0;
    int k = 0;
    float scale = 1.0f;
    uint64_t producer_gen = 0;      // view_gen at the producer's stamp; 0 = the dispatch's own quantize
    mutable uint64_t view_gen = 1;  // +1 per touch of src, so a stamp never reads 0

    [[nodiscard]] bool holds(const void* d, int rows, int cols, float s) const {
        return src == d && m == rows && k == cols && scale == s;
    }
    // A producer wrote src and the scratch in one launch and nothing touched src since.
    [[nodiscard]] bool producer_live() const { return producer_gen == view_gen; }
    void stamp_producer(const void* d, int rows, int cols, float s) {
        set_(d, rows, cols, s);
        producer_gen = view_gen;
    }
    void stamp_quantize(const void* d, int rows, int cols, float s) {
        set_(d, rows, cols, s);
        producer_gen = 0;
    }
    void clear() {
        src = nullptr;
        producer_gen = 0;
    }
    // A view handed out of, or a write to, `base`: a producer stamp on it is no longer live.
    void touch(const void* base) const {
        if (base != nullptr && base == src)
            ++view_gen;
    }
    // A consumer op that skipped its norm on a prenorm match takes over the producer's stamp, which
    // its own view of src retired. `stamped`: producer_gen as the producer left it (0 = no quantize).
    void adopt(const void* d, uint64_t stamped) {
        if (stamped != 0 && stamped == producer_gen && src == d)
            producer_gen = view_gen;
    }

private:
    void set_(const void* d, int rows, int cols, float s) {
        src = d;
        m = rows;
        k = cols;
        scale = s;
    }
};

}  // namespace imp
