#pragma once

#include <cuda_runtime.h>

namespace imp {

struct MmaVariantsBenchResult {
    static constexpr int kMaxEntries = 16;
    struct Entry {
        const char* label;
        float ms;
        double tops;
    };
    Entry entries[kMaxEntries];
    int count = 0;
};

// Benchmarks all sm120 FP4/FP8 MMA variants exposed by CUTLASS headers.
// `warps`: number of warps (CTAs of 32 threads each) to launch.
// `iterations`: tight-loop iterations per warp.
// Result.entries[i].tops < 0 indicates the variant failed to launch
// (likely missing CUTE_ARCH_*_ENABLED defines or invalid PTX).
MmaVariantsBenchResult bench_mma_variants(int warps, int iterations,
                                           cudaStream_t stream);

} // namespace imp
