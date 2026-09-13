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

// Benchmarks all sm120 FP4/FP8 MMA variants CUTLASS headers expose.
// warps: number of 32-thread CTAs; iterations: tight-loop iterations per warp.
// entries[i].tops < 0 means the variant failed to launch (missing CUTE_ARCH_* define or
// invalid PTX).
MmaVariantsBenchResult bench_mma_variants(int warps, int iterations, cudaStream_t stream);

}  // namespace imp
