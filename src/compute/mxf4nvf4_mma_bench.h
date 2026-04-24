#pragma once

#include <cuda_runtime.h>

namespace imp {

struct MmaBenchResult {
    float legacy_ms;         // avg ms per rep — kind::f8f6f4.m16n8k32
    float blockscale_ms;     // avg ms per rep — kind::mxf4nvf4.block_scale.m16n8k64
    double legacy_tops;      // effective TOPS across all warps
    double blockscale_tops;
    double speedup;          // blockscale_tops / legacy_tops
};

// Run a raw-MMA throughput microbench for both instructions and return
// per-warp effective TOPS. Answers the "is Project B Stage 4 worth the
// integration effort?" question with a concrete number.
//
// warps: number of resident warps (1 per SM for 170 warps = full
//        sm_120f on RTX 5090; smaller values isolate single-SM perf).
// iterations: MMA issues per warp per rep. 1M is a reasonable default.
MmaBenchResult bench_mma_comparison(int warps, int iterations, cudaStream_t stream);

} // namespace imp
