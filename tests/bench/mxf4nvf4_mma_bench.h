#pragma once

#include <cuda_runtime.h>

namespace imp {

struct MmaBenchResult {
    float legacy_ms;      // avg ms per rep — kind::f8f6f4.m16n8k32
    float blockscale_ms;  // avg ms per rep — kind::mxf4nvf4.block_scale.m16n8k64
    double legacy_tops;   // effective TOPS across all warps
    double blockscale_tops;
    double speedup;  // blockscale_tops / legacy_tops
};

// Raw-MMA throughput microbench for both MMA variants; answers the Project B Stage 4
// integration-effort gate with per-warp effective TOPS.
// warps: resident warps (170 = full sm_120f on RTX 5090). iterations: MMA issues/warp/rep.
MmaBenchResult bench_mma_comparison(int warps, int iterations, cudaStream_t stream);

}  // namespace imp
