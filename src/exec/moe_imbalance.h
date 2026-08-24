#pragma once

// Per-launch expert imbalance, as arithmetic with no CUDA in it (#1548).
//
// What decides grouped-GEMM cost is max(M_e) at ONE launch: the kernel pads
// every expert to a single M tile, so one hot expert sets the tile for all of
// them and everything below it is padding. imp computed that number on the host
// at three sites, used it to pick the tile, and dropped it; the only surviving
// record was a whole-process activation histogram, which averages exactly the
// skew that matters away.
//
// Split out here so the arithmetic is testable without a GPU or an executor.
// The accumulation lives in GraphExecutor; this is the part that can be wrong.

#include <cstdint>

namespace imp {

struct MoeLaunchRows {
    int32_t max_rows = 0;    // max over experts of rows routed to it
    int64_t total_rows = 0;  // rows over all experts
};

// `offsets` is the exclusive-scan array of length ne + 1 that the routing
// produces, so rows for expert e are offsets[e+1] - offsets[e].
inline MoeLaunchRows moe_launch_rows(const int32_t* offsets, int ne) {
    MoeLaunchRows r;
    if (!offsets || ne <= 0)
        return r;
    for (int e = 0; e < ne; ++e) {
        const int32_t m = offsets[e + 1] - offsets[e];
        if (m > r.max_rows)
            r.max_rows = m;
        r.total_rows += m;
    }
    return r;
}

// The device kernel that actually records this at runtime. Declared here so a
// test can run it against moe_launch_rows() above: the reference is the rule,
// the kernel is what ships, and nothing else makes them agree.
//
// `acc` is [n_layers * 4]: [peak_max, sum_max, sum_rows, launches] per layer.
#ifdef __CUDACC__
__global__ void moe_imbalance_kernel(const int32_t* __restrict__ offsets, unsigned int* __restrict__ acc,
                                     int ne, int layer);
#endif

}  // namespace imp
