#pragma once
#include <cstdint>

namespace imp {

struct TmaBlockScaleResult {
    double ms_separate;   // separate cp.async streams for FP4 data + UE4M3 scales
    double ms_fused;      // interleaved / co-issued cp.async for data + scales
    double bytes_loaded;  // total bytes per iteration (data + scales, both variants equal)
};

// Microbench: load 16KiB FP4 data + 1KiB UE4M3 scales `iters` times; compares
// two-stream-sequential vs fused-interleaved cp.async.
// Fused should be >5% faster if HW pipelines descriptor fetches; else revisit the spec.
TmaBlockScaleResult bench_tma_block_scale(int iters = 1024);

}  // namespace imp
