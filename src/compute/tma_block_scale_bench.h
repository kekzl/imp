#pragma once
#include <cstdint>

namespace imp {

struct TmaBlockScaleResult {
    double ms_separate;   // separate cp.async streams for FP4 data + UE4M3 scales
    double ms_fused;      // interleaved / co-issued cp.async for data + scales
    double bytes_loaded;  // total bytes per iteration (data + scales, both variants equal)
};

// Microbench: load 16 KiB of FP4 data + 1 KiB of UE4M3 block scales `iters` times.
// Compares two-stream-sequential vs fused-interleaved cp.async patterns.
// On SM120 the fused variant should be >5% faster if the HW can pipeline
// the descriptor fetches; otherwise the spec assumption needs revisiting.
TmaBlockScaleResult bench_tma_block_scale(int iters = 1024);

}  // namespace imp
