#pragma once

#include <cuda_runtime.h>

namespace imp {

struct MmaPeakResult {
    double fp4_blockscale_tops;  // mxf4nvf4.block_scale m16n8k64, f32 acc
    double fp16_f16acc_tflops;   // m16n8k16 f16.f16.f16.f16
    double fp16_f32acc_tflops;   // m16n8k16 f32.f16.f16.f32
    double fp8_f32acc_tops;      // m16n8k32 f32.e4m3.e4m3.f32
};

// SATURATED tensor-core peak per dtype: 8 warps/SM x 4 independent
// accumulator chains (vs. mxf4nvf4_mma_bench's 1 warp/SM serial chain,
// which is latency-bound and reads ~7x low). This is the empirical
// compute-roofline ceiling used to calibrate tools/roofline/config.json
// (issues #595/#596): on GeForce sm_120, f32 accumulate runs at 1/4 the
// f16-accumulate rate, and the FP4 block-scale path peaks at HALF the
// 3354-TOPS datasheet number (~2019 TOPS measured at ~2.85 GHz boost).
MmaPeakResult bench_mma_peak_saturated(cudaStream_t stream);

}  // namespace imp
