#pragma once

// FP8 E4M3 MoE decode GEMV with 128x128 block scales (DeepSeek `weight_scale_inv` layout), used by
// the Qwen4Exp MTP head's 512 routed experts. Dequant: W[r, k] = fp8(W[r, k]) * scale[r / 128][k / 128].
//
//   y[s, p * rows + r] = sum_k W[e_s, p][r, k] * x[s * x_stride + k]      e_s = expert_ids[s]
//
// One warp per output row, FP32 accumulation, FP16 output. rows and K must be multiples of 128.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace imp {

inline constexpr int kFp8BlockSize = 128;

struct Fp8BlockMoeArgs {
    const uint8_t* const* w = nullptr;  // device table [n_experts * n_proj], each [rows, K] E4M3
    const float* scales = nullptr;      // [n_experts * n_proj, rows / 128, K / 128] FP32
    const int32_t* expert_ids = nullptr;  // [top_k] device
    const half* x = nullptr;
    int x_stride = 0;  // 0: one input shared by every slot
    half* y = nullptr;  // [top_k, n_proj * rows]
    int rows = 0;
    int K = 0;
    int n_proj = 1;
    int top_k = 0;
};

// False (nothing launched) when a shape is not a multiple of 128 or a pointer is null.
bool gemv_fp8_block_moe(const Fp8BlockMoeArgs& a, cudaStream_t stream);

// act[s, j] = silu(gu[s, j]) * gu[s, eff + j] for the packed [top_k, 2 * eff] gate|up output.
void swiglu_packed_rows(const half* gu, half* act, int eff, int top_k, cudaStream_t stream);

}  // namespace imp
