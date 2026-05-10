#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace imp {

// Per-expert FP16 -> NVFP4 quantize, native row-major UE4M3 scale layout.
// Input:  [expanded, K] FP16 activations, expert_offsets[ne+1] partitions rows.
// Output: per-expert packed NVFP4 (FP4 nibbles, K/2 bytes per row) +
//         per-expert UE4M3 scales (K/16 bytes per row), both row-major dense.
//
// d_packed_ptrs[e] points to a [M_e, K/2] tightly-packed FP4 buffer.
// d_sf_ptrs[e]     points to a [M_e, K/16] UE4M3 row-major buffer.
// expert_offsets[e..e+1] gives the row range in src_fp16.
//
// Two-level scaling matches quantize_fp16_to_nvfp4 (nvfp4_quant.cu):
//   tensor_scale   = per-expert absmax / 6.0  (FP32)
//   micro_scale    = local_absmax / (tensor_scale * 6.0), encoded FP8 UE4M3
//   quantized FP4  = val / (tensor_scale * micro_scale_actual), E2M1 HW sat
//
// Layout convention (native, no SfAtom padding):
//   packed[m][k/2] = packed nibbles (low=even, high=odd element index)
//   sf[m][kb]      = UE4M3 byte for elements [kb*16, kb*16+15]
//   where kb = k_block = k/16
//
// This layout is read directly by gemm_grouped_nvfp4_smallM (cache_moe_native_nvfp4 /
// nvfp4_moe_ms_native buffers). See bench/sm120_smallM_audit.md.
void quantize_fp16_to_nvfp4_moe_native(
    const __half* src_fp16,              // [expanded, K]
    void* const* d_packed_ptrs,          // [n_experts] per-expert packed FP4
    void* const* d_sf_ptrs,              // [n_experts] per-expert UE4M3
    const int* d_expert_offsets,         // [n_experts + 1] device pointer
    int expanded,
    int K,
    int n_experts,
    cudaStream_t stream);

}  // namespace imp
