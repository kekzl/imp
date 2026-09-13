#pragma once

// gpt-oss checkpoint conversion (#547): HF-MXFP4 packed experts -> imp's native NVFP4 MoE
// cache. Lossless on the nibbles: both formats store e2m1 in linear pair order (element
// 2i = low nibble of byte i, verified against transformers' mxfp4.py unpack). Only SCALES
// differ: MXFP4 = ue8m0 per 32 elements, NVFP4 = e4m3 per 16 elements + one FP32 tensor
// scale/expert; each ue8m0 scale expands to two identical e4m3 micro-scales, normalized
// into e4m3's range by the per-expert tensor scale (clamped+logged on out-of-range blocks).
// gate_up arrives row-INTERLEAVED (g0,u0,g1,u1,...); the converter de-interleaves into
// separate gate/up results for imp's standard MoE machinery.

#include "quant/nvfp4_quant.h"
#include <cstdint>
#include <vector>

namespace imp {

// Converts one packed projection [ne,N,K/32,16] blocks + [ne,N,K/32] ue8m0 scales (host)
// into a device NvFP4MoEQuantResult. row_stride/row_offset select interleaved sub-matrices:
// gate offset0/stride2, up offset1/stride2, down offset0/stride1. extra_scale folds a
// constant into the per-expert tensor scale (gpt-oss's residual 2^-4 rescale on down).
// h_tscales_out (optional): host copy of per-expert FP32 tensor scales for CUTLASS
// per-expert weight registration.
bool gpt_oss_convert_experts_to_nvfp4(const uint8_t* h_blocks, const uint8_t* h_scales, int ne,
                                      int64_t n_rows_total, int64_t K, int row_offset, int row_stride,
                                      NvFP4MoEQuantResult& out, float extra_scale = 1.0f,
                                      std::vector<float>* h_tscales_out = nullptr);

}  // namespace imp
