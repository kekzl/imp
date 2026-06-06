#pragma once

// gpt-oss checkpoint conversion (issue #547): HF-MXFP4 packed experts →
// imp's native NVFP4 MoE cache (NvFP4MoEQuantResult).
//
// Why this works losslessly on the nibbles: both formats store e2m1 4-bit
// values in linear pair order (element 2i = low nibble of byte i — verified
// against transformers' integrations/mxfp4.py unpack and imp's split-layout
// dequant). Only the SCALES differ: MXFP4 = ue8m0 per 32 elements,
// NVFP4 = e4m3 per 16 elements + one FP32 tensor scale per expert. Each
// ue8m0 scale expands to two identical e4m3 micro-scales; the per-expert
// tensor scale normalizes the ue8m0 range into e4m3's dynamic range
// (clamped + logged on the rare out-of-range block).
//
// gate_up arrives row-INTERLEAVED (g0,u0,g1,u1,… along N — HF slices the
// output ::2/1::2); the converter de-interleaves into separate gate and up
// results so imp's standard MoE machinery applies unchanged.

#include "quant/nvfp4_quant.h"
#include <cstdint>

namespace imp {

// Convert one packed projection [ne, N, K/32, 16] blocks + [ne, N, K/32]
// ue8m0 scales (host memory) into a device NvFP4MoEQuantResult.
// row_stride/row_offset select interleaved sub-matrices:
//   gate: offset 0, stride 2;  up: offset 1, stride 2;  down: offset 0, stride 1.
// Returns false on allocation failure.
bool gpt_oss_convert_experts_to_nvfp4(const uint8_t* h_blocks, const uint8_t* h_scales, int ne,
                                      int64_t n_rows_total, int64_t K, int row_offset, int row_stride,
                                      NvFP4MoEQuantResult& out);

}  // namespace imp
