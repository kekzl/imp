#pragma once

#include <cstdint>

namespace imp {

enum class StorageTier : uint8_t {
    Undefined = 0,  // handle not yet populated — FATAL if dispatched
    FP32 = 1,
    FP16 = 2,
    FP8 = 3,            // E4M3 with per-tensor scale
    NVFP4 = 4,          // two-level micro-scale, native decode-GEMV path
    CUTLASS_NVFP4 = 5,  // SfAtom NVFP4 on the own tile GEMM (name historical, CUTLASS gone #2938)
    MXFP4 = 6,          // block-scaled MXFP4 weights: gemv_mxfp4_kpar decode, gemm_mxfp4_tile prefill
};

using TierMask = uint32_t;

constexpr TierMask mask(StorageTier t) { return TierMask{1} << static_cast<int>(t); }

constexpr bool mask_contains(TierMask m, StorageTier t) { return (m & mask(t)) != 0; }

}  // namespace imp
