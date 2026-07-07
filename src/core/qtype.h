#pragma once

#include <cstdint>
#include <cstddef>

namespace imp {

// Unified data-type tag for tensors. Replaces the previous split between
// the compute-side dtype enum and the block-quant qtype enum.
//
// Wire-stable values 0..31 match the on-disk block-quant numbering used by
// GGUF and similar formats. Values >= 64 are engine-internal types that have
// no on-disk representation.
enum class QType : uint16_t {
    // Wire-stable 0..31 (kompatibel mit GGUF block-quant Wire-Format)
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    IQ4_NL = 20,
    IQ4_XS = 23,
    BF16 = 30,
    MXFP4 = 31,

    // Engine-internal (>= 64, never collides with future wire values)
    NONE = 64,
    FP8_E4M3 = 65,
    FP8_E5M2 = 66,
    INT8 = 67,
    INT4 = 68,
    INT32 = 69,
    FP4_E2M1 = 70,
    NVFP4 = 71,
    MXFP4_KV = 72,
    // Values 80 and 81 (TURBOQUANT / TURBOQUANT_LITE) were removed in Phase 5
    // (2026-05-17). The C-API constants IMP_DTYPE_TURBOQUANT(_LITE) are kept
    // for ABI compatibility but map to MXFP4_KV at the imp_api.cpp boundary.
};

// True for block-quantised disk formats (Q*_K, Q*_0, Q*_1, IQ4_*, MXFP4, NVFP4).
// These types require a Sidecar scales tensor for reconstruction.
constexpr bool is_block_quant(QType q) {
    auto v = static_cast<uint16_t>(q);
    return (v >= 2 && v <= 15) || q == QType::IQ4_NL || q == QType::IQ4_XS || q == QType::MXFP4 ||
           q == QType::NVFP4;
}

// True for compute-side dtypes that map directly to a hardware FP/INT type.
// These tensors store raw values; no Sidecar scales needed.
constexpr bool is_compute_dtype(QType q) {
    switch (q) {
        case QType::F32:
        case QType::F16:
        case QType::BF16:
        case QType::FP8_E4M3:
        case QType::FP8_E5M2:
        case QType::INT8:
        case QType::INT4:
        case QType::INT32:
        case QType::FP4_E2M1:
            return true;
        default:
            return false;
    }
}

// Does this source qtype benefit from an NVFP4 decode-cache conversion?
// (> 4.5 bits/elem, or no fast native decode kernel.) Single source of truth
// for the pre-dequant phases AND the VRAM-budget heuristic — keep any policy
// change here so the two can't drift.
inline bool nvfp4_beneficial(QType qt, bool decode_all = false) {
    switch (qt) {
        case QType::Q8_0:
        case QType::Q8_K:
        case QType::Q6_K:
        case QType::Q5_K:
            return true;
        case QType::Q4_K:
        case QType::Q3_K:
        case QType::Q2_K:
            return decode_all;
        case QType::IQ4_NL:
        case QType::IQ4_XS:
            // i-quants have no dp4a/MMVQ decode kernels — without the NVFP4
            // decode cache they fall to dequant->cuBLAS GEMV (uncapturable
            // under graph capture, ~2x slower). Similar bit-width to Q4_K,
            // but here the conversion is the only fast decode path.
            return true;
        default:
            return false;
    }
}

// Bytes-per-row for a tensor of `cols` elements at the given qtype.
// For block-quantised types this includes the per-block overhead.
// Returns 0 for QType::NONE or unknown types.
size_t qtype_row_bytes(QType q, int64_t cols);

// Bytes-per-element for compute dtypes. INT4/FP4_E2M1 return 1 (two
// values packed per byte — caller must account for the packing).
size_t qtype_elem_bytes(QType q);

// Human-readable name (for logs/debug).
const char* qtype_name(QType q);

// Compile-time wire-format stability assertions. If a future change shifts
// any value, the assert fires at build time.
static_assert(static_cast<uint16_t>(QType::F32) == 0, "QType::F32 wire value");
static_assert(static_cast<uint16_t>(QType::F16) == 1, "QType::F16 wire value");
static_assert(static_cast<uint16_t>(QType::Q4_0) == 2, "QType::Q4_0 wire value");
static_assert(static_cast<uint16_t>(QType::Q4_K) == 12, "QType::Q4_K wire value");
static_assert(static_cast<uint16_t>(QType::Q6_K) == 14, "QType::Q6_K wire value");
static_assert(static_cast<uint16_t>(QType::Q8_0) == 8, "QType::Q8_0 wire value");
static_assert(static_cast<uint16_t>(QType::IQ4_NL) == 20, "QType::IQ4_NL wire value");
static_assert(static_cast<uint16_t>(QType::IQ4_XS) == 23, "QType::IQ4_XS wire value");
static_assert(static_cast<uint16_t>(QType::BF16) == 30, "QType::BF16 wire value");
static_assert(static_cast<uint16_t>(QType::MXFP4) == 31, "QType::MXFP4 wire value");

}  // namespace imp
