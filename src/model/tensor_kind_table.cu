#include "model/tensor_kind_table.h"
#include "core/logging.h"

#include <array>
#include <cassert>

namespace imp {

namespace {

constexpr TierMask ALL_QUANT = mask(StorageTier::FP16) | mask(StorageTier::FP8) | mask(StorageTier::NVFP4) |
                               mask(StorageTier::CUTLASS_NVFP4) | mask(StorageTier::MXFP4);

constexpr TierMask NO_MXFP4 = mask(StorageTier::FP16) | mask(StorageTier::FP8) | mask(StorageTier::NVFP4) |
                              mask(StorageTier::CUTLASS_NVFP4);

constexpr TierMask FP16_ONLY = mask(StorageTier::FP16);
constexpr TierMask FP32_ONLY = mask(StorageTier::FP32);
constexpr TierMask FP16_OR_FP32 = mask(StorageTier::FP16) | mask(StorageTier::FP32);

constexpr KindCapabilities build(TierMask s, StorageTier f, bool fus = false) { return {s, f, fus}; }

constexpr std::array<KindCapabilities, static_cast<size_t>(TensorKind::_COUNT)> kKindTable = [] {
    std::array<KindCapabilities, static_cast<size_t>(TensorKind::_COUNT)> t{};
    t[(size_t)TensorKind::UNKNOWN] = build(FP16_ONLY, StorageTier::FP16);
    t[(size_t)TensorKind::WQ] = build(ALL_QUANT, StorageTier::NVFP4);
    t[(size_t)TensorKind::WK] = build(ALL_QUANT, StorageTier::FP8, true);
    t[(size_t)TensorKind::WV] = build(ALL_QUANT, StorageTier::FP8, true);
    t[(size_t)TensorKind::WO] = build(ALL_QUANT, StorageTier::NVFP4);
    t[(size_t)TensorKind::QKV_FUSED] = build(NO_MXFP4, StorageTier::FP8);
    t[(size_t)TensorKind::W_GATE] = build(ALL_QUANT, StorageTier::NVFP4, true);
    t[(size_t)TensorKind::W_UP] = build(ALL_QUANT, StorageTier::NVFP4, true);
    t[(size_t)TensorKind::W_DOWN] = build(ALL_QUANT, StorageTier::NVFP4);
    t[(size_t)TensorKind::EXPERT_GATE] = build(ALL_QUANT, StorageTier::NVFP4, true);
    t[(size_t)TensorKind::EXPERT_UP] = build(ALL_QUANT, StorageTier::NVFP4, true);
    t[(size_t)TensorKind::EXPERT_DOWN] = build(ALL_QUANT, StorageTier::NVFP4);
    t[(size_t)TensorKind::FUSED_KV] = build(NO_MXFP4, StorageTier::FP8);
    t[(size_t)TensorKind::FUSED_GATE_UP] = build(ALL_QUANT, StorageTier::NVFP4);
    t[(size_t)TensorKind::TOK_EMBED] = build(FP16_ONLY, StorageTier::FP16);
    t[(size_t)TensorKind::LM_HEAD] = build(FP16_ONLY, StorageTier::FP16);
    t[(size_t)TensorKind::ROUTER] = build(FP16_OR_FP32, StorageTier::FP32);
    t[(size_t)TensorKind::SHARED_EXPERT_GATE] = build(FP32_ONLY, StorageTier::FP32);
    t[(size_t)TensorKind::SSM_IN] = build(FP16_ONLY, StorageTier::FP16);
    t[(size_t)TensorKind::SSM_OUT] = build(FP16_ONLY, StorageTier::FP16);
    t[(size_t)TensorKind::CONV1D_W] = build(FP16_ONLY, StorageTier::FP16);
    t[(size_t)TensorKind::CONV1D_B] = build(FP16_ONLY, StorageTier::FP16);
    t[(size_t)TensorKind::A_LOG] = build(FP32_ONLY, StorageTier::FP32);
    t[(size_t)TensorKind::DT_BIAS] = build(FP32_ONLY, StorageTier::FP32);
    t[(size_t)TensorKind::BETA] = build(FP16_ONLY, StorageTier::FP16);
    t[(size_t)TensorKind::ALPHA] = build(FP16_ONLY, StorageTier::FP16);
    t[(size_t)TensorKind::SSM_GROUP_NORM] = build(FP16_ONLY, StorageTier::FP16);
    t[(size_t)TensorKind::GDN_GATE] = build(ALL_QUANT, StorageTier::NVFP4);
    t[(size_t)TensorKind::ATTN_NORM] = build(FP32_ONLY, StorageTier::FP32);
    t[(size_t)TensorKind::FFN_NORM] = build(FP32_ONLY, StorageTier::FP32);
    t[(size_t)TensorKind::POST_ATTN_NORM] = build(FP32_ONLY, StorageTier::FP32);
    t[(size_t)TensorKind::POST_FFN_NORM] = build(FP32_ONLY, StorageTier::FP32);
    t[(size_t)TensorKind::QK_NORM_Q] = build(FP32_ONLY, StorageTier::FP32);
    t[(size_t)TensorKind::QK_NORM_K] = build(FP32_ONLY, StorageTier::FP32);
    t[(size_t)TensorKind::ROPE_FREQS] = build(FP32_ONLY, StorageTier::FP32);
    t[(size_t)TensorKind::SIGLIP_ATTN] = build(NO_MXFP4, StorageTier::FP16);
    t[(size_t)TensorKind::SIGLIP_FFN] = build(NO_MXFP4, StorageTier::FP16);
    t[(size_t)TensorKind::SIGLIP_NORM] = build(FP32_ONLY, StorageTier::FP32);
    t[(size_t)TensorKind::MM_PROJ] = build(FP16_ONLY, StorageTier::FP16);
    return t;
}();

}  // namespace

const KindCapabilities& capabilities_of(TensorKind k) {
    IMP_CHECK(k != TensorKind::_COUNT && static_cast<size_t>(k) < kKindTable.size(),
              "capabilities_of: invalid TensorKind=%zu (table size=%zu)",
              static_cast<size_t>(k), kKindTable.size());
    return kKindTable[static_cast<size_t>(k)];
}

KindCapabilities effective_capabilities(TensorKind k, QType source_qtype) {
    KindCapabilities cap = capabilities_of(k);

    // Native FP4-family sources: capabilities ARE the source tier.
    // No upgrade path exists (we wouldn't dequant FP4 to FP16 just to cache it).
    if (source_qtype == QType::NVFP4) {
        cap.supported = mask(StorageTier::NVFP4) | mask(StorageTier::CUTLASS_NVFP4);
        cap.required_floor = StorageTier::NVFP4;
        return cap;
    }
    if (source_qtype == QType::MXFP4) {
        cap.supported = mask(StorageTier::MXFP4);
        cap.required_floor = StorageTier::MXFP4;
        return cap;
    }
    if (source_qtype == QType::FP8_E4M3) {
        cap.supported = mask(StorageTier::FP8) | mask(StorageTier::FP16);
        cap.required_floor = StorageTier::FP8;
        return cap;
    }

    // NVFP4 overlay only benefits Q5_K / Q6_K / Q8_0 / Q8_K (>= 5.5 bits/elem
    // source). Sub-5-bit GGUF formats (Q4_K, Q4_0, Q5_0, Q5_1, Q3_K, Q2_K) are
    // representation changes at similar bit-width — no compression win, possible
    // quality risk. Mirror the runtime `nvfp4_beneficial(qtype)` policy.
    const bool nvfp4_overlay_ok =
        (source_qtype == QType::Q8_0 || source_qtype == QType::Q8_K ||
         source_qtype == QType::Q6_K || source_qtype == QType::Q5_K);

    if (!nvfp4_overlay_ok) {
        cap.supported = cap.supported & ~mask(StorageTier::NVFP4) &
                        ~mask(StorageTier::CUTLASS_NVFP4) & ~mask(StorageTier::MXFP4);
        if (cap.required_floor == StorageTier::NVFP4 ||
            cap.required_floor == StorageTier::CUTLASS_NVFP4 ||
            cap.required_floor == StorageTier::MXFP4) {
            // Fall back to FP16 (smallest cuBLAS-friendly tier still in supported).
            // If FP16 was also stripped (it shouldn't be — all dense kinds keep it),
            // the caller will fail loud via required_floor not in supported.
            if (mask_contains(cap.supported, StorageTier::FP16))
                cap.required_floor = StorageTier::FP16;
            else if (mask_contains(cap.supported, StorageTier::FP8))
                cap.required_floor = StorageTier::FP8;
            else
                cap.required_floor = StorageTier::FP32;
        }
    }

    return cap;
}

}  // namespace imp
