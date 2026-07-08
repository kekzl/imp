#include "model/tensor_kind_table.h"
#include "core/logging.h"

#include <array>
#include <cassert>
#include <utility>

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

constexpr std::array<KindCapabilities, std::to_underlying(TensorKind::COUNT)> kKindTable = [] {
    std::array<KindCapabilities, std::to_underlying(TensorKind::COUNT)> t{};
    t[std::to_underlying(TensorKind::UNKNOWN)] = build(FP16_ONLY, StorageTier::FP16);
    t[std::to_underlying(TensorKind::WQ)] = build(ALL_QUANT, StorageTier::NVFP4);
    t[std::to_underlying(TensorKind::WK)] = build(ALL_QUANT, StorageTier::FP8, true);
    t[std::to_underlying(TensorKind::WV)] = build(ALL_QUANT, StorageTier::FP8, true);
    t[std::to_underlying(TensorKind::WO)] = build(ALL_QUANT, StorageTier::NVFP4);
    t[std::to_underlying(TensorKind::QKV_FUSED)] = build(NO_MXFP4, StorageTier::FP8);
    // MLA (DeepSeek-V2/V3) latent projections.
    // kv_a_proj is precision-sensitive (latent down-proj) → no aggressive FP4 default.
    t[std::to_underlying(TensorKind::KV_A_PROJ)] = build(NO_MXFP4, StorageTier::FP16);
    // kv_a_layernorm is an RMSNorm weight → never quantized.
    t[std::to_underlying(TensorKind::KV_A_NORM)] = build(FP32_ONLY, StorageTier::FP32);
    // kv_b_proj is a standard up-proj → mirror WO.
    t[std::to_underlying(TensorKind::KV_B_PROJ)] = build(ALL_QUANT, StorageTier::NVFP4);
    t[std::to_underlying(TensorKind::W_GATE)] = build(ALL_QUANT, StorageTier::NVFP4, true);
    t[std::to_underlying(TensorKind::W_UP)] = build(ALL_QUANT, StorageTier::NVFP4, true);
    t[std::to_underlying(TensorKind::W_DOWN)] = build(ALL_QUANT, StorageTier::NVFP4);
    t[std::to_underlying(TensorKind::EXPERT_GATE)] = build(ALL_QUANT, StorageTier::NVFP4, true);
    t[std::to_underlying(TensorKind::EXPERT_UP)] = build(ALL_QUANT, StorageTier::NVFP4, true);
    t[std::to_underlying(TensorKind::EXPERT_DOWN)] = build(ALL_QUANT, StorageTier::NVFP4);
    t[std::to_underlying(TensorKind::FUSED_KV)] = build(NO_MXFP4, StorageTier::FP8);
    t[std::to_underlying(TensorKind::FUSED_GATE_UP)] = build(ALL_QUANT, StorageTier::NVFP4);
    t[std::to_underlying(TensorKind::TOK_EMBED)] = build(FP16_ONLY, StorageTier::FP16);
    t[std::to_underlying(TensorKind::LM_HEAD)] = build(FP16_ONLY, StorageTier::FP16);
    t[std::to_underlying(TensorKind::ROUTER)] = build(FP16_OR_FP32, StorageTier::FP32);
    t[std::to_underlying(TensorKind::SHARED_EXPERT_GATE)] = build(FP32_ONLY, StorageTier::FP32);
    t[std::to_underlying(TensorKind::SSM_IN)] = build(FP16_ONLY, StorageTier::FP16);
    t[std::to_underlying(TensorKind::SSM_OUT)] = build(FP16_ONLY, StorageTier::FP16);
    t[std::to_underlying(TensorKind::CONV1D_W)] = build(FP16_ONLY, StorageTier::FP16);
    t[std::to_underlying(TensorKind::CONV1D_B)] = build(FP16_ONLY, StorageTier::FP16);
    t[std::to_underlying(TensorKind::A_LOG)] = build(FP32_ONLY, StorageTier::FP32);
    t[std::to_underlying(TensorKind::DT_BIAS)] = build(FP32_ONLY, StorageTier::FP32);
    t[std::to_underlying(TensorKind::BETA)] = build(FP16_ONLY, StorageTier::FP16);
    t[std::to_underlying(TensorKind::ALPHA)] = build(FP16_ONLY, StorageTier::FP16);
    t[std::to_underlying(TensorKind::SSM_GROUP_NORM)] = build(FP16_ONLY, StorageTier::FP16);
    t[std::to_underlying(TensorKind::GDN_GATE)] = build(ALL_QUANT, StorageTier::NVFP4);
    t[std::to_underlying(TensorKind::GDN_ALPHA)] = build(FP16_ONLY, StorageTier::FP16);
    t[std::to_underlying(TensorKind::GDN_BETA)] = build(FP16_ONLY, StorageTier::FP16);
    t[std::to_underlying(TensorKind::GDN_ALPHA_BETA_PACKED)] = build(FP16_ONLY, StorageTier::FP16);
    t[std::to_underlying(TensorKind::GDN_INPUT_PACKED)] = build(FP16_ONLY, StorageTier::FP16);
    t[std::to_underlying(TensorKind::ATTN_NORM)] = build(FP32_ONLY, StorageTier::FP32);
    t[std::to_underlying(TensorKind::FFN_NORM)] = build(FP32_ONLY, StorageTier::FP32);
    t[std::to_underlying(TensorKind::POST_ATTN_NORM)] = build(FP32_ONLY, StorageTier::FP32);
    t[std::to_underlying(TensorKind::POST_FFN_NORM)] = build(FP32_ONLY, StorageTier::FP32);
    t[std::to_underlying(TensorKind::QK_NORM_Q)] = build(FP32_ONLY, StorageTier::FP32);
    t[std::to_underlying(TensorKind::QK_NORM_K)] = build(FP32_ONLY, StorageTier::FP32);
    t[std::to_underlying(TensorKind::ROPE_FREQS)] = build(FP32_ONLY, StorageTier::FP32);
    t[std::to_underlying(TensorKind::SIGLIP_ATTN)] = build(NO_MXFP4, StorageTier::FP16);
    t[std::to_underlying(TensorKind::SIGLIP_FFN)] = build(NO_MXFP4, StorageTier::FP16);
    t[std::to_underlying(TensorKind::SIGLIP_NORM)] = build(FP32_ONLY, StorageTier::FP32);
    t[std::to_underlying(TensorKind::MM_PROJ)] = build(FP16_ONLY, StorageTier::FP16);
    return t;
}();

}  // namespace

const KindCapabilities& capabilities_of(TensorKind k) {
    IMP_CHECK(k != TensorKind::COUNT && static_cast<size_t>(k) < kKindTable.size(),
              "capabilities_of: invalid TensorKind=%zu (table size=%zu)", static_cast<size_t>(k),
              kKindTable.size());
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
         source_qtype == QType::Q6_K || source_qtype == QType::Q5_K ||
         // i-quants: no dp4a/MMVQ kernels — the NVFP4 decode cache is the
         // only fast decode path, so the overlay is allowed despite the
         // similar bit-width (mirror nvfp4_beneficial).
         source_qtype == QType::IQ4_NL || source_qtype == QType::IQ4_XS);

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
