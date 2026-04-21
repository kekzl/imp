#include "model/tensor_kind_table.h"

#include <array>

namespace imp {

namespace {

constexpr TierMask ALL_QUANT =
    mask(StorageTier::FP16) | mask(StorageTier::FP8) |
    mask(StorageTier::NVFP4) | mask(StorageTier::CUTLASS_NVFP4) |
    mask(StorageTier::MXFP4);

constexpr TierMask NO_MXFP4 =
    mask(StorageTier::FP16) | mask(StorageTier::FP8) |
    mask(StorageTier::NVFP4) | mask(StorageTier::CUTLASS_NVFP4);

constexpr TierMask FP16_ONLY = mask(StorageTier::FP16);
constexpr TierMask FP32_ONLY = mask(StorageTier::FP32);
constexpr TierMask FP16_OR_FP32 = mask(StorageTier::FP16) | mask(StorageTier::FP32);

constexpr KindCapabilities build(TierMask s, StorageTier f, bool fus = false) {
    return {s, f, fus};
}

constexpr std::array<KindCapabilities, static_cast<size_t>(TensorKind::_COUNT)>
kKindTable = [] {
    std::array<KindCapabilities, static_cast<size_t>(TensorKind::_COUNT)> t{};
    t[(size_t)TensorKind::UNKNOWN]             = build(FP16_ONLY,   StorageTier::FP16);
    t[(size_t)TensorKind::WQ]                  = build(ALL_QUANT,   StorageTier::NVFP4);
    t[(size_t)TensorKind::WK]                  = build(NO_MXFP4,    StorageTier::FP8,  true);
    t[(size_t)TensorKind::WV]                  = build(NO_MXFP4,    StorageTier::FP8,  true);
    t[(size_t)TensorKind::WO]                  = build(ALL_QUANT,   StorageTier::NVFP4);
    t[(size_t)TensorKind::QKV_FUSED]           = build(NO_MXFP4,    StorageTier::FP8);
    t[(size_t)TensorKind::W_GATE]              = build(ALL_QUANT,   StorageTier::NVFP4, true);
    t[(size_t)TensorKind::W_UP]                = build(ALL_QUANT,   StorageTier::NVFP4, true);
    t[(size_t)TensorKind::W_DOWN]              = build(ALL_QUANT,   StorageTier::NVFP4);
    t[(size_t)TensorKind::EXPERT_GATE]         = build(ALL_QUANT,   StorageTier::NVFP4, true);
    t[(size_t)TensorKind::EXPERT_UP]           = build(ALL_QUANT,   StorageTier::NVFP4, true);
    t[(size_t)TensorKind::EXPERT_DOWN]         = build(ALL_QUANT,   StorageTier::NVFP4);
    t[(size_t)TensorKind::FUSED_KV]            = build(NO_MXFP4,    StorageTier::FP8);
    t[(size_t)TensorKind::FUSED_GATE_UP]       = build(ALL_QUANT,   StorageTier::NVFP4);
    t[(size_t)TensorKind::TOK_EMBED]           = build(FP16_ONLY,   StorageTier::FP16);
    t[(size_t)TensorKind::LM_HEAD]             = build(FP16_ONLY,   StorageTier::FP16);
    t[(size_t)TensorKind::ROUTER]              = build(FP16_OR_FP32,StorageTier::FP32);
    t[(size_t)TensorKind::SHARED_EXPERT_GATE]  = build(FP32_ONLY,   StorageTier::FP32);
    t[(size_t)TensorKind::SSM_IN]              = build(FP16_ONLY,   StorageTier::FP16);
    t[(size_t)TensorKind::SSM_OUT]             = build(FP16_ONLY,   StorageTier::FP16);
    t[(size_t)TensorKind::CONV1D_W]            = build(FP16_ONLY,   StorageTier::FP16);
    t[(size_t)TensorKind::CONV1D_B]            = build(FP16_ONLY,   StorageTier::FP16);
    t[(size_t)TensorKind::A_LOG]               = build(FP32_ONLY,   StorageTier::FP32);
    t[(size_t)TensorKind::DT_BIAS]             = build(FP32_ONLY,   StorageTier::FP32);
    t[(size_t)TensorKind::BETA]                = build(FP16_ONLY,   StorageTier::FP16);
    t[(size_t)TensorKind::ALPHA]               = build(FP16_ONLY,   StorageTier::FP16);
    t[(size_t)TensorKind::SSM_GROUP_NORM]      = build(FP32_ONLY,   StorageTier::FP32);
    t[(size_t)TensorKind::ATTN_NORM]           = build(FP32_ONLY,   StorageTier::FP32);
    t[(size_t)TensorKind::FFN_NORM]            = build(FP32_ONLY,   StorageTier::FP32);
    t[(size_t)TensorKind::POST_ATTN_NORM]      = build(FP32_ONLY,   StorageTier::FP32);
    t[(size_t)TensorKind::POST_FFN_NORM]       = build(FP32_ONLY,   StorageTier::FP32);
    t[(size_t)TensorKind::QK_NORM_Q]           = build(FP32_ONLY,   StorageTier::FP32);
    t[(size_t)TensorKind::QK_NORM_K]           = build(FP32_ONLY,   StorageTier::FP32);
    t[(size_t)TensorKind::ROPE_FREQS]          = build(FP32_ONLY,   StorageTier::FP32);
    t[(size_t)TensorKind::SIGLIP_ATTN]         = build(NO_MXFP4,    StorageTier::FP16);
    t[(size_t)TensorKind::SIGLIP_FFN]          = build(NO_MXFP4,    StorageTier::FP16);
    t[(size_t)TensorKind::SIGLIP_NORM]         = build(FP32_ONLY,   StorageTier::FP32);
    t[(size_t)TensorKind::MM_PROJ]             = build(FP16_ONLY,   StorageTier::FP16);
    return t;
}();

} // namespace

const KindCapabilities& capabilities_of(TensorKind k) {
    return kKindTable[static_cast<size_t>(k)];
}

} // namespace imp
