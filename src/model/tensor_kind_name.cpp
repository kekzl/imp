#include "core/tensor_kind.h"

namespace imp {

const char* tensor_kind_name(TensorKind k) {
    switch (k) {
        case TensorKind::UNKNOWN:          return "UNKNOWN";
        case TensorKind::WQ:               return "WQ";
        case TensorKind::WK:               return "WK";
        case TensorKind::WV:               return "WV";
        case TensorKind::WO:               return "WO";
        case TensorKind::QKV_FUSED:        return "QKV_FUSED";
        case TensorKind::W_GATE:           return "W_GATE";
        case TensorKind::W_UP:             return "W_UP";
        case TensorKind::W_DOWN:           return "W_DOWN";
        case TensorKind::EXPERT_GATE:      return "EXPERT_GATE";
        case TensorKind::EXPERT_UP:        return "EXPERT_UP";
        case TensorKind::EXPERT_DOWN:      return "EXPERT_DOWN";
        case TensorKind::FUSED_KV:         return "FUSED_KV";
        case TensorKind::FUSED_GATE_UP:    return "FUSED_GATE_UP";
        case TensorKind::TOK_EMBED:        return "TOK_EMBED";
        case TensorKind::LM_HEAD:          return "LM_HEAD";
        case TensorKind::ROUTER:           return "ROUTER";
        case TensorKind::SHARED_EXPERT_GATE: return "SHARED_EXPERT_GATE";
        case TensorKind::SSM_IN:           return "SSM_IN";
        case TensorKind::SSM_OUT:          return "SSM_OUT";
        case TensorKind::CONV1D_W:         return "CONV1D_W";
        case TensorKind::CONV1D_B:         return "CONV1D_B";
        case TensorKind::A_LOG:            return "A_LOG";
        case TensorKind::DT_BIAS:          return "DT_BIAS";
        case TensorKind::BETA:             return "BETA";
        case TensorKind::ALPHA:            return "ALPHA";
        case TensorKind::SSM_GROUP_NORM:   return "SSM_GROUP_NORM";
        case TensorKind::GDN_GATE:         return "GDN_GATE";
        case TensorKind::ATTN_NORM:        return "ATTN_NORM";
        case TensorKind::FFN_NORM:         return "FFN_NORM";
        case TensorKind::POST_ATTN_NORM:   return "POST_ATTN_NORM";
        case TensorKind::POST_FFN_NORM:    return "POST_FFN_NORM";
        case TensorKind::QK_NORM_Q:        return "QK_NORM_Q";
        case TensorKind::QK_NORM_K:        return "QK_NORM_K";
        case TensorKind::ROPE_FREQS:       return "ROPE_FREQS";
        case TensorKind::SIGLIP_ATTN:      return "SIGLIP_ATTN";
        case TensorKind::SIGLIP_FFN:       return "SIGLIP_FFN";
        case TensorKind::SIGLIP_NORM:      return "SIGLIP_NORM";
        case TensorKind::MM_PROJ:          return "MM_PROJ";
    }
    return "UNKNOWN";
}

} // namespace imp
