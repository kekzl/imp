#include "compute/gemm.h"
#include "compute/gemm_q4k.h"
#include "compute/gemm_q6k.h"
#include "core/qtype.h"

#include <stdexcept>
#include <string>

// MoE decode (#2444) and prefill (#2459) kernel selectors: one switch per family, nullptr = no arm.
namespace imp {
namespace {

MoeDp4aDecodeFn find_dp4a_decode(QType qt) {
    switch (qt) {
    case QType::Q6_K: return gemv_q6k_q8_1_moe_decode;
    case QType::Q8_0: return gemv_q8_0_q8_1_moe_decode;
    case QType::Q4_0: return gemv_q4_0_q8_1_moe_decode;
    case QType::Q4_K: return gemv_q4_k_q8_1_moe_decode;
    case QType::Q5_K: return gemv_q5_k_q8_1_moe_decode;
    case QType::Q2_K: return gemv_q2_k_q8_1_moe_decode;
    case QType::Q3_K: return gemv_q3_k_q8_1_moe_decode;
    case QType::Q5_1: return gemv_q5_1_q8_1_moe_decode;
    default: return nullptr;
    }
}

MoeDp4aGateUpFn find_dp4a_gate_up(QType qt) {
    switch (qt) {
    case QType::Q6_K: return gemv_q6k_q8_1_moe_gate_up_fused;
    case QType::Q8_0: return gemv_q8_0_q8_1_moe_gate_up_fused;
    case QType::Q4_0: return gemv_q4_0_q8_1_moe_gate_up_fused;
    case QType::Q4_K: return gemv_q4_k_q8_1_moe_gate_up_fused;
    case QType::Q5_K: return gemv_q5_k_q8_1_moe_gate_up_fused;
    case QType::Q2_K: return gemv_q2_k_q8_1_moe_gate_up_fused;
    case QType::Q3_K: return gemv_q3_k_q8_1_moe_gate_up_fused;
    case QType::Q5_1: return gemv_q5_1_q8_1_moe_gate_up_fused;
    default: return nullptr;
    }
}

MoeFp16DecodeFn find_fp16_decode(QType qt) {
    switch (qt) {
    case QType::Q6_K: return gemv_q6k_moe_decode;
    case QType::Q8_0: return gemv_q8_0_moe_decode;
    default: return nullptr;
    }
}

MoeFp16GateUpFn find_fp16_gate_up(QType qt) {
    switch (qt) {
    case QType::Q6_K: return gemv_q6k_moe_gate_up_fused;
    case QType::Q8_0: return gemv_q8_0_moe_gate_up_fused;
    default: return nullptr;
    }
}

// #2459: prefill selectors, same contract as decode.
MoeFusedDp4aPrefillFn find_fused_dp4a_prefill(QType qt) {
    switch (qt) {
        case QType::Q4_K:
            return gemm_q4k_dp4a_moe_fused;
        case QType::Q5_K:
            return gemm_q5k_dp4a_moe_fused;
        case QType::Q6_K:
            return gemm_q6k_moe_fused;
        default:
            return nullptr;
    }
}

// qkind of mmq_imma_moe_gemm; -1 = no arm.
int find_imma_prefill_qkind(QType qt) {
    switch (qt) {
        case QType::Q8_0:
            return 0;
        case QType::Q4_K:
            return 1;
        case QType::Q6_K:
            return 2;
        case QType::Q5_1:
            return 3;
        case QType::Q5_K:
            return 4;
        default:
            return -1;
    }
}

[[noreturn]] void refuse(QType qt, const char* family) {
    throw std::invalid_argument(std::string("MoE: no ") + family + " kernel for qtype " + qtype_name(qt));
}

template <typename Fn>
Fn require(Fn fn, QType qt, const char* family) {
    if (fn == nullptr)
        refuse(qt, family);
    return fn;
}

}  // namespace

bool moe_dp4a_decode_supported(QType qt) {
    return find_dp4a_decode(qt) != nullptr && find_dp4a_gate_up(qt) != nullptr;
}
MoeDp4aDecodeFn moe_dp4a_decode_kernel(QType qt) { return require(find_dp4a_decode(qt), qt, "dp4a decode"); }
MoeDp4aGateUpFn moe_dp4a_gate_up_kernel(QType qt) {
    return require(find_dp4a_gate_up(qt), qt, "dp4a gate_up");
}
bool moe_fp16_decode_supported(QType qt) {
    return find_fp16_decode(qt) != nullptr && find_fp16_gate_up(qt) != nullptr;
}
MoeFp16DecodeFn moe_fp16_decode_kernel(QType qt) { return require(find_fp16_decode(qt), qt, "fp16 decode"); }
MoeFp16GateUpFn moe_fp16_gate_up_kernel(QType qt) {
    return require(find_fp16_gate_up(qt), qt, "fp16 gate_up");
}
bool moe_fused_dp4a_prefill_supported(QType qt) { return find_fused_dp4a_prefill(qt) != nullptr; }
MoeFusedDp4aPrefillFn moe_fused_dp4a_prefill_kernel(QType qt) {
    return require(find_fused_dp4a_prefill(qt), qt, "fused dp4a prefill");
}
bool moe_imma_prefill_supported(QType qt) { return find_imma_prefill_qkind(qt) >= 0; }
int moe_imma_prefill_qkind(QType qt) {
    const int qkind = find_imma_prefill_qkind(qt);
    if (qkind < 0)
        refuse(qt, "IMMA prefill");
    return qkind;
}

}  // namespace imp
