#include "graph/op.h"

namespace imp {

const char* op_type_name(OpType type) {
    switch (type) {
        case OpType::EMBEDDING:         return "EMBEDDING";
        case OpType::RMSNORM:           return "RMSNORM";
        case OpType::RMSNORM_RESIDUAL:  return "RMSNORM_RESIDUAL";
        case OpType::ROPE:              return "ROPE";
        case OpType::QKV_PROJ:          return "QKV_PROJ";
        case OpType::ATTENTION_PREFILL: return "ATTENTION_PREFILL";
        case OpType::ATTENTION_DECODE:  return "ATTENTION_DECODE";
        case OpType::O_PROJ:            return "O_PROJ";
        case OpType::GATE_PROJ:         return "GATE_PROJ";
        case OpType::UP_PROJ:           return "UP_PROJ";
        case OpType::DOWN_PROJ:         return "DOWN_PROJ";
        case OpType::SWIGLU:            return "SWIGLU";
        case OpType::RESIDUAL_ADD:      return "RESIDUAL_ADD";
        case OpType::MOE_GATE:          return "MOE_GATE";
        case OpType::MOE_GATHER:        return "MOE_GATHER";
        case OpType::GROUPED_GEMM:      return "GROUPED_GEMM";
        case OpType::MOE_SCATTER:       return "MOE_SCATTER";
        case OpType::OUTPUT_NORM:       return "OUTPUT_NORM";
        case OpType::LM_HEAD:           return "LM_HEAD";
        case OpType::SAMPLING:          return "SAMPLING";
        default:                        return "UNKNOWN";
    }
}

} // namespace imp
