#pragma once
// Inkling2-Loop's dense projections in the engine's WeightRegistry and its LoRA contract (B-111 S5).
// Host-only: the CPU lane pins both (tests/test_inkling_registry.cpp).

#include "core/tensor_kind.h"
#include "exec/weight_handle.h"
#include "lora/lora_adapter.h"
#include "model/inkling_config.h"

#include <cstddef>
#include <vector>

namespace imp {

struct InklingWeights;

// Handles of one layer's projections, as the forward's GEMMs read them (InklingForward::apply_):
// qkv the whole row block [wq; wr; sink; gate_attn; wk; k taps; wv; v taps] (sliding first pass),
// q its first 2 QD + RD + n_heads rows, k and v their KD (+ taps) rows, o [hidden, QD].
struct InklingProjIds {
    TensorID qkv = kInvalidTensorID, q = kInvalidTensorID, k = kInvalidTensorID, v = kInvalidTensorID,
             o = kInvalidTensorID;
};

// Appends 5 borrowed FP16 handles per layer (payload and source = the weights' device pointers, owned
// by InklingWeights); one entry per layer. The I8B32 decode copies (InklingDense8) stay in front.
[[nodiscard]] std::vector<InklingProjIds> register_inkling_weights(const InklingWeights& w,
                                                                   WeightRegistry& reg);

// Rows of the q block a split call reads: wq, wr, sink, gate_attn.
[[nodiscard]] inline int inkling_q_rows(const InklingConfig& c) {
    return 2 * c.n_heads * c.head_dim + c.n_heads * c.d_rel + c.n_heads;
}

// An adapter on Inkling: q / k / v / o of the HF modules attn.{wq_du, wk_dv, wv_dv, wo_ud}; any other
// lora tensor refuses the load (no dense FFN, experts and controllers are not adapted).
inline constexpr LoraTargets kInklingLoraTargets{.mask = lora_proj_bit(LoraProj::Q) |
                                                         lora_proj_bit(LoraProj::K) |
                                                         lora_proj_bit(LoraProj::V) |
                                                         lora_proj_bit(LoraProj::O),
                                                 .strict = true};
[[nodiscard]] inline LoraDims inkling_lora_dims(const InklingConfig& c) {
    return {c.hidden, c.n_heads * c.head_dim, c.n_kv_heads * c.head_dim, 0};
}

// fp32 rank scratch of the forward's q, k and v streams: 3 x [rows][rank].
[[nodiscard]] inline size_t inkling_lora_scratch_bytes(int rows, int rank) {
    return 3 * static_cast<size_t>(rows) * static_cast<size_t>(rank) * sizeof(float);
}

}  // namespace imp
