#pragma once

#include <string>

namespace imp {

// LFM2 / LFM2-MoE SafeTensors name -> the HF-canonical name weight_map.cpp matches (conv -> mamba.*,
// feed_forward.w1/w3/w2 -> mlp.gate/up/down_proj, expert_bias -> mlp.gate.bias, embedding_norm ->
// model.norm). Unknown names come back unchanged.
[[nodiscard]] std::string lfm2_canonical_name(const std::string& name);

}  // namespace imp
