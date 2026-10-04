#pragma once

#include "core/tensor.h"

#include <cstdint>
#include <string>
#include <vector>

namespace imp {

struct ModelConfig;
struct TransformerLayer;

// Qwen3-Next fused GDN inputs, one row block per key-head group g (n_k groups, r = n_v / n_k):
//   in_proj_qkvz: [q(dk) k(dk) v(r*dv) z(r*dv)],  in_proj_ba: [b(r) a(r)].
// Re-laid into the Qwen3.5 slots: ssm_in = [Q | K | V], gdn_gate = Z, gdn_beta = B, gdn_alpha = A.
struct GdnFusedDims {
    int64_t n_k = 0, dk = 0, n_v = 0, dv = 0;
};

[[nodiscard]] GdnFusedDims gdn_fused_dims(const ModelConfig& cfg);

// Splits `t` per the layout above. Dense row-major sources only (F32/F16/BF16, no scales);
// false on a quantized source or a shape that disagrees with `d`. New buffers go to `owned`.
[[nodiscard]] bool split_gdn_qkvz(const Tensor& t, const GdnFusedDims& d, std::vector<void*>& owned,
                                  Tensor& qkv, Tensor& z);
[[nodiscard]] bool split_gdn_ba(const Tensor& t, const GdnFusedDims& d, std::vector<void*>& owned, Tensor& b,
                                Tensor& a);

// weight_map entry: proj is "in_proj_qkvz" or "in_proj_ba". False = refuse the checkpoint.
[[nodiscard]] bool assign_gdn_fused_input(const std::string& proj, const Tensor& t, const ModelConfig& cfg,
                                          std::vector<void*>& owned, TransformerLayer& layer);

}  // namespace imp
