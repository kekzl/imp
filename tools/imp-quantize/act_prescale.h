#pragma once

#include "awq.h"
#include "model/safetensors_raw.h"
#include "options.h"

#include <functional>
#include <memory>
#include <set>
#include <vector>
#include <map>
#include <string>

namespace imp::quantize {

// LFM2 (model_type lfm2 / lfm2_moe): power-of-two activation pre-scales folded into producer rows and
// consumer columns (exact in BF16). imp's NVFP4 activation quantize has no tensor scale, so a block
// whose absmax / 6 is below 2^-9 flushes; LFM2-8B-A1B: 75 % of the w2 input blocks, 23 % conv.out_proj,
// 11 % self_attn.out_proj. Pairs: w3 x64 / w2 /64 (dense + experts), conv.in_proj C rows x16 /
// conv.out_proj /16, v_proj x8 / out_proj /8 (PPL 38.92 -> 22.13). A pair is skipped unless both are
// quantized. Returns the number of pairs folded.
int add_lfm2_act_prescale(awq::Plan& plan, const std::map<std::string, const RawTensor*>& index,
                          const std::function<bool(const RawTensor&)>& quantized);

// The call imp-quantize makes: model_type lfm2 / lfm2_moe / glm4_moe_lite / cohere2_moe, --format modelopt
// only (compressed-tensors shares one tensor scale across q/k/v and gate/up, which an x8 / x64 producer would
// dominate); a tensor counts as quantized by the writer's own rule (gated q_proj, --keep-gdn-proj,
// should_quantize).
void apply_act_prescale(const Options& opt, const std::vector<std::unique_ptr<RawSafeTensors>>& opened,
                        const std::set<std::string>& gated_q_proj, awq::Plan& plan);

}  // namespace imp::quantize
