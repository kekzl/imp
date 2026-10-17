#pragma once

// Classifier-free guidance (roadmap row 117): a shadow request decodes the negative prompt in
// lockstep with its owner; before sampling, the owner's logit row becomes l_a + scale * (l_a - l_b)
// and the shadow takes the owner's token. Engine side: runtime/engine_cfg.cpp.

#include <memory>
#include <vector>

namespace imp {

struct Request;

inline constexpr float kMaxCfgScale = 10.0f;

struct CfgState {
    float scale = 0.0f;               // owner: weight w = cfg_scale - 1 (cfg_scale 1 = unguided)
    std::shared_ptr<Request> shadow;  // owner: the negative-prompt request
    std::weak_ptr<Request> owner;     // shadow: its owner
    std::vector<float> first_logits;  // shadow: raw logits of its prompt's last row (host)
    bool ready = false;               // shadow: prefill done, first_logits valid
};

}  // namespace imp
