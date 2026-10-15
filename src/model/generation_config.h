#pragma once

#include <cstdint>
#include <vector>

namespace imp {

// Sampling/stop-condition defaults from generation_config.json. Sentinels mean "not
// present": <0 for floats, -1 for ints, empty vector for eos. Caller falls back to arch
// defaults or the hardcoded ImpGenerateParams baseline.
struct GenerationConfig {
    float temperature = -1.0f;
    float top_p = -1.0f;
    int top_k = -1;
    float repetition_penalty = -1.0f;
    std::vector<int32_t> eos_token_ids;
};

}  // namespace imp
