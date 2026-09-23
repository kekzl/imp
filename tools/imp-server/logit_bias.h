#pragma once

// OpenAI "logit_bias" validation, shared by /v1/chat/completions and /v1/completions.

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

// Object of base-10 token-id keys to numbers in [-100, 100], at most `cap` entries (0 = no cap).
// Returns "" and fills `out`, or the 400 message: no entry is dropped.
std::string parse_logit_bias(const nlohmann::json& body, int cap,
                             std::vector<std::pair<int32_t, float>>& out);

// 400 message naming the first id >= `vocab`, else "": the sampler skips such ids without a word.
std::string logit_bias_vocab_error(const std::vector<std::pair<int32_t, float>>& bias, int vocab);
