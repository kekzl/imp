#pragma once

// OpenAI "logit_bias" validation, shared by /v1/chat/completions and /v1/completions.

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

namespace imp {
class PhraseRegex;
}  // namespace imp

// Object of base-10 token-id keys to numbers in [-100, 100], at most `cap` entries (0 = no cap).
// Returns "" and fills `out`, or the 400 message: no entry is dropped.
std::string parse_logit_bias(const nlohmann::json& body, int cap,
                             std::vector<std::pair<int32_t, float>>& out);

// 400 message naming the first id >= `vocab`, else "": the sampler skips such ids without a word.
std::string logit_bias_vocab_error(const std::vector<std::pair<int32_t, float>>& bias, int vocab);

// "banned_strings" (row 118): an array of 1..kMaxBannedPhrases non-empty strings of at most
// kMaxBannedPhraseBytes bytes each (runtime/phrase_ban.h). Returns "" and fills `out`, or the 400 message.
std::string parse_banned_strings(const nlohmann::json& v, std::vector<std::string>& out);

// "banned_regex" (row 118): 1..kMaxBannedRegex PCRE2 patterns of at most kMaxBannedPhraseBytes bytes,
// compiled (runtime/phrase_regex.h); a pattern that fails to compile or matches "" is a 400.
std::string parse_banned_regex(const nlohmann::json& v,
                               std::vector<std::shared_ptr<const imp::PhraseRegex>>& out);
