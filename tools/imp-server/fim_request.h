#pragma once

// Fill-in-the-middle request fields (#2201): POST /infill (llama.cpp shape) and the
// /v1/completions `suffix` field. Prompt assembly lives in src/model/fim.h.

#include "model/fim.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace imp {
class Tokenizer;
}  // namespace imp

struct FimRequest {
    bool active = false;
    imp::FimInput input;
    size_t bytes() const;  // all request text, for the pre-tokenize byte budget
};

// /infill: input_prefix, input_suffix (string, default ""), input_extra [{filename, text}],
// prompt (string), n_predict (> 0 maps to max_tokens). Returns "" or the 400 message.
std::string parse_infill_fields(nlohmann::json& body, FimRequest& out);

// /v1/completions `suffix`: a non-empty string turns on FIM with `prompt` as the prefix;
// "" or absent = plain completion. FIM needs a text prompt. Returns "" or the 400 message.
std::string parse_completion_suffix(const nlohmann::json& body, const std::string& prompt, bool token_prompt,
                                    FimRequest& out);

// Tokens for an active FIM request plus the FIM marker texts appended to `stops`.
// Returns "" or the 400 message when the tokenizer has no FIM tokens (code fim_not_supported).
std::string build_fim_request_tokens(const FimRequest& fim, const imp::Tokenizer& tok,
                                     std::vector<int32_t>& tokens, std::vector<std::string>& stops);
