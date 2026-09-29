// Candidate-token guard for /v1/decide and /v1/score (#2198): a candidate is scored as
// the logit of ONE token at the last prompt position, which is only meaningful if that
// token is what the tokenizer would produce for the candidate right after the prompt.
#pragma once

#include "model/tokenizer.h"

#include <cstdint>
#include <string>
#include <vector>

namespace imp::server {

// Tail tokens decoded: prompt ids after the last control token, at most this many.
inline constexpr int kCandidateTailTokens = 32;

// Text the candidate continues: decode of the prompt ids after its last control token.
std::string prompt_tail_text(const imp::Tokenizer& tok, const std::vector<int32_t>& prompt);

struct CandidateResolution {
    std::vector<int32_t> ids;  // one per candidate when error is empty
    std::string error;         // empty = every candidate passed
};

// Each candidate encodes to exactly one id, ids are distinct, and
// encode(tail + c) == encode(tail) + [id] (boundary merge, e.g. ":A" in Qwen).
CandidateResolution resolve_candidate_tokens(const imp::Tokenizer& tok, const std::string& tail,
                                             const std::vector<std::string>& candidates);

// Caller-given ids: in [0, vocab_size), distinct. Empty string = ok.
std::string validate_candidate_ids(const imp::Tokenizer& tok, const std::vector<int32_t>& ids);

}  // namespace imp::server
