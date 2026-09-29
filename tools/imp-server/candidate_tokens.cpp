#include "candidate_tokens.h"

#include <unordered_set>

namespace imp::server {

std::string prompt_tail_text(const imp::Tokenizer& tok, const std::vector<int32_t>& prompt) {
    size_t start = prompt.size();
    const size_t floor = prompt.size() > static_cast<size_t>(kCandidateTailTokens)
                             ? prompt.size() - static_cast<size_t>(kCandidateTailTokens)
                             : 0;
    while (start > floor && !tok.is_control_token(prompt[start - 1]))
        start--;
    if (start == prompt.size())
        return {};
    return tok.decode(
        std::vector<int32_t>(prompt.begin() + static_cast<std::ptrdiff_t>(start), prompt.end()));
}

namespace {

std::string quoted(const std::string& s) { return "\"" + s + "\""; }

}  // namespace

CandidateResolution resolve_candidate_tokens(const imp::Tokenizer& tok, const std::string& tail,
                                             const std::vector<std::string>& candidates) {
    CandidateResolution out;
    const std::vector<int32_t> base = tok.encode(tail, /*no_prefix=*/true);
    std::unordered_set<int32_t> seen;
    for (const auto& c : candidates) {
        const std::vector<int32_t> alone = tok.encode(c, /*no_prefix=*/true);
        if (alone.size() != 1) {
            out.error = "candidate " + quoted(c) + " is " + std::to_string(alone.size()) +
                        " tokens in this tokenizer, scoring needs exactly one";
            out.ids.clear();
            return out;
        }
        const int32_t id = alone[0];
        std::vector<int32_t> expect = base;
        expect.push_back(id);
        if (tok.encode(tail + c, /*no_prefix=*/true) != expect) {
            out.error = "candidate " + quoted(c) +
                        " merges with the end of the prompt: tokenize(prefix + candidate) != "
                        "tokenize(prefix) + [" +
                        std::to_string(id) + "]";
            out.ids.clear();
            return out;
        }
        if (!seen.insert(id).second) {
            out.error = "candidate " + quoted(c) + " maps to token " + std::to_string(id) +
                        " which another candidate already uses";
            out.ids.clear();
            return out;
        }
        out.ids.push_back(id);
    }
    return out;
}

std::string validate_candidate_ids(const imp::Tokenizer& tok, const std::vector<int32_t>& ids) {
    std::unordered_set<int32_t> seen;
    for (int32_t id : ids) {
        if (id < 0 || id >= tok.vocab_size())
            return "candidate token id " + std::to_string(id) + " is outside the vocabulary [0, " +
                   std::to_string(tok.vocab_size()) + ")";
        if (!seen.insert(id).second)
            return "candidate token id " + std::to_string(id) + " appears twice";
    }
    return {};
}

}  // namespace imp::server
