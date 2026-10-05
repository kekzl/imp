#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace imp {
class Tokenizer;
}  // namespace imp

// Chat-path output aliases: a control token the shared reasoning / tool-call parsers do not know is
// decoded as the text they do. Cohere2: <|START_THINKING|> -> <think>, <|START_ACTION|> ->
// <tool_call>, <|START_TEXT|> -> "". Completions and the tokenizer itself stay verbatim.
struct OutputMarkerAlias {
    int32_t id = -1;
    std::string text;
};
using OutputMarkerAliases = std::vector<OutputMarkerAlias>;

// Cohere2 table when all six markers are single added tokens; empty otherwise.
OutputMarkerAliases cohere_output_aliases(const imp::Tokenizer& tok);

// Alias text of `id`, nullptr when `id` has none.
const std::string* output_alias(const OutputMarkerAliases& aliases, int32_t id);

// decode_token / decode with the aliases applied.
std::string decode_chat_piece(const imp::Tokenizer& tok, const OutputMarkerAliases& aliases, int32_t id);
std::string decode_chat_output(const imp::Tokenizer& tok, const OutputMarkerAliases& aliases,
                               const std::vector<int32_t>& ids);
