#pragma once

#include <array>
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
    // Inkling2: one closing token (closes) ends whatever content marker opened last and decodes as that
    // marker's `close`. header: plain text up to the next marker is held, and dropped when that marker
    // sets drops_header (a tool name before its call marker).
    std::string close;
    bool closes = false;
    bool header = false;
    bool drops_header = false;
    // The marker, its body and its closer decode as nothing (thinking off on a marker protocol).
    bool silent = false;
    // First occurrence of rewrite_from after this marker becomes rewrite_to (tool JSON key).
    std::string rewrite_from, rewrite_to;
};
using OutputMarkerAliases = std::vector<OutputMarkerAlias>;

// Decode state of one output stream; only tables with closes / header entries use it.
struct OutputMarkerState {
    const OutputMarkerAlias* open = nullptr;
    bool holding = false;
    std::string held;
    bool rewrite_pending = false;
    std::string buf;
};

// Cohere2 table when all six markers are single added tokens; empty otherwise.
OutputMarkerAliases cohere_output_aliases(const imp::Tokenizer& tok);
// Inkling2 table (<|message_model|> NAME? <|content_*|> body <|end_message|>); empty when a marker is
// missing. think_start / think_end receive the ids of <|content_thinking|> and <|end_message|>.
OutputMarkerAliases inkling_output_aliases(const imp::Tokenizer& tok, int32_t& think_start,
                                           int32_t& think_end);

// The Inkling2 table over the ids of <|message_model|>, <|content_thinking|>, <|content_text|>,
// <|content_invoke_tool_json|>, <|end_message|> (in that order).
OutputMarkerAliases inkling_alias_table(const std::array<int32_t, 5>& ids);
// One decode step: marker a (non-null) or plain text between markers.
std::string output_marker_step(OutputMarkerState& st, const OutputMarkerAlias* a, std::string plain);

// Thinking off on a marker protocol (one closer shared by every content kind): the model opens the
// thinking block itself, so the block is decoded as nothing. True when the table was changed.
bool silence_marker_thinking(OutputMarkerAliases& aliases, bool explicit_off);

// Alias text of `id`, nullptr when `id` has none.
const std::string* output_alias(const OutputMarkerAliases& aliases, int32_t id);

// decode_token / decode with the aliases applied. st carries the open marker across calls; null
// decodes every alias as its text.
std::string decode_chat_piece(const imp::Tokenizer& tok, const OutputMarkerAliases& aliases, int32_t id,
                              OutputMarkerState* st = nullptr);
std::string decode_chat_output(const imp::Tokenizer& tok, const OutputMarkerAliases& aliases,
                               const std::vector<int32_t>& ids);
