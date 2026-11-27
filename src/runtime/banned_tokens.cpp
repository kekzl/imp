#include "runtime/banned_tokens.h"

#include "model/tokenizer.h"

#include <algorithm>
#include <iterator>
#include <string>

namespace imp {

namespace {

// Markers the model is trained to emit; banning one traps or derails generation.
constexpr const char* kTrainedMarkers[] = {
    "<think>", "</think>", "<|think|>", "<|/think|>",
    // Gemma-4 reasoning channel.
    "<|channel>", "<channel|>",
    // Harmony (gpt-oss, #547): banned, the model never left the analysis channel.
    "<|channel|>", "<|message|>", "<|start|>", "<|end|>", "<|constrain|>",
    // Inkling2: every reply is <|message_model|> + a content marker + <|end_message|>.
    "<|message_model|>", "<|content_thinking|>", "<|content_text|>", "<|end_message|>",
    "<|content_invoke_tool_json|>", "<|content_model_end_sampling|>",
    // Gemma-4 call `<|tool_call>call:NAME{k:<|"|>v<|"|>}<tool_call|>`: banned, 0 of 20 auto calls
    // (B-143). <|tool>, <tool|>, <tool_response|> stay banned: only the template writes them.
    "<|tool_call>", "<tool_call|>", "<|\"|>"};

// Legacy GGUF without token types: `<|..|>` markers and Gemma's angle-bracket specials.
bool is_legacy_control_text(const std::string& t) {
    if (t.size() < 3 || t[0] != '<' || t.back() != '>')
        return false;
    if (t.size() >= 4 && t[1] == '|' && t[t.size() - 2] == '|')
        return true;
    static const char* const kNames[] = {
        "<pad>",           "<unk>",         "<mask>",           "<unused0>",
        "<start_of_turn>", "<end_of_turn>", "<start_of_image>", "<end_of_image>"};
    return std::any_of(std::begin(kNames), std::end(kNames), [&](const char* n) { return t == n; });
}

std::vector<int32_t> kept_ids(const Tokenizer* tok, const std::vector<int32_t>& stop_ids) {
    std::vector<int32_t> keep = stop_ids;
    if (!tok)
        return keep;
    for (int32_t eid : tok->eos_ids())
        keep.push_back(eid);
    for (const char* name : kTrainedMarkers) {
        const int32_t tid = tok->find_token(name);
        if (tid >= 0)
            keep.push_back(tid);
    }
    return keep;
}

}  // namespace

std::vector<int32_t> build_banned_token_ids(const Tokenizer* tok, const std::vector<int32_t>& stop_ids,
                                            const std::vector<int32_t>& delimiters) {
    const std::vector<int32_t> keep = kept_ids(tok, stop_ids);
    auto is_kept = [&](int32_t id) { return std::find(keep.begin(), keep.end(), id) != keep.end(); };

    std::vector<int32_t> banned;
    for (int32_t id : delimiters)
        if (id >= 0 && !is_kept(id))
            banned.push_back(id);
    // Authoritative: token types (GGUF token_type, tokenizer.json `special`).
    const int vocab_size = tok ? tok->vocab_size() : 0;
    for (int i = 0; i < vocab_size; i++) {
        const bool control = tok->has_token_types() ? tok->is_control_token(i)
                                                    : is_legacy_control_text(tok->token_text(i));
        if (control && !is_kept(i))
            banned.push_back(i);
    }
    std::sort(banned.begin(), banned.end());
    banned.erase(std::unique(banned.begin(), banned.end()), banned.end());
    return banned;
}

}  // namespace imp
