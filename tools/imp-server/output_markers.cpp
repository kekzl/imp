#include "output_markers.h"

#include "model/tokenizer.h"

#include <iterator>

OutputMarkerAliases cohere_output_aliases(const imp::Tokenizer& tok) {
    static constexpr std::pair<const char*, const char*> kCohere[] = {
        {"<|START_THINKING|>", "<think>"},
        {"<|END_THINKING|>", "</think>"},
        {"<|START_TEXT|>", ""},
        {"<|END_TEXT|>", ""},
        {"<|START_ACTION|>", "<tool_call>"},
        {"<|END_ACTION|>", "</tool_call>"},
    };
    OutputMarkerAliases out;
    for (const auto& [marker, text] : kCohere) {
        const int32_t id = tok.find_token(marker);
        if (id < 0 || !tok.is_added_token(id))
            return {};
        out.push_back({id, text});
    }
    return out;
}

const std::string* output_alias(const OutputMarkerAliases& aliases, int32_t id) {
    for (const auto& a : aliases)
        if (a.id == id)
            return &a.text;
    return nullptr;
}

std::string decode_chat_piece(const imp::Tokenizer& tok, const OutputMarkerAliases& aliases, int32_t id) {
    const std::string* alias = output_alias(aliases, id);
    return alias ? *alias : tok.decode_token(id);
}

std::string decode_chat_output(const imp::Tokenizer& tok, const OutputMarkerAliases& aliases,
                               const std::vector<int32_t>& ids) {
    if (aliases.empty())
        return tok.decode(ids);
    // Runs between aliased ids decode as one span (byte-level merges stay intact).
    std::string out;
    auto run = ids.begin();
    for (auto it = ids.begin(); it != ids.end(); ++it) {
        const std::string* alias = output_alias(aliases, *it);
        if (alias == nullptr)
            continue;
        if (run != it)
            out += tok.decode(std::vector<int32_t>(run, it));
        out += *alias;
        run = std::next(it);
    }
    if (run != ids.end())
        out += tok.decode(std::vector<int32_t>(run, ids.end()));
    return out;
}
