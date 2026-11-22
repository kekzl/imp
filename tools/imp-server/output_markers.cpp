#include "output_markers.h"

#include "model/tokenizer.h"

#include <iterator>
#include <utility>

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

OutputMarkerAliases inkling_output_aliases(const imp::Tokenizer& tok, int32_t& think_start, int32_t& think_end) {
    OutputMarkerAliases out;
    auto add = [&](const char* marker, OutputMarkerAlias a) {
        a.id = tok.find_token(marker);
        out.push_back(std::move(a));
        return out.back().id;
    };
    add("<|message_model|>", {.header = true});
    const int32_t ts = add("<|content_thinking|>", {.text = "<think>", .close = "</think>"});
    add("<|content_text|>", {});
    // The body is {"name":NAME,"args":{...}}; the shared parser reads "arguments".
    add("<|content_invoke_tool_json|>", {.text = "<tool_call>",
                                         .close = "</tool_call>",
                                         .drops_header = true,
                                         .rewrite_from = "\"args\":",
                                         .rewrite_to = "\"arguments\":"});
    const int32_t te = add("<|end_message|>", {.closes = true});
    for (const auto& a : out)
        if (a.id < 0 || !tok.is_added_token(a.id))
            return {};
    think_start = ts;
    think_end = te;
    return out;
}

const std::string* output_alias(const OutputMarkerAliases& aliases, int32_t id) {
    for (const auto& a : aliases)
        if (a.id == id)
            return &a.text;
    return nullptr;
}

namespace {

const OutputMarkerAlias* find_alias(const OutputMarkerAliases& aliases, int32_t id) {
    for (const auto& a : aliases)
        if (a.id == id)
            return &a;
    return nullptr;
}

// Text between markers: held behind a header, or behind a pending key rewrite (at most 256 bytes).
std::string feed_plain(OutputMarkerState& st, std::string piece) {
    if (st.holding) {
        st.held += piece;
        return {};
    }
    if (!st.rewrite_pending)
        return piece;
    st.buf += piece;
    const size_t at = st.buf.find(st.open->rewrite_from);
    if (at == std::string::npos && st.buf.size() <= 256)
        return {};
    if (at != std::string::npos)
        st.buf.replace(at, st.open->rewrite_from.size(), st.open->rewrite_to);
    st.rewrite_pending = false;
    return std::exchange(st.buf, {});
}

std::string feed_marker(OutputMarkerState& st, const OutputMarkerAlias& a) {
    if (a.header) {
        st.holding = true;
        st.held.clear();
        return a.text;
    }
    std::string out;
    if (a.closes) {
        out = std::exchange(st.buf, {});
        st.rewrite_pending = false;
        if (st.open)
            out += st.open->close;
        st.open = nullptr;
        return out;
    }
    if (st.holding && !a.drops_header)
        out = st.held;
    st.holding = false;
    st.held.clear();
    st.open = &a;
    st.rewrite_pending = !a.rewrite_from.empty();
    st.buf.clear();
    return out + a.text;
}

}  // namespace

std::string decode_chat_piece(const imp::Tokenizer& tok, const OutputMarkerAliases& aliases, int32_t id,
                              OutputMarkerState* st) {
    const OutputMarkerAlias* a = find_alias(aliases, id);
    if (!st)
        return a ? a->text : tok.decode_token(id);
    return a ? feed_marker(*st, *a) : feed_plain(*st, tok.decode_token(id));
}

std::string decode_chat_output(const imp::Tokenizer& tok, const OutputMarkerAliases& aliases,
                               const std::vector<int32_t>& ids) {
    if (aliases.empty())
        return tok.decode(ids);
    // Runs between aliased ids decode as one span (byte-level merges stay intact).
    std::string out;
    OutputMarkerState st;
    auto run = ids.begin();
    for (auto it = ids.begin(); it != ids.end(); ++it) {
        const OutputMarkerAlias* a = find_alias(aliases, *it);
        if (a == nullptr)
            continue;
        if (run != it)
            out += feed_plain(st, tok.decode(std::vector<int32_t>(run, it)));
        out += feed_marker(st, *a);
        run = std::next(it);
    }
    if (run != ids.end())
        out += feed_plain(st, tok.decode(std::vector<int32_t>(run, ids.end())));
    return out + st.held + st.buf;
}
