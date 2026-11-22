#include "output_markers.h"

#include "model/tokenizer.h"

#include <array>
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

OutputMarkerAliases inkling_alias_table(const std::array<int32_t, 5>& ids) {
    OutputMarkerAliases out(5);
    out[0].header = true;
    out[1].text = "<think>";
    out[1].close = "</think>";
    // The call body is {"name":NAME,"args":{...}}; the shared parser reads "arguments".
    out[3].text = "<tool_call>";
    out[3].close = "</tool_call>";
    out[3].drops_header = true;
    out[3].rewrite_from = "\"args\":";
    out[3].rewrite_to = "\"arguments\":";
    out[4].closes = true;
    for (size_t i = 0; i < out.size(); ++i)
        out[i].id = ids[i];
    return out;
}

OutputMarkerAliases inkling_output_aliases(const imp::Tokenizer& tok, int32_t& think_start,
                                           int32_t& think_end) {
    static constexpr const char* kMarkers[5] = {"<|message_model|>", "<|content_thinking|>",
                                                "<|content_text|>", "<|content_invoke_tool_json|>",
                                                "<|end_message|>"};
    std::array<int32_t, 5> ids{};
    for (size_t i = 0; i < ids.size(); ++i) {
        ids[i] = tok.find_token(kMarkers[i]);
        if (ids[i] < 0 || !tok.is_added_token(ids[i]))
            return {};
    }
    think_start = ids[1];
    think_end = ids[4];
    return inkling_alias_table(ids);
}

bool silence_marker_thinking(OutputMarkerAliases& aliases, bool explicit_off) {
    bool protocol = false, changed = false;
    for (const auto& a : aliases)
        protocol = protocol || a.closes;
    if (!explicit_off || !protocol)
        return false;
    for (auto& a : aliases)
        if (a.text == "<think>") {
            a.text.clear();
            a.close.clear();
            a.silent = changed = true;
        }
    return changed;
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
    if (st.open && st.open->silent)
        return {};
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

std::string output_marker_step(OutputMarkerState& st, const OutputMarkerAlias* a, std::string plain) {
    return a ? feed_marker(st, *a) : feed_plain(st, std::move(plain));
}

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
