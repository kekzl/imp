#include "tool_call.h"
#include "utils.h"

#include <cstring>
#include <string>
#include <utility>
#include <vector>

// Gemma-4's tool-call dialect is its own value syntax, not JSON, so it needs a
// small recursive-descent parser of its own. Split out of tool_call.cpp: that
// file otherwise carries four unrelated concerns (prompt building, three parse
// dialects, the streaming scanner, validation) in one translation unit, which
// is what the file-size gate is actually measuring.

// ---------------------------------------------------------------------------
// Gemma-4 native tool-call format. Pipe-delimited markers + a
// non-JSON value syntax matching chat_template.jinja's format_argument macro:
//
//   <|tool_call>call:NAME{key:value,key:value,...}<tool_call|>
//
// Values:
//   string  -> <|"|>contents<|"|>            (Google's quote-escape sequence)
//   bool    -> true | false
//   number  -> bare digits / -/./e
//   array   -> [v,v,...]
//   object  -> {key:v,key:v,...}             (recursive; bare keys, no quoting)
//
// We re-emit args as JSON for the OpenAI-style ParsedToolCall.arguments field.
// ---------------------------------------------------------------------------

namespace {

void skip_ws(const std::string& s, size_t& p) {
    while (p < s.size() && (s[p] == ' ' || s[p] == '\t' || s[p] == '\n' || s[p] == '\r'))
        ++p;
}

bool match(const std::string& s, size_t p, const char* lit) {
    size_t L = std::strlen(lit);
    return p + L <= s.size() && std::memcmp(s.data() + p, lit, L) == 0;
}

// Forward decl
bool parse_gemma_value(const std::string& s, size_t& p, json& out);

// Read a bare key up to ':'. Strips whitespace.
bool parse_gemma_key(const std::string& s, size_t& p, std::string& out) {
    skip_ws(s, p);
    size_t start = p;
    while (p < s.size() && s[p] != ':' && s[p] != ',' && s[p] != '}' && s[p] != '{')
        ++p;
    if (p == start)
        return false;
    size_t end = p;
    while (end > start && (s[end - 1] == ' ' || s[end - 1] == '\t'))
        --end;
    out.assign(s, start, end - start);
    return !out.empty();
}

bool parse_gemma_string(const std::string& s, size_t& p, json& out) {
    if (!match(s, p, kGemmaQuote))
        return false;
    p += kGemmaQuoteLen;
    size_t start = p;
    size_t end = s.find(kGemmaQuote, p);
    if (end == std::string::npos)
        return false;
    out = s.substr(start, end - start);
    p = end + kGemmaQuoteLen;
    return true;
}

bool parse_gemma_object(const std::string& s, size_t& p, json& out) {
    if (p >= s.size() || s[p] != '{')
        return false;
    ++p;
    out = json::object();
    skip_ws(s, p);
    if (p < s.size() && s[p] == '}') {
        ++p;
        return true;
    }
    while (p < s.size()) {
        std::string key;
        if (!parse_gemma_key(s, p, key))
            return false;
        skip_ws(s, p);
        if (p >= s.size() || s[p] != ':')
            return false;
        ++p;
        json value;
        if (!parse_gemma_value(s, p, value))
            return false;
        out[key] = std::move(value);
        skip_ws(s, p);
        if (p < s.size() && s[p] == ',') {
            ++p;
            skip_ws(s, p);
            continue;
        }
        if (p < s.size() && s[p] == '}') {
            ++p;
            return true;
        }
        return false;
    }
    return false;
}

bool parse_gemma_array(const std::string& s, size_t& p, json& out) {
    if (p >= s.size() || s[p] != '[')
        return false;
    ++p;
    out = json::array();
    skip_ws(s, p);
    if (p < s.size() && s[p] == ']') {
        ++p;
        return true;
    }
    while (p < s.size()) {
        json item;
        if (!parse_gemma_value(s, p, item))
            return false;
        out.push_back(std::move(item));
        skip_ws(s, p);
        if (p < s.size() && s[p] == ',') {
            ++p;
            skip_ws(s, p);
            continue;
        }
        if (p < s.size() && s[p] == ']') {
            ++p;
            return true;
        }
        return false;
    }
    return false;
}

// Tries string -> object -> array -> bool -> number (in that order).
bool parse_gemma_value(const std::string& s, size_t& p, json& out) {
    skip_ws(s, p);
    if (p >= s.size())
        return false;
    if (match(s, p, kGemmaQuote))
        return parse_gemma_string(s, p, out);
    if (s[p] == '{')
        return parse_gemma_object(s, p, out);
    if (s[p] == '[')
        return parse_gemma_array(s, p, out);
    if (match(s, p, "true")) {
        out = true;
        p += 4;
        return true;
    }
    if (match(s, p, "false")) {
        out = false;
        p += 5;
        return true;
    }
    if (match(s, p, "null")) {
        out = nullptr;
        p += 4;
        return true;
    }
    // Number: bare token until terminator
    size_t start = p;
    while (p < s.size() && s[p] != ',' && s[p] != '}' && s[p] != ']' && s[p] != ' ' && s[p] != '\t' &&
           s[p] != '\n' && s[p] != '\r')
        ++p;
    if (p == start)
        return false;
    std::string tok = s.substr(start, p - start);
    try {
        if (tok.find('.') != std::string::npos || tok.find('e') != std::string::npos ||
            tok.find('E') != std::string::npos) {
            out = std::stod(tok);
        } else {
            out = std::stoll(tok);
        }
        return true;
    } catch (...) {
        out = tok;  // fall back to string
        return true;
    }
}

}  // namespace

// Single Gemma-4 tool-call body: "call:NAME{key:value,...}" (markers already
// stripped). Shared by the non-streaming parser below and the streaming paths.
bool parse_gemma_tool_call_body(const std::string& body_in, ParsedToolCall& tc) {
    size_t bs = body_in.find_first_not_of("\n\r\t ");
    if (bs == std::string::npos)
        return false;
    std::string body = body_in.substr(bs);

    if (body.compare(0, 5, "call:") != 0)
        return false;
    size_t brace = body.find('{', 5);
    if (brace == std::string::npos)
        return false;

    std::string name = body.substr(5, brace - 5);
    auto ne = name.find_last_not_of("\n\r\t ");
    name = (ne != std::string::npos) ? name.substr(0, ne + 1) : std::string();
    if (name.empty())
        return false;

    size_t bp = brace;
    json args;
    bool ok = parse_gemma_object(body, bp, args);
    tc.name = std::move(name);
    tc.arguments = ok ? dump_safe(args) : "{}";
    return true;
}

std::pair<std::string, std::vector<ParsedToolCall>> parse_tool_calls_gemma(
    const std::string& text, std::atomic<int>& next_tool_call_id) {
    std::vector<ParsedToolCall> calls;
    std::string content;

    constexpr const char* kOpen = "<|tool_call>";
    constexpr const char* kClose = "<tool_call|>";
    size_t open_len = std::strlen(kOpen);
    size_t close_len = std::strlen(kClose);

    size_t first = text.find(kOpen);
    if (first == std::string::npos)
        return {text, {}};

    content = text.substr(0, first);
    auto last = content.find_last_not_of("\n\r\t ");
    if (last != std::string::npos)
        content = content.substr(0, last + 1);
    else
        content.clear();

    size_t pos = first;
    while (pos < text.size()) {
        size_t start = text.find(kOpen, pos);
        if (start == std::string::npos)
            break;
        start += open_len;
        size_t end = text.find(kClose, start);
        if (end == std::string::npos)
            break;

        // Expect "call:NAME{...}" — shared single-call body parser.
        ParsedToolCall tc;
        if (parse_gemma_tool_call_body(text.substr(start, end - start), tc)) {
            tc.id = "call_imp_" + std::to_string(next_tool_call_id.fetch_add(1));
            calls.push_back(std::move(tc));
        }

        pos = end + close_len;
    }

    return {content, calls};
}
