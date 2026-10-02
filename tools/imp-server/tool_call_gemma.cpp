#include "tool_call_dialect.h"
#include "utils.h"

#include <cstring>
#include <string>
#include <utility>
#include <vector>

// Gemma-4's tool-call dialect is its own value syntax, not JSON, needing its own recursive-
// descent parser. Split out of tool_call.cpp, which otherwise mixed four unrelated concerns in
// one TU (the file-size gate's actual target).

// Gemma-4 native tool-call format (matches chat_template.jinja's format_argument macro):
// <|tool_call>call:NAME{key:value,...}<tool_call|>. Strings use <|"|>...<|"|>; bool/number/array/
// object as bare tokens (object recursive, unquoted keys). Re-emitted as JSON for ParsedToolCall.arguments.

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
    // Only a whole JSON number coerces; stod/stoll read "600acab9" as 600 (#2103).
    out = json::parse(tok, nullptr, false);
    if (out.is_discarded() || !out.is_number())
        out = tok;
    return true;
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

    // A forced tool_choice constrains a JSON arguments object here (#2279); Gemma's own syntax has
    // unquoted keys, so a body that parses as a JSON object is JSON.
    json args = json::parse(body.substr(brace), nullptr, false);
    bool ok = !args.is_discarded() && args.is_object();
    if (!ok) {
        size_t bp = brace;
        ok = parse_gemma_object(body, bp, args);
    }
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

namespace {

// JSON value -> Gemma's format_argument() output (with escape_keys=False
// for keys, since tool-call argument keys are bare identifiers in the
// chat-template macro).
std::string json_to_gemma_value(const json& v) {
    if (v.is_string()) {
        return std::string(kGemmaQuote) + v.get<std::string>() + kGemmaQuote;
    }
    if (v.is_boolean())
        return v.get<bool>() ? "true" : "false";
    if (v.is_null())
        return "null";
    if (v.is_number())
        return dump_safe(v);
    if (v.is_array()) {
        std::string out = "[";
        bool first = true;
        for (const auto& item : v) {
            if (!first)
                out += ",";
            out += json_to_gemma_value(item);
            first = false;
        }
        out += "]";
        return out;
    }
    if (v.is_object()) {
        std::string out = "{";
        bool first = true;
        for (auto it = v.begin(); it != v.end(); ++it) {
            if (!first)
                out += ",";
            out += it.key() + ":" + json_to_gemma_value(it.value());
            first = false;
        }
        out += "}";
        return out;
    }
    return dump_safe(v);
}

std::pair<std::string, std::vector<ParsedToolCall>> parse_gemma(const std::string& text,
                                                                std::atomic<int>& next_id,
                                                                const std::vector<std::string>&) {
    return parse_tool_calls_gemma(text, next_id);
}

bool parse_gemma_body(const std::string& body, const std::string&, ParsedToolCall& tc) {
    return parse_gemma_tool_call_body(body, tc);
}

// <|tool_call> (native, Gemma-4) or <tool_call> (the ChatML prompt this family is also given).
ToolTagScan scan_gemma(const std::string& buf) {
    ToolTagScan r = scan_tool_call_tag(buf, /*gemma_native=*/true);
    if (r.gemma_body)
        r.parse_body = parse_gemma_body;
    return r;
}

void gemma_render_call(std::string& out, const std::string& name, const std::string& args, bool) {
    json args_json = json::parse(args, nullptr, false);
    std::string args_body;
    if (!args_json.is_discarded() && args_json.is_object()) {
        bool first = true;
        for (auto it = args_json.begin(); it != args_json.end(); ++it) {
            if (!first)
                args_body += ",";
            args_body += it.key() + ":" + json_to_gemma_value(it.value());
            first = false;
        }
    }
    out += "<|tool_call>call:";
    out += name;
    out += "{";
    out += args_body;
    out += "}<tool_call|>";
}

// Gemma's chat template reaches the role=tool branch only via forward-scan from a preceding
// assistant-with-tool_calls message (template line ~215): the caller APPENDS this string to that
// assistant message (tool_response_joins_assistant).
std::string gemma_tool_response(const json& msg, const std::string& content) {
    std::string name = msg.value("name", "tool");
    return "<|tool_response>response:" + name + "{value:" + std::string(kGemmaQuote) + content + kGemmaQuote +
           "}<tool_response|>";
}

// A forced tool_choice constrains a JSON body here; parse_gemma_tool_call_body accepts it (#2279).
void gemma_forced_envelope(const std::string& name, std::string& open, std::string& close) {
    open = "<|tool_call>call:" + name;
    close = "<tool_call|>";
}

}  // namespace

// Gemma-4 <|tool_call>call:NAME{...}<tool_call|>; gemma-3 shares the family without the native token,
// so the forced envelope needs <|tool_call> in the vocab.
extern const ToolCallDialect kGemmaToolDialect = {
    .name = "gemma",
    .prompt = chatml_tool_prompt,
    .parse = parse_gemma,
    .scan = scan_gemma,
    .render_call = gemma_render_call,
    .tool_response = gemma_tool_response,
    .json_envelope_open = nullptr,
    .json_envelope_close = nullptr,
    .xml_body = false,
    .forced_envelope = gemma_forced_envelope,
    .native_token = "<|tool_call>",
    .tool_response_joins_assistant = true,
    .gate_open = "<|tool_call>",
    .gate_close = "<tool_call|>",
    .gate_tokens = true,
};
