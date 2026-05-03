#include "tool_call.h"

#include <cstring>
#include <cstdlib>

std::string build_tool_prompt(imp::ChatTemplateFamily family, const json& tools, const json& tool_choice) {
    if (tools.empty())
        return "";

    // tool_choice "none" means no tool injection
    if (tool_choice.is_string() && tool_choice.get<std::string>() == "none")
        return "";

    std::string prompt;

    if (family == imp::ChatTemplateFamily::LLAMA3) {
        // Llama3 function calling format
        prompt = "\n\nYou have access to the following functions:\n\n";
        for (const auto& tool : tools) {
            if (!tool.contains("function"))
                continue;
            const auto& fn = tool["function"];
            json fn_desc = {{"name", fn.value("name", "")},
                            {"description", fn.value("description", "")},
                            {"parameters", fn.value("parameters", json::object())}};
            prompt += fn_desc.dump() + "\n\n";
        }
        prompt +=
            "For each function call, return a JSON object within <function=function_name> tags:\n"
            "<function=function_name>{\"param\": \"value\"}</function>\n\n"
            "If no function call is needed, respond normally without any function tags.";
    } else {
        // ChatML (Qwen3, Hermes) and all other families — use <tool_call> format
        prompt =
            "\n\n# Tools\n\n"
            "You may call one or more functions to assist with the user query.\n\n"
            "<tools>\n" +
            tools.dump() +
            "\n</tools>\n\n"
            "For each function call, return a JSON object within <tool_call></tool_call> XML tags:\n"
            "<tool_call>\n"
            "{\"name\": \"function_name\", \"arguments\": {\"param\": \"value\"}}\n"
            "</tool_call>\n\n"
            "If no function call is needed, respond normally without any tool_call tags.";
    }

    // Add constraints based on tool_choice
    if (tool_choice.is_string()) {
        std::string choice = tool_choice.get<std::string>();
        if (choice == "required") {
            prompt += "\n\nYou MUST call at least one tool.";
        }
    } else if (tool_choice.is_object() && tool_choice.contains("function")) {
        std::string fn_name = tool_choice["function"].value("name", "");
        if (!fn_name.empty()) {
            prompt += "\n\nYou MUST call the " + fn_name + " tool.";
        }
    }

    return prompt;
}

// Parse Qwen3.6's XML-flavored tool-call body:
//   <function=NAME>
//   <parameter=KEY1>
//   VALUE1
//   </parameter>
//   ...
//   </function>
// Strings round-trip as strings; bare numerics get coerced to JSON numbers.
static bool parse_qwen36_xml_call(const std::string& body, ParsedToolCall& tc) {
    size_t fn = body.find("<function=");
    if (fn == std::string::npos)
        return false;
    fn += 10;
    size_t fn_end = body.find('>', fn);
    if (fn_end == std::string::npos)
        return false;
    tc.name = body.substr(fn, fn_end - fn);
    auto trim = [](std::string& s) {
        auto a = s.find_first_not_of("\n\r\t ");
        auto b = s.find_last_not_of("\n\r\t ");
        if (a == std::string::npos) {
            s.clear();
            return;
        }
        s = s.substr(a, b - a + 1);
    };
    trim(tc.name);
    if (tc.name.empty())
        return false;

    json args = json::object();
    size_t pos = fn_end + 1;
    size_t fn_close = body.find("</function>", pos);
    size_t scan_end = (fn_close == std::string::npos) ? body.size() : fn_close;
    while (pos < scan_end) {
        size_t pk = body.find("<parameter=", pos);
        if (pk == std::string::npos || pk >= scan_end)
            break;
        pk += 11;
        size_t pk_end = body.find('>', pk);
        if (pk_end == std::string::npos || pk_end >= scan_end)
            break;
        std::string key = body.substr(pk, pk_end - pk);
        trim(key);
        size_t val_start = pk_end + 1;
        size_t pv_end = body.find("</parameter>", val_start);
        if (pv_end == std::string::npos || pv_end > scan_end)
            break;
        std::string val = body.substr(val_start, pv_end - val_start);
        trim(val);
        // Coerce bare numerics / true/false; otherwise keep as string.
        json jv;
        try {
            if (val == "true")
                jv = true;
            else if (val == "false")
                jv = false;
            else if (val == "null")
                jv = nullptr;
            else if (!val.empty() && (val[0] == '-' || val[0] == '.' || (val[0] >= '0' && val[0] <= '9'))) {
                if (val.find('.') != std::string::npos || val.find('e') != std::string::npos ||
                    val.find('E') != std::string::npos) {
                    jv = std::stod(val);
                } else {
                    jv = std::stoll(val);
                }
            } else {
                jv = val;
            }
        } catch (...) {
            jv = val;
        }
        args[key] = std::move(jv);
        pos = pv_end + 12;
    }
    tc.arguments = args.dump();
    return true;
}

std::pair<std::string, std::vector<ParsedToolCall>> parse_tool_calls_chatml(
    const std::string& text, std::atomic<int>& next_tool_call_id) {
    std::vector<ParsedToolCall> calls;
    std::string content;

    size_t pos = 0;
    size_t first_tag = text.find("<tool_call>");
    if (first_tag == std::string::npos) {
        return {text, {}};
    }

    // Content is everything before the first <tool_call>
    content = text.substr(0, first_tag);
    // Trim trailing whitespace
    auto last = content.find_last_not_of("\n\r\t ");
    if (last != std::string::npos)
        content = content.substr(0, last + 1);
    else
        content.clear();

    pos = first_tag;
    while (pos < text.size()) {
        size_t start = text.find("<tool_call>", pos);
        if (start == std::string::npos)
            break;
        start += 11;  // skip "<tool_call>"

        // Locate the closing tag. Some models (Qwen3.6) drift and emit a second
        // opening <tool_call> instead of </tool_call>; treat either as the
        // body delimiter so we still parse the call rather than dropping it.
        size_t end_proper = text.find("</tool_call>", start);
        size_t end_drift = text.find("<tool_call>", start);
        size_t end = end_proper;
        size_t skip_len = 12;  // "</tool_call>"
        if (end_proper == std::string::npos || (end_drift != std::string::npos && end_drift < end_proper)) {
            end = end_drift;
            skip_len = 11;
        }
        if (end == std::string::npos)
            break;

        std::string body = text.substr(start, end - start);
        // Trim whitespace
        auto bs = body.find_first_not_of("\n\r\t ");
        auto be = body.find_last_not_of("\n\r\t ");
        if (bs != std::string::npos && be != std::string::npos)
            body = body.substr(bs, be - bs + 1);

        // Two flavours: classic ChatML JSON ({"name": ..., "arguments": ...})
        // and Qwen3.6's XML-styled <function=...><parameter=...>... layout.
        bool parsed = false;
        if (!body.empty() && body[0] == '{') {
            try {
                json j = json::parse(body);
                ParsedToolCall tc;
                tc.id = "call_imp_" + std::to_string(next_tool_call_id.fetch_add(1));
                tc.name = j.value("name", "");
                if (j.contains("arguments")) {
                    tc.arguments = j["arguments"].dump();
                } else {
                    json args = j;
                    args.erase("name");
                    tc.arguments = args.dump();
                }
                if (!tc.name.empty()) {
                    calls.push_back(std::move(tc));
                    parsed = true;
                }
            } catch (...) { /* fall through */
            }
        }
        if (!parsed && body.find("<function=") != std::string::npos) {
            ParsedToolCall tc;
            tc.id = "call_imp_" + std::to_string(next_tool_call_id.fetch_add(1));
            if (parse_qwen36_xml_call(body, tc)) {
                calls.push_back(std::move(tc));
            }
        }

        pos = end + skip_len;
    }

    return {content, calls};
}

std::pair<std::string, std::vector<ParsedToolCall>> parse_tool_calls_llama3(
    const std::string& text, std::atomic<int>& next_tool_call_id) {
    std::vector<ParsedToolCall> calls;
    std::string content;

    size_t first_tag = text.find("<function=");
    if (first_tag == std::string::npos) {
        return {text, {}};
    }

    content = text.substr(0, first_tag);
    auto last = content.find_last_not_of("\n\r\t ");
    if (last != std::string::npos)
        content = content.substr(0, last + 1);
    else
        content.clear();

    size_t pos = first_tag;
    while (pos < text.size()) {
        size_t start = text.find("<function=", pos);
        if (start == std::string::npos)
            break;
        start += 10;  // skip "<function="

        size_t name_end = text.find('>', start);
        if (name_end == std::string::npos)
            break;

        std::string name = text.substr(start, name_end - start);

        size_t body_start = name_end + 1;
        size_t end = text.find("</function>", body_start);
        if (end == std::string::npos)
            break;

        std::string body = text.substr(body_start, end - body_start);
        auto bs = body.find_first_not_of("\n\r\t ");
        auto be = body.find_last_not_of("\n\r\t ");
        if (bs != std::string::npos && be != std::string::npos)
            body = body.substr(bs, be - bs + 1);

        try {
            // Validate it's valid JSON
            json j = json::parse(body);
            ParsedToolCall tc;
            tc.id = "call_imp_" + std::to_string(next_tool_call_id.fetch_add(1));
            tc.name = name;
            tc.arguments = j.dump();
            calls.push_back(std::move(tc));
        } catch (...) {
            // Malformed JSON — skip
        }

        pos = end + 11;  // skip "</function>"
    }

    return {content, calls};
}

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

constexpr const char* kGemmaQuote = "<|\"|>";
constexpr size_t kGemmaQuoteLen = 5;

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

        std::string body = text.substr(start, end - start);
        size_t bs = body.find_first_not_of("\n\r\t ");
        if (bs == std::string::npos) {
            pos = end + close_len;
            continue;
        }
        body = body.substr(bs);

        // Expect "call:NAME{...}"
        if (body.compare(0, 5, "call:") != 0) {
            pos = end + close_len;
            continue;
        }
        size_t brace = body.find('{', 5);
        if (brace == std::string::npos) {
            pos = end + close_len;
            continue;
        }

        std::string name = body.substr(5, brace - 5);
        auto ne = name.find_last_not_of("\n\r\t ");
        if (ne != std::string::npos)
            name = name.substr(0, ne + 1);

        // Parse the {...} args block as a Gemma object.
        size_t bp = brace;
        json args;
        bool ok = parse_gemma_object(body, bp, args);
        if (!ok || !name.empty()) {
            ParsedToolCall tc;
            tc.id = "call_imp_" + std::to_string(next_tool_call_id.fetch_add(1));
            tc.name = std::move(name);
            tc.arguments = ok ? args.dump() : "{}";
            if (!tc.name.empty())
                calls.push_back(std::move(tc));
        }

        pos = end + close_len;
    }

    return {content, calls};
}

std::pair<std::string, std::vector<ParsedToolCall>> parse_tool_calls(imp::ChatTemplateFamily family,
                                                                     const std::string& text,
                                                                     std::atomic<int>& next_tool_call_id) {
    if (family == imp::ChatTemplateFamily::LLAMA3)
        return parse_tool_calls_llama3(text, next_tool_call_id);
    if (family == imp::ChatTemplateFamily::GEMMA)
        return parse_tool_calls_gemma(text, next_tool_call_id);
    return parse_tool_calls_chatml(text, next_tool_call_id);
}

// JSON value -> Gemma's format_argument() output (with escape_keys=False
// for keys, since tool-call argument keys are bare identifiers in the
// chat-template macro).
static std::string json_to_gemma_value(const json& v) {
    if (v.is_string()) {
        return std::string(kGemmaQuote) + v.get<std::string>() + kGemmaQuote;
    }
    if (v.is_boolean())
        return v.get<bool>() ? "true" : "false";
    if (v.is_null())
        return "null";
    if (v.is_number())
        return v.dump();
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
    return v.dump();
}

std::string reconstruct_tool_call_output(imp::ChatTemplateFamily family, const json& tool_calls,
                                         const std::string& content) {
    std::string result;
    if (!content.empty() && content != "null") {
        result = content;
    }

    for (const auto& tc : tool_calls) {
        if (!tc.contains("function"))
            continue;
        std::string name = tc["function"].value("name", "");
        std::string args = tc["function"].value("arguments", "{}");

        if (family == imp::ChatTemplateFamily::LLAMA3) {
            result += "\n<function=" + name + ">" + args + "</function>";
        } else if (family == imp::ChatTemplateFamily::GEMMA) {
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
            result += "<|tool_call>call:" + name + "{" + args_body + "}<tool_call|>";
        } else {
            // ChatML format
            json call_obj = {{"name", name}, {"arguments", json::parse(args, nullptr, false)}};
            if (call_obj["arguments"].is_discarded())
                call_obj["arguments"] = args;
            result += "\n<tool_call>\n" + call_obj.dump() + "\n</tool_call>";
        }
    }

    return result;
}

std::string format_tool_response(imp::ChatTemplateFamily family, const json& msg) {
    std::string content = msg.value("content", "");

    if (family == imp::ChatTemplateFamily::LLAMA3) {
        return content;
    }
    if (family == imp::ChatTemplateFamily::GEMMA) {
        // Gemma native format. The chat-template's role=tool branch is reachable
        // only via forward-scan from a preceding assistant-with-tool_calls
        // message; standalone tool-role messages get skipped (template
        // line ~215). Caller in handlers.cpp must therefore APPEND this
        // string to the previous assistant ChatMessage's content rather
        // than push a separate ChatMessage.
        std::string name = msg.value("name", "tool");
        return "<|tool_response>response:" + name + "{value:" + std::string(kGemmaQuote) + content +
               kGemmaQuote + "}<tool_response|>";
    }
    // ChatML: wrap in <tool_response> tags
    return "<tool_response>\n" + content + "\n</tool_response>";
}
