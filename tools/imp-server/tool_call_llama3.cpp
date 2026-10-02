#include "tool_call_dialect.h"
#include "utils.h"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

std::pair<std::string, std::vector<ParsedToolCall>> parse_tool_calls_llama3(
    const std::string& text, std::atomic<int>& next_tool_call_id,
    const std::vector<std::string>& known_tool_names) {
    std::vector<ParsedToolCall> calls;
    std::string content;

    size_t first_tag = text.find("<function=");
    if (first_tag == std::string::npos) {
        // Llama 3.2 emits a bare JSON object ({"name":F,"parameters":{...}}) instead of 3.1's
        // <function=F> envelope; without recognizing it the call reached the caller as plain `content`.
        // Deliberately strict (non-empty string name + object parameters/arguments, nothing else) so a
        // plain JSON answer is never mistaken for a call.
        std::string trimmed = text;
        auto b = trimmed.find_first_not_of("\n\r\t ");
        auto e = trimmed.find_last_not_of("\n\r\t ");
        if (b == std::string::npos)
            return {text, {}};
        trimmed = trimmed.substr(b, e - b + 1);
        if (trimmed.front() != '{')
            return {text, {}};
        // Takes the FIRST balanced JSON object: a small model asked for one call can emit several
        // separated by "; " (Llama-3.2-3B does this), and parsing the whole string then fails outright.
        // Brace counting is string-aware so a '}' inside a value doesn't end the object early.
        size_t depth = 0, end_obj = std::string::npos;
        bool in_str = false, esc = false;
        for (size_t i = 0; i < trimmed.size(); i++) {
            char c = trimmed[i];
            if (in_str) {
                if (esc)
                    esc = false;
                else if (c == '\\')
                    esc = true;
                else if (c == '"')
                    in_str = false;
                continue;
            }
            if (c == '"')
                in_str = true;
            else if (c == '{')
                depth++;
            else if (c == '}' && --depth == 0) {
                end_obj = i;
                break;
            }
        }
        if (end_obj == std::string::npos)
            return {text, {}};
        trimmed = trimmed.substr(0, end_obj + 1);
        try {
            json j = json::parse(trimmed);
            if (j.is_object() && j.contains("name") && j["name"].is_string() &&
                !j["name"].get<std::string>().empty()) {
                const char* key = j.contains("parameters")  ? "parameters"
                                  : j.contains("arguments") ? "arguments"
                                                            : nullptr;
                const std::string cand = j["name"].get<std::string>();
                const bool known = std::find(known_tool_names.begin(), known_tool_names.end(), cand) !=
                                   known_tool_names.end();
                if (key && j[key].is_object() && known) {
                    ParsedToolCall tc;
                    tc.name = cand;
                    tc.arguments = dump_safe(j[key]);
                    tc.id = "call_imp_" + std::to_string(next_tool_call_id.fetch_add(1));
                    calls.push_back(std::move(tc));
                    return {std::string(), calls};
                }
            }
        } catch (...) {
            return {text, {}};  // not JSON: plain content
        }
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

        // Same shared body-parser: with fn_name set it takes the Llama3 branch
        // (name from the open tag, body is the bare JSON args). id on success.
        ParsedToolCall tc;
        if (parse_stream_tool_body(body, /*gemma_body=*/false, /*fn_name=*/name, tc)) {
            tc.id = "call_imp_" + std::to_string(next_tool_call_id.fetch_add(1));
            calls.push_back(std::move(tc));
        }

        pos = end + 11;  // skip "</function>"
    }

    return {content, calls};
}

namespace {

std::string llama3_tool_prompt(const json& tools) {
    std::string prompt = "\n\nYou have access to the following functions:\n\n";
    for (const auto& tool : tools) {
        if (!tool.contains("function"))
            continue;
        const auto& fn = tool["function"];
        json fn_desc = {{"name", fn.value("name", "")},
                        {"description", fn.value("description", "")},
                        {"parameters", fn.value("parameters", json::object())}};
        prompt += dump_safe(fn_desc) + "\n\n";
    }
    prompt +=
        "For each function call, return a JSON object within <function=function_name> tags:\n"
        "<function=function_name>{\"param\": \"value\"}</function>\n\n"
        "If no function call is needed, respond normally without any function tags.";
    return prompt;
}

ToolTagScan scan_llama3(const std::string& buf) {
    ToolTagScan r;
    constexpr const char* kFn = "<function=";
    constexpr size_t kFnLen = 10;
    size_t fn_pos = buf.find(kFn);
    if (fn_pos != std::string::npos) {
        size_t gt = buf.find('>', fn_pos + kFnLen);
        if (gt == std::string::npos) {
            r.kind = ToolTagScan::Kind::PARTIAL;  // still waiting for '>'
            return r;
        }
        r.kind = ToolTagScan::Kind::OPEN;
        r.content_len = fn_pos;
        r.body_start = gt + 1;
        r.close_tag = "</function>";
        r.fn_name = buf.substr(fn_pos + kFnLen, gt - (fn_pos + kFnLen));
        return r;
    }
    r.kind = suffix_is_marker_prefix(buf, kFn, kFnLen) ? ToolTagScan::Kind::PARTIAL : ToolTagScan::Kind::NONE;
    return r;
}

void llama3_render_call(std::string& out, const std::string& name, const std::string& args, bool) {
    out += "\n<function=";
    out += name;
    out += ">";
    out += args;
    out += "</function>";
}

void llama3_forced_envelope(const std::string& name, std::string& open, std::string& close) {
    open = "<function=" + name + ">";
    close = "</function>";
}

}  // namespace

// <function=NAME>{args}</function>: dynamic NAME, so the preamble gate is a char prefix only.
extern const ToolCallDialect kLlama3ToolDialect = {
    .name = "llama3",
    .prompt = llama3_tool_prompt,
    .parse = parse_tool_calls_llama3,
    .scan = scan_llama3,
    .render_call = llama3_render_call,
    .tool_response = nullptr,
    .json_envelope_open = nullptr,
    .json_envelope_close = nullptr,
    .xml_body = false,
    .forced_envelope = llama3_forced_envelope,
    .native_token = nullptr,
    .tool_response_joins_assistant = false,
    .gate_open = "<function=",
    .gate_close = "</function>",
    .gate_tokens = false,
};
