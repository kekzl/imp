#include "tool_call_dialect.h"
#include "utils.h"

#include <string>
#include <utility>
#include <vector>

// Parses Qwen3.6's XML-flavored tool-call body (<function=NAME><parameter=KEY>VALUE</parameter>...):
// strings round-trip as strings, bare numerics coerce to JSON numbers.
bool parse_qwen36_xml_call(const std::string& body, ParsedToolCall& tc) {
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
    // Tries newline-anchored close tags first: the constrained-decode grammar
    // (schema_constrain.cu) only recognizes "\n</parameter>"/"\n</function>" as delimiters, so a raw
    // value may legally contain a bare close tag (e.g. code). Unanchored find is the fallback for
    // sloppy unconstrained output.
    size_t fn_close = body.find("\n</function>", pos);
    if (fn_close == std::string::npos)
        fn_close = body.find("</function>", pos);
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
        size_t pv_end = body.find("\n</parameter>", val_start);
        size_t pv_adv = 13;
        if (pv_end == std::string::npos || pv_end > scan_end) {
            pv_end = body.find("</parameter>", val_start);
            pv_adv = 12;
        }
        if (pv_end == std::string::npos || pv_end > scan_end)
            break;
        std::string val = body.substr(val_start, pv_end - val_start);
        trim(val);
        // Only a whole JSON scalar literal coerces; "600acab9" stays a string (#2103).
        // validate_tool_call restores `val` for string-typed params.
        json jv = json::parse(val, nullptr, false);
        if (jv.is_discarded() || !(jv.is_number() || jv.is_boolean() || jv.is_null()))
            jv = val;
        tc.raw_params.emplace_back(key, val);
        args[key] = std::move(jv);
        pos = pv_end + pv_adv;
    }
    tc.arguments = dump_safe(args);
    return true;
}

// GLM-4.x body: NAME<arg_key>K</arg_key><arg_value>V</arg_value>... The template writes strings raw
// and everything else as JSON (`tojson if v is not string`), so a value that parses as JSON is JSON.
bool parse_glm_arg_key_call(const std::string& body, ParsedToolCall& tc) {
    const size_t first = body.find("<arg_key>");
    const size_t name_end = first == std::string::npos ? body.size() : first;
    const size_t a = body.find_first_not_of("\n\r\t ");
    const size_t b = body.find_last_not_of("\n\r\t ", name_end == 0 ? 0 : name_end - 1);
    if (a == std::string::npos || b == std::string::npos || a > b || a >= name_end)
        return false;
    tc.name = body.substr(a, b - a + 1);
    json args = json::object();
    size_t pos = first;
    while (pos != std::string::npos) {
        const size_t k0 = pos + 9;  // "<arg_key>"
        const size_t k1 = body.find("</arg_key>", k0);
        const size_t v0 = body.find("<arg_value>", k1 == std::string::npos ? k0 : k1);
        const size_t v1 = v0 == std::string::npos ? v0 : body.find("</arg_value>", v0 + 11);
        if (k1 == std::string::npos || v1 == std::string::npos)
            return false;
        const std::string key = body.substr(k0, k1 - k0);
        const std::string val = body.substr(v0 + 11, v1 - v0 - 11);
        json jv = json::parse(val, nullptr, false);
        tc.raw_params.emplace_back(key, val);
        args[key] = jv.is_discarded() ? json(val) : std::move(jv);
        pos = body.find("<arg_key>", v1 + 12);
    }
    tc.arguments = dump_safe(args);
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

        // Decodes the body through the shared streaming body-parser (ChatML JSON, then Qwen3.6 XML
        // fallback) so the streaming and non-streaming tool-call paths cannot drift. id set only on success.
        ParsedToolCall tc;
        if (parse_stream_tool_body(body, /*gemma_body=*/false, /*fn_name=*/"", tc)) {
            tc.id = "call_imp_" + std::to_string(next_tool_call_id.fetch_add(1));
            calls.push_back(std::move(tc));
        }

        pos = end + skip_len;
    }

    return {content, calls};
}

std::string chatml_tool_prompt(const json& tools) {
    // ChatML (Qwen3, Hermes) and every family without its own dialect: <tool_call> format.
    return "\n\n# Tools\n\n"
           "You may call one or more functions to assist with the user query.\n\n"
           "<tools>\n" +
           dump_safe(tools) +
           "\n</tools>\n\n"
           "For each function call, return a JSON object within <tool_call></tool_call> XML tags:\n"
           "<tool_call>\n"
           "{\"name\": \"function_name\", \"arguments\": {\"param\": \"value\"}}\n"
           "</tool_call>\n\n"
           "If no function call is needed, respond normally without any tool_call tags.";
}

ToolTagScan scan_tool_call_tag(const std::string& buf, bool gemma_native) {
    ToolTagScan r;
    constexpr const char* kChatml = "<tool_call>";
    constexpr size_t kChatmlLen = 11;
    constexpr const char* kGemma = "<|tool_call>";
    constexpr size_t kGemmaLen = 12;

    size_t chatml_pos = buf.find(kChatml);
    size_t gemma_pos = gemma_native ? buf.find(kGemma) : std::string::npos;
    if (chatml_pos != std::string::npos || gemma_pos != std::string::npos) {
        r.kind = ToolTagScan::Kind::OPEN;
        if (gemma_pos != std::string::npos && (chatml_pos == std::string::npos || gemma_pos < chatml_pos)) {
            r.content_len = gemma_pos;
            r.body_start = gemma_pos + kGemmaLen;
            r.close_tag = "<tool_call|>";
            r.gemma_body = true;
        } else {
            r.content_len = chatml_pos;
            r.body_start = chatml_pos + kChatmlLen;
            r.close_tag = "</tool_call>";
        }
        return r;
    }

    if (suffix_is_marker_prefix(buf, kChatml, kChatmlLen) ||
        (gemma_native && suffix_is_marker_prefix(buf, kGemma, kGemmaLen))) {
        r.kind = ToolTagScan::Kind::PARTIAL;
        return r;
    }
    r.kind = ToolTagScan::Kind::NONE;
    return r;
}

void chatml_render_call(std::string& out, const std::string& name, const std::string& args, bool xml) {
    if (xml) {
        // Qwen-Coder XML dialect: mirror the template's tool_calls branch (raw-text values,
        // non-strings stringified).
        out += "\n<tool_call>\n<function=" + name + ">\n";
        json args_json = json::parse(args, nullptr, false);
        if (!args_json.is_discarded() && args_json.is_object()) {
            for (auto it = args_json.begin(); it != args_json.end(); ++it) {
                std::string val = it.value().is_string() ? it.value().get<std::string>()
                                                         : dump_safe(it.value());
                out += "<parameter=" + it.key() + ">\n" + val + "\n</parameter>\n";
            }
        }
        out += "</function>\n</tool_call>";
        return;
    }
    json call_obj = {{"name", name}, {"arguments", json::parse(args, nullptr, false)}};
    if (call_obj["arguments"].is_discarded())
        call_obj["arguments"] = args;
    out += "\n<tool_call>\n" + dump_safe(call_obj) + "\n</tool_call>";
}

namespace {

std::pair<std::string, std::vector<ParsedToolCall>> parse_chatml(const std::string& text,
                                                                 std::atomic<int>& next_id,
                                                                 const std::vector<std::string>&) {
    return parse_tool_calls_chatml(text, next_id);
}

ToolTagScan scan_chatml(const std::string& buf) { return scan_tool_call_tag(buf, /*gemma_native=*/false); }

}  // namespace

// ChatML <tool_call>{"name","arguments"}</tool_call>; the only dialect with the JSON constraint envelope.
extern const ToolCallDialect kChatmlToolDialect = {
    .name = "chatml",
    .prompt = chatml_tool_prompt,
    .parse = parse_chatml,
    .scan = scan_chatml,
    .render_call = chatml_render_call,
    .tool_response = nullptr,
    .json_envelope_open = "<tool_call>\n",
    .json_envelope_close = "\n</tool_call>",
    .xml_body = true,
    .forced_envelope = nullptr,
    .native_token = nullptr,
    .tool_response_joins_assistant = false,
    .gate_open = "<tool_call>",
    .gate_close = "</tool_call>",
    .gate_tokens = true,
};

// Families without a tool dialect of their own: ChatML wire format, prompt hint only (no enforcement).
extern const ToolCallDialect kChatmlHintToolDialect = {
    .name = "chatml-hint",
    .prompt = chatml_tool_prompt,
    .parse = parse_chatml,
    .scan = scan_chatml,
    .render_call = chatml_render_call,
    .tool_response = nullptr,
    .json_envelope_open = nullptr,
    .json_envelope_close = nullptr,
    .xml_body = false,
    .forced_envelope = nullptr,
    .native_token = nullptr,
    .tool_response_joins_assistant = false,
    .gate_open = "<tool_call>",
    .gate_close = "</tool_call>",
    .gate_tokens = true,
};
