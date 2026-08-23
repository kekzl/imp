#include "tool_call.h"
#include "utils.h"

#include <algorithm>
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
            prompt += dump_safe(fn_desc) + "\n\n";
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
            dump_safe(tools) +
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

bool tool_choice_is_enforceable(imp::ChatTemplateFamily family, const json& tool_choice) {
    if (tool_choice.is_object() && tool_choice.contains("function")) {
        if (tool_choice["function"].value("name", "").empty())
            return true;  // an object without a name forces nothing
        return family == imp::ChatTemplateFamily::CHATML || family == imp::ChatTemplateFamily::LLAMA3;
    }
    if (tool_choice.is_string() && tool_choice.get<std::string>() == "required")
        return family == imp::ChatTemplateFamily::CHATML;
    return true;
}

std::vector<std::pair<std::string, std::string>> collect_tool_constraint(imp::ChatTemplateFamily family,
                                                                         const json& tools,
                                                                         const json& tool_choice) {
    std::vector<std::pair<std::string, std::string>> out;
    if (tools.empty())
        return out;
    // Stage 1 (#1002): only the ChatML `<tool_call>` JSON envelope — the
    // dialect build_tool_prompt instructs for this family. Llama3/Gemma
    // envelopes and the Qwen3.6 XML flavor keep the prompt-hint path.
    if (family != imp::ChatTemplateFamily::CHATML)
        return out;

    std::string forced_name;
    if (tool_choice.is_object() && tool_choice.contains("function"))
        forced_name = tool_choice["function"].value("name", "");
    const bool required = tool_choice.is_string() && tool_choice.get<std::string>() == "required";
    if (forced_name.empty() && !required)
        return out;

    for (const auto& tool : tools) {
        if (!tool.contains("function"))
            continue;
        const auto& fn = tool["function"];
        std::string name = fn.value("name", "");
        if (name.empty() || (!forced_name.empty() && name != forced_name))
            continue;
        // Enforceable parameters only — an absent/free-form schema would
        // dead-end the FSM's key phase (engine-side builder re-checks).
        if (!fn.contains("parameters") || !fn["parameters"].is_object() ||
            !fn["parameters"].contains("properties") || fn["parameters"]["properties"].empty()) {
            out.clear();
            return out;  // one unenforceable tool → whole request falls back
        }
        out.emplace_back(std::move(name), dump_safe(fn["parameters"]));
    }
    if (!forced_name.empty() && out.size() != 1)
        out.clear();  // forced function not found — fall back to the hint
    return out;
}

std::pair<std::string, std::string> collect_llama3_forced_tool(imp::ChatTemplateFamily family,
                                                               const json& tools, const json& tool_choice) {
    // Llama3 tool calls are `<function=NAME>{JSON args}</function>` — the body
    // IS the arguments object (name is in the tag), so a forced single function
    // maps onto the plain parameter schema with a per-tool envelope. Only the
    // forced-function case is enforceable here: "required"/auto would need a
    // name-in-tag enum binding (follow-up); Gemma/Qwen3.6-XML bodies are
    // non-JSON and need a separate grammar (out of scope for the JSON FSM).
    if (family != imp::ChatTemplateFamily::LLAMA3 || tools.empty())
        return {};
    if (!tool_choice.is_object() || !tool_choice.contains("function"))
        return {};
    std::string forced = tool_choice["function"].value("name", "");
    if (forced.empty())
        return {};
    for (const auto& tool : tools) {
        if (!tool.contains("function"))
            continue;
        const auto& fn = tool["function"];
        if (fn.value("name", "") != forced)
            continue;
        // Enforceable parameters only (an absent/free-form schema dead-ends the
        // FSM's key phase — the engine-side parser re-checks).
        if (!fn.contains("parameters") || !fn["parameters"].is_object() ||
            !fn["parameters"].contains("properties") || fn["parameters"]["properties"].empty())
            return {};
        return {forced, dump_safe(fn["parameters"])};
    }
    return {};
}

std::vector<std::pair<std::string, std::string>> collect_strict_tool_constraint(
    imp::ChatTemplateFamily family, const json& tools, const json& tool_choice) {
    std::vector<std::pair<std::string, std::string>> out;
    if (tools.empty())
        return out;
    // ChatML `<tool_call>` JSON envelope only (as the forced path).
    if (family != imp::ChatTemplateFamily::CHATML)
        return out;
    // Optional strict enforcement applies only when the model is FREE to decide:
    // tool_choice auto/absent. A forced function or "required" is mandatory and
    // goes the forced-envelope path (collect_tool_constraint); "none"/unknown
    // suppress tools.
    if (tool_choice.is_object())
        return out;
    if (tool_choice.is_string() && tool_choice.get<std::string>() != "auto")
        return out;

    // Every callable tool must declare `strict: true` AND carry enforceable
    // params. A mixed strict/non-strict set falls back to the prompt hint —
    // the uniform TOOL_CALL enum would otherwise over-constrain the arguments
    // of a tool whose caller never asked for schema adherence.
    for (const auto& tool : tools) {
        if (!tool.contains("function"))
            continue;
        const auto& fn = tool["function"];
        std::string name = fn.value("name", "");
        if (name.empty() || !fn.value("strict", false))
            return {};
        if (!fn.contains("parameters") || !fn["parameters"].is_object() ||
            !fn["parameters"].contains("properties") || fn["parameters"]["properties"].empty())
            return {};  // one unenforceable tool → whole request falls back
        out.emplace_back(std::move(name), dump_safe(fn["parameters"]));
    }
    return out;
}

// Parse Qwen3.6's XML-flavored tool-call body:
//   <function=NAME>
//   <parameter=KEY1>
//   VALUE1
//   </parameter>
//   ...
//   </function>
// Strings round-trip as strings; bare numerics get coerced to JSON numbers.
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
    // Newline-anchored close tags first — the constrained-decode grammar
    // (schema_constrain.cu XML phases) only recognizes "\n</parameter>" /
    // "\n</function>" as delimiters, so a raw value may legally CONTAIN a
    // bare close tag (code writing about tool calls). The unanchored find is
    // kept as a fallback for sloppy unconstrained output.
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
        pos = pv_end + pv_adv;
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

        // Decode the body through the shared streaming body-parser — classic
        // ChatML JSON ({"name":..., "arguments":...}) then Qwen3.6's XML-styled
        // <function=...><parameter=...>... fallback — so the streaming and
        // non-streaming paths cannot drift. id is assigned only on success.
        ParsedToolCall tc;
        if (parse_stream_tool_body(body, /*gemma_body=*/false, /*fn_name=*/"", tc)) {
            tc.id = "call_imp_" + std::to_string(next_tool_call_id.fetch_add(1));
            calls.push_back(std::move(tc));
        }

        pos = end + skip_len;
    }

    return {content, calls};
}

std::vector<std::string> tool_names_from_request(const json& tools) {
    std::vector<std::string> names;
    if (!tools.is_array())
        return names;
    for (const auto& t : tools) {
        if (!t.is_object())
            continue;
        const json& fn = t.contains("function") ? t["function"] : t;
        if (fn.is_object() && fn.contains("name") && fn["name"].is_string())
            names.push_back(fn["name"].get<std::string>());
    }
    return names;
}

std::pair<std::string, std::vector<ParsedToolCall>> parse_tool_calls_llama3(
    const std::string& text, std::atomic<int>& next_tool_call_id,
    const std::vector<std::string>& known_tool_names) {
    std::vector<ParsedToolCall> calls;
    std::string content;

    size_t first_tag = text.find("<function=");
    if (first_tag == std::string::npos) {
        // Llama 3.2 emits a bare JSON object — {"name": F, "parameters": {...}}
        // — where 3.1 used the <function=F> envelope above. Without this the
        // call is handed back as `content` and an agent never sees a tool call,
        // even though the model and the constrained grammar did their job.
        // Deliberately strict: an object with a non-empty string `name` and an
        // object `parameters`/`arguments`, nothing else, so a plain JSON answer
        // is not mistaken for a call.
        std::string trimmed = text;
        auto b = trimmed.find_first_not_of("\n\r\t ");
        auto e = trimmed.find_last_not_of("\n\r\t ");
        if (b == std::string::npos)
            return {text, {}};
        trimmed = trimmed.substr(b, e - b + 1);
        if (trimmed.front() != '{')
            return {text, {}};
        // Take the FIRST balanced object: a small model asked for one call can
        // emit several, separated by "; ", and parsing the whole string then
        // fails outright (Llama-3.2-3B does this with a two-property schema).
        // Brace counting is string-aware so a '}' inside a value doesn't end it.
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
            // not JSON — plain content
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

std::pair<std::string, std::vector<ParsedToolCall>> parse_tool_calls(
    imp::ChatTemplateFamily family, const std::string& text, std::atomic<int>& next_tool_call_id,
    const std::vector<std::string>& known_tool_names) {
    if (family == imp::ChatTemplateFamily::LLAMA3)
        return parse_tool_calls_llama3(text, next_tool_call_id, known_tool_names);
    if (family == imp::ChatTemplateFamily::GEMMA)
        return parse_tool_calls_gemma(text, next_tool_call_id);
    return parse_tool_calls_chatml(text, next_tool_call_id);
}

// ---------------------------------------------------------------------------
// Streaming tag scanner + body parser (used by StreamToolCallFilter).
// ---------------------------------------------------------------------------

namespace {

// True iff some suffix of buf is a proper prefix of marker m — i.e. more
// bytes could still complete the marker. Used for the streaming holdback.
bool suffix_is_marker_prefix(const std::string& buf, const char* m, size_t mlen) {
    size_t maxk = std::min(buf.size(), mlen - 1);
    for (size_t k = maxk; k >= 1; --k) {
        if (std::memcmp(buf.data() + buf.size() - k, m, k) == 0)
            return true;
    }
    return false;
}

}  // namespace

ToolTagScan scan_tool_tag(const std::string& buf, imp::ChatTemplateFamily family) {
    ToolTagScan r;

    if (family == imp::ChatTemplateFamily::LLAMA3) {
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
        r.kind = suffix_is_marker_prefix(buf, kFn, kFnLen) ? ToolTagScan::Kind::PARTIAL
                                                           : ToolTagScan::Kind::NONE;
        return r;
    }

    // ChatML "<tool_call>" for all non-Llama3 families; the GEMMA family
    // additionally recognises the native pipe-delimited "<|tool_call>".
    constexpr const char* kChatml = "<tool_call>";
    constexpr size_t kChatmlLen = 11;
    constexpr const char* kGemma = "<|tool_call>";
    constexpr size_t kGemmaLen = 12;
    const bool gemma = (family == imp::ChatTemplateFamily::GEMMA);

    size_t chatml_pos = buf.find(kChatml);
    size_t gemma_pos = gemma ? buf.find(kGemma) : std::string::npos;
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
        (gemma && suffix_is_marker_prefix(buf, kGemma, kGemmaLen))) {
        r.kind = ToolTagScan::Kind::PARTIAL;
        return r;
    }
    r.kind = ToolTagScan::Kind::NONE;
    return r;
}

bool parse_stream_tool_body(const std::string& body, bool gemma_body, const std::string& fn_name,
                            ParsedToolCall& tc) {
    if (gemma_body)
        return parse_gemma_tool_call_body(body, tc);

    if (!fn_name.empty()) {
        // Llama3: name came from the open tag, the body is the bare JSON args.
        try {
            json j = json::parse(body);
            tc.name = fn_name;
            tc.arguments = dump_safe(j);
            return true;
        } catch (...) {
            return false;
        }
    }

    // ChatML: classic JSON body first ...
    try {
        json j = json::parse(body);
        tc.name = j.value("name", "");
        if (j.contains("arguments")) {
            tc.arguments = dump_safe(j["arguments"]);
        } else {
            json args = j;
            args.erase("name");
            tc.arguments = dump_safe(args);
        }
        if (!tc.name.empty())
            return true;
    } catch (...) {
        // fall through to the Qwen3.6 XML layout
    }
    // ... then Qwen3.6's <function=NAME><parameter=K>V</parameter> layout.
    if (body.find("<function=") != std::string::npos && parse_qwen36_xml_call(body, tc))
        return true;
    return false;
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

std::string reconstruct_tool_call_output(imp::ChatTemplateFamily family, const json& tool_calls,
                                         const std::string& content, bool xml) {
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
            result += "\n<function=";
            result += name;
            result += ">";
            result += args;
            result += "</function>";
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
            result += "<|tool_call>call:";
            result += name;
            result += "{";
            result += args_body;
            result += "}<tool_call|>";
        } else if (xml) {
            // Qwen-Coder XML dialect — mirror the template's tool_calls
            // branch: raw-text values, non-strings stringified.
            result += "\n<tool_call>\n<function=" + name + ">\n";
            json args_json = json::parse(args, nullptr, false);
            if (!args_json.is_discarded() && args_json.is_object()) {
                for (auto it = args_json.begin(); it != args_json.end(); ++it) {
                    std::string val = it.value().is_string() ? it.value().get<std::string>()
                                                             : dump_safe(it.value());
                    result += "<parameter=" + it.key() + ">\n" + val + "\n</parameter>\n";
                }
            }
            result += "</function>\n</tool_call>";
        } else {
            // ChatML format
            json call_obj = {{"name", name}, {"arguments", json::parse(args, nullptr, false)}};
            if (call_obj["arguments"].is_discarded())
                call_obj["arguments"] = args;
            result += "\n<tool_call>\n" + dump_safe(call_obj) + "\n</tool_call>";
        }
    }

    return result;
}

// ---------------------------------------------------------------------------
// Tool-call argument validation (self-contained; no engine constraint code).
// ---------------------------------------------------------------------------

// Does a parsed JSON value match a JSON-schema "type" string? Only the
// top-level scalar/container kinds are checked — enough to catch the common
// hallucination failure modes (string where a number is required, missing
// object, etc.) without reimplementing a full validator.
static bool json_type_matches(const json& v, const std::string& type) {
    if (type == "string")
        return v.is_string();
    if (type == "integer")
        return v.is_number_integer() || v.is_number_unsigned();
    if (type == "number")
        return v.is_number();
    if (type == "boolean")
        return v.is_boolean();
    if (type == "object")
        return v.is_object();
    if (type == "array")
        return v.is_array();
    if (type == "null")
        return v.is_null();
    return true;  // unknown/compound type ("any", union via array) — accept
}

// Locate the tool definition for `name` and return its `parameters` schema.
// Returns an empty object if not found.
static json find_tool_schema(const json& tools, const std::string& name) {
    if (!tools.is_array())
        return json::object();
    for (const auto& t : tools) {
        if (!t.is_object())
            continue;
        // OpenAI shape: {"type":"function","function":{"name","parameters"}}
        if (t.contains("function") && t["function"].is_object()) {
            const auto& fn = t["function"];
            if (fn.value("name", "") == name)
                return fn.value("parameters", json::object());
        }
        // Bare shape: {"name","parameters"}
        if (t.value("name", "") == name)
            return t.value("parameters", json::object());
    }
    return json::object();
}

void validate_tool_call(ParsedToolCall& tc, const json& tools) {
    json schema = find_tool_schema(tools, tc.name);
    if (!schema.is_object() || schema.empty())
        return;  // no schema to validate against — leave as-is

    json args = json::parse(tc.arguments, nullptr, false);
    if (args.is_discarded() || !args.is_object()) {
        tc.valid = false;
        tc.error = "arguments are not a valid JSON object";
        return;
    }

    // Required properties must be present.
    if (schema.contains("required") && schema["required"].is_array()) {
        for (const auto& r : schema["required"]) {
            if (!r.is_string())
                continue;
            const std::string key = r.get<std::string>();
            if (!args.contains(key)) {
                tc.valid = false;
                tc.error = "missing required argument \"" + key + "\"";
                return;
            }
        }
    }

    // Top-level property types (best-effort).
    if (schema.contains("properties") && schema["properties"].is_object()) {
        const auto& props = schema["properties"];
        for (auto it = args.begin(); it != args.end(); ++it) {
            if (!props.contains(it.key()))
                continue;  // additional property — don't reject
            const auto& pschema = props[it.key()];
            if (pschema.is_object() && pschema.contains("type") && pschema["type"].is_string()) {
                if (!json_type_matches(it.value(), pschema["type"].get<std::string>())) {
                    tc.valid = false;
                    tc.error = "argument \"" + it.key() + "\" has wrong type (expected " +
                               pschema["type"].get<std::string>() + ")";
                    return;
                }
            }
        }
    }
}

std::string format_tool_response(imp::ChatTemplateFamily family, const json& msg) {
    // Tool responses arrive either as a plain string (OpenAI canonical) or as
    // a structured JSON object/array (some clients pass through tool output
    // verbatim). `msg.value("content", "")` silently returns "" for the
    // non-string case, dropping the entire payload — serialise it instead.
    std::string content;
    if (msg.contains("content") && !msg["content"].is_null()) {
        const auto& c = msg["content"];
        content = c.is_string() ? c.get<std::string>() : dump_safe(c);
    }

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
    // ChatML (Qwen3, Qwen3.6): the chat template wraps role=tool messages
    // with `<tool_response>` markers itself (see tool branch in
    // chat_template.jinja). Returning a pre-wrapped string here would nest
    // the markers (`<tool_response><tool_response>...</tool_response></tool_response>`)
    // and the model fails to recognise the result — silently degenerates to
    // its training prior (e.g. claiming the search target doesn't exist).
    return content;
}
