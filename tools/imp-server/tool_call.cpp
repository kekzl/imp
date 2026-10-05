#include "tool_call.h"
#include "tool_call_dialect.h"
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

    std::string prompt = tool_call_dialect(family).prompt(tools);

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

bool tool_choice_is_enforceable(imp::ChatTemplateFamily family, const json& tool_choice,
                                bool gemma_native_tool_call) {
    const ToolCallDialect& d = tool_call_dialect(family);
    if (tool_choice.is_object() && tool_choice.contains("function")) {
        if (tool_choice["function"].value("name", "").empty())
            return true;  // an object without a name forces nothing
        return d.json_envelope_open != nullptr ||
               (d.forced_envelope != nullptr && (d.native_token == nullptr || gemma_native_tool_call));
    }
    if (tool_choice.is_string() && tool_choice.get<std::string>() == "required")
        return d.json_envelope_open != nullptr;
    return true;
}

std::vector<std::pair<std::string, std::string>> collect_tool_constraint(imp::ChatTemplateFamily family,
                                                                         const json& tools,
                                                                         const json& tool_choice) {
    std::vector<std::pair<std::string, std::string>> out;
    if (tools.empty())
        return out;
    // Stage 1 (#1002): only a dialect with a JSON {"name","arguments"} envelope (ChatML
    // <tool_call>); the others keep the prompt-hint path.
    if (tool_call_dialect(family).json_envelope_open == nullptr)
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

ForcedToolEnvelope collect_forced_bare_args_tool(imp::ChatTemplateFamily family, const json& tools,
                                                 const json& tool_choice, bool gemma_native_tool_call) {
    // The envelope carries the name, so the body IS the arguments object and only a FORCED single
    // function maps onto the plain parameter schema. "required"/auto would need a name enum inside the
    // header (follow-up). Gemma-4 gets a JSON body: tool_call_gemma.cpp parses it next to its own syntax.
    const ToolCallDialect& d = tool_call_dialect(family);
    if (d.forced_envelope == nullptr || (d.native_token != nullptr && !gemma_native_tool_call) ||
        tools.empty())
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
        ForcedToolEnvelope out{forced, dump_safe(fn["parameters"]), "", ""};
        d.forced_envelope(forced, out.open, out.close);
        return out;
    }
    return {};
}

std::vector<std::pair<std::string, std::string>> collect_strict_tool_constraint(
    imp::ChatTemplateFamily family, const json& tools, const json& tool_choice) {
    std::vector<std::pair<std::string, std::string>> out;
    if (tools.empty())
        return out;
    // JSON envelope dialects only (as the forced path).
    if (tool_call_dialect(family).json_envelope_open == nullptr)
        return out;
    // Optional strict enforcement applies only when the model is FREE to choose (tool_choice
    // auto/absent); a forced function or "required" takes the forced-envelope path
    // (collect_tool_constraint) instead, and "none"/unknown suppress tools.
    if (tool_choice.is_object())
        return out;
    if (tool_choice.is_string() && tool_choice.get<std::string>() != "auto")
        return out;

    // `strict` is per-function (OpenAI API): a tool without enforcement, or with no properties to
    // enforce, still enters the name enum with a free-form schema, leaving only ITS arguments
    // unconstrained. Bailing out entirely on one loose tool used to disable constrained decoding for
    // the whole set (#1597: a schema-bound write_file next to a free-text bash tool). Free-form
    // schemas became representable in #1729.
    static constexpr const char* kFreeFormParams = R"({"type":"object","additionalProperties":true})";
    bool any_strict = false;
    for (const auto& tool : tools) {
        if (!tool.contains("function"))
            continue;
        const auto& fn = tool["function"];
        std::string name = fn.value("name", "");
        if (name.empty())
            return {};  // an unnameable tool cannot enter the enum
        const bool strict = fn.value("strict", false);
        const bool enforceable_params = fn.contains("parameters") && fn["parameters"].is_object() &&
                                        fn["parameters"].contains("properties") &&
                                        !fn["parameters"]["properties"].empty();
        any_strict = any_strict || strict;
        out.emplace_back(std::move(name), (strict && enforceable_params)
                                              ? dump_safe(fn["parameters"])
                                              : std::string(kFreeFormParams));
    }
    // No tool asked for enforcement: the prompt hint is what the caller wanted.
    if (!any_strict)
        return {};
    return out;
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

std::pair<std::string, std::vector<ParsedToolCall>> parse_tool_calls(
    imp::ChatTemplateFamily family, const std::string& text, std::atomic<int>& next_tool_call_id,
    const std::vector<std::string>& known_tool_names) {
    return tool_call_dialect(family).parse(text, next_tool_call_id, known_tool_names);
}

// ---------------------------------------------------------------------------
// Streaming tag scanner + body parser (used by StreamToolCallFilter).
// ---------------------------------------------------------------------------

ToolTagScan scan_tool_tag(const std::string& buf, imp::ChatTemplateFamily family) {
    return tool_call_dialect(family).scan(buf);
}

// ChatML JSON body {"name": ..., "arguments": {...}}: true on a non-empty name, false on any
// parse or type error (the caller then tries the Qwen3.6 XML layout).
static bool parse_chatml_json_call(const std::string& body, ParsedToolCall& tc) {
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
        return !tc.name.empty();
    } catch (...) {
        return false;
    }
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
    if (parse_chatml_json_call(body, tc))
        return true;
    // ... then Qwen3.6's <function=NAME><parameter=K>V</parameter> layout.
    if (body.find("<function=") != std::string::npos && parse_qwen36_xml_call(body, tc))
        return true;
    // ... then GLM-4.x's NAME<arg_key>K</arg_key><arg_value>V</arg_value>.
    return body.find("<arg_key>") != std::string::npos && parse_glm_arg_key_call(body, tc);
}

std::string reconstruct_tool_call_output(imp::ChatTemplateFamily family, const json& tool_calls,
                                         const std::string& content, bool xml) {
    std::string result;
    if (!content.empty() && content != "null") {
        result = content;
    }

    const ToolRenderCallFn render = tool_call_dialect(family).render_call;
    for (const auto& tc : tool_calls) {
        if (!tc.contains("function"))
            continue;
        const json& fn = tc["function"];
        std::string name = fn.contains("name") && fn["name"].is_string() ? fn["name"].get<std::string>() : "";
        // arguments is a JSON string per OpenAI; Ollama-style clients send the object itself, which
        // value(..., "{}") answered with a raw 400 type_error.302.
        std::string args = "{}";
        if (fn.contains("arguments") && fn["arguments"].is_string())
            args = fn["arguments"].get<std::string>();
        else if (fn.contains("arguments") && !fn["arguments"].is_null())
            args = dump_safe(fn["arguments"]);

        render(result, name, args, xml);
    }

    return result;
}

// ---------------------------------------------------------------------------
// Tool-call argument validation (self-contained; no engine constraint code).
// ---------------------------------------------------------------------------

// json_type_matches: checks only top-level scalar/container "type" kinds - enough to catch
// common hallucinations (string where a number is required, missing object) without a full validator.
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

void restore_raw_string_params(ParsedToolCall& tc, const json& tools) {
    if (tc.raw_params.empty())
        return;
    const json schema = find_tool_schema(tools, tc.name);
    if (!schema.is_object() || !schema.contains("properties") || !schema["properties"].is_object())
        return;
    json args = json::parse(tc.arguments, nullptr, false);
    if (!args.is_object())
        return;
    bool changed = false;
    for (const auto& [key, raw] : tc.raw_params) {
        const json& p = schema["properties"].value(key, json::object());
        if (!p.is_object() || !args.contains(key) || args[key].is_string())
            continue;
        const json types = p.contains("type") ? p["type"] : json();
        const auto allows = [&](const char* t) {
            return types == t ||
                   (types.is_array() && std::find(types.begin(), types.end(), t) != types.end());
        };
        if (!allows("string"))
            continue;
        bool other_ok = false;
        for (const char* t : {"integer", "number", "boolean", "null"})
            other_ok |= allows(t) && json_type_matches(args[key], t);
        if (!other_ok) {
            args[key] = raw;
            changed = true;
        }
    }
    if (changed)
        tc.arguments = dump_safe(args);
}

void validate_tool_call(ParsedToolCall& tc, const json& tools) {
    restore_raw_string_params(tc, tools);
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
    // Tool responses arrive as either a plain string (OpenAI canonical) or structured JSON;
    // msg.value("content","") silently drops the whole payload for the non-string case - serialize it
    // instead.
    std::string content;
    if (msg.contains("content") && !msg["content"].is_null()) {
        const auto& c = msg["content"];
        // An array of text parts is the OpenAI spec shape: the texts, not the serialized array.
        if (c.is_string())
            content = c.get<std::string>();
        else if (!join_text_parts(c, content))
            content = dump_safe(c);
    }

    // ChatML (Qwen3/Qwen3.6) templates wrap role=tool messages with <tool_response> themselves: a
    // pre-wrapped string nests the markers and the model misses the result (tool_response=nullptr).
    const ToolCallDialect& d = tool_call_dialect(family);
    return d.tool_response ? d.tool_response(msg, content) : content;
}
