#include "responses.h"
#include "sampling_fields.h"
#include "spec_usage_keys.h"

#include <algorithm>
#include <cstdio>
#include <random>
#include <stdexcept>

namespace imp_server::responses {

namespace {

// Responses message `content` -> chat `content`: the joined text when every part is text, else
// chat parts in order (text, image_url). input_image carries a URL or data: URI in `image_url`;
// file_id and input_file need a file store this server does not have.
json content_to_chat(const json& content) {
    if (content.is_string())
        return content;
    std::string text;
    json parts = json::array();
    bool has_image = false;
    if (content.is_array()) {
        for (const auto& part : content) {
            if (!part.is_object())
                continue;
            std::string type = part.value("type", "");
            std::string piece;
            if (type == "input_text" || type == "output_text" || type == "text" ||
                type == "summary_text") {
                piece = part.value("text", "");
            } else if (type == "refusal") {
                piece = part.value("refusal", "");
            } else if (type == "input_image") {
                const json url = part.value("image_url", json());
                if (!url.is_string() || url.get<std::string>().empty())
                    throw std::invalid_argument(
                        "input_image needs image_url (a URL or data: URI); file_id is not supported");
                json image = {{"url", url}};
                if (part.contains("detail"))
                    image["detail"] = part["detail"];
                parts.push_back({{"type", "image_url"}, {"image_url", std::move(image)}});
                has_image = true;
                continue;
            } else {
                throw std::invalid_argument("unsupported input content part type: " + type);
            }
            text += piece;
            parts.push_back({{"type", "text"}, {"text", std::move(piece)}});
        }
    }
    return has_image ? parts : json(text);
}

// Chat function names of the two client tools with their own item types.
constexpr const char* kLocalShellTool = "shell";
constexpr const char* kToolSearchTool = "tool_search";

// Codex 0.166.0 hides MCP and agent tools behind {"type":"tool_search","execution":"client"}.
bool is_client_tool_search(const json& t) {
    const auto it = t.find("execution");
    return t.value("type", "") == "tool_search" && it != t.end() && it->is_string() && *it == "client";
}

// Server-side tools the server cannot execute: offered, never demanded, so they are dropped
// (Codex 0.163.0 sends web_search with every request).
bool is_hosted_tool(const std::string& type) {
    static const char* const kHosted[] = {"web_search",           "web_search_preview",
                                          "tool_search",          "file_search",
                                          "code_interpreter",     "image_generation",
                                          "computer_use_preview", "mcp"};
    for (const char* h : kHosted)
        if (type == h)
            return true;
    return false;
}

// A custom (freeform) tool becomes a function with one string parameter `input`; its grammar
// goes into the parameter description.
json custom_tool_parameters(const json& t) {
    std::string desc = "Raw tool input, passed verbatim";
    const json fmt = t.value("format", json::object());
    if (fmt.is_object() && fmt.value("type", "") == "grammar")
        desc += ". It must match this " + fmt.value("syntax", std::string("grammar")) + " grammar:\n" +
                fmt.value("definition", std::string());
    return {{"type", "object"},
            {"properties", {{"input", {{"type", "string"}, {"description", std::move(desc)}}}}},
            {"required", json::array({"input"})}};
}

json local_shell_tool() {
    json props = {{"command",
                   {{"type", "array"},
                    {"items", {{"type", "string"}}},
                    {"description", "The command and its arguments, as for execvp"}}},
                  {"workdir", {{"type", "string"}, {"description", "Working directory"}}},
                  {"timeout_ms", {{"type", "integer"}, {"description", "Timeout in milliseconds"}}}};
    json fn = {
        {"name", kLocalShellTool},
        {"description", "Runs a shell command and returns its output."},
        {"parameters",
         {{"type", "object"}, {"properties", std::move(props)}, {"required", json::array({"command"})}}}};
    return {{"type", "function"}, {"function", std::move(fn)}};
}

// `shell` function arguments -> local_shell_call action. A bare string runs through bash.
json local_shell_action(const std::string& arguments) {
    json action = {{"type", "exec"}, {"command", json::array()}, {"env", json::object()}};
    const json a = json::parse(arguments, nullptr, /*allow_exceptions=*/false);
    if (!a.is_object())
        return action;
    const json command = a.value("command", json());
    if (command.is_array())
        action["command"] = command;
    else if (command.is_string())
        action["command"] = json::array({"bash", "-lc", command});
    if (a.contains("workdir") && a["workdir"].is_string())
        action["working_directory"] = a["workdir"];
    if (a.contains("timeout_ms") && a["timeout_ms"].is_number())
        action["timeout_ms"] = a["timeout_ms"];
    return action;
}

// Inverse for a replayed local_shell_call.
std::string local_shell_arguments(const json& action) {
    json args = {{"command", action.value("command", json::array())}};
    if (action.contains("working_directory") && action["working_directory"].is_string())
        args["workdir"] = action["working_directory"];
    if (action.contains("timeout_ms") && action["timeout_ms"].is_number())
        args["timeout_ms"] = action["timeout_ms"];
    return args.dump();
}

// Responses tools (flat function, custom, namespace) -> chat tools (nested function).
void append_chat_tools(const json& list, json& tools) {
    for (const auto& t : list) {
        if (!t.is_object())
            throw std::invalid_argument("a tools entry must be an object");
        const std::string ttype = t.value("type", "function");
        if (ttype == "namespace") {
            if (t.contains("tools") && t["tools"].is_array())
                append_chat_tools(t["tools"], tools);
            continue;
        }
        if (ttype == "local_shell") {
            tools.push_back(local_shell_tool());
            continue;
        }
        if (is_client_tool_search(t)) {
            json fn = {{"name", kToolSearchTool}};
            if (t.contains("description"))
                fn["description"] = t["description"];
            fn["parameters"] = t.value("parameters", json{{"type", "object"},
                                                          {"properties", {{"query", {{"type", "string"}}}}},
                                                          {"required", json::array({"query"})}});
            tools.push_back({{"type", "function"}, {"function", std::move(fn)}});
            continue;
        }
        if (is_hosted_tool(ttype))
            continue;
        if (ttype != "function" && ttype != "custom")
            throw std::invalid_argument("unsupported tool type: " + ttype +
                                        " (function, custom, namespace, local_shell and client tool_search; "
                                        "hosted tools are ignored)");
        json fn = {{"name", t.value("name", "")}};
        if (t.contains("description"))
            fn["description"] = t["description"];
        if (ttype == "custom")
            fn["parameters"] = custom_tool_parameters(t);
        else if (t.contains("parameters"))
            fn["parameters"] = t["parameters"];
        if (ttype == "function" && t.contains("strict"))
            fn["strict"] = t["strict"];
        tools.push_back({{"type", "function"}, {"function", std::move(fn)}});
    }
}

// A replayed call item's chat function name and arguments. False for any other item type.
bool replayed_call(const std::string& type, const json& item, std::string& name, std::string& args) {
    if (type == "function_call") {
        name = item.value("name", "");
        args = item.value("arguments", std::string("{}"));
    } else if (type == "custom_tool_call") {
        name = item.value("name", "");
        args = json{{"input", item.value("input", std::string())}}.dump();
    } else if (type == "local_shell_call") {
        name = kLocalShellTool;
        args = local_shell_arguments(item.value("action", json::object()));
    } else if (type == "tool_search_call") {
        name = kToolSearchTool;
        const json a = item.value("arguments", json::object());
        args = a.is_string() ? a.get<std::string>() : a.dump();
    } else {
        return false;
    }
    return true;
}

// What the model reads as the result of tool_search; the tools themselves join the request's list.
std::string tool_search_result_text(const json& item) {
    json found = json::array();
    if (item.contains("tools") && item["tools"].is_array())
        append_chat_tools(item["tools"], found);
    if (found.empty())
        return "No matching tools found.";
    std::string text = "These tools can be called now:";
    for (const auto& t : found)
        text += " " + t["function"].value("name", "");
    return text;
}

// Replayed tool items -> chat messages: every call item becomes an assistant tool call, its output
// a tool message. False for any other item type.
bool tool_item_to_message(const std::string& type, const json& item, json& messages) {
    std::string name;
    std::string args;
    if (replayed_call(type, item, name, args)) {
        json tc = {{"id", item.value("call_id", item.value("id", ""))},
                   {"type", "function"},
                   {"function", {{"name", name}, {"arguments", args}}}};
        // One response's preamble and calls are one assistant turn. As separate turns the template
        // closed the preamble with an end marker, and Qwen3.8-Flash-Next then ended 15 of 40 replies
        // after its preamble, without the call (#2807).
        if (!messages.empty() && messages.back().value("role", "") == "assistant") {
            json& prev = messages.back();
            if (!prev.contains("tool_calls"))
                prev["tool_calls"] = json::array();
            prev["tool_calls"].push_back(std::move(tc));
            return true;
        }
        messages.push_back({{"role", "assistant"}, {"content", nullptr}, {"tool_calls", json::array({tc})}});
        return true;
    }
    if (type == "function_call_output" || type == "custom_tool_call_output" ||
        type == "local_shell_call_output") {
        json out = item.value("output", json());
        std::string out_text = out.is_string() ? out.get<std::string>() : out.dump();
        // local_shell_call_output names its call by `id` in the API reference.
        messages.push_back({{"role", "tool"},
                            {"tool_call_id", item.value("call_id", item.value("id", ""))},
                            {"content", out_text}});
        return true;
    }
    if (type == "tool_search_output") {
        messages.push_back({{"role", "tool"},
                            {"tool_call_id", item.value("call_id", "")},
                            {"content", tool_search_result_text(item)}});
        return true;
    }
    return false;
}

// Tools a replayed tool_search_output loaded are callable from then on: the client keeps them out
// of `tools`, so they are added here, once each.
void append_searched_tools(const json& input, json& tools) {
    if (!input.is_array())
        return;
    std::set<std::string> seen;
    for (const auto& t : tools)
        seen.insert(t["function"].value("name", ""));
    for (const auto& item : input) {
        if (!item.is_object() || item.value("type", "") != "tool_search_output" || !item.contains("tools") ||
            !item["tools"].is_array())
            continue;
        json found = json::array();
        append_chat_tools(item["tools"], found);
        for (auto& t : found)
            if (seen.insert(t["function"].value("name", "")).second)
                tools.push_back(std::move(t));
    }
}

// '-' as '_': the model spells MCP server names both ways (#3056).
std::string name_key(std::string name) {
    std::replace(name.begin(), name.end(), '-', '_');
    return name;
}

void collect_tool_kinds(const json& list, const std::string& ns, ToolKinds& kinds) {
    for (const auto& t : list) {
        if (!t.is_object())
            continue;
        const std::string ttype = t.value("type", "function");
        if (ttype == "namespace" && t.contains("tools") && t["tools"].is_array()) {
            collect_tool_kinds(t["tools"], t.value("name", ""), kinds);
            continue;
        }
        kinds.local_shell = kinds.local_shell || ttype == "local_shell";
        kinds.tool_search = kinds.tool_search || is_client_tool_search(t);
        const std::string name = t.value("name", "");
        if (ttype == "custom")
            kinds.custom.insert(name);
        if (!ns.empty() && (ttype == "custom" || ttype == "function")) {
            kinds.ns[name] = ns;
            kinds.qualified[name_key(ns + "__" + name)] = name;
            kinds.qualified[name_key(ns + "_" + name)] = name;
        }
    }
}

// Every system message (instructions, system and developer items) joined into one leading system
// message, in order: Qwen templates raise on a system message past index 0 and the renderer then
// drops it. Codex 0.163.0 sends two developer items (base instructions, sandbox permissions).
json hoist_system_messages(json messages) {
    std::string system;
    json rest = json::array();
    for (auto& m : messages) {
        if (m.value("role", "") != "system") {
            rest.push_back(std::move(m));
            continue;
        }
        if (!m["content"].is_string())
            throw std::invalid_argument("system and developer messages take text parts only");
        if (!system.empty())
            system += "\n\n";
        system += m["content"].get<std::string>();
    }
    if (system.empty())
        return rest;
    json out = json::array({{{"role", "system"}, {"content", std::move(system)}}});
    for (auto& m : rest)
        out.push_back(std::move(m));
    return out;
}

}  // namespace

ToolKinds tool_kinds(const json& rsp) {
    ToolKinds kinds;
    if (rsp.contains("tools") && rsp["tools"].is_array())
        collect_tool_kinds(rsp["tools"], "", kinds);
    if (rsp.contains("input") && rsp["input"].is_array())
        for (const auto& item : rsp["input"])
            if (item.is_object() && item.value("type", "") == "tool_search_output" &&
                item.contains("tools") && item["tools"].is_array())
                collect_tool_kinds(item["tools"], "", kinds);
    return kinds;
}

std::string ToolKinds::resolve(const std::string& name) const {
    const auto it = qualified.find(name_key(name));
    return it == qualified.end() ? name : it->second;
}

CallShape ToolKinds::shape(const std::string& name) const {
    const std::string n = resolve(name);
    if (local_shell && n == kLocalShellTool)
        return CallShape::kLocalShell;
    if (tool_search && n == kToolSearchTool)
        return CallShape::kToolSearch;
    return custom.count(n) != 0 ? CallShape::kCustom : CallShape::kFunction;
}

std::string custom_tool_input(const std::string& arguments) {
    const json a = json::parse(arguments, nullptr, /*allow_exceptions=*/false);
    if (a.is_object() && a.contains("input") && a["input"].is_string())
        return a["input"].get<std::string>();
    return arguments;  // the model ignored the wrapper: pass its text through
}

json tool_call_item(const ToolKinds& kinds, const std::string& item_id, const std::string& call_id,
                    const std::string& name, const std::string& arguments, const char* status) {
    // Codex answers "unsupported call" to <ns>__<name> without `namespace` (0.166.0, #3056).
    const std::string tool = kinds.resolve(name);
    json item = {{"id", item_id}, {"call_id", call_id}, {"status", status}};
    switch (kinds.shape(tool)) {
        case CallShape::kCustom:
            item["type"] = "custom_tool_call";
            item["name"] = tool;
            item["input"] = custom_tool_input(arguments);
            break;
        case CallShape::kLocalShell:
            item["type"] = "local_shell_call";
            item["action"] = local_shell_action(arguments);
            break;
        case CallShape::kToolSearch: {
            item["type"] = "tool_search_call";
            item["execution"] = "client";
            const json a = json::parse(arguments, nullptr, /*allow_exceptions=*/false);
            item["arguments"] = a.is_object() ? a : json::object();
            break;
        }
        case CallShape::kFunction:
            item["type"] = "function_call";
            item["name"] = tool;
            item["arguments"] = arguments;
            break;
    }
    if (const auto it = kinds.ns.find(tool); it != kinds.ns.end())
        item["namespace"] = it->second;
    return item;
}

std::string make_response_id(uint64_t counter) {
    // random_device (getrandom on Linux), not a seeded PRNG whose state other ids would reveal.
    std::random_device rd;
    const uint64_t rnd = (static_cast<uint64_t>(rd()) << 32) ^ rd();
    char buf[64];
    std::snprintf(buf, sizeof(buf), "resp_imp%012llx%016llx", static_cast<unsigned long long>(counter),
                  static_cast<unsigned long long>(rnd));
    return buf;
}

std::string make_item_id(const char* prefix, uint64_t counter) {
    char buf[48];
    std::snprintf(buf, sizeof(buf), "%s_imp%012llx", prefix,
                  static_cast<unsigned long long>(counter));
    return buf;
}

json responses_to_openai_body(const json& rsp) {
    // `store` is the handler's (responses_store.h). previous_response_id must already be resolved
    // into `input` by continue_conversation(); reaching here unresolved is a handler bug.
    if (rsp.contains("previous_response_id") && !rsp["previous_response_id"].is_null())
        throw std::invalid_argument("previous_response_id was not resolved against the response store");

    json oai;
    if (rsp.contains("model"))
        oai["model"] = rsp["model"];

    json messages = json::array();
    if (rsp.contains("instructions") && rsp["instructions"].is_string())
        messages.push_back({{"role", "system"}, {"content", rsp["instructions"]}});

    const json& input = rsp.contains("input") ? rsp["input"] : json();
    if (input.is_string()) {
        messages.push_back({{"role", "user"}, {"content", input.get<std::string>()}});
    } else if (input.is_array()) {
        for (const auto& item : input) {
            if (!item.is_object())
                continue;
            // Plain {role, content} entries (the SDKs' EasyInputMessage) have
            // no `type`; typed items say type:"message".
            std::string type = item.value("type", item.contains("role") ? "message" : "");
            if (type == "message") {
                std::string role = item.value("role", "user");
                if (role == "developer")
                    role = "system";  // Responses' developer role ≈ system
                messages.push_back(
                    {{"role", role}, {"content", content_to_chat(item.value("content", json()))}});
            } else if (tool_item_to_message(type, item, messages)) {
                // A prior tool call or its output the client is replaying back.
                continue;
            } else if (type == "reasoning") {
                // Replayed reasoning items carry no information the chat path
                // can use — skip (the SDKs round-trip them opaquely).
                continue;
            } else if (type == "item_reference") {
                throw std::invalid_argument(
                    "item_reference input items are not supported (stateless server)");
            } else {
                throw std::invalid_argument("unsupported input item type: " + type);
            }
        }
    }
    oai["messages"] = hoist_system_messages(std::move(messages));
    // Assistant prefill (imp extension, the chat field): continue a trailing assistant input item.
    if (rsp.contains("continue_final_message"))
        oai["continue_final_message"] = rsp["continue_final_message"];

    // Tools: Responses uses a FLAT function shape ({type, name, parameters});
    // chat/completions nests it under `function`.
    json tools = json::array();
    if (rsp.contains("tools") && rsp["tools"].is_array())
        append_chat_tools(rsp["tools"], tools);
    append_searched_tools(input, tools);
    if (!tools.empty())
        oai["tools"] = std::move(tools);
    if (rsp.contains("tool_choice")) {
        const json& tc = rsp["tool_choice"];
        if (tc.is_string()) {
            oai["tool_choice"] = tc;
        } else if (tc.is_object() && tc.value("type", "") == "function") {
            // Flat {type:"function", name} -> nested chat shape.
            oai["tool_choice"] = {{"type", "function"},
                                  {"function", {{"name", tc.value("name", "")}}}};
        } else {
            // tool_choice {"type":"allowed_tools","mode":"required"} (the Agents SDK shape) fell through
            // both transform arms writing nothing, so the chat parser defaulted to "auto" and a caller
            // demanding a call got a fluent 200 with none - validate_tool_choice runs post-transform and
            // never saw the original field.
            const std::string tct = tc.is_object() ? tc.value("type", "") : "";
            throw std::invalid_argument("unsupported tool_choice " +
                                        (tct.empty() ? std::string("shape") : "type: \"" + tct + "\"") +
                                        " (a string, or {\"type\":\"function\",\"name\":...})");
        }
    }
    if (rsp.contains("parallel_tool_calls"))
        oai["parallel_tool_calls"] = rsp["parallel_tool_calls"];

    // text.format -> response_format.
    if (rsp.contains("text") && rsp["text"].is_object() && rsp["text"].contains("format")) {
        const json& fmt = rsp["text"]["format"];
        std::string ftype = fmt.value("type", "text");
        if (ftype == "text") {
            // The Responses default and an explicit "text" both mean
            // unconstrained; chat/completions expresses that by having no
            // response_format at all.
        } else if (ftype == "json_object") {
            oai["response_format"] = {{"type", "json_object"}};
        } else if (ftype == "json_schema") {
            json js = {{"name", fmt.value("name", "response")}};
            if (fmt.contains("schema"))
                js["schema"] = fmt["schema"];
            if (fmt.contains("strict"))
                js["strict"] = fmt["strict"];
            oai["response_format"] = {{"type", "json_schema"}, {"json_schema", std::move(js)}};
        } else if (ftype == "regex" || ftype == "grammar") {
            // response_format/grammar are imp extensions that already work on /v1/chat/completions; this
            // transform must pass them through whole (chat parser owns validation) or the same request is
            // constrained on one endpoint and free text on the other.
            oai["response_format"] = fmt;
        } else {
            // Anything else wrote no response_format, so the chat parser's
            // "unknown response_format.type" 400 never fired and the reply came
            // back unconstrained at 200. Same shape as tool_choice above.
            throw std::invalid_argument("unknown \"text.format.type\": \"" + ftype +
                                        "\" (known: text, json_object, json_schema, regex, grammar)");
        }
    }

    // temperature, top_p, priority, stream, speculative: kSamplingKeys rows with kViaResponses.
    pass_sampling_keys(rsp, kViaResponses, oai);
    if (rsp.contains("max_output_tokens"))
        oai["max_tokens"] = rsp["max_output_tokens"];

    // reasoning.effort -> think budget (fraction of max_tokens for the think
    // phase; see --think-budget). minimal/low keep answers snappy.
    if (rsp.contains("reasoning") && rsp["reasoning"].is_object()) {
        std::string effort = rsp["reasoning"].value("effort", "");
        if (effort == "none" || effort == "minimal")
            oai["think_budget"] = 0.0;
        else if (effort == "low")
            oai["think_budget"] = 0.25;
        else if (effort == "medium")
            oai["think_budget"] = 0.5;
        else if (effort == "high")
            oai["think_budget"] = 0.8;
    }

    return oai;
}

json normalize_input_items(const json& input) {
    if (input.is_string())
        return json::array({{{"role", "user"}, {"content", input}}});
    if (input.is_array())
        return input;
    return json::array();
}

json continue_conversation(const json& prev_input_items, const json& prev_output_items,
                           const json& new_input) {
    json items = prev_input_items.is_array() ? prev_input_items : json::array();
    if (prev_output_items.is_array())
        items.insert(items.end(), prev_output_items.begin(), prev_output_items.end());
    json tail = normalize_input_items(new_input);
    items.insert(items.end(), tail.begin(), tail.end());
    return items;
}

json openai_to_responses_response(const json& oai, const std::string& req_model,
                                  const std::string& response_id, const ToolKinds& kinds) {
    json out = {{"id", response_id},
                {"object", "response"},
                {"created_at", oai.value("created", 0)},
                {"model", req_model.empty() ? oai.value("model", "") : req_model},
                {"status", "completed"},
                {"error", nullptr},
                {"incomplete_details", nullptr},
                {"output", json::array()},
                {"parallel_tool_calls", true},
                {"tool_choice", "auto"},
                {"tools", json::array()}};

    uint64_t item_seq = 0;
    std::string finish;
    if (oai.contains("choices") && !oai["choices"].empty()) {
        const json& choice = oai["choices"][0];
        finish = choice.value("finish_reason", "");
        const json& msg = choice.value("message", json::object());

        // Reasoning first (mirrors the OpenAI item order): raw CoT goes out
        // as a reasoning item with a single summary_text part.
        std::string reasoning = msg.value("reasoning_content", "");
        if (!reasoning.empty()) {
            out["output"].push_back(
                {{"type", "reasoning"},
                 {"id", make_item_id("rs", item_seq++)},
                 {"summary", json::array({{{"type", "summary_text"}, {"text", reasoning}}})}});
        }

        if (msg.contains("content") && msg["content"].is_string() &&
            !msg["content"].get<std::string>().empty()) {
            out["output"].push_back(
                {{"type", "message"},
                 {"id", make_item_id("msg", item_seq++)},
                 {"status", "completed"},
                 {"role", "assistant"},
                 {"content", json::array({{{"type", "output_text"},
                                           {"text", msg["content"]},
                                           {"annotations", json::array()}}})}});
        }

        if (msg.contains("tool_calls") && msg["tool_calls"].is_array()) {
            for (const auto& tc : msg["tool_calls"]) {
                const json& fn = tc.value("function", json::object());
                out["output"].push_back(tool_call_item(kinds, make_item_id("fc", item_seq++),
                                                       tc.value("id", ""), fn.value("name", ""),
                                                       fn.value("arguments", "{}"), "completed"));
            }
        }
    }

    if (finish == "length") {
        out["status"] = "incomplete";
        out["incomplete_details"] = {{"reason", "max_output_tokens"}};
    }

    if (oai.contains("usage")) {
        const json& u = oai["usage"];
        json usage = {{"input_tokens", u.value("prompt_tokens", 0)},
                      {"output_tokens", u.value("completion_tokens", 0)},
                      {"total_tokens", u.value("total_tokens", 0)}};
        int cached = 0;
        int evicted = 0;
        if (u.contains("prompt_tokens_details")) {
            cached = u["prompt_tokens_details"].value("cached_tokens", 0);
            evicted = u["prompt_tokens_details"].value("evicted_tokens", 0);
        }
        usage["input_tokens_details"] = {{"cached_tokens", cached}};
        // imp extension, forwarded rather than dropped: context this request
        // lost to StreamingLLM eviction. Present only when it happened, so a
        // client that never hits the KV ceiling never sees the key.
        if (evicted > 0)
            usage["input_tokens_details"]["evicted_tokens"] = evicted;
        int reasoning_toks = 0;
        if (u.contains("completion_tokens_details"))
            reasoning_toks = u["completion_tokens_details"].value("reasoning_tokens", 0);
        usage["output_tokens_details"] = {{"reasoning_tokens", reasoning_toks}};
        // imp extension (C-6): per-request speculation counters, same keys as
        // the chat shape, from the shared table (spec_usage_keys.h).
        if (u.contains("completion_tokens_details"))
            imp_server::copy_spec_usage_keys(u["completion_tokens_details"],
                                             usage["output_tokens_details"]);
        out["usage"] = std::move(usage);
    }
    return out;
}

}  // namespace imp_server::responses
