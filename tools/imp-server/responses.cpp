#include "responses.h"
#include "sampling_fields.h"
#include "spec_usage_keys.h"

#include <cstdio>
#include <random>
#include <stdexcept>

namespace imp_server::responses {

namespace {

// Extract the plain text of a Responses message `content` field: either a
// bare string or an array of {type: input_text|output_text|text, text} parts.
// Non-text parts (input_image, input_file) are unsupported in v1.
std::string content_to_text(const json& content) {
    if (content.is_string())
        return content.get<std::string>();
    std::string out;
    if (content.is_array()) {
        for (const auto& part : content) {
            if (!part.is_object())
                continue;
            std::string type = part.value("type", "");
            if (type == "input_text" || type == "output_text" || type == "text" ||
                type == "summary_text") {
                out += part.value("text", "");
            } else if (type == "refusal") {
                out += part.value("refusal", "");
            } else {
                throw std::invalid_argument("unsupported input content part type: " + type);
            }
        }
    }
    return out;
}

// Server-side tools the server cannot execute: offered, never demanded, so they are dropped
// (Codex 0.163.0 sends web_search and tool_search with every request).
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
        if (is_hosted_tool(ttype))
            continue;
        if (ttype != "function" && ttype != "custom")
            throw std::invalid_argument("unsupported tool type: " + ttype +
                                        " (function, custom and namespace; hosted tools are ignored)");
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

// Replayed tool items -> chat messages: function_call / custom_tool_call become an assistant
// tool call, *_output a tool message. False for any other item type.
bool tool_item_to_message(const std::string& type, const json& item, json& messages) {
    if (type == "function_call" || type == "custom_tool_call") {
        const std::string args = type == "function_call"
                                     ? item.value("arguments", std::string("{}"))
                                     : json{{"input", item.value("input", std::string())}}.dump();
        json tc = {{"id", item.value("call_id", item.value("id", ""))},
                   {"type", "function"},
                   {"function", {{"name", item.value("name", "")}, {"arguments", args}}}};
        messages.push_back({{"role", "assistant"}, {"content", nullptr}, {"tool_calls", json::array({tc})}});
        return true;
    }
    if (type == "function_call_output" || type == "custom_tool_call_output") {
        json out = item.value("output", json());
        std::string out_text = out.is_string() ? out.get<std::string>() : out.dump();
        messages.push_back(
            {{"role", "tool"}, {"tool_call_id", item.value("call_id", "")}, {"content", out_text}});
        return true;
    }
    return false;
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
        const std::string name = t.value("name", "");
        if (ttype == "custom")
            kinds.custom.insert(name);
        if (!ns.empty() && (ttype == "custom" || ttype == "function"))
            kinds.ns[name] = ns;
    }
}

}  // namespace

ToolKinds tool_kinds(const json& rsp) {
    ToolKinds kinds;
    if (rsp.contains("tools") && rsp["tools"].is_array())
        collect_tool_kinds(rsp["tools"], "", kinds);
    return kinds;
}

std::string custom_tool_input(const std::string& arguments) {
    const json a = json::parse(arguments, nullptr, /*allow_exceptions=*/false);
    if (a.is_object() && a.contains("input") && a["input"].is_string())
        return a["input"].get<std::string>();
    return arguments;  // the model ignored the wrapper: pass its text through
}

json tool_call_item(const ToolKinds& kinds, const std::string& item_id, const std::string& call_id,
                    const std::string& name, const std::string& arguments, const char* status) {
    json item = kinds.custom.count(name) != 0 ? json{{"type", "custom_tool_call"},
                                                     {"id", item_id},
                                                     {"call_id", call_id},
                                                     {"name", name},
                                                     {"input", custom_tool_input(arguments)}}
                                              : json{{"type", "function_call"},
                                                     {"id", item_id},
                                                     {"call_id", call_id},
                                                     {"name", name},
                                                     {"arguments", arguments}};
    item["status"] = status;
    if (const auto it = kinds.ns.find(name); it != kinds.ns.end())
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
                messages.push_back({{"role", role}, {"content", content_to_text(item.value("content", json()))}});
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
    oai["messages"] = std::move(messages);
    // Assistant prefill (imp extension, the chat field): continue a trailing assistant input item.
    if (rsp.contains("continue_final_message"))
        oai["continue_final_message"] = rsp["continue_final_message"];

    // Tools: Responses uses a FLAT function shape ({type, name, parameters});
    // chat/completions nests it under `function`.
    if (rsp.contains("tools") && rsp["tools"].is_array()) {
        json tools = json::array();
        append_chat_tools(rsp["tools"], tools);
        if (!tools.empty())
            oai["tools"] = std::move(tools);
    }
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
