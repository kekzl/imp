#include "responses.h"

#include <cstdio>
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

}  // namespace

std::string make_response_id(uint64_t counter) {
    char buf[48];
    std::snprintf(buf, sizeof(buf), "resp_imp%012llx", static_cast<unsigned long long>(counter));
    return buf;
}

std::string make_item_id(const char* prefix, uint64_t counter) {
    char buf[48];
    std::snprintf(buf, sizeof(buf), "%s_imp%012llx", prefix,
                  static_cast<unsigned long long>(counter));
    return buf;
}

json responses_to_openai_body(const json& rsp) {
    // Statelessness guards: imp keeps no response store, so continuation via
    // previous_response_id / retrieval via store=true cannot be honored.
    // Agentic clients (Codex CLI, Agents SDK) run with store=false and send
    // the full transcript as `input` each turn — exactly what works here.
    if (rsp.contains("previous_response_id") && !rsp["previous_response_id"].is_null())
        throw std::invalid_argument(
            "previous_response_id is not supported (imp-server is stateless — send the "
            "full transcript in `input`, e.g. store=false clients like Codex do)");
    if (rsp.value("store", false))
        throw std::invalid_argument(
            "store=true is not supported (imp-server keeps no response store); use "
            "store=false and resend the transcript in `input`");

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
            } else if (type == "function_call") {
                // A prior assistant tool call the client is replaying back.
                json tc = {{"id", item.value("call_id", item.value("id", ""))},
                           {"type", "function"},
                           {"function",
                            {{"name", item.value("name", "")},
                             {"arguments", item.value("arguments", "{}")}}}};
                messages.push_back({{"role", "assistant"},
                                    {"content", nullptr},
                                    {"tool_calls", json::array({tc})}});
            } else if (type == "function_call_output") {
                json out = item.value("output", json());
                std::string out_text = out.is_string() ? out.get<std::string>() : out.dump();
                messages.push_back({{"role", "tool"},
                                    {"tool_call_id", item.value("call_id", "")},
                                    {"content", out_text}});
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

    // Tools: Responses uses a FLAT function shape ({type, name, parameters});
    // chat/completions nests it under `function`.
    if (rsp.contains("tools") && rsp["tools"].is_array() && !rsp["tools"].empty()) {
        json tools = json::array();
        for (const auto& t : rsp["tools"]) {
            std::string ttype = t.value("type", "function");
            if (ttype != "function")
                throw std::invalid_argument("unsupported tool type: " + ttype +
                                            " (only function tools)");
            json fn = {{"name", t.value("name", "")}};
            if (t.contains("description"))
                fn["description"] = t["description"];
            if (t.contains("parameters"))
                fn["parameters"] = t["parameters"];
            if (t.contains("strict"))
                fn["strict"] = t["strict"];
            tools.push_back({{"type", "function"}, {"function", std::move(fn)}});
        }
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
        }
    }
    if (rsp.contains("parallel_tool_calls"))
        oai["parallel_tool_calls"] = rsp["parallel_tool_calls"];

    // text.format -> response_format.
    if (rsp.contains("text") && rsp["text"].is_object() && rsp["text"].contains("format")) {
        const json& fmt = rsp["text"]["format"];
        std::string ftype = fmt.value("type", "text");
        if (ftype == "json_object") {
            oai["response_format"] = {{"type", "json_object"}};
        } else if (ftype == "json_schema") {
            json js = {{"name", fmt.value("name", "response")}};
            if (fmt.contains("schema"))
                js["schema"] = fmt["schema"];
            if (fmt.contains("strict"))
                js["strict"] = fmt["strict"];
            oai["response_format"] = {{"type", "json_schema"}, {"json_schema", std::move(js)}};
        }
    }

    if (rsp.contains("temperature"))
        oai["temperature"] = rsp["temperature"];
    if (rsp.contains("top_p"))
        oai["top_p"] = rsp["top_p"];
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

    if (rsp.contains("stream"))
        oai["stream"] = rsp["stream"];
    return oai;
}

json openai_to_responses_response(const json& oai, const std::string& req_model,
                                  const std::string& response_id) {
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
                out["output"].push_back({{"type", "function_call"},
                                         {"id", make_item_id("fc", item_seq++)},
                                         {"call_id", tc.value("id", "")},
                                         {"name", fn.value("name", "")},
                                         {"arguments", fn.value("arguments", "{}")},
                                         {"status", "completed"}});
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
        out["usage"] = std::move(usage);
    }
    return out;
}

}  // namespace imp_server::responses
