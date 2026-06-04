// Anthropic Messages API (/v1/messages) transforms.
// See anthropic.h for the public surface.

#include "anthropic.h"

#include <atomic>
#include <cstdio>

namespace imp_server::anthropic {

// ---------------------------------------------------------------------------
// Request: Anthropic Messages body -> OpenAI chat.completion body
// ---------------------------------------------------------------------------

namespace {

// Turn an Anthropic message `content` field (string | [content_block]) into
// the OpenAI content field (string | [OpenAI_part]).
json convert_message_content(const json& anth_content) {
    if (anth_content.is_string())
        return anth_content;
    if (!anth_content.is_array())
        return "";

    // Special case: a single text block collapses to a plain string, which
    // is what the OpenAI code path prefers.
    if (anth_content.size() == 1 && anth_content[0].is_object() &&
        anth_content[0].value("type", "") == "text") {
        return anth_content[0].value("text", "");
    }

    json oai_parts = json::array();
    for (const auto& block : anth_content) {
        std::string type = block.value("type", "");
        if (type == "text") {
            oai_parts.push_back({{"type", "text"}, {"text", block.value("text", "")}});
        } else if (type == "image" && block.contains("source")) {
            // Anthropic image blocks:
            //   {"type":"image","source":{"type":"base64","media_type":"image/png","data":"..."}}
            //   {"type":"image","source":{"type":"url","url":"https://..."}}
            const auto& src = block["source"];
            std::string src_type = src.value("type", "");
            if (src_type == "base64") {
                std::string mime = src.value("media_type", "image/png");
                std::string data = src.value("data", "");
                oai_parts.push_back(
                    {{"type", "image_url"}, {"image_url", {{"url", "data:" + mime + ";base64," + data}}}});
            } else if (src_type == "url") {
                oai_parts.push_back({{"type", "image_url"}, {"image_url", {{"url", src.value("url", "")}}}});
            }
        } else if (type == "tool_use") {
            // Assistant message previously produced a tool_use block. The
            // OpenAI-equivalent is a tool_calls entry on the assistant turn,
            // handled one level up in the caller.
        } else if (type == "tool_result") {
            // See caller — tool_result blocks become an OpenAI `role: "tool"`
            // message, handled one level up.
        }
    }
    return oai_parts;
}

// Expand an Anthropic assistant turn that contains tool_use blocks into the
// OpenAI form (content string + tool_calls array) and push into `out`.
void push_assistant_turn(json& out, const json& anth_msg) {
    json oai_msg = {{"role", "assistant"}};
    std::string text_accum;
    json tool_calls = json::array();

    const auto& content = anth_msg["content"];
    if (content.is_string()) {
        oai_msg["content"] = content;
        out.push_back(oai_msg);
        return;
    }
    if (!content.is_array()) {
        oai_msg["content"] = "";
        out.push_back(oai_msg);
        return;
    }

    for (const auto& block : content) {
        std::string type = block.value("type", "");
        if (type == "text") {
            if (!text_accum.empty())
                text_accum += "\n";
            text_accum += block.value("text", "");
        } else if (type == "tool_use") {
            std::string id = block.value("id", "");
            std::string name = block.value("name", "");
            json input = block.value("input", json::object());
            tool_calls.push_back({
                {"id", id},
                {"type", "function"},
                {"function",
                 {
                     {"name", name},
                     {"arguments", input.dump()},
                 }},
            });
        } else if (type == "thinking") {
            // Map Anthropic thinking blocks back onto OpenAI's
            // reasoning_content so a prior assistant turn's chain of
            // thought round-trips through the OpenAI code path.
            std::string th = block.value("thinking", "");
            if (!th.empty()) {
                if (oai_msg.contains("reasoning_content"))
                    oai_msg["reasoning_content"] =
                        oai_msg["reasoning_content"].get<std::string>() + "\n" + th;
                else
                    oai_msg["reasoning_content"] = th;
            }
        }
    }
    oai_msg["content"] = text_accum.empty() ? json(nullptr) : json(text_accum);
    if (!tool_calls.empty())
        oai_msg["tool_calls"] = std::move(tool_calls);
    out.push_back(std::move(oai_msg));
}

// Expand an Anthropic user turn whose content has tool_result blocks into
// the OpenAI form (role: "tool" messages + optional regular user message).
void push_user_turn(json& out, const json& anth_msg) {
    const auto& content = anth_msg["content"];
    if (content.is_string()) {
        out.push_back({{"role", "user"}, {"content", content}});
        return;
    }
    if (!content.is_array()) {
        out.push_back({{"role", "user"}, {"content", ""}});
        return;
    }

    // Partition tool_result blocks from regular (text/image) blocks. Tool
    // results become OpenAI role="tool" messages; the rest stays on a user
    // turn.
    std::vector<json> tool_results;
    json other_parts = json::array();
    for (const auto& block : content) {
        std::string type = block.value("type", "");
        if (type == "tool_result") {
            // Anthropic tool_result.content can be string OR array of blocks
            // (mostly text). Collapse to a string for the OpenAI tool turn.
            std::string body;
            if (block.contains("content")) {
                const auto& c = block["content"];
                if (c.is_string())
                    body = c.get<std::string>();
                else if (c.is_array()) {
                    for (const auto& p : c) {
                        if (p.is_object() && p.value("type", "") == "text") {
                            if (!body.empty())
                                body += "\n";
                            body += p.value("text", "");
                        }
                    }
                }
            }
            tool_results.push_back({
                {"role", "tool"},
                {"tool_call_id", block.value("tool_use_id", "")},
                {"content", body},
            });
        } else if (type == "text") {
            other_parts.push_back({{"type", "text"}, {"text", block.value("text", "")}});
        } else if (type == "image" && block.contains("source")) {
            // Reuse the message-content converter
            json tmp = json::array({block});
            auto converted = convert_message_content(tmp);
            if (converted.is_array()) {
                for (const auto& p : converted)
                    other_parts.push_back(p);
            }
        }
    }

    // Tool-result messages come first (they refer back to the preceding
    // assistant turn); then any leftover user content.
    for (auto& tr : tool_results)
        out.push_back(std::move(tr));
    if (!other_parts.empty()) {
        // Collapse a single text part to a bare string for readability
        if (other_parts.size() == 1 && other_parts[0].value("type", "") == "text") {
            out.push_back({{"role", "user"}, {"content", other_parts[0].value("text", "")}});
        } else {
            out.push_back({{"role", "user"}, {"content", other_parts}});
        }
    }
}

// Convert Anthropic tools[] entries (input_schema -> parameters) and wrap
// them into OpenAI tools[] entries.
json convert_tools(const json& anth_tools) {
    json out = json::array();
    if (!anth_tools.is_array())
        return out;
    for (const auto& t : anth_tools) {
        if (!t.is_object())
            continue;
        out.push_back({
            {"type", "function"},
            {"function",
             {
                 {"name", t.value("name", "")},
                 {"description", t.value("description", "")},
                 {"parameters", t.value("input_schema", json::object())},
             }},
        });
    }
    return out;
}

// Convert Anthropic tool_choice -> OpenAI tool_choice. Defaults to "auto".
json convert_tool_choice(const json& anth_choice) {
    if (anth_choice.is_null())
        return "auto";
    if (anth_choice.is_string())
        return anth_choice;  // user supplied OpenAI-style
    if (!anth_choice.is_object())
        return "auto";

    std::string type = anth_choice.value("type", "auto");
    if (type == "auto")
        return "auto";
    if (type == "none")
        return "none";
    if (type == "any")
        return "required";  // "must call some tool"
    if (type == "tool") {
        return json{
            {"type", "function"},
            {"function", {{"name", anth_choice.value("name", "")}}},
        };
    }
    return "auto";
}

// Anthropic allows `system` as string OR [content_block]; collapse to text.
std::string flatten_system(const json& system_field) {
    if (system_field.is_null())
        return "";
    if (system_field.is_string())
        return system_field.get<std::string>();
    if (!system_field.is_array())
        return "";
    std::string out;
    for (const auto& block : system_field) {
        if (block.is_object() && block.value("type", "") == "text") {
            if (!out.empty())
                out += "\n";
            out += block.value("text", "");
        }
    }
    return out;
}

}  // namespace

// Any cache_control marker anywhere in the request? Anthropic allows it on
// system blocks, message content blocks, and tool definitions. imp maps the
// marker to prompt-KV pinning (block-granular prefix cache), so the marker's
// exact position doesn't matter — any marker pins the request's prompt
// prefix against eviction.
static bool has_cache_control(const json& anth) {
    auto any_block = [](const json& blocks) {
        if (!blocks.is_array())
            return false;
        for (const auto& b : blocks)
            if (b.is_object() && b.contains("cache_control"))
                return true;
        return false;
    };
    if (anth.contains("system") && any_block(anth["system"]))
        return true;
    if (anth.contains("tools") && any_block(anth["tools"]))
        return true;
    if (anth.contains("messages") && anth["messages"].is_array()) {
        for (const auto& m : anth["messages"])
            if (m.is_object() && m.contains("content") && any_block(m["content"]))
                return true;
    }
    return false;
}

json anthropic_to_openai_body(const json& anth) {
    json oai = json::object();

    // Core fields ----------------------------------------------------------
    oai["model"] = anth.value("model", "");
    // Anthropic requires max_tokens; OpenAI's default is unbounded, but we
    // prefer to pass the user's intent through.
    if (anth.contains("max_tokens"))
        oai["max_tokens"] = anth["max_tokens"];
    if (anth.contains("temperature"))
        oai["temperature"] = anth["temperature"];
    if (anth.contains("top_p"))
        oai["top_p"] = anth["top_p"];
    if (anth.contains("top_k"))
        oai["top_k"] = anth["top_k"];
    if (anth.contains("stream"))
        oai["stream"] = anth["stream"];
    if (anth.contains("metadata") && anth["metadata"].is_object() && anth["metadata"].contains("user_id")) {
        oai["user"] = anth["metadata"]["user_id"];
    }
    if (anth.contains("stop_sequences"))
        oai["stop"] = anth["stop_sequences"];

    // Messages -------------------------------------------------------------
    json oai_messages = json::array();

    // Prepend system as a role=system message. `system` is a top-level
    // field in Anthropic, not part of messages.
    std::string system_text = flatten_system(anth.value("system", json(nullptr)));
    if (!system_text.empty()) {
        oai_messages.push_back({{"role", "system"}, {"content", system_text}});
    }

    if (anth.contains("messages") && anth["messages"].is_array()) {
        for (const auto& m : anth["messages"]) {
            std::string role = m.value("role", "user");
            if (role == "assistant") {
                push_assistant_turn(oai_messages, m);
            } else {
                // Anthropic uses "user" for both genuine user turns and
                // tool_result carriers; push_user_turn splits them.
                push_user_turn(oai_messages, m);
            }
        }
    }
    oai["messages"] = std::move(oai_messages);

    // Tools ----------------------------------------------------------------
    if (anth.contains("tools") && anth["tools"].is_array() && !anth["tools"].empty()) {
        oai["tools"] = convert_tools(anth["tools"]);
    }
    if (anth.contains("tool_choice")) {
        oai["tool_choice"] = convert_tool_choice(anth["tool_choice"]);
    }

    // cache_control → prompt-KV pinning (internal "cache_prompt" field,
    // same name the OpenAI route accepts directly).
    if (has_cache_control(anth))
        oai["cache_prompt"] = true;

    return oai;
}

// ---------------------------------------------------------------------------
// Response: OpenAI chat.completion body -> Anthropic Messages body
// ---------------------------------------------------------------------------

std::string make_message_id(uint64_t counter) {
    char buf[48];
    std::snprintf(buf, sizeof(buf), "msg_imp_%016llx", static_cast<unsigned long long>(counter));
    return std::string(buf);
}

std::string tool_call_id_to_anthropic(const std::string& openai_id) {
    // OpenAI-style id from imp: "call_imp_<counter>". Rewrite to Anthropic's
    // "toolu_..." prefix so well-behaved Anthropic clients accept it.
    constexpr const char* kSrcPrefix = "call_imp_";
    if (openai_id.rfind("toolu_", 0) == 0)
        return openai_id;  // already OK
    if (openai_id.rfind(kSrcPrefix, 0) == 0) {
        return std::string("toolu_") + openai_id.substr(std::char_traits<char>::length(kSrcPrefix));
    }
    return std::string("toolu_") + openai_id;
}

json openai_to_anthropic_response(const json& oai, const std::string& anth_model) {
    // Pass through OpenAI error envelopes unchanged but flip the type.
    if (oai.contains("error")) {
        return {
            {"type", "error"},
            {"error", oai["error"]},
        };
    }

    // Dig into the first choice (we reject n>1 for /v1/messages upstream).
    json choice = json::object();
    if (oai.contains("choices") && oai["choices"].is_array() && !oai["choices"].empty()) {
        choice = oai["choices"][0];
    }
    const json& msg = choice.contains("message") ? choice["message"] : json::object();

    // Build content[] blocks. Thinking first (Anthropic emits the thinking
    // block before the visible answer), then text, then tool_use entries.
    json content = json::array();
    if (msg.contains("reasoning_content") && msg["reasoning_content"].is_string()) {
        std::string thinking = msg["reasoning_content"].get<std::string>();
        if (!thinking.empty()) {
            content.push_back({{"type", "thinking"}, {"thinking", thinking}});
        }
    }
    if (msg.contains("content") && msg["content"].is_string()) {
        std::string text = msg["content"].get<std::string>();
        if (!text.empty()) {
            content.push_back({{"type", "text"}, {"text", text}});
        }
    }
    if (msg.contains("tool_calls") && msg["tool_calls"].is_array()) {
        for (const auto& tc : msg["tool_calls"]) {
            std::string id = tc.value("id", "");
            std::string name;
            json input = json::object();
            if (tc.contains("function") && tc["function"].is_object()) {
                name = tc["function"].value("name", "");
                if (tc["function"].contains("arguments")) {
                    const auto& args = tc["function"]["arguments"];
                    if (args.is_string()) {
                        try {
                            input = json::parse(args.get<std::string>());
                        } catch (...) {
                            input = json::object();
                        }
                    } else if (args.is_object()) {
                        input = args;
                    }
                }
            }
            content.push_back({
                {"type", "tool_use"},
                {"id", tool_call_id_to_anthropic(id)},
                {"name", name},
                {"input", input},
            });
        }
    }

    // Map finish_reason -> stop_reason.
    std::string finish = choice.value("finish_reason", "stop");
    std::string stop_reason;
    if (finish == "stop")
        stop_reason = "end_turn";
    else if (finish == "length")
        stop_reason = "max_tokens";
    else if (finish == "tool_calls")
        stop_reason = "tool_use";
    else if (finish == "cancelled")
        stop_reason = "end_turn";
    else
        stop_reason = finish;  // pass through

    json usage_out = {
        {"input_tokens", 0},
        {"output_tokens", 0},
    };
    if (oai.contains("usage") && oai["usage"].is_object()) {
        const auto& u = oai["usage"];
        int prompt_tokens = u.value("prompt_tokens", 0);
        usage_out["output_tokens"] = u.value("completion_tokens", 0);
        // Anthropic splits the prompt token count: tokens served from the
        // prefix cache are reported separately (cache_read_input_tokens) and
        // excluded from input_tokens. imp surfaces the prefix-cache hit count
        // via OpenAI's prompt_tokens_details.cached_tokens — pass it through.
        int cached = 0;
        int creation = 0;
        if (u.contains("prompt_tokens_details") && u["prompt_tokens_details"].is_object()) {
            cached = u["prompt_tokens_details"].value("cached_tokens", 0);
            creation = u["prompt_tokens_details"].value("cache_creation_tokens", 0);
        }
        if (cached > prompt_tokens)
            cached = prompt_tokens;
        usage_out["input_tokens"] = prompt_tokens - cached;
        usage_out["cache_read_input_tokens"] = cached;
        usage_out["cache_creation_input_tokens"] = creation;
    }

    // Use the OpenAI id if present; otherwise synthesize one.
    std::string id = oai.value("id", "");
    if (id.empty()) {
        id = make_message_id(static_cast<uint64_t>(oai.value("created", 0)));
    } else if (id.rfind("chatcmpl", 0) == 0) {
        id = std::string("msg_") + id.substr(std::char_traits<char>::length("chatcmpl"));
    }

    return {
        {"id", id},
        {"type", "message"},
        {"role", "assistant"},
        {"content", std::move(content)},
        {"model", anth_model},
        {"stop_reason", std::move(stop_reason)},
        {"stop_sequence", nullptr},
        {"usage", std::move(usage_out)},
    };
}

}  // namespace imp_server::anthropic
