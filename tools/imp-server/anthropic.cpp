// Anthropic Messages API (/v1/messages) transforms.
// See anthropic.h for the public surface.

#include "anthropic.h"
#include "utils.h"

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
                std::string data_url = "data:";
                data_url += mime;
                data_url += ";base64,";
                data_url += data;
                oai_parts.push_back(
                    {{"type", "image_url"}, {"image_url", {{"url", data_url}}}});
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
                     {"arguments", dump_safe(input)},
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
            // (text and/or images — screenshot-returning tools). Text collapses
            // to a string for the OpenAI tool turn; IMAGE blocks cannot ride a
            // role:"tool" message, so they are re-homed onto the trailing user
            // turn (the multimodal path the engine already serves) with a
            // marker in the tool body tying them back to this result (#1006 —
            // previously they were silently dropped).
            std::string body;
            int n_images = 0;
            if (block.contains("content")) {
                const auto& c = block["content"];
                if (c.is_string())
                    body = c.get<std::string>();
                else if (c.is_array()) {
                    for (const auto& p : c) {
                        if (!p.is_object())
                            continue;
                        const std::string ptype = p.value("type", "");
                        if (ptype == "text") {
                            if (!body.empty())
                                body += "\n";
                            body += p.value("text", "");
                        } else if (ptype == "image" && p.contains("source")) {
                            json tmp = json::array({p});
                            auto converted = convert_message_content(tmp);
                            if (converted.is_array())
                                for (const auto& img : converted)
                                    other_parts.push_back(img);
                            n_images++;
                        }
                    }
                }
            }
            if (n_images > 0) {
                if (!body.empty())
                    body += "\n";
                body += "[" + std::to_string(n_images) +
                        " image(s) from this tool result follow in the next user message]";
            }
            // is_error was read by nothing, so a tool that failed reached the
            // model as an ordinary successful result and it went on as if the
            // call had worked (#1557). OpenAI's role:"tool" turn has no field
            // for this - the content IS the channel - so the failure is
            // labelled in the text, which is what the model reads either way.
            if (block.value("is_error", false)) {
                body = body.empty() ? "[tool error]" : "[tool error] " + body;
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

// Any cache_control marker in this block array? Anthropic allows markers on
// system blocks, message content blocks, and tool definitions. The `ttl`
// field ("5m"/"1h") is accepted but not modeled — pins recycle via the FIFO
// pin budget; there is no billing distinction locally.
static bool blocks_have_cache_marker(const json& blocks) {
    if (!blocks.is_array())
        return false;
    for (const auto& b : blocks)
        if (b.is_object() && b.contains("cache_control"))
            return true;
    return false;
}

// Any cache_control marker anywhere in the request? imp maps markers to
// prompt-KV pinning (block-granular prefix cache). The LAST marked
// system/message block additionally defines a per-breakpoint pin boundary
// (see anthropic_to_openai_body, #1046).
static bool has_cache_control(const json& anth) {
    if (anth.contains("system") && blocks_have_cache_marker(anth["system"]))
        return true;
    if (anth.contains("tools") && blocks_have_cache_marker(anth["tools"]))
        return true;
    if (anth.contains("messages") && anth["messages"].is_array()) {
        for (const auto& m : anth["messages"])
            if (m.is_object() && m.contains("content") && blocks_have_cache_marker(m["content"]))
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
    // imp extension: per-request speculative-decode override, passed through
    // to the shared OpenAI param parser.
    if (anth.contains("speculative"))
        oai["speculative"] = anth["speculative"];
    if (anth.contains("stream"))
        oai["stream"] = anth["stream"];
    if (anth.contains("metadata") && anth["metadata"].is_object() && anth["metadata"].contains("user_id")) {
        oai["user"] = anth["metadata"]["user_id"];
    }
    if (anth.contains("stop_sequences"))
        oai["stop"] = anth["stop_sequences"];

    // Constrained decoding. These are imp/vLLM/llama.cpp extensions with no
    // Anthropic equivalent, and the shim used to leave them behind — so the
    // SAME server honoured `guided_regex` on /v1/chat/completions and ignored
    // it on /v1/messages (measured: 'ZZZ6' vs free-form prose). Nothing
    // documented was violated, but one server answering the same extension two
    // ways is a trap, and the silence was total: a malformed pattern was not
    // rejected here either, because the admission check never saw the field.
    // Carrying them through fixes both halves at once.
    for (const char* key : {"guided_regex", "guided_grammar", "grammar", "response_format"})
        if (anth.contains(key))
            oai[key] = anth[key];

    // Extended-thinking control. Anthropic uses a `thinking` object:
    //   {"type":"enabled","budget_tokens":N}  |  {"type":"disabled"}
    // imp's orchestrator (handlers.cpp) reads `enable_thinking` (bool) and
    // `think_budget` on the OpenAI body. `think_budget` is a FRACTION of
    // max_tokens (default 0.5), so map Anthropic's absolute budget_tokens to
    // budget_tokens/max_tokens, clamped to [0,1]. Without this mapping a
    // think-model always reasoned on /v1/messages regardless of the request.
    if (anth.contains("thinking") && anth["thinking"].is_object()) {
        const auto& think = anth["thinking"];
        const std::string ttype = think.value("type", "");
        if (ttype == "disabled") {
            // Suppressing thinking on a think-model needs BOTH signals: the
            // orchestrator only sets suppress_thinking (which injects /no_think
            // so the template emits no <think>) when enable_thinking is false
            // AND think_budget <= 0 (handlers.cpp). The server's default budget
            // is 0.5, so without zeroing it the model would still reason.
            oai["enable_thinking"] = false;
            oai["think_budget"] = 0.0;
        } else if (ttype == "enabled") {
            oai["enable_thinking"] = true;
            if (think.contains("budget_tokens") && think["budget_tokens"].is_number()) {
                double budget = think["budget_tokens"].get<double>();
                double max_tokens = anth.value("max_tokens", 0.0);
                if (budget > 0.0 && max_tokens > 0.0) {
                    double frac = budget / max_tokens;
                    oai["think_budget"] = frac > 1.0 ? 1.0 : frac;
                }
            }
        }
    }

    // Messages -------------------------------------------------------------
    json oai_messages = json::array();

    // cache_control breakpoint tracking: after converting a marked system/
    // message block, remember how many converted messages form the cacheable
    // prefix. The LAST marker wins (Anthropic caches everything before it).
    int cache_prefix_msgs = -1;

    // Prepend system as a role=system message. `system` is a top-level
    // field in Anthropic, not part of messages.
    std::string system_text = flatten_system(anth.value("system", json(nullptr)));

    // The Messages API has no "system" role — it says so explicitly, and sending
    // one is an error there. imp accepted it and, falling through to the
    // not-assistant branch below, rendered it as a USER turn: the text reached
    // the model (verified by prompt-token count, so this was never data loss),
    // but with user semantics instead of system semantics, and nothing said so.
    // Clients ported from the OpenAI dialect, where the role IS legal, write
    // their system prompt exactly this way.
    //
    // Fold LEADING system messages into the system prompt, which is what the
    // caller means. Only leading ones: a "system" role appearing mid-conversation
    // is not a system prompt in any dialect, and keeps its existing handling.
    size_t leading_system = 0;
    if (anth.contains("messages") && anth["messages"].is_array()) {
        for (const auto& m : anth["messages"]) {
            if (!m.is_object() || m.value("role", "user") != "system")
                break;
            const std::string folded = flatten_system(m.value("content", json(nullptr)));
            if (!folded.empty()) {
                if (!system_text.empty())
                    system_text += "\n\n";
                system_text += folded;
            }
            leading_system++;
        }
    }

    if (!system_text.empty()) {
        oai_messages.push_back({{"role", "system"}, {"content", system_text}});
        if (anth.contains("system") && blocks_have_cache_marker(anth["system"]))
            cache_prefix_msgs = static_cast<int>(oai_messages.size());
    }

    if (anth.contains("messages") && anth["messages"].is_array()) {
        size_t idx = 0;
        for (const auto& m : anth["messages"]) {
            if (idx++ < leading_system)
                continue;  // already folded into the system prompt above
            std::string role = m.value("role", "user");
            if (role == "assistant") {
                push_assistant_turn(oai_messages, m);
            } else {
                // Anthropic uses "user" for both genuine user turns and
                // tool_result carriers; push_user_turn splits them.
                push_user_turn(oai_messages, m);
            }
            if (m.is_object() && m.contains("content") && blocks_have_cache_marker(m["content"]))
                cache_prefix_msgs = static_cast<int>(oai_messages.size());
        }
    }
    oai["messages"] = std::move(oai_messages);

    // Tools ----------------------------------------------------------------
    if (anth.contains("tools") && anth["tools"].is_array() && !anth["tools"].empty()) {
        oai["tools"] = convert_tools(anth["tools"]);
    }
    if (anth.contains("tool_choice")) {
        oai["tool_choice"] = convert_tool_choice(anth["tool_choice"]);
        // Anthropic expresses "one tool at a time" as
        // tool_choice.disable_parallel_tool_use; map it to the OpenAI
        // parallel_tool_calls flag the (non-)streaming tool loops honor so the
        // suppression actually reaches the /v1/messages path (#892).
        if (anth["tool_choice"].is_object() &&
            anth["tool_choice"].value("disable_parallel_tool_use", false)) {
            oai["parallel_tool_calls"] = false;
        }
    }

    // cache_control → prompt-KV pinning (internal "cache_prompt" field,
    // same name the OpenAI route accepts directly). Per-breakpoint
    // granularity (#1046): the last marked system/message block bounds the
    // pin to the prompt tokens before it ("cache_prefix_messages" = count of
    // leading converted messages). A marker on tools keeps the whole-prompt
    // pin — tools render into the system preamble, which is not expressible
    // as a message boundary.
    if (has_cache_control(anth)) {
        oai["cache_prompt"] = true;
        bool tools_marked = anth.contains("tools") && blocks_have_cache_marker(anth["tools"]);
        if (!tools_marked && cache_prefix_msgs > 0)
            oai["cache_prefix_messages"] = cache_prefix_msgs;
    }

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

const char* anthropic_stop_reason(const std::string& openai_finish, bool stop_sequence_matched) {
    if (openai_finish == "stop")
        return stop_sequence_matched ? "stop_sequence" : "end_turn";
    if (openai_finish == "length")
        return "max_tokens";
    if (openai_finish == "tool_calls")
        return "tool_use";
    // Anthropic's name for the same thing. "content_filter" is OpenAI's and is
    // not in Anthropic's stop_reason enum, so it used to pass through as an
    // unknown value.
    if (openai_finish == "content_filter")
        return "refusal";
    // "cancelled" is a client disconnect and "capacity" an admission refusal.
    // Neither is an Anthropic stop_reason, and "capacity" used to ship verbatim
    // on the streaming path while the non-streaming path answered 503 for the
    // same condition (#1552). The stream cannot change its status once
    // message_start is out, so it ends the turn and reports the fault as an
    // `error` event (#1553).
    return "end_turn";
}

json openai_to_anthropic_response(const json& oai, const std::string& anth_model,
                                  const std::string& stop_sequence) {
    // Pass through OpenAI error envelopes unchanged but flip the type.
    if (oai.contains("error")) {
        return {
            {"type", "error"},
            {"error", oai["error"]},
        };
    }

    // Dig into the first choice. The Anthropic Messages API has no `n`
    // parameter, so the engine only ever produces a single choice here.
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
    std::string stop_reason = anthropic_stop_reason(finish, !stop_sequence.empty());

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
        int evicted = 0;
        if (u.contains("prompt_tokens_details") && u["prompt_tokens_details"].is_object()) {
            cached = u["prompt_tokens_details"].value("cached_tokens", 0);
            creation = u["prompt_tokens_details"].value("cache_creation_tokens", 0);
            evicted = u["prompt_tokens_details"].value("evicted_tokens", 0);
        }
        if (cached > prompt_tokens)
            cached = prompt_tokens;
        usage_out["input_tokens"] = prompt_tokens - cached;
        usage_out["cache_read_input_tokens"] = cached;
        usage_out["cache_creation_input_tokens"] = creation;
        // Anthropic's usage shape has no slot for "we dropped context", so this
        // is an imp-namespaced extension rather than a guess at their schema.
        // Only present when eviction actually fired.
        if (evicted > 0)
            usage_out["imp_evicted_tokens"] = evicted;
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
        {"stop_sequence", stop_sequence.empty() ? json(nullptr) : json(stop_sequence)},
        {"usage", std::move(usage_out)},
    };
}

}  // namespace imp_server::anthropic
