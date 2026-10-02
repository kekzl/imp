// AUTO-SPLIT from handlers_chat_core.cpp (800-code-LOC hard gate, #1017/#1018). Body parsing:
// parse_chat_request_params populates ChatRequestContext (sampling, response_format, tools,
// logit_bias, vision, stop sequences, thinking knobs) from the request JSON.

#include "runtime/engine.h"
#include "handlers.h"
#include "handlers_internal.h"
#include "utils.h"
#include "tool_call.h"
#include "tool_call_dialect.h"
#include "anthropic.h"
#include "stream_pipeline.h"
#include "image_fetch.h"
#include "reasoning_split.h"

#include "api/imp_internal.h"
#include "vision/image_processor.h"
#include "runtime/request.h"
#include "memory/kv_cache.h"
#include "model/hf_hub.h"
#include "model/image_placeholders.h"
#include "runtime/config.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <functional>
#include <vector>

#include <cuda_runtime.h>

// Defined in handlers_chat_core.cpp — suppresses inner request-log entries
// while handle_messages delegates through the OpenAI shim.
extern thread_local bool g_in_anthropic_shim;

// Reasoning a client sends back on a prior assistant message (OpenAI
// reasoning_content; the Anthropic shim folds thinking blocks into it). Handed to
// the Jinja template as message.reasoning_content; "" when absent.
// data: URI or (with --allow-remote-images) http(s) URL -> bytes. A failure leaves `bytes` empty and
// sets `error` once; the string never echoes the URL (#1610: distinguishable errors made this a port
// scanner of the server's own network).
static void read_vision_url(ServerState& state, const std::string& url, const char* what,
                            std::vector<uint8_t>& bytes, std::string& error) {
    const bool remote = url.rfind("http://", 0) == 0 || url.rfind("https://", 0) == 0;
    if (url.rfind("data:", 0) == 0) {
        auto comma = url.find(',');
        if (comma != std::string::npos)
            bytes = base64_decode(url.substr(comma + 1));
    } else if (remote) {
        // Off by default, bounded when on, no redirects (image_fetch.h).
        auto fetched = imp_server::fetch_remote_image(url, state.default_args.allow_remote_images);
        if (fetched.ok)
            bytes = std::move(fetched.bytes);
        else
            IMP_LOG_WARN("%s not fetched: %s", what, fetched.detail.c_str());
    }
    if (bytes.empty() && error.empty())
        error = (remote && !state.default_args.allow_remote_images)
                    ? std::string("could not read ") + what +
                          ": remote URLs are disabled on this server; send a data: URI, or start it with "
                          "--allow-remote-images"
                    : std::string("could not read ") + what;
}

static std::string prior_reasoning(const json& msg) {
    if (msg.contains("reasoning_content") && msg["reasoning_content"].is_string())
        return msg["reasoning_content"].get<std::string>();
    return "";
}

// Populates ctx.params/log_*/req_id/snap.tpl_family (early best-effort). Returns false with a
// 400 JSON error on validation failure; true means proceed to state snapshot + tokenize.
bool parse_chat_request_params(const httplib::Request& req, httplib::Response& res, ServerState& state,
                               ChatRequestContext& ctx) {
    // Capture inputs for opt-in JSONL request logging. Only used when
    // state.request_logger.enabled and the call is not an inner shim.
    ctx.t_log_start = std::chrono::system_clock::now();
    ctx.log_endpoint = req.path;
    // Same key the rate limiter uses: an untrusted X-Forwarded-For in the
    // request log is a forged identity in the audit trail (#1614).
    ctx.log_client_ip = state.rate_limit_key(req.remote_addr, req.get_header_value("X-Forwarded-For"));
        ctx.log_client_request_id = sanitize_for_echo(req.get_header_value("X-Request-Id"), 128);
    ctx.trace.traceparent = req.get_header_value("traceparent");
    ctx.log_raw_body = req.body;
    ctx.client_gone = req.is_connection_closed;
    ctx.log_skip = g_in_anthropic_shim;

    // #1607: bound the nesting before any recursive parser sees it.
    if (reject_body_too_deep(req, res))
        return false;

    // Parse request body
    json body;
    try {
        body = json::parse(req.body);
        drop_null_fields(body);
    } catch (const json::parse_error& e) {
        send_json_error(res, 400, "invalid_request_error", std::string("Invalid JSON: ") + e.what());
        return false;
    }

    // Validate sampling parameters
    if (!validate_sampling_params(body, res))
        return false;

    // Extract parameters
    auto messages = body.value("messages", json::array());
    if (messages.empty()) {
        res.status = 400;
        json err = {{"error",
                     {{"message", "messages array is required and must not be empty"},
                      {"type", "invalid_request_error"}}}};
        res.set_content(dump_safe(err), "application/json");
        return false;
    }
    // Bound the conversation length: each message is tokenized + template-expanded
    // on the host, so an unbounded array is a CPU/memory DoS within the body cap.
    constexpr size_t kMaxMessages = 10000;
    if (messages.size() > kMaxMessages) {
        res.status = 400;
        json err = {{"error",
                     {{"message", "messages array exceeds maximum of 10000 entries"},
                      {"type", "invalid_request_error"}}}};
        res.set_content(dump_safe(err), "application/json");
        return false;
    }

    parse_sampling_fields(body, state.default_think_budget, ctx.params);
    // "max_completion_tokens" (current OpenAI SDKs) takes precedence over the
    // deprecated "max_tokens"; without this, SDK requests silently ran with
    // the server default.
    ctx.params.max_tokens = parse_max_tokens_field(body, state.default_max_tokens);
    ctx.params.n_completions = body.value("n", 1);
    if (ctx.params.n_completions < 1)
        ctx.params.n_completions = 1;
    // Each n is a full independent generation, run sequentially, but the whole request still counts
    // as ONE against --rate-limit and --max-concurrent. max_tokens is clamped to context; n was not
    // clamped at all before (#1616).
    if (state.max_n > 0 && ctx.params.n_completions > state.max_n) {
        send_json_error(res, 400, "invalid_request_error",
                        "\"n\" is " + std::to_string(ctx.params.n_completions) +
                            ", above the server limit of " + std::to_string(state.max_n) + " (--max-n)");
        return false;
    }

    // Streaming with n > 1 is not supported
    if (ctx.params.stream && ctx.params.n_completions > 1) {
        res.status = 400;
        json err = {
            {"error",
             {{"message", "streaming with n > 1 is not supported"}, {"type", "invalid_request_error"}}}};
        res.set_content(dump_safe(err), "application/json");
        return false;
    }

    // OpenAI caps stop sequences at 4; Anthropic /v1/messages does not, and its stop_sequences
    // convert through this parser - allow up to kMaxStopSequences=16 and warn when truncating.
    constexpr size_t kMaxStopSequences = 16;
    if (parse_stop_field(body, kMaxStopSequences, ctx.params.stop_sequences)) {
        IMP_LOG_WARN("request sent %zu stop sequences; keeping the first %zu", body["stop"].size(),
                     kMaxStopSequences);
    }
    ctx.params.max_stop_len = 0;
    for (const auto& s : ctx.params.stop_sequences)
        ctx.params.max_stop_len = std::max(ctx.params.max_stop_len, s.size());

    // Parse logprobs parameters
    if (const auto lp = body.find("logprobs"); lp != body.end() && !lp->is_null() && !lp->is_boolean()) {
        send_json_error(res, 400, "invalid_request_error",
                        std::string("\"logprobs\" must be a boolean, got ") + lp->type_name());
        return false;
    }
    ctx.params.req_logprobs = body.value("logprobs", false);
    ctx.params.top_logprobs = body.value("top_logprobs", 0);
    if (ctx.params.top_logprobs < 0)
        ctx.params.top_logprobs = 0;
    if (ctx.params.top_logprobs > 20)
        ctx.params.top_logprobs = 20;

    // Parse response_format for JSON mode / JSON Schema / regex
    if (body.contains("response_format") && body["response_format"].is_object()) {
        std::string fmt_type = body["response_format"].value("type", "text");
        if (fmt_type == "regex") {
            // {"type":"regex","regex":"..."} — the whole reply must match.
            // Accepted at "pattern" too, since that is the JSON-Schema spelling
            // and callers reach for it.
            const auto& rf = body["response_format"];
            if (rf.contains("regex") && rf["regex"].is_string())
                ctx.params.regex_pattern = rf["regex"].get<std::string>();
            else if (rf.contains("pattern") && rf["pattern"].is_string())
                ctx.params.regex_pattern = rf["pattern"].get<std::string>();
            else {
                send_json_error(res, 400, "invalid_request_error",
                                "\"response_format\" is type \"regex\" but carries no string "
                                "\"regex\" (or \"pattern\")");
                return false;
            }
        } else if (fmt_type == "grammar") {
            // {"type":"grammar","grammar":"root ::= ..."} — a GBNF grammar the
            // whole reply must derive. "gbnf" is accepted as a spelling too,
            // since that is what the format is called everywhere else.
            const auto& rf = body["response_format"];
            if (rf.contains("grammar") && rf["grammar"].is_string())
                ctx.params.grammar = rf["grammar"].get<std::string>();
            else if (rf.contains("gbnf") && rf["gbnf"].is_string())
                ctx.params.grammar = rf["gbnf"].get<std::string>();
            else {
                send_json_error(res, 400, "invalid_request_error",
                                "\"response_format\" is type \"grammar\" but carries no string "
                                "\"grammar\" (or \"gbnf\")");
                return false;
            }
        } else if (fmt_type == "json_object") {
            ctx.params.json_mode = true;
        } else if (fmt_type == "json_schema") {
            ctx.params.json_mode = true;
            const auto& rf = body["response_format"];
            if (!rf.contains("json_schema") || !rf["json_schema"].is_object()) {
                send_json_error(res, 400, "invalid_request_error",
                                "\"response_format\" is type \"json_schema\" but carries no object "
                                "\"json_schema\"");
                return false;
            }
            if (!rf["json_schema"].contains("schema") || !rf["json_schema"]["schema"].is_object()) {
                send_json_error(res, 400, "invalid_request_error",
                                "\"response_format.json_schema\" carries no object \"schema\"");
                return false;
            }
            if (body["response_format"].contains("json_schema") &&
                body["response_format"]["json_schema"].is_object()) {
                auto& js = body["response_format"]["json_schema"];
                if (js.contains("schema") && js["schema"].is_object()) {
                    const auto& sch = js["schema"];
                    // Free-form object schema ({"type":"object"}, no properties/enum) carries no structure
                    // the
                    // constrainer could enforce (its key phase would reject every token). Treated as
                    // json_object:
                    // leave json_schema_str empty so the request takes the any-JSON constrainer path.
                    const bool free_form = sch.value("type", "") == "object" &&
                                           (!sch.contains("properties") || sch["properties"].empty()) &&
                                           !sch.contains("enum");
                    if (!free_form) {
                        ctx.params.json_schema_str = dump_safe(sch);
                    }
                }
            }
        } else if (fmt_type != "text") {
            // A response_format type this build does not know is a DIFFERENT request, not a weaker one:
            // answering it as free text with 200 tells the caller their constraint held (#1591).
            send_json_error(res, 400, "invalid_request_error",
                            "unknown \"response_format.type\": \"" + sanitize_for_echo(fmt_type, 64) +
                                "\" (known: text, json_object, json_schema, regex, grammar)");
            return false;
        }
    }

    // vLLM/SGLang spell it `guided_regex` at the top level; accept that too so
    // an existing client works unchanged. response_format wins if both appear.
    if (ctx.params.regex_pattern.empty() && body.contains("guided_regex") &&
        body["guided_regex"].is_string())
        ctx.params.regex_pattern = body["guided_regex"].get<std::string>();

    // Grammars have two established spellings and no response_format convention: llama.cpp takes
    // `grammar`, vLLM takes `guided_grammar`. Accept both.
    if (ctx.params.grammar.empty() && body.contains("grammar") && body["grammar"].is_string())
        ctx.params.grammar = body["grammar"].get<std::string>();
    if (ctx.params.grammar.empty() && body.contains("guided_grammar") && body["guided_grammar"].is_string())
        ctx.params.grammar = body["guided_grammar"].get<std::string>();

    if (const std::string err = parse_logit_bias(body, state.max_logit_bias, ctx.params.logit_bias);
        !err.empty()) {
        send_json_error(res, 400, "invalid_request_error", err, "logit_bias");
        return false;
    }

    // Parse stream_options for include_usage
    if (body.contains("stream_options") && body["stream_options"].is_object()) {
        ctx.params.include_usage = body["stream_options"].value("include_usage", false);
    }

    // cache_prompt (parse_sampling_fields) pins the first cache_prefix_messages messages.
    if (body.contains("cache_prefix_messages") && body["cache_prefix_messages"].is_number_integer())
        ctx.params.cache_prefix_messages = body["cache_prefix_messages"].get<int>();

    // Per-request speculative-decode contract (imp extension). Absent -> tri-state stays -1 (server
    // default). true/false forces every drafter on/off; {"mtp_k":N} sets MTP depth for this request
    // alone. A depth outside the armed range is a 400 naming the range, never silently ignored (#1384).
    {
        const int armed_mtp_k = state.armed_mtp_k.load(std::memory_order_relaxed);
        const SpecFieldParse sp = parse_spec_field_(body, armed_mtp_k);
        if (!sp.ok) {
            send_json_error(res, 400, "invalid_request_error", sp.error);
            return false;
        }
        ctx.params.spec_override = sp.spec_override;
        ctx.params.spec_mtp_k = sp.mtp_k;
    }

    // OpenAI Predicted Outputs: {"prediction":{"type":"content","content": string | [{"type":"text",...}]}}.
    // The text only speeds up verify-accept, never changes output - unknown shapes are ignored, not rejected.
    if (body.contains("prediction") && body["prediction"].is_object()) {
        const auto& pred = body["prediction"];
        if (pred.value("type", "content") == "content" && pred.contains("content")) {
            const auto& content = pred["content"];
            if (content.is_string()) {
                ctx.params.prediction_text = content.get<std::string>();
            } else if (content.is_array()) {
                for (const auto& part : content) {
                    if (part.is_object() && part.value("type", "text") == "text" && part.contains("text") &&
                        part["text"].is_string())
                        ctx.params.prediction_text += part["text"].get<std::string>();
                }
            }
        }
    }

    // Parse tool calling parameters
    ctx.params.tools = body.value("tools", json::array());
    ctx.params.tool_choice = body.value("tool_choice", json("auto"));
    ctx.params.parallel_tool_calls = body.value("parallel_tool_calls", true);
    ctx.params.has_tools = !ctx.params.tools.empty() &&
                           !(ctx.params.tool_choice.is_string() &&
                             ctx.params.tool_choice.get<std::string>() == "none");

    // tools + response_format=json_schema/json_object: the engine-side gate stays "no-mask" through
    // tool-call bodies (ConstraintManager::prepare, PreambleGate::configure_with_tools) and decides
    // at runtime which path the model takes; tool-call dialect comes from tpl_family.

    // Snapshot template family (may be re-snapshotted under lock in the orchestrator)
    bool tool_xml_dialect = false;
    {
        std::lock_guard<std::timed_mutex> lock(state.mtx);
        ctx.snap.tpl_family = state.have_template ? state.chat_tpl.family() : imp::ChatTemplateFamily::CHATML;
        tool_xml_dialect = state.have_template && state.chat_tpl.tool_xml_dialect();
    }

    // logprobs on a constrained request drops it out of the ConstrainedPipeline
    // fast path to eager decode (~102 vs ~235 tok/s) — silent until now.
    // Surface it: one WARN per request + a /metrics counter (#1006).
    if (ctx.params.req_logprobs && (ctx.params.json_mode || !ctx.params.json_schema_str.empty())) {
        state.metrics.constrained_eager_fallback++;
        IMP_LOG_WARN(
            "constrained request with logprobs: leaving the ConstrainedPipeline "
            "fast path for eager decode (expect ~2x slower decode)");
    }

    // Enforced tool calling (#1002) is collected POST-snapshot in handlers_chat_core: the request
    // may auto-load or name a different model, so the constraint dialect must come from the
    // template that will actually render this prompt, not the parse-time family guess here.

    // Convert JSON messages to ChatMessage vector, extracting image data if present
    for (const auto& msg : messages) {
        std::string role = msg.value("role", "user");

        if (role == "tool") {
            // Tool response message — format for the model
            std::string content = format_tool_response(ctx.snap.tpl_family, msg);
            // Gemma's chat template skips standalone role=tool messages and expects tool_response markers
            // glued onto the assistant message that produced the call; ChatML/Llama3 templates render
            // standalone tool messages, so keep the push there.
            if (tool_call_dialect(ctx.snap.tpl_family).tool_response_joins_assistant &&
                !ctx.params.chat_msgs.empty() && ctx.params.chat_msgs.back().role == "assistant") {
                ctx.params.chat_msgs.back().content += content;
            } else {
                ctx.params.chat_msgs.push_back({"tool", content});
            }
        } else if (role == "assistant" && msg.contains("tool_calls")) {
            // XML-dialect templates (Qwen-Coder) must replay prior tool_calls in the XML shape the model
            // itself emits, not the ChatML JSON body - a JSON replay teaches the model the wrong dialect for
            // its NEXT call, exactly what the armed XML grammar forbids.
            std::string content_str;
            // Array content (text parts) is valid OpenAI; get<std::string>() answered it with a raw 400.
            if (msg.contains("content") && msg["content"].is_string())
                content_str = msg["content"].get<std::string>();
            else if (msg.contains("content"))
                join_text_parts(msg["content"], content_str);
            std::string reconstructed = reconstruct_tool_call_output(ctx.snap.tpl_family, msg["tool_calls"],
                                                                     content_str, tool_xml_dialect);
            ctx.params.chat_msgs.push_back({"assistant", reconstructed, prior_reasoning(msg)});
        } else if (msg.contains("content") && msg["content"].is_array()) {
            // OpenAI multimodal format: content is array of parts
            std::string text_parts;
            for (const auto& part : msg["content"]) {
                std::string type = part.value("type", "");
                if (type == "text") {
                    if (!text_parts.empty())
                        text_parts += "\n";
                    text_parts += part.value("text", "");
                } else if (type == "image_url" && part.contains("image_url")) {
                    std::string url = part["image_url"].value("url", "");
                    // Each image part is decoded at full resolution on a worker thread, so the count is
                    // bounded
                    // before the bytes are read (AUDIT_arch_2026 F2-4); the mmproj one-image rule applies on
                    // top.
                    if (state.max_images > 0 &&
                        static_cast<int>(ctx.params.images.size()) >= state.max_images) {
                        send_json_error(res, 400, "invalid_request_error",
                                        "request carries more than " + std::to_string(state.max_images) +
                                            " images, the server limit (--max-images-per-request)");
                        return false;
                    }
                    // One slot per part, appended before it is filled: if the
                    // fetch below fails the request is rejected, so a half-read
                    // list never reaches the prompt builder.
                    ctx.params.images.emplace_back();
                    ctx.params.vision_order.push_back('i');
                    read_vision_url(state, url, "image_url", ctx.params.images.back(),
                                    ctx.params.image_error);
                } else if (type == "video") {
                    // {"type":"video","video":[frame URL, ...],"fps":F | "timestamps":[s, ...]}: frames the
                    // client sampled itself (no decoder here). Frame k sits at k/F s or timestamps[k].
                    const json frames = part.value("video", json());
                    if (!frames.is_array() || frames.size() < 2 ||
                        frames.size() > imp::Engine::kQwenVideoMaxFrames) {
                        send_json_error(res, 400, "invalid_request_error",
                                        "a video part needs \"video\": an array of 2.." +
                                            std::to_string(imp::Engine::kQwenVideoMaxFrames) +
                                            " frame image URLs");
                        return false;
                    }
                    ChatVideoInput v;
                    if (part.contains("timestamps")) {
                        const json& ts = part["timestamps"];
                        if (!ts.is_array() || ts.size() != frames.size()) {
                            send_json_error(res, 400, "invalid_request_error",
                                            "video \"timestamps\" needs one number per frame");
                            return false;
                        }
                        for (const auto& t : ts) {
                            if (!t.is_number()) {
                                send_json_error(res, 400, "invalid_request_error",
                                                "video \"timestamps\" needs one number per frame");
                                return false;
                            }
                            v.seconds.push_back(t.get<double>());
                        }
                    } else {
                        const json fps_j = part.value("fps", json(imp::kQwenVideoDefaultFps));
                        const double fps = fps_j.is_number() ? fps_j.get<double>() : -1.0;
                        if (!(fps > 0.0)) {
                            send_json_error(res, 400, "invalid_request_error", "video \"fps\" must be > 0");
                            return false;
                        }
                        for (size_t k = 0; k < frames.size(); ++k)
                            v.seconds.push_back(static_cast<double>(k) / fps);
                    }
                    for (const auto& f : frames) {
                        v.frames.emplace_back();
                        read_vision_url(state, f.is_string() ? f.get<std::string>() : std::string(),
                                        "video frame", v.frames.back(), ctx.params.image_error);
                    }
                    ctx.params.videos.push_back(std::move(v));
                    ctx.params.vision_order.push_back('v');
                }
            }
            ctx.params.chat_msgs.push_back({role, text_parts});
        } else {
            std::string content;
            if (msg.contains("content") && !msg["content"].is_null()) {
                content = msg["content"].get<std::string>();
            }
            ctx.params.chat_msgs.push_back({role, content, prior_reasoning(msg)});
        }
    }

    // Log request received (structured)
    ctx.req_id = make_completion_id(state);
    // Trace join on the wire: the client's id when it sent one, the server's
    // req_id otherwise — every generation response carries SOME id a caller
    // can quote back. Set here so both the JSON and the SSE path get it.
    res.set_header("X-Request-Id", ctx.log_client_request_id.empty()
                                       ? ctx.req_id
                                       : ctx.log_client_request_id);
    IMP_LOG_INFO("[%s] chat/completions: prompt_msgs=%zu stream=%s max_tokens=%d temp=%.2f",
                 ctx.req_id.c_str(), messages.size(), ctx.params.stream ? "true" : "false",
                 ctx.params.max_tokens, ctx.params.temperature);

    // Validate model field (required per OpenAI spec)
    ctx.params.requested_model = body.value("model", "");
    if (ctx.params.requested_model.empty()) {
        res.status = 400;
        json err = {{"error", {{"message", "\"model\" is required"}, {"type", "invalid_request_error"}}}};
        res.set_content(dump_safe(err), "application/json");
        return false;
    }

    // Parse enable_thinking (only meaningful for think models; checked in orchestrator). The top-level
    // field wins; else chat_template_kwargs.enable_thinking (vLLM/SGLang form, OpenAI SDK extra_body),
    // which was ignored: Qwen3.8 reasoned with it false.
    const json* think_src = nullptr;
    if (body.contains("enable_thinking") && body["enable_thinking"].is_boolean())
        think_src = &body["enable_thinking"];
    else if (body.contains("chat_template_kwargs") && body["chat_template_kwargs"].is_object() &&
             body["chat_template_kwargs"].contains("enable_thinking") &&
             body["chat_template_kwargs"]["enable_thinking"].is_boolean())
        think_src = &body["chat_template_kwargs"]["enable_thinking"];
    ctx.params.enable_thinking_set = think_src != nullptr;
    ctx.params.enable_thinking_requested = think_src != nullptr && think_src->get<bool>();

    // reasoning_effort: handed to the chat template verbatim. A non-string is ignored rather than
    // rejected, matching the other optional scalars; a template that dislikes the value raises its
    // own exception (imp logs it and renders without the branch).
    if (body.contains("reasoning_effort") && body["reasoning_effort"].is_string())
        ctx.params.reasoning_effort = body["reasoning_effort"].get<std::string>();

    // Per-request LoRA adapter selection ("lora": "<name>"; absent/"" = base).
    ctx.params.lora_name = body.value("lora", std::string());

    return true;
}
