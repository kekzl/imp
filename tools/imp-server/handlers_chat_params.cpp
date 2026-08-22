// AUTO-SPLIT from handlers_chat_core.cpp (verbatim move; see
// handlers_internal.h) — the file crossed the 800-code-LOC hard gate after
// #1017/#1018. Body parsing for the shared chat-completion machinery:
// parse_chat_request_params populates ChatRequestContext from the request
// JSON (sampling params, response_format, tools + enforced tool constraints,
// logit_bias, vision parts, stop sequences, thinking knobs).

#include "handlers.h"
#include "handlers_internal.h"
#include "utils.h"
#include "tool_call.h"
#include "anthropic.h"
#include "stream_pipeline.h"
#include "image_fetch.h"
#include "reasoning_split.h"

#include "api/imp_internal.h"
#include "vision/image_processor.h"
#include "runtime/request.h"
#include "memory/kv_cache.h"
#include "model/hf_hub.h"
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

// Populates ctx.params, ctx.log_*, ctx.req_id, ctx.snap.tpl_family (early best-
// effort snapshot used to format tool-role messages in the conversion loop).
// On parse/validation failure: sets res with 400 + error JSON and returns false.
// On success: returns true; caller proceeds to state snapshot + tokenize.
bool parse_chat_request_params(const httplib::Request& req, httplib::Response& res, ServerState& state,
                               ChatRequestContext& ctx) {
    // Capture inputs for opt-in JSONL request logging. Only used when
    // state.request_logger.enabled and the call is not an inner shim.
    ctx.t_log_start = std::chrono::system_clock::now();
    ctx.log_endpoint = req.path;
    // Same key the rate limiter uses: an untrusted X-Forwarded-For in the
    // request log is a forged identity in the audit trail (#1614).
    ctx.log_client_ip = state.rate_limit_key(req.remote_addr, req.get_header_value("X-Forwarded-For"));
    ctx.log_raw_body = req.body;
    ctx.log_skip = g_in_anthropic_shim;

    // #1607: bound the nesting before any recursive parser sees it.
    if (reject_body_too_deep(req, res))
        return false;

    // Parse request body
    json body;
    try {
        body = json::parse(req.body);
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

    ctx.params.temperature = body.value("temperature", 0.7f);
    ctx.params.top_p_explicit = body.contains("top_p");
    ctx.params.top_k_explicit = body.contains("top_k");
    ctx.params.rep_pen_explicit = body.contains("repetition_penalty");
    // 1.05 default is mild — breaks pathological repetition loops on
    // verbose-think models (Qwen3.6-NVFP4 falling into "Wie wär es mit
    // diesem hier?" 40-iteration spirals on multi-turn sensitive prompts)
    // without disrupting structurally-repetitive valid output (JSON keys,
    // markdown lists, code idioms). Callers that need deterministic
    // sampling (validation harness, perf tests) can pass 1.0 explicitly.
    ctx.params.top_p = body.value("top_p", 0.95f);
    ctx.params.top_k = body.value("top_k", 40);
    // "max_completion_tokens" (current OpenAI SDKs) takes precedence over the
    // deprecated "max_tokens"; without this, SDK requests silently ran with
    // the server default.
    ctx.params.max_tokens = parse_max_tokens_field(body, state.default_max_tokens);
    ctx.params.seed = body.value("seed", -1);
    ctx.params.stream = body.value("stream", false);
    ctx.params.n_completions = body.value("n", 1);
    if (ctx.params.n_completions < 1)
        ctx.params.n_completions = 1;
    // Each n is a full independent generation, run sequentially, and the whole
    // request still counts as ONE against --rate-limit and --max-concurrent.
    // The neighbouring max_tokens is clamped to the context window; this was
    // not clamped at all (#1616).
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

    ctx.params.min_p = body.value("min_p", 0.0f);
    ctx.params.typical_p = body.value("typical_p", 1.0f);
    // Mild anti-repetition default; explicit request values still win.
    ctx.params.repetition_penalty = body.value("repetition_penalty", 1.05f);
    ctx.params.frequency_penalty = body.value("frequency_penalty", 0.0f);
    ctx.params.presence_penalty = body.value("presence_penalty", 0.0f);
    ctx.params.repeat_last_n = body.value("repeat_last_n", 0);
    ctx.params.dry_multiplier = body.value("dry_multiplier", 0.0f);
    ctx.params.dry_base = body.value("dry_base", 1.75f);
    ctx.params.dry_allowed_length = body.value("dry_allowed_length", 2);
    ctx.params.dry_penalty_last_n = body.value("dry_penalty_last_n", 0);
    ctx.params.mirostat = body.value("mirostat", 0);
    ctx.params.mirostat_tau = body.value("mirostat_tau", 5.0f);
    ctx.params.mirostat_eta = body.value("mirostat_eta", 0.1f);
    ctx.params.think_budget = body.value("think_budget", state.default_think_budget);

    // Parse stop sequences (string or array). OpenAI caps at 4 but Anthropic
    // /v1/messages does not, and its stop_sequences convert through this
    // parser — allow up to 16 and warn when truncating (the stop-scan
    // machinery handles arbitrary lists; max_stop_len derives from the vector).
    constexpr size_t kMaxStopSequences = 16;
    if (parse_stop_field(body, kMaxStopSequences, ctx.params.stop_sequences)) {
        fprintf(stderr, "warning: request sent %zu stop sequences; keeping the first %zu\n",
                body["stop"].size(), kMaxStopSequences);
    }
    ctx.params.max_stop_len = 0;
    for (const auto& s : ctx.params.stop_sequences)
        ctx.params.max_stop_len = std::max(ctx.params.max_stop_len, s.size());

    // Parse logprobs parameters
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
        } else if (fmt_type == "grammar") {
            // {"type":"grammar","grammar":"root ::= ..."} — a GBNF grammar the
            // whole reply must derive. "gbnf" is accepted as a spelling too,
            // since that is what the format is called everywhere else.
            const auto& rf = body["response_format"];
            if (rf.contains("grammar") && rf["grammar"].is_string())
                ctx.params.grammar = rf["grammar"].get<std::string>();
            else if (rf.contains("gbnf") && rf["gbnf"].is_string())
                ctx.params.grammar = rf["gbnf"].get<std::string>();
        } else if (fmt_type == "json_object") {
            ctx.params.json_mode = true;
        } else if (fmt_type == "json_schema") {
            ctx.params.json_mode = true;
            if (body["response_format"].contains("json_schema") &&
                body["response_format"]["json_schema"].is_object()) {
                auto& js = body["response_format"]["json_schema"];
                if (js.contains("schema") && js["schema"].is_object()) {
                    const auto& sch = js["schema"];
                    // Free-form object schema ({"type":"object"} without
                    // properties/enum) carries no structure the schema
                    // constrainer could enforce — its key phase would reject
                    // every token. Semantically this is json_object: leave
                    // json_schema_str empty so the whole request (scheduler
                    // included) takes the any-JSON constrainer path.
                    const bool free_form = sch.value("type", "") == "object" &&
                                           (!sch.contains("properties") || sch["properties"].empty()) &&
                                           !sch.contains("enum");
                    if (!free_form) {
                        ctx.params.json_schema_str = dump_safe(sch);
                    }
                }
            }
        }
    }

    // vLLM/SGLang spell it `guided_regex` at the top level; accept that too so
    // an existing client works unchanged. response_format wins if both appear.
    if (ctx.params.regex_pattern.empty() && body.contains("guided_regex") &&
        body["guided_regex"].is_string())
        ctx.params.regex_pattern = body["guided_regex"].get<std::string>();

    // Grammars have two established top-level spellings and no response_format
    // convention at all: llama.cpp takes `grammar`, vLLM takes
    // `guided_grammar`. Accept both, so a client written against either server
    // works here unchanged.
    if (ctx.params.grammar.empty() && body.contains("grammar") && body["grammar"].is_string())
        ctx.params.grammar = body["grammar"].get<std::string>();
    if (ctx.params.grammar.empty() && body.contains("guided_grammar") && body["guided_grammar"].is_string())
        ctx.params.grammar = body["guided_grammar"].get<std::string>();

    // Parse logit_bias: map of token_id (string) -> bias (float)
    if (body.contains("logit_bias") && body["logit_bias"].is_object()) {
        // Every entry costs a blocking device-to-host copy per decode step, so
        // the map size multiplies the cost of every token, not of the request
        // (#1617). Refuse rather than truncate: a silently dropped bias changes
        // the output without saying so.
        if (state.max_logit_bias > 0 && static_cast<int>(body["logit_bias"].size()) > state.max_logit_bias) {
            send_json_error(res, 400, "invalid_request_error",
                            "\"logit_bias\" has " + std::to_string(body["logit_bias"].size()) +
                                " entries, above the server limit of " +
                                std::to_string(state.max_logit_bias) + " (--max-logit-bias)");
            return false;
        }
        for (auto& [key, val] : body["logit_bias"].items()) {
            try {
                int32_t token_id = std::stoi(key);
                float bias = val.get<float>();
                ctx.params.logit_bias.emplace_back(token_id, bias);
            } catch (...) {
                // Skip invalid entries
            }
        }
    }

    // Parse stream_options for include_usage
    if (body.contains("stream_options") && body["stream_options"].is_object()) {
        ctx.params.include_usage = body["stream_options"].value("include_usage", false);
    }

    // Prompt KV pinning: Anthropic cache_control (mapped to "cache_prompt"
    // by anthropic_to_openai_body) or a direct llama.cpp-style field.
    ctx.params.cache_prompt = body.value("cache_prompt", false);
    if (body.contains("cache_prefix_messages") && body["cache_prefix_messages"].is_number_integer())
        ctx.params.cache_prefix_messages = body["cache_prefix_messages"].get<int>();

    // Per-request speculative-decode override (imp extension). Absent → leave
    // tri-state at -1 (server default). Present bool → force on/off.
    if (body.contains("speculative") && body["speculative"].is_boolean())
        ctx.params.spec_ngram_override = body["speculative"].get<bool>() ? 1 : 0;

    // OpenAI Predicted Outputs: {"prediction": {"type": "content", "content":
    // string | [{"type":"text","text":...}...]}}. The text is a draft hint —
    // it never changes the output, only speeds up verify-accept — so unknown
    // shapes are ignored rather than rejected.
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

    // tools + response_format=json_schema/json_object: the engine-side gate
    // stays "no-mask" through tool-call bodies (see ConstraintManager::prepare
    // and PreambleGate::configure_with_tools), so we keep both signals set
    // and the gate decides at runtime which path the model takes. Tool-call
    // dialect comes from tpl_family, captured below into the request.

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

    // Enforced tool calling (#1002) is collected POST-snapshot in
    // handlers_chat_core (collect_tool_enforcement): the request may auto-load
    // or name a different model, and the constraint dialect must come from the
    // template that will actually render this prompt — the parse-time family
    // above is a pre-load best guess (fine for message flattening, wrong to
    // bake a grammar from).

    // Convert JSON messages to ChatMessage vector, extracting image data if present
    for (const auto& msg : messages) {
        std::string role = msg.value("role", "user");

        if (role == "tool") {
            // Tool response message — format for the model
            std::string content = format_tool_response(ctx.snap.tpl_family, msg);
            // Gemma's chat-template skips standalone role=tool messages and
            // expects tool_response markers to be glued onto the assistant
            // message that produced the call. Append to previous assistant
            // entry instead of pushing a fresh ChatMessage; ChatML/Llama3
            // templates render standalone tool messages so keep the push.
            if (ctx.snap.tpl_family == imp::ChatTemplateFamily::GEMMA && !ctx.params.chat_msgs.empty() &&
                ctx.params.chat_msgs.back().role == "assistant") {
                ctx.params.chat_msgs.back().content += content;
            } else {
                ctx.params.chat_msgs.push_back({"tool", content});
            }
        } else if (role == "assistant" && msg.contains("tool_calls")) {
            // Assistant message with tool_calls — reconstruct model output
            // format. On XML-dialect templates (Qwen-Coder) prior calls must
            // replay in the XML shape the model itself emits, not the ChatML
            // JSON body — a JSON replay teaches the model the wrong dialect
            // for its NEXT call, exactly what the armed XML grammar forbids.
            // (Parse-time dialect: flattening happens pre-model-load; a
            // cross-model first request may replay in the previous template's
            // dialect — moving message conversion post-snapshot is the deeper
            // fix, tracked in the PR.)
            std::string content_str;
            if (msg.contains("content") && !msg["content"].is_null()) {
                content_str = msg["content"].get<std::string>();
            }
            std::string reconstructed = reconstruct_tool_call_output(ctx.snap.tpl_family, msg["tool_calls"],
                                                                     content_str, tool_xml_dialect);
            ctx.params.chat_msgs.push_back({"assistant", reconstructed});
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
                    // One slot per part, appended before it is filled: if the
                    // fetch below fails the request is rejected, so a half-read
                    // list never reaches the prompt builder.
                    ctx.params.images.emplace_back();
                    std::vector<uint8_t>& image_bytes = ctx.params.images.back();
                    if (url.rfind("data:", 0) == 0) {
                        // Data URI: data:image/...;base64,...
                        auto comma = url.find(',');
                        if (comma != std::string::npos) {
                            image_bytes = base64_decode(url.substr(comma + 1));
                        }
                    } else if (url.rfind("http://", 0) == 0 || url.rfind("https://", 0) == 0) {
                        // #1610: this used to build an httplib client straight
                        // from the request's host, follow redirects, and buffer
                        // whatever came back. That is an SSRF primitive on an
                        // endpoint that is unauthenticated by default: the
                        // caller picks the host and port and the server has
                        // reach the caller does not. Off by default now, and
                        // bounded when on. See image_fetch.h.
                        auto fetched = imp_server::fetch_remote_image(url,
                                                                      state.default_args.allow_remote_images);
                        if (fetched.ok) {
                            image_bytes = std::move(fetched.bytes);
                        } else {
                            IMP_LOG_WARN("image_url not fetched: %s", fetched.detail.c_str());
                        }
                    }
                    // A scheme we do not fetch (file://, plain paths) leaves the
                    // slot empty, same as a failed request. Both are refused
                    // below rather than silently dropping a picture.
                    //
                    // One string for every cause, and it does NOT echo the URL.
                    // Distinguishable errors turned this into a port scanner of
                    // the server's own network: "connection refused" and "200
                    // with unparseable bytes" read differently from outside.
                    //
                    // The two variants below differ by SERVER CONFIGURATION,
                    // never by what the URL named, so neither tells a caller
                    // anything about the destination.
                    if (image_bytes.empty() && ctx.params.image_error.empty()) {
                        const bool remote = url.rfind("http://", 0) == 0 || url.rfind("https://", 0) == 0;
                        ctx.params.image_error =
                            (remote && !state.default_args.allow_remote_images)
                                ? "could not read image_url: remote URLs are disabled on this "
                                  "server; send a data: URI, or start it with --allow-remote-images"
                                : "could not read image_url";
                    }
                }
            }
            ctx.params.chat_msgs.push_back({role, text_parts});
        } else {
            std::string content;
            if (msg.contains("content") && !msg["content"].is_null()) {
                content = msg["content"].get<std::string>();
            }
            ctx.params.chat_msgs.push_back({role, content});
        }
    }

    // Log request received (structured)
    ctx.req_id = make_completion_id(state);
    fprintf(stderr, "[%s] chat/completions: prompt_msgs=%zu stream=%s max_tokens=%d temp=%.2f\n",
            ctx.req_id.c_str(), messages.size(), ctx.params.stream ? "true" : "false", ctx.params.max_tokens,
            ctx.params.temperature);

    // Validate model field (required per OpenAI spec)
    ctx.params.requested_model = body.value("model", "");
    if (ctx.params.requested_model.empty()) {
        res.status = 400;
        json err = {{"error", {{"message", "\"model\" is required"}, {"type", "invalid_request_error"}}}};
        res.set_content(dump_safe(err), "application/json");
        return false;
    }

    // Parse enable_thinking (only meaningful for think models; checked in orchestrator)
    ctx.params.enable_thinking_requested = body.value("enable_thinking", false);
    ctx.params.enable_thinking_set = body.contains("enable_thinking") && body["enable_thinking"].is_boolean();

    // Per-request LoRA adapter selection ("lora": "<name>"; absent/"" = base).
    ctx.params.lora_name = body.value("lora", std::string());

    return true;
}
