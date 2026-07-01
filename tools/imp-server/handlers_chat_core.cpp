// AUTO-SPLIT from handlers.cpp (verbatim move; see handlers_internal.h).
// Shared chat-completion machinery: request-log, body parse, state snapshot +
// tokenize, and the non-streaming chat response builder. Used by the OpenAI
// chat endpoint (handlers_chat.cpp) and the Anthropic messages endpoint
// (handlers_messages.cpp).

#include "handlers.h"
#include "handlers_internal.h"
#include "utils.h"
#include "tool_call.h"
#include "anthropic.h"
#include "stream_pipeline.h"

#include "api/imp_internal.h"
#include "vision/image_processor.h"
#include "runtime/request.h"
#include "memory/kv_cache.h"
#include "model/hf_hub.h"
#include "runtime/config.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <functional>
#include <vector>

#include <cuda_runtime.h>

// Set true on the calling thread when handle_messages is delegating to
// handle_chat_completions via a shim — suppresses inner request-log entries
// so the Anthropic call only logs once at the outer handler.
thread_local bool g_in_anthropic_shim = false;

// Write one JSONL line capturing this request: timing, endpoint, raw client
// body, token counts, finish reason, and (for non-streaming) the response.
// Streaming responses pass an empty `response_body` since per-chunk text is
// not accumulated.
void log_request_jsonl(ServerState& state, bool skip,
                              const std::chrono::system_clock::time_point& t_start,
                              const std::string& req_id, const std::string& endpoint,
                              const std::string& client_ip, const std::string& raw_body,
                              double latency_ms, int prompt_tokens, int completion_tokens,
                              const char* finish_reason, const json& response_body) {
    if (skip || !state.request_logger.enabled)
        return;
    json record;
    record["ts_ms"] =
        std::chrono::duration_cast<std::chrono::milliseconds>(t_start.time_since_epoch()).count();
    record["req_id"] = req_id;
    record["endpoint"] = endpoint;
    record["client_ip"] = client_ip;
    record["latency_ms"] = latency_ms;
    record["prompt_tokens"] = prompt_tokens;
    record["completion_tokens"] = completion_tokens;
    record["finish_reason"] = finish_reason ? finish_reason : "";
    try {
        record["request"] = json::parse(raw_body);
    } catch (...) {
        record["request"] = raw_body;
    }
    record["response"] = response_body;
    state.request_logger.log(record);
}

// Parses request body, validates params, builds chat_msgs from messages array.
// Populates ctx.params, ctx.log_*, ctx.req_id, ctx.snap.tpl_family (early best-
// effort snapshot used to format tool-role messages in the conversion loop).
// On parse/validation failure: sets res with 400 + error JSON and returns false.
// On success: returns true; caller proceeds to state snapshot + tokenize.
bool parse_chat_request_params(
    const httplib::Request& req,
    httplib::Response& res,
    ServerState& state,
    ChatRequestContext& ctx)
{
    // Capture inputs for opt-in JSONL request logging. Only used when
    // state.request_logger.enabled and the call is not an inner shim.
    ctx.t_log_start = std::chrono::system_clock::now();
    ctx.log_endpoint = req.path;
    ctx.log_client_ip = req.get_header_value("X-Forwarded-For");
    if (ctx.log_client_ip.empty())
        ctx.log_client_ip = req.remote_addr;
    ctx.log_raw_body = req.body;
    ctx.log_skip = g_in_anthropic_shim;

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

    // Parse response_format for JSON mode / JSON Schema
    if (body.contains("response_format") && body["response_format"].is_object()) {
        std::string fmt_type = body["response_format"].value("type", "text");
        if (fmt_type == "json_object") {
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
                                           (!sch.contains("properties") ||
                                            sch["properties"].empty()) &&
                                           !sch.contains("enum");
                    if (!free_form) {
                        ctx.params.json_schema_str = dump_safe(sch);
                    }
                }
            }
        }
    }

    // Parse logit_bias: map of token_id (string) -> bias (float)
    if (body.contains("logit_bias") && body["logit_bias"].is_object()) {
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

    // Per-request speculative-decode override (imp extension). Absent → leave
    // tri-state at -1 (server default). Present bool → force on/off.
    if (body.contains("speculative") && body["speculative"].is_boolean())
        ctx.params.spec_ngram_override = body["speculative"].get<bool>() ? 1 : 0;

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
    {
        std::lock_guard<std::timed_mutex> lock(state.mtx);
        ctx.snap.tpl_family = state.have_template ? state.chat_tpl.family() : imp::ChatTemplateFamily::CHATML;
    }

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
            // Assistant message with tool_calls — reconstruct model output format
            std::string content_str;
            if (msg.contains("content") && !msg["content"].is_null()) {
                content_str = msg["content"].get<std::string>();
            }
            std::string reconstructed = reconstruct_tool_call_output(ctx.snap.tpl_family, msg["tool_calls"],
                                                                     content_str);
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
                    if (url.rfind("data:", 0) == 0) {
                        // Data URI: data:image/...;base64,...
                        auto comma = url.find(',');
                        if (comma != std::string::npos) {
                            ctx.params.image_data = base64_decode(url.substr(comma + 1));
                        }
                    } else if (url.rfind("http://", 0) == 0 || url.rfind("https://", 0) == 0) {
                        // Remote URL: fetch image via HTTP
                        // Parse URL into host + path
                        bool is_https = (url.rfind("https://", 0) == 0);
                        std::string rest = url.substr(is_https ? 8 : 7);
                        auto slash = rest.find('/');
                        std::string host = (slash != std::string::npos) ? rest.substr(0, slash) : rest;
                        std::string path_str = (slash != std::string::npos) ? rest.substr(slash) : "/";
                        if (is_https) {
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
                            httplib::SSLClient cli(host);
                            cli.set_follow_location(true);
                            cli.set_connection_timeout(10);
                            auto img_res = cli.Get(path_str);
                            if (img_res && img_res->status == 200) {
                                ctx.params.image_data.assign(img_res->body.begin(), img_res->body.end());
                            }
#endif
                        } else {
                            httplib::Client cli(host);
                            cli.set_follow_location(true);
                            cli.set_connection_timeout(10);
                            auto img_res = cli.Get(path_str);
                            if (img_res && img_res->status == 200) {
                                ctx.params.image_data.assign(img_res->body.begin(), img_res->body.end());
                            }
                        }
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

    // Parse enable_thinking (only meaningful for think models; checked in orchestrator)
    ctx.params.enable_thinking_requested = body.value("enable_thinking", false);
    ctx.params.enable_thinking_set = body.contains("enable_thinking") && body["enable_thinking"].is_boolean();

    // Per-request LoRA adapter selection ("lora": "<name>"; absent/"" = base).
    ctx.params.lora_name = body.value("lora", std::string());

    return true;
}

// Acquires state.mtx lock, snapshots engine state into ctx.snap, sets up
// tool defs / vision lock / thinking detection, tokenizes the prompt with
// the chat template, validates prompt length, clamps max_tokens to remaining
// context, and starts timing. Returns true if OK; sets res with 400/503 and
// returns false on failure (model not loaded, prompt too long, vision lock
// timeout, image processing failure).
bool snapshot_state_and_tokenize_(
    httplib::Response& res,
    ServerState& state,
    ChatRequestContext& ctx)
{
    // Snapshot all state fields needed for request processing under lock.
    // This protects against concurrent model load/unload invalidating pointers.
    {
        std::lock_guard<std::timed_mutex> lock(state.mtx);
        if (!ensure_model_loaded(state, ctx.params.requested_model, res))
            return false;
        ctx.snap.tok = state.tok;
        ctx.snap.chat_tpl = state.chat_tpl;
        ctx.snap.have_template = state.have_template;
        ctx.snap.model_name = state.model_name;
        ctx.snap.is_think_model = state.is_think_model;
        ctx.snap.think_start_id = state.think_start_id;
        ctx.snap.think_end_id = state.think_end_id;
        ctx.snap.channel_open_id = state.channel_open_id;
        ctx.snap.channel_close_id = state.channel_close_id;
        ctx.snap.channel_newline_id = state.channel_newline_id;
        ctx.snap.max_seq_len = state.max_seq_len;
        ctx.snap.tpl_family = ctx.snap.have_template ? ctx.snap.chat_tpl.family() : imp::ChatTemplateFamily::CHATML;
        if (ctx.snap.have_template)
            ctx.snap.stop_token_ids = ctx.snap.chat_tpl.stop_token_ids();
        // Provisionally add <think> as a stop token. Removed below if the
        // request enables thinking. Without this, think-trained models at high
        // temp can hallucinate phantom turns ("Human\n<think>...").
        if (state.think_start_id >= 0) {
            ctx.snap.stop_token_ids.push_back(state.think_start_id);
        }
        ctx.snap.has_vision_request = !ctx.params.image_data.empty() && state.ctx && state.ctx->engine->has_vision();
    }

    // Channel models (Gemma-4) are more susceptible to sampling-driven
    // degeneration on casual prompts than DeepSeek-style reasoning models.
    // If the caller didn't specify a sampler parameter, tighten the default
    // to suppress the tail of the distribution. Qwen3 / DeepSeek / non-channel
    // models retain the 0.95 / 40 / 1.0 defaults.
    if (ctx.snap.channel_open_id >= 0) {
        if (!ctx.params.top_p_explicit)
            ctx.params.top_p = 0.9f;
        if (!ctx.params.top_k_explicit)
            ctx.params.top_k = 20;
        if (!ctx.params.rep_pen_explicit)
            ctx.params.repetition_penalty = 1.05f;
    }

    // Build tool definitions for Jinja2-native tool calling
    if (ctx.params.has_tools && ctx.snap.have_template && ctx.snap.chat_tpl.supports_tools()) {
        for (const auto& t : ctx.params.tools) {
            if (t.contains("function") && t["function"].is_object()) {
                imp::ToolFunction tf;
                tf.name = t["function"].value("name", "");
                tf.description = t["function"].value("description", "");
                if (t["function"].contains("parameters")) {
                    tf.parameters_json = dump_safe(t["function"]["parameters"]);
                }
                ctx.snap.tool_defs.push_back(std::move(tf));
            }
        }
    }
    // tools_via_jinja tracks whether we'll attempt the Jinja2 tools path
    ctx.snap.tools_via_jinja = !ctx.snap.tool_defs.empty();

    // Vision (per-request, F-A5): CPU-preprocess the image into
    // ctx.snap.vision_image — no engine pause, no global image. Each
    // request-build site copies it to req->image; the batch worker encodes +
    // binds it per-request on admission, so a vision request batches like text.
    // Soft-token placeholders are injected by apply_with_image() below.
    if (ctx.snap.has_vision_request) {
        auto img = std::make_shared<imp::ImageData>();
        if (!state.ctx->engine->preprocess_image(ctx.params.image_data.data(),
                                                  ctx.params.image_data.size(), *img)) {
            res.status = 400;
            json error = {
                {"error", {{"message", "Failed to process image"}, {"type", "invalid_request_error"}}}};
            res.set_content(dump_safe(error), "application/json");
            return false;
        }
        ctx.snap.vision_image = std::move(img);
    }

    // Thinking mode default: ON for think models in plain chat. These models
    // are trained with the <think> prefix; serving them without it produces
    // bare reasoning that cannot be separated and leaks into user-visible
    // content ("Okay, let's see. The user is asking..." as the answer — the
    // recurring think-leak bug class). Exceptions, where entering reasoning
    // mode breaks the requested output format: structured output (json_mode)
    // and tool calls keep the old default OFF. An explicit "enable_thinking"
    // in the request always wins in both directions.
    // Template evidence guard: vocab-level <think> specials alone are not
    // proof of a think-trained model — Qwen3-*-Instruct-2507 ships the Qwen3
    // vocab (incl. <think>) but never opens a think block; defaulting it to
    // thinking traps the entire answer in reasoning_content (content "").
    // Default ON only when the chat template itself references thinking.
    // Models without a Jinja template keep the previous default (no evidence
    // either way); an explicit "enable_thinking" still wins in both cases.
    // Only a present-but-silent Jinja template counts as evidence AGAINST
    // thinking; hardcoded families / templateless runs keep the old default.
    const bool template_think_evidence = !ctx.snap.have_template ||
                                         !ctx.snap.chat_tpl.has_jinja() ||
                                         ctx.snap.chat_tpl.mentions_thinking();
    const bool thinking_default = ctx.snap.is_think_model && template_think_evidence &&
                                  !ctx.params.json_mode && !ctx.params.has_tools;
    const bool want_thinking = ctx.params.enable_thinking_set
                                   ? ctx.params.enable_thinking_requested
                                   : thinking_default;
    // think_budget is the fraction of max_tokens reserved for reasoning;
    // think_budget <= 0 means "no reasoning headroom" → disable thinking entirely
    // (documented "0 = disabled"). Folding it into enable_thinking keeps the two
    // flags consistent: without this, budget=0 left thinking ON yet never armed
    // the force-close, so the model reasoned to max_tokens and returned empty
    // content (#752). The Anthropic "disabled" path already zeroes the budget.
    const bool budget_disables_thinking = ctx.params.think_budget <= 0.0f;
    ctx.snap.enable_thinking = ctx.snap.is_think_model && ctx.snap.think_start_id >= 0 &&
                               want_thinking && !budget_disables_thinking;
    ctx.snap.suppress_thinking =
        ctx.snap.is_think_model && !ctx.snap.enable_thinking && budget_disables_thinking;

    // If thinking IS enabled, remove the provisional <think> stop token.
    if (ctx.snap.enable_thinking && ctx.snap.think_start_id >= 0) {
        auto& ids = ctx.snap.stop_token_ids;
        ids.erase(std::remove(ids.begin(), ids.end(), ctx.snap.think_start_id), ids.end());
    }

    // Guard against hallucinated turn boundaries ("Human\n") that thinking
    // models emit at high temperature. Only inject if the caller didn't
    // already provide stop sequences (respect user intent).
    if (ctx.snap.is_think_model && ctx.params.stop_sequences.empty()) {
        ctx.params.stop_sequences.push_back("\nHuman");
    }

    // Tokenize with chat template (with image tokens if vision is active)
    if (ctx.snap.have_template && ctx.snap.has_vision_request) {
        ctx.snap.tokens = ctx.snap.chat_tpl.apply_with_image(*ctx.snap.tok, ctx.params.chat_msgs, 256, ctx.snap.suppress_thinking);
    } else if (ctx.snap.have_template && ctx.snap.tools_via_jinja) {
        std::string tc_str = ctx.params.tool_choice.is_string() ? ctx.params.tool_choice.get<std::string>() : "auto";
        ctx.snap.tokens = ctx.snap.chat_tpl.apply_with_tools(*ctx.snap.tok, ctx.params.chat_msgs, ctx.snap.tool_defs, tc_str, ctx.snap.suppress_thinking);
        // If Jinja2 tools render failed, fall back to text-based tool prompt injection
        if (ctx.snap.tokens.empty()) {
            IMP_LOG_INFO("Jinja2 tools path failed, falling back to text-based tool prompt");
            std::string tool_prompt = build_tool_prompt(ctx.snap.tpl_family, ctx.params.tools, ctx.params.tool_choice);
            if (!tool_prompt.empty()) {
                bool found_system = false;
                for (auto& m : ctx.params.chat_msgs) {
                    if (m.role == "system") {
                        m.content += tool_prompt;
                        found_system = true;
                        break;
                    }
                }
                if (!found_system) {
                    std::string sys = ctx.snap.chat_tpl.default_system_message();
                    if (sys.empty())
                        sys = "You are a helpful assistant.";
                    sys += tool_prompt;
                    ctx.params.chat_msgs.insert(ctx.params.chat_msgs.begin(), {"system", sys});
                }
            }
            ctx.snap.tokens = ctx.snap.chat_tpl.apply(*ctx.snap.tok, ctx.params.chat_msgs, ctx.snap.suppress_thinking);
        }
    } else if (ctx.snap.have_template) {
        // No tools, or no Jinja2 support — inject text-based tool prompt if tools present
        if (ctx.params.has_tools) {
            std::string tool_prompt = build_tool_prompt(ctx.snap.tpl_family, ctx.params.tools, ctx.params.tool_choice);
            if (!tool_prompt.empty()) {
                bool found_system = false;
                for (auto& m : ctx.params.chat_msgs) {
                    if (m.role == "system") {
                        m.content += tool_prompt;
                        found_system = true;
                        break;
                    }
                }
                if (!found_system) {
                    std::string sys = ctx.snap.chat_tpl.default_system_message();
                    if (sys.empty())
                        sys = "You are a helpful assistant.";
                    sys += tool_prompt;
                    ctx.params.chat_msgs.insert(ctx.params.chat_msgs.begin(), {"system", sys});
                }
            }
        }
        ctx.snap.tokens = ctx.snap.chat_tpl.apply(*ctx.snap.tok, ctx.params.chat_msgs, ctx.snap.suppress_thinking);
    } else {
        // Concatenate all message content as raw text
        std::string raw;
        for (const auto& m : ctx.params.chat_msgs)
            raw += m.content + "\n";
        ctx.snap.tokens = ctx.snap.tok->encode(raw);
    }

    // Detect chat-template-injected <think> prefix (Qwen3 / Qwen3.5 / Qwen3.6
    // / DeepSeek-R1 add `<think>\n` via add_generation_prompt by default). When
    // present, the model output starts mid-thinking with no opener — only a
    // closing `</think>` mid-stream. Matches vLLM's qwen3 reasoning_parser
    // auto-detection (see vllm/reasoning/qwen3_reasoning_parser.py docstring).
    // Treating these models as thinking-enabled lets the SSE stream emit
    // `reasoning_content` chunks until `</think>` is seen, then `content`.
    //
    // Detection is done over decoded text (not token-ID equality) because
    // Qwen3.6 ships `<think>`/`</think>` as `added_tokens` with `special=False`,
    // so the BPE tokenizer breaks them into 3 pieces (`<`, `think`, `>`)
    // rather than the single special-token id. vLLM's parser sidesteps this
    // by promoting them at AutoTokenizer load; imp's tokenizer doesn't, so
    // we match on the rendered string instead.
    auto prompt_tail_contains = [&](const char* needle, int max_tail_tokens) -> bool {
        int n = static_cast<int>(ctx.snap.tokens.size());
        int start = std::max(0, n - max_tail_tokens);
        std::string tail_text;
        for (int i = start; i < n; ++i) {
            tail_text += ctx.snap.tok->decode_token(ctx.snap.tokens[i]);
        }
        return tail_text.find(needle) != std::string::npos;
    };
    // No special-token requirement here: Nemotron-style models think at TEXT
    // level ("<think>" renders as plain text pieces, "</think>" closes it) —
    // when their chat template injects the prefix, the output is reasoning
    // from token 0 and must flow into reasoning_content, not content.
    //
    // Only an OPEN think prefix counts: when thinking is suppressed, Qwen3's
    // template injects a *closed* empty block `<think>\n\n</think>\n\n` (so the
    // model answers directly). That tail contains "<think>" too — re-enabling on
    // it would defeat suppression entirely (the model thinks despite the closed
    // block). Require "<think>" present AND no matching "</think>" in the tail.
    // Window 16 (not 8) so both tags of the adjacent closed block fall inside
    // the same tail — otherwise "<think>" could be in-window while "</think>"
    // just falls off, mis-reading a closed block as an open prefix.
    if (!ctx.snap.enable_thinking) {
        if (prompt_tail_contains("<think>", 16) && !prompt_tail_contains("</think>", 16)) {
            ctx.snap.enable_thinking = true;
        }
    }

    // Append <think>\n to trigger reasoning mode (matches llama.cpp behavior).
    // Without this prefix, think-trained models produce degenerate output.
    // Skip if the chat template already added it (Qwen3.x default path).
    if (ctx.snap.enable_thinking && ctx.snap.think_start_id >= 0) {
        if (!prompt_tail_contains("<think>", 8)) {
            ctx.snap.tokens.push_back(ctx.snap.think_start_id);
            // Append newline after <think> — the model expects "\n" before reasoning
            auto nl_ids = ctx.snap.tok->encode("\n");
            ctx.snap.tokens.insert(ctx.snap.tokens.end(), nl_ids.begin(), nl_ids.end());
        }
    }

    ctx.snap.n_prompt_tokens = static_cast<int>(ctx.snap.tokens.size());

    // Server-side input-token limit (--max-input-tokens). Reject before
    // prefill so an oversized prompt never reaches the engine.
    if (state.max_input_tokens > 0 && ctx.snap.n_prompt_tokens > state.max_input_tokens) {
        res.status = 400;
        json error = {{"error",
                       {{"message", "Prompt exceeds max input tokens (" +
                                        std::to_string(ctx.snap.n_prompt_tokens) + " > " +
                                        std::to_string(state.max_input_tokens) + ")"},
                        {"type", "invalid_request_error"}}}};
        res.set_content(dump_safe(error), "application/json");
        return false;
    }

    // Validate prompt length against context window
    if (ctx.snap.n_prompt_tokens >= ctx.snap.max_seq_len) {
        res.status = 400;
        json error = {{"error",
                       {{"message", "Prompt exceeds context window (" + std::to_string(ctx.snap.n_prompt_tokens) +
                                        " tokens >= " + std::to_string(ctx.snap.max_seq_len) + " max)"},
                        {"type", "invalid_request_error"}}}};
        res.set_content(dump_safe(error), "application/json");
        return false;
    }

    // Per-request LoRA selection (#522): swap the engine-global adapter
    // before generation. Single-user semantics — the swap re-captures
    // decode graphs on the next step, so back-to-back requests with
    // different adapters work; concurrent mixed-adapter batches are out of
    // scope (imp is batch=1-first by mission).
    {
        int32_t want = 0;
        if (!ctx.params.lora_name.empty()) {
            auto it = state.lora_ids.find(ctx.params.lora_name);
            if (it == state.lora_ids.end()) {
                res.status = 400;
                json error = {{"error",
                               {{"message", "Unknown LoRA adapter '" + ctx.params.lora_name +
                                                "' (load at startup via --lora NAME=PATH)"},
                                {"type", "invalid_request_error"}}}};
                res.set_content(dump_safe(error), "application/json");
                return false;
            }
            want = it->second;
        }
        imp_lora_set(state.ctx, want);
    }

    // Clamp max_tokens to remaining context window
    int remaining = ctx.snap.max_seq_len - ctx.snap.n_prompt_tokens;
    if (ctx.params.max_tokens > remaining)
        ctx.params.max_tokens = remaining;

    // Start timing
    ctx.t_start = std::chrono::high_resolution_clock::now();

    return true;
}


// Non-streaming chat completion: run n_completions independent generations
// sequentially via the batching engine, build the choices array with
// reasoning_content / tool_calls / logprobs as appropriate, send a single
// JSON response. Caller has already submitted server_req via state.batching.
void nonstream_chat_response_(
    httplib::Response& res,
    ServerState& state,
    ChatRequestContext& ctx,
    std::shared_ptr<imp::Request>& imp_req,
    std::shared_ptr<ServerRequest>& server_req,
    const std::vector<int32_t>& saved_tokens,
    const std::string& comp_id,
    int64_t created)
{
    // Helper to create an imp::Request with the given completion index
    auto make_imp_request = [&](int completion_idx) {
        auto req = std::make_shared<imp::Request>();
        req->image = ctx.snap.vision_image;  // per-request vision (null for text)
        req->input_tokens = saved_tokens;
        req->max_tokens = ctx.params.max_tokens;
        req->temperature = ctx.params.temperature;
        req->top_p = ctx.params.top_p;
        req->top_k = ctx.params.top_k;
        req->seed = (ctx.params.seed != -1) ? ctx.params.seed + completion_idx : -1;
        req->pin_kv_prefix = ctx.params.cache_prompt;
        req->spec_ngram_override = ctx.params.spec_ngram_override;
        req->min_p = ctx.params.min_p;
        req->typical_p = ctx.params.typical_p;
        req->repetition_penalty = ctx.params.repetition_penalty;
        req->frequency_penalty = ctx.params.frequency_penalty;
        req->presence_penalty = ctx.params.presence_penalty;
        req->repeat_last_n = ctx.params.repeat_last_n;
        req->dry_multiplier = ctx.params.dry_multiplier;
        req->dry_base = ctx.params.dry_base;
        req->dry_allowed_length = ctx.params.dry_allowed_length;
        req->dry_penalty_last_n = ctx.params.dry_penalty_last_n;
        req->mirostat = ctx.params.mirostat;
        req->mirostat_tau = ctx.params.mirostat_tau;
        req->mirostat_eta = ctx.params.mirostat_eta;
        req->logprobs = ctx.params.req_logprobs;
        req->top_logprobs = ctx.params.top_logprobs;
        req->json_mode = ctx.params.json_mode;
        req->json_schema = ctx.params.json_schema_str;
        req->has_tools = ctx.params.has_tools;
        req->tpl_family = ctx.snap.tpl_family;
        req->logit_bias = ctx.params.logit_bias;
        req->think_budget = ctx.params.think_budget;
        // Generation starts INSIDE the think block when the prompt carries the
        // <think> prefix (template-injected or server-appended). Without this
        // the engine's think-budget enforcement never sees an opener in the
        // output, counts zero reasoning tokens, and lets the model think until
        // max_tokens (content empty, finish=length).
        req->started_in_think = ctx.snap.enable_thinking;
        req->in_think_block = ctx.snap.enable_thinking;
        req->status = imp::RequestStatus::PENDING;
        return req;
    };

    // Non-streaming: decode all tokens, return complete response
    // For n > 1, run multiple independent generations sequentially
    json choices = json::array();
    int total_output_tokens = 0;

    for (int ci = 0; ci < ctx.params.n_completions; ci++) {
        // For subsequent completions, create a new request and submit it
        if (ci > 0) {
            imp_req = make_imp_request(ci);
            server_req = std::make_shared<ServerRequest>();
            server_req->request = imp_req;
            {
                std::lock_guard<std::timed_mutex> lock(state.mtx);
                if (!state.batching || !state.batching->is_running()) {
                    break;
                }
                state.batching->submit(server_req);
            }
        }

        auto active_req = server_req->request;
        std::vector<int32_t> output_ids;
        const char* finish = nullptr;
        std::string output_text;  // accumulated output for stop matching

        auto ns_request_start = std::chrono::steady_clock::now();
        for (;;) {
            // Check request timeout
            if (state.request_timeout > 0) {
                auto elapsed = std::chrono::steady_clock::now() - ns_request_start;
                if (elapsed > std::chrono::seconds(state.request_timeout)) {
                    server_req->cancel();
                    finish = "length";
                    break;
                }
            }

            // Read next token from the batching engine
            TokenEvent evt{};
            if (!server_req->pop_token(evt)) {
                continue;  // timeout — loop back to check request timeout
            }

            if (evt.token_id < 0) {
                finish = evt.finish_reason ? evt.finish_reason : "stop";
                break;
            }

            int32_t token = evt.token_id;

            // Silently drop structural stop tokens that slipped through.
            // The engine's think-block implicit-close passes ONE EOS-like
            // token through to recover from empty thinking; it must not
            // appear as user-visible content.
            if (!evt.is_last) {
                bool is_structural_stop = (token == ctx.snap.tok->eos_id());
                if (!is_structural_stop && ctx.snap.have_template) {
                    for (int32_t stop_id : ctx.snap.stop_token_ids) {
                        if (token == stop_id) {
                            is_structural_stop = true;
                            break;
                        }
                    }
                }
                if (is_structural_stop)
                    continue;
            }

            // Check stop conditions
            if (evt.is_last) {
                if (token == ctx.snap.tok->eos_id()) {
                    finish = evt.finish_reason ? evt.finish_reason : "stop";
                    break;
                }
                bool is_stop = false;
                if (ctx.snap.have_template) {
                    for (int32_t stop_id : ctx.snap.stop_token_ids) {
                        if (token == stop_id) {
                            is_stop = true;
                            break;
                        }
                    }
                }
                if (is_stop) {
                    finish = evt.finish_reason ? evt.finish_reason : "stop";
                    break;
                }
                finish = evt.finish_reason ? evt.finish_reason : "length";
            }

            output_ids.push_back(token);

            // Check text-level stop sequences
            if (!ctx.params.stop_sequences.empty()) {
                output_text += ctx.snap.tok->decode_token(token);
                bool stop_found = false;
                for (const auto& stop : ctx.params.stop_sequences) {
                    auto pos = output_text.find(stop);
                    if (pos != std::string::npos) {
                        output_text = output_text.substr(0, pos);
                        stop_found = true;
                        break;
                    }
                }
                if (stop_found) {
                    finish = "stop";
                    break;
                }
            }

            // Break after processing the last non-EOS token
            if (finish)
                break;
        }

        if (!finish)
            finish = "length";

        int n_output_tokens = static_cast<int>(output_ids.size());
        total_output_tokens += n_output_tokens;
        std::string content = !ctx.params.stop_sequences.empty() ? output_text : ctx.snap.tok->decode(output_ids);

        // Extract reasoning content (DeepSeek format) or strip think blocks.
        // enable_thinking also covers text-level thinkers (Nemotron) whose
        // template injects "<think>" as plain text — is_think_model is false
        // but the output is reasoning until the literal "</think>".
        std::string reasoning_content;
        if (ctx.snap.tpl_family == imp::ChatTemplateFamily::HARMONY) {
            // gpt-oss Harmony: split the <|channel|>analysis|final<|message|>…
            // blocks so the analysis channel becomes reasoning_content and the
            // final channel becomes the answer. Without this the raw Harmony
            // markup leaks verbatim into content (#760).
            auto segs = split_harmony_channels(content);
            content = std::move(segs.content);
            if (state.default_args.reasoning_format != "none")
                reasoning_content = std::move(segs.reasoning);
        } else if ((ctx.snap.is_think_model || ctx.snap.enable_thinking) &&
                   state.default_args.reasoning_format == "deepseek") {
            // Generation that started inside an injected <think> prefix
            // (chat-template or server-appended; see prompt_tail_contains
            // above) carries no opener in its output. If it also never
            // reached </think> — budget exhausted mid-think, or the model
            // stopped while reasoning — the WHOLE text is reasoning.
            // extract_reasoning() can't tell that from text alone and would
            // spill it into user-visible content (the streaming path gets
            // this right via its in-think state machine).
            if (ctx.snap.enable_thinking && content.find("</think>") == std::string::npos &&
                content.find("<think>") == std::string::npos) {
                reasoning_content = std::move(content);
                content.clear();
            } else {
                auto [reasoning, cleaned] = extract_reasoning(content);
                reasoning_content = reasoning;
                content = cleaned;
            }
        } else if (ctx.snap.is_think_model && state.default_args.reasoning_format != "none") {
            strip_think_block(content);
        }

        // Gemma-4 channel headers: structural "<|channel>NAME[<channel|>]…"
        // wraps both the chain-of-thought and the user-facing answer. Split
        // them so "thought" content goes to reasoning_content (OpenAI-
        // compat) and "final" content stays in content. Falls back to
        // strip-only if the request asked reasoning_format=none.
        if (ctx.snap.channel_open_id >= 0) {
            if (state.default_args.reasoning_format == "none") {
                strip_channel_headers(content);
            } else {
                auto segs = split_channel_segments(content);
                if (!segs.reasoning.empty() && reasoning_content.empty()) {
                    reasoning_content = std::move(segs.reasoning);
                }
                content = std::move(segs.content);
            }
        }

        // Build logprobs object if requested
        json logprobs_obj = nullptr;
        if (ctx.params.req_logprobs && active_req) {
            const auto& lp_data = active_req->output_logprobs;
            json content_logprobs = json::array();
            for (size_t idx = 0; idx < lp_data.size() && idx < output_ids.size(); idx++) {
                const auto& lp = lp_data[idx];
                json top_arr = json::array();
                for (const auto& t : lp.top) {
                    top_arr.push_back({{"token", safe_token_json(t.text)},
                                       {"logprob", t.logprob},
                                       {"bytes", token_bytes_json(t.text)}});
                }
                content_logprobs.push_back({{"token", safe_token_json(lp.text)},
                                            {"logprob", lp.logprob},
                                            {"bytes", token_bytes_json(lp.text)},
                                            {"top_logprobs", top_arr}});
            }
            logprobs_obj = {{"content", content_logprobs}};
        }

        // Parse tool calls from model output. Run even on finish=length:
        // the model may have emitted a complete tool_call and then kept
        // generating until the budget ran out (common before we hook the
        // family-specific close marker as a stop token). The parser is
        // tolerant of trailing garbage after the closing marker.
        std::vector<ParsedToolCall> tool_calls;
        std::string tool_validation_error;
        if (ctx.params.has_tools) {
            auto [pre_content, parsed_calls] = parse_tool_calls(ctx.snap.tpl_family, content,
                                                                state.next_tool_call_id);
            if (!parsed_calls.empty()) {
                tool_calls = std::move(parsed_calls);
                // OpenAI parallel_tool_calls=false: emit at most one call.
                if (!ctx.params.parallel_tool_calls && tool_calls.size() > 1)
                    tool_calls.resize(1);
                content = pre_content;
                finish = "tool_calls";
                // Validate parsed arguments against each tool's input schema.
                // A failure means the model hallucinated/garbled the call —
                // surface it rather than silently shipping bad arguments.
                for (auto& tc : tool_calls) {
                    validate_tool_call(tc, ctx.params.tools);
                    if (!tc.valid) {
                        if (!tool_validation_error.empty())
                            tool_validation_error += "; ";
                        tool_validation_error += tc.name + ": " + tc.error;
                    }
                }
            }
        }

        json msg = {{"role", "assistant"}};
        if (!tool_calls.empty()) {
            // content is null when only tool calls (no preceding text)
            msg["content"] = content.empty() ? json(nullptr) : json(content);
            json tc_array = json::array();
            for (const auto& tc : tool_calls) {
                json tc_json = {{"id", tc.id},
                                {"type", "function"},
                                {"function", {{"name", tc.name}, {"arguments", tc.arguments}}}};
                if (!tc.valid)
                    tc_json["invalid_arguments"] = tc.error;
                tc_array.push_back(std::move(tc_json));
            }
            msg["tool_calls"] = tc_array;
        } else {
            msg["content"] = content;
        }
        if (!reasoning_content.empty()) {
            msg["reasoning_content"] = reasoning_content;
        }
        if (!tool_validation_error.empty()) {
            msg["tool_call_validation_error"] = tool_validation_error;
        }

        json choice = {{"index", ci}, {"message", msg}, {"finish_reason", finish}};
        if (!logprobs_obj.is_null()) {
            choice["logprobs"] = logprobs_obj;
        }

        choices.push_back(choice);

        // Log each completion
        fprintf(stderr, "[%s] completion %d/%d: %d tokens\n", comp_id.c_str(), ci + 1,
                ctx.params.n_completions, n_output_tokens);
    }

    // Log aggregate request
    auto t_end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t_end - ctx.t_start).count();
    fprintf(stderr, "[%s] %d prompt + %d completion tokens (%d choices), %.1f ms\n", comp_id.c_str(),
            ctx.snap.n_prompt_tokens, total_output_tokens, ctx.params.n_completions, ms);
    state.metrics.requests_total++;
    state.metrics.tokens_prompt_total += ctx.snap.n_prompt_tokens;
    state.metrics.tokens_completion_total += total_output_tokens;
    state.metrics.last_request_duration_ms = static_cast<int64_t>(ms);
    state.metrics.request_duration.observe(ms / 1000.0);

    json usage = {{"prompt_tokens", ctx.snap.n_prompt_tokens},
                  {"completion_tokens", total_output_tokens},
                  {"total_tokens", ctx.snap.n_prompt_tokens + total_output_tokens}};
    // Prefix-cache reporting (OpenAI prompt_tokens_details; the Anthropic
    // converter maps these to cache_read/cache_creation_input_tokens).
    if (imp_req && (imp_req->cached_tokens > 0 || imp_req->pin_kv_prefix)) {
        json details = {{"cached_tokens", imp_req->cached_tokens}};
        int creation = cache_creation_tokens_(imp_req, ctx.snap.n_prompt_tokens);
        if (creation > 0)
            details["cache_creation_tokens"] = creation;
        usage["prompt_tokens_details"] = std::move(details);
        state.metrics.tokens_cached_total += imp_req->cached_tokens;
    }

    json response = {{"id", comp_id},      {"object", "chat.completion"},
                     {"created", created}, {"model", ctx.snap.model_name},
                     {"choices", choices}, {"usage", usage}};

    // Pull the final finish_reason from choice 0 for log correlation;
    // multi-completion requests still record only the aggregate.
    const char* nonstream_finish = nullptr;
    if (!choices.empty() && choices[0].contains("finish_reason") &&
        choices[0]["finish_reason"].is_string()) {
        nonstream_finish = choices[0]["finish_reason"].get_ref<const std::string&>().c_str();
    }
    log_request_jsonl(state, ctx.log_skip, ctx.t_log_start, comp_id, ctx.log_endpoint,
                      ctx.log_client_ip, ctx.log_raw_body,
                      ms, ctx.snap.n_prompt_tokens, total_output_tokens, nonstream_finish, response);

    res.set_content(dump_safe(response), "application/json");
}
