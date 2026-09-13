// AUTO-SPLIT from handlers.cpp. Shared chat-completion machinery: request-log, snapshot+tokenize,
// non-streaming response builder. Body parsing split into handlers_chat_params.cpp
// (800-LOC hard gate). Used by OpenAI chat and Anthropic messages endpoints.

#include "runtime/engine.h"
#include "handlers.h"
#include "handlers_internal.h"

#include <tuple>
#include "model/image_placeholders.h"
#include "utils.h"
#include "tool_call.h"
#include "anthropic.h"
#include "stream_pipeline.h"
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

// Set true on the calling thread when handle_messages is delegating to
// handle_chat_completions via a shim — suppresses inner request-log entries
// so the Anthropic call only logs once at the outer handler.
thread_local bool g_in_anthropic_shim = false;

// g_shim_stop_sequence: the stop sequence that ended the last non-stream generation on this
// thread. OpenAI's finish_reason "stop" doesn't distinguish natural end from a stop-sequence
// match; Anthropic does (#1550), so this travels beside the JSON body.
thread_local std::string g_shim_stop_sequence;

// Writes one JSONL request-log line: timing, endpoint, raw client body, token counts, finish
// reason, and (non-streaming only) the response body.
void log_request_jsonl(ServerState& state, bool skip, const std::chrono::system_clock::time_point& t_start,
                       const std::string& req_id, const std::string& endpoint, const std::string& client_ip,
                       const std::string& raw_body, double latency_ms, int prompt_tokens,
                                              int completion_tokens, const char* finish_reason, const json& response_body,
                       const std::string& client_request_id, const RequestSpan* trace) {
    if (skip)
        return;
    // The span first: it does not depend on the JSONL being on.
    std::string trace_id, span_id;
    if (state.tracer.enabled()) {
        RequestSpan sp = trace ? *trace : RequestSpan{};
        sp.endpoint = endpoint;
        sp.req_id = req_id;
        sp.client_request_id = client_request_id;
        sp.t_start = t_start;
        sp.latency_ms = latency_ms;
        sp.prompt_tokens = prompt_tokens;
        sp.completion_tokens = completion_tokens;
        sp.finish_reason = finish_reason ? finish_reason : "";
        std::tie(trace_id, span_id) = state.tracer.record(sp);
    }
    if (!state.request_logger.enabled)
        return;
    json record;
    if (!trace_id.empty()) {
        record["trace_id"] = trace_id;
        record["span_id"] = span_id;
    }
    record["ts_ms"] =
        std::chrono::duration_cast<std::chrono::milliseconds>(t_start.time_since_epoch()).count();
    record["req_id"] = req_id;
    if (!client_request_id.empty())
        record["client_request_id"] = client_request_id;
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

// Enforced tool calling (#1002): derives the FSM constraint from the POST-snapshot template
// (family+dialect), run after ensure_model_loaded so an auto-loaded/switched model gets its own
// template's grammar rather than the parse-time guess.
static void collect_tool_enforcement_(ChatRequestContext& ctx) {
    if (!ctx.params.has_tools)
        return;
    // tool_choice=required / forced function on the ChatML <tool_call>
    // dialect. Empty result = prompt-hint fallback.
    ctx.params.tool_constraint_tools = collect_tool_constraint(ctx.snap.tpl_family, ctx.params.tools,
                                                               ctx.params.tool_choice);
    if (!ctx.params.tool_constraint_tools.empty()) {
        ctx.params.tool_envelope_open = "<tool_call>\n";
        ctx.params.tool_envelope_close = "\n</tool_call>";
    } else {
        // No forced/required constraint — try strict optional (OpenAI
        // strict:true with a model-chosen call): the envelope is not forced,
        // the body FSM engages only if the model opens a tool call (#1002).
        ctx.params.tool_constraint_tools = collect_strict_tool_constraint(ctx.snap.tpl_family,
                                                                          ctx.params.tools,
                                                                          ctx.params.tool_choice);
        if (!ctx.params.tool_constraint_tools.empty()) {
            ctx.params.tool_constraint_optional = true;
            ctx.params.tool_envelope_open = "<tool_call>\n";
            ctx.params.tool_envelope_close = "\n</tool_call>";
        }
    }
    // Qwen-Coder/Qwen3.6 XML templates share the <tool_call> envelope but the BODY grammar is XML
    // (<function=NAME><parameter=KEY>, raw-text values): flag it so the engine builds the XML FSM,
    // not the JSON one, which masks newlines and mangles multi-line arguments.
    if (!ctx.params.tool_constraint_tools.empty() &&
        ctx.snap.tpl_family == imp::ChatTemplateFamily::CHATML && ctx.snap.have_template &&
        ctx.snap.chat_tpl.tool_xml_dialect())
        ctx.params.tool_constraint_xml = true;
    // Llama3 `<function=NAME>{args}</function>` forced function: constrain the
    // bare parameter schema with a per-tool envelope (#1002). Only when the
    // ChatML paths above found nothing (different family).
    if (ctx.params.tool_constraint_tools.empty()) {
        auto [ln, lparams] = collect_llama3_forced_tool(ctx.snap.tpl_family, ctx.params.tools,
                                                        ctx.params.tool_choice);
        if (!ln.empty()) {
            ctx.params.tool_constraint_tools = {{ln, lparams}};
            ctx.params.tool_constraint_bare_args = true;
            ctx.params.tool_envelope_open = "<function=" + ln + ">";
            ctx.params.tool_envelope_close = "</function>";
        }
    }
}

bool snapshot_state_and_tokenize_(httplib::Response& res, ServerState& state, ChatRequestContext& ctx) {
    // model_has_vision: whether the LOADED model can see images at all. Without this check, a
    // request carrying images for a model with no vision tower answered as if nothing had been
    // sent (#1198).
    bool model_has_vision = false;
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
        snapshot_mtp_state_(state, ctx.snap);
        ctx.snap.tpl_family = ctx.snap.have_template ? ctx.snap.chat_tpl.family()
                                                     : imp::ChatTemplateFamily::CHATML;
        if (ctx.snap.have_template)
            ctx.snap.stop_token_ids = ctx.snap.chat_tpl.stop_token_ids();
        // Provisionally add <think> as a stop token. Removed below if the
        // request enables thinking. Without this, think-trained models at high
        // temp can hallucinate phantom turns ("Human\n<think>...").
        if (state.think_start_id >= 0) {
            ctx.snap.stop_token_ids.push_back(state.think_start_id);
        }
        model_has_vision = state.ctx && state.ctx->engine->has_vision();
        ctx.snap.has_vision_request = !ctx.params.images.empty() && model_has_vision;
    }

    // Refuse (not silently drop) images when the loaded model has no vision tower: a fluent
    // text-only answer about a picture the model never received is indistinguishable from a real
    // one. The load-time WARN never reaches the client.
    if (!ctx.params.images.empty() && !model_has_vision) {
        res.status = 400;
        json error = {
            {"error",
             {{"message",
               "This model cannot see images — it is loaded without a vision tower, so the " +
                   std::to_string(ctx.params.images.size()) +
                   " image part(s) in this request would be ignored. Vision needs either a "
                   "Qwen3-VL checkpoint or a GGUF started with --mmproj; a multimodal "
                   "SafeTensors checkpoint of any other family loads text-only (the load log "
                   "says so). Send text only, or load a model that can see."},
              {"type", "invalid_request_error"},
              {"code", "vision_unavailable"}}}};
        res.set_content(dump_safe(error), "application/json");
        return false;
    }

    // Called after ensure_model_loaded (#1002): the request may have auto-loaded or switched the
    // model, so the grammar must derive from the template that actually renders this prompt, not
    // the parse-time family guess.
    collect_tool_enforcement_(ctx);

    // #1592: refuse (400) tool_choice "required"/named-function when the family's template has no
    // tool-call grammar, rather than degrade to a prose hint with 200. Measured 0/40 across
    // gemma-3/gemma-4/gpt-oss (no grammar) vs 10/10 on Qwen3-4B ChatML (has one); "auto" is untouched.
    if (ctx.params.has_tools && !tool_choice_is_enforceable(ctx.snap.tpl_family, ctx.params.tool_choice)) {
        const char* fam = imp::chat_template_family_name(ctx.snap.tpl_family);
        const bool named = ctx.params.tool_choice.is_object();
        res.status = 400;
        json error = {
            {"error",
             {{"message", std::string("\"tool_choice\": ") + (named ? "a named function" : "\"required\"") +
                              " cannot be enforced on this model's chat template family (" + fam +
                              "). \"required\" is enforced on chatml; a named function on chatml and "
                              "llama3. On every other family it would degrade to a prompt hint and the "
                              "model answers with prose instead of calling the tool. Send \"tool_choice\": "
                              "\"auto\", or load a model whose template this server can constrain."},
              {"type", "invalid_request_error"},
              {"param", "tool_choice"},
              {"code", "tool_choice_unenforceable"}}}};
        res.set_content(dump_safe(error), "application/json");
        return false;
    }

    // Channel models (Gemma-4) degenerate more easily on casual prompts under default sampling;
    // tighten the default when the caller doesn't specify a sampler param. Qwen3/DeepSeek keep
    // 0.95/40/1.0 defaults.
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

    // Vision (per-request, F-A5): CPU-preprocess into ctx.snap.vision_image, no engine pause and no
    // global image state. Batch worker encodes+binds per-request on admission so vision batches like text.
    if (ctx.snap.has_vision_request) {
        auto fail = [&](const std::string& why) {
            res.status = 400;
            json error = {{"error", {{"message", why}, {"type", "invalid_request_error"}}}};
            res.set_content(dump_safe(error), "application/json");
            return false;
        };
        // An image_url that could not be read is fatal, not skipped: dropping
        // one would slide every later picture onto the wrong placeholder.
        if (!ctx.params.image_error.empty())
            return fail(ctx.params.image_error);
        // Salts the prefix cache so a hit needs the same pictures, not just the
        // same token ids (every image token shares one id). Folded in order, so
        // the same two images the other way round are a different key.
        ctx.snap.vision_content_hash = 0;
        for (const auto& bytes : ctx.params.images)
            ctx.snap.vision_content_hash = imp::combine_image_hash(ctx.snap.vision_content_hash,
                                                                   imp::image_content_hash(bytes));
        if (state.ctx->engine->has_qwen_vision()) {
            // Dynamic resolution: patchify now (CPU only) so the token counts
            // are known before the prompt is tokenized — each placeholder has
            // to be expanded to exactly its own picture's count.
            for (const auto& bytes : ctx.params.images) {
                auto patches = std::make_shared<imp::QwenPatches>();
                if (!state.ctx->engine->preprocess_image_qwen(bytes, *patches))
                    return fail("Failed to process image");
                const int tokens = state.ctx->engine->image_tokens_of(*patches);
                if (tokens <= 0)
                    return fail("Failed to process image");
                ctx.snap.qwen_image_tokens.push_back(tokens);
                ctx.snap.qwen_patches.push_back(std::move(patches));
            }
        } else if (ctx.params.images.size() > 1) {
            // The mmproj tower encodes one image into a fixed token count and
            // has no notion of a second. Refusing beats answering about one of
            // the pictures as if it were all of them.
            return fail("this model accepts one image per request");
        } else {
            auto img = std::make_shared<imp::ImageData>();
            if (!state.ctx->engine->preprocess_image(ctx.params.images[0], *img))
                return fail("Failed to process image");
            ctx.snap.vision_image = std::move(img);
        }
    }

    // Thinking defaults ON for think-trained models in plain chat (avoids reasoning leaking into
    // content); OFF for json_mode/tool-calls. Explicit enable_thinking always wins. Flips OFF only
    // when the Jinja template itself never mentions thinking (guards e.g. Qwen3-Instruct-2507).
    const bool template_think_evidence = !ctx.snap.have_template || !ctx.snap.chat_tpl.has_jinja() ||
                                         ctx.snap.chat_tpl.mentions_thinking();
    // #1431: any structured-output constraint (regex/grammar/json_schema) covers the WHOLE reply, so
    // thinking must be disabled - the constrainer would hold the mask open through reasoning. On a
    // model whose </think> is multi-token BPE the block never closed in text (0/8 wrong vs 10/10 fixed).
    const bool thinking_default =
        ctx.snap.is_think_model && template_think_evidence &&
        !imp::server::structured_output_excludes_thinking(
            ctx.params.json_mode, ctx.params.has_tools, !ctx.params.json_schema_str.empty(),
            !ctx.params.regex_pattern.empty(), !ctx.params.grammar.empty());
    const bool want_thinking = ctx.params.enable_thinking_set ? ctx.params.enable_thinking_requested
                                                              : thinking_default;
    // think_budget<=0 disables thinking entirely (folded into enable_thinking): budget=0 used to
    // leave thinking ON without arming the force-close, so the model reasoned to max_tokens with
    // empty content (#752).
    const bool budget_disables_thinking = ctx.params.think_budget <= 0.0f;
    ctx.snap.enable_thinking = ctx.snap.is_think_model && ctx.snap.think_start_id >= 0 && want_thinking &&
                               !budget_disables_thinking;
    // Suppressing thinking means stamping enable_thinking=false into the Jinja context: an unstamped
    // template falls back to its own default, and e.g. Qwen3.8 defaults to an OPEN <think> block,
    // which the reconcile step below would then read as reasoning that never closes.
    ctx.snap.suppress_thinking = imp::server::should_stamp_thinking_off(
        ctx.snap.is_think_model, ctx.snap.enable_thinking, budget_disables_thinking, want_thinking);
    ctx.snap.reasoning_effort = ctx.params.reasoning_effort;

    // If thinking IS enabled, remove the provisional <think> stop token.
    if (ctx.snap.enable_thinking && ctx.snap.think_start_id >= 0) {
        auto& ids = ctx.snap.stop_token_ids;
        ids.erase(std::remove(ids.begin(), ids.end(), ctx.snap.think_start_id), ids.end());
    }

    // Injects an implicit stop on hallucinated turn boundaries ("Human\n") that thinking models emit
    // at high temperature, only when the caller supplied no stop sequences. Skipped under ignore_eos
    // (a benchmark run past EOS would stop early).
    if (ctx.snap.is_think_model && ctx.params.stop_sequences.empty() && !ctx.params.ignore_eos) {
        ctx.params.stop_sequences.push_back("\nHuman");
    }

    // force_thinking stamps enable_thinking=true into the Jinja render so a template defaulting to a
    // pre-closed block (Qwen3.5-4B) actually opens thinking when the caller explicitly asks - an
    // explicit enable_thinking:true was otherwise a silent no-op on such templates.
    const bool force_thinking = ctx.params.enable_thinking_set && ctx.params.enable_thinking_requested &&
                                ctx.snap.enable_thinking;

    // The byte bound before any render or merge walk: --max-input-tokens is
    // checked on the token count below, which exists only after the cost has
    // been paid (AUDIT_arch_2026 F2-9). count_tokens takes this path too.
    size_t prompt_bytes = 0;
    for (const auto& m : ctx.params.chat_msgs)
        prompt_bytes += m.content.size();
    if (!prompt_within_input_budget(res, prompt_bytes, state.max_input_tokens, "messages"))
        return false;

    // Tokenize with chat template (with image tokens if vision is active)
    if (ctx.snap.have_template && !ctx.snap.qwen_patches.empty()) {
        // Chat template renders one <|image_pad|> per image before sizes are known (smart_resize runs
        // after). Placed on the first user turn (the position the parser reliably tracks), rendered,
        // then each placeholder expands to its real token count.
        std::string blocks;
        for (size_t i = 0; i < ctx.snap.qwen_patches.size(); ++i)
            blocks += "<|vision_start|><|image_pad|><|vision_end|>";
        auto msgs = ctx.params.chat_msgs;
        for (auto& m : msgs)
            if (m.role == "user") {
                m.content = blocks + m.content;
                break;
            }
        ctx.snap.tokens = ctx.snap.chat_tpl.apply(*ctx.snap.tok, msgs, ctx.snap.suppress_thinking,
                                                  force_thinking, ctx.snap.reasoning_effort);
        const int32_t pad_id = ctx.snap.tok->find_token("<|image_pad|>");
        const auto expanded = pad_id < 0 ? std::unexpected(std::string("tokenizer has no <|image_pad|>"))
                                         : imp::expand_image_placeholders(ctx.snap.tokens, pad_id,
                                                                          ctx.snap.qwen_image_tokens);
        if (!expanded) {
            res.status = 400;
            json error = {{"error", {{"message", expanded.error()}, {"type", "invalid_request_error"}}}};
            res.set_content(dump_safe(error), "application/json");
            return false;
        }
    } else if (ctx.snap.have_template && ctx.snap.has_vision_request) {
        ctx.snap.tokens = ctx.snap.chat_tpl.apply_with_image(*ctx.snap.tok, ctx.params.chat_msgs, 256,
                                                             ctx.snap.suppress_thinking, force_thinking,
                                                             ctx.snap.reasoning_effort);
    } else if (ctx.snap.have_template && ctx.snap.tools_via_jinja) {
        std::string tc_str = ctx.params.tool_choice.is_string() ? ctx.params.tool_choice.get<std::string>()
                                                                : "auto";
        ctx.snap.tokens = ctx.snap.chat_tpl.apply_with_tools(*ctx.snap.tok, ctx.params.chat_msgs,
                                                             ctx.snap.tool_defs, tc_str,
                                                             ctx.snap.suppress_thinking, force_thinking,
                                                             ctx.snap.reasoning_effort);
        // If Jinja2 tools render failed, fall back to text-based tool prompt injection
        if (ctx.snap.tokens.empty()) {
            IMP_LOG_INFO("Jinja2 tools path failed, falling back to text-based tool prompt");
            // Text-fallback hint advertises the ChatML JSON body; when an XML tool constraint is armed, drop
            // to the hint (not the JSON FSM) so an XML-finetuned model isn't fought with the wrong grammar.
            if (ctx.params.tool_constraint_xml) {
                ctx.params.tool_constraint_xml = false;
                ctx.params.tool_constraint_tools.clear();
                ctx.params.tool_envelope_open.clear();
                ctx.params.tool_envelope_close.clear();
                ctx.params.tool_constraint_optional = false;
            }
            std::string tool_prompt = build_tool_prompt(ctx.snap.tpl_family, ctx.params.tools,
                                                        ctx.params.tool_choice);
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
                    if (ctx.params.cache_prefix_messages >= 0)
                        ctx.params.cache_prefix_messages++;  // boundary shifts with the insert
                }
            }
            ctx.snap.tokens = ctx.snap.chat_tpl.apply(*ctx.snap.tok, ctx.params.chat_msgs,
                                                      ctx.snap.suppress_thinking, force_thinking,
                                                      ctx.snap.reasoning_effort);
        }
    } else if (ctx.snap.have_template) {
        // No tools, or no Jinja2 support — inject text-based tool prompt if tools present
        if (ctx.params.has_tools) {
            std::string tool_prompt = build_tool_prompt(ctx.snap.tpl_family, ctx.params.tools,
                                                        ctx.params.tool_choice);
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
                    if (ctx.params.cache_prefix_messages >= 0)
                        ctx.params.cache_prefix_messages++;  // boundary shifts with the insert
                }
            }
        }
        ctx.snap.tokens = ctx.snap.chat_tpl.apply(*ctx.snap.tok, ctx.params.chat_msgs,
                                                  ctx.snap.suppress_thinking, force_thinking,
                                                  ctx.snap.reasoning_effort);
    } else {
        // Concatenate all message content as raw text
        std::string raw;
        for (const auto& m : ctx.params.chat_msgs)
            raw += m.content + "\n";
        ctx.snap.tokens = ctx.snap.tok->encode(raw);
    }

    // cache_control breakpoint (#1046): re-render leading messages up to the marked block and count
    // tokens; engine pins that many tokens (rounded down to full KV blocks) against eviction.
    // Approximate is fine - pins are eviction protection, not correctness state.
    if (ctx.params.cache_prompt && ctx.params.cache_prefix_messages > 0 &&
        ctx.params.cache_prefix_messages < static_cast<int>(ctx.params.chat_msgs.size()) &&
        ctx.snap.have_template) {
        std::vector<imp::ChatMessage> prefix_msgs(
            ctx.params.chat_msgs.begin(),
            ctx.params.chat_msgs.begin() + ctx.params.cache_prefix_messages);
        ctx.snap.pin_prefix_tokens = static_cast<int>(
            ctx.snap.chat_tpl
                .apply(*ctx.snap.tok, prefix_msgs, ctx.snap.suppress_thinking, force_thinking,
                       ctx.snap.reasoning_effort)
                .size());
    }

    // Thinking-state pipeline: INTENT (explicit request or heuristic) -> RENDER (apply() stamps
    // enable_thinking only when forced/suppressed) -> GROUND TRUTH (below: reconcile against the
    // actual rendered prompt tail, matching vLLM's qwen3_reasoning_parser auto-detection).
    // Detects on decoded text, not token-ID equality: Qwen3.6 ships <think>/</think> as non-special
    // added_tokens, so BPE splits them into 3 pieces instead of one special-token id.
    auto prompt_tail_contains = [&](const char* needle, int max_tail_tokens) -> bool {
        int n = static_cast<int>(ctx.snap.tokens.size());
        int start = std::max(0, n - max_tail_tokens);
        std::string tail_text;
        for (int i = start; i < n; ++i) {
            tail_text += ctx.snap.tok->decode_token(ctx.snap.tokens[i]);
        }
        return tail_text.find(needle) != std::string::npos;
    };
    // Reconcile: an OPEN <think> prefix (no matching </think> in the tail) turns thinking ON; a
    // pre-closed block (<think>...</think>) turns it OFF even if the heuristic defaulted ON (#934:
    // Qwen3.5-4B mentions enable_thinking but defaults closed, else content is empty).
    // Window 16 (not 8) so both tags of an adjacent closed block land in the same tail scan.
    {
        const bool tail_has_think = prompt_tail_contains("<think>", 16);
        const bool tail_has_close = prompt_tail_contains("</think>", 16);
        const bool was_thinking = ctx.snap.enable_thinking;
        ctx.snap.enable_thinking = imp::server::reconcile_thinking_with_prompt_tail(
            ctx.snap.enable_thinking, ctx.params.enable_thinking_set, tail_has_think, tail_has_close);
        // If reconcile flips thinking OFF after the snapshot already removed the provisional <think>
        // stop token (added while ON), restore it - a non-thinking think-model still needs the
        // phantom-"<think>"-turn guard.
        if (was_thinking && !ctx.snap.enable_thinking && ctx.snap.think_start_id >= 0) {
            auto& ids = ctx.snap.stop_token_ids;
            if (std::find(ids.begin(), ids.end(), ctx.snap.think_start_id) == ids.end())
                ids.push_back(ctx.snap.think_start_id);
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

    // Predicted Outputs: tokenize `prediction.content` (plain encode, no template/specials) to seed
    // the n-gram draft corpus only - never forwarded as output. Clamped to model context.
    if (!ctx.params.prediction_text.empty()) {
        ctx.snap.prediction_tokens = ctx.snap.tok->encode(ctx.params.prediction_text);
        if (ctx.snap.max_seq_len > 0 &&
            ctx.snap.prediction_tokens.size() > static_cast<size_t>(ctx.snap.max_seq_len))
            ctx.snap.prediction_tokens.resize(ctx.snap.max_seq_len);
    }

    // Server-side input-token limit (--max-input-tokens). Reject before
    // prefill so an oversized prompt never reaches the engine.
    if (state.max_input_tokens > 0 && ctx.snap.n_prompt_tokens > state.max_input_tokens) {
        send_json_error(res, 400, "invalid_request_error",
                        "Prompt exceeds max input tokens (" + std::to_string(ctx.snap.n_prompt_tokens) +
                            " > " + std::to_string(state.max_input_tokens) + ")",
                        "messages", "context_length_exceeded");
        return false;
    }

    // Validate prompt length against context window
    if (ctx.snap.n_prompt_tokens >= ctx.snap.max_seq_len) {
        send_json_error(res, 400, "invalid_request_error",
                        "Prompt exceeds context window (" + std::to_string(ctx.snap.n_prompt_tokens) +
                            " tokens >= " + std::to_string(ctx.snap.max_seq_len) + " max)",
                        "messages", "context_length_exceeded");
        return false;
    }

    // Per-request LoRA selection (#522): name resolved here; batching worker switches the
    // engine-global adapter at admission once nothing of another adapter is in flight
    // (AUDIT_arch_2026 E-1). One adapter active at a time; others queue behind the barrier.
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
        ctx.snap.lora_id = want;
    }

    // Clamp max_tokens to remaining context window
    int remaining = ctx.snap.max_seq_len - ctx.snap.n_prompt_tokens;
    if (ctx.params.max_tokens > remaining)
        ctx.params.max_tokens = remaining;

    // Start timing
    ctx.t_start = std::chrono::high_resolution_clock::now();

    return true;
}

// Single params->request mapping for the ctx-based submission sites (see
// handlers_internal.h).
std::shared_ptr<imp::Request> build_imp_request_(const ChatRequestContext& ctx,
                                                 const std::vector<int32_t>& input_tokens, int completion_idx,
                                                 bool stream) {
    auto req = std::make_shared<imp::Request>();
    // The id the client sees, carried into the engine so its log lines can be
    // joined to this HTTP request (#1582). With n>1 several engine requests
    // share one completion id, which is what the response says too.
    req->trace_id = ctx.req_id;
    req->image = ctx.snap.vision_image;         // per-request vision (null for text)
    req->qwen_patches = ctx.snap.qwen_patches;  // dynamic-resolution route (empty otherwise)
    req->vision_content_hash = ctx.snap.vision_content_hash;
    req->lora_id = ctx.snap.lora_id;
    req->input_tokens = input_tokens;
    req->max_tokens = ctx.params.max_tokens;
    req->temperature = ctx.params.temperature;
    req->top_p = ctx.params.top_p;
    req->top_k = ctx.params.top_k;
    req->seed = (ctx.params.seed != -1) ? ctx.params.seed + completion_idx : -1;
    req->pin_kv_prefix = ctx.params.cache_prompt;
    req->pin_kv_prefix_tokens = ctx.snap.pin_prefix_tokens;
    // Per-request speculation: the override, the MTP depth, and the reason a
    // decline happened, resolved by the same pure rule the engine uses
    // (spec_request.h) so the two cannot disagree about one request.
    apply_spec_contract_(*req, ctx.params.spec_override, ctx.params.spec_mtp_k, ctx.snap.mtp_armed_k,
                         ctx.snap.mtp_head_present, ctx.snap.mtp_head_loaded);
    req->priority = ctx.params.priority;
    req->prediction_tokens = ctx.snap.prediction_tokens;
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
    req->ignore_eos = ctx.params.ignore_eos;
    req->json_mode = ctx.params.json_mode;
    req->json_schema = ctx.params.json_schema_str;
    req->regex_pattern = ctx.params.regex_pattern;
    req->grammar = ctx.params.grammar;
    req->has_tools = ctx.params.has_tools;
    req->tool_constraint_tools = ctx.params.tool_constraint_tools;
    req->tool_envelope_open = ctx.params.tool_envelope_open;
    req->tool_envelope_close = ctx.params.tool_envelope_close;
    req->tool_constraint_optional = ctx.params.tool_constraint_optional;
    req->tool_constraint_parallel = ctx.params.parallel_tool_calls;
    req->tool_constraint_bare_args = ctx.params.tool_constraint_bare_args;
    req->tool_constraint_xml = ctx.params.tool_constraint_xml;
    req->tpl_family = ctx.snap.tpl_family;
    req->logit_bias = ctx.params.logit_bias;
    req->think_budget = ctx.params.think_budget;
    // req->started_in_think = enable_thinking: without it the engine's think-budget enforcement
    // never sees an opener in the output and lets the model reason to max_tokens (content empty).
    req->started_in_think = ctx.snap.enable_thinking;
    req->in_think_block = ctx.snap.enable_thinking;
    // Stream requests stay on per-step decode for real per-token SSE (#754).
    req->stream = stream;
    req->status = imp::RequestStatus::PENDING;
    return req;
}

// Runs n_completions independent generations sequentially via the batching engine, then sends
// one JSON response with the combined choices array.
void nonstream_chat_response_(httplib::Response& res, ServerState& state, ChatRequestContext& ctx,
                              std::shared_ptr<imp::Request>& imp_req,
                              std::shared_ptr<ServerRequest>& server_req,
                              const std::vector<int32_t>& saved_tokens, const std::string& comp_id,
                              int64_t created) {
    // Non-streaming: decode all tokens, return complete response
    // For n > 1, run multiple independent generations sequentially
    json choices = json::array();
    int total_output_tokens = 0;
    int total_reasoning_tokens = 0;
    double ttft_ms = -1.0;  // first token of the FIRST completion (#1578)
    g_shim_stop_sequence.clear();

    for (int ci = 0; ci < ctx.params.n_completions; ci++) {
        // For subsequent completions, create a new request and submit it
        if (ci > 0) {
            imp_req = build_imp_request_(ctx, saved_tokens, ci, /*stream=*/false);
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
        int n_eos_counted = 0;  // EOS / stop tokens counted under ignore_eos
        const char* finish = nullptr;
        std::string output_text;  // accumulated output for stop matching
        std::string matched_stop;  // which stop sequence ended it, for the Anthropic shim (#1550)

        auto ns_request_start = std::chrono::steady_clock::now();
        auto t_prev_token = std::chrono::high_resolution_clock::now();  // last delivered token (ITL)
        for (;;) {
            // Check request timeout
            if (state.request_timeout > 0) {
                auto elapsed = std::chrono::steady_clock::now() - ns_request_start;
                if (elapsed > std::chrono::seconds(state.request_timeout)) {
                    server_req->cancel();
                    state.metrics.requests_timed_out++;
                    state.metrics.observe_unadmitted_queue_wait(server_req->t_submit,
                                                                server_req->queue_ms.load(
                                                                    std::memory_order_relaxed));
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

            // Drops structural stop tokens that slip through: the engine's think-block implicit-close passes
            // one EOS-like token to recover from empty thinking, which must never reach user-visible content.
            bool is_structural_stop = (token == ctx.snap.tok->eos_id());
            if (!is_structural_stop && ctx.snap.have_template) {
                for (int32_t stop_id : ctx.snap.stop_token_ids) {
                    if (token == stop_id) {
                        is_structural_stop = true;
                        break;
                    }
                }
            }
            // ignore_eos (vLLM semantics): EOS and stop tokens count as
            // output tokens and carry no text; only max_tokens ends the
            // request.
            if (ctx.params.ignore_eos && is_structural_stop) {
                n_eos_counted++;
                if (evt.is_last) {
                    finish = evt.finish_reason ? evt.finish_reason : "length";
                    break;
                }
                continue;
            }
            if (!evt.is_last && is_structural_stop)
                continue;

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

            // TTFT was recorded on the streaming path only, while this one
            // still incremented requests_total - so the histogram described
            // half the traffic and said nothing about which half (#1578).
            const auto t_tok = std::chrono::high_resolution_clock::now();
            if (output_ids.empty() && ttft_ms < 0.0) {
                ttft_ms = std::chrono::duration<double, std::milli>(t_tok - ctx.t_start).count();
                // Same point as the streaming path: by the first token the
                // worker has admitted the request, so queue_ms is final
                // (#1580).
                const double q = server_req->queue_ms.load(std::memory_order_relaxed);
                if (q >= 0.0)
                    state.metrics.queue_time.observe(q / 1000.0);
            } else if (!output_ids.empty()) {
                // ITL was observed on the streaming path only (#1577); this loop adds it for non-streaming
                // too.
                // Per completion: the first token of a second (n>1) completion is not a gap.
                state.metrics.inter_token.observe(
                    std::chrono::duration<double>(t_tok - t_prev_token).count());
            }
            t_prev_token = t_tok;
            output_ids.push_back(token);

            // Check text-level stop sequences
            if (!ctx.params.stop_sequences.empty()) {
                output_text += ctx.snap.tok->decode_token(token);
                bool stop_found = false;
                // Finds the earliest-occurring stop sequence and which one matched (#1550, Anthropic shim
                // reports it) - taking the first list entry regardless of position cuts at the wrong offset
                // when two stop sequences are both present.
                size_t best = std::string::npos;
                for (const auto& stop : ctx.params.stop_sequences) {
                    auto pos = output_text.find(stop);
                    if (pos != std::string::npos && (best == std::string::npos || pos < best)) {
                        best = pos;
                        matched_stop = stop;
                    }
                }
                if (best != std::string::npos) {
                    output_text = output_text.substr(0, best);
                    stop_found = true;
                }
                if (stop_found) {
                    finish = "stop";
                    g_shim_stop_sequence = matched_stop;
                    break;
                }
            }

            // Break after processing the last non-EOS token
            if (finish)
                break;
        }

        if (!finish)
            finish = "length";

        // Admission refusal (invariant I6): the KV pool can never hold this prompt, so retrying at the
        // same length never helps. 503 with the reason (not 200 with an empty completion, which read as
        // "the model chose to say nothing").
        if (std::strcmp(finish, "capacity") == 0) {
            // floored: true when the pool fell back to its rescue floor (a few hundred tokens) - "shorten
            // the prompt" is not actionable advice there, it's a startup fault. Name which situation this is.
            bool floored = false;
            if (state.ctx && state.ctx->engine)
                floored = state.ctx->engine->kv_pool_floored();
            send_json_error(res, 503, "capacity_error",
                            floored
                                ? "The KV pool fell back to its rescue floor at startup, so it holds only "
                                  "a few hundred tokens. This lasts as long as the process and retrying "
                                  "will not help: restart the server on a free card. GET /health reports "
                                  "code kv_pool_floored and the exact capacity."
                                : "Request does not fit the KV cache: the prompt needs more blocks than "
                                  "the pool can hold. Shorten the prompt, lower --max-seq-len, or give "
                                  "the server more VRAM (see the engine log for the exact block counts).",
                            /*param=*/nullptr, floored ? "kv_pool_floored" : "context_length_exceeded");
            return;
        }

        // "internal_error": the worker cancelled this request from inside a
        // failed step (host throw or device fault, AUDIT_arch_2026 D-1). A 200
        // with an empty completion would read as a model that chose silence.
        if (std::strcmp(finish, "internal_error") == 0) {
            const bool faulted = state.batching && state.batching->faulted();
            send_json_error(res, 500, "internal_error",
                            faulted ? "The engine faulted (CUDA context poisoned) and this process cannot "
                                      "recover: restart it. GET /health reports code engine_faulted."
                                    : "The engine step failed and this request was cancelled; retry.",
                            /*param=*/nullptr, faulted ? "engine_faulted" : "engine_step_failed");
            return;
        }

        int n_output_tokens = static_cast<int>(output_ids.size()) + n_eos_counted;
        total_output_tokens += n_output_tokens;
        std::string content = !ctx.params.stop_sequences.empty() ? output_text
                                                                 : ctx.snap.tok->decode(output_ids);
        // max_tokens can stop mid-codepoint; the streaming path holds those
        // bytes back, so this one must drop them or the two transports return
        // different text for the same request (#1310).
        drop_incomplete_utf8_tail(content);

        // Extracts reasoning_content (DeepSeek marker format) or strips think blocks. enable_thinking
        // also covers text-level thinkers (Nemotron): is_think_model is false but output is reasoning
        // until the literal "</think>".
        std::string reasoning_content;
        // harmony_raw: kept because gpt-oss's tool call IS a Harmony channel and the split below consumes
        // channels. The tool-call parser runs later on `content`, where the markup is already gone
        // (#1716 survived a green unit test of the parser alone).
        std::string harmony_raw;
        if (ctx.snap.tpl_family == imp::ChatTemplateFamily::HARMONY) {
            harmony_raw = content;
            // gpt-oss Harmony: splits <|channel|>analysis|final<|message|>... into reasoning_content
            // (analysis) and content (final); without this the raw markup leaks verbatim (#760).
            auto segs = split_harmony_channels(content);
            content = std::move(segs.content);
            if (state.default_args.reasoning_format != "none")
                reasoning_content = std::move(segs.reasoning);
        } else if ((ctx.snap.is_think_model || ctx.snap.enable_thinking) &&
                   state.default_args.reasoning_format == "deepseek") {
            // If generation started inside an injected <think> prefix and never reached </think> (budget
            // exhausted, or stopped mid-think), the WHOLE text is reasoning - extract_reasoning() cannot
            // tell that from text alone (streaming path handles it via its state machine).
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

        // Gemma-4 channel headers "<|channel>NAME[<channel|>]..." wrap both CoT and answer: "thought"
        // goes to reasoning_content, "final" stays in content. Falls back to strip-only for
        // reasoning_format=none.
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

        // completion_tokens_details.reasoning_tokens: streaming has reported this since #1593; without
        // it here the same request answered different numbers depending on transport, and non-stream
        // /v1/responses always said 0. Counting rule lives in utils.h.
        total_reasoning_tokens += nonstream_reasoning_tokens(
            output_ids, ctx.snap.think_start_id, ctx.snap.think_end_id,
            active_req && active_req->started_in_think, reasoning_content.size(),
            [&ctx](int32_t id) { return ctx.snap.tok ? ctx.snap.tok->decode_token(id).size() : 0u; });

        // Build logprobs object if requested
        // Chat shape here; /v1/completions builds the other one (#1589). Both
        // come out of utils.cpp now, so the two cannot drift apart again.
        json logprobs_obj = nullptr;
        if (ctx.params.req_logprobs && active_req) {
            logprobs_obj = chat_logprobs_json(active_req->output_logprobs, output_ids.size());
        }

        // Tool-call parsing runs even on finish=length: the model may emit a complete tool_call then
        // keep generating until the budget runs out. Parser tolerates trailing garbage after the closing
        // marker.
        std::vector<ParsedToolCall> tool_calls;
        std::string tool_validation_error;
        if (ctx.params.has_tools) {
            auto [pre_content, parsed_calls] = parse_tool_calls(ctx.snap.tpl_family,
                                                                harmony_raw.empty() ? content : harmony_raw,
                                                                state.next_tool_call_id,
                                                                tool_names_from_request(ctx.params.tools));
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
        // Detection + WARN live beside the predicate in utils.cpp.
        const bool budget_exhausted = report_answer_lost_to_reasoning(!tool_calls.empty(), content,
                                                                      reasoning_content, finish);
        if (budget_exhausted)
            state.metrics.requests_reasoning_exhausted++;
        if (!tool_validation_error.empty()) {
            msg["tool_call_validation_error"] = tool_validation_error;
        }

        json choice = {{"index", ci}, {"message", msg}, {"finish_reason", openai_finish_reason(finish)}};
        // finish_reason stays "stop"/"length" for SDK compatibility (an unknown enum value breaks strict
        // SDKs); the imp-namespaced detail beside it is the machine-readable half a caller can act on.
        // Decision and write both live in utils.h, reachable from a CPU test.
        attach_reasoning_finish_detail(choice, !tool_calls.empty(), content.empty(),
                                       !reasoning_content.empty());
        if (!logprobs_obj.is_null()) {
            choice["logprobs"] = logprobs_obj;
        }

        choices.push_back(choice);

        // Log each completion
        IMP_LOG_INFO("[%s] completion %d/%d: %d tokens", comp_id.c_str(), ci + 1, ctx.params.n_completions,
                     n_output_tokens);
    }

    // Log aggregate request
    auto t_end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t_end - ctx.t_start).count();
    IMP_LOG_INFO("[%s] %d prompt + %d completion tokens (%d choices), %.1f ms", comp_id.c_str(),
                 ctx.snap.n_prompt_tokens, total_output_tokens, ctx.params.n_completions, ms);
    state.metrics.record_completion(ctx.log_endpoint, ms, ttft_ms, ctx.snap.n_prompt_tokens,
                                    total_output_tokens);

    json usage = {{"prompt_tokens", ctx.snap.n_prompt_tokens},
                  {"completion_tokens", total_output_tokens},
                  {"total_tokens", ctx.snap.n_prompt_tokens + total_output_tokens}};
    // Prefix-cache reporting (OpenAI prompt_tokens_details; the Anthropic
    // converter maps these to cache_read/cache_creation_input_tokens).
    // Also carries `evicted_tokens` when StreamingLLM dropped context mid-run.
    if (json details = prompt_tokens_details_(imp_req, ctx.snap.n_prompt_tokens); !details.is_null())
        usage["prompt_tokens_details"] = std::move(details);
    if (imp_req && (imp_req->cached_tokens > 0 || imp_req->pin_kv_prefix))
        state.metrics.tokens_cached_total += imp_req->cached_tokens;
    // Predicted Outputs accounting (only when the request carried a
    // prediction): accepted/rejected draft tokens whose draft came from the
    // prediction region of the n-gram corpus.
    if (total_reasoning_tokens > 0)
        usage["completion_tokens_details"]["reasoning_tokens"] = total_reasoning_tokens;
    if (imp_req && !imp_req->prediction_tokens.empty()) {
        // Element assignment, not whole-object: reasoning_tokens is already in
        // there and a replacing assignment would drop it.
        usage["completion_tokens_details"]["accepted_prediction_tokens"] = imp_req->pred_accepted;
        usage["completion_tokens_details"]["rejected_prediction_tokens"] = imp_req->pred_rejected;
    }
    add_spec_usage_(usage, imp_req);

    json response = {{"id", comp_id},
                     {"object", "chat.completion"},
                     {"created", created},
                     {"model", ctx.snap.model_name},
                     {"system_fingerprint", system_fingerprint(ctx.snap.model_name)},
                     {"choices", choices},
                     {"usage", usage}};

    // Pull the final finish_reason from choice 0 for log correlation;
    // multi-completion requests still record only the aggregate.
    const char* nonstream_finish = nullptr;
    if (!choices.empty() && choices[0].contains("finish_reason") && choices[0]["finish_reason"].is_string()) {
        nonstream_finish = choices[0]["finish_reason"].get_ref<const std::string&>().c_str();
    }
    ctx.trace.model = ctx.snap.model_name;
    ctx.trace.stream = false;
    // The request objects are the parameters (the last completion's for
    // n > 1), not context fields: nothing ever assigned those.
    if (server_req)
        ctx.trace.queue_ms = server_req->queue_ms.load(std::memory_order_relaxed);
    if (imp_req && imp_req->cached_tokens > 0)
        ctx.trace.cached_tokens = imp_req->cached_tokens;
    if (imp_req)
        ctx.trace.set_spec(imp_req->spec_drafted, imp_req->spec_accepted, imp_req->spec_verifies);
    log_request_jsonl(state, ctx.log_skip, ctx.t_log_start, comp_id, ctx.log_endpoint, ctx.log_client_ip,
                      ctx.log_raw_body, ms, ctx.snap.n_prompt_tokens, total_output_tokens, nonstream_finish,
                      response, ctx.log_client_request_id, &ctx.trace);

    res.set_content(dump_safe(response), "application/json");
}
