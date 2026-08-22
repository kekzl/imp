// AUTO-SPLIT from handlers.cpp (verbatim move; see handlers_internal.h).
// Shared chat-completion machinery: request-log, state snapshot + tokenize,
// and the non-streaming chat response builder. Body parsing lives in
// handlers_chat_params.cpp (split when this TU crossed the 800-LOC hard
// gate). Used by the OpenAI chat endpoint (handlers_chat.cpp) and the
// Anthropic messages endpoint (handlers_messages.cpp).

#include "handlers.h"
#include "handlers_internal.h"
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

// Write one JSONL line capturing this request: timing, endpoint, raw client
// body, token counts, finish reason, and (for non-streaming) the response.
// Streaming responses pass an empty `response_body` since per-chunk text is
// not accumulated.
void log_request_jsonl(ServerState& state, bool skip, const std::chrono::system_clock::time_point& t_start,
                       const std::string& req_id, const std::string& endpoint, const std::string& client_ip,
                       const std::string& raw_body, double latency_ms, int prompt_tokens,
                       int completion_tokens, const char* finish_reason, const json& response_body) {
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
// Acquires state.mtx lock, snapshots engine state into ctx.snap, sets up
// tool defs / vision lock / thinking detection, tokenizes the prompt with
// the chat template, validates prompt length, clamps max_tokens to remaining
// context, and starts timing. Returns true if OK; sets res with 400/503 and
// returns false on failure (model not loaded, prompt too long, vision lock
// timeout, image processing failure).
// Enforced tool calling (#1002): derive the FSM constraint from the
// POST-SNAPSHOT template (family + body dialect). Runs after
// ensure_model_loaded so an auto-loaded or switched model gets the grammar
// its own template teaches; the parse-time family is never baked in.
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
    // Qwen-Coder / Qwen3.6 XML templates: same <tool_call> envelope and
    // selection logic, but the BODY grammar is the XML dialect
    // (<function=NAME><parameter=KEY> with raw-text values) — flag it so the
    // engine builds the XML FSM instead of the JSON body FSM, which masks raw
    // newlines and mangles multi-line (code) arguments.
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
    // Read under the lock, acted on after it: whether the LOADED model can see
    // images at all. A request that carries pictures a model has no tower for
    // used to fall through to the text-only path and be answered as if nothing
    // had been sent (#1198).
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

    // Refuse rather than answer from the text alone. Silently dropping the
    // image is the worst of the three options: a refusal is trivial for a
    // caller to handle, but a fluent answer about a picture the model never
    // received is indistinguishable from a real one — it reads as a verdict.
    // The load-time WARN ("the vision tower will be skipped") is in the server
    // log, not in the response, so nothing reached the client at all.
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

    // Enforced tool calling (#1002): tool_choice=required / forced function /
    // strict:true constrains generation via the schema FSM (JSON or XML body
    // per the template's dialect). Collected HERE, after ensure_model_loaded:
    // the request may have auto-loaded or switched the model, so the grammar
    // must derive from the template that actually renders this prompt — the
    // parse-time family in handlers_chat_params is a pre-load best guess.
    collect_tool_enforcement_(ctx);

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
    const bool template_think_evidence = !ctx.snap.have_template || !ctx.snap.chat_tpl.has_jinja() ||
                                         ctx.snap.chat_tpl.mentions_thinking();
    // A regex or grammar constraint is structured output too: it covers the
    // WHOLE reply, so a reasoning preamble cannot be emitted without violating
    // it — the constrainer's gate would hold the mask open while the model
    // spends the budget thinking, and the caller gets prose instead of the
    // format.
    //
    // `json_schema` belongs in that list and was missing from it, which is the
    // whole of #1431: a schema request kept thinking on, the gate held the mask
    // open for the reasoning, and on a model whose `</think>` is a multi-token
    // BPE sequence (Qwen3.8: `think_start_id` is -1) the block never closed in
    // the text the splitter reads. The answer was written inside the reasoning
    // channel and `content` came back EMPTY. Measured before the fix: 0 of 8 at
    // temperature 0, against 10 of 10 on Qwen3.6-27B, whose `</think>` IS a
    // single token.
    const bool thinking_default =
        ctx.snap.is_think_model && template_think_evidence &&
        !imp::server::structured_output_excludes_thinking(
            ctx.params.json_mode, ctx.params.has_tools, !ctx.params.json_schema_str.empty(),
            !ctx.params.regex_pattern.empty(), !ctx.params.grammar.empty());
    const bool want_thinking = ctx.params.enable_thinking_set ? ctx.params.enable_thinking_requested
                                                              : thinking_default;
    // think_budget is the fraction of max_tokens reserved for reasoning;
    // think_budget <= 0 means "no reasoning headroom" → disable thinking entirely
    // (documented "0 = disabled"). Folding it into enable_thinking keeps the two
    // flags consistent: without this, budget=0 left thinking ON yet never armed
    // the force-close, so the model reasoned to max_tokens and returned empty
    // content (#752). The Anthropic "disabled" path already zeroes the budget.
    const bool budget_disables_thinking = ctx.params.think_budget <= 0.0f;
    ctx.snap.enable_thinking = ctx.snap.is_think_model && ctx.snap.think_start_id >= 0 && want_thinking &&
                               !budget_disables_thinking;
    // Suppressing means STAMPING `enable_thinking=false` into the Jinja context,
    // which is the only thing that makes a template render its pre-closed
    // `<think></think>` block. Leaving it unstamped lets the template fall back
    // to its own default, and Qwen3.8's default is an OPEN `<think>`: the server
    // then believes thinking is off while the prompt says it is on, and the
    // reconcile step correctly flips the splitter to REASONING for a block that
    // never closes.
    //
    // So it is not enough that thinking was disabled: it has to be disabled
    // where the template can see it. Any reason to not want thinking now
    // suppresses, rather than only a zero budget.
    ctx.snap.suppress_thinking = imp::server::should_stamp_thinking_off(
        ctx.snap.is_think_model, ctx.snap.enable_thinking, budget_disables_thinking, want_thinking);

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

    // force_thinking stamps enable_thinking=true INTO the Jinja render, so a
    // template that defaults the variable to a pre-closed block (Qwen3.5-4B)
    // actually opens the think block when the caller EXPLICITLY asked for
    // thinking — otherwise `enable_thinking:true` was a silent no-op on such
    // templates. Only for an explicit request: the default case stays undefined
    // so each template author's own default wins (Qwen3 open vs Gemma-4 closed).
    const bool force_thinking = ctx.params.enable_thinking_set && ctx.params.enable_thinking_requested &&
                                ctx.snap.enable_thinking;

    // Tokenize with chat template (with image tokens if vision is active)
    if (ctx.snap.have_template && !ctx.snap.qwen_patches.empty()) {
        // The chat template renders one <|image_pad|> per block — at template
        // time nobody knows how big each image will be after smart_resize. Put
        // one block per image in front of the first user turn, render, then
        // expand each placeholder to its own count.
        //
        // All of them go on the first user turn, which is where a single image
        // already went: the request parser keeps the pictures in order but not
        // which message they came from, so this is the position that is
        // actually known rather than guessed.
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
                                                  force_thinking);
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
                                                             ctx.snap.suppress_thinking, force_thinking);
    } else if (ctx.snap.have_template && ctx.snap.tools_via_jinja) {
        std::string tc_str = ctx.params.tool_choice.is_string() ? ctx.params.tool_choice.get<std::string>()
                                                                : "auto";
        ctx.snap.tokens = ctx.snap.chat_tpl.apply_with_tools(*ctx.snap.tok, ctx.params.chat_msgs,
                                                             ctx.snap.tool_defs, tc_str,
                                                             ctx.snap.suppress_thinking, force_thinking);
        // If Jinja2 tools render failed, fall back to text-based tool prompt injection
        if (ctx.snap.tokens.empty()) {
            IMP_LOG_INFO("Jinja2 tools path failed, falling back to text-based tool prompt");
            // The text fallback advertises the ChatML JSON body — an armed XML
            // body constraint would fight the prompt. Drop to the hint (not to
            // the JSON FSM: an XML-finetuned model would fight that grammar
            // too). bare_args is never set on this path (xml implies the
            // ChatML collectors matched, see collect_tool_enforcement_).
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
                                                      ctx.snap.suppress_thinking, force_thinking);
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
                                                  ctx.snap.suppress_thinking, force_thinking);
    } else {
        // Concatenate all message content as raw text
        std::string raw;
        for (const auto& m : ctx.params.chat_msgs)
            raw += m.content + "\n";
        ctx.snap.tokens = ctx.snap.tok->encode(raw);
    }

    // cache_control per-breakpoint boundary (#1046): re-render the leading
    // messages up to the marked block and count tokens — the engine then pins
    // only that many prompt tokens (rounded down to full KV blocks) instead
    // of the whole prompt. Render jitter vs. the full prompt (generation
    // prompt, template joins) is at most ~a block; pins are eviction
    // protection, not correctness state, so approximate is fine.
    if (ctx.params.cache_prompt && ctx.params.cache_prefix_messages > 0 &&
        ctx.params.cache_prefix_messages < static_cast<int>(ctx.params.chat_msgs.size()) &&
        ctx.snap.have_template) {
        std::vector<imp::ChatMessage> prefix_msgs(
            ctx.params.chat_msgs.begin(),
            ctx.params.chat_msgs.begin() + ctx.params.cache_prefix_messages);
        ctx.snap.pin_prefix_tokens = static_cast<int>(
            ctx.snap.chat_tpl
                .apply(*ctx.snap.tok, prefix_msgs, ctx.snap.suppress_thinking, force_thinking)
                .size());
    }

    // Thinking-state pipeline (single source of truth = the rendered prompt):
    //   1. INTENT  — `want_thinking` above: explicit request, else heuristic.
    //   2. RENDER  — apply() stamps enable_thinking only when we force/suppress
    //      (above); otherwise the template author's own default decides.
    //   3. GROUND TRUTH — the block below reconciles enable_thinking against what
    //      the render actually produced in the prompt tail (open <think> prefix →
    //      thinking on; pre-closed block → off), then force-appends <think> for
    //      templateless think-models. Stages 2–3 keep the splitter's start phase
    //      consistent with the bytes the model will actually continue from.
    //
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
    // Reconcile enable_thinking with what the template actually rendered — the
    // prompt tail is ground truth. An OPEN prefix (<think>, no </think>) is
    // mid-reasoning and turns thinking ON; a pre-closed block (<think> AND
    // </think>) means the template chose answer-directly and turns it OFF, even
    // if the heuristic defaulted it ON (#934: Qwen3.5-4B mentions enable_thinking
    // but defaults to a closed block — without this the whole answer is trapped
    // in reasoning_content). Window 16 so both tags of the adjacent closed block
    // fall in the same tail (a lone in-window "<think>" would else read as open).
    {
        const bool tail_has_think = prompt_tail_contains("<think>", 16);
        const bool tail_has_close = prompt_tail_contains("</think>", 16);
        const bool was_thinking = ctx.snap.enable_thinking;
        ctx.snap.enable_thinking = imp::server::reconcile_thinking_with_prompt_tail(
            ctx.snap.enable_thinking, ctx.params.enable_thinking_set, tail_has_think, tail_has_close);
        // If the reconcile turned thinking OFF after the snapshot had removed the
        // provisional <think> stop token (removed above while it was ON), restore
        // it: a non-thinking think-model still needs the phantom-"<think>"-turn
        // guard. (The upgrade direction keeps the prior behavior untouched.)
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

    // Predicted Outputs: tokenize the prediction text while we hold the
    // tokenizer. Plain encode (no template, no specials) — these tokens only
    // seed the n-gram draft corpus, they are never forwarded. Clamped to the
    // model context so a hostile prediction can't blow up the host-side scan.
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

// Single params->request mapping for the ctx-based submission sites (see
// handlers_internal.h).
std::shared_ptr<imp::Request> build_imp_request_(const ChatRequestContext& ctx,
                                                 const std::vector<int32_t>& input_tokens, int completion_idx,
                                                 bool stream) {
    auto req = std::make_shared<imp::Request>();
    req->image = ctx.snap.vision_image;         // per-request vision (null for text)
    req->qwen_patches = ctx.snap.qwen_patches;  // dynamic-resolution route (empty otherwise)
    req->vision_content_hash = ctx.snap.vision_content_hash;
    req->input_tokens = input_tokens;
    req->max_tokens = ctx.params.max_tokens;
    req->temperature = ctx.params.temperature;
    req->top_p = ctx.params.top_p;
    req->top_k = ctx.params.top_k;
    req->seed = (ctx.params.seed != -1) ? ctx.params.seed + completion_idx : -1;
    req->pin_kv_prefix = ctx.params.cache_prompt;
    req->pin_kv_prefix_tokens = ctx.snap.pin_prefix_tokens;
    req->spec_override = ctx.params.spec_override;
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
    // Generation starts INSIDE the think block when the prompt carries the
    // <think> prefix (template-injected or server-appended). Without this
    // the engine's think-budget enforcement never sees an opener in the
    // output, counts zero reasoning tokens, and lets the model think until
    // max_tokens (content empty, finish=length).
    req->started_in_think = ctx.snap.enable_thinking;
    req->in_think_block = ctx.snap.enable_thinking;
    // Stream requests stay on per-step decode for real per-token SSE (#754).
    req->stream = stream;
    req->status = imp::RequestStatus::PENDING;
    return req;
}

// Non-streaming chat completion: run n_completions independent generations
// sequentially via the batching engine, build the choices array with
// reasoning_content / tool_calls / logprobs as appropriate, send a single
// JSON response. Caller has already submitted server_req via state.batching.
void nonstream_chat_response_(httplib::Response& res, ServerState& state, ChatRequestContext& ctx,
                              std::shared_ptr<imp::Request>& imp_req,
                              std::shared_ptr<ServerRequest>& server_req,
                              const std::vector<int32_t>& saved_tokens, const std::string& comp_id,
                              int64_t created) {
    // Non-streaming: decode all tokens, return complete response
    // For n > 1, run multiple independent generations sequentially
    json choices = json::array();
    int total_output_tokens = 0;

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

        // Admission refusal (I6): the KV pool can never hold this prompt, so no
        // amount of waiting or retrying at the same length helps. Returning a
        // 200 with an empty completion — which is what a generic "cancelled"
        // produced — makes an unservable request look like a model that chose
        // to say nothing. 503 with the reason is the honest answer, and it is
        // the code a client is expected to back off / re-route on.
        if (std::strcmp(finish, "capacity") == 0) {
            // On a pool that fell back to its rescue floor, "shorten the prompt"
            // is advice that cannot be followed: the pool holds a few hundred
            // tokens and the fault is the startup, not the request. Say which
            // of the two situations this is, or the caller tunes the prompt
            // against a server that will refuse every length worth sending.
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

        int n_output_tokens = static_cast<int>(output_ids.size());
        total_output_tokens += n_output_tokens;
        std::string content = !ctx.params.stop_sequences.empty() ? output_text
                                                                 : ctx.snap.tok->decode(output_ids);
        // max_tokens can stop mid-codepoint; the streaming path holds those
        // bytes back, so this one must drop them or the two transports return
        // different text for the same request (#1310).
        drop_incomplete_utf8_tail(content);

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
        // Chat shape here; /v1/completions builds the other one (#1589). Both
        // come out of utils.cpp now, so the two cannot drift apart again.
        json logprobs_obj = nullptr;
        if (ctx.params.req_logprobs && active_req) {
            logprobs_obj = chat_logprobs_json(active_req->output_logprobs, output_ids.size());
        }

        // Parse tool calls from model output. Run even on finish=length:
        // the model may have emitted a complete tool_call and then kept
        // generating until the budget ran out (common before we hook the
        // family-specific close marker as a stop token). The parser is
        // tolerant of trailing garbage after the closing marker.
        std::vector<ParsedToolCall> tool_calls;
        std::string tool_validation_error;
        if (ctx.params.has_tools) {
            auto [pre_content, parsed_calls] =
                parse_tool_calls(ctx.snap.tpl_family, content, state.next_tool_call_id,
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
        // An empty answer beside a full reasoning channel is not a defect, and
        // it reads exactly like one. The reply shares the token budget with the
        // thinking, so once the thinking fills it the answer never starts, and
        // the caller sees `content: ""` with `finish_reason: stop`. Measured on
        // Qwen3.8-27B: a 74-turn session returns empty replies at max_tokens
        // 260 and is 74/74 clean at 600 (docs/TROUBLESHOOTING.md). Say which of
        // the two it was, rather than leave someone bisecting an engine that
        // did what it was asked.
        if (answer_lost_to_reasoning(!tool_calls.empty(), content, reasoning_content)) {
            IMP_LOG_WARN(
                "empty content: the answer never started because the token budget went "
                "to reasoning (%zu chars of thinking, finish_reason=%s). Raise "
                "max_tokens — a thinking model needs room to answer AFTER it thinks.",
                reasoning_content.size(), finish);
        }
        if (!tool_validation_error.empty()) {
            msg["tool_call_validation_error"] = tool_validation_error;
        }

        json choice = {{"index", ci}, {"message", msg}, {"finish_reason", openai_finish_reason(finish)}};
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
    // Also carries `evicted_tokens` when StreamingLLM dropped context mid-run.
    if (json details = prompt_tokens_details_(imp_req, ctx.snap.n_prompt_tokens); !details.is_null())
        usage["prompt_tokens_details"] = std::move(details);
    if (imp_req && (imp_req->cached_tokens > 0 || imp_req->pin_kv_prefix))
        state.metrics.tokens_cached_total += imp_req->cached_tokens;
    // Predicted Outputs accounting (only when the request carried a
    // prediction): accepted/rejected draft tokens whose draft came from the
    // prediction region of the n-gram corpus.
    if (imp_req && !imp_req->prediction_tokens.empty()) {
        usage["completion_tokens_details"] = {{"accepted_prediction_tokens", imp_req->pred_accepted},
                                              {"rejected_prediction_tokens", imp_req->pred_rejected}};
    }

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
    log_request_jsonl(state, ctx.log_skip, ctx.t_log_start, comp_id, ctx.log_endpoint, ctx.log_client_ip,
                      ctx.log_raw_body, ms, ctx.snap.n_prompt_tokens, total_output_tokens, nonstream_finish,
                      response);

    res.set_content(dump_safe(response), "application/json");
}
