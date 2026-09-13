#pragma once

// Internal (server-private) declarations shared across the handlers.cpp split (chat core/stream/
// endpoints, Anthropic messages, misc). NOT the public handler API - that is handlers.h.

#include "handlers.h"
#include "spec_usage_keys.h"

#include "api/imp_internal.h"
#include "vision/image_processor.h"
#include "runtime/request.h"
#include "runtime/spec_request.h"
#include "memory/kv_cache.h"

#include <algorithm>
#include <chrono>
#include <memory>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Chat completion context (bundles state for handle_chat_completions phases)
// ---------------------------------------------------------------------------

// Body-parsed input parameters (no lock needed to populate).
struct ChatRequestParams {
    // Sampling
    float temperature = 0.7f, top_p = 0.95f, min_p = 0.0f, typical_p = 1.0f;
    float repetition_penalty = 1.05f;
    float frequency_penalty = 0.0f, presence_penalty = 0.0f;
    float dry_multiplier = 0.0f, dry_base = 1.75f;
    float mirostat_tau = 5.0f, mirostat_eta = 0.1f;
    float think_budget = 0.0f;
    int top_k = 40, max_tokens = 0, seed = -1, repeat_last_n = 0;
    int dry_allowed_length = 2, dry_penalty_last_n = 0, mirostat = 0;
    int n_completions = 1, top_logprobs = 0;
    // Admission priority (vLLM-compatible "priority" body field): lower value
    // schedules earlier, default 0. Strictly dominates the scheduler's
    // shortest-first-with-aging order across classes.
    int priority = 0;
    bool stream = false, json_mode = false, req_logprobs = false, include_usage = false;
    bool ignore_eos = false;  // vLLM-style: run to max_tokens, never stop on EOS (benchmarks)
    bool top_p_explicit = false, top_k_explicit = false, rep_pen_explicit = false;
    // Pin the prompt's KV blocks against eviction (Anthropic cache_control →
    // mapped by anthropic_to_openai_body; also a direct llama.cpp-style
    // "cache_prompt" body field on the OpenAI route).
    bool cache_prompt = false;
    // cache_prefix_messages: leading chat-message count forming the cacheable prefix (-1 = whole
    // prompt). Set by anthropic_to_openai_body; shifted when a system message is injected in front
    // (tool-prompt fallback).
    int cache_prefix_messages = -1;
    // spec_override: per-request n-gram speculation tri-state (-1 server default, 0 off, 1 on) from
    // the "speculative" bool body field.
    int spec_override = -1;
    // spec_mtp_k: per-request MTP chain depth from {"speculative":{"mtp_k":N}} (-1 = server
    // default). Orthogonal to spec_override - addresses the head only. Resolved against the
    // process-armed depth (src/runtime/spec_request.h); a request can lower it, never raise it.
    int spec_mtp_k = -1;
    // prediction_text: OpenAI Predicted Outputs "prediction" body field (concatenated text).
    // Tokenized in the snapshot stage and fed to the n-gram draft corpus; never forwarded as output.
    std::string prediction_text;
    bool enable_thinking_requested = false;  // value of "enable_thinking" if present
    // reasoning_effort: passed to the chat template verbatim (empty = template default). Legal
    // values are the template's business - Qwen3.8 takes xhigh/medium/low, OpenAI low/medium/high.
    std::string reasoning_effort;
    std::string lora_name;                   // "lora" body field (empty = base model)
    bool enable_thinking_set = false;        // true iff body contained "enable_thinking"
    // Stop sequences
    std::vector<std::string> stop_sequences;
    size_t max_stop_len = 0;
    // Logit bias / format
    std::vector<std::pair<int32_t, float>> logit_bias;
    std::string json_schema_str;
    // Constrain the whole reply to this regex (empty = off). Set via
    // response_format {"type":"regex","regex":...} or vLLM's guided_regex.
    std::string regex_pattern;
    // Constrain the whole reply to this GBNF grammar (empty = off). Set via
    // response_format {"type":"grammar","grammar":...}, llama.cpp's top-level
    // "grammar", or vLLM's guided_grammar.
    std::string grammar;
    // Tools
    nlohmann::json tools;
    nlohmann::json tool_choice;
    bool has_tools = false;
    bool parallel_tool_calls = true;  // OpenAI: false → emit at most one tool call
    // Enforced tool calling (#1002): filled for tool_choice=required / forced
    // function on the <tool_call>-JSON dialect; empty = prompt hint only.
    std::vector<std::pair<std::string, std::string>> tool_constraint_tools;
    std::string tool_envelope_open;
    std::string tool_envelope_close;
    // Strict OPTIONAL tool call (OpenAI strict:true, tool_choice=auto): the
    // envelope is not forced; the body FSM engages only if the model calls.
    bool tool_constraint_optional = false;
    // Llama3 `<function=NAME>{args}</function>` forced call: the constraint root
    // is the bare parameter schema (the body is the arguments object), not a
    // TOOL_CALL {"name","arguments"} wrapper.
    bool tool_constraint_bare_args = false;
    // Qwen-Coder XML dialect (template teaches <function=/<parameter= bodies):
    // enforce with the XML grammar, never the JSON body FSM.
    bool tool_constraint_xml = false;
    // Messages + image
    std::vector<imp::ChatMessage> chat_msgs;
    // Every `image_url` part in the request, in prompt order. A vector because
    // one buffer meant the last picture silently overwrote the rest: the
    // request named several, the model saw one, and nothing said so.
    std::vector<std::vector<uint8_t>> images;
    // Set when an `image_url` could not be resolved. Fatal rather than skipped:
    // dropping one image would shift every later picture onto the wrong
    // placeholder, which reads as a coherent answer about the wrong thing.
    std::string image_error;
    std::string requested_model;
};

// Lock-acquired engine state (populated under state.mtx).
struct ChatStateSnapshot {
    // MTP head facts of the load this request was admitted against, so the
    // per-request speculation contract resolves against a consistent picture
    // even if a model swap lands mid-request (spec_request.h).
    int mtp_armed_k = 0;
    bool mtp_head_present = false;
    bool mtp_head_loaded = false;
    imp::Tokenizer* tok = nullptr;
    imp::ChatTemplate chat_tpl;
    bool have_template = false;
    std::string model_name;
    bool is_think_model = false;
    int32_t think_start_id = -1, think_end_id = -1;
    int32_t channel_open_id = -1, channel_close_id = -1, channel_newline_id = -1;
    int max_seq_len = 0;
    bool has_vision_request = false;
    // Per-request vision (F-A5): CPU-preprocessed image pixels, copied to
    // req->image at every request-build site. The batch worker encodes + binds
    // it per-request (no engine pause). Null for text-only requests.
    std::shared_ptr<imp::ImageData> vision_image;
    // Qwen3-VL: dynamic-resolution images are patchified CPU-side before tokenizing (token counts
    // must be known up front to reserve exact placeholder counts per image, hence a list).
    std::vector<std::shared_ptr<imp::QwenPatches>> qwen_patches;
    // Token count per image, in prompt order — the k-th placeholder expands to
    // the k-th entry, so this must stay parallel to `qwen_patches`.
    std::vector<int> qwen_image_tokens;
    size_t vision_content_hash = 0;
    std::vector<int32_t> stop_token_ids;
    imp::ChatTemplateFamily tpl_family = imp::ChatTemplateFamily::CHATML;
    std::vector<imp::ToolFunction> tool_defs;
    bool tools_via_jinja = false;
    bool enable_thinking = false, suppress_thinking = false;
    std::string reasoning_effort;  // copied from params; stamped into the Jinja context
    std::vector<int32_t> tokens;
    int n_prompt_tokens = 0;
    // Tokenized Predicted-Outputs text (params.prediction_text) — encoded here
    // because the tokenizer only exists inside the snapshot stage.
    std::vector<int32_t> prediction_tokens;
    // pin_prefix_tokens: cache_control breakpoint in tokens (#1046). -1 = pin whole prompt; >=0 =
    // pin only the first N tokens' full KV blocks. Computed by re-rendering the leading
    // cache_prefix_messages messages.
    int pin_prefix_tokens = -1;
    // Engine adapter id the request named via `lora` (0 = base). Resolved
    // here, switched by the batching worker at admission (E-1).
    int lora_id = 0;
};

// Top-level context bundling params + snap + transients.
struct ChatRequestContext {
    ChatRequestParams params;
    ChatStateSnapshot snap;
    std::string req_id;
    std::string comp_id;
    int64_t created = 0;
    std::chrono::high_resolution_clock::time_point t_start;
    std::chrono::system_clock::time_point t_log_start;
    std::string log_endpoint, log_client_ip, log_raw_body;
    // Client-sent X-Request-Id (sanitized; empty = none). Echoed on the
    // response and written to the request JSONL, so an external trace joins
    // the server's req_id.
        std::string log_client_request_id;
    // Tracing: the incoming traceparent plus the timing the span needs
    // (queue, first token); filled where the numbers become known.
    RequestSpan trace;
    bool log_skip = false;
};

// cache_creation_input_tokens (Anthropic): full prompt blocks newly written
// and pinned by this request — block-rounded prompt minus prefix-cache hits.
inline int cache_creation_tokens_(const std::shared_ptr<imp::Request>& req, int n_prompt_tokens) {
    if (!req || !req->pin_kv_prefix)
        return 0;
    int rounded = (n_prompt_tokens / imp::kKVBlockSize) * imp::kKVBlockSize;
    int creation = rounded - req->cached_tokens;
    return creation > 0 ? creation : 0;
}

// usage.prompt_tokens_details.evicted_tokens: context lost to StreamingLLM eviction mid-
// generation (roadmap gap 6), else only a WARN the caller never sees. Null when nothing to report.
// add_spec_usage_ (AUDIT_arch_2026 C-6): per-request speculation counters, vendor-prefixed keys
// in completion_tokens_details, present only when a verify ran or mtp_k=auto declined the head.
inline void add_spec_usage_(nlohmann::json& usage, const std::shared_ptr<imp::Request>& req) {
    if (!req)
        return;
    const bool declined = imp::spec_decline_is_reportable(req->spec_decline);
    if (!declined && req->spec_verifies == 0 && req->spec_drafted == 0)
        return;
    auto& d = usage["completion_tokens_details"];
    d["imp_spec_drafted"] = req->spec_drafted;
    d["imp_spec_accepted"] = req->spec_accepted;
    d["imp_spec_emitted"] = req->spec_emitted;
    d["imp_spec_verify_steps"] = req->spec_verifies;
    if (declined) {
        d["imp_spec_declined"] = imp::spec_decline_name(req->spec_decline);
        d["imp_spec_declined_detail"] = imp::spec_decline_detail(req->spec_decline);
    }
}

// SpecFieldParse: "speculative" body field, true/false (every drafter on/off) or {"mtp_k":N}
// (0<=N<=armed, head only). armed_mtp_k bounds the accepted range (0 = no head); without a
// model loaded the bound is the device chain cap, so this is testable model-less.
struct SpecFieldParse {
    bool ok = true;
    int spec_override = -1;
    int mtp_k = -1;
    std::string error;
};

// The three MTP facts the contract below needs, read once per request off the
// server atomics (a model swap can move them mid-request).
template <typename State, typename Snap>
inline void snapshot_mtp_state_(const State& state, Snap& snap) {
    snap.mtp_armed_k = state.armed_mtp_k.load(std::memory_order_relaxed);
    snap.mtp_head_present = state.mtp_head_present.load(std::memory_order_relaxed);
    snap.mtp_head_loaded = state.mtp_head_loaded.load(std::memory_order_relaxed);
}

// Resolve the per-request speculation contract onto an engine Request. One
// helper so the three submission sites (chat core, /v1/completions, and the
// dialect shims through the first) cannot resolve it three ways.
inline void apply_spec_contract_(imp::Request& req, int spec_override, int spec_mtp_k, int armed_k,
                                 bool head_present, bool head_loaded) {
    req.spec_override = spec_override;
    req.spec_mtp_k = spec_mtp_k;
    imp::MtpRequestState ms;
    ms.requested_k = spec_mtp_k;
    ms.armed_k = armed_k;
    ms.head_present = head_present;
    ms.head_loaded = head_loaded;
    ms.forced_off = spec_override == 0;
    req.spec_decline = imp::mtp_resolve_request(ms).reason;
}

inline SpecFieldParse parse_spec_field_(const nlohmann::json& body, int armed_mtp_k) {
    SpecFieldParse out;
    if (!body.contains("speculative"))
        return out;
    const auto& sp = body["speculative"];
    if (sp.is_boolean()) {
        out.spec_override = sp.get<bool>() ? 1 : 0;
        return out;
    }
    if (!sp.is_object()) {
        out.ok = false;
        out.error = "\"speculative\" must be a boolean or an object of the form {\"mtp_k\": N}";
        return out;
    }
    if (!sp.contains("mtp_k"))
        return out;  // an empty object asks for nothing
    const auto& k = sp["mtp_k"];
    const int ceiling = armed_mtp_k > 0 ? armed_mtp_k : imp::kSpecRequestMaxMtpK;
    if (!k.is_number_integer()) {
        out.ok = false;
        out.error = "\"speculative.mtp_k\" must be an integer in 0.." + std::to_string(ceiling);
        return out;
    }
    const int v = k.get<int>();
    if (v < 0 || v > ceiling) {
        out.ok = false;
        out.error = "\"speculative.mtp_k\" is " + std::to_string(v) + ", outside the accepted range 0.." +
                    std::to_string(ceiling) +
                    (armed_mtp_k > 0 ? " (the MTP chain depth this server armed)"
                                     : " (no MTP head is armed; the bound is the device chain cap)");
        return out;
    }
    out.mtp_k = v;
    return out;
}

inline nlohmann::json prompt_tokens_details_(const std::shared_ptr<imp::Request>& req, int n_prompt_tokens) {
    if (!req)
        return nlohmann::json();
    // cached_tokens is always present, 0 on a miss: OpenAI sends the field with
    // 0 and a client probing once at startup reads "absent" as "unsupported" (#1980).
    nlohmann::json details = {{"cached_tokens", std::max(0, req->cached_tokens)}};
    if (req->cached_tokens > 0 || req->pin_kv_prefix) {
        const int creation = cache_creation_tokens_(req, n_prompt_tokens);
        if (creation > 0)
            details["cache_creation_tokens"] = creation;
    }
    if (req->evicted_kv_tokens > 0)
        details["evicted_tokens"] = req->evicted_kv_tokens;
    return details;
}

// g_in_anthropic_shim: set when a shim handler (messages/responses/count_tokens) delegates to
// handle_chat_completions, suppressing the inner request-log entry so the call logs once.
extern thread_local bool g_in_anthropic_shim;

// Set by the non-streaming path to the stop sequence that ended the
// generation, empty otherwise. Read by the Anthropic shim, which cannot
// recover it from the OpenAI body (#1550).
extern thread_local std::string g_shim_stop_sequence;

// ---------------------------------------------------------------------------
// Shared server-private helpers (definitions split across handlers_*.cpp).
// ---------------------------------------------------------------------------

// Defined in handlers.cpp.
bool ensure_model_loaded(ServerState& state, const std::string& requested_model, httplib::Response& res);
bool validate_sampling_params(const json& body, httplib::Response& res);

// Rejects a constraint imp cannot compile with 400 instead of answering
// unconstrained (#1256). Called from validate_sampling_params.
bool validate_constraints(const json& body, httplib::Response& res);

// Rejects a content part this server cannot read (video_url, a malformed
// image_url part) instead of answering as if it had been understood.
bool validate_content_parts(const json& body, httplib::Response& res);

// Same content-part rule as validate_content_parts, for the Anthropic body, run BEFORE
// anthropic_to_openai_body deletes an unreadable block and leaves nothing to find downstream.
bool anthropic_unreadable_block(const json& body, std::string& why);

// Rejects a tool_choice that contradicts the request (names a tool that is not
// there, or demands a call with no tools).
bool validate_tool_choice(const json& body, httplib::Response& res);

// client_request_id: sanitized client X-Request-Id (empty = none), written as
// "client_request_id" so an external trace joins the server req_id. trace carries the
// OTLP timing/traceparent from the same accounting point.
void log_request_jsonl(ServerState& state, bool skip, const std::chrono::system_clock::time_point& t_start,
                       const std::string& req_id, const std::string& endpoint, const std::string& client_ip,
                       const std::string& raw_body, double latency_ms, int prompt_tokens,
                       int completion_tokens, const char* finish_reason, const json& response_body,
                                              const std::string& client_request_id = "", const RequestSpan* trace = nullptr);
bool parse_chat_request_params(const httplib::Request& req, httplib::Response& res, ServerState& state,
                               ChatRequestContext& ctx);
bool snapshot_state_and_tokenize_(httplib::Response& res, ServerState& state, ChatRequestContext& ctx);
// build_imp_request_: single params->request mapping for all four ctx-based submission sites
// (chat stream/non-stream, /v1/messages stream, /v1/responses stream) - was hand-copied per
// site and drifted (#941). completion_idx offsets the seed for n>1; stream forces per-step decode (#754).
std::shared_ptr<imp::Request> build_imp_request_(const ChatRequestContext& ctx,
                                                 const std::vector<int32_t>& input_tokens, int completion_idx,
                                                 bool stream);
void nonstream_chat_response_(httplib::Response& res, ServerState& state, ChatRequestContext& ctx,
                              std::shared_ptr<imp::Request>& imp_req,
                              std::shared_ptr<ServerRequest>& server_req,
                              const std::vector<int32_t>& saved_tokens, const std::string& comp_id,
                              int64_t created);

// Defined in handlers_chat_stream.cpp.
void stream_chat_response_(httplib::Response& res, ServerState& state, ChatRequestContext& ctx,
                           const std::shared_ptr<ServerRequest>& server_req);
bool run_chat_stream_(httplib::DataSink& sink, ChatRequestContext& ctx, ServerState& state,
                      const std::shared_ptr<ServerRequest>& server_req);

// Defined in handlers_messages.cpp.
bool run_anthropic_stream_(httplib::DataSink& sink, ChatRequestContext& ctx, ServerState& state,
                           const std::shared_ptr<ServerRequest>& server_req, const std::string& anth_model,
                           const std::string& msg_id, bool omit_thinking);
