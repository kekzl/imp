#pragma once

// Internal (server-private) shared declarations for the imp-server HTTP
// handlers. These were file-local (anonymous-namespace structs + `static`
// helpers) inside the original monolithic handlers.cpp. handlers.cpp was split
// across several translation units to cut recompile blast radius (chat core /
// chat streaming / chat endpoints / Anthropic messages / misc endpoints), and
// the pieces shared between those TUs live here.
//
// This header is NOT part of the public handler API — that stays in handlers.h.

#include "handlers.h"

#include "api/imp_internal.h"
#include "vision/image_processor.h"
#include "runtime/request.h"
#include "memory/kv_cache.h"

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
    bool top_p_explicit = false, top_k_explicit = false, rep_pen_explicit = false;
    // Pin the prompt's KV blocks against eviction (Anthropic cache_control →
    // mapped by anthropic_to_openai_body; also a direct llama.cpp-style
    // "cache_prompt" body field on the OpenAI route).
    bool cache_prompt = false;
    // cache_control breakpoint boundary (#1046): number of leading chat
    // messages that form the cacheable prefix (-1 = whole prompt). Set by
    // anthropic_to_openai_body ("cache_prefix_messages"); shifted when a
    // system message is injected in front (tool-prompt fallback).
    int cache_prefix_messages = -1;
    // Per-request n-gram speculation override (tri-state): -1 = server default,
    // 0 = force off, 1 = force on. From the imp extension body field
    // "speculative" (bool). Lets code-gen calls opt into speculation while
    // short tool-arg generations skip it on the same server.
    int spec_override = -1;
    // OpenAI Predicted Outputs: concatenated text of the "prediction" body
    // field ({"type":"content","content": string | [{"type":"text","text"}]}).
    // Tokenized later in the snapshot stage (needs the tokenizer) and fed to
    // the n-gram draft corpus. Empty = no prediction.
    std::string prediction_text;
    bool enable_thinking_requested = false;  // value of "enable_thinking" if present
    // "reasoning_effort" body field, passed through to the chat template
    // verbatim (empty = not sent, template default applies). Legal values are
    // the template's business, not ours: Qwen3.8 takes xhigh/medium/low, the
    // OpenAI convention is low/medium/high.
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
    // Qwen3-VL takes the other route: dynamic-resolution images are patchified
    // here (CPU only) and their token counts are known BEFORE tokenizing,
    // because the prompt has to reserve exactly that many placeholders — and
    // each picture reserves its own number, so this stays a list.
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
    // cache_control breakpoint boundary in TOKENS (#1046): -1 = pin the whole
    // prompt; >=0 = pin only the first N prompt tokens' full KV blocks.
    // Computed by re-rendering the leading params.cache_prefix_messages
    // messages (tokenizer lives in the snapshot stage).
    int pin_prefix_tokens = -1;
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
    bool log_skip = false;
    std::shared_ptr<imp::Request> imp_req;
    std::shared_ptr<ServerRequest> server_req;
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

// usage.prompt_tokens_details — prefix-cache accounting, plus `evicted_tokens`:
// context this request LOST to StreamingLLM eviction while it was generating.
// That last one is the point of the field existing at all. When the KV pool
// runs out mid-generation the engine frees the middle of the sequence and keeps
// decoding; it logs a WARN, but a server-side WARN is not something the caller
// sees, and the answer it gets back was written against less context than it
// sent. `evicted_tokens` is how the caller finds out (roadmap gap 6).
//
// Returns a null json when there is nothing to report, so the field appears
// only when it says something.
inline nlohmann::json prompt_tokens_details_(const std::shared_ptr<imp::Request>& req, int n_prompt_tokens) {
    if (!req)
        return nlohmann::json();
    nlohmann::json details = nlohmann::json::object();
    if (req->cached_tokens > 0 || req->pin_kv_prefix) {
        details["cached_tokens"] = req->cached_tokens;
        const int creation = cache_creation_tokens_(req, n_prompt_tokens);
        if (creation > 0)
            details["cache_creation_tokens"] = creation;
    }
    if (req->evicted_kv_tokens > 0)
        details["evicted_tokens"] = req->evicted_kv_tokens;
    if (details.empty())
        return nlohmann::json();
    return details;
}

// Set true on the calling thread when a shim handler (handle_messages,
// handle_responses, handle_count_tokens) is delegating to
// handle_chat_completions — suppresses inner request-log entries so the
// call only logs once at the outer handler. Defined in handlers_chat_core.cpp.
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

// Rejects a tool_choice that contradicts the request (names a tool that is not
// there, or demands a call with no tools).
bool validate_tool_choice(const json& body, httplib::Response& res);

// Defined in handlers_chat_core.cpp. client_request_id: sanitized client
// X-Request-Id (empty = none sent; written as "client_request_id" so an
// external trace joins the server req_id).
void log_request_jsonl(ServerState& state, bool skip, const std::chrono::system_clock::time_point& t_start,
                       const std::string& req_id, const std::string& endpoint, const std::string& client_ip,
                       const std::string& raw_body, double latency_ms, int prompt_tokens,
                       int completion_tokens, const char* finish_reason, const json& response_body,
                       const std::string& client_request_id = "");
bool parse_chat_request_params(const httplib::Request& req, httplib::Response& res, ServerState& state,
                               ChatRequestContext& ctx);
bool snapshot_state_and_tokenize_(httplib::Response& res, ServerState& state, ChatRequestContext& ctx);
// Build an imp::Request from a parsed+snapshotted chat request context — the
// single params->request mapping for all four ctx-based submission sites
// (chat streaming + non-streaming, /v1/messages streaming, /v1/responses
// streaming); this was hand-copied per site and drifted (#941).
// completion_idx offsets the seed for n>1 choice generation; stream keeps the
// request on per-step decode for real per-token SSE (#754).
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
