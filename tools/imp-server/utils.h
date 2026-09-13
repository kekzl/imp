#pragma once

#include <httplib.h>
#include <nlohmann/json.hpp>

#include "runtime/request.h"

#include <functional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <cstddef>
#include <cstdint>

using json = nlohmann::json;

// dump_safe: serializes without throwing on invalid UTF-8 (nlohmann's dump() throws
// type_error.316 on ill-formed bytes, which client/model text routinely contains). Invalid bytes
// become U+FFFD; well-formed input dumps byte-identical to dump(). Use for ANY body carrying
// client- or model-supplied text.
std::string dump_safe(const json& j);

// Printable-ASCII, length-capped copy of a client-supplied string, for the
// cases where one is echoed back into a response (#1618).
std::string sanitize_for_echo(std::string_view in, size_t max_len);

// Utf8Stitch rejoins a multi-byte character a BPE tokenizer split across two tokens: each
// streamed delta is serialized alone, so dump_safe sees half a character and emits U+FFFD
// (affects any non-ASCII script) while the non-streaming path, which decodes all tokens
// together, is correct. feed() emits the safe part and holds an incomplete trailing sequence;
// bytes still held when the stream ends are dropped (no character to show).
class Utf8Stitch {
public:
    std::string feed(const std::string& piece);

private:
    std::string carry_;
};

// json_nesting_depth (#1607): counts JSON nesting depth WITHOUT parsing (the parse itself is
// where the stack dies - 50000 nested arrays parse fine, 100000 segfault, ~100KB vs a 100MiB body
// cap). Scans left to right skipping string contents, stops once `stop_at` is exceeded so a
// hostile body costs only as many bytes as needed to prove it hostile.
int json_nesting_depth(const std::string& body, int stop_at);

// reject_body_too_deep (#1607): rejects an over-nested body with the dialect's own error
// envelope. Cannot live in the pre-routing handler - httplib calls that before the body is read
// (req.body is empty there); a 10000-level body measured 200 when tried there first.
bool reject_body_too_deep(const httplib::Request& req, httplib::Response& res);

// utf8_chunk_len (#1554): chunk length at most `max` bytes AND ending on a codepoint boundary.
// Tool arguments were sliced every 48 bytes independently, splitting multibyte characters into
// U+FFFD (the per-token path has stitched since #1310, this path did not). Always returns >=1
// when bytes remain; `max` yields to a whole character when none fits within it.
size_t utf8_chunk_len(const std::string& s, size_t off, size_t max);

// send_json_error: sends the OpenAI error envelope via dump_safe, so an invalid-UTF-8 byte
// echoed into the message (e.g. a parse-error what()) cannot throw and turn a 400 into a bare
// 500. `param`/`code` are optional but make the error machine-readable (#1595) - without them a
// context-window refusal, a bad argument, and an auth failure differ only in English prose.
void send_json_error(httplib::Response& res, int status, const char* type, const std::string& message,
                     const char* param = nullptr, const char* code = nullptr);

// kMaxPromptBytesPerToken=16: --max-input-tokens is checked AFTER tokenizing, and tokenizing
// itself is the cost (BPE walk over a 100MiB body on a worker thread, AUDIT_arch_2026 F2-9) -
// this pre-refuses a body that cannot pass the token check anyway. 16 bytes/token is a bound, not
// an estimate (English ~4, code ~3; only long whitespace runs beat it).
inline constexpr size_t kMaxPromptBytesPerToken = 16;
inline bool prompt_bytes_exceed_input_budget(size_t bytes, int max_input_tokens) {
    return max_input_tokens > 0 && bytes > static_cast<size_t>(max_input_tokens) * kMaxPromptBytesPerToken;
}
// The check plus the 400 for it, so every prompt-taking handler spends one line.
// `param` names the request field in the envelope.
bool prompt_within_input_budget(httplib::Response& res, size_t bytes, int max_input_tokens,
                                const char* param);

// servable_context_tokens: the smaller of the resolver's plan and what the KV pool actually
// holds - they can differ hugely once the pool is clamped after planning (97204 vs 52256,
// Qwen3.8-27B-NVFP4 example), and a prompt between the two used to be accepted and then fail
// (#1542). kv_capacity_tokens<=0 means unknown; the plan stands.
int servable_context_tokens(int planned_max_seq_len, long long kv_capacity_tokens);

// kv_capacity_ceiling_tokens: what a growable pool MAY reach (the ceiling), not what is
// committed now - kv_cache.growable_initial_pct=25 makes the committed count a moving floor
// (e.g. 875 of a 13264-block ceiling). A fixed pool has ceiling == total.
long long kv_capacity_ceiling_tokens(int total_blocks, int ceiling_blocks, int block_size);

// is_anthropic_path: Anthropic endpoints use a different error envelope
// ({"type":"error","error":{...}}). Four hand-spelled checks in main.cpp (two of them missing)
// let a 429 on /v1/messages come back OpenAI-shaped, unclassifiable by an Anthropic SDK (#1551).
bool is_anthropic_path(const std::string& path);

// send_anthropic_error: `type` must be one of Anthropic's defined error types
// (invalid_request_error, authentication_error, billing_error, permission_error,
// not_found_error, request_too_large, rate_limit_error, api_error, overloaded_error,
// timeout_error) - server_error/capacity_error are not among them and were emitted at seven
// sites (#1556). request_id (#1561) is what support/log correlation asks for first.
void send_anthropic_error(httplib::Response& res, int status, const char* type, const std::string& message,
                          const std::string& request_id = {});

// anthropic_error_type_for: translates an OpenAI-dialect error.type to Anthropic's, since the
// non-streaming /v1/messages path forwards the OpenAI handler's type verbatim otherwise (#1556).
// Unrecognized types fall back on status: 5xx -> api_error, else invalid_request_error.
const char* anthropic_error_type_for(std::string_view openai_type, int status);

// Send whichever envelope `path` calls for. `openai_type` and `anthropic_type`
// are the two dialects' names for the same condition.
void send_dialect_error(httplib::Response& res, const std::string& path, int status, const char* openai_type,
                        const char* anthropic_type, const std::string& message,
                        const std::string& request_id = {});

// bearer_token_matches: constant-time compare (runs the full expected length regardless of where
// the input differs - std::string::operator== short-circuits and would leak the key prefix via
// timing). Extracted from main.cpp's pre-routing handler so the security-critical compare is
// unit-testable.
bool bearer_token_matches(const std::string& authorization, const std::string& api_key);

// answer_lost_to_reasoning: true when a reply has no tool calls, empty content, and non-empty
// reasoning - not a defect, the reply shares the token budget with thinking (measured on
// Qwen3.8-27B: empty at max_tokens 260, clean 74/74 at 600, docs/TROUBLESHOOTING.md). Split out
// because the state depends on how long the model chooses to think, so it needs a dedicated test, not a run.
bool answer_lost_to_reasoning(bool has_tool_calls, const std::string& content, const std::string& reasoning);

// The same predicate over the three FACTS, so the streaming path (which never
// holds the finished strings, only "did any content byte go out") asks the same
// question as the non-streaming one. One source of truth for four emitters.
inline bool answer_lost_to_reasoning_flags(bool has_tool_calls, bool content_empty, bool has_reasoning) {
    return !has_tool_calls && content_empty && has_reasoning;
}

// The wire value of the exhaustion signal, or nullptr when the request does not
// qualify. `finish_reason` / `stop_reason` keep their upstream enums, so this
// rides beside them as an imp-namespaced extra.
inline const char* reasoning_finish_detail(bool has_tool_calls, bool content_empty, bool has_reasoning) {
    return answer_lost_to_reasoning_flags(has_tool_calls, content_empty, has_reasoning)
               ? "reasoning_budget_exhausted"
               : nullptr;
}

// attach_reasoning_finish_detail: the ONE site that writes imp_finish_detail onto a response
// object. Kept with the decision on purpose - the handler TUs have no CPU test target, so a
// mutant emitting the field unconditionally must get past this function (test_sse_stream_utils).
inline void attach_reasoning_finish_detail(json& obj, bool has_tool_calls, bool content_empty,
                                           bool has_reasoning) {
    if (const char* detail = reasoning_finish_detail(has_tool_calls, content_empty, has_reasoning))
        obj["imp_finish_detail"] = detail;
}

// report_answer_lost_to_reasoning: answer_lost_to_reasoning plus the server-side WARN naming
// which situation applies. Returns the decision so the caller attaches the wire signal
// (imp_finish_detail) and the metric without evaluating the predicate twice.
bool report_answer_lost_to_reasoning(bool has_tool_calls, const std::string& content,
                                     const std::string& reasoning, const char* finish);

// nonstream_reasoning_tokens: counts the same tokens two ways. think_end_id>=0 -> the engine's
// own exact recount over output ids. think_end_id<0 (</think> split across BPE pieces, no single
// id) -> charges the leading tokens whose decoded bytes cover reasoning_chars (an estimate).
int nonstream_reasoning_tokens(const std::vector<int32_t>& output_ids, int32_t think_start_id,
                               int32_t think_end_id, bool started_in_think, size_t reasoning_chars,
                               const std::function<size_t(int32_t)>& decoded_len);

// health_unservable_reason: why the server cannot serve, or "" when it can - distinct from
// whether the last request failed. Transient OOM keeps /health at 200 on purpose (restarting
// would make it worse); a KV pool floored at init stays unservable for the process's lifetime
// (every prompt past a few hundred tokens is cancelled at admission) and only a restart on a
// freed card fixes it, which is what 503 tells an orchestrator to do. health_unservable_code()
// is the machine-readable half a client needs to know not to retry.
std::string health_unservable_reason(bool engine_faulted, bool kv_pool_floored, int kv_blocks,
                                     int kv_block_size);

// The stable identifier for the same state, "" when the server can serve.
// Values: "engine_faulted", "kv_pool_floored".
const char* health_unservable_code(bool engine_faulted, bool kv_pool_floored);

// api_key_matches: accepts either OpenAI's `Authorization: Bearer` or Anthropic's `x-api-key`
// header (the official Anthropic SDK sends the latter) - a Bearer-only check 401s real Anthropic
// clients on /v1/messages. Both comparisons are constant-time.
bool api_key_matches(const std::string& authorization, const std::string& x_api_key,
                     const std::string& api_key);

// openai_finish_reason: maps the engine's "cancelled"/"capacity" (not in OpenAI's enum) onto
// "length" - both used to ship verbatim on a 200, so a client switching on the enum fell to its
// default branch and treated a failed generation as normal (#1590). "length" is what makes a
// client retry/shorten rather than accept the text; non-streaming chat answers "capacity" with
// 503 before reaching here, this is the backstop for paths that don't.
const char* openai_finish_reason(const char* engine_finish);

// system_fingerprint: hash of engine version + loaded model, so a client can notice the backend
// changed under it (model swap, quant change, server upgrade) - emitted nowhere before #1602.
// Stable for the life of a configuration, different across any change to either input.
std::string system_fingerprint(const std::string& model_name);

json safe_token_json(const std::string& text);
json token_bytes_json(const std::string& text);

// Chat and Completions logprobs are DIFFERENT shapes: Chat is
// {"content":[{"token","logprob","bytes","top_logprobs"}]}, Completions is
// {"tokens":[],"token_logprobs":[],"top_logprobs":[{tok:lp}],"text_offset":[]}.
// /v1/completions returned the Chat shape until #1589. `text_offset` is byte offset from `text`'s start.
json chat_logprobs_json(const std::vector<imp::TokenLogprobInfo>& lps, size_t limit);
json completions_logprobs_json(const std::vector<imp::TokenLogprobInfo>& lps, size_t limit,
                               const std::string& text);

// One token in the Completions shape, for a streamed chunk. Streaming emits one
// chunk per token: a chunk carrying two tokens has nowhere to put two offsets.
json completions_logprobs_json_one(const imp::TokenLogprobInfo& lp, size_t text_offset);
size_t utf8_complete_len(const std::string& s);

// Trim a trailing incomplete UTF-8 sequence from a finished string (#1310).
void drop_incomplete_utf8_tail(std::string& s);
void json_escape_into(std::string& out, const char* s, size_t len);

int b64_val(unsigned char c);
std::vector<uint8_t> base64_decode(const std::string& encoded);
// Standard base64 (with '=' padding) of a raw byte buffer. Used to serve the
// OpenAI `encoding_format: "base64"` embeddings response (the little-endian
// float32 array encoded as bytes — the default in the OpenAI Python SDK).
std::string base64_encode(const uint8_t* data, size_t len);

void strip_think_block(std::string& text);
std::pair<std::string, std::string> extract_reasoning(const std::string& text);

// strip_channel_headers: removes Gemma-4 "<|channel>NAME\n..." and "<channel|>\n..." structural
// headers (up to and including the newline), preserving body text. Handles both single-header
// and both-tags emission variants.
void strip_channel_headers(std::string& text);

// Channel-aware split: parses Gemma-4 "<|channel>NAME[<channel|>]BODY..." segments into
// reasoning ("thought" channel) and content ("final" channel plus any pre-channel text), each
// trimmed, markers stripped.
// HarmonyToolCall: one Harmony tool call (<|channel|>commentary to=functions.NAME
// ...<|message|>{args}<|call|>); only split_harmony_channels() fills it.
struct HarmonyToolCall {
    std::string name;       // the part after "functions."
    std::string arguments;  // the message body, verbatim
};

struct ChannelSegments {
    std::string reasoning;  // "thought" channel(s)
    std::string content;    // "final" channel(s) + un-channeled text
    std::string other;      // any unrecognised channel name (debug)
    // Harmony only: bodies addressed to a function recipient. These used to
    // land in `other`, which nothing reads, so a gpt-oss tool call was
    // silently dropped and the response carried an empty content (#1716).
    std::vector<HarmonyToolCall> tool_calls;
};
ChannelSegments split_channel_segments(const std::string& text);

// split_harmony_channels (gpt-oss): parses "<|channel|>NAME<|message|>BODY<|end|>" blocks (and
// the <|start|>role plumbing between them) into reasoning (analysis/commentary) vs content
// (final). All Harmony markup and role names stripped, each segment trimmed.
ChannelSegments split_harmony_channels(const std::string& text);

// Effective max output tokens for an OpenAI-shaped body: current OpenAI SDKs
// send "max_completion_tokens" (max_tokens is deprecated on chat/completions);
// it takes precedence over "max_tokens". `def` when neither is present.
int parse_max_tokens_field(const json& body, int def);

// Parse the OpenAI "stop" field (string or array of strings) into `out`,
// keeping at most `cap` entries. Returns true iff entries were dropped.
bool parse_stop_field(const json& body, size_t cap, std::vector<std::string>& out);

// `finish_detail` rides beside finish_reason on the choice as
// `imp_finish_detail` (nullptr = absent). finish_reason itself keeps the OpenAI
// enum: a client that switches on it must not have to learn a new member.
std::string sse_chunk(const std::string& id, int64_t created, const std::string& model, const json& delta,
                      const char* finish_reason, const json& logprobs = nullptr,
                      const char* finish_detail = nullptr);

std::string sse_completion_chunk(const std::string& id, int64_t created, const std::string& model,
                                 const std::string& text, const char* finish_reason,
                                 const json& logprobs = nullptr);

// Pre-formatted SSE chunk writer. Builds envelope templates once per request;
// hot-path write_content/write_reasoning only JSON-escape the token text and
// concatenate with the pre-built prefix/suffix — no json objects or .dump().
struct SSEChunkWriter {
    // content:            ...{"content":"<TEXT>"}...
    // reasoning_content:  ...{"reasoning_content":"<TEXT>"}...
    std::string content_prefix;
    std::string content_suffix;
    std::string reasoning_prefix;
    std::string reasoning_suffix;
    std::string buf_;

    SSEChunkWriter(const std::string& id, int64_t created, const std::string& model) {
        // JSON-escape id and model (they could theoretically contain quotes)
        std::string esc_id, esc_model;
        json_escape_into(esc_id, id.data(), id.size());
        json_escape_into(esc_model, model.data(), model.size());

        // system_fingerprint must be emitted by BOTH response builders or they drift
        // (ContentFrameMatchesJsonBuiltChunk caught exactly that when only sse_chunk() gained it, #1602).
        // Constant for the request, so it belongs in the pre-built SSE prefix, not the hot path.
        std::string esc_fp;
        const std::string fp = system_fingerprint(model);
        json_escape_into(esc_fp, fp.data(), fp.size());

        std::string envelope_prefix = "data: {\"id\":\"" + esc_id +
                                      "\",\"object\":\"chat.completion.chunk\",\"created\":" +
                                      std::to_string(created) + ",\"model\":\"" + esc_model +
                                      "\",\"system_fingerprint\":\"" + esc_fp +
                                      "\",\"choices\":[{\"index\":0,\"delta\":{\"";

        std::string envelope_suffix = "\"},\"finish_reason\":null}]}\n\n";

        content_prefix = envelope_prefix + "content\":\"";
        content_suffix = envelope_suffix;
        reasoning_prefix = envelope_prefix + "reasoning_content\":\"";
        reasoning_suffix = envelope_suffix;

        buf_.reserve(512);
    }

    bool write_content(const char* text, size_t len, httplib::DataSink& sink) {
        buf_.clear();
        buf_ += content_prefix;
        json_escape_into(buf_, text, len);
        buf_ += content_suffix;
        return sink.write(buf_.data(), buf_.size());
    }

    bool write_content(const std::string& text, httplib::DataSink& sink) {
        return write_content(text.data(), text.size(), sink);
    }

    bool write_reasoning(const char* text, size_t len, httplib::DataSink& sink) {
        buf_.clear();
        buf_ += reasoning_prefix;
        json_escape_into(buf_, text, len);
        buf_ += reasoning_suffix;
        return sink.write(buf_.data(), buf_.size());
    }

    bool write_reasoning(const std::string& text, httplib::DataSink& sink) {
        return write_reasoning(text.data(), text.size(), sink);
    }
};
