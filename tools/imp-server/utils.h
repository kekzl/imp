#pragma once

#include <httplib.h>
#include <nlohmann/json.hpp>

#include "runtime/request.h"

#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <cstddef>
#include <cstdint>

using json = nlohmann::json;

// Serialize `j` to a string that never throws on invalid UTF-8. nlohmann's
// default dump() throws json::type_error.316 the moment it hits an ill-formed
// UTF-8 byte; user- and model-derived strings (byte-truncated prompts, decoded
// tokens) routinely contain those. Invalid bytes are replaced with U+FFFD. For
// well-formed UTF-8 the output is byte-identical to dump(). Use this for ANY
// response/SSE/error body that can carry client- or model-supplied text.
std::string dump_safe(const json& j);

// Printable-ASCII, length-capped copy of a client-supplied string, for the
// cases where one is echoed back into a response (#1618).
std::string sanitize_for_echo(std::string_view in, size_t max_len);

// Rejoins UTF-8 characters that a tokenizer split across two tokens.
//
// BPE vocabularies routinely cut a multi-byte character in half: "größer"
// arrives as a piece ending in 0xC3 followed by a piece starting with 0xB6.
// Each streamed delta is serialized on its own, so dump_safe() sees half a
// character and replaces it with U+FFFD — the client receives "gr��ßer"
// while the same generation is correct over the non-streaming path, which
// decodes all tokens together. Any non-ASCII script hits this: German umlauts,
// accents, CJK, emoji.
//
// feed() returns the part that is safe to emit and holds back a trailing
// incomplete sequence until the next piece completes it. Bytes still held when
// the stream ends are dropped: generation stopped inside a character, so there
// is no character to show — emitting the fragment would reproduce the very
// artifact this class removes.
class Utf8Stitch {
public:
    std::string feed(const std::string& piece);

private:
    std::string carry_;
};

// Nesting depth of a JSON document, counted WITHOUT parsing it (#1607).
//
// The parsers on this surface are all recursive and none of them bounds depth,
// nlohmann included: measured on this tree, 50 000 nested arrays parse and
// dump() fine and 100 000 segfault, i.e. ~100 KB of body against a 100 MiB body
// cap. The parse is where the stack dies, so the check has to happen before it,
// on the raw bytes.
//
// Scans left to right, skipping string contents so a brace inside a string does
// not count, and stops as soon as `stop_at` is exceeded - so a hostile body
// costs only as many bytes as it takes to prove it hostile. Returns the depth
// reached, capped at `stop_at + 1`.
int json_nesting_depth(const std::string& body, int stop_at);

// Reject a request body that nests deeper than the cap, with the dialect's own
// error envelope (#1607). Returns true when the request was answered and the
// caller must stop.
//
// NOT in the pre-routing handler, where the other cross-cutting checks live:
// httplib calls that handler from Server::routing() BEFORE the body has been
// read, so `req.body` is empty there. Measured, after writing it there first -
// a 10 000-level body still returned 200.
bool reject_body_too_deep(const httplib::Request& req, httplib::Response& res);

// Length of a chunk starting at `off` that is at most `max` bytes AND ends on a
// UTF-8 codepoint boundary (#1554).
//
// Tool arguments were sliced every 48 bytes and each slice JSON-encoded on its
// own, so a multi-byte character straddling a boundary was cut in half and
// dump_safe turned each half into U+FFFD. The per-token content path has
// stitched for exactly this reason since #1310; the tool-argument path did not.
//
// Requires `off` to be on a boundary, which holds inductively when every chunk
// comes from this function. Always returns at least 1 when bytes remain, so a
// pathological input cannot stall the loop.
//
// `max` yields to a character: when no whole character fits, the single
// character is returned even if it is longer than `max`. A chunk size is a hint
// about frame size, and half a character is wrong at any size.
size_t utf8_chunk_len(const std::string& s, size_t off, size_t max);

// Send an OpenAI-style error envelope {"error":{"message":..,"type":..}} with
// the given HTTP status. Dumps via dump_safe so an invalid-UTF-8 byte echoed
// into the message (e.g. a parse-error what() on byte-truncated input) can
// never make the dump throw — that throw used to escape the handler and turn a
// 400-class bad-input case into a bare 500.
// The shared error envelope.
//
// `param` and `code` are optional and default to absent, which is what every
// existing caller gets. They are what makes an error machine-readable: OpenAI
// clients branch on `error.code`, and without it a context-window refusal, a
// bad argument and an auth failure differ only in an English sentence (#1595).
// Pass them wherever the answer is "this specific field, for this specific
// reason".
void send_json_error(httplib::Response& res, int status, const char* type, const std::string& message,
                     const char* param = nullptr, const char* code = nullptr);

// True for the endpoints that speak the Anthropic dialect, whose errors have a
// different envelope: `{"type":"error","error":{...}}` rather than
// `{"error":{...}}`. Four call sites in main.cpp used to spell this test out
// and two more forgot it, so a 429 on /v1/messages came back in the OpenAI
// shape and no Anthropic SDK could classify it (#1551).
bool is_anthropic_path(const std::string& path);

// The Anthropic error envelope, with `request_id` when one is known.
//
// `type` must be one of Anthropic's error types - invalid_request_error,
// authentication_error, billing_error, permission_error, not_found_error,
// request_too_large, rate_limit_error, api_error, overloaded_error,
// timeout_error. `server_error` and `capacity_error` are not among them and
// were being emitted at seven sites (#1556).
//
// request_id is what support and log correlation are asked for first; no error
// body carried one and no response carried a request-id header (#1561).
void send_anthropic_error(httplib::Response& res, int status, const char* type, const std::string& message,
                          const std::string& request_id = {});

// Translate an OpenAI-dialect `error.type` into the Anthropic one.
//
// The non-streaming /v1/messages path runs through the OpenAI handler and
// forwards whatever it produced, so `server_error` and `capacity_error` - which
// are not Anthropic error types - reached Anthropic SDK clients verbatim
// (#1556). Anything unrecognised falls back on the status: 5xx is api_error,
// everything else invalid_request_error.
const char* anthropic_error_type_for(std::string_view openai_type, int status);

// Send whichever envelope `path` calls for. `openai_type` and `anthropic_type`
// are the two dialects' names for the same condition.
void send_dialect_error(httplib::Response& res, const std::string& path, int status, const char* openai_type,
                        const char* anthropic_type, const std::string& message,
                        const std::string& request_id = {});

// Constant-time Bearer-token check. Returns true iff `authorization` equals
// "Bearer " + api_key, compared without early-out so response timing cannot leak
// the key prefix (std::string::operator== short-circuits on the first differing
// byte — a timing oracle). The comparison runs over the full expected length
// regardless of where (or whether) the input differs. Extracted from main.cpp's
// pre-routing handler so the security-critical compare is unit-testable.
bool bearer_token_matches(const std::string& authorization, const std::string& api_key);

// True when a reply came back with nothing to show and everything spent on
// thinking: no tool calls, empty content, non-empty reasoning.
//
// Not a defect. The answer shares the token budget with the thinking, so on a
// long conversation a small max_tokens can be consumed before the reply starts,
// and the caller sees `content: ""` with `finish_reason: stop`, which reads
// exactly like a broken engine. Measured on Qwen3.8-27B: a 74-turn session
// returns empty replies at max_tokens 260 and is clean at 600
// (docs/TROUBLESHOOTING.md).
//
// Split out because the state is real but not reliably reproducible on demand:
// it depends on how long the model chooses to think. A rule that fires rarely
// is exactly the one that has to be covered by a test rather than by a run.
bool answer_lost_to_reasoning(bool has_tool_calls, const std::string& content, const std::string& reasoning);

// Why this server cannot serve, or "" when it can. Not the same question as
// whether the last request failed.
//
// A transient OOM keeps /health at 200 on purpose: the server is alive, the
// pressure passes, and an orchestrator restarting on it makes things worse. A
// KV pool that fell back to its rescue floor is the opposite. The pool is sized
// once at init, so the condition lasts as long as the process; every prompt
// past a few hundred tokens is cancelled at admission with a message naming the
// prompt; and /v1/models goes on advertising the full context. Restarting on a
// card that has since been freed is the only fix, which is exactly what a 503
// asks an orchestrator to do.
//
// Reported from production by a peer running imp behind an agent loop:
// `docker compose restart` while the previous process still held the card came
// up with 16 KV blocks against a planned 3066, /health saying ok throughout. It
// cost two failures that looked like defects in another component.
//
// The string is the operator-facing detail; the machine-readable half is
// health_unservable_code() below, because a client has to tell this apart from
// a transient 503 to know not to retry it.
std::string health_unservable_reason(bool engine_faulted, bool kv_pool_floored, int kv_blocks,
                                     int kv_block_size);

// The stable identifier for the same state, "" when the server can serve.
// Values: "engine_faulted", "kv_pool_floored".
const char* health_unservable_code(bool engine_faulted, bool kv_pool_floored);

// Accepts EITHER the OpenAI-style `Authorization: Bearer <key>` header OR the
// Anthropic-style `x-api-key: <key>` header (the official Anthropic SDK sends
// the latter, so a Bearer-only check 401s real Anthropic clients on /v1/messages).
// Both comparisons are constant-time. Pass the raw header values.
bool api_key_matches(const std::string& authorization, const std::string& x_api_key,
                     const std::string& api_key);

// Map an engine finish reason onto the OpenAI `finish_reason` enum.
//
// The engine has two reasons OpenAI does not: "cancelled" (the request was
// aborted) and "capacity" (the KV pool cannot hold it). Both used to ship
// verbatim on a 200, so a client switching on the enum fell through its
// default branch and treated a failed generation as a normal one (#1590).
//
// Both map to "length": the generation stopped before the model chose to stop,
// which is exactly what "length" means to a client, and it is the value that
// makes them retry or shorten rather than accept the text. The non-streaming
// chat path answers "capacity" with 503 before it gets here; this is the
// backstop for the paths that do not.
const char* openai_finish_reason(const char* engine_finish);

// `system_fingerprint`: what a client compares across calls to notice that the
// backend changed under it. Emitted nowhere before #1602, so a model swap, a
// quantisation change or a server upgrade was invisible in the response.
//
// The value is the engine version plus the loaded model, hashed: the two things
// that change what the same request returns. Stable for the life of a
// configuration, different across any change to either.
std::string system_fingerprint(const std::string& model_name);

json safe_token_json(const std::string& text);
json token_bytes_json(const std::string& text);

// The two logprobs SHAPES, which are not the same object.
//
// Chat (`/v1/chat/completions`):
//   {"content": [{"token","logprob","bytes","top_logprobs":[{...}]}]}
// Completions (`/v1/completions`), a different shape entirely:
//   {"tokens":[], "token_logprobs":[], "top_logprobs":[{tok: lp}], "text_offset":[]}
//
// /v1/completions returned the Chat object on a `text_completion` response
// until #1589, so an OpenAI SDK reading `.logprobs.tokens` found nothing and
// one reading `.logprobs.content` got a field its own type does not declare.
//
// `text` is the completion string the offsets index into; the offsets are byte
// offsets from its start, which is what the OpenAI field means for ASCII and
// the only defensible reading for anything else.
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

// Strip Gemma-4 "<|channel>NAME\n..." and "<channel|>\n..." structural headers
// from a content string. Only the header (up to and including the newline) is
// removed; the body text is preserved. Model variants that never emit
// <channel|> produce a single leading header that this function drops; ones
// that emit both get both stripped, leaving only the body text concatenated.
void strip_channel_headers(std::string& text);

// Channel-aware split: parses Gemma-4 style `<|channel>NAME[<channel|>]BODY...`
// segments and returns the reasoning (= "thought" channel) separately from the
// user-facing content (= "final" channel + any pre-channel text). Bodies are
// preserved verbatim minus the markers/header names. Each segment is trimmed.
struct ChannelSegments {
    std::string reasoning;  // "thought" channel(s)
    std::string content;    // "final" channel(s) + un-channeled text
    std::string other;      // any unrecognised channel name (debug)
};
ChannelSegments split_channel_segments(const std::string& text);

// Harmony-aware split (gpt-oss): parses `<|channel|>NAME<|message|>BODY<|end|>`
// blocks (and the `<|start|>role` plumbing between them) into reasoning
// (analysis / commentary channels) vs content (final channel). All Harmony
// control markup and role names are stripped. Each segment is trimmed.
ChannelSegments split_harmony_channels(const std::string& text);

// Effective max output tokens for an OpenAI-shaped body: current OpenAI SDKs
// send "max_completion_tokens" (max_tokens is deprecated on chat/completions);
// it takes precedence over "max_tokens". `def` when neither is present.
int parse_max_tokens_field(const json& body, int def);

// Parse the OpenAI "stop" field (string or array of strings) into `out`,
// keeping at most `cap` entries. Returns true iff entries were dropped.
bool parse_stop_field(const json& body, size_t cap, std::vector<std::string>& out);

std::string sse_chunk(const std::string& id, int64_t created, const std::string& model, const json& delta,
                      const char* finish_reason, const json& logprobs = nullptr);

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

        // system_fingerprint is part of the envelope, so it has to be in BOTH
        // builders or they drift; ContentFrameMatchesJsonBuiltChunk is the
        // guard that caught exactly that when only sse_chunk() gained it
        // (#1602). It is constant for the request, so it belongs in the
        // pre-built prefix rather than the hot path.
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
