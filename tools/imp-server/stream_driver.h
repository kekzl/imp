#pragma once

// Shared per-token SSE loop for all three streaming dialects (chat/messages/responses):
// disconnect/timeout/keepalive, pop_token, structural-stop filtering, Harmony/Gemma-4 channel
// filters, reasoning demux, tool-call demux, stop holdback, UTF-8 buffering, end-of-stream flush.
// Used to be hand-copied per dialect (~600 LOC each) and drifted repeatedly (#941: /v1/responses
// had no metrics or keepalive). Now lives once; each dialect supplies its wire format via
// StreamDialect callbacks.

#include "handlers.h"
#include "handlers_internal.h"
#include "tool_call.h"

#include <functional>
#include <memory>
#include <string>
#include <vector>

// Dialect emission callbacks. Every emitter returns false when the client
// write failed; the driver then aborts the stream and returns false (no
// terminal events, no accounting — matching the previous per-dialect code).
struct StreamDialect {
    // User-visible content / reasoning deltas.
    std::function<bool(const std::string&)> emit_text;
    std::function<bool(const std::string&)> emit_reasoning;
    // emit_content_token carries the token index explicitly rather than reading a live counter:
    // without stop sequences the counter is current, but held-back bytes (stop matching) ship after
    // the counter has moved on - the stop path used to bypass this sink and ship no logprobs at all
    // (#1588). -1 means the driver cannot attribute the bytes to one token.
    std::function<bool(const std::string&, int token_index)> emit_content_token;
    // Idle keepalive, sent when no token arrived for ~10s. A false return is
    // treated as a client disconnect (request cancelled).
    std::function<bool()> keepalive;
    // Tool calls: a streamed (JSON) call is on_call_begin -> on_call_args_delta* -> on_call_end; a
    // buffered (non-JSON) call is one on_call_buffered. The driver appends to
    // StreamLoopResult::tool_calls BEFORE invoking the begin/buffered callback, so it gets a
    // reference to the recorded element (index = size()-1) and may mutate it (validation).
    std::function<bool(const ParsedToolCall&)> on_call_begin;
    std::function<bool(const std::string&)> on_call_args_delta;
    std::function<bool(ParsedToolCall*)> on_call_end;
    std::function<bool(ParsedToolCall&)> on_call_buffered;
    // Harmony (gpt-oss) reasoning-channel gate: the chat dialect drops
    // analysis/commentary text when reasoning_format == "none"; the native
    // thinking dialects always emit it.
    bool harmony_reasoning_on = true;
};

// Loop outcome, consumed by the dialect's terminal-event section and by
// finish_stream_accounting_. n_output_tokens and tool_calls are updated live
// (see StreamDialect::emit_content_token / on_call_begin).
struct StreamLoopResult {
    const char* finish = nullptr;
    // stop_sequence: the matched text, reported by the Anthropic wire format
    // (stop_reason:"stop_sequence", stop_sequence:"<text>") - with only finish=="stop" a match was
    // indistinguishable from the model ending its turn (#1550).
    std::string stop_sequence;
    // error_type: set when the stream ended on a server fault, not the model finishing. Anthropic
    // turns it into an `error` SSE event; without it a timeout read as stop_reason "max_tokens" and
    // a refusal as "capacity" (#1552, #1553). OpenAI ignores both deliberately (#1590, no enum member).
    const char* error_type = nullptr;
    std::string error_message;
    int n_output_tokens = 0;
    int n_reasoning_tokens = 0;
    double ttft_ms = 0.0;
    bool tool_calls_emitted = false;
    // content_emitted: the streaming equivalent of the non-streaming content.empty() test.
    // reasoning_truncated is not a substitute - it only covers finish=="length", missing a reasoning
    // model that hit EOS mid-thought (finish "stop" via the in-think suppression + 16-token grace).
    bool content_emitted = false;
    // Generation hit max_tokens while still inside reasoning and produced no
    // content (the chat dialect emits its "[Reasoning truncated ...]" notice).
    bool reasoning_truncated = false;
    std::vector<ParsedToolCall> tool_calls;
};

// Drive the token loop until a finish reason is recorded. Returns false when a
// client write failed mid-stream (adapter returns false to httplib without
// terminal events); true otherwise, with out.finish always set.
bool run_stream_loop_(httplib::DataSink& sink, ChatRequestContext& ctx, ServerState& state,
                      const std::shared_ptr<ServerRequest>& server_req, StreamDialect& d,
                      StreamLoopResult& out);

// Shared post-stream accounting: server metrics (request/token counters, TTFT,
// inter-token latency), the JSONL request log, and the stderr summary line.
// label prefixes the stderr line ("" for chat, "messages stream: ", ...).
void finish_stream_accounting_(ServerState& state, ChatRequestContext& ctx,
                               const std::shared_ptr<imp::Request>& active_req, const StreamLoopResult& out,
                               const std::string& req_id, const char* label);
