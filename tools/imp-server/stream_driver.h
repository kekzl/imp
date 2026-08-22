#pragma once

// Shared per-token SSE streaming loop for the three streaming dialects
// (/v1/chat/completions, /v1/messages, /v1/responses). The outer token loop —
// disconnect/timeout/keepalive handling, batching-engine pop_token, structural
// stop-token filtering, the Harmony and Gemma-4 channel filters, the
// reasoning/content demux, the streaming tool-call demux, stop-sequence
// holdback and UTF-8 buffering, and end-of-stream flushing — used to be
// hand-copied per dialect (~600 LOC each) and drifted repeatedly (#941 was the
// 3rd/4th drift bug: /v1/responses had no metrics and no keepalive). The loop
// now lives once in stream_driver.cpp; each dialect supplies only its wire
// format via StreamDialect callbacks and emits its own terminal events after
// the loop returns.

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
    // Content carrying a token index, so the chat dialect can attach the
    // right per-token logprob. The other dialects alias this to emit_text and
    // ignore the index.
    //
    // The index is passed rather than read off StreamLoopResult::n_output_tokens
    // because the two emission paths disagree about "now": without stop
    // sequences the driver emits as it decodes, so the live counter is right,
    // but with them it holds bytes back until a stop match is ruled out, and by
    // the time those bytes go out the counter has moved on. That is why the
    // stop path used to bypass this sink entirely and ship no logprobs at all
    // (#1588). -1 means the driver cannot attribute the bytes to one token.
    std::function<bool(const std::string&, int token_index)> emit_content_token;
    // Idle keepalive, sent when no token arrived for ~10s. A false return is
    // treated as a client disconnect (request cancelled).
    std::function<bool()> keepalive;
    // Tool calls. A streamed (JSON-layout) call arrives as on_call_begin ->
    // on_call_args_delta* -> on_call_end; a buffered (non-JSON layout) call as
    // a single on_call_buffered. The driver assigns tc.id and appends the call
    // to StreamLoopResult::tool_calls BEFORE invoking on_call_begin /
    // on_call_buffered — the callback receives a reference to the recorded
    // element (its index is tool_calls.size() - 1) and may mutate it
    // (validation). on_call_end receives the completed call (arguments
    // recorded), or nullptr when no call was recorded.
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
    int n_output_tokens = 0;
    int n_reasoning_tokens = 0;
    double ttft_ms = 0.0;
    bool tool_calls_emitted = false;
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
