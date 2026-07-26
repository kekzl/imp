#pragma once

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <string>
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

// Send an OpenAI-style error envelope {"error":{"message":..,"type":..}} with
// the given HTTP status. Dumps via dump_safe so an invalid-UTF-8 byte echoed
// into the message (e.g. a parse-error what() on byte-truncated input) can
// never make the dump throw — that throw used to escape the handler and turn a
// 400-class bad-input case into a bare 500 (DEBUG-500-on-bad-input.md).
void send_json_error(httplib::Response& res, int status, const char* type, const std::string& message);

// Constant-time Bearer-token check. Returns true iff `authorization` equals
// "Bearer " + api_key, compared without early-out so response timing cannot leak
// the key prefix (std::string::operator== short-circuits on the first differing
// byte — a timing oracle). The comparison runs over the full expected length
// regardless of where (or whether) the input differs. Extracted from main.cpp's
// pre-routing handler so the security-critical compare is unit-testable.
bool bearer_token_matches(const std::string& authorization, const std::string& api_key);

// Accepts EITHER the OpenAI-style `Authorization: Bearer <key>` header OR the
// Anthropic-style `x-api-key: <key>` header (the official Anthropic SDK sends
// the latter, so a Bearer-only check 401s real Anthropic clients on /v1/messages).
// Both comparisons are constant-time. Pass the raw header values.
bool api_key_matches(const std::string& authorization, const std::string& x_api_key,
                     const std::string& api_key);

json safe_token_json(const std::string& text);
json token_bytes_json(const std::string& text);
size_t utf8_complete_len(const std::string& s);
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
                                 const std::string& text, const char* finish_reason);

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

        std::string envelope_prefix = "data: {\"id\":\"" + esc_id +
                                      "\",\"object\":\"chat.completion.chunk\",\"created\":" +
                                      std::to_string(created) + ",\"model\":\"" + esc_model +
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
