#include "utils.h"
#include "imp/imp.h"
#include "stream_pipeline.h"
#include "core/logging.h"
#include "runtime/think_stop_logic.h"

#include <algorithm>
#include <cstring>
#include <cstdio>

// sanitize_for_echo: makes a client-supplied string safe to echo back. Two problems, one helper:
// ill-formed UTF-8 throws in dump(), and control bytes/unbounded length reach log viewers or
// terminals. Printable ASCII only; everything else becomes '.', truncated with a marker (#1618).
std::string sanitize_for_echo(std::string_view in, size_t max_len) {
    std::string out;
    const size_t n = std::min(in.size(), max_len);
    out.reserve(n + 3);
    for (size_t i = 0; i < n; i++) {
        const unsigned char c = static_cast<unsigned char>(in[i]);
        out.push_back((c >= 0x20 && c < 0x7f) ? static_cast<char>(c) : '.');
    }
    if (in.size() > max_len)
        out += "...";
    return out;
}

std::string dump_safe(const json& j) {
    // error_handler_t::replace: emit U+FFFD for ill-formed UTF-8 instead of
    // throwing json::type_error.316. Identical output for valid UTF-8.
    return j.dump(-1, ' ', false, json::error_handler_t::replace);
}

// drop_incomplete_utf8_tail (#1310): drops an incomplete trailing UTF-8 sequence from a FINISHED
// string. Utf8Stitch defers such a tail for the next streaming piece, but generation end has no
// next piece - dump_safe would otherwise substitute U+FFFD for a character no token produced.
// Same 3-byte bound as Utf8Stitch::feed; a longer tail is genuinely ill-formed and left for dump_safe.
void drop_incomplete_utf8_tail(std::string& s) {
    const size_t complete = imp::stream::utf8_complete_len(s);
    if (complete < s.size() && s.size() - complete <= 3)
        s.resize(complete);
}

bool reject_body_too_deep(const httplib::Request& req, httplib::Response& res) {
    // kMaxBodyNesting=100: every downstream parser (nlohmann included) is recursive and unbounded -
    // measured 50000 nested arrays parse fine, 100000 segfault (~100KB against the 100MiB body cap,
    // unauthenticated, and one process crash takes every in-flight stream with it).
    constexpr int kMaxBodyNesting = 100;
    if (req.body.empty() || json_nesting_depth(req.body, kMaxBodyNesting) <= kMaxBodyNesting)
        return false;

    send_dialect_error(res, req.path, 400, "invalid_request_error", "invalid_request_error",
                       "request body nests deeper than 100 levels");
    return true;
}

int json_nesting_depth(const std::string& body, int stop_at) {
    int depth = 0, max_depth = 0;
    bool in_string = false, escaped = false;
    for (char c : body) {
        if (in_string) {
            if (escaped) {
                escaped = false;
            } else if (c == '\\') {
                escaped = true;
            } else if (c == '"') {
                in_string = false;
            }
            continue;
        }
        switch (c) {
            case '"':
                in_string = true;
                break;
            case '{':
            case '[':
                depth++;
                if (depth > max_depth) {
                    max_depth = depth;
                    if (max_depth > stop_at)
                        return max_depth;  // proven hostile, stop reading
                }
                break;
            case '}':
            case ']':
                if (depth > 0)
                    depth--;
                break;
            default:
                break;
        }
    }
    return max_depth;
}

size_t utf8_chunk_len(const std::string& s, size_t off, size_t max) {
    if (off >= s.size())
        return 0;
    const size_t remaining = s.size() - off;
    if (remaining <= max)
        return remaining;  // the tail fits; whatever it is, it is not a split
    const size_t complete = imp::stream::utf8_complete_len(s.substr(off, max));
    if (complete > 0)
        return complete;
    // complete==0 (no whole character fits in `max`): yields one whole character anyway rather than
    // splitting it, since a chunk-size hint is wrong at any size if it cuts a character. Falls back
    // to one raw byte only when the input is ill-formed at `off`.
    const unsigned char lead = static_cast<unsigned char>(s[off]);
    size_t char_len = 1;
    if ((lead & 0xE0) == 0xC0)
        char_len = 2;
    else if ((lead & 0xF0) == 0xE0)
        char_len = 3;
    else if ((lead & 0xF8) == 0xF0)
        char_len = 4;
    return std::min(char_len, remaining);
}

std::string Utf8Stitch::feed(const std::string& piece) {
    std::string buf = carry_ + piece;
    carry_.clear();

    const size_t complete = imp::stream::utf8_complete_len(buf);
    // A split character is at most 3 bytes short; a longer tail is invalid input, not a split
    // character, and holding it back would stall the stream forever - pass it through for dump_safe
    // to replace instead.
    if (complete < buf.size() && buf.size() - complete <= 3) {
        carry_.assign(buf, complete, buf.size() - complete);
        buf.resize(complete);
    }
    return buf;
}

void send_json_error(httplib::Response& res, int status, const char* type, const std::string& message,
                     const char* param, const char* code) {
    json e = {{"message", message}, {"type", type}};
    // Emitted only when supplied: OpenAI's own envelope carries them as null
    // rather than absent, but a client that checks `"code" in err` should not
    // see a key that says nothing.
    if (param)
        e["param"] = param;
    if (code)
        e["code"] = code;
    json err = {{"error", std::move(e)}};
    res.status = status;
    res.set_content(dump_safe(err), "application/json");
}

bool prompt_within_input_budget(httplib::Response& res, size_t bytes, int max_input_tokens,
                                const char* param) {
    if (!prompt_bytes_exceed_input_budget(bytes, max_input_tokens))
        return true;
    send_json_error(res, 400, "invalid_request_error",
                    std::string(param) + " carries " + std::to_string(bytes) + " bytes, above what " +
                        std::to_string(max_input_tokens) + " tokens can hold (--max-input-tokens)",
                    param, "context_length_exceeded");
    return false;
}

int servable_context_tokens(int planned_max_seq_len, long long kv_capacity_tokens) {
    if (kv_capacity_tokens <= 0 || kv_capacity_tokens >= planned_max_seq_len)
        return planned_max_seq_len;
    return static_cast<int>(kv_capacity_tokens);
}

long long kv_capacity_ceiling_tokens(int total_blocks, int ceiling_blocks, int block_size) {
    return static_cast<long long>(std::max(total_blocks, ceiling_blocks)) * block_size;
}

bool is_anthropic_path(const std::string& path) { return path.rfind("/v1/messages", 0) == 0; }

void send_anthropic_error(httplib::Response& res, int status, const char* type, const std::string& message,
                          const std::string& request_id) {
    json e = {{"type", type}, {"message", message}};
    json err = {{"type", "error"}, {"error", std::move(e)}};
    if (!request_id.empty()) {
        err["request_id"] = request_id;
        res.set_header("request-id", request_id);
    }
    res.status = status;
    res.set_content(dump_safe(err), "application/json");
}

const char* anthropic_error_type_for(std::string_view openai_type, int status) {
    // Anthropic's complete set. A type already in it passes through; the two
    // this server invented do not (#1556).
    static constexpr const char* kAnthropicTypes[] = {"invalid_request_error", "authentication_error",
                                                      "billing_error",         "permission_error",
                                                      "not_found_error",       "request_too_large",
                                                      "rate_limit_error",      "api_error",
                                                      "overloaded_error",      "timeout_error"};
    for (const char* t : kAnthropicTypes)
        if (openai_type == t)
            return t;
    if (openai_type == "capacity_error")
        return "overloaded_error";
    if (openai_type == "server_error")
        return "api_error";
    return status >= 500 ? "api_error" : "invalid_request_error";
}

void send_dialect_error(httplib::Response& res, const std::string& path, int status, const char* openai_type,
                        const char* anthropic_type, const std::string& message,
                        const std::string& request_id) {
    if (is_anthropic_path(path)) {
        send_anthropic_error(res, status, anthropic_type, message, request_id);
        return;
    }
    if (!request_id.empty())
        res.set_header("request-id", request_id);
    send_json_error(res, status, openai_type, message);
}

bool answer_lost_to_reasoning(bool has_tool_calls, const std::string& content, const std::string& reasoning) {
    return answer_lost_to_reasoning_flags(has_tool_calls, content.empty(), !reasoning.empty());
}

bool report_answer_lost_to_reasoning(bool has_tool_calls, const std::string& content,
                                     const std::string& reasoning, const char* finish) {
    // An empty answer beside a full reasoning channel is not a defect - the reply shares the token
    // budget with thinking. Measured on Qwen3.8-27B: a 74-turn session returned empty replies at
    // max_tokens 260, clean 74/74 at 600 (docs/TROUBLESHOOTING.md). Report which case it was rather
    // than leave the caller bisecting a working engine.
    if (!answer_lost_to_reasoning(has_tool_calls, content, reasoning))
        return false;
    IMP_LOG_WARN(
        "empty content: the answer never started because the token budget went to "
        "reasoning (%zu chars of thinking, finish_reason=%s). Raise max_tokens: a "
        "thinking model needs room to answer AFTER it thinks.",
        reasoning.size(), finish ? finish : "");
    return true;
}

int nonstream_reasoning_tokens(const std::vector<int32_t>& output_ids, int32_t think_start_id,
                               int32_t think_end_id, bool started_in_think, size_t reasoning_chars,
                               const std::function<size_t(int32_t)>& decoded_len) {
    if (reasoning_chars == 0)
        return 0;
    if (think_end_id >= 0) {
        bool still_thinking = false;
        const int n = imp::think_logic::count_reasoning_tokens(output_ids, think_start_id, think_end_id,
                                                               started_in_think, still_thinking);
        if (n > 0)
            return n;
    }
    if (!decoded_len)
        return 0;
    int n = 0;
    size_t chars = 0;
    for (int32_t id : output_ids) {
        if (chars >= reasoning_chars)
            break;
        chars += decoded_len(id);
        n++;
    }
    return n;
}

const char* health_unservable_code(bool engine_faulted, bool kv_pool_floored) {
    // Faulted first: a wedged engine is the louder fault, and a floored pool
    // does not stop being true underneath it.
    if (engine_faulted)
        return "engine_faulted";
    if (kv_pool_floored)
        return "kv_pool_floored";
    return "";
}

std::string health_unservable_reason(bool engine_faulted, bool kv_pool_floored, int kv_blocks,
                                     int kv_block_size) {
    if (engine_faulted)
        return "the engine is faulted and this process cannot recover; restart it";
    if (!kv_pool_floored)
        return "";
    // Name the pool and the arithmetic. The admission error a caller sees says
    // "KV cache too small for prompt", which reads as a statement about the
    // prompt: at this size no prompt worth sending fits.
    const long long tokens = static_cast<long long>(kv_blocks) * kv_block_size;
    return "the KV pool fell back to its rescue floor: " + std::to_string(kv_blocks) + " blocks of " +
           std::to_string(kv_block_size) + " = " + std::to_string(tokens) +
           " tokens of capacity, so every longer prompt is cancelled at admission. Nothing was "
           "left to size the pool from at startup, usually another process still holding the "
           "card. This lasts as long as the process: restart it on a free card. Retrying will "
           "not help.";
}

bool bearer_token_matches(const std::string& authorization, const std::string& api_key) {
    const std::string expected = "Bearer " + api_key;
    // Accumulate the difference over the full expected length; do NOT early-out
    // on the first mismatch (that leaks the key prefix via response timing).
    unsigned char diff = static_cast<unsigned char>(authorization.size() != expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        unsigned char ac = (i < authorization.size()) ? static_cast<unsigned char>(authorization[i]) : 0;
        diff |= ac ^ static_cast<unsigned char>(expected[i]);
    }
    return diff == 0;
}

// Constant-time equality over max(|a|,|b|) bytes — no early-out on the first
// differing byte (that would leak the key prefix via timing).
static bool constant_time_equals(const std::string& a, const std::string& b) {
    const size_t n = a.size() > b.size() ? a.size() : b.size();
    unsigned char diff = static_cast<unsigned char>(a.size() != b.size());
    for (size_t i = 0; i < n; ++i) {
        unsigned char ac = (i < a.size()) ? static_cast<unsigned char>(a[i]) : 0;
        unsigned char bc = (i < b.size()) ? static_cast<unsigned char>(b[i]) : 0;
        diff |= ac ^ bc;
    }
    return diff == 0;
}

bool api_key_matches(const std::string& authorization, const std::string& x_api_key,
                     const std::string& api_key) {
    if (bearer_token_matches(authorization, api_key))
        return true;
    // x-api-key carries the raw key (no "Bearer " prefix). Only consider it when
    // present so an empty header can't match an empty configured key by accident.
    if (!x_api_key.empty() && constant_time_equals(x_api_key, api_key))
        return true;
    return false;
}

std::string system_fingerprint(const std::string& model_name) {
    // FNV-1a over version + model. Not a security hash; it exists so a client
    // can tell "same backend" from "different backend" in one comparison.
    const std::string material = std::string(imp_version()) + "\x1f" + model_name;
    uint64_t h = 1469598103934665603ULL;
    for (unsigned char c : material) {
        h ^= c;
        h *= 1099511628211ULL;
    }
    char buf[24];
    std::snprintf(buf, sizeof(buf), "fp_%016llx", static_cast<unsigned long long>(h));
    return buf;
}

const char* openai_finish_reason(const char* engine_finish) {
    if (engine_finish == nullptr)
        return "stop";
    if (std::strcmp(engine_finish, "cancelled") == 0 || std::strcmp(engine_finish, "capacity") == 0)
        return "length";
    return engine_finish;
}

json safe_token_json(const std::string& text) {
    std::string safe;
    safe.reserve(text.size());
    size_t i = 0;
    while (i < text.size()) {
        unsigned char c = static_cast<unsigned char>(text[i]);
        int expected = 0;
        if (c < 0x80) {
            expected = 1;
        } else if ((c & 0xE0) == 0xC0) {
            expected = 2;
        } else if ((c & 0xF0) == 0xE0) {
            expected = 3;
        } else if ((c & 0xF8) == 0xF0) {
            expected = 4;
        } else {
            safe += "\xEF\xBF\xBD";
            i++;
            continue;
        }  // invalid lead -> U+FFFD
        if (i + expected > text.size()) {
            // Incomplete sequence at end -> U+FFFD for each remaining byte
            for (; i < text.size(); i++)
                safe += "\xEF\xBF\xBD";
            break;
        }
        // Validate continuation bytes
        bool valid = true;
        for (int j = 1; j < expected; j++) {
            if ((static_cast<unsigned char>(text[i + j]) & 0xC0) != 0x80) {
                valid = false;
                break;
            }
        }
        if (valid) {
            safe.append(text, i, expected);
            i += expected;
        } else {
            safe += "\xEF\xBF\xBD";
            i++;
        }
    }
    return json(safe);
}

json token_bytes_json(const std::string& text) {
    json arr = json::array();
    for (unsigned char c : text)
        arr.push_back(static_cast<int>(c));
    return arr;
}

json completions_logprobs_json_one(const imp::TokenLogprobInfo& lp, size_t text_offset) {
    json top_obj = json::object();
    for (const auto& t : lp.top) {
        const json key = safe_token_json(t.text);
        if (key.is_string())
            top_obj[key.get<std::string>()] = t.logprob;
    }
    return json{{"tokens", json::array({safe_token_json(lp.text)})},
                {"token_logprobs", json::array({lp.logprob})},
                {"top_logprobs", json::array({top_obj})},
                {"text_offset", json::array({static_cast<int>(text_offset)})}};
}

json chat_logprobs_json(const std::vector<imp::TokenLogprobInfo>& lps, size_t limit) {
    json content = json::array();
    for (size_t i = 0; i < lps.size() && i < limit; i++) {
        const auto& lp = lps[i];
        json top_arr = json::array();
        for (const auto& t : lp.top) {
            top_arr.push_back({{"token", safe_token_json(t.text)},
                               {"logprob", t.logprob},
                               {"bytes", token_bytes_json(t.text)}});
        }
        content.push_back({{"token", safe_token_json(lp.text)},
                           {"logprob", lp.logprob},
                           {"bytes", token_bytes_json(lp.text)},
                           {"top_logprobs", top_arr}});
    }
    return json{{"content", content}};
}

json completions_logprobs_json(const std::vector<imp::TokenLogprobInfo>& lps, size_t limit,
                               const std::string& text) {
    json tokens = json::array();
    json token_logprobs = json::array();
    json top_logprobs = json::array();
    json text_offset = json::array();

    // Token offsets are tracked by advancing through the assembled `text`, not by summing decoded
    // token lengths - a stop-sequence trim or a detokenizer's dropped leading space can make the two
    // disagree; when they do, offsets stop advancing rather than running past the string's end.
    size_t cursor = 0;
    for (size_t i = 0; i < lps.size() && i < limit; i++) {
        const auto& lp = lps[i];
        tokens.push_back(safe_token_json(lp.text));
        token_logprobs.push_back(lp.logprob);
        text_offset.push_back(static_cast<int>(cursor));

        // top_logprobs here is an OBJECT per position (token -> logprob), not
        // the array of objects the Chat shape uses.
        json top_obj = json::object();
        for (const auto& t : lp.top) {
            const json key = safe_token_json(t.text);
            if (key.is_string())
                top_obj[key.get<std::string>()] = t.logprob;
        }
        top_logprobs.push_back(top_obj);

        if (!lp.text.empty() && cursor + lp.text.size() <= text.size() &&
            text.compare(cursor, lp.text.size(), lp.text) == 0) {
            cursor += lp.text.size();
        }
    }

    return json{{"tokens", tokens},
                {"token_logprobs", token_logprobs},
                {"top_logprobs", top_logprobs},
                {"text_offset", text_offset}};
}

size_t utf8_complete_len(const std::string& s) {
    // Single source of truth lives in stream_pipeline.h (pure, unit-tested).
    return imp::stream::utf8_complete_len(s);
}

void json_escape_into(std::string& out, const char* s, size_t len) {
    out.reserve(out.size() + len + 8);
    for (size_t i = 0; i < len; i++) {
        char c = s[i];
        switch (c) {
            case '"':
                out += "\\\"";
                break;
            case '\\':
                out += "\\\\";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\r':
                out += "\\r";
                break;
            case '\t':
                out += "\\t";
                break;
            default:
                if ((unsigned char)c < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", (unsigned char)c);
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
}

int b64_val(unsigned char c) {
    if (c >= 'A' && c <= 'Z')
        return c - 'A';
    if (c >= 'a' && c <= 'z')
        return c - 'a' + 26;
    if (c >= '0' && c <= '9')
        return c - '0' + 52;
    if (c == '+')
        return 62;
    if (c == '/')
        return 63;
    return -1;
}

std::string base64_encode(const uint8_t* data, size_t len) {
    static const char tbl[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve((len + 2) / 3 * 4);
    size_t i = 0;
    for (; i + 3 <= len; i += 3) {
        uint32_t n = (static_cast<uint32_t>(data[i]) << 16) |
                     (static_cast<uint32_t>(data[i + 1]) << 8) | static_cast<uint32_t>(data[i + 2]);
        out.push_back(tbl[(n >> 18) & 0x3F]);
        out.push_back(tbl[(n >> 12) & 0x3F]);
        out.push_back(tbl[(n >> 6) & 0x3F]);
        out.push_back(tbl[n & 0x3F]);
    }
    if (i < len) {
        uint32_t n = static_cast<uint32_t>(data[i]) << 16;
        bool have_two = (i + 1 < len);
        if (have_two)
            n |= static_cast<uint32_t>(data[i + 1]) << 8;
        out.push_back(tbl[(n >> 18) & 0x3F]);
        out.push_back(tbl[(n >> 12) & 0x3F]);
        out.push_back(have_two ? tbl[(n >> 6) & 0x3F] : '=');
        out.push_back('=');
    }
    return out;
}

std::vector<uint8_t> base64_decode(const std::string& encoded) {
    std::vector<uint8_t> out;
    out.reserve(encoded.size() * 3 / 4);
    uint32_t accum = 0;
    int bits = 0;
    for (unsigned char c : encoded) {
        int val = b64_val(c);
        if (val < 0)
            continue;
        accum = (accum << 6) | static_cast<uint32_t>(val);
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back(static_cast<uint8_t>((accum >> bits) & 0xFF));
        }
    }
    return out;
}

// split_last_think: splits `text` at the LAST "</think>" (reasoning = text before, leading
// "<think>" stripped and trimmed; content = text after, trimmed). Returns false, out-params
// untouched, when there is no "</think>". Shared by both non-streaming demuxers so the split
// point cannot drift between them.
static bool split_last_think(const std::string& text, std::string& reasoning, std::string& content) {
    auto last_end = text.rfind("</think>");
    if (last_end == std::string::npos)
        return false;

    reasoning = text.substr(0, last_end);
    auto think_start = reasoning.find("<think>");
    if (think_start != std::string::npos)
        reasoning = reasoning.substr(think_start + 7);
    // Markers left inside (a budget-forced </think> the model closed again
    // after, a re-opened block) are structure, not prose: drop them, as the
    // streaming splitter does.
    for (const char* marker : {"</think>", "<think>"}) {
        const size_t len = std::char_traits<char>::length(marker);
        for (size_t p; (p = reasoning.find(marker)) != std::string::npos;)
            reasoning.erase(p, len);
    }
    auto rs = reasoning.find_first_not_of("\n\r\t ");
    auto re = reasoning.find_last_not_of("\n\r\t ");
    reasoning =
        (rs != std::string::npos && re != std::string::npos) ? reasoning.substr(rs, re - rs + 1) : std::string();

    content = text.substr(last_end + 8);
    auto cs = content.find_first_not_of("\n\r\t ");
    content = (cs != std::string::npos) ? content.substr(cs) : std::string();
    return true;
}

void strip_think_block(std::string& text) {
    std::string reasoning, content;
    if (split_last_think(text, reasoning, content)) {
        // Content after the last </think>. If it re-opens an (unclosed) <think>,
        // the model never finished that block — discard it.
        text = (content.compare(0, 7, "<think>") == 0) ? std::string() : content;
        return;
    }

    // No </think> — an unclosed leading <think> means thinking never finished.
    auto first = text.find_first_not_of("\n\r\t ");
    if (first != std::string::npos && text.compare(first, 7, "<think>") == 0)
        text.clear();
}

ChannelSegments split_channel_segments(const std::string& text) {
    static const char kOpen[] = "<|channel>";
    static const char kClose[] = "<channel|>";
    constexpr size_t kOpenLen = sizeof(kOpen) - 1;
    constexpr size_t kCloseLen = sizeof(kClose) - 1;

    ChannelSegments out;
    // Empty channel name = "before any header" — Gemma-4's chat template often
    // routes pre-thought turn boilerplate through this state. Treat it as
    // user-facing content (matches the legacy strip_channel_headers behaviour).
    std::string current_channel;

    auto append_to_channel = [&](char c) {
        if (current_channel == "thought" || current_channel == "analysis") {
            out.reasoning.push_back(c);
        } else if (current_channel == "final" || current_channel.empty()) {
            out.content.push_back(c);
        } else {
            out.other.push_back(c);
        }
    };

    size_t i = 0;
    while (i < text.size()) {
        const bool is_open = (i + kOpenLen <= text.size() && text.compare(i, kOpenLen, kOpen) == 0);
        const bool is_close = (!is_open && i + kCloseLen <= text.size() &&
                               text.compare(i, kCloseLen, kClose) == 0);
        if (is_open) {
            // The header runs from "<|channel>" up to the FIRST of:
            //   1. a newline, or
            //   2. a "<channel|>" marker (Q5_K_M variant — see strip_channel_headers comment).
            size_t name_start = i + kOpenLen;
            size_t nl = text.find('\n', name_start);
            size_t cls = text.find(kClose, name_start);
            size_t end = std::min<size_t>(nl == std::string::npos ? text.size() : nl,
                                          cls == std::string::npos ? text.size() : cls);
            std::string name = text.substr(name_start, end - name_start);
            // Trim header name (whitespace, args after first space)
            size_t s = name.find_first_not_of("\n\r\t ");
            size_t e = name.find_last_not_of("\n\r\t ");
            if (s == std::string::npos) {
                name.clear();
            } else {
                name = name.substr(s, e - s + 1);
                size_t sp = name.find_first_of(" \t");
                if (sp != std::string::npos)
                    name = name.substr(0, sp);
            }
            current_channel = std::move(name);
            // Skips past the header (including its trailing \n if present). If a <channel|> marker ended the
            // header instead, it is left for the close-marker branch next iteration, so this rule stays a
            // no-op rather than swallowing body text.
            if (end == nl && nl != std::string::npos) {
                i = nl + 1;
            } else {
                i = end;
            }
            continue;
        }
        if (is_close) {
            // A standalone <channel|> CLOSES the current channel and everything after is the user-facing
            // answer (observed Gemma-4 shape: "<|channel>thought\nTHOUGHT<channel|>FINAL" has no explicit
            // "final" opener - the template prefix already supplied it). Treated as switch-to-content.
            current_channel.clear();
            i += kCloseLen;
            continue;
        }
        append_to_channel(text[i++]);
    }

    auto trim = [](std::string& s) {
        size_t a = s.find_first_not_of("\n\r\t ");
        if (a == std::string::npos) {
            s.clear();
            return;
        }
        size_t b = s.find_last_not_of("\n\r\t ");
        s = s.substr(a, b - a + 1);
    };
    trim(out.reasoning);
    trim(out.content);
    trim(out.other);
    return out;
}

ChannelSegments split_harmony_channels(const std::string& text) {
    // gpt-oss Harmony output: "<|channel|>analysis<|message|>...<|end|><|start|>assistant<|channel|>
    // final<|message|>..."; analysis/commentary -> reasoning_content, final -> content, all control
    // markup and <|start|>role plumbing stripped.
    static const std::string CH = "<|channel|>";
    static const std::string MSG = "<|message|>";
    static const std::string END = "<|end|>";
    static const std::string START = "<|start|>";
    static const std::string RET = "<|return|>";
    static const std::string CALL = "<|call|>";

    ChannelSegments out;
    std::string cur;        // current channel name; empty = no active channel
    std::string recipient;  // "functions.NAME" from the channel header, if any
    std::string tool_body;  // body of a call addressed to a recipient
    bool in_msg = false;
    const size_t n = text.size();
    size_t i = 0;
    auto at = [&](const std::string& m) { return text.compare(i, m.size(), m) == 0; };
    auto emit = [&](char c) {
        if (!recipient.empty())
            tool_body.push_back(c);
        else if (cur == "analysis" || cur == "commentary")
            out.reasoning.push_back(c);
        else if (cur == "final")
            out.content.push_back(c);
        else
            out.other.push_back(c);
    };
    // Close whatever block is open. A body addressed to `functions.NAME`
    // becomes a tool call rather than text.
    auto close_block = [&]() {
        constexpr const char* kFns = "functions.";
        if (!recipient.empty() && recipient.compare(0, 10, kFns) == 0) {
            size_t a = tool_body.find_first_not_of("\n\r\t ");
            size_t b = tool_body.find_last_not_of("\n\r\t ");
            if (a != std::string::npos)
                out.tool_calls.push_back({recipient.substr(10), tool_body.substr(a, b - a + 1)});
        }
        recipient.clear();
        tool_body.clear();
        cur.clear();
        in_msg = false;
    };
    while (i < n) {
        if (at(CH)) {
            i += CH.size();
            // The Harmony header between <|channel|> and <|message|> can carry a recipient/constraint too
            // ("commentary to=functions.X <|constrain|>json"); splitting only on '<' matched no known
            // channel and dropped the body (#1716). Take the first token as the channel, read `to=` from the
            // rest.
            std::string header;
            while (i < n && !at(MSG) && !at(END) && !at(START) && !at(CH) && text[i] != '<')
                header.push_back(text[i++]);
            size_t s = header.find_first_not_of("\n\r\t ");
            if (s == std::string::npos) {
                cur.clear();
                recipient.clear();
            } else {
                size_t sp = header.find_first_of("\n\r\t ", s);
                cur = header.substr(s, sp == std::string::npos ? std::string::npos : sp - s);
                recipient.clear();
                if (sp != std::string::npos) {
                    const size_t to = header.find("to=", sp);
                    if (to != std::string::npos) {
                        size_t e = header.find_first_of("\n\r\t ", to + 3);
                        recipient = header.substr(to + 3,
                                                  e == std::string::npos ? std::string::npos : e - (to + 3));
                    }
                }
            }
            tool_body.clear();
            in_msg = false;
            continue;
        }
        if (at(MSG)) {
            i += MSG.size();
            in_msg = true;
            continue;
        }
        if (at(END)) {
            i += END.size();
            close_block();
            continue;
        }
        if (at(CALL)) {
            // The marker that ends a tool call. It was in no control set, so
            // it used to fall through as literal text.
            i += CALL.size();
            close_block();
            continue;
        }
        if (at(RET)) {
            i += RET.size();
            close_block();
            continue;
        }
        if (at(START)) {
            i += START.size();
            close_block();
            // Drop the role name up to the next control marker.
            while (i < n && !at(CH) && !at(MSG) && text[i] != '<')
                i++;
            continue;
        }
        if (in_msg)
            emit(text[i]);
        i++;
    }
    // A truncated generation - max_tokens, or a stop before <|call|> - leaves
    // the last block open. Close it, or the tool call the model did emit is
    // lost to a missing terminator.
    close_block();

    auto trim = [](std::string& s) {
        size_t a = s.find_first_not_of("\n\r\t ");
        if (a == std::string::npos) {
            s.clear();
            return;
        }
        size_t b = s.find_last_not_of("\n\r\t ");
        s = s.substr(a, b - a + 1);
    };
    trim(out.reasoning);
    trim(out.content);
    trim(out.other);
    return out;
}

void strip_channel_headers(std::string& text) {
    // Scan for "<|channel>" and "<channel|>" markers. Each one begins a header
    // that runs until the next '\n'. Remove the markers and the characters up
    // to (and including) that newline. Body text between headers is kept.
    static const char kOpen[] = "<|channel>";
    static const char kClose[] = "<channel|>";
    constexpr size_t kOpenLen = sizeof(kOpen) - 1;
    constexpr size_t kCloseLen = sizeof(kClose) - 1;
    std::string out;
    out.reserve(text.size());
    size_t i = 0;
    while (i < text.size()) {
        const bool is_open = (i + kOpenLen <= text.size() && text.compare(i, kOpenLen, kOpen) == 0);
        const bool is_close = (!is_open && i + kCloseLen <= text.size() &&
                               text.compare(i, kCloseLen, kClose) == 0);
        if (is_open) {
            // Open marker: "<|channel>NAME\n" — strip the whole header.
            // If no newline follows the name, the header is malformed/
            // truncated; drop just the marker so we don't swallow body text.
            size_t nl = text.find('\n', i + kOpenLen);
            i = (nl == std::string::npos) ? (i + kOpenLen) : (nl + 1);
            continue;
        }
        if (is_close) {
            // Drops the close/channel-switch marker token itself without waiting for a trailing newline:
            // Gemma-4 Q5_K_M emits "<channel|>answer body" with none, and waiting would eat the answer
            // (observed on "What is 5+3?").
            i += kCloseLen;
            continue;
        }
        out.push_back(text[i++]);
    }
    text = std::move(out);
    // Trim a single leading newline left behind by a dropped header.
    if (!text.empty() && (text.front() == '\n' || text.front() == '\r')) {
        size_t s = text.find_first_not_of("\n\r");
        text = (s == std::string::npos) ? std::string() : text.substr(s);
    }
}

std::pair<std::string, std::string> extract_reasoning(const std::string& text) {
    std::string reasoning, content;
    if (split_last_think(text, reasoning, content))
        return {reasoning, content};

    // No </think> — an unclosed <think> makes everything after it reasoning.
    auto think_start = text.find("<think>");
    if (think_start != std::string::npos) {
        std::string reasoning_only = text.substr(think_start + 7);
        auto rs = reasoning_only.find_first_not_of("\n\r\t ");
        auto re = reasoning_only.find_last_not_of("\n\r\t ");
        reasoning_only = (rs != std::string::npos && re != std::string::npos)
                             ? reasoning_only.substr(rs, re - rs + 1)
                             : std::string();
        return {reasoning_only, ""};
    }

    return {"", text};
}

std::string sse_chunk(const std::string& id, int64_t created, const std::string& model, const json& delta,
                      const char* finish_reason, const json& logprobs, const char* finish_detail) {
    json choice = {{"index", 0},
                   {"delta", delta},
                   {"finish_reason", finish_reason ? json(finish_reason) : json(nullptr)}};
    if (finish_detail) {
        choice["imp_finish_detail"] = finish_detail;
    }
    if (!logprobs.is_null()) {
        choice["logprobs"] = logprobs;
    }
    json obj = {{"id", id},
                {"object", "chat.completion.chunk"},
                {"created", created},
                {"model", model},
                {"system_fingerprint", system_fingerprint(model)},
                {"choices", json::array({choice})}};
    return "data: " + dump_safe(obj) + "\n\n";
}

std::string sse_completion_chunk(const std::string& id, int64_t created, const std::string& model,
                                 const std::string& text, const char* finish_reason, const json& logprobs) {
    json choice = {{"index", 0},
                   {"text", text},
                   {"logprobs", logprobs},
                   {"finish_reason", finish_reason ? json(finish_reason) : json(nullptr)}};
    json obj = {{"id", id},
                {"object", "text_completion"},
                {"created", created},
                {"model", model},
                {"system_fingerprint", system_fingerprint(model)},
                {"choices", json::array({choice})}};
    return "data: " + dump_safe(obj) + "\n\n";
}

int parse_max_tokens_field(const json& body, int def) {
    int v = def;
    if (body.contains("max_tokens") && body["max_tokens"].is_number())
        v = body["max_tokens"].get<int>();
    // Current OpenAI SDKs send "max_completion_tokens" (max_tokens is
    // deprecated on chat/completions) — honor it with precedence.
    if (body.contains("max_completion_tokens") && body["max_completion_tokens"].is_number())
        v = body["max_completion_tokens"].get<int>();
    return v;
}

bool parse_stop_field(const json& body, size_t cap, std::vector<std::string>& out) {
    if (!body.contains("stop") || body["stop"].is_null())
        return false;
    const json& stop = body["stop"];
    if (stop.is_string()) {
        out.push_back(stop.get<std::string>());
        return false;
    }
    if (!stop.is_array())
        return false;
    bool truncated = false;
    for (const auto& s : stop) {
        if (!s.is_string())
            continue;
        if (out.size() >= cap) {
            truncated = true;
            break;
        }
        out.push_back(s.get<std::string>());
    }
    return truncated;
}
