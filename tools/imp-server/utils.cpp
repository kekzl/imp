#include "utils.h"
#include "stream_pipeline.h"

#include <algorithm>
#include <cstdio>

// Make a client-supplied string safe to put back into a response body.
//
// Two independent problems, one helper. Ill-formed UTF-8 reaches `dump()` and
// throws; control bytes and unbounded length reach whatever reads the response
// (a log viewer, a terminal, a dashboard). Printable ASCII only, everything
// else one '.', truncated with a marker so a reader can tell (#1618).
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

// Drop an incomplete trailing UTF-8 sequence from a FINISHED string (#1310).
//
// Utf8Stitch carries such a tail forward to the next piece, which is right for
// a stream. At the end of a generation there is no next piece: max_tokens can
// stop mid-codepoint, and those bytes then reach dump_safe(), whose
// error_handler_t::replace substitutes U+FFFD - so `message.content` carries a
// character no generated token produced. The streaming path never showed this
// because the stitcher simply never releases the tail.
//
// Same 3-byte bound as Utf8Stitch::feed: a split codepoint is at most 3 bytes
// short, and a longer tail is genuinely ill-formed input rather than a
// truncation, so it is left alone for dump_safe to handle.
void drop_incomplete_utf8_tail(std::string& s) {
    const size_t complete = imp::stream::utf8_complete_len(s);
    if (complete < s.size() && s.size() - complete <= 3)
        s.resize(complete);
}

bool reject_body_too_deep(const httplib::Request& req, httplib::Response& res) {
    // Every parser downstream is recursive and none bounds depth, nlohmann
    // included and it runs first: measured on this tree, 50 000 nested arrays
    // parse and dump() fine, 100 000 segfault the process. That is ~100 KB
    // against a 100 MiB body cap, from an unauthenticated request, and one
    // process means the SIGSEGV takes every in-flight stream with it.
    constexpr int kMaxBodyNesting = 100;
    if (req.body.empty() || json_nesting_depth(req.body, kMaxBodyNesting) <= kMaxBodyNesting)
        return false;

    res.status = 400;
    const char* msg = "request body nests deeper than 100 levels";
    json err;
    if (req.path.rfind("/v1/messages", 0) == 0) {
        err = {{"type", "error"}, {"error", {{"type", "invalid_request_error"}, {"message", msg}}}};
    } else {
        err = {{"error", {{"message", msg}, {"type", "invalid_request_error"}}}};
    }
    res.set_content(err.dump(), "application/json");
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
    // complete == 0 means no whole character fits in `max`. Emitting `max`
    // bytes here would be the very split this function exists to prevent, so
    // the cap yields: one character, whole, even if it is longer than `max`.
    // A chunk size is a hint about frame size; a half character is wrong at any
    // size. Falls back to one byte only for input that is ill-formed at `off`,
    // where there is no character to keep intact and stalling is worse.
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
    // A split character is at most 3 bytes short. A longer tail is not a split
    // character but invalid input, and utf8_complete_len parks on it — holding
    // that back would stall the stream forever, so pass it through and let
    // dump_safe replace it.
    if (complete < buf.size() && buf.size() - complete <= 3) {
        carry_.assign(buf, complete, buf.size() - complete);
        buf.resize(complete);
    }
    return buf;
}

void send_json_error(httplib::Response& res, int status, const char* type, const std::string& message) {
    json err = {{"error", {{"message", message}, {"type", type}}}};
    res.status = status;
    res.set_content(dump_safe(err), "application/json");
}

bool answer_lost_to_reasoning(bool has_tool_calls, const std::string& content, const std::string& reasoning) {
    return !has_tool_calls && content.empty() && !reasoning.empty();
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

// Split `text` at the LAST "</think>". On success fills `reasoning` (the text
// before it, a leading "<think>" stripped, trimmed both ends) and `content`
// (the text after it, leading whitespace trimmed) and returns true; returns
// false (out-params untouched) when there is no "</think>". Shared by the two
// non-streaming reasoning demuxers below so their split point cannot drift.
static bool split_last_think(const std::string& text, std::string& reasoning, std::string& content) {
    auto last_end = text.rfind("</think>");
    if (last_end == std::string::npos)
        return false;

    reasoning = text.substr(0, last_end);
    auto think_start = reasoning.find("<think>");
    if (think_start != std::string::npos)
        reasoning = reasoning.substr(think_start + 7);
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
            // Skip past the header — including the trailing \n if that's what
            // ended it. If a <channel|> marker ended the header, leave it for
            // the close-marker branch on the next iteration so the "header
            // separator" rule below stays a no-op rather than swallowing body.
            if (end == nl && nl != std::string::npos) {
                i = nl + 1;
            } else {
                i = end;
            }
            continue;
        }
        if (is_close) {
            // <channel|> on its own. The model's observed Gemma-4 emission is
            //   <|channel>thought\nTHOUGHT<channel|>FINAL
            // i.e. <channel|> CLOSES the current channel and the body that
            // follows is the user-facing answer (no explicit
            // <|channel>final\n<channel|> opener for the final answer — the
            // chat-template prefix already supplied that). Treat a standalone
            // close-marker as "switch back to default (content)".
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
    // gpt-oss Harmony output looks like:
    //   <|channel|>analysis<|message|>REASONING<|end|>
    //   <|start|>assistant<|channel|>final<|message|>ANSWER<|return|>
    // analysis/commentary channels carry chain-of-thought (-> reasoning_content);
    // the final channel carries the user-facing answer (-> content). All Harmony
    // control markup and the <|start|>role plumbing are stripped.
    static const std::string CH = "<|channel|>";
    static const std::string MSG = "<|message|>";
    static const std::string END = "<|end|>";
    static const std::string START = "<|start|>";
    static const std::string RET = "<|return|>";

    ChannelSegments out;
    std::string cur;  // current channel name; empty = no active channel
    bool in_msg = false;
    const size_t n = text.size();
    size_t i = 0;
    auto at = [&](const std::string& m) { return text.compare(i, m.size(), m) == 0; };
    auto emit = [&](char c) {
        if (cur == "analysis" || cur == "commentary")
            out.reasoning.push_back(c);
        else if (cur == "final")
            out.content.push_back(c);
        else
            out.other.push_back(c);
    };
    while (i < n) {
        if (at(CH)) {
            i += CH.size();
            // Channel name runs up to <|message|> (or any other control marker).
            std::string name;
            while (i < n && !at(MSG) && !at(END) && !at(START) && !at(CH) && text[i] != '<')
                name.push_back(text[i++]);
            size_t s = name.find_first_not_of("\n\r\t ");
            size_t e = name.find_last_not_of("\n\r\t ");
            cur = (s == std::string::npos) ? std::string() : name.substr(s, e - s + 1);
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
            in_msg = false;
            cur.clear();
            continue;
        }
        if (at(RET)) {
            i += RET.size();
            in_msg = false;
            cur.clear();
            continue;
        }
        if (at(START)) {
            i += START.size();
            in_msg = false;
            cur.clear();
            // Drop the role name up to the next control marker.
            while (i < n && !at(CH) && !at(MSG) && text[i] != '<')
                i++;
            continue;
        }
        if (in_msg)
            emit(text[i]);
        i++;
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
            // Close / channel-switch marker: just drop the marker token itself.
            // Gemma-4 Q5_K_M emits "<channel|>answer body" directly without a
            // trailing newline, so we must NOT wait for one here — otherwise
            // the answer body gets eaten (observed on "What is 5+3?").
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
                      const char* finish_reason, const json& logprobs) {
    json choice = {{"index", 0},
                   {"delta", delta},
                   {"finish_reason", finish_reason ? json(finish_reason) : json(nullptr)}};
    if (!logprobs.is_null()) {
        choice["logprobs"] = logprobs;
    }
    json obj = {{"id", id},
                {"object", "chat.completion.chunk"},
                {"created", created},
                {"model", model},
                {"choices", json::array({choice})}};
    return "data: " + dump_safe(obj) + "\n\n";
}

std::string sse_completion_chunk(const std::string& id, int64_t created, const std::string& model,
                                 const std::string& text, const char* finish_reason) {
    json choice = {{"index", 0},
                   {"text", text},
                   {"finish_reason", finish_reason ? json(finish_reason) : json(nullptr)}};
    json obj = {{"id", id},
                {"object", "text_completion"},
                {"created", created},
                {"model", model},
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
