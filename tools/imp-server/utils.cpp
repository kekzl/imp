#include "utils.h"
#include "stream_pipeline.h"

#include <cstdio>

std::string dump_safe(const json& j) {
    // error_handler_t::replace: emit U+FFFD for ill-formed UTF-8 instead of
    // throwing json::type_error.316. Identical output for valid UTF-8.
    return j.dump(-1, ' ', false, json::error_handler_t::replace);
}

void send_json_error(httplib::Response& res, int status, const char* type, const std::string& message) {
    json err = {{"error", {{"message", message}, {"type", type}}}};
    res.status = status;
    res.set_content(dump_safe(err), "application/json");
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
