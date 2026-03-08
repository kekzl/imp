#include "utils.h"

#include <cstdio>

json safe_token_json(const std::string& text) {
    std::string safe;
    safe.reserve(text.size());
    size_t i = 0;
    while (i < text.size()) {
        unsigned char c = static_cast<unsigned char>(text[i]);
        int expected = 0;
        if (c < 0x80) { expected = 1; }
        else if ((c & 0xE0) == 0xC0) { expected = 2; }
        else if ((c & 0xF0) == 0xE0) { expected = 3; }
        else if ((c & 0xF8) == 0xF0) { expected = 4; }
        else { safe += "\xEF\xBF\xBD"; i++; continue; }  // invalid lead -> U+FFFD
        if (i + expected > text.size()) {
            // Incomplete sequence at end -> U+FFFD for each remaining byte
            for (; i < text.size(); i++) safe += "\xEF\xBF\xBD";
            break;
        }
        // Validate continuation bytes
        bool valid = true;
        for (int j = 1; j < expected; j++) {
            if ((static_cast<unsigned char>(text[i + j]) & 0xC0) != 0x80) {
                valid = false; break;
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
    for (unsigned char c : text) arr.push_back(static_cast<int>(c));
    return arr;
}

size_t utf8_complete_len(const std::string& s) {
    size_t len = s.size();
    if (len == 0) return 0;
    // Walk back from end to find the start of the last codepoint
    size_t i = len - 1;
    while (i > 0 && (static_cast<unsigned char>(s[i]) & 0xC0) == 0x80)
        --i;
    unsigned char lead = static_cast<unsigned char>(s[i]);
    int expected;
    if (lead < 0x80) expected = 1;
    else if ((lead & 0xE0) == 0xC0) expected = 2;
    else if ((lead & 0xF0) == 0xE0) expected = 3;
    else if ((lead & 0xF8) == 0xF0) expected = 4;
    else return i; // invalid byte — emit up to it
    if (i + static_cast<size_t>(expected) <= len) return len; // complete
    return i; // incomplete — emit up to start of this sequence
}

void json_escape_into(std::string& out, const char* s, size_t len) {
    out.reserve(out.size() + len + 8);
    for (size_t i = 0; i < len; i++) {
        char c = s[i];
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
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
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '+') return 62;
    if (c == '/') return 63;
    return -1;
}

std::vector<uint8_t> base64_decode(const std::string& encoded) {
    std::vector<uint8_t> out;
    out.reserve(encoded.size() * 3 / 4);
    uint32_t accum = 0;
    int bits = 0;
    for (unsigned char c : encoded) {
        int val = b64_val(c);
        if (val < 0) continue;
        accum = (accum << 6) | static_cast<uint32_t>(val);
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back(static_cast<uint8_t>((accum >> bits) & 0xFF));
        }
    }
    return out;
}

void strip_think_block(std::string& text) {
    // Find the last </think> — everything after it is the actual response
    auto last_end = text.rfind("</think>");
    if (last_end != std::string::npos) {
        std::string after = text.substr(last_end + 8);
        auto start = after.find_first_not_of("\n\r\t ");
        if (start != std::string::npos) {
            after = after.substr(start);
            // If remaining text starts with another unclosed <think>, strip it
            if (after.compare(0, 7, "<think>") == 0) {
                auto next_end = after.find("</think>", 7);
                if (next_end == std::string::npos) {
                    // Unclosed trailing <think> block — discard
                    text.clear();
                    return;
                }
                // Recursive case: more think blocks after the last </think>
                text = after;
                strip_think_block(text);
                return;
            }
            text = after;
        } else {
            text.clear();
        }
        return;
    }

    // No </think> found — check if there's an opening <think>
    auto first = text.find_first_not_of("\n\r\t ");
    if (first != std::string::npos && text.compare(first, 7, "<think>") == 0) {
        // Unclosed <think> block — model didn't finish thinking, clear output
        text.clear();
    }
}

std::pair<std::string, std::string> extract_reasoning(const std::string& text) {
    // Find the last </think>
    auto last_end = text.rfind("</think>");
    if (last_end != std::string::npos) {
        std::string reasoning = text.substr(0, last_end);
        // Strip leading <think> tag
        auto think_start = reasoning.find("<think>");
        if (think_start != std::string::npos) {
            reasoning = reasoning.substr(think_start + 7);
        }
        // Trim leading/trailing whitespace from reasoning
        auto rs = reasoning.find_first_not_of("\n\r\t ");
        auto re = reasoning.find_last_not_of("\n\r\t ");
        if (rs != std::string::npos && re != std::string::npos) {
            reasoning = reasoning.substr(rs, re - rs + 1);
        } else {
            reasoning.clear();
        }

        std::string content = text.substr(last_end + 8);
        auto cs = content.find_first_not_of("\n\r\t ");
        content = (cs != std::string::npos) ? content.substr(cs) : "";

        return {reasoning, content};
    }

    // No </think> — check for unclosed <think>
    auto think_start = text.find("<think>");
    if (think_start != std::string::npos) {
        std::string reasoning = text.substr(think_start + 7);
        auto rs = reasoning.find_first_not_of("\n\r\t ");
        auto re = reasoning.find_last_not_of("\n\r\t ");
        if (rs != std::string::npos && re != std::string::npos) {
            reasoning = reasoning.substr(rs, re - rs + 1);
        } else {
            reasoning.clear();
        }
        return {reasoning, ""};
    }

    // Check for </think> without opening (special token was skipped)
    // — text before </think> is reasoning
    // (This case shouldn't happen since we checked rfind above, but handle gracefully)

    return {"", text};
}

std::string sse_chunk(const std::string& id, int64_t created,
                      const std::string& model,
                      const json& delta,
                      const char* finish_reason,
                      const json& logprobs) {
    json choice = {
        {"index", 0},
        {"delta", delta},
        {"finish_reason", finish_reason ? json(finish_reason) : json(nullptr)}
    };
    if (!logprobs.is_null()) {
        choice["logprobs"] = logprobs;
    }
    json obj = {
        {"id", id},
        {"object", "chat.completion.chunk"},
        {"created", created},
        {"model", model},
        {"choices", json::array({choice})}
    };
    return "data: " + obj.dump() + "\n\n";
}

std::string sse_completion_chunk(const std::string& id, int64_t created,
                                 const std::string& model,
                                 const std::string& text,
                                 const char* finish_reason) {
    json choice = {
        {"index", 0},
        {"text", text},
        {"finish_reason", finish_reason ? json(finish_reason) : json(nullptr)}
    };
    json obj = {
        {"id", id},
        {"object", "text_completion"},
        {"created", created},
        {"model", model},
        {"choices", json::array({choice})}
    };
    return "data: " + obj.dump() + "\n\n";
}
