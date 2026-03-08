#pragma once

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <string>
#include <utility>
#include <vector>
#include <cstddef>
#include <cstdint>

using json = nlohmann::json;

json safe_token_json(const std::string& text);
json token_bytes_json(const std::string& text);
size_t utf8_complete_len(const std::string& s);
void json_escape_into(std::string& out, const char* s, size_t len);

int b64_val(unsigned char c);
std::vector<uint8_t> base64_decode(const std::string& encoded);

void strip_think_block(std::string& text);
std::pair<std::string, std::string> extract_reasoning(const std::string& text);

std::string sse_chunk(const std::string& id, int64_t created,
                      const std::string& model,
                      const json& delta,
                      const char* finish_reason,
                      const json& logprobs = nullptr);

std::string sse_completion_chunk(const std::string& id, int64_t created,
                                 const std::string& model,
                                 const std::string& text,
                                 const char* finish_reason);

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

        std::string envelope_prefix =
            "data: {\"id\":\"" + esc_id +
            "\",\"object\":\"chat.completion.chunk\",\"created\":" +
            std::to_string(created) +
            ",\"model\":\"" + esc_model +
            "\",\"choices\":[{\"index\":0,\"delta\":{\"";

        std::string envelope_suffix =
            "\"},\"finish_reason\":null}]}\n\n";

        content_prefix   = envelope_prefix + "content\":\"";
        content_suffix   = envelope_suffix;
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
