#include "tool_call_dialect.h"
#include "utils.h"

#include <string>
#include <utility>
#include <vector>

// Harmony (gpt-oss): a tool call is a channel addressed to a function, not a tag
// (<|channel|>commentary to=functions.NAME ...<|message|>{args}<|call|>).

std::pair<std::string, std::vector<ParsedToolCall>> parse_tool_calls_harmony(
    const std::string& text, std::atomic<int>& next_tool_call_id) {
    ChannelSegments segs = split_harmony_channels(text);
    std::vector<ParsedToolCall> calls;
    for (auto& tc : segs.tool_calls) {
        ParsedToolCall p;
        p.name = std::move(tc.name);
        // Keeps the tool-call body as text verbatim rather than re-serializing: a re-dump would silently
        // normalize whatever the model produced, and invalid JSON is the caller's to see.
        // validate_tool_call() checks it against the schema afterward.
        p.arguments = std::move(tc.arguments);
        p.id = "call_" + std::to_string(next_tool_call_id.fetch_add(1));
        calls.push_back(std::move(p));
    }
    // The visible answer is the final channel. Without this the caller would
    // get the raw Harmony markup back alongside the calls.
    return {std::move(segs.content), std::move(calls)};
}

namespace {

std::pair<std::string, std::vector<ParsedToolCall>> parse_harmony(const std::string& text,
                                                                  std::atomic<int>& next_id,
                                                                  const std::vector<std::string>&) {
    return parse_tool_calls_harmony(text, next_id);
}

ToolTagScan scan_harmony(const std::string& buf) {
    ToolTagScan r;
    // The name is in the channel header, so the body is the bare arguments object - the same shape
    // parse_stream_tool_body handles for Llama3.
    constexpr const char* kTo = "to=functions.";
    constexpr size_t kToLen = 13;
    const size_t to_pos = buf.find(kTo);
    if (to_pos != std::string::npos) {
        static const std::string MSG = "<|message|>";
        const size_t msg = buf.find(MSG, to_pos);
        if (msg == std::string::npos) {
            r.kind = ToolTagScan::Kind::PARTIAL;  // header not finished yet
            return r;
        }
        size_t e = buf.find_first_of("\n\r\t <", to_pos + kToLen);
        if (e == std::string::npos || e > msg)
            e = msg;
        // Content before the marker is whatever preceded the channel
        // header, not the header itself: rewind to the <|channel|> that
        // opened it so the markup never reaches the stream.
        static const std::string CH = "<|channel|>";
        const size_t ch = buf.rfind(CH, to_pos);
        r.kind = ToolTagScan::Kind::OPEN;
        r.content_len = ch == std::string::npos ? to_pos : ch;
        r.body_start = msg + MSG.size();
        r.close_tag = "<|call|>";
        r.fn_name = buf.substr(to_pos + kToLen, e - (to_pos + kToLen));
        return r;
    }
    if (suffix_is_marker_prefix(buf, kTo, kToLen)) {
        r.kind = ToolTagScan::Kind::PARTIAL;
        return r;
    }
    r.kind = ToolTagScan::Kind::NONE;
    return r;
}

void harmony_forced_envelope(const std::string& name, std::string& open, std::string& close) {
    // gpt-oss emits calls in this header form; split_harmony_channels reads it (#1716).
    open = "<|channel|>commentary to=functions." + name + " <|constrain|>json<|message|>";
    close = "<|call|>";
}

}  // namespace

// Prompt and prior-call replay stay ChatML (the template renders tools itself); the preamble gate
// watches <tool_call>.
extern const ToolCallDialect kHarmonyToolDialect = {
    .name = "harmony",
    .prompt = chatml_tool_prompt,
    .parse = parse_harmony,
    .scan = scan_harmony,
    .render_call = chatml_render_call,
    .tool_response = nullptr,
    .json_envelope_open = nullptr,
    .json_envelope_close = nullptr,
    .xml_body = false,
    .forced_envelope = harmony_forced_envelope,
    .native_token = nullptr,
    .tool_response_joins_assistant = false,
    .gate_open = "<tool_call>",
    .gate_close = "</tool_call>",
    .gate_tokens = true,
};
