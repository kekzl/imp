#include "args.h"
#include "model/chat_template.h"
#include "model/tokenizer.h"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <string>
#include <vector>

using json = nlohmann::json;

// Convert a token text to a JSON-safe string. Tokens may contain partial UTF-8
// sequences (e.g. emoji fragments) that nlohmann::json rejects. Replace invalid
// bytes with U+FFFD to avoid serialization errors.
static json safe_token_json(const std::string& text) {
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
        else { safe += "\xEF\xBF\xBD"; i++; continue; }  // invalid lead → U+FFFD
        if (i + expected > text.size()) {
            // Incomplete sequence at end → U+FFFD for each remaining byte
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

// Build a JSON bytes array from raw token bytes (for OpenAI logprobs "bytes" field).
static json token_bytes_json(const std::string& text) {
    json arr = json::array();
    for (unsigned char c : text) arr.push_back(static_cast<int>(c));
    return arr;
}

// Returns the number of leading bytes that form complete UTF-8 characters.
// Bytes from complete_len onwards are an incomplete trailing sequence.
static size_t utf8_complete_len(const std::string& s) {
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

// Append JSON-escaped text to output buffer. Handles ", \, control chars; passes
// through valid UTF-8 bytes unescaped.
static void json_escape_into(std::string& out, const char* s, size_t len) {
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

// Simple base64 decoder for image data URIs
static int b64_val(unsigned char c) {
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '+') return 62;
    if (c == '/') return 63;
    return -1;
}

static std::vector<uint8_t> base64_decode(const std::string& encoded) {
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

// Internal handle types (shared with imp_api.cpp and imp-cli)
#include "api/imp_internal.h"

struct ParsedToolCall {
    std::string id;         // "call_imp_0", "call_imp_1", ...
    std::string name;       // Function name
    std::string arguments;  // JSON string
};

struct ServerState {
    ImpModel model = nullptr;
    ImpContext ctx = nullptr;
    imp::Tokenizer* tok = nullptr;
    imp::ChatTemplate chat_tpl;
    bool have_template = false;
    std::string model_name;
    std::timed_mutex mtx;
    int default_max_tokens = 2048;
    int max_seq_len = 0;
    std::atomic<int> next_id{0};
    std::atomic<int> next_tool_call_id{0};
    ServerArgs default_args;
    std::string models_dir;  // directory to scan for available .gguf files
    std::string api_key;     // if non-empty, require Bearer token auth
    bool is_think_model = false;  // model has <think> token (DeepSeek R1 etc.)
    int32_t think_start_id = -1;  // <think> token ID (-1 if not present)
    int32_t think_end_id = -1;    // </think> token ID (-1 if not present)
    bool model_loaded() const { return ctx != nullptr; }
};

// Graceful shutdown
static std::atomic<httplib::Server*> g_server{nullptr};

static void signal_handler(int /*sig*/) {
    fprintf(stderr, "\nShutting down...\n");
    if (auto* svr = g_server.exchange(nullptr, std::memory_order_relaxed))
        svr->stop();
}

static std::string make_completion_id(ServerState& state) {
    return "imp-" + std::to_string(state.next_id.fetch_add(1));
}

static int64_t unix_timestamp() {
    return std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

// Build a single SSE data line
static std::string sse_chunk(const std::string& id, int64_t created,
                             const std::string& model,
                             const json& delta,
                             const char* finish_reason,
                             const json& logprobs = nullptr) {
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

// Strip all <think>...</think> blocks from model output (DeepSeek R1, Nanbeige etc.)
// Finds the LAST </think> and returns everything after it, handling:
//   1. Single <think>...reasoning...</think> response
//   2. Multiple <think>...</think> blocks (some models repeat)
//   3. Missing opening <think> (special token skipped by decode)
//   4. Missing </think> (model hit token limit before closing)
//   5. Trailing unclosed <think> after the last </think>
static void strip_think_block(std::string& text) {
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

// Extract reasoning and content from model output (for non-streaming DeepSeek format).
// Returns (reasoning, content) pair. If no think block found, reasoning is empty.
static std::pair<std::string, std::string> extract_reasoning(const std::string& text) {
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

// ============================================================================
// Tool / Function Calling support
// ============================================================================

// Build tool-definition prompt text for injection into the system message.
// Returns empty string if tools array is empty or tool_choice is "none".
static std::string build_tool_prompt(imp::ChatTemplateFamily family,
                                     const json& tools,
                                     const json& tool_choice) {
    if (tools.empty()) return "";

    // tool_choice "none" means no tool injection
    if (tool_choice.is_string() && tool_choice.get<std::string>() == "none")
        return "";

    std::string prompt;

    if (family == imp::ChatTemplateFamily::LLAMA3) {
        // Llama3 function calling format
        prompt = "\n\nYou have access to the following functions:\n\n";
        for (const auto& tool : tools) {
            if (!tool.contains("function")) continue;
            const auto& fn = tool["function"];
            json fn_desc = {
                {"name", fn.value("name", "")},
                {"description", fn.value("description", "")},
                {"parameters", fn.value("parameters", json::object())}
            };
            prompt += fn_desc.dump() + "\n\n";
        }
        prompt += "For each function call, return a JSON object within <function=function_name> tags:\n"
                  "<function=function_name>{\"param\": \"value\"}</function>\n\n"
                  "If no function call is needed, respond normally without any function tags.";
    } else {
        // ChatML (Qwen3, Hermes) and all other families — use <tool_call> format
        prompt = "\n\n# Tools\n\n"
                 "You may call one or more functions to assist with the user query.\n\n"
                 "<tools>\n" + tools.dump() + "\n</tools>\n\n"
                 "For each function call, return a JSON object within <tool_call></tool_call> XML tags:\n"
                 "<tool_call>\n"
                 "{\"name\": \"function_name\", \"arguments\": {\"param\": \"value\"}}\n"
                 "</tool_call>\n\n"
                 "If no function call is needed, respond normally without any tool_call tags.";
    }

    // Add constraints based on tool_choice
    if (tool_choice.is_string()) {
        std::string choice = tool_choice.get<std::string>();
        if (choice == "required") {
            prompt += "\n\nYou MUST call at least one tool.";
        }
    } else if (tool_choice.is_object() && tool_choice.contains("function")) {
        std::string fn_name = tool_choice["function"].value("name", "");
        if (!fn_name.empty()) {
            prompt += "\n\nYou MUST call the " + fn_name + " tool.";
        }
    }

    return prompt;
}

// Parse tool calls from ChatML model output (<tool_call>JSON</tool_call>).
// Returns (content_before_first_tag, vector_of_tool_calls).
static std::pair<std::string, std::vector<ParsedToolCall>>
parse_tool_calls_chatml(const std::string& text, ServerState& state) {
    std::vector<ParsedToolCall> calls;
    std::string content;

    size_t pos = 0;
    size_t first_tag = text.find("<tool_call>");
    if (first_tag == std::string::npos) {
        return {text, {}};
    }

    // Content is everything before the first <tool_call>
    content = text.substr(0, first_tag);
    // Trim trailing whitespace
    auto last = content.find_last_not_of("\n\r\t ");
    if (last != std::string::npos) content = content.substr(0, last + 1);
    else content.clear();

    pos = first_tag;
    while (pos < text.size()) {
        size_t start = text.find("<tool_call>", pos);
        if (start == std::string::npos) break;
        start += 11; // skip "<tool_call>"

        size_t end = text.find("</tool_call>", start);
        if (end == std::string::npos) break; // incomplete tag

        std::string body = text.substr(start, end - start);
        // Trim whitespace
        auto bs = body.find_first_not_of("\n\r\t ");
        auto be = body.find_last_not_of("\n\r\t ");
        if (bs != std::string::npos && be != std::string::npos)
            body = body.substr(bs, be - bs + 1);

        // Parse JSON
        try {
            json j = json::parse(body);
            ParsedToolCall tc;
            tc.id = "call_imp_" + std::to_string(state.next_tool_call_id.fetch_add(1));
            tc.name = j.value("name", "");
            if (j.contains("arguments")) {
                tc.arguments = j["arguments"].dump();
            } else {
                // Some models put params at top level minus "name"
                json args = j;
                args.erase("name");
                tc.arguments = args.dump();
            }
            if (!tc.name.empty()) {
                calls.push_back(std::move(tc));
            }
        } catch (...) {
            // Malformed JSON — skip
        }

        pos = end + 12; // skip "</tool_call>"
    }

    return {content, calls};
}

// Parse tool calls from Llama3 model output (<function=name>JSON</function>).
static std::pair<std::string, std::vector<ParsedToolCall>>
parse_tool_calls_llama3(const std::string& text, ServerState& state) {
    std::vector<ParsedToolCall> calls;
    std::string content;

    size_t first_tag = text.find("<function=");
    if (first_tag == std::string::npos) {
        return {text, {}};
    }

    content = text.substr(0, first_tag);
    auto last = content.find_last_not_of("\n\r\t ");
    if (last != std::string::npos) content = content.substr(0, last + 1);
    else content.clear();

    size_t pos = first_tag;
    while (pos < text.size()) {
        size_t start = text.find("<function=", pos);
        if (start == std::string::npos) break;
        start += 10; // skip "<function="

        size_t name_end = text.find('>', start);
        if (name_end == std::string::npos) break;

        std::string name = text.substr(start, name_end - start);

        size_t body_start = name_end + 1;
        size_t end = text.find("</function>", body_start);
        if (end == std::string::npos) break;

        std::string body = text.substr(body_start, end - body_start);
        auto bs = body.find_first_not_of("\n\r\t ");
        auto be = body.find_last_not_of("\n\r\t ");
        if (bs != std::string::npos && be != std::string::npos)
            body = body.substr(bs, be - bs + 1);

        try {
            // Validate it's valid JSON
            json j = json::parse(body);
            ParsedToolCall tc;
            tc.id = "call_imp_" + std::to_string(state.next_tool_call_id.fetch_add(1));
            tc.name = name;
            tc.arguments = j.dump();
            calls.push_back(std::move(tc));
        } catch (...) {
            // Malformed JSON — skip
        }

        pos = end + 11; // skip "</function>"
    }

    return {content, calls};
}

// Dispatch tool call parsing based on template family.
static std::pair<std::string, std::vector<ParsedToolCall>>
parse_tool_calls(imp::ChatTemplateFamily family, const std::string& text,
                 ServerState& state) {
    if (family == imp::ChatTemplateFamily::LLAMA3)
        return parse_tool_calls_llama3(text, state);
    return parse_tool_calls_chatml(text, state);
}

// Reconstruct assistant message content from structured tool_calls (for multi-turn).
// This rebuilds what the model originally output so the conversation stays consistent.
static std::string reconstruct_tool_call_output(imp::ChatTemplateFamily family,
                                                const json& tool_calls,
                                                const std::string& content) {
    std::string result;
    if (!content.empty() && content != "null") {
        result = content;
    }

    for (const auto& tc : tool_calls) {
        if (!tc.contains("function")) continue;
        std::string name = tc["function"].value("name", "");
        std::string args = tc["function"].value("arguments", "{}");

        if (family == imp::ChatTemplateFamily::LLAMA3) {
            result += "\n<function=" + name + ">" + args + "</function>";
        } else {
            // ChatML format
            json call_obj = {{"name", name}, {"arguments", json::parse(args, nullptr, false)}};
            if (call_obj["arguments"].is_discarded()) call_obj["arguments"] = args;
            result += "\n<tool_call>\n" + call_obj.dump() + "\n</tool_call>";
        }
    }

    return result;
}

// Format a tool-role message content for the model.
// Wraps the content in <tool_response> tags for ChatML, passes as-is for Llama3.
static std::string format_tool_response(imp::ChatTemplateFamily family,
                                        const json& msg) {
    std::string content = msg.value("content", "");
    std::string tool_call_id = msg.value("tool_call_id", "");

    if (family == imp::ChatTemplateFamily::LLAMA3) {
        return content;
    }
    // ChatML: wrap in <tool_response> tags
    return "<tool_response>\n" + content + "\n</tool_response>";
}

static void handle_health(const httplib::Request& /*req*/, httplib::Response& res,
                          ServerState& state) {
    json body = {
        {"status", "ok"},
        {"model_loaded", state.model_loaded()}
    };
    res.set_content(body.dump(), "application/json");
}

// Recursively find all .gguf files in a directory, returning (filename, full_path) pairs.
// Resolves symlinks and rejects any path that escapes the base directory (path traversal).
static std::vector<std::pair<std::string, std::string>> scan_gguf_files(const std::string& dir) {
    std::vector<std::pair<std::string, std::string>> results;
    if (dir.empty()) return results;
    std::error_code ec;
    auto base = std::filesystem::canonical(dir, ec);
    if (ec) return results;
    std::string base_prefix = base.string() + "/";

    for (const auto& entry : std::filesystem::recursive_directory_iterator(dir, ec)) {
        if (!entry.is_regular_file() && !entry.is_symlink()) continue;
        auto path = entry.path();
        if (path.extension() != ".gguf" || path.string().find(".no_exist") != std::string::npos)
            continue;
        // Resolve to real path and verify it stays under the base directory
        std::error_code ec2;
        auto real = std::filesystem::canonical(path, ec2);
        if (ec2) continue;
        std::string real_str = real.string();
        if (real_str.compare(0, base_prefix.size(), base_prefix) != 0) continue;
        results.emplace_back(path.filename().string(), real_str);
    }
    std::sort(results.begin(), results.end());
    return results;
}

static void handle_models(const httplib::Request& /*req*/, httplib::Response& res,
                          ServerState& state) {
    json data = json::array();

    // Scan models directory for all available .gguf files
    auto available = scan_gguf_files(state.models_dir);
    if (!available.empty()) {
        for (const auto& [name, path] : available) {
            data.push_back({
                {"id", name},
                {"object", "model"},
                {"owned_by", "imp"},
                {"path", path}
            });
        }
    } else if (state.model_loaded()) {
        // Fallback: only show loaded model if no models_dir configured
        data.push_back({
            {"id", state.model_name},
            {"object", "model"},
            {"owned_by", "imp"}
        });
    }

    json body = {
        {"object", "list"},
        {"data", data}
    };
    res.set_content(body.dump(), "application/json");
}

// Forward declaration (defined after build_config)
static std::string load_model_into_state(ServerState& state, const std::string& path,
                                         const json& config_overrides = json::object());

// Find a GGUF file by name in models_dir. Returns full path or empty string.
static std::string find_model_path(const ServerState& state, const std::string& name) {
    auto available = scan_gguf_files(state.models_dir);
    for (const auto& [fname, fpath] : available) {
        if (fname == name) return fpath;
    }
    return "";
}

static void handle_chat_completions(const httplib::Request& req, httplib::Response& res,
                                    ServerState& state) {
    // Parse request body
    json body;
    try {
        body = json::parse(req.body);
    } catch (const json::parse_error& e) {
        res.status = 400;
        json err = {{"error", {{"message", std::string("Invalid JSON: ") + e.what()},
                                {"type", "invalid_request_error"}}}};
        res.set_content(err.dump(), "application/json");
        return;
    }

    // Extract parameters
    auto messages = body.value("messages", json::array());
    if (messages.empty()) {
        res.status = 400;
        json err = {{"error", {{"message", "messages array is required and must not be empty"},
                                {"type", "invalid_request_error"}}}};
        res.set_content(err.dump(), "application/json");
        return;
    }

    float temperature = body.value("temperature", 0.7f);
    float top_p = body.value("top_p", 0.9f);
    int top_k = body.value("top_k", 40);
    int max_tokens = body.value("max_tokens", state.default_max_tokens);
    int seed = body.value("seed", -1);
    bool stream = body.value("stream", false);
    float min_p = body.value("min_p", 0.0f);
    float typical_p = body.value("typical_p", 1.0f);
    float repetition_penalty = body.value("repetition_penalty", 1.0f);
    float frequency_penalty = body.value("frequency_penalty", 0.0f);
    float presence_penalty = body.value("presence_penalty", 0.0f);
    float dry_multiplier = body.value("dry_multiplier", 0.0f);
    float dry_base = body.value("dry_base", 1.75f);
    int dry_allowed_length = body.value("dry_allowed_length", 2);
    int dry_penalty_last_n = body.value("dry_penalty_last_n", 0);
    int mirostat = body.value("mirostat", 0);
    float mirostat_tau = body.value("mirostat_tau", 5.0f);
    float mirostat_eta = body.value("mirostat_eta", 0.1f);

    // Prevent repetition degeneration when no penalty is explicitly set.
    // Reasoning models (DeepSeek-R1 distills) are prone to infinite loops
    // in their thinking phase without this.
    if (!body.contains("repetition_penalty") && !body.contains("frequency_penalty")
        && !body.contains("presence_penalty")) {
        frequency_penalty = 0.3f;
    }

    // Parse stop sequences (string or array of up to 4 strings)
    std::vector<std::string> stop_sequences;
    if (body.contains("stop") && !body["stop"].is_null()) {
        if (body["stop"].is_string()) {
            stop_sequences.push_back(body["stop"].get<std::string>());
        } else if (body["stop"].is_array()) {
            for (const auto& s : body["stop"]) {
                if (s.is_string()) {
                    stop_sequences.push_back(s.get<std::string>());
                    if (stop_sequences.size() >= 4) break;
                }
            }
        }
    }
    size_t max_stop_len = 0;
    for (const auto& s : stop_sequences) max_stop_len = std::max(max_stop_len, s.size());

    // Parse logprobs parameters
    bool req_logprobs = body.value("logprobs", false);
    int top_logprobs = body.value("top_logprobs", 0);
    if (top_logprobs < 0) top_logprobs = 0;
    if (top_logprobs > 20) top_logprobs = 20;

    // Parse response_format for JSON mode
    bool json_mode = false;
    if (body.contains("response_format") && body["response_format"].is_object()) {
        std::string fmt_type = body["response_format"].value("type", "text");
        if (fmt_type == "json_object") json_mode = true;
    }

    // Parse stream_options for include_usage
    bool include_usage = false;
    if (body.contains("stream_options") && body["stream_options"].is_object()) {
        include_usage = body["stream_options"].value("include_usage", false);
    }

    // Parse tool calling parameters
    json tools = body.value("tools", json::array());
    json tool_choice = body.value("tool_choice", json("auto"));
    bool has_tools = !tools.empty() &&
        !(tool_choice.is_string() && tool_choice.get<std::string>() == "none");

    // Convert JSON messages to ChatMessage vector, extracting image data if present
    std::vector<imp::ChatMessage> chat_msgs;
    std::vector<uint8_t> image_data;  // decoded image bytes (if any)
    imp::ChatTemplateFamily tpl_family = state.have_template
        ? state.chat_tpl.family() : imp::ChatTemplateFamily::CHATML;

    for (const auto& msg : messages) {
        std::string role = msg.value("role", "user");

        if (role == "tool") {
            // Tool response message — format for the model
            std::string content = format_tool_response(tpl_family, msg);
            chat_msgs.push_back({"tool", content});
        } else if (role == "assistant" && msg.contains("tool_calls")) {
            // Assistant message with tool_calls — reconstruct model output format
            std::string content_str;
            if (msg.contains("content") && !msg["content"].is_null()) {
                content_str = msg["content"].get<std::string>();
            }
            std::string reconstructed = reconstruct_tool_call_output(
                tpl_family, msg["tool_calls"], content_str);
            chat_msgs.push_back({"assistant", reconstructed});
        } else if (msg.contains("content") && msg["content"].is_array()) {
            // OpenAI multimodal format: content is array of parts
            std::string text_parts;
            for (const auto& part : msg["content"]) {
                std::string type = part.value("type", "");
                if (type == "text") {
                    if (!text_parts.empty()) text_parts += "\n";
                    text_parts += part.value("text", "");
                } else if (type == "image_url" && part.contains("image_url")) {
                    std::string url = part["image_url"].value("url", "");
                    // Handle data URI: data:image/...;base64,...
                    auto comma = url.find(',');
                    if (url.rfind("data:", 0) == 0 && comma != std::string::npos) {
                        image_data = base64_decode(url.substr(comma + 1));
                    }
                }
            }
            chat_msgs.push_back({role, text_parts});
        } else {
            std::string content;
            if (msg.contains("content") && !msg["content"].is_null()) {
                content = msg["content"].get<std::string>();
            }
            chat_msgs.push_back({role, content});
        }
    }

    // Inject tool definitions into system message
    if (has_tools && state.have_template) {
        std::string tool_prompt = build_tool_prompt(tpl_family, tools, tool_choice);
        if (!tool_prompt.empty()) {
            // Find or create system message
            bool found_system = false;
            for (auto& m : chat_msgs) {
                if (m.role == "system") {
                    m.content += tool_prompt;
                    found_system = true;
                    break;
                }
            }
            if (!found_system) {
                std::string sys = state.chat_tpl.default_system_message();
                if (sys.empty()) sys = "You are a helpful assistant.";
                sys += tool_prompt;
                chat_msgs.insert(chat_msgs.begin(), {"system", sys});
            }
        }
    }

    // Acquire inference lock — wait up to 5 minutes (covers model load + generation)
    std::unique_lock<std::timed_mutex> lock(state.mtx, std::chrono::minutes(5));
    if (!lock.owns_lock()) {
        res.status = 503;
        json err = {{"error", {{"message", "Server is busy processing another request. Please retry."},
                                {"type", "server_error"}}}};
        res.set_content(err.dump(), "application/json");
        return;
    }

    // Auto-load model if request specifies a different one
    std::string requested_model = body.value("model", "");
    if (!requested_model.empty() && requested_model != state.model_name) {
        std::string path = find_model_path(state, requested_model);
        if (!path.empty()) {
            printf("Auto-loading model: %s\n", requested_model.c_str());
            fflush(stdout);
            std::string error = load_model_into_state(state, path);
            if (!error.empty()) {
                res.status = 500;
                json err = {{"error", {{"message", error}, {"type", "server_error"}}}};
                res.set_content(err.dump(), "application/json");
                return;
            }
            printf("Model loaded: %s\n", state.model_name.c_str());
            fflush(stdout);
        }
    }

    // Check if a model is loaded
    if (!state.model_loaded()) {
        res.status = 503;
        json err = {{"error", {{"message", "No model loaded. Use POST /v1/models to load one."},
                                {"type", "server_error"}}}};
        res.set_content(err.dump(), "application/json");
        return;
    }

    // Handle vision: clear previous, encode new image if present
    state.ctx->engine->clear_image();
    if (!image_data.empty() && state.ctx->engine->has_vision()) {
        if (!state.ctx->engine->set_image_from_memory(image_data.data(), image_data.size())) {
            res.status = 400;
            json error = {{"error", {{"message", "Failed to process image"},
                                      {"type", "invalid_request_error"}}}};
            res.set_content(error.dump(), "application/json");
            return;
        }
    }

    // Tokenize with chat template (with image tokens if vision is active)
    std::vector<int32_t> tokens;
    if (state.have_template && state.ctx->engine->has_vision_input()) {
        tokens = state.chat_tpl.apply_with_image(*state.tok, chat_msgs, 256);
    } else if (state.have_template) {
        tokens = state.chat_tpl.apply(*state.tok, chat_msgs);
    } else {
        // Concatenate all message content as raw text
        std::string raw;
        for (const auto& m : chat_msgs) raw += m.content + "\n";
        tokens = state.tok->encode(raw);
    }

    // Optionally append <think> token to trigger reasoning mode.
    // Enabled when: model supports thinking AND reasoning format is deepseek AND
    // either the request explicitly asks for it, or no system message disables it.
    bool enable_thinking = false;
    if (state.is_think_model && state.default_args.reasoning_format == "deepseek" &&
        state.think_start_id >= 0) {
        // Check for explicit enable/disable via request body
        if (body.contains("enable_thinking")) {
            enable_thinking = body.value("enable_thinking", false);
        }
        if (enable_thinking) {
            tokens.push_back(state.think_start_id);
        }
    }

    int n_prompt_tokens = static_cast<int>(tokens.size());

    // Validate prompt length against context window
    if (n_prompt_tokens >= state.max_seq_len) {
        res.status = 400;
        json error = {{"error", {{"message", "Prompt exceeds context window (" +
                                              std::to_string(n_prompt_tokens) + " tokens >= " +
                                              std::to_string(state.max_seq_len) + " max)"},
                                  {"type", "invalid_request_error"}}}};
        res.set_content(error.dump(), "application/json");
        return;
    }

    // Clamp max_tokens to remaining context window
    int remaining = state.max_seq_len - n_prompt_tokens;
    if (max_tokens > remaining) max_tokens = remaining;

    // Start timing
    auto t_start = std::chrono::high_resolution_clock::now();

    // Reset context and prefill
    ImpError err = imp_context_reset(state.ctx);
    if (err != IMP_SUCCESS) {
        res.status = 500;
        json error = {{"error", {{"message", std::string("Context reset failed: ") + imp_error_string(err)},
                                  {"type", "server_error"}}}};
        res.set_content(error.dump(), "application/json");
        return;
    }

    err = imp_prefill(state.ctx, tokens.data(), n_prompt_tokens);
    if (err != IMP_SUCCESS) {
        res.status = 500;
        json error = {{"error", {{"message", std::string("Prefill failed: ") + imp_error_string(err)},
                                  {"type", "server_error"}}}};
        res.set_content(error.dump(), "application/json");
        return;
    }

    ImpGenerateParams params = imp_generate_params_default();
    params.temperature = temperature;
    params.top_p = top_p;
    params.top_k = top_k;
    params.max_tokens = max_tokens;
    params.seed = seed;
    params.min_p = min_p;
    params.typical_p = typical_p;
    params.repetition_penalty = repetition_penalty;
    params.frequency_penalty = frequency_penalty;
    params.presence_penalty = presence_penalty;
    params.dry_multiplier = dry_multiplier;
    params.dry_base = dry_base;
    params.dry_allowed_length = dry_allowed_length;
    params.dry_penalty_last_n = dry_penalty_last_n;
    params.mirostat = mirostat;
    params.mirostat_tau = mirostat_tau;
    params.mirostat_eta = mirostat_eta;
    params.logprobs = req_logprobs ? 1 : 0;
    params.top_logprobs = top_logprobs;
    params.json_mode = json_mode ? 1 : 0;

    // Capture first token produced during prefill (engine samples one token as part of prefill)
    int32_t prefill_token = -1;
    if (state.ctx->active_request && !state.ctx->active_request->output_tokens.empty()) {
        prefill_token = state.ctx->active_request->output_tokens.back();
    }

    std::string comp_id = make_completion_id(state);
    int64_t created = unix_timestamp();

    if (stream) {
        // SSE streaming response
        res.set_header("Cache-Control", "no-cache");
        res.set_header("Connection", "keep-alive");

        res.set_chunked_content_provider(
            "text/event-stream",
            [&state, params, comp_id, created, max_tokens, n_prompt_tokens, t_start,
             stop_sequences, max_stop_len, req_logprobs, include_usage,
             prefill_token, enable_thinking, has_tools, tpl_family](
                size_t /*offset*/, httplib::DataSink& sink) -> bool {

                // Save active request ref for logprobs access
                auto active_req = state.ctx->active_request;

                // Pre-build SSE envelope templates for fast content/reasoning emission
                SSEChunkWriter sse_writer(comp_id, created, state.model_name);

                // Send initial chunk with role
                json role_delta = {{"role", "assistant"}};
                std::string chunk = sse_chunk(comp_id, created, state.model_name,
                                              role_delta, nullptr);
                sink.write(chunk.data(), chunk.size());

                int n_output_tokens = 0;
                const char* finish = nullptr;

                // Buffer for incomplete UTF-8 sequences across token boundaries
                std::string utf8_buf;

                // Buffered output for stop sequence matching in streaming mode.
                // We hold back text until we're sure it doesn't contain a stop match.
                std::string pending_text;
                bool text_stop_matched = false;

                // Tool call detection state machine for streaming
                enum class ToolPhase { CONTENT, TAG_SCANNING, TOOL_CALL_BODY };
                ToolPhase tool_phase = ToolPhase::CONTENT;
                std::string tool_tag_buf;           // buffer for partial tag match
                std::string tool_body_buf;           // buffer for tool call body
                std::string tool_close_tag;          // expected closing tag
                std::string tool_fn_name;            // Llama3: extracted function name from open tag
                std::vector<ParsedToolCall> stream_tool_calls;
                bool tool_calls_emitted = false;
                // The full accumulated output (only used when has_tools, for fallback)
                std::string full_output;

                // Reasoning content extraction (DeepSeek format)
                enum class ThinkPhase { SCAN, REASONING, CONTENT };
                bool use_reasoning = (state.default_args.reasoning_format == "deepseek"
                                      && state.is_think_model);
                ThinkPhase think_phase;
                if (enable_thinking) {
                    think_phase = ThinkPhase::REASONING;  // <think> in prefill → start reasoning
                } else if (use_reasoning) {
                    think_phase = ThinkPhase::SCAN;       // model decides whether to think
                } else {
                    think_phase = ThinkPhase::CONTENT;    // no reasoning extraction
                }
                std::string reasoning_utf8_buf;
                std::string think_scan_buf;
                int think_scan_count = 0;
                int n_reasoning_tokens = 0;
                const int kThinkScanLimit = 8;

                // Helper: emit reasoning_content SSE chunk
                auto emit_reasoning = [&](const std::string& text) -> bool {
                    if (text.empty()) return true;
                    return sse_writer.write_reasoning(text, sink);
                };

                // Helper: flush confirmed text up to a byte position
                auto flush_text = [&](size_t up_to) {
                    if (up_to == 0) return true;
                    bool ok = sse_writer.write_content(
                        pending_text.data(), up_to, sink);
                    pending_text.erase(0, up_to);
                    return ok;
                };

                for (int step = -1; step < max_tokens; step++) {
                    int32_t token = 0;
                    if (step == -1) {
                        // First iteration: use the token produced during prefill
                        if (prefill_token < 0) continue;
                        token = prefill_token;
                    } else {
                        ImpError err = imp_decode_step(state.ctx, &params, &token);
                        if (err != IMP_SUCCESS) {
                            finish = "stop";
                            break;
                        }
                    }

                    // Check stop conditions
                    if (token == state.tok->eos_id()) {
                        finish = "stop";
                        break;
                    }
                    if (state.have_template) {
                        bool is_stop = false;
                        for (int32_t stop_id : state.chat_tpl.stop_token_ids()) {
                            if (token == stop_id) { is_stop = true; break; }
                        }
                        if (is_stop) {
                            finish = "stop";
                            break;
                        }
                    }

                    n_output_tokens++;
                    std::string piece = state.tok->decode_token(token);

                    // Reasoning content extraction (DeepSeek format)
                    if (think_phase == ThinkPhase::SCAN) {
                        if (token == state.think_start_id) {
                            think_phase = ThinkPhase::REASONING;
                            n_reasoning_tokens++;
                            continue;
                        }
                        think_scan_buf += piece;
                        think_scan_count++;
                        if (think_scan_buf.find("<think>") != std::string::npos) {
                            think_phase = ThinkPhase::REASONING;
                            n_reasoning_tokens += think_scan_count;
                            auto pos = think_scan_buf.find("<think>");
                            std::string after = think_scan_buf.substr(pos + 7);
                            think_scan_buf.clear();
                            if (!after.empty()) reasoning_utf8_buf += after;
                            continue;
                        }
                        if (think_scan_count == 1 && piece.empty()) {
                            think_phase = ThinkPhase::REASONING;
                            n_reasoning_tokens++;
                            continue;
                        }
                        if (think_scan_count >= kThinkScanLimit) {
                            think_phase = ThinkPhase::CONTENT;
                            piece = think_scan_buf;
                            think_scan_buf.clear();
                        } else {
                            continue;
                        }
                    }

                    if (think_phase == ThinkPhase::REASONING) {
                        n_reasoning_tokens++;
                        if (token == state.think_end_id) {
                            if (!emit_reasoning(reasoning_utf8_buf)) return false;
                            reasoning_utf8_buf.clear();
                            think_phase = ThinkPhase::CONTENT;
                            continue;
                        }
                        // Skip duplicate <think> tokens while already reasoning
                        if (token == state.think_start_id) continue;
                        reasoning_utf8_buf += piece;
                        // Strip <think> text that appears via multi-token encoding
                        for (;;) {
                            auto tp = reasoning_utf8_buf.find("<think>");
                            if (tp == std::string::npos) break;
                            reasoning_utf8_buf.erase(tp, 7);
                        }
                        auto end_pos = reasoning_utf8_buf.find("</think>");
                        if (end_pos != std::string::npos) {
                            std::string before = reasoning_utf8_buf.substr(0, end_pos);
                            if (!emit_reasoning(before)) return false;
                            think_phase = ThinkPhase::CONTENT;
                            std::string after = reasoning_utf8_buf.substr(end_pos + 8);
                            reasoning_utf8_buf.clear();
                            auto start = after.find_first_not_of("\n\r\t ");
                            if (start != std::string::npos) {
                                piece = after.substr(start);
                            } else {
                                continue;
                            }
                        } else {
                            size_t complete = utf8_complete_len(reasoning_utf8_buf);
                            if (complete > 0) {
                                std::string to_emit = reasoning_utf8_buf.substr(0, complete);
                                reasoning_utf8_buf = reasoning_utf8_buf.substr(complete);
                                if (!emit_reasoning(to_emit)) return false;
                            }
                            continue;
                        }
                    }

                    // CONTENT phase: handle stray think tokens from confused models
                    if (use_reasoning) {
                        if (token == state.think_start_id) {
                            think_phase = ThinkPhase::REASONING;
                            n_reasoning_tokens++;
                            continue;
                        }
                        if (token == state.think_end_id) {
                            n_reasoning_tokens++;
                            continue;
                        }
                        // Strip text-level think tags from content piece
                        for (;;) {
                            auto p = piece.find("<think>");
                            if (p != std::string::npos) { piece.erase(p, 7); continue; }
                            p = piece.find("</think>");
                            if (p != std::string::npos) { piece.erase(p, 8); continue; }
                            break;
                        }
                        if (piece.empty()) continue;
                    }

                    // CONTENT phase — with tool call tag detection
                    if (has_tools) full_output += piece;

                    // Tool call state machine (only active when tools are present)
                    if (has_tools && tool_phase == ToolPhase::TOOL_CALL_BODY) {
                        tool_body_buf += piece;
                        // Check for close tag
                        auto close_pos = tool_body_buf.find(tool_close_tag);
                        if (close_pos != std::string::npos) {
                            std::string body = tool_body_buf.substr(0, close_pos);
                            auto bs = body.find_first_not_of("\n\r\t ");
                            auto be = body.find_last_not_of("\n\r\t ");
                            if (bs != std::string::npos && be != std::string::npos)
                                body = body.substr(bs, be - bs + 1);

                            // Parse and emit tool call
                            try {
                                json j = json::parse(body);
                                ParsedToolCall tc;
                                tc.id = "call_imp_" + std::to_string(
                                    state.next_tool_call_id.fetch_add(1));
                                if (tpl_family == imp::ChatTemplateFamily::LLAMA3) {
                                    tc.name = tool_fn_name;
                                    tc.arguments = j.dump();
                                } else {
                                    tc.name = j.value("name", "");
                                    if (j.contains("arguments")) {
                                        tc.arguments = j["arguments"].dump();
                                    } else {
                                        json args = j;
                                        args.erase("name");
                                        tc.arguments = args.dump();
                                    }
                                }
                                if (!tc.name.empty()) {
                                    int idx = static_cast<int>(stream_tool_calls.size());
                                    // Emit name chunk
                                    json name_delta = {{"tool_calls", json::array({
                                        {{"index", idx}, {"id", tc.id},
                                         {"type", "function"},
                                         {"function", {{"name", tc.name}, {"arguments", ""}}}}
                                    })}};
                                    std::string sse = sse_chunk(comp_id, created,
                                        state.model_name, name_delta, nullptr);
                                    sink.write(sse.data(), sse.size());

                                    // Emit arguments chunk
                                    json args_delta = {{"tool_calls", json::array({
                                        {{"index", idx},
                                         {"function", {{"arguments", tc.arguments}}}}
                                    })}};
                                    sse = sse_chunk(comp_id, created,
                                        state.model_name, args_delta, nullptr);
                                    sink.write(sse.data(), sse.size());

                                    stream_tool_calls.push_back(std::move(tc));
                                    tool_calls_emitted = true;
                                }
                            } catch (...) {
                                // Malformed JSON — skip
                            }

                            // Check for more content after close tag
                            std::string after = tool_body_buf.substr(
                                close_pos + tool_close_tag.size());
                            tool_body_buf.clear();
                            tool_phase = ToolPhase::CONTENT;
                            // If there's remaining text, it might contain more tool calls
                            if (!after.empty()) {
                                auto ws = after.find_first_not_of("\n\r\t ");
                                if (ws != std::string::npos) {
                                    piece = after.substr(ws);
                                    // Fall through to CONTENT handling below
                                } else {
                                    continue;
                                }
                            } else {
                                continue;
                            }
                        } else {
                            continue; // Still collecting body
                        }
                    }

                    if (has_tools && tool_phase == ToolPhase::TAG_SCANNING) {
                        tool_tag_buf += piece;
                        // ChatML: check for <tool_call>
                        if (tpl_family != imp::ChatTemplateFamily::LLAMA3) {
                            if (tool_tag_buf.size() >= 11) { // len("<tool_call>")
                                if (tool_tag_buf.find("<tool_call>") != std::string::npos) {
                                    auto pos = tool_tag_buf.find("<tool_call>");
                                    // Flush content before the tag
                                    std::string before = tool_tag_buf.substr(0, pos);
                                    if (!before.empty()) {
                                        json cd = {{"content", before}};
                                        std::string sse = sse_chunk(comp_id, created,
                                            state.model_name, cd, nullptr);
                                        sink.write(sse.data(), sse.size());
                                    }
                                    tool_body_buf = tool_tag_buf.substr(pos + 11);
                                    tool_close_tag = "</tool_call>";
                                    tool_tag_buf.clear();
                                    tool_phase = ToolPhase::TOOL_CALL_BODY;
                                    continue;
                                }
                                // Check if it's definitely not a tool_call tag
                                if (tool_tag_buf.find("<tool_call") == std::string::npos &&
                                    tool_tag_buf.find("<tool_c") == std::string::npos &&
                                    tool_tag_buf.find("<tool_") == std::string::npos &&
                                    tool_tag_buf.find("<tool") == std::string::npos &&
                                    tool_tag_buf.find("<too") == std::string::npos &&
                                    tool_tag_buf.find("<to") == std::string::npos &&
                                    tool_tag_buf.find("<t") == std::string::npos) {
                                    // Not a tool tag — flush as content
                                    piece = tool_tag_buf;
                                    tool_tag_buf.clear();
                                    tool_phase = ToolPhase::CONTENT;
                                    // Fall through to content emission
                                } else {
                                    continue; // Still scanning
                                }
                            } else {
                                // Check partial match
                                const char* tc_tag = "<tool_call>";
                                bool could_match = true;
                                for (size_t ci = 0; ci < tool_tag_buf.size() && ci < 11; ci++) {
                                    if (tool_tag_buf[ci] != tc_tag[ci]) {
                                        could_match = false;
                                        break;
                                    }
                                }
                                if (!could_match) {
                                    piece = tool_tag_buf;
                                    tool_tag_buf.clear();
                                    tool_phase = ToolPhase::CONTENT;
                                } else {
                                    continue; // Still matching prefix
                                }
                            }
                        } else {
                            // Llama3: check for <function=
                            if (tool_tag_buf.size() >= 10) { // len("<function=")
                                auto fn_pos = tool_tag_buf.find("<function=");
                                if (fn_pos != std::string::npos) {
                                    auto gt = tool_tag_buf.find('>', fn_pos + 10);
                                    if (gt != std::string::npos) {
                                        std::string before = tool_tag_buf.substr(0, fn_pos);
                                        if (!before.empty()) {
                                            json cd = {{"content", before}};
                                            std::string sse = sse_chunk(comp_id, created,
                                                state.model_name, cd, nullptr);
                                            sink.write(sse.data(), sse.size());
                                        }
                                        tool_fn_name = tool_tag_buf.substr(fn_pos + 10,
                                            gt - (fn_pos + 10));
                                        tool_body_buf = tool_tag_buf.substr(gt + 1);
                                        tool_close_tag = "</function>";
                                        tool_tag_buf.clear();
                                        tool_phase = ToolPhase::TOOL_CALL_BODY;
                                        continue;
                                    } else {
                                        continue; // Still scanning for >
                                    }
                                }
                                // Check prefix match
                                const char* fn_tag = "<function=";
                                bool could_match = true;
                                for (size_t ci = 0; ci < tool_tag_buf.size() && ci < 10; ci++) {
                                    if (tool_tag_buf[ci] != fn_tag[ci]) {
                                        could_match = false;
                                        break;
                                    }
                                }
                                if (!could_match) {
                                    piece = tool_tag_buf;
                                    tool_tag_buf.clear();
                                    tool_phase = ToolPhase::CONTENT;
                                } else {
                                    continue;
                                }
                            } else {
                                const char* fn_tag = "<function=";
                                bool could_match = true;
                                for (size_t ci = 0; ci < tool_tag_buf.size() && ci < 10; ci++) {
                                    if (tool_tag_buf[ci] != fn_tag[ci]) {
                                        could_match = false;
                                        break;
                                    }
                                }
                                if (!could_match) {
                                    piece = tool_tag_buf;
                                    tool_tag_buf.clear();
                                    tool_phase = ToolPhase::CONTENT;
                                } else {
                                    continue;
                                }
                            }
                        }
                    }

                    // In CONTENT phase, check for start of tool call tag
                    if (has_tools && tool_phase == ToolPhase::CONTENT) {
                        // Look for < that might start a tool call tag
                        size_t lt_pos = piece.find('<');
                        if (lt_pos != std::string::npos) {
                            // Emit everything before the <
                            if (lt_pos > 0) {
                                std::string before = piece.substr(0, lt_pos);
                                if (stop_sequences.empty()) {
                                    utf8_buf += before;
                                } else {
                                    pending_text += before;
                                }
                            }
                            // Start tag scanning with the < and everything after
                            tool_tag_buf = piece.substr(lt_pos);
                            tool_phase = ToolPhase::TAG_SCANNING;
                            // Flush any buffered content before entering tag scan
                            if (stop_sequences.empty() && !utf8_buf.empty()) {
                                size_t complete = utf8_complete_len(utf8_buf);
                                if (complete > 0) {
                                    if (!sse_writer.write_content(
                                            utf8_buf.data(), complete, sink))
                                        return false;
                                    utf8_buf.erase(0, complete);
                                }
                            } else if (!stop_sequences.empty()) {
                                bool stop_found = false;
                                for (const auto& stop : stop_sequences) {
                                    auto pos = pending_text.find(stop);
                                    if (pos != std::string::npos) {
                                        if (!flush_text(pos)) return false;
                                        stop_found = true;
                                        break;
                                    }
                                }
                                if (stop_found) {
                                    text_stop_matched = true;
                                    finish = "stop";
                                    break;
                                }
                                if (pending_text.size() > max_stop_len) {
                                    size_t safe = pending_text.size() - max_stop_len + 1;
                                    if (!flush_text(safe)) return false;
                                }
                            }
                            continue;
                        }
                    }

                    // Normal content emission (no tool tag detected)
                    if (stop_sequences.empty()) {
                        // No stop sequences: stream directly (with UTF-8 buffering)
                        utf8_buf += piece;
                        size_t complete = utf8_complete_len(utf8_buf);
                        if (complete > 0) {
                            if (req_logprobs && active_req) {
                                // Logprobs path: fall back to sse_chunk (rare)
                                std::string to_emit = utf8_buf.substr(0, complete);
                                utf8_buf.erase(0, complete);
                                json content_delta = {{"content", to_emit}};
                                json lp_chunk = nullptr;
                                size_t lp_idx = n_output_tokens - 1;
                                if (lp_idx < active_req->output_logprobs.size()) {
                                    const auto& lp = active_req->output_logprobs[lp_idx];
                                    json top_arr = json::array();
                                    for (const auto& t : lp.top) {
                                        top_arr.push_back({
                                            {"token", safe_token_json(t.text)},
                                            {"logprob", t.logprob},
                                            {"bytes", token_bytes_json(t.text)}
                                        });
                                    }
                                    lp_chunk = {{"content", json::array({
                                        {{"token", safe_token_json(lp.text)},
                                         {"logprob", lp.logprob},
                                         {"bytes", token_bytes_json(lp.text)},
                                         {"top_logprobs", top_arr}}
                                    })}};
                                }
                                std::string chunk = sse_chunk(comp_id, created,
                                    state.model_name, content_delta, nullptr, lp_chunk);
                                if (!sink.write(chunk.data(), chunk.size()))
                                    return false;
                            } else {
                                // Fast path: pre-formatted template
                                if (!sse_writer.write_content(
                                        utf8_buf.data(), complete, sink))
                                    return false;
                                utf8_buf.erase(0, complete);
                            }
                        }
                    } else {
                        // Buffer text and check for stop matches
                        pending_text += piece;

                        // Check for complete stop match
                        bool stop_found = false;
                        for (const auto& stop : stop_sequences) {
                            auto pos = pending_text.find(stop);
                            if (pos != std::string::npos) {
                                // Flush text before the stop string
                                if (!flush_text(pos)) return false;
                                stop_found = true;
                                break;
                            }
                        }
                        if (stop_found) {
                            text_stop_matched = true;
                            finish = "stop";
                            break;
                        }

                        // Flush text that can't be part of a partial stop match.
                        // Keep only the last (max_stop_len - 1) chars as potential prefix.
                        if (pending_text.size() > max_stop_len) {
                            size_t safe = pending_text.size() - max_stop_len + 1;
                            if (!flush_text(safe)) return false;
                        }
                    }
                }

                // Flush scan buffer if we never left SCAN phase (model didn't think)
                if (think_phase == ThinkPhase::SCAN && !think_scan_buf.empty()) {
                    utf8_buf += think_scan_buf;
                    think_scan_buf.clear();
                }

                // Flush remaining reasoning buffer (model ended while still thinking)
                if (!reasoning_utf8_buf.empty()) {
                    emit_reasoning(reasoning_utf8_buf);
                    reasoning_utf8_buf.clear();
                }

                // Handle incomplete tool call at end (max_tokens hit while in tag)
                if (tool_phase != ToolPhase::CONTENT && !tool_calls_emitted) {
                    // Partial tool call — emit as content, finish_reason stays "length"
                    std::string leftover;
                    if (!tool_tag_buf.empty()) leftover += tool_tag_buf;
                    if (!tool_body_buf.empty()) leftover += tool_body_buf;
                    if (!leftover.empty()) {
                        utf8_buf += leftover;
                    }
                }

                // Flush any remaining UTF-8 buffer (only if no tool calls were emitted)
                if (!utf8_buf.empty() && !text_stop_matched && !tool_calls_emitted) {
                    sse_writer.write_content(utf8_buf, sink);
                }

                // Flush any remaining buffered text (skip if text-level stop was matched)
                if (!pending_text.empty() && !text_stop_matched && !tool_calls_emitted) {
                    sse_writer.write_content(pending_text, sink);
                }

                if (!finish) {
                    finish = tool_calls_emitted ? "tool_calls" : "length";
                } else if (tool_calls_emitted && strcmp(finish, "stop") == 0) {
                    finish = "tool_calls";
                }

                // Send final chunk with finish_reason
                json empty_delta = json::object();
                std::string final_chunk = sse_chunk(comp_id, created, state.model_name,
                                                    empty_delta, finish);
                sink.write(final_chunk.data(), final_chunk.size());

                // Send usage chunk if requested
                if (include_usage) {
                    json usage = {
                        {"prompt_tokens", n_prompt_tokens},
                        {"completion_tokens", n_output_tokens},
                        {"total_tokens", n_prompt_tokens + n_output_tokens}
                    };
                    if (n_reasoning_tokens > 0) {
                        usage["completion_tokens_details"] = {
                            {"reasoning_tokens", n_reasoning_tokens}
                        };
                    }
                    json usage_obj = {
                        {"id", comp_id},
                        {"object", "chat.completion.chunk"},
                        {"created", created},
                        {"model", state.model_name},
                        {"choices", json::array()},
                        {"usage", usage}
                    };
                    std::string usage_chunk = "data: " + usage_obj.dump() + "\n\n";
                    sink.write(usage_chunk.data(), usage_chunk.size());
                }

                // Send [DONE]
                std::string done = "data: [DONE]\n\n";
                sink.write(done.data(), done.size());
                sink.done();

                // Log request
                auto t_end = std::chrono::high_resolution_clock::now();
                double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
                fprintf(stderr, "[%s] %d prompt + %d completion tokens, %.1f ms\n",
                        comp_id.c_str(), n_prompt_tokens, n_output_tokens, ms);

                return true;
            }
        );
    } else {
        // Non-streaming: decode all tokens, return complete response
        // Save request reference for logprobs access (active_request gets cleared on finish)
        auto active_req = state.ctx->active_request;
        std::vector<int32_t> output_ids;
        const char* finish = nullptr;
        std::string output_text;  // accumulated output for stop matching

        for (int step = -1; step < max_tokens; step++) {
            int32_t token = 0;
            if (step == -1) {
                // First iteration: use the token produced during prefill
                if (prefill_token < 0) continue;
                token = prefill_token;
            } else {
                err = imp_decode_step(state.ctx, &params, &token);
                if (err != IMP_SUCCESS) {
                    finish = "stop";
                    break;
                }
            }

            if (token == state.tok->eos_id()) {
                finish = "stop";
                break;
            }
            if (state.have_template) {
                bool is_stop = false;
                for (int32_t stop_id : state.chat_tpl.stop_token_ids()) {
                    if (token == stop_id) { is_stop = true; break; }
                }
                if (is_stop) {
                    finish = "stop";
                    break;
                }
            }

            output_ids.push_back(token);

            // Check text-level stop sequences
            if (!stop_sequences.empty()) {
                output_text += state.tok->decode_token(token);
                bool stop_found = false;
                for (const auto& stop : stop_sequences) {
                    auto pos = output_text.find(stop);
                    if (pos != std::string::npos) {
                        output_text = output_text.substr(0, pos);
                        stop_found = true;
                        break;
                    }
                }
                if (stop_found) {
                    finish = "stop";
                    break;
                }
            }
        }

        if (!finish) finish = "length";

        int n_output_tokens = static_cast<int>(output_ids.size());
        std::string content = !stop_sequences.empty()
            ? output_text : state.tok->decode(output_ids);

        // Extract reasoning content (DeepSeek format) or strip think blocks
        std::string reasoning_content;
        if (state.is_think_model &&
            state.default_args.reasoning_format == "deepseek") {
            auto [reasoning, cleaned] = extract_reasoning(content);
            reasoning_content = reasoning;
            content = cleaned;
        } else if (state.is_think_model &&
                   state.default_args.reasoning_format != "none") {
            strip_think_block(content);
        }

        // Log request
        auto t_end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        fprintf(stderr, "[%s] %d prompt + %d completion tokens, %.1f ms\n",
                comp_id.c_str(), n_prompt_tokens, n_output_tokens, ms);

        // Build logprobs object if requested
        json logprobs_obj = nullptr;
        if (req_logprobs && active_req) {
            const auto& lp_data = active_req->output_logprobs;
            json content_logprobs = json::array();
            for (size_t idx = 0; idx < lp_data.size() && idx < output_ids.size(); idx++) {
                const auto& lp = lp_data[idx];
                json top_arr = json::array();
                for (const auto& t : lp.top) {
                    top_arr.push_back({
                        {"token", safe_token_json(t.text)},
                        {"logprob", t.logprob},
                        {"bytes", token_bytes_json(t.text)}
                    });
                }
                content_logprobs.push_back({
                    {"token", safe_token_json(lp.text)},
                    {"logprob", lp.logprob},
                    {"bytes", token_bytes_json(lp.text)},
                    {"top_logprobs", top_arr}
                });
            }
            logprobs_obj = {{"content", content_logprobs}};
        }

        // Parse tool calls from model output
        std::vector<ParsedToolCall> tool_calls;
        if (has_tools && strcmp(finish, "length") != 0) {
            auto [pre_content, parsed_calls] = parse_tool_calls(tpl_family, content, state);
            if (!parsed_calls.empty()) {
                tool_calls = std::move(parsed_calls);
                content = pre_content;
                finish = "tool_calls";
            }
        }

        json msg = {{"role", "assistant"}};
        if (!tool_calls.empty()) {
            // content is null when only tool calls (no preceding text)
            msg["content"] = content.empty() ? json(nullptr) : json(content);
            json tc_array = json::array();
            for (const auto& tc : tool_calls) {
                tc_array.push_back({
                    {"id", tc.id},
                    {"type", "function"},
                    {"function", {{"name", tc.name}, {"arguments", tc.arguments}}}
                });
            }
            msg["tool_calls"] = tc_array;
        } else {
            msg["content"] = content;
        }
        if (!reasoning_content.empty()) {
            msg["reasoning_content"] = reasoning_content;
        }

        json choice = {
            {"index", 0},
            {"message", msg},
            {"finish_reason", finish}
        };
        if (!logprobs_obj.is_null()) {
            choice["logprobs"] = logprobs_obj;
        }

        json usage = {
            {"prompt_tokens", n_prompt_tokens},
            {"completion_tokens", n_output_tokens},
            {"total_tokens", n_prompt_tokens + n_output_tokens}
        };
        if (!reasoning_content.empty()) {
            // Estimate reasoning tokens (non-streaming doesn't have exact count)
            auto reasoning_ids = state.tok->encode(reasoning_content);
            int n_reasoning_tokens = static_cast<int>(reasoning_ids.size()) + 2;  // +2 for <think>/<think>
            usage["completion_tokens_details"] = {
                {"reasoning_tokens", n_reasoning_tokens}
            };
        }

        json response = {
            {"id", comp_id},
            {"object", "chat.completion"},
            {"created", created},
            {"model", state.model_name},
            {"choices", json::array({choice})},
            {"usage", usage}
        };

        res.set_content(response.dump(), "application/json");
    }
}

// Build ImpConfig from default args + optional JSON overrides
static ImpConfig build_config(const ServerArgs& args, const json& overrides = json::object()) {
    ImpConfig config = imp_config_default();
    config.device_id = args.device;
    config.max_batch_size = 1;
    config.max_seq_len = overrides.value("max_seq_len", 4096);
    config.gpu_layers = args.gpu_layers;
    if (args.ssm_fp16) config.ssm_state_dtype = IMP_DTYPE_FP16;
    if (args.no_cuda_graphs) config.enable_cuda_graphs = 0;

    // KV cache dtype: check overrides first, then startup args
    bool kv_fp8 = overrides.value("kv_fp8", args.kv_fp8);
    bool kv_int8 = overrides.value("kv_int8", args.kv_int8);
    if (kv_fp8) config.kv_cache_dtype = IMP_DTYPE_FP8_E4M3;
    if (kv_int8) config.kv_cache_dtype = IMP_DTYPE_INT8;

    int chunk = overrides.value("prefill_chunk_size", args.prefill_chunk_size);
    if (chunk > 0) config.prefill_chunk_size = chunk;

    config.use_nvfp4_decode = overrides.value("decode_nvfp4", args.decode_nvfp4);

    if (!args.mmproj_path.empty())
        config.mmproj_path = args.mmproj_path.c_str();

    return config;
}

// Load a model into ServerState. Caller must hold state.mtx.
// Returns error message on failure, empty string on success.
static std::string load_model_into_state(ServerState& state, const std::string& path,
                                         const json& config_overrides) {
    // Free existing model/context
    if (state.ctx) { imp_context_free(state.ctx); state.ctx = nullptr; }
    if (state.model) { imp_model_free(state.model); state.model = nullptr; }
    state.tok = nullptr;
    state.have_template = false;
    state.model_name.clear();

    // Load model
    ImpError err = imp_model_load(path.c_str(), IMP_FORMAT_GGUF, &state.model);
    if (err != IMP_SUCCESS) {
        std::string msg = std::string("Failed to load model: ") + imp_error_string(err);
        state.model = nullptr;
        return msg;
    }

    // Create context
    ImpConfig config = build_config(state.default_args, config_overrides);
    err = imp_context_create(state.model, &config, &state.ctx);
    if (err != IMP_SUCCESS) {
        std::string msg = std::string("Failed to create context: ") + imp_error_string(err);
        imp_model_free(state.model);
        state.model = nullptr;
        return msg;
    }

    // Extract model name from path
    size_t slash = path.find_last_of('/');
    state.model_name = (slash != std::string::npos) ? path.substr(slash + 1) : path;

    // Set up tokenizer and chat template
    state.tok = state.model->model->tokenizer();
    const imp::ChatTemplate& engine_tpl = state.ctx->engine->chat_template();

    std::string chat_tpl_name = config_overrides.value("chat_template",
                                                        state.default_args.chat_template);
    if (chat_tpl_name == "none") {
        // No template
    } else if (chat_tpl_name != "auto") {
        auto family = imp::ChatTemplate::parse_family(chat_tpl_name);
        if (family != imp::ChatTemplateFamily::RAW) {
            state.have_template = state.chat_tpl.init(family, *state.tok);
        }
    } else {
        if (!engine_tpl.is_raw()) {
            state.chat_tpl = engine_tpl;
            state.have_template = true;
        }
    }

    // Store max sequence length for token clamping
    state.max_seq_len = imp_model_max_seq_len(state.model);
    if (state.max_seq_len <= 0) state.max_seq_len = static_cast<int>(config.max_seq_len);

    // Detect thinking model (DeepSeek R1 etc.) by checking for <think> token in vocab
    state.think_start_id = state.tok->find_token("<think>");
    state.think_end_id = state.tok->find_token("</think>");
    state.is_think_model = (state.think_start_id >= 0);
    if (state.is_think_model) {
        printf("Reasoning model: <think>=%d, </think>=%d\n",
               state.think_start_id, state.think_end_id);
    }

    if (state.have_template) {
        printf("Chat template: %s\n",
               imp::chat_template_family_name(state.chat_tpl.family()));
    } else {
        printf("No chat template (raw mode)\n");
    }

    return "";
}

// Build a single SSE data line for text completions
static std::string sse_completion_chunk(const std::string& id, int64_t created,
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

static void handle_completions(const httplib::Request& req, httplib::Response& res,
                                ServerState& state) {
    // Parse request body
    json body;
    try {
        body = json::parse(req.body);
    } catch (const json::parse_error& e) {
        res.status = 400;
        json err = {{"error", {{"message", std::string("Invalid JSON: ") + e.what()},
                                {"type", "invalid_request_error"}}}};
        res.set_content(err.dump(), "application/json");
        return;
    }

    // Extract prompt
    std::string prompt = body.value("prompt", "");
    if (prompt.empty()) {
        res.status = 400;
        json err = {{"error", {{"message", "\"prompt\" is required and must not be empty"},
                                {"type", "invalid_request_error"}}}};
        res.set_content(err.dump(), "application/json");
        return;
    }

    // Extract parameters
    float temperature = body.value("temperature", 0.7f);
    float top_p = body.value("top_p", 0.9f);
    int top_k = body.value("top_k", 40);
    int max_tokens = body.value("max_tokens", state.default_max_tokens);
    int seed = body.value("seed", -1);
    bool stream = body.value("stream", false);
    bool echo = body.value("echo", false);
    float min_p = body.value("min_p", 0.0f);
    float typical_p = body.value("typical_p", 1.0f);
    float repetition_penalty = body.value("repetition_penalty", 1.0f);
    float frequency_penalty = body.value("frequency_penalty", 0.0f);
    float presence_penalty = body.value("presence_penalty", 0.0f);
    float dry_multiplier = body.value("dry_multiplier", 0.0f);
    float dry_base = body.value("dry_base", 1.75f);
    int dry_allowed_length = body.value("dry_allowed_length", 2);
    int dry_penalty_last_n = body.value("dry_penalty_last_n", 0);
    int mirostat = body.value("mirostat", 0);
    float mirostat_tau = body.value("mirostat_tau", 5.0f);
    float mirostat_eta = body.value("mirostat_eta", 0.1f);

    if (!body.contains("repetition_penalty") && !body.contains("frequency_penalty")
        && !body.contains("presence_penalty")) {
        frequency_penalty = 0.3f;
    }

    bool req_logprobs = body.value("logprobs", false);
    int top_logprobs = body.value("top_logprobs", 0);
    if (top_logprobs < 0) top_logprobs = 0;
    if (top_logprobs > 20) top_logprobs = 20;

    // Parse stop sequences
    std::vector<std::string> stop_sequences;
    if (body.contains("stop") && !body["stop"].is_null()) {
        if (body["stop"].is_string()) {
            stop_sequences.push_back(body["stop"].get<std::string>());
        } else if (body["stop"].is_array()) {
            for (const auto& s : body["stop"]) {
                if (s.is_string()) {
                    stop_sequences.push_back(s.get<std::string>());
                    if (stop_sequences.size() >= 4) break;
                }
            }
        }
    }
    size_t max_stop_len = 0;
    for (const auto& s : stop_sequences) max_stop_len = std::max(max_stop_len, s.size());

    // Parse stream_options for include_usage
    bool include_usage = false;
    if (body.contains("stream_options") && body["stream_options"].is_object()) {
        include_usage = body["stream_options"].value("include_usage", false);
    }

    // Acquire inference lock
    std::unique_lock<std::timed_mutex> lock(state.mtx, std::chrono::minutes(5));
    if (!lock.owns_lock()) {
        res.status = 503;
        json err = {{"error", {{"message", "Server is busy processing another request. Please retry."},
                                {"type", "server_error"}}}};
        res.set_content(err.dump(), "application/json");
        return;
    }

    // Auto-load model if request specifies a different one
    std::string requested_model = body.value("model", "");
    if (!requested_model.empty() && requested_model != state.model_name) {
        std::string path = find_model_path(state, requested_model);
        if (!path.empty()) {
            printf("Auto-loading model: %s\n", requested_model.c_str());
            fflush(stdout);
            std::string error = load_model_into_state(state, path);
            if (!error.empty()) {
                res.status = 500;
                json err = {{"error", {{"message", error}, {"type", "server_error"}}}};
                res.set_content(err.dump(), "application/json");
                return;
            }
            printf("Model loaded: %s\n", state.model_name.c_str());
            fflush(stdout);
        }
    }

    if (!state.model_loaded()) {
        res.status = 503;
        json err = {{"error", {{"message", "No model loaded. Use POST /v1/models to load one."},
                                {"type", "server_error"}}}};
        res.set_content(err.dump(), "application/json");
        return;
    }

    // Tokenize raw prompt (no chat template)
    std::vector<int32_t> tokens = state.tok->encode(prompt);
    int n_prompt_tokens = static_cast<int>(tokens.size());

    if (n_prompt_tokens >= state.max_seq_len) {
        res.status = 400;
        json error = {{"error", {{"message", "Prompt exceeds context window (" +
                                              std::to_string(n_prompt_tokens) + " tokens >= " +
                                              std::to_string(state.max_seq_len) + " max)"},
                                  {"type", "invalid_request_error"}}}};
        res.set_content(error.dump(), "application/json");
        return;
    }

    int remaining = state.max_seq_len - n_prompt_tokens;
    if (max_tokens > remaining) max_tokens = remaining;

    auto t_start = std::chrono::high_resolution_clock::now();

    // Reset and prefill
    ImpError err = imp_context_reset(state.ctx);
    if (err != IMP_SUCCESS) {
        res.status = 500;
        json error = {{"error", {{"message", std::string("Context reset failed: ") + imp_error_string(err)},
                                  {"type", "server_error"}}}};
        res.set_content(error.dump(), "application/json");
        return;
    }

    err = imp_prefill(state.ctx, tokens.data(), n_prompt_tokens);
    if (err != IMP_SUCCESS) {
        res.status = 500;
        json error = {{"error", {{"message", std::string("Prefill failed: ") + imp_error_string(err)},
                                  {"type", "server_error"}}}};
        res.set_content(error.dump(), "application/json");
        return;
    }

    // Capture first token produced during prefill
    int32_t prefill_token = -1;
    if (state.ctx->active_request && !state.ctx->active_request->output_tokens.empty()) {
        prefill_token = state.ctx->active_request->output_tokens.back();
    }

    ImpGenerateParams params = imp_generate_params_default();
    params.temperature = temperature;
    params.top_p = top_p;
    params.top_k = top_k;
    params.max_tokens = max_tokens;
    params.seed = seed;
    params.min_p = min_p;
    params.typical_p = typical_p;
    params.repetition_penalty = repetition_penalty;
    params.frequency_penalty = frequency_penalty;
    params.presence_penalty = presence_penalty;
    params.dry_multiplier = dry_multiplier;
    params.dry_base = dry_base;
    params.dry_allowed_length = dry_allowed_length;
    params.dry_penalty_last_n = dry_penalty_last_n;
    params.mirostat = mirostat;
    params.mirostat_tau = mirostat_tau;
    params.mirostat_eta = mirostat_eta;
    params.logprobs = req_logprobs ? 1 : 0;
    params.top_logprobs = top_logprobs;

    std::string comp_id = make_completion_id(state);
    int64_t created = unix_timestamp();

    if (stream) {
        res.set_header("Cache-Control", "no-cache");
        res.set_header("Connection", "keep-alive");

        res.set_chunked_content_provider(
            "text/event-stream",
            [&state, params, comp_id, created, max_tokens, n_prompt_tokens, t_start,
             stop_sequences, max_stop_len, echo, prompt, include_usage, prefill_token](
                size_t /*offset*/, httplib::DataSink& sink) -> bool {

                int n_output_tokens = 0;
                const char* finish = nullptr;

                // Echo prompt as first chunk if requested
                if (echo && !prompt.empty()) {
                    std::string chunk = sse_completion_chunk(comp_id, created,
                                                             state.model_name, prompt, nullptr);
                    sink.write(chunk.data(), chunk.size());
                }

                std::string utf8_buf;
                std::string pending_text;
                bool text_stop_matched = false;

                // Strip <think> blocks for completions (no reasoning_content field)
                bool think_strip = (state.is_think_model &&
                                    state.default_args.reasoning_format != "none");
                bool think_confirmed = think_strip;
                std::string think_buf;
                int think_tokens = 0;
                const int kThinkScanLimit = 8;

                auto flush_text = [&](size_t up_to) {
                    if (up_to == 0) return true;
                    std::string to_send = pending_text.substr(0, up_to);
                    pending_text = pending_text.substr(up_to);
                    std::string sse = sse_completion_chunk(comp_id, created,
                                                           state.model_name, to_send, nullptr);
                    return sink.write(sse.data(), sse.size());
                };

                for (int step = -1; step < max_tokens; step++) {
                    int32_t token = 0;
                    if (step == -1) {
                        if (prefill_token < 0) continue;
                        token = prefill_token;
                    } else {
                        ImpError err = imp_decode_step(state.ctx, &params, &token);
                        if (err != IMP_SUCCESS) {
                            finish = "stop";
                            break;
                        }
                    }

                    if (token == state.tok->eos_id()) {
                        finish = "stop";
                        break;
                    }

                    n_output_tokens++;
                    std::string piece = state.tok->decode_token(token);

                    // Strip <think>...</think> block for text completions
                    if (think_strip) {
                        think_buf += piece;
                        think_tokens++;

                        if (!think_confirmed) {
                            if (think_buf.find("<think>") != std::string::npos)
                                think_confirmed = true;
                            else if (think_tokens == 1 && piece.empty())
                                think_confirmed = true;
                        }

                        auto end_pos = think_buf.find("</think>");
                        if (end_pos != std::string::npos) {
                            think_strip = false;
                            std::string after = think_buf.substr(end_pos + 8);
                            think_buf.clear();
                            auto start = after.find_first_not_of("\n\r\t ");
                            piece = (start != std::string::npos) ? after.substr(start) : "";
                            if (piece.empty()) continue;
                        } else if (think_confirmed) {
                            continue;
                        } else if (think_tokens < kThinkScanLimit) {
                            continue;
                        } else {
                            think_strip = false;
                            piece = think_buf;
                            think_buf.clear();
                        }
                    }

                    if (stop_sequences.empty()) {
                        utf8_buf += piece;
                        size_t complete = utf8_complete_len(utf8_buf);
                        if (complete > 0) {
                            std::string to_emit = utf8_buf.substr(0, complete);
                            utf8_buf = utf8_buf.substr(complete);
                            std::string chunk = sse_completion_chunk(comp_id, created,
                                                                      state.model_name,
                                                                      to_emit, nullptr);
                            if (!sink.write(chunk.data(), chunk.size())) return false;
                        }
                    } else {
                        pending_text += piece;
                        bool stop_found = false;
                        for (const auto& stop : stop_sequences) {
                            auto pos = pending_text.find(stop);
                            if (pos != std::string::npos) {
                                if (!flush_text(pos)) return false;
                                stop_found = true;
                                break;
                            }
                        }
                        if (stop_found) {
                            text_stop_matched = true;
                            finish = "stop";
                            break;
                        }
                        if (pending_text.size() > max_stop_len) {
                            size_t safe = pending_text.size() - max_stop_len + 1;
                            if (!flush_text(safe)) return false;
                        }
                    }
                }

                // Flush think buffer: strip think blocks and emit remaining content
                if (!think_buf.empty()) {
                    strip_think_block(think_buf);
                    if (!think_buf.empty()) {
                        utf8_buf += think_buf;
                    }
                    think_buf.clear();
                }

                // Flush remaining buffers
                if (!utf8_buf.empty() && !text_stop_matched) {
                    std::string sse = sse_completion_chunk(comp_id, created,
                                                           state.model_name, utf8_buf, nullptr);
                    sink.write(sse.data(), sse.size());
                }
                if (!pending_text.empty() && !text_stop_matched) {
                    std::string sse = sse_completion_chunk(comp_id, created,
                                                           state.model_name, pending_text, nullptr);
                    sink.write(sse.data(), sse.size());
                }

                if (!finish) finish = "length";

                // Final chunk with finish_reason
                std::string final_chunk = sse_completion_chunk(comp_id, created,
                                                                state.model_name, "", finish);
                sink.write(final_chunk.data(), final_chunk.size());

                // Usage chunk if requested
                if (include_usage) {
                    json usage_obj = {
                        {"id", comp_id},
                        {"object", "text_completion"},
                        {"created", created},
                        {"model", state.model_name},
                        {"choices", json::array()},
                        {"usage", {
                            {"prompt_tokens", n_prompt_tokens},
                            {"completion_tokens", n_output_tokens},
                            {"total_tokens", n_prompt_tokens + n_output_tokens}
                        }}
                    };
                    std::string usage_chunk = "data: " + usage_obj.dump() + "\n\n";
                    sink.write(usage_chunk.data(), usage_chunk.size());
                }

                std::string done = "data: [DONE]\n\n";
                sink.write(done.data(), done.size());
                sink.done();

                auto t_end = std::chrono::high_resolution_clock::now();
                double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
                fprintf(stderr, "[%s] %d prompt + %d completion tokens, %.1f ms\n",
                        comp_id.c_str(), n_prompt_tokens, n_output_tokens, ms);

                return true;
            }
        );
    } else {
        // Non-streaming
        auto active_req = state.ctx->active_request;
        std::vector<int32_t> output_ids;
        const char* finish = nullptr;
        std::string output_text;

        for (int step = -1; step < max_tokens; step++) {
            int32_t token = 0;
            if (step == -1) {
                if (prefill_token < 0) continue;
                token = prefill_token;
            } else {
                err = imp_decode_step(state.ctx, &params, &token);
                if (err != IMP_SUCCESS) {
                    finish = "stop";
                    break;
                }
            }

            if (token == state.tok->eos_id()) {
                finish = "stop";
                break;
            }

            output_ids.push_back(token);

            if (!stop_sequences.empty()) {
                output_text += state.tok->decode_token(token);
                bool stop_found = false;
                for (const auto& stop : stop_sequences) {
                    auto pos = output_text.find(stop);
                    if (pos != std::string::npos) {
                        output_text = output_text.substr(0, pos);
                        stop_found = true;
                        break;
                    }
                }
                if (stop_found) {
                    finish = "stop";
                    break;
                }
            }
        }

        if (!finish) finish = "length";

        int n_output_tokens = static_cast<int>(output_ids.size());
        std::string text = !stop_sequences.empty()
            ? output_text : state.tok->decode(output_ids);

        // Strip <think>...</think> for text completions (no reasoning_content field)
        if (state.is_think_model && state.default_args.reasoning_format != "none") {
            strip_think_block(text);
        }

        // Prepend prompt if echo requested
        if (echo) text = prompt + text;

        auto t_end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        fprintf(stderr, "[%s] %d prompt + %d completion tokens, %.1f ms\n",
                comp_id.c_str(), n_prompt_tokens, n_output_tokens, ms);

        // Build logprobs if requested
        json logprobs_obj = nullptr;
        if (req_logprobs && active_req) {
            const auto& lp_data = active_req->output_logprobs;
            json content_logprobs = json::array();
            for (size_t idx = 0; idx < lp_data.size() && idx < output_ids.size(); idx++) {
                const auto& lp = lp_data[idx];
                json top_arr = json::array();
                for (const auto& t : lp.top) {
                    top_arr.push_back({
                        {"token", safe_token_json(t.text)},
                        {"logprob", t.logprob},
                        {"bytes", token_bytes_json(t.text)}
                    });
                }
                content_logprobs.push_back({
                    {"token", safe_token_json(lp.text)},
                    {"logprob", lp.logprob},
                    {"bytes", token_bytes_json(lp.text)},
                    {"top_logprobs", top_arr}
                });
            }
            logprobs_obj = {{"content", content_logprobs}};
        }

        json choice = {
            {"index", 0},
            {"text", text},
            {"finish_reason", finish}
        };
        if (!logprobs_obj.is_null()) {
            choice["logprobs"] = logprobs_obj;
        }

        json response = {
            {"id", comp_id},
            {"object", "text_completion"},
            {"created", created},
            {"model", state.model_name},
            {"choices", json::array({choice})},
            {"usage", {
                {"prompt_tokens", n_prompt_tokens},
                {"completion_tokens", n_output_tokens},
                {"total_tokens", n_prompt_tokens + n_output_tokens}
            }}
        };

        res.set_content(response.dump(), "application/json");
    }
}

static void handle_tokenize(const httplib::Request& req, httplib::Response& res,
                            ServerState& state) {
    json body;
    try {
        body = json::parse(req.body);
    } catch (const json::parse_error& e) {
        res.status = 400;
        json err = {{"error", {{"message", std::string("Invalid JSON: ") + e.what()},
                                {"type", "invalid_request_error"}}}};
        res.set_content(err.dump(), "application/json");
        return;
    }

    std::string content = body.value("content", "");
    if (content.empty()) {
        res.status = 400;
        json err = {{"error", {{"message", "\"content\" is required"},
                                {"type", "invalid_request_error"}}}};
        res.set_content(err.dump(), "application/json");
        return;
    }

    if (!state.model) {
        res.status = 503;
        json err = {{"error", {{"message", "No model loaded"},
                                {"type", "server_error"}}}};
        res.set_content(err.dump(), "application/json");
        return;
    }

    std::vector<int32_t> tokens(32768);
    int n_tokens = 0;
    ImpError err = imp_tokenize(state.model, content.c_str(), tokens.data(), &n_tokens, 32768);
    if (err != IMP_SUCCESS) {
        res.status = 500;
        json error = {{"error", {{"message", std::string("Tokenize failed: ") + imp_error_string(err)},
                                  {"type", "server_error"}}}};
        res.set_content(error.dump(), "application/json");
        return;
    }

    tokens.resize(n_tokens);
    json response = {{"tokens", tokens}};
    res.set_content(response.dump(), "application/json");
}

static void handle_detokenize(const httplib::Request& req, httplib::Response& res,
                               ServerState& state) {
    json body;
    try {
        body = json::parse(req.body);
    } catch (const json::parse_error& e) {
        res.status = 400;
        json err = {{"error", {{"message", std::string("Invalid JSON: ") + e.what()},
                                {"type", "invalid_request_error"}}}};
        res.set_content(err.dump(), "application/json");
        return;
    }

    if (!body.contains("tokens") || !body["tokens"].is_array()) {
        res.status = 400;
        json err = {{"error", {{"message", "\"tokens\" array is required"},
                                {"type", "invalid_request_error"}}}};
        res.set_content(err.dump(), "application/json");
        return;
    }

    if (!state.model) {
        res.status = 503;
        json err = {{"error", {{"message", "No model loaded"},
                                {"type", "server_error"}}}};
        res.set_content(err.dump(), "application/json");
        return;
    }

    std::vector<int32_t> tokens = body["tokens"].get<std::vector<int32_t>>();
    std::vector<char> buf(tokens.size() * 32 + 256);
    ImpError err = imp_detokenize(state.model, tokens.data(), static_cast<int>(tokens.size()),
                                   buf.data(), buf.size());
    if (err != IMP_SUCCESS) {
        res.status = 500;
        json error = {{"error", {{"message", std::string("Detokenize failed: ") + imp_error_string(err)},
                                  {"type", "server_error"}}}};
        res.set_content(error.dump(), "application/json");
        return;
    }

    json response = {{"content", std::string(buf.data())}};
    res.set_content(response.dump(), "application/json");
}

static void handle_load_model(const httplib::Request& req, httplib::Response& res,
                              ServerState& state) {
    json body;
    try {
        body = json::parse(req.body);
    } catch (const json::parse_error& e) {
        res.status = 400;
        json err = {{"error", {{"message", std::string("Invalid JSON: ") + e.what()},
                                {"type", "invalid_request_error"}}}};
        res.set_content(err.dump(), "application/json");
        return;
    }

    std::string path = body.value("path", "");
    if (path.empty()) {
        res.status = 400;
        json err = {{"error", {{"message", "\"path\" is required"},
                                {"type", "invalid_request_error"}}}};
        res.set_content(err.dump(), "application/json");
        return;
    }

    json config_overrides = body.value("config", json::object());

    // Block inference during model swap
    std::lock_guard<std::timed_mutex> lock(state.mtx);

    printf("Loading model: %s\n", path.c_str());
    fflush(stdout);

    std::string error = load_model_into_state(state, path, config_overrides);
    if (!error.empty()) {
        res.status = 500;
        json err = {{"error", {{"message", error}, {"type", "server_error"}}}};
        res.set_content(err.dump(), "application/json");
        return;
    }

    printf("Model loaded: %s\n", state.model_name.c_str());
    fflush(stdout);

    json response = {
        {"status", "loaded"},
        {"model", state.model_name}
    };
    res.set_content(response.dump(), "application/json");
}

static void handle_unload_model(const httplib::Request& /*req*/, httplib::Response& res,
                                ServerState& state) {
    std::lock_guard<std::timed_mutex> lock(state.mtx);

    if (state.ctx) { imp_context_free(state.ctx); state.ctx = nullptr; }
    if (state.model) { imp_model_free(state.model); state.model = nullptr; }
    state.tok = nullptr;
    state.have_template = false;
    state.model_name.clear();
    state.max_seq_len = 0;

    printf("Model unloaded\n");
    fflush(stdout);

    json response = {{"status", "unloaded"}};
    res.set_content(response.dump(), "application/json");
}

int main(int argc, char** argv) {
    ServerArgs args = parse_server_args(argc, argv);

    printf("IMP Server %s\n", imp_version());

    ServerState state;
    state.default_max_tokens = args.max_tokens;
    state.default_args = args;

    // Set models directory (explicit flag, or infer from model path, or /models default)
    if (!args.models_dir.empty()) {
        state.models_dir = args.models_dir;
    } else if (!args.model_path.empty()) {
        auto parent = std::filesystem::path(args.model_path).parent_path().string();
        if (!parent.empty()) state.models_dir = parent;
    } else if (std::filesystem::is_directory("/models")) {
        state.models_dir = "/models";
    }

    if (!state.models_dir.empty()) {
        printf("Models directory: %s\n", state.models_dir.c_str());
    }

    // Load model at startup if provided
    if (!args.model_path.empty()) {
        printf("Loading model: %s\n", args.model_path.c_str());
        std::string error = load_model_into_state(state, args.model_path);
        if (!error.empty()) {
            fprintf(stderr, "%s\n", error.c_str());
            return 1;
        }
    } else {
        printf("No model specified — server will wait for POST /v1/models\n");
    }

    // Set up HTTP server
    httplib::Server svr;

    // Limit request body size to 100 MiB (prevents DoS via large base64 images)
    svr.set_payload_max_length(100 * 1024 * 1024);

    // Store API key in state
    state.api_key = args.api_key;

    // CORS headers + API key auth on every response
    svr.set_pre_routing_handler([&state](const httplib::Request& req, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization");

        // Skip auth for health checks and CORS preflight
        if (req.path == "/health" || req.method == "OPTIONS")
            return httplib::Server::HandlerResponse::Unhandled;

        // Enforce API key if configured
        if (!state.api_key.empty()) {
            std::string auth = req.get_header_value("Authorization");
            std::string expected = "Bearer " + state.api_key;
            if (auth != expected) {
                res.status = 401;
                json err = {{"error", {{"message", "Invalid API key"},
                                        {"type", "invalid_request_error"}}}};
                res.set_content(err.dump(), "application/json");
                return httplib::Server::HandlerResponse::Handled;
            }
        }

        return httplib::Server::HandlerResponse::Unhandled;
    });

    // CORS preflight
    svr.Options(R"(.*)", [](const httplib::Request&, httplib::Response& res) {
        res.status = 204;
    });

    svr.Get("/health", [&state](const httplib::Request& req, httplib::Response& res) {
        handle_health(req, res, state);
    });

    svr.Get("/v1/models", [&state](const httplib::Request& req, httplib::Response& res) {
        handle_models(req, res, state);
    });

    svr.Post("/v1/models", [&state](const httplib::Request& req, httplib::Response& res) {
        handle_load_model(req, res, state);
    });

    svr.Delete("/v1/models", [&state](const httplib::Request& req, httplib::Response& res) {
        handle_unload_model(req, res, state);
    });

    svr.Post("/v1/chat/completions",
        [&state](const httplib::Request& req, httplib::Response& res) {
            handle_chat_completions(req, res, state);
        });

    svr.Post("/v1/completions",
        [&state](const httplib::Request& req, httplib::Response& res) {
            handle_completions(req, res, state);
        });

    svr.Post("/tokenize",
        [&state](const httplib::Request& req, httplib::Response& res) {
            handle_tokenize(req, res, state);
        });

    svr.Post("/detokenize",
        [&state](const httplib::Request& req, httplib::Response& res) {
            handle_detokenize(req, res, state);
        });

    // Graceful shutdown on SIGINT/SIGTERM
    g_server.store(&svr, std::memory_order_relaxed);
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    if (!state.api_key.empty()) printf("API key: enabled\n");

    printf("Server listening on http://%s:%d\n", args.host.c_str(), args.port);
    printf("Endpoints:\n");
    printf("  GET    /health\n");
    printf("  GET    /v1/models\n");
    printf("  POST   /v1/models          Load/swap model\n");
    printf("  DELETE /v1/models          Unload model\n");
    printf("  POST   /v1/chat/completions\n");
    printf("  POST   /v1/completions\n");
    printf("  POST   /tokenize\n");
    printf("  POST   /detokenize\n");
    fflush(stdout);

    if (!svr.listen(args.host, args.port)) {
        // listen() returns false on stop() or bind failure
        if (!g_server.load(std::memory_order_relaxed)) {
            // Server was nulled by signal — clean shutdown
        } else {
            fprintf(stderr, "Failed to start server on %s:%d\n",
                    args.host.c_str(), args.port);
        }
    }

    g_server.store(nullptr, std::memory_order_relaxed);
    imp_context_free(state.ctx);
    imp_model_free(state.model);
    return 0;
}
