// AUTO-SPLIT from handlers.cpp (verbatim move; see handlers_internal.h).
// Anthropic /v1/messages streaming machinery: the AnthropicSSE event writer
// and run_anthropic_stream_ (drives the real token loop, emits native
// Anthropic SSE events). The handle_messages endpoint lives in handlers_chat.cpp.

#include "handlers.h"
#include "handlers_internal.h"
#include "utils.h"
#include "tool_call.h"
#include "anthropic.h"
#include "stream_pipeline.h"

#include "api/imp_internal.h"
#include "vision/image_processor.h"
#include "runtime/request.h"
#include "memory/kv_cache.h"
#include "model/hf_hub.h"
#include "runtime/config.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <functional>
#include <vector>

#include <cuda_runtime.h>

// ===========================================================================
// Anthropic /v1/messages — native SSE streaming
// ===========================================================================
//
// Non-streaming reuses the OpenAI code path (Anthropic→OpenAI body in,
// OpenAI→Anthropic response out). Streaming drives the same real per-token
// batching-engine loop the OpenAI streaming path uses (pop_token), but emits
// native Anthropic SSE events incrementally so TTFT == real first-token
// latency rather than full-generation latency.
// ---------------------------------------------------------------------------

namespace {

// Anthropic SSE event writer. Emits "event: <name>\ndata: <json>\n\n".
struct AnthropicSSE {
    httplib::DataSink& sink;
    bool emit(const char* event_name, const json& payload) const {
        std::string buf = "event: ";
        buf += event_name;
        buf += "\ndata: ";
        buf += dump_safe(payload);
        buf += "\n\n";
        return sink.write(buf.data(), buf.size());
    }
};

// Tracks which content block (if any) is currently open in the stream so we
// can close it before opening one of a different kind. Anthropic requires a
// content_block_start before deltas and a content_block_stop after.
enum class AnthBlock { NONE, THINKING, TEXT, TOOL_USE };

}  // anonymous namespace

// Drives the real token loop and emits native Anthropic SSE events. Mirrors
// the token-handling of run_chat_stream_ (reasoning extraction, channel
// filter, tool-call tag state machine) but maps it onto Anthropic blocks:
//   reasoning -> thinking block (thinking_delta)
//   content   -> text block (text_delta)
//   tool call -> tool_use block (input_json_delta, chunked)
bool run_anthropic_stream_(httplib::DataSink& sink, ChatRequestContext& ctx, ServerState& state,
                                  const std::shared_ptr<ServerRequest>& server_req,
                                  const std::string& anth_model, const std::string& msg_id) {
    AnthropicSSE out{sink};

    const auto& stop_sequences = ctx.params.stop_sequences;
    // Derive from the FINAL stop list (server-injected stops update it late) —
    // see the matching comment in the chat streaming path.
    size_t max_stop_len = 0;
    for (const auto& s : stop_sequences)
        max_stop_len = std::max(max_stop_len, s.size());
    bool enable_thinking = ctx.snap.enable_thinking;
    bool has_tools = ctx.params.has_tools;
    auto tpl_family = ctx.snap.tpl_family;
    float think_budget = ctx.params.think_budget;
    auto snap_tok = ctx.snap.tok;
    bool snap_have_template = ctx.snap.have_template;
    bool snap_is_think_model = ctx.snap.is_think_model;
    int snap_think_start_id = ctx.snap.think_start_id;
    int snap_think_end_id = ctx.snap.think_end_id;
    int snap_channel_open_id = ctx.snap.channel_open_id;
    int snap_channel_close_id = ctx.snap.channel_close_id;
    int snap_channel_newline_id = ctx.snap.channel_newline_id;
    const auto& snap_stop_token_ids = ctx.snap.stop_token_ids;
    auto active_req = server_req->request;
    auto t_start = ctx.t_start;
    int n_prompt_tokens = ctx.snap.n_prompt_tokens;

    // ---- message_start ----------------------------------------------------
    {
        json msg = {
            {"id", msg_id},
            {"type", "message"},
            {"role", "assistant"},
            {"content", json::array()},
            {"model", anth_model},
            {"stop_reason", nullptr},
            {"stop_sequence", nullptr},
            {"usage", {{"input_tokens", n_prompt_tokens}, {"output_tokens", 0}}},
        };
        if (!out.emit("message_start", json{{"type", "message_start"}, {"message", std::move(msg)}}))
            return false;
    }

    // ---- ping (initial keepalive) -----------------------------------------
    // Anthropic streams emit periodic `ping` events; sending one immediately
    // signals liveness before the first token (TTFT can be >1s under load /
    // long prefills), and the loop below re-pings during idle gaps so clients
    // and intermediary proxies don't time out the connection.
    if (!out.emit("ping", json{{"type", "ping"}}))
        return false;
    auto last_ping = std::chrono::steady_clock::now();

    int block_index = -1;
    AnthBlock open_block = AnthBlock::NONE;

    auto stop_block = [&]() -> bool {
        if (open_block == AnthBlock::NONE)
            return true;
        bool ok = out.emit("content_block_stop",
                           json{{"type", "content_block_stop"}, {"index", block_index}});
        open_block = AnthBlock::NONE;
        return ok;
    };
    auto start_text_block = [&]() -> bool {
        if (open_block == AnthBlock::TEXT)
            return true;
        if (!stop_block())
            return false;
        ++block_index;
        open_block = AnthBlock::TEXT;
        return out.emit("content_block_start",
                        json{{"type", "content_block_start"},
                             {"index", block_index},
                             {"content_block", {{"type", "text"}, {"text", ""}}}});
    };
    auto start_thinking_block = [&]() -> bool {
        if (open_block == AnthBlock::THINKING)
            return true;
        if (!stop_block())
            return false;
        ++block_index;
        open_block = AnthBlock::THINKING;
        return out.emit("content_block_start",
                        json{{"type", "content_block_start"},
                             {"index", block_index},
                             {"content_block", {{"type", "thinking"}, {"thinking", ""}}}});
    };
    auto emit_text = [&](const std::string& text) -> bool {
        if (text.empty())
            return true;
        if (!start_text_block())
            return false;
        return out.emit("content_block_delta",
                        json{{"type", "content_block_delta"},
                             {"index", block_index},
                             {"delta", {{"type", "text_delta"}, {"text", text}}}});
    };
    auto emit_thinking = [&](const std::string& text) -> bool {
        if (text.empty())
            return true;
        if (!start_thinking_block())
            return false;
        return out.emit("content_block_delta",
                        json{{"type", "content_block_delta"},
                             {"index", block_index},
                             {"delta", {{"type", "thinking_delta"}, {"thinking", text}}}});
    };

    // gpt-oss Harmony streaming filter (analysis/commentary -> thinking block,
    // final -> text block); markers arrive as atomic special-token pieces. See
    // the matching filter in run_chat_stream_ (#760).
    const bool harmony = (ctx.snap.tpl_family == imp::ChatTemplateFamily::HARMONY);
    std::string hm_channel, hm_name, hm_buf;
    bool hm_in_msg = false, hm_reading_name = false;
    auto hm_flush = [&](bool force) -> bool {
        size_t complete = force ? hm_buf.size() : utf8_complete_len(hm_buf);
        if (complete == 0)
            return true;
        std::string chunk = hm_buf.substr(0, complete);
        hm_buf.erase(0, complete);
        if (hm_channel == "analysis" || hm_channel == "commentary")
            return emit_thinking(chunk);
        return emit_text(chunk);
    };
    // Open a tool_use block and stream its arguments as chunked
    // input_json_delta events (Task 6: incremental arg deltas).
    auto emit_tool_use = [&](const ParsedToolCall& tc) -> bool {
        if (!stop_block())
            return false;
        ++block_index;
        open_block = AnthBlock::TOOL_USE;
        namespace anth = imp_server::anthropic;
        if (!out.emit("content_block_start",
                      json{{"type", "content_block_start"},
                           {"index", block_index},
                           {"content_block",
                            {{"type", "tool_use"},
                             {"id", anth::tool_call_id_to_anthropic(tc.id)},
                             {"name", tc.name},
                             {"input", json::object()}}}}))
            return false;
        const std::string& args = tc.arguments;
        constexpr size_t kChunk = 48;
        for (size_t off = 0; off < args.size(); off += kChunk) {
            size_t n = std::min(kChunk, args.size() - off);
            if (!out.emit("content_block_delta",
                          json{{"type", "content_block_delta"},
                               {"index", block_index},
                               {"delta",
                                {{"type", "input_json_delta"},
                                 {"partial_json", args.substr(off, n)}}}}))
                return false;
        }
        return stop_block();
    };

    int n_output_tokens = 0;
    const char* finish = nullptr;
    double ttft_ms = 0.0;

    std::string utf8_buf;        // confirmed-UTF8 content buffer
    std::string pending_text;    // stop-sequence holdback
    bool text_stop_matched = false;

    // Tool call detection state machine (same shape as run_chat_stream_).
    enum class ToolPhase { CONTENT, TAG_SCANNING, TOOL_CALL_BODY };
    ToolPhase tool_phase = ToolPhase::CONTENT;
    std::string tool_tag_buf, tool_body_buf, tool_close_tag, tool_fn_name;
    std::vector<ParsedToolCall> stream_tool_calls;
    bool tool_calls_emitted = false;

    // Reasoning extraction (DeepSeek <think>). enable_thinking also covers
    // text-level thinkers (Nemotron) — see the chat streaming path.
    enum class ThinkPhase { SCAN, REASONING, CONTENT };
    bool use_reasoning = (state.default_args.reasoning_format == "deepseek" &&
                          (snap_is_think_model || enable_thinking));
    ThinkPhase think_phase;
    if (enable_thinking)
        think_phase = ThinkPhase::REASONING;
    else if (use_reasoning && think_budget > 0.0f)
        think_phase = ThinkPhase::SCAN;
    else
        think_phase = ThinkPhase::CONTENT;
    std::string reasoning_utf8_buf, think_scan_buf;
    int think_scan_count = 0;
    bool content_started = (think_phase == ThinkPhase::CONTENT);
    int think_reentries = 0;
    const int kMaxThinkReentries = 1;
    const int kThinkScanLimit = 8;
    bool channel_header_active = false;

    auto flush_text = [&](size_t up_to) -> bool {
        if (up_to == 0)
            return true;
        bool ok = emit_text(pending_text.substr(0, up_to));
        pending_text.erase(0, up_to);
        return ok;
    };

    auto request_start = std::chrono::steady_clock::now();
    for (;;) {
        // #755: when a thinking model exhausts its token budget the final
        // (is_last) token lands inside the REASONING phase, which `continue`s
        // and skips the trailing `if (finish) break`. The loop would then spin
        // on pop_token forever — message_delta/message_stop never emitted, so
        // the Anthropic client (SDK or curl -N) hangs indefinitely. Breaking
        // here guarantees the terminal events are always sent.
        if (finish)
            break;

        if (!sink.is_writable()) {
            server_req->cancel();
            state.metrics.requests_cancelled++;
            finish = "cancelled";
            break;
        }
        if (state.request_timeout > 0) {
            auto elapsed = std::chrono::steady_clock::now() - request_start;
            if (elapsed > std::chrono::seconds(state.request_timeout)) {
                server_req->cancel();
                finish = "length";
                break;
            }
        }

        TokenEvent evt{};
        if (!server_req->pop_token(evt)) {
            // No token ready yet (prefill / generation gap). Re-ping at most
            // every ~10s so idle streams stay alive without spamming.
            auto now = std::chrono::steady_clock::now();
            if (now - last_ping > std::chrono::seconds(10)) {
                last_ping = now;
                if (!out.emit("ping", json{{"type", "ping"}}))
                    break;
            }
            continue;
        }

        if (evt.token_id < 0) {
            finish = evt.finish_reason ? evt.finish_reason : "stop";
            break;
        }
        int32_t token = evt.token_id;

        if (!evt.is_last) {
            bool is_structural_stop = (token == snap_tok->eos_id());
            if (!is_structural_stop && snap_have_template) {
                for (int32_t stop_id : snap_stop_token_ids)
                    if (token == stop_id) {
                        is_structural_stop = true;
                        break;
                    }
            }
            if (is_structural_stop)
                continue;
        }
        if (evt.is_last) {
            if (token == snap_tok->eos_id()) {
                finish = evt.finish_reason ? evt.finish_reason : "stop";
                break;
            }
            bool is_stop = false;
            if (snap_have_template) {
                for (int32_t stop_id : snap_stop_token_ids)
                    if (token == stop_id) {
                        is_stop = true;
                        break;
                    }
            }
            if (is_stop) {
                finish = evt.finish_reason ? evt.finish_reason : "stop";
                break;
            }
            finish = evt.finish_reason ? evt.finish_reason : "length";
        }

        n_output_tokens++;
        if (n_output_tokens == 1)
            ttft_ms = std::chrono::duration<double, std::milli>(
                          std::chrono::high_resolution_clock::now() - t_start)
                          .count();
        std::string piece = snap_tok->decode_token(token);

        // gpt-oss Harmony channel routing (analysis/commentary -> thinking,
        // final -> text). Markers arrive as atomic special-token pieces.
        if (harmony) {
            if (piece == "<|channel|>" || piece == "<|message|>" || piece == "<|end|>" ||
                piece == "<|return|>" || piece == "<|start|>") {
                if (hm_in_msg && !hm_flush(/*force=*/true))
                    return false;
                if (piece == "<|channel|>") {
                    hm_reading_name = true;
                    hm_in_msg = false;
                    hm_name.clear();
                } else if (piece == "<|message|>") {
                    size_t s = hm_name.find_first_not_of("\n\r\t ");
                    size_t e = hm_name.find_last_not_of("\n\r\t ");
                    hm_channel = (s == std::string::npos) ? std::string() : hm_name.substr(s, e - s + 1);
                    hm_reading_name = false;
                    hm_in_msg = true;
                } else {
                    hm_in_msg = false;
                    hm_reading_name = false;
                    hm_channel.clear();
                }
                continue;
            }
            if (hm_reading_name) {
                hm_name += piece;
                continue;
            }
            if (!hm_in_msg)
                continue;
            hm_buf += piece;
            if (!hm_flush(/*force=*/false))
                return false;
            continue;
        }

        // Gemma-4 channel filter.
        if (snap_channel_open_id >= 0) {
            if (channel_header_active) {
                if (token == snap_channel_newline_id || (!piece.empty() && piece.back() == '\n'))
                    channel_header_active = false;
                continue;
            }
            if (token == snap_channel_open_id) {
                channel_header_active = true;
                continue;
            }
            if (token == snap_channel_close_id)
                continue;
        }

        // Reasoning extraction.
        if (think_phase == ThinkPhase::SCAN) {
            if (token == snap_think_start_id) {
                think_phase = ThinkPhase::REASONING;
                continue;
            }
            think_scan_buf += piece;
            think_scan_count++;
            if (think_scan_buf.find("<think>") != std::string::npos) {
                think_phase = ThinkPhase::REASONING;
                auto pos = think_scan_buf.find("<think>");
                std::string after = think_scan_buf.substr(pos + 7);
                think_scan_buf.clear();
                if (!after.empty())
                    reasoning_utf8_buf += after;
                continue;
            }
            if (think_scan_count == 1 && piece.empty()) {
                think_phase = ThinkPhase::REASONING;
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
            if (token == snap_think_end_id) {
                size_t complete = utf8_complete_len(reasoning_utf8_buf);
                if (!emit_thinking(reasoning_utf8_buf.substr(0, complete)))
                    return false;
                reasoning_utf8_buf.clear();
                think_phase = ThinkPhase::CONTENT;
                continue;
            }
            if (token == snap_think_start_id)
                continue;
            reasoning_utf8_buf += piece;
            for (;;) {
                auto tp = reasoning_utf8_buf.find("<think>");
                if (tp == std::string::npos)
                    break;
                reasoning_utf8_buf.erase(tp, 7);
            }
            auto end_pos = reasoning_utf8_buf.find("</think>");
            if (end_pos != std::string::npos) {
                if (!emit_thinking(reasoning_utf8_buf.substr(0, end_pos)))
                    return false;
                think_phase = ThinkPhase::CONTENT;
                std::string after = reasoning_utf8_buf.substr(end_pos + 8);
                reasoning_utf8_buf.clear();
                auto start = after.find_first_not_of("\n\r\t ");
                if (start != std::string::npos)
                    piece = after.substr(start);
                else
                    continue;
            } else {
                constexpr size_t kOverlap = 7;
                size_t complete = utf8_complete_len(reasoning_utf8_buf);
                if (complete > kOverlap) {
                    size_t emit_end = complete - kOverlap;
                    while (emit_end > 0 &&
                           (static_cast<unsigned char>(reasoning_utf8_buf[emit_end]) & 0xC0) == 0x80)
                        --emit_end;
                    if (emit_end > 0) {
                        std::string to_emit = reasoning_utf8_buf.substr(0, emit_end);
                        reasoning_utf8_buf = reasoning_utf8_buf.substr(emit_end);
                        if (!emit_thinking(to_emit))
                            return false;
                    }
                }
                continue;
            }
        }

        if (!content_started && think_phase == ThinkPhase::CONTENT) {
            auto ns = piece.find_first_not_of("\n\r\t ");
            if (ns == std::string::npos)
                continue;
            piece = piece.substr(ns);
            content_started = true;
        }

        if (use_reasoning) {
            if (token == snap_think_start_id) {
                if (think_reentries < kMaxThinkReentries) {
                    think_phase = ThinkPhase::REASONING;
                    think_reentries++;
                }
                continue;
            }
            if (token == snap_think_end_id) {
                continue;
            }
            for (;;) {
                auto p = piece.find("<think>");
                if (p != std::string::npos) {
                    piece.erase(p, 7);
                    continue;
                }
                p = piece.find("</think>");
                if (p != std::string::npos) {
                    piece.erase(p, 8);
                    continue;
                }
                break;
            }
            if (piece.empty())
                continue;
        }

        // Tool call body accumulation.
        if (has_tools && tool_phase == ToolPhase::TOOL_CALL_BODY) {
            tool_body_buf += piece;
            auto close_pos = tool_body_buf.find(tool_close_tag);
            if (close_pos != std::string::npos) {
                std::string body = tool_body_buf.substr(0, close_pos);
                auto bs = body.find_first_not_of("\n\r\t ");
                auto be = body.find_last_not_of("\n\r\t ");
                if (bs != std::string::npos && be != std::string::npos)
                    body = body.substr(bs, be - bs + 1);
                try {
                    json j = json::parse(body);
                    ParsedToolCall tc;
                    tc.id = "call_imp_" + std::to_string(state.next_tool_call_id.fetch_add(1));
                    if (tpl_family == imp::ChatTemplateFamily::LLAMA3) {
                        tc.name = tool_fn_name;
                        tc.arguments = dump_safe(j);
                    } else {
                        tc.name = j.value("name", "");
                        if (j.contains("arguments"))
                            tc.arguments = dump_safe(j["arguments"]);
                        else {
                            json args = j;
                            args.erase("name");
                            tc.arguments = dump_safe(args);
                        }
                    }
                    if (!tc.name.empty()) {
                        validate_tool_call(tc, ctx.params.tools);
                        if (!tc.valid) {
                            fprintf(stderr, "[%s] tool-call arg validation failed: %s: %s\n",
                                    msg_id.c_str(), tc.name.c_str(), tc.error.c_str());
                        }
                        if (!emit_tool_use(tc))
                            return false;
                        stream_tool_calls.push_back(std::move(tc));
                        tool_calls_emitted = true;
                    }
                } catch (...) {
                    // Malformed tool-call JSON — skip this block and continue streaming
                }
                std::string after = tool_body_buf.substr(close_pos + tool_close_tag.size());
                tool_body_buf.clear();
                tool_phase = ToolPhase::CONTENT;
                if (!after.empty()) {
                    auto ws = after.find_first_not_of("\n\r\t ");
                    if (ws != std::string::npos)
                        piece = after.substr(ws);
                    else
                        continue;
                } else {
                    continue;
                }
            } else {
                continue;
            }
        }

        if (has_tools && tool_phase == ToolPhase::TAG_SCANNING) {
            tool_tag_buf += piece;
            if (tpl_family != imp::ChatTemplateFamily::LLAMA3) {
                if (tool_tag_buf.size() >= 11) {
                    auto pos = tool_tag_buf.find("<tool_call>");
                    if (pos != std::string::npos) {
                        std::string before = tool_tag_buf.substr(0, pos);
                        if (!before.empty() && !emit_text(before))
                            return false;
                        tool_body_buf = tool_tag_buf.substr(pos + 11);
                        tool_close_tag = "</tool_call>";
                        tool_tag_buf.clear();
                        tool_phase = ToolPhase::TOOL_CALL_BODY;
                        continue;
                    }
                    const char* tc_tag = "<tool_call>";
                    bool could_match = true;
                    for (size_t k = 0; k < tool_tag_buf.size() && k < 11; k++)
                        if (tool_tag_buf[k] != tc_tag[k]) {
                            could_match = false;
                            break;
                        }
                    if (!could_match) {
                        piece = tool_tag_buf;
                        tool_tag_buf.clear();
                        tool_phase = ToolPhase::CONTENT;
                    } else {
                        continue;
                    }
                } else {
                    const char* tc_tag = "<tool_call>";
                    bool could_match = true;
                    for (size_t k = 0; k < tool_tag_buf.size() && k < 11; k++)
                        if (tool_tag_buf[k] != tc_tag[k]) {
                            could_match = false;
                            break;
                        }
                    if (!could_match) {
                        piece = tool_tag_buf;
                        tool_tag_buf.clear();
                        tool_phase = ToolPhase::CONTENT;
                    } else {
                        continue;
                    }
                }
            } else {
                if (tool_tag_buf.size() >= 10) {
                    auto fn_pos = tool_tag_buf.find("<function=");
                    if (fn_pos != std::string::npos) {
                        auto gt = tool_tag_buf.find('>', fn_pos + 10);
                        if (gt != std::string::npos) {
                            std::string before = tool_tag_buf.substr(0, fn_pos);
                            if (!before.empty() && !emit_text(before))
                                return false;
                            tool_fn_name = tool_tag_buf.substr(fn_pos + 10, gt - (fn_pos + 10));
                            tool_body_buf = tool_tag_buf.substr(gt + 1);
                            tool_close_tag = "</function>";
                            tool_tag_buf.clear();
                            tool_phase = ToolPhase::TOOL_CALL_BODY;
                            continue;
                        } else {
                            continue;
                        }
                    }
                }
                const char* fn_tag = "<function=";
                bool could_match = true;
                for (size_t k = 0; k < tool_tag_buf.size() && k < 10; k++)
                    if (tool_tag_buf[k] != fn_tag[k]) {
                        could_match = false;
                        break;
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

        if (has_tools && tool_phase == ToolPhase::CONTENT) {
            size_t lt_pos = piece.find('<');
            if (lt_pos != std::string::npos) {
                if (lt_pos > 0) {
                    std::string before = piece.substr(0, lt_pos);
                    if (stop_sequences.empty())
                        utf8_buf += before;
                    else
                        pending_text += before;
                }
                tool_tag_buf = piece.substr(lt_pos);
                tool_phase = ToolPhase::TAG_SCANNING;
                if (stop_sequences.empty() && !utf8_buf.empty()) {
                    size_t complete = utf8_complete_len(utf8_buf);
                    if (complete > 0) {
                        if (!emit_text(utf8_buf.substr(0, complete)))
                            return false;
                        utf8_buf.erase(0, complete);
                    }
                } else if (!stop_sequences.empty()) {
                    auto d = imp::stream::holdback_decision(pending_text, max_stop_len,
                                                            stop_sequences);
                    if (!flush_text(d.flush_len))
                        return false;
                    if (d.complete_match) {
                        text_stop_matched = true;
                        finish = "stop";
                        break;
                    }
                }
                continue;
            }
        }

        // Normal content emission.
        if (stop_sequences.empty()) {
            utf8_buf += piece;
            size_t complete = utf8_complete_len(utf8_buf);
            if (complete > 0) {
                if (!emit_text(utf8_buf.substr(0, complete)))
                    return false;
                utf8_buf.erase(0, complete);
            }
        } else {
            pending_text += piece;
            auto d = imp::stream::holdback_decision(pending_text, max_stop_len, stop_sequences);
            if (!flush_text(d.flush_len))
                return false;
            if (d.complete_match) {
                text_stop_matched = true;
                finish = "stop";
                break;
            }
        }

        if (finish)
            break;
    }

    // Flush trailing buffers.
    // Harmony: flush the final channel's tail (ends at EOS/<|return|> with no
    // trailing <|end|>); the other buffers below stay empty for harmony.
    if (harmony && !hm_buf.empty())
        hm_flush(/*force=*/true);

    if (think_phase == ThinkPhase::SCAN && !think_scan_buf.empty()) {
        utf8_buf += think_scan_buf;
        think_scan_buf.clear();
    }
    if (!reasoning_utf8_buf.empty()) {
        emit_thinking(reasoning_utf8_buf);
        reasoning_utf8_buf.clear();
    }
    if (tool_phase != ToolPhase::CONTENT && !tool_calls_emitted) {
        std::string leftover = tool_tag_buf + tool_body_buf;
        if (!leftover.empty())
            utf8_buf += leftover;
    }
    if (!utf8_buf.empty() && !text_stop_matched && !tool_calls_emitted)
        emit_text(utf8_buf);
    if (!pending_text.empty() && !text_stop_matched && !tool_calls_emitted)
        emit_text(pending_text);

    // Close any block still open.
    stop_block();

    if (!finish)
        finish = tool_calls_emitted ? "tool_calls" : "length";
    else if (tool_calls_emitted && strcmp(finish, "stop") == 0)
        finish = "tool_calls";

    // Map finish_reason -> Anthropic stop_reason.
    std::string stop_reason;
    if (strcmp(finish, "stop") == 0)
        stop_reason = "end_turn";
    else if (strcmp(finish, "length") == 0)
        stop_reason = "max_tokens";
    else if (strcmp(finish, "tool_calls") == 0)
        stop_reason = "tool_use";
    else if (strcmp(finish, "cancelled") == 0)
        stop_reason = "end_turn";
    else
        stop_reason = finish;

    // ---- message_delta + message_stop ------------------------------------
    // Cache accounting is only known after prefill ran, so it rides on the
    // final usage update instead of message_start.
    json delta_usage = {{"output_tokens", n_output_tokens}};
    {
        int cached_now = (active_req && active_req->cached_tokens > 0) ? active_req->cached_tokens : 0;
        int creation = cache_creation_tokens_(active_req, n_prompt_tokens);
        if (cached_now > 0 || creation > 0) {
            delta_usage["input_tokens"] = n_prompt_tokens - cached_now;
            delta_usage["cache_read_input_tokens"] = cached_now;
            delta_usage["cache_creation_input_tokens"] = creation;
        }
    }
    out.emit("message_delta",
             json{{"type", "message_delta"},
                  {"delta", {{"stop_reason", stop_reason}, {"stop_sequence", nullptr}}},
                  {"usage", std::move(delta_usage)}});
    out.emit("message_stop", json{{"type", "message_stop"}});
    sink.done();

    // Metrics + log.
    auto t_end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    int cached = (active_req && active_req->cached_tokens > 0) ? active_req->cached_tokens : 0;
    state.metrics.requests_total++;
    state.metrics.tokens_prompt_total += n_prompt_tokens;
    state.metrics.tokens_completion_total += n_output_tokens;
    state.metrics.tokens_cached_total += cached;
    state.metrics.last_request_duration_ms = static_cast<int64_t>(ms);
    state.metrics.last_ttft_ms = static_cast<int64_t>(ttft_ms);
    state.metrics.request_duration.observe(ms / 1000.0);
    if (n_output_tokens > 0)
        state.metrics.ttft.observe(ttft_ms / 1000.0);
    log_request_jsonl(state, ctx.log_skip, ctx.t_log_start, msg_id, ctx.log_endpoint, ctx.log_client_ip,
                      ctx.log_raw_body, ms, n_prompt_tokens, n_output_tokens, finish, json());
    fprintf(stderr, "[%s] messages stream: %d prompt + %d completion tokens, %.1f ms (ttft=%.1f ms)\n",
            msg_id.c_str(), n_prompt_tokens, n_output_tokens, ms, ttft_ms);
    return true;
}
