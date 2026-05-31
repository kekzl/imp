#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace imp {

// Gate that lets a model emit a free-form preamble before strict JSON
// enforcement kicks in. Two-axis configuration: think-close vs. budget
// (selects when the gate "gives up" on a non-tool path) and tool-aware
// vs. legacy (controls whether tool-tag transitions are recognised).
//
//   Think-close mode (close_token >= 0)
//     Reasoning models (Qwen3.6, DeepSeek-R1, Gemma-4 thinking) prepend
//     `<think>...</think>` to every response. Gate stays "no-mask"
//     until the close token; budget is a safety cap.
//
//   Budget-only mode (close_token < 0, max_tokens > 0)
//     Non-reasoning models that wrap JSON in markdown fences (` ```json `)
//     or short verbal preambles ("Sure! ") need a small slack window
//     before strict enforcement. Gate exits on the first `{`/`[`, on
//     budget exhaustion, or — in tool-aware mode — on a tool opener.
//
// Tool-aware overlay (configure_with_tools): hand the gate sets of
// tool-opener/close token IDs plus optional char-level prefix/suffix
// for multi-token dialects (Llama3 <function=NAME>). Internal FSM:
//
//   ACTIVE        — preamble running; mask off; transitions:
//                     · `{`/`[`/close-token/budget → OFF (mask kicks in)
//                     · tool-opener (token or char-prefix) → TOOL_BODY
//   TOOL_BODY     — inside a tool call; mask off; transitions:
//                     · tool-close (token or char-suffix) → TERMINAL_OFF
//   TERMINAL_OFF  — tool call closed; mask off forever (parallel calls,
//                   trailing text, EOS all pass through).
//   OFF           — preamble exited normally; FSM mask now applies.
//
// External API stays binary:
//   active() = true  → mask is bypassed (ACTIVE / TOOL_BODY / TERMINAL_OFF)
//   active() = false → mask is enforced by FSM (OFF only)
//   absorb() returns true if the token was consumed by the gate, false
//   only on the ACTIVE → OFF transition via `{`/`[` (forwarded to FSM).
class PreambleGate {
public:
    // Existing two-arg overload — preserved for non-tool callers.
    void configure(int32_t close_token, int max_tokens, bool thinking_open = true) {
        configure_with_tools(close_token, max_tokens,
                             /*open_tokens=*/{},
                             /*close_tokens=*/{},
                             /*open_prefix=*/"",
                             /*close_suffix=*/"", thinking_open);
    }

    // Tool-aware configure. open_tokens/close_tokens are token IDs of
    // tool-tag boundaries (single-token dialects: ChatML <tool_call>,
    // Gemma <|tool_call>). open_prefix/close_suffix are char-level
    // fallbacks for multi-token dialects (Llama3 <function=).
    //
    // Empty open_tokens AND empty open_prefix means "tool detection
    // disabled" — gate behaves exactly like the legacy two-arg configure.
    void configure_with_tools(int32_t close_token, int max_tokens,
                              std::vector<int32_t> open_tokens,
                              std::vector<int32_t> close_tokens,
                              std::string open_prefix,
                              std::string close_suffix,
                              bool thinking_open = true) {
        max_tokens_ = max_tokens > 0 ? max_tokens : 0;
        close_token_ = close_token;
        open_tokens_ = std::move(open_tokens);
        close_tokens_ = std::move(close_tokens);
        open_prefix_ = std::move(open_prefix);
        close_suffix_ = std::move(close_suffix);
        thinking_open_ = thinking_open;
        configured_ = (close_token >= 0) || (max_tokens_ > 0);
        reset();
    }

    void reset() {
        // Reasoning models gate on </think>. But if generation begins with the
        // thinking block ALREADY closed (e.g. /no_think — the template emits an
        // empty <think></think> in the prompt, so no </think> is ever generated),
        // there is nothing to absorb: waiting for a close token that never comes
        // would let the model ramble unconstrained until the budget. Start OFF so
        // the structural mask enforces immediately. Tool-aware mode keeps ACTIVE
        // (tool openers may still appear post-think) and budget-only mode is
        // unaffected (close_token_ < 0).
        const bool reasoning_already_closed =
            (close_token_ >= 0) && !thinking_open_ && !tool_detection_active();
        state_ = (configured_ && !reasoning_already_closed) ? State::ACTIVE : State::OFF;
        seen_ = 0;
        char_buf_.clear();
    }

    // active() returns true whenever the FSM mask should be skipped.
    // That's three of the four internal states: ACTIVE (preamble),
    // TOOL_BODY (inside a tool call), TERMINAL_OFF (after a tool call
    // closed). Only OFF — reached via {/[/think-close/budget — lets
    // the FSM mask through.
    bool active() const noexcept { return state_ != State::OFF; }

    // Returns true if the token was fully consumed by the gate (FSM should
    // NOT process it). Returns false only for the one transition where
    // the token must be forwarded to the FSM: ACTIVE → OFF via `{` or `[`.
    bool absorb(int32_t token, const std::string& text) {
        switch (state_) {
            case State::ACTIVE:
                return absorb_active(token, text);
            case State::TOOL_BODY:
                return absorb_tool_body(token, text);
            case State::TERMINAL_OFF:
                return true;  // permanently absorbing
            case State::OFF:
                return false;
        }
        return false;
    }

private:
    enum class State : uint8_t { ACTIVE, TOOL_BODY, TERMINAL_OFF, OFF };

    bool absorb_active(int32_t token, const std::string& text) {
        seen_++;

        // Close token (e.g. </think>) — in tool-aware mode, stay ACTIVE so
        // tool-opener detection runs in the post-think window. Reset the
        // budget counter so the post-think slack is fresh. In legacy mode
        // (no tool detection), exit to OFF and let the FSM enforce the
        // structural mask.
        if (close_token_ >= 0 && token == close_token_) {
            if (tool_detection_active()) {
                seen_ = 0;
                char_buf_.clear();
                return true;
            }
            state_ = State::OFF;
            return true;
        }

        // Tool-opener detection (token-set fast-path).
        if (is_tool_open_token(token)) {
            state_ = State::TOOL_BODY;
            char_buf_.clear();
            return true;
        }

        // Tool-opener detection (char-prefix fallback).
        if (!open_prefix_.empty()) {
            append_char_buf(text);
            if (char_buf_.find(open_prefix_) != std::string::npos) {
                state_ = State::TOOL_BODY;
                char_buf_.clear();
                return true;
            }
        }

        // JSON start — exit to FSM and forward this token.
        for (char c : text) {
            if (c == '{' || c == '[') {
                state_ = State::OFF;
                return false;
            }
        }

        // Budget exhausted — give up on preamble.
        if (max_tokens_ > 0 && seen_ >= max_tokens_) {
            state_ = State::OFF;
            return true;
        }

        return true;
    }

    bool absorb_tool_body(int32_t token, const std::string& text) {
        // Close-token detection (token-set fast-path).
        if (is_tool_close_token(token)) {
            state_ = State::TERMINAL_OFF;
            char_buf_.clear();
            return true;
        }

        // Close-suffix detection (char-suffix fallback).
        if (!close_suffix_.empty()) {
            append_char_buf(text);
            if (char_buf_.find(close_suffix_) != std::string::npos) {
                state_ = State::TERMINAL_OFF;
                char_buf_.clear();
                return true;
            }
        }

        return true;  // body content is always absorbed
    }

    bool tool_detection_active() const {
        return !open_tokens_.empty() || !open_prefix_.empty();
    }

    bool is_tool_open_token(int32_t token) const {
        if (token < 0)
            return false;
        for (int32_t t : open_tokens_) {
            if (t == token)
                return true;
        }
        return false;
    }

    bool is_tool_close_token(int32_t token) const {
        if (token < 0)
            return false;
        for (int32_t t : close_tokens_) {
            if (t == token)
                return true;
        }
        return false;
    }

    void append_char_buf(const std::string& text) {
        // Sliding window: keep up to 32 chars (covers any tag we care about).
        char_buf_ += text;
        if (char_buf_.size() > 32)
            char_buf_.erase(0, char_buf_.size() - 32);
    }

    bool configured_ = false;
    bool thinking_open_ = true;  // false = thinking already closed at gen start (e.g. /no_think)
    State state_ = State::OFF;
    int32_t close_token_ = -1;
    int max_tokens_ = 0;
    int seen_ = 0;

    std::vector<int32_t> open_tokens_;
    std::vector<int32_t> close_tokens_;
    std::string open_prefix_;
    std::string close_suffix_;
    std::string char_buf_;
};

}  // namespace imp
