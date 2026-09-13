#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace imp {

// Gate allowing a free-form preamble before strict JSON enforcement. Two axes:
// think-close (close_token>=0) vs budget-only (close_token<0, max_tokens>0); tool-aware
// (configure_with_tools) vs legacy.
// Think-close: reasoning models (Qwen3.6/DeepSeek-R1/Gemma-4) prepend <think>...</think>;
// gate stays no-mask until close_token, budget is a safety cap.
// Budget-only: non-reasoning models wrapping JSON in markdown/verbal preambles; gate exits
// on first {/[, budget exhaustion, or (tool-aware) a tool opener.
// Tool-aware overlay: tool-opener/close token ids + optional char prefix/suffix (multi-token
// dialects, e.g. Llama3 <function=NAME>). FSM:
//   ACTIVE      - preamble running, mask off. {/[/close/budget->OFF; tool-opener->TOOL_BODY
//                 (legacy) or TOOL_ARGS (strict_tool).
//   TOOL_BODY   - inside tool call, mask off. tool-close->TERMINAL_OFF.
//   TOOL_ARGS   - inside tool call (strict_tool): mask ENGAGED, FSM constrains args (#1002).
//   TERMINAL_OFF- tool call closed, mask off forever.
//   OFF         - preamble exited, FSM mask applies.
// active()=true bypasses mask (ACTIVE/TOOL_BODY/TERMINAL_OFF); false = FSM enforces (OFF/TOOL_ARGS).
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

    // Tool-aware configure: open_tokens/close_tokens = tool-tag boundary token ids
    // (ChatML <tool_call>, Gemma <|tool_call>); open_prefix/close_suffix = char-level
    // fallback for multi-token dialects (Llama3 <function=). Empty open_tokens AND
    // open_prefix = tool detection disabled (behaves like legacy 2-arg configure).
    // strict_tool: on tool opener, enter TOOL_ARGS (mask ENGAGED) instead of TOOL_BODY, so
    // the TOOL_CALL body FSM constrains an optional tool call's args (OpenAI strict:true, #1002).
    void configure_with_tools(int32_t close_token, int max_tokens, std::vector<int32_t> open_tokens,
                              std::vector<int32_t> close_tokens, std::string open_prefix,
                              std::string close_suffix, bool thinking_open = true, bool strict_tool = false) {
        max_tokens_ = max_tokens > 0 ? max_tokens : 0;
        close_token_ = close_token;
        open_tokens_ = std::move(open_tokens);
        close_tokens_ = std::move(close_tokens);
        open_prefix_ = std::move(open_prefix);
        close_suffix_ = std::move(close_suffix);
        thinking_open_ = thinking_open;
        strict_tool_ = strict_tool;
        configured_ = (close_token >= 0) || (max_tokens_ > 0);
        reset();
    }

    void reset() {
        // Reasoning models gate on </think>, but if generation starts with thinking already
        // closed (/no_think: template emits empty <think></think> in the prompt, no </think>
        // generated), waiting for a close token that never comes lets the model ramble
        // unconstrained until budget. Start OFF instead so the structural mask applies immediately.
        // Tool-aware mode keeps ACTIVE (tool openers may follow); budget-only unaffected.
        const bool reasoning_already_closed = (close_token_ >= 0) && !thinking_open_ &&
                                              !tool_detection_active();
        state_ = (configured_ && !reasoning_already_closed) ? State::ACTIVE : State::OFF;
        seen_ = 0;
        char_buf_.clear();
    }

    // active()=true skips the FSM mask: ACTIVE (preamble), TOOL_BODY (unconstrained tool
    // call), TERMINAL_OFF (after tool call closed). OFF and TOOL_ARGS (strict tool-call body)
    // let the FSM mask through.
    bool active() const noexcept {
        return state_ == State::ACTIVE || state_ == State::TOOL_BODY || state_ == State::TERMINAL_OFF;
    }

    // True once a strict optional tool call opened and its body is now
    // FSM-constrained (#1002) — the constrainer engages its TOOL_CALL body
    // frame on the ACTIVE → TOOL_ARGS transition.
    bool in_tool_args() const noexcept { return state_ == State::TOOL_ARGS; }

    // Re-arm after a strict tool-call body completes (parallel_tool_calls, #1002): TOOL_ARGS
    // -> ACTIVE so free text/EOS/another tool opener pass again, and the next <tool_call>
    // starts a fresh body FSM. Preamble budget restarts for the inter-call gap.
    void rearm_after_call() noexcept {
        state_ = State::ACTIVE;
        seen_ = 0;
        char_buf_.clear();
    }

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
            case State::TOOL_ARGS:
                return false;  // strict body: every token is driven by the FSM
            case State::OFF:
                return false;
        }
        return false;
    }

private:
    enum class State : uint8_t { ACTIVE, TOOL_BODY, TOOL_ARGS, TERMINAL_OFF, OFF };

    bool absorb_active(int32_t token, const std::string& text) {
        seen_++;

        // On close token (e.g. </think>): tool-aware mode stays ACTIVE (tool-opener detection
        // continues post-think) and resets the budget for fresh post-think slack. Legacy mode
        // (no tool detection) exits to OFF so the FSM enforces the structural mask.
        if (close_token_ >= 0 && token == close_token_) {
            if (tool_detection_active()) {
                seen_ = 0;
                char_buf_.clear();
                return true;
            }
            state_ = State::OFF;
            return true;
        }

        // Tool-opener detection (token-set fast-path). strict_tool → TOOL_ARGS
        // (mask engaged, FSM constrains the body); legacy → TOOL_BODY (mask off).
        if (is_tool_open_token(token)) {
            state_ = strict_tool_ ? State::TOOL_ARGS : State::TOOL_BODY;
            char_buf_.clear();
            return true;
        }

        // Tool-opener detection (char-prefix fallback).
        if (!open_prefix_.empty()) {
            append_char_buf(text);
            if (char_buf_.find(open_prefix_) != std::string::npos) {
                state_ = strict_tool_ ? State::TOOL_ARGS : State::TOOL_BODY;
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

    bool tool_detection_active() const { return !open_tokens_.empty() || !open_prefix_.empty(); }

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
    bool strict_tool_ = false;   // true = opener enters TOOL_ARGS (FSM constrains body)
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
