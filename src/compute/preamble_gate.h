#pragma once

#include <cstdint>
#include <string>

namespace imp {

// Gate that lets a model emit a free-form preamble before strict JSON
// enforcement kicks in — needed for reasoning models (Qwen3.6, DeepSeek-R1,
// Gemma-4 thinking) that prepend `<think>...</think>` to every response.
//
// State:
//   active=true   →  apply_mask is a no-op, all tokens accepted
//   active=false  →  underlying JSON/schema FSM enforces structure
//
// Transitions out of active (any of):
//   - close_token observed (e.g. </think>) — token is consumed by gate
//   - JSON-start char ({ or [) seen in token text — token is forwarded to FSM
//   - max_tokens budget exhausted — current token forwarded, next gets masked
class PreambleGate {
public:
    void configure(int32_t close_token, int max_tokens) {
        configured_ = (close_token >= 0);
        close_token_ = close_token;
        max_tokens_ = max_tokens > 0 ? max_tokens : 0;
        reset();
    }

    void reset() {
        active_ = configured_;
        seen_ = 0;
    }

    bool active() const noexcept { return active_; }

    // Returns true if the token was fully consumed by the preamble (the
    // underlying FSM should NOT process it). Returns false if the gate just
    // transitioned out of preamble and the token must be forwarded.
    bool absorb(int32_t token, const std::string& text) {
        if (!active_)
            return false;

        seen_++;

        // Close token (e.g. </think>) — consume and transition.
        if (token == close_token_) {
            active_ = false;
            return true;
        }

        // Model went straight to JSON — transition and forward this token.
        for (char c : text) {
            if (c == '{' || c == '[') {
                active_ = false;
                return false;
            }
        }

        // Budget exhausted — give up on preamble; next mask will force JSON.
        if (max_tokens_ > 0 && seen_ >= max_tokens_) {
            active_ = false;
            return true;
        }

        return true;
    }

private:
    bool configured_ = false;
    bool active_ = false;
    int32_t close_token_ = -1;
    int max_tokens_ = 0;
    int seen_ = 0;
};

}  // namespace imp
