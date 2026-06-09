#pragma once

// Pure (GPU-free) decision logic for the think/stop state machine.
//
// These free functions hold the host-side reasoning of engine_sampling_stop.cpp
// (think-budget recount, text-tail </think> detection, post-think grace period)
// and engine_workspace_warmup.cpp (think-token-type acceptance), extracted so the
// branchy logic can be unit-tested on the CPU without standing up an Engine + GPU.
//
// The Engine methods are thin wrappers that supply tokenizer/model state and
// call into here. Behaviour must stay byte-identical to the wrappers — these are
// a seam, not a redesign.

#include <cstdint>
#include <string>
#include <vector>

namespace imp::think_logic {

// --- Warmup: should <think> be treated as a control/think token? ---
//
// Mirrors engine_workspace_warmup.cpp's acceptance test. <think>/</think> are
// the think markers ONLY when <think> is a special token, not plain text:
//   - GGUF metadata tags them CONTROL (type 3)  -> accept
//   - Qwen3 GGUFs tag them USER_DEFINED (type 4) -> accept (was the day-N bug:
//     requiring CONTROL left think_end_id_ == -1, so the budget never fired)
//   - Nemotron has "<think>" at ID 12 tagged NORMAL (type 1) -> reject
// When the tokenizer carries no type table, fall back to the heuristic
// "lives in the top 1% of the vocab id range" (added/special tokens cluster there).
//
// `start_id` < 0 means the literal "<think>" is absent -> never a think model.
//
// `is_added` == the token was declared in tokenizer.json's added_tokens array.
// Qwen3/Qwen3.x NVFP4 SafeTensors ship <think>/</think> as added tokens with
// special=false (is_special=false), so the is_special gate alone left
// think_end_id_ == -1 and forced those reasoning models onto the eager decode
// path (no conditional-graph loop, −27..36% decode). An *added* marker is a
// deliberate control marker even when special=false; only a NORMAL BPE piece
// that merely spells "<think>" (Nemotron ID 12, not added) must stay rejected.
inline bool accept_think_token(int32_t start_id, bool has_token_types, bool is_special, bool is_added,
                               int vocab_size) {
    if (start_id < 0)
        return false;
    if (has_token_types)
        return is_special || is_added;  // special OR an explicit added marker
    return start_id > vocab_size * 99 / 100;
}

// --- Think-budget recount (fill_sampling_params) ---
//
// Walk the emitted output and count the tokens generated while inside a think
// block. The opener may live in the PROMPT (injected <think>\n prefix), in which
// case the output has no opener and the recount must START in-think
// (`started_in_think`) or the budget never fires.
//
// `currently_thinking_out` receives the in/out think state at the end of the
// scan (the budget only fires while still thinking).
inline int count_reasoning_tokens(const std::vector<int32_t>& output_tokens, int32_t think_start_id,
                                  int32_t think_end_id, bool started_in_think, bool& currently_thinking_out) {
    bool currently_thinking = started_in_think;
    int n_reasoning = 0;
    for (int32_t t : output_tokens) {
        if (t == think_start_id)
            currently_thinking = true;
        else if (t == think_end_id)
            currently_thinking = false;
        else if (currently_thinking)
            n_reasoning++;
    }
    currently_thinking_out = currently_thinking;
    return n_reasoning;
}

// Should the sampler force a </think> token this step? True when budgeting is
// active, a </think> id exists, the model is still thinking, and the reasoning
// count has reached the limit (= max_tokens * think_budget).
inline bool should_force_think_end(float think_budget, int32_t think_end_id, int max_tokens,
                                   const std::vector<int32_t>& output_tokens, int32_t think_start_id,
                                   bool started_in_think) {
    if (!(think_budget > 0.0f) || think_end_id < 0 || output_tokens.empty())
        return false;
    int think_limit = static_cast<int>(max_tokens * think_budget);
    bool currently_thinking = false;
    int n_reasoning = count_reasoning_tokens(output_tokens, think_start_id, think_end_id, started_in_think,
                                             currently_thinking);
    return currently_thinking && n_reasoning >= think_limit;
}

// --- Text-tail </think> / <think> detection (track_think_state fallback) ---
//
// For tokenizers that ship <think>/</think> as added_tokens with special=False
// (Qwen3.6, Qwen3-Coder NVFP4 SafeTensors): there is no single token id, the
// markers arrive split across multiple BPE pieces (e.g. ['</','think','>']).
// We accumulate decoded pieces in a sliding window and match the literal string.
//
// Holds exactly the state track_think_state mutates on a Request.
struct TextThinkState {
    bool in_think_block = false;
    std::string think_text_tail;

    static constexpr size_t kWindow = 32;

    // Feed one decoded piece. Returns true if a transition just fired (entered
    // or exited a think block) this call.
    bool feed_piece(const std::string& piece) {
        if (piece.empty())
            return false;
        think_text_tail += piece;
        if (think_text_tail.size() > kWindow)
            think_text_tail.erase(0, think_text_tail.size() - kWindow);
        if (in_think_block) {
            if (think_text_tail.find("</think>") != std::string::npos) {
                in_think_block = false;
                think_text_tail.clear();
                return true;
            }
        } else {
            if (think_text_tail.find("<think>") != std::string::npos &&
                think_text_tail.find("</think>") == std::string::npos) {
                in_think_block = true;
                think_text_tail.clear();
                return true;
            }
        }
        return false;
    }
};

// --- Post-</think> grace period (should_stop) ---
//
// After the think block closes, suppress an early stop until at least
// kMinAnswerAfterThink content tokens have been emitted: numerically-noisy
// NVFP4 quants can close an empty thinking block in ~3 tokens and then EOS to a
// zero-content completion. `think_exit_idx` is the output index at which think
// last closed (-1 if never), `output_size` the current output length.
inline constexpr int kMinAnswerAfterThink = 16;

inline bool grace_blocks_stop(int think_exit_idx, int output_size) {
    if (think_exit_idx < 0)
        return false;
    int tokens_since_exit = output_size - think_exit_idx;
    return tokens_since_exit < kMinAnswerAfterThink;
}

}  // namespace imp::think_logic
