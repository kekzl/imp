#pragma once

// Pure (GPU-free) decision logic for the think/stop state machine, shared by
// engine_sampling_stop.cpp and engine_workspace_warmup.cpp for CPU unit tests.
// Engine methods are thin wrappers; behaviour must stay byte-identical to them.

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace imp::think_logic {

// Chat templates (Qwen3.x, DeepSeek-R1) end the prompt with `<think>\n`: set
// in_think_block AND started_in_think together (Engine::add_request), or the
// budget recount misses the seed. Opener is "last" only if open_pos > close_pos+1.
struct ThinkSeed {
    bool in_think = false;          // suppress stop tokens; the block is open
    bool started_in_think = false;  // the budget recount starts in-think
};

inline ThinkSeed seed_from_prompt_tail(std::string_view tail) {
    const size_t open_pos = tail.rfind("<think>");
    const size_t close_pos = tail.rfind("</think>");
    const bool open_is_last = (open_pos != std::string_view::npos) &&
                              (close_pos == std::string_view::npos || open_pos > close_pos + 1);
    return ThinkSeed{open_is_last, open_is_last};
}

// <think>/</think> count as think markers only when <think> is a special
// token, not plain text (engine_workspace_warmup.cpp mirrors this test):
//   - GGUF CONTROL (type 3) or USER_DEFINED (type 4) -> accept
//   - NORMAL (type 1, e.g. Nemotron ID 12) -> reject
// No type table: fall back to "top 1% of vocab id range" heuristic.
// start_id < 0 -> "<think>" absent, never a think model.
// is_added (tokenizer.json added_tokens) counts as a control marker even
// with special=false (Qwen3.x NVFP4 SafeTensors); only a NORMAL BPE piece
// that merely spells "<think>" stays rejected.
inline bool accept_think_token(int32_t start_id, bool has_token_types, bool is_special, bool is_added,
                               int vocab_size) {
    if (start_id < 0)
        return false;
    if (has_token_types)
        return is_special || is_added;  // special OR an explicit added marker
    return start_id > vocab_size * 99 / 100;
}

// Counts tokens generated while inside a think block (fill_sampling_params).
// The opener may live in the PROMPT (`<think>\n`); recount must start in-think
// (started_in_think) or the budget never fires. currently_thinking_out returns final state.
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

// Floor under the fractional think_budget: a pure fraction over-reserves at
// large max_tokens and forces early </think>, causing reasoning to bleed into
// content. Fallback default for runtime.think_answer_reserve when no RuntimeConfig is at hand.
inline constexpr int kMaxAnswerReserve = 256;

// think_limit takes the LATER of the fractional budget and max_tokens minus
// answer_reserve_for's reserve (max_tokens/4, floored at kMaxAnswerReserve, #1248).
// reserve = runtime.think_answer_reserve; negative clamps to 0 (never adds thinking room).
inline constexpr int answer_reserve_for(int max_tokens, int reserve = kMaxAnswerReserve) {
    const int floor = reserve > 0 ? reserve : 0;
    const int scaled = max_tokens / 4;
    return scaled > floor ? scaled : floor;
}

// Reasoning-token limit shared by all enforcement paths (eager sampler,
// CUDA-graph device counter, n-gram/scheduler gates): the LATER of the
// fractional budget and max_tokens minus the answer reserve.
inline constexpr int think_limit(int max_tokens, float think_budget, int answer_reserve = kMaxAnswerReserve) {
    const int frac_limit = static_cast<int>(max_tokens * think_budget);
    const int reserve_limit = max_tokens - answer_reserve_for(max_tokens, answer_reserve);
    return frac_limit > reserve_limit ? frac_limit : reserve_limit;
}

// True when budgeting is active, a </think> id exists, still thinking, and the
// reasoning count reached think_limit (LATER of fraction / reserve-based cap),
// so a larger max_tokens only ever grants more thinking room, never less.
inline bool should_force_think_end(float think_budget, int32_t think_end_id, int max_tokens,
                                   const std::vector<int32_t>& output_tokens, int32_t think_start_id,
                                   bool started_in_think, int answer_reserve = kMaxAnswerReserve) {
    if (!(think_budget > 0.0f) || think_end_id < 0 || output_tokens.empty())
        return false;
    const int limit = think_limit(max_tokens, think_budget, answer_reserve);
    bool currently_thinking = false;
    int n_reasoning = count_reasoning_tokens(output_tokens, think_start_id, think_end_id, started_in_think,
                                             currently_thinking);
    return currently_thinking && n_reasoning >= limit;
}

// Fallback for tokenizers that ship <think>/</think> as added_tokens with
// special=False (Qwen3.6, Qwen3-Coder NVFP4): no single token id, so markers
// arrive split across BPE pieces, matched via a sliding decoded-text window (mirrors track_think_state on Request).
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

// Suppresses a too-eager stop after </think> only until real content appears
// (content_after_think) - NVFP4 noise can close an empty think block in ~3
// tokens then EOS. kMinAnswerAfterThink is a HARD CAP (not a min answer length): grace lifts after that many tokens regardless of content.
inline constexpr int kMinAnswerAfterThink = 16;

// post_decode_step_kernel (cuda_graph.cu) evaluates the same rules on device.
#if defined(__CUDACC__)
#define IMP_THINK_HD __host__ __device__
#else
#define IMP_THINK_HD
#endif

IMP_THINK_HD inline bool grace_blocks_stop(int think_exit_idx, int output_size, bool content_after_think) {
    if (think_exit_idx < 0 || content_after_think)
        return false;
    int tokens_since_exit = output_size - think_exit_idx;
    return tokens_since_exit < kMinAnswerAfterThink;
}

// Masks a stop token before sampling wherever should_stop would suppress it
// (fill_sampling_params, post_decode_step_kernel): a suppressed stop left in
// context can pick <|endoftext|> on a near-tie (AUDIT_qwen38_nvfp4 P3). Bounded: in-think only while budget_can_close, else the kMinAnswerAfterThink grace window.
IMP_THINK_HD inline bool stop_mask_active(bool in_think, bool budget_can_close, int think_exit_idx,
                                          int output_size, bool content_after_think, bool ignore_eos) {
    if (ignore_eos)
        return false;
    if (in_think)
        return budget_can_close;
    return grace_blocks_stop(think_exit_idx, output_size, content_after_think);
}

// Whitespace-only decoded text (e.g. "\n" right after </think>) must NOT count
// as real content or release the grace, or a stop right after it yields a
// 0-content completion (#798). Only a token with a non-whitespace byte counts.
inline bool piece_is_whitespace(const std::string& piece) {
    return piece.find_first_not_of(" \t\n\r\f\v") == std::string::npos;
}

}  // namespace imp::think_logic
