#pragma once

#include <vector>
#include <cstdint>
#include <string>
#include "model/chat_template.h"

namespace imp {

// Per-token log probability information (for logprobs output)
struct TokenLogprob {
    int32_t token;
    float logprob;
    std::string text;
};

struct TokenLogprobInfo {
    float logprob;                  // logprob of the sampled token
    std::string text;               // decoded text of the sampled token
    std::vector<TokenLogprob> top;  // top_logprobs alternatives
};

enum class RequestStatus { PENDING, PREFILLING, DECODING, FINISHED, CANCELLED };

const char* request_status_name(RequestStatus status);

struct Request {
    int id = 0;
    RequestStatus status = RequestStatus::PENDING;

    std::vector<int32_t> input_tokens;
    std::vector<int32_t> output_tokens;

    int max_tokens = 256;
    float temperature = 1.0f;
    float top_p = 1.0f;
    int top_k = 0;
    int seed = -1;
    float min_p = 0.0f;               // Min probability threshold (0 = disabled)
    float typical_p = 1.0f;           // Locally typical sampling (1.0 = disabled)
    float repetition_penalty = 1.0f;  // >1 penalizes repeats (multiplicative)
    float frequency_penalty = 0.0f;   // Subtractive per-occurrence
    float presence_penalty = 0.0f;    // Subtractive binary (appeared or not)
    int repeat_last_n = 0;            // How many recent tokens to scan for penalties (0 = all)
    float dry_multiplier = 0.0f;      // DRY penalty scale (0 = disabled)
    float dry_base = 1.75f;           // DRY exponential base
    int dry_allowed_length = 2;       // N-grams at or below this not penalized
    int dry_penalty_last_n = 0;       // How far back to scan (0 = all)
    int mirostat = 0;                 // 0=off, 2=Mirostat v2
    float mirostat_tau = 5.0f;        // Target entropy
    float mirostat_eta = 0.1f;        // Learning rate
    float mirostat_mu = 0.0f;         // Running variable (persists across tokens, init = 2*tau)
    bool ignore_eos = false;          // Don't stop on EOS (benchmark mode)
    bool in_think_block = false;      // Currently inside <think>...</think> (suppress stop tokens)
    // Generation began inside an injected <think> prefix (the opener lives in
    // the PROMPT, not the output) — seeds the think-budget recount loop and
    // the CUDA-graph decode config so the budget engages from token 0.
    bool started_in_think = false;
    // Sliding-window decoded-text buffer for multi-token </think> detection.
    // Required for tokenizers that ship <think>/</think> as added_tokens with
    // special=False (Qwen3.6, Qwen3-Coder NVFP4 SafeTensors): the model emits
    // </think> as multiple BPE tokens (e.g. ['</', 'think', '>']) so a single-
    // token-id compare in track_think_state can never trip. We accumulate the
    // last ~32 decoded chars and scan for the literal string instead.
    std::string think_text_tail;
    // Output-tokens index at which the think block was last exited, or -1 if
    // we have never been in a think block. Used to enforce a minimum answer
    // budget after </think> on numerically-noisy NVFP4 quants whose model
    // sometimes closes an empty thinking block and would otherwise EOS to
    // a 0-content completion.
    int think_exit_idx = -1;
    float think_budget = 0.0f;  // Fraction of max_tokens for reasoning (0=unlimited)
    int prefill_offset = 0;     // Chunked prefill: tokens processed so far
    int cached_tokens = 0;      // Tokens served from prefix cache (skipped in prefill)
    // Pin this request's full prompt blocks in the prefix cache at finish
    // (Anthropic cache_control / OpenAI-route cache_prompt). Pinned blocks
    // survive eviction until the pin budget recycles them (FIFO).
    bool pin_kv_prefix = false;

    // Logprobs
    bool logprobs = false;                          // Return logprobs for sampled tokens
    int top_logprobs = 0;                           // 0-20, number of top alternatives
    std::vector<TokenLogprobInfo> output_logprobs;  // parallel to output_tokens

    // JSON mode
    bool json_mode = false;   // Constrain output to valid JSON
    std::string json_schema;  // JSON Schema string (empty = disabled)

    // Tool-call coordination: when true and (json_mode || !json_schema.empty()),
    // the preamble gate enters tool-aware mode so the schema/JSON FSM mask
    // does not block the model's tool-tag opener (`<tool_call>`, `<|tool_call>`,
    // `<function=`).
    bool has_tools = false;
    ChatTemplateFamily tpl_family = ChatTemplateFamily::CHATML;

    // Logit bias: token_id -> bias value, added to logits before sampling
    std::vector<std::pair<int32_t, float>> logit_bias;

    int context_len() const { return static_cast<int>(input_tokens.size() + output_tokens.size()); }
};

}  // namespace imp
