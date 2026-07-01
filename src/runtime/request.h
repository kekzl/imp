#pragma once

#include <vector>
#include <cstdint>
#include <string>
#include <memory>
#include "model/chat_template.h"

namespace imp {

// Forward declares to keep CUDA / buffer headers out of this widely-included file.
struct ImageData;  // src/vision/image_processor.h
class Buffer;       // src/core/buffer.h

// Per-token log probability information (for logprobs output)
struct TokenLogprob {
    int32_t token;
    float logprob;
    std::string text;
};

struct TokenLogprobInfo {
    float logprob{};                // logprob of the sampled token
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
    // n-gram speculation bookkeeping: consecutive draft misses and per-
    // request acceptance economics, plus the sticky give-up flag that hands
    // the request back to the async graph loop once the context proved
    // draft-poor (speculative.give_up_after) or acceptance-poor (structured
    // content whose continuations never match, e.g. number tables).
    int spec_consecutive_misses = 0;
    int spec_verifies = 0;
    long long spec_drafted = 0;
    long long spec_accepted = 0;
    bool spec_ngram_given_up = false;
    int spec_last_giveup_pos = 0;  // output size at last give-up (burst re-arm)
    // Sticky acceptance verdict: structured-but-mutating content (number
    // tables) re-trips the acceptance economics after every re-arm window —
    // once doomed, give-up is final for this request.
    bool spec_acceptance_doomed = false;
    // Per-request n-gram speculation override (tri-state): -1 = use the global
    // speculative.ngram default, 0 = force OFF, 1 = force ON. Lets a code-gen
    // request opt into speculation while a short tool-arg generation skips it
    // (and vice-versa) on the same server. Resolved via Engine::spec_ngram_enabled_.
    int spec_ngram_override = -1;
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
    // Whether a real (non-stop) answer token has been emitted since the last
    // </think>. The post-think grace releases the moment this is true, so a
    // complete short answer stops on its own <|im_end|> instead of being padded
    // or repeated. Reset to false at every think exit.
    bool content_after_think = false;
    float think_budget = 0.0f;  // Fraction of max_tokens for reasoning (0=unlimited)
    // gpt-oss Harmony answer-headroom force: when reasoning hits the budget we
    // can't just force <|end|> (the model re-opens the analysis channel) — we
    // force the whole <|end|><|start|>assistant<|channel|>final<|message|>
    // opener so the model commits to the final (answer) channel. Index into
    // Engine::harmony_force_seq_ for the in-flight forced opener (-1 = idle).
    int harmony_force_idx = -1;
    int prefill_offset = 0;     // Chunked prefill: tokens processed so far
    int cached_tokens = 0;      // Tokens served from prefix cache (skipped in prefill)
    // Pin this request's full prompt blocks in the prefix cache at finish
    // (Anthropic cache_control / OpenAI-route cache_prompt). Pinned blocks
    // survive eviction until the pin budget recycles them (FIFO).
    bool pin_kv_prefix = false;

    // SSE streaming request: the client consumes tokens as they are produced.
    // Streaming runs the async conditional graph loop like everything else:
    // step_async_graph_resume polls the mapped ring buffer (device publishes
    // each token behind a __threadfence_system + separate poll counter) and
    // surfaces tokens per step while the burst is in flight (#754 resolved —
    // the old blocking per-burst sync delivered tokens only in burst-sized
    // groups, which is why streaming used to stay on per-step decode).
    bool stream = false;

    // Logprobs
    bool logprobs = false;                          // Return logprobs for sampled tokens
    int top_logprobs = 0;                           // 0-20, number of top alternatives
    std::vector<TokenLogprobInfo> output_logprobs;  // parallel to output_tokens

    // JSON mode
    bool json_mode = false;   // Constrain output to valid JSON
    std::string json_schema;  // JSON Schema string (empty = disabled)
    // Per-request constraint FSM (JsonConstrainer/SchemaConstrainer wrapper).
    // Owned by the request so concurrent prefills/finishes of OTHER requests
    // cannot clobber the state, and batched decode can mask per row. Checked
    // out of Engine::constraint_pool_ on first need, returned at finish.
    std::shared_ptr<class ConstraintManager> constraints;

    // Tool-call coordination: when true and (json_mode || !json_schema.empty()),
    // the preamble gate enters tool-aware mode so the schema/JSON FSM mask
    // does not block the model's tool-tag opener (`<tool_call>`, `<|tool_call>`,
    // `<function=`).
    bool has_tools = false;
    ChatTemplateFamily tpl_family = ChatTemplateFamily::CHATML;

    // Logit bias: token_id -> bias value, added to logits before sampling
    std::vector<std::pair<int32_t, float>> logit_bias;

    // Vision (multimodal), per-request binding. `image` carries CPU-preprocessed
    // pixels set by the server/CLI; the batch worker encodes it into `vision_emb`
    // (device [n_vision_tokens, d_model] fp16) on admission and clears `image`.
    // step_prefill_one binds vision_emb at offset==0. shared_ptr → auto-freed on
    // the last request reference (no manual lifecycle across cancel paths).
    // Null for text-only requests.
    std::shared_ptr<ImageData> image;
    std::shared_ptr<Buffer> vision_emb;
    int32_t vision_token_id = -1;
    int n_vision_tokens = 0;

    int context_len() const { return static_cast<int>(input_tokens.size() + output_tokens.size()); }
};

}  // namespace imp
