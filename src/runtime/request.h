#pragma once

#include <vector>
#include <cstdint>
#include <string>
#include <memory>
#include "model/chat_template.h"

namespace imp {

// Forward declares to keep CUDA / buffer headers out of this widely-included file.
struct ImageData;
struct QwenPatches;  // vision/image_processor.h
class Buffer;      // src/core/buffer.h

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

// Why a CANCELLED request was cancelled. Most cancellations are indistinguishable
// to a caller and stay `None` (client disconnect, engine-internal abort); the one
// that is actionable — the prompt needs more KV blocks than the pool can ever
// hold — gets its own reason so the API can return a typed error instead of a
// generic cancel. Invariant I6.
enum class CancelReason { None, KvCapacity };

const char* request_status_name(RequestStatus status);

struct Request {
    int id = 0;
    // The id the CALLER knows this request by - the server's `imp-N`, which is
    // also what the HTTP response carries. `id` above is the engine's own
    // counter and the two never met, so no engine log line could be attributed
    // to an HTTP request (#1582). Empty for direct API users; add_request()
    // emits the join line when it is set.
    std::string trace_id;
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
    // Token-Recycling feed cursor (speculative.token_recycling): output
    // tokens up to this index are already ingested into the engine's
    // adjacency table; prompt bigrams are fed once when it is 0.
    int spec_recycle_fed = 0;
    // Per-request n-gram speculation override (tri-state): -1 = use the global
    // speculative.ngram default, 0 = force OFF, 1 = force ON. Lets a code-gen
    // request opt into speculation while a short tool-arg generation skips it
    // (and vice-versa) on the same server. Resolved via Engine::spec_ngram_enabled_.
    // Per-request `"speculative": true/false`. -1 = unset, 0 = off, 1 = on.
    // Named for the field, not for one of the drafters: "false" switches off
    // the n-gram matcher AND the MTP head AND token recycling (#1639). "true"
    // enables what the model and the operator config allow; it cannot conjure
    // an MTP head the checkpoint does not carry.
    int spec_override = -1;
    // Admission priority (vLLM semantics: LOWER value schedules earlier,
    // default 0). Read by Scheduler::schedule as the primary sort key of the
    // pending queue; aging and shortest-first order requests WITHIN a
    // priority class only. Strict dominance is deliberate: a caller that
    // sets priorities owns starvation across classes, same as vLLM. From the
    // "priority" body field on all three server dialects.
    int priority = 0;
    // Scheduler round this request was enqueued in. Only the scheduler writes
    // it; it exists so admission can tell "short" from "short AND recent"
    // (#1634).
    uint64_t enqueued_round = 0;
    // OpenAI Predicted Outputs: client-supplied predicted completion tokens.
    // Never forwarded through the model — they only extend the n-gram draft
    // search corpus (input + prediction + output), so a response that tracks
    // the prediction gets max_match-length drafts every verify step. Verify
    // semantics are unchanged: output stays a faithful greedy decode.
    std::vector<int32_t> prediction_tokens;
    // Accept accounting for usage.completion_tokens_details — counts only
    // verify steps whose draft was sourced from the prediction region.
    long long pred_accepted = 0;
    long long pred_rejected = 0;
    bool in_think_block = false;  // Currently inside <think>...</think> (suppress stop tokens)
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
    int prefill_offset = 0;  // Chunked prefill: tokens processed so far
    int cached_tokens = 0;   // Tokens served from prefix cache (skipped in prefill)
    // Context this request LOST to StreamingLLM eviction mid-generation: the
    // KV of the middle of its own conversation was freed to keep decoding.
    // The engine logs a WARN when eviction auto-enables, but a WARN is not
    // something the caller sees, and the answer it gets back was written
    // against less context than it sent. Surfaced as
    // usage.prompt_tokens_details.evicted_tokens (roadmap gap 6).
    int evicted_kv_tokens = 0;
    // Hybrid (SSM/GDN) prefix caching: snapshot of the recurrent state at
    // exactly `cached_tokens` tokens, set at admission when the prompt prefix
    // matches a stored snapshot. Restored into the request's recurrent slot
    // on the first prefill chunk instead of zeroing (engine_sampling_stop).
    std::shared_ptr<const struct RecurrentSnapshotEntry> recurrent_restore;
    // SWA window snapshot (kv_cache.swa_snapshot_mb): packed windowed-layer
    // KV at exactly `cached_tokens`, restored into fresh SWA blocks on the
    // first prefill chunk so a prefix-cache hit is valid under SWA sizing.
    std::shared_ptr<const struct RecurrentSnapshotEntry> swa_restore;
    // Pin this request's full prompt blocks in the prefix cache at finish
    // (Anthropic cache_control / OpenAI-route cache_prompt). Pinned blocks
    // survive eviction until the pin budget recycles them (FIFO).
    bool pin_kv_prefix = false;
    // Pin boundary in tokens (#1046): -1 = whole prompt; >=0 = pin only the
    // first N prompt tokens' full blocks (Anthropic cache_control breakpoint).
    int pin_kv_prefix_tokens = -1;

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

    // Embedding request (#1005): prefill-only. Hidden states are mean-pooled
    // across ALL prefill chunks (device partial sums, host accumulation) and
    // land in `embedding_out`; the request finishes without sampling a single
    // token, so it batches with concurrent decodes instead of pausing them.
    bool embedding_request = false;
    std::vector<float> embedding_out;

    // Rerank scoring (/v1/rerank): prefill only, then read the logits of these
    // token ids at the LAST position into score_out. A cross-encoder reranker
    // scores query+document jointly in one forward and never generates, so
    // this rides the same no-sampling path embeddings take — but it DOES want
    // the prefix cache, because reranking N documents replays one shared
    // system+query prefix N times.
    std::vector<int32_t> score_token_ids;
    std::vector<float> score_out;

    // JSON mode
    bool json_mode = false;   // Constrain output to valid JSON
    std::string json_schema;  // JSON Schema string (empty = disabled)
    // Constrain the whole output to this regular expression (empty = disabled).
    // Mutually exclusive with json_mode/json_schema; the engine prefers it.
    std::string regex_pattern;
    // Constrain the whole output to this GBNF grammar (empty = disabled).
    // Same deal one step up the Chomsky hierarchy: a grammar can balance
    // brackets and nest, which no regex can. Engine precedence is
    // tool-call > regex > grammar > json.
    std::string grammar;
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

    // Enforced tool calling (#1002, tool_choice=required / forced function):
    // when non-empty, generation is constrained to ONE tool-call envelope
    // (`tool_envelope_open` + `{"name":...,"arguments":{...}}` +
    // `tool_envelope_close`) via a TOOL_CALL schema FSM. Each entry is
    // (tool name, parameter-schema JSON). Takes precedence over
    // json_mode/json_schema; falls back to the prompt hint when the schemas
    // are not enforceable (see build_tool_call_schema).
    std::vector<std::pair<std::string, std::string>> tool_constraint_tools;
    std::string tool_envelope_open;
    std::string tool_envelope_close;
    // Strict OPTIONAL tool call (OpenAI `strict: true`, tool_choice=auto): the
    // envelope is NOT forced — the model may answer in text, but IF it opens the
    // tool tag the body FSM enforces the arguments. false = forced (mandatory).
    bool tool_constraint_optional = false;
    // parallel_tool_calls (strict optional only): true = the gate re-arms after
    // each enforced call so the model may emit several; false = at most one.
    bool tool_constraint_parallel = true;
    // Llama3 forced call: the constraint root is the bare parameter schema (the
    // `<function=NAME>{args}</function>` body is the arguments object), not a
    // TOOL_CALL {"name","arguments"} wrapper.
    bool tool_constraint_bare_args = false;
    // Qwen-Coder XML dialect: the body inside <tool_call> is
    // `<function=NAME><parameter=KEY>raw value</parameter>...</function>`
    // (raw-text values) — enforce with the XML_TOOL_CALL grammar instead of
    // the JSON body FSM (which masks raw newlines and mangles code arguments).
    bool tool_constraint_xml = false;

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

    // Qwen3-VL, per-request. `qwen_patches` holds the CPU-preprocessed images
    // set by the server thread, in prompt order; the batch worker encodes them
    // into ONE concatenated `vision_emb` (+ one `deepstack_emb` per tap) on
    // admission and clears them. Per-request rather than engine-global because
    // the server admits image requests concurrently.
    //
    // A vector rather than one image because the kernels address embeddings by
    // "the k-th image token in the prompt", which is already a global index
    // across every picture — concatenating in prompt order is all they need.
    std::vector<std::shared_ptr<QwenPatches>> qwen_patches;
    // Hash of this request's image bytes, seeded into the prefix cache's block
    // chain. Without it the cache matches on token ids alone, and every image
    // token has the same id — so a second request would inherit the first
    // one's picture. Non-zero whenever the request carries an image; a request
    // that has one but reports 0 is refused the cache entirely, so a missed
    // plumbing site degrades to "no reuse" instead of "wrong picture".
    size_t vision_content_hash = 0;
    // `deepstack_emb` holds one buffer per vision tap, each the same
    // shape as `vision_emb`; they are ADDED at the LM's first layers.
    std::vector<std::shared_ptr<Buffer>> deepstack_emb;
    // Per-token (t, h, w) positions for the PROMPT, [3, input_tokens.size()].
    // Empty for a model without M-RoPE.
    std::vector<int32_t> mrope_positions;
    // What to add to a generated token's position to continue the M-RoPE
    // sequence. An image costs max(rows, cols) positions but occupies
    // rows*cols tokens, so this is negative whenever the prompt held one, and
    // `context_len()` alone would leave a gap the model never saw in training.
    int mrope_pos_delta = 0;
    // The same number, on the device, owned by THIS request. Engine-global
    // would be a cross-request corruption: decode replays read the pointer at
    // replay time, so a concurrent request's prefill would overwrite the value
    // a different sequence is mid-generation on.
    std::shared_ptr<Buffer> mrope_delta_dev;

    int context_len() const { return static_cast<int>(input_tokens.size() + output_tokens.size()); }

    // Deliberately LAST rather than next to `status`: Request is touched every
    // decode step, so inserting into the middle shifts every following field
    // and there is no reason to perturb a hot layout for a field only read on
    // error paths. NOT a measured win — an interleaved 4-pair A/B against a
    // rebuilt parent put branch and main within 0.2% either way, and the
    // non-interleaved reading that suggested ~2% was host drift between two
    // measurements 20 minutes apart.
    CancelReason cancel_reason = CancelReason::None;
};

}  // namespace imp
