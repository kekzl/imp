#pragma once

#include "imp/types.h"
#include "imp/config.h"
#include "imp/error.h"

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Thread-safety contract:
 *
 * - ImpModel handles are read-only after creation and safe to share
 *   across threads.
 * - ImpContext handles are NOT thread-safe. Each context must be used
 *   from a single thread at a time. Create one context per thread for
 *   concurrent inference.
 * - imp_model_load() and imp_model_free() are NOT thread-safe with
 *   respect to the same model handle.
 * - Global state (CUDA device, cuBLAS handles) is initialized once
 *   during the first imp_context_create() call.
 */

// Opaque handles
typedef struct ImpModel_T* ImpModel;
typedef struct ImpContext_T* ImpContext;

// --- Model Loading ---

ImpError imp_model_load(const char* path, ImpModelFormat format, ImpModel* out_model);

// Like imp_model_load, but lets the caller opt into loading the MTP
// (multi-token-prediction) head sidecar for SafeTensors models that ship one
// (DeepSeek-V3 family, e.g. Qwen3.6). The head is ~1.57 GiB BF16 of VRAM and is
// only useful when MTP spec-decode is subsequently enabled via
// imp_enable_mtp_spec_decode. imp_model_load() is equivalent to passing
// load_mtp_head=0. Has no effect for GGUF or models without an MTP sidecar.
ImpError imp_model_load_ex(const char* path, ImpModelFormat format, int load_mtp_head,
                           ImpModel* out_model);

void imp_model_free(ImpModel model);

// Query model metadata
ImpModelArch imp_model_arch(ImpModel model);
int imp_model_n_layers(ImpModel model);
int imp_model_d_model(ImpModel model);
int imp_model_vocab_size(ImpModel model);
int imp_model_max_seq_len(ImpModel model);
/* BOS token id when the model's tokenizer prepends one (add_bos), else -1.
 * Teacher-forced eval (perplexity) must prepend it for BOS-dependent
 * families (Gemma, Llama) or the NLL measures an out-of-distribution
 * sequence. */
int32_t imp_model_bos_token(ImpModel model);

/* --- LoRA adapters (runtime low-rank deltas, no weight patching) ---
 * imp_lora_load: load a HuggingFace PEFT adapter directory (or a bare
 * adapter_model.safetensors). Returns an adapter id >= 1 via out_id.
 * imp_lora_set: activate an adapter (0 = base model). Swapping re-captures
 * decode CUDA graphs on the next request — swap between requests, not
 * mid-generation. Adapters live until the context is freed. */
ImpError imp_lora_load(ImpContext ctx, const char* path, int32_t* out_id);
ImpError imp_lora_set(ImpContext ctx, int32_t adapter_id);

// --- Context / Runtime ---

ImpError imp_context_create(ImpModel model, const ImpConfig* config, ImpContext* out_ctx);

void imp_context_free(ImpContext ctx);

// --- Generation ---

typedef struct {
    float temperature;
    float top_p;
    int top_k;
    int max_tokens;
    int seed;                  // -1 = random
    float min_p;               // min probability threshold (0 = disabled)
    float typical_p;           // locally typical sampling (1.0 = disabled)
    float repetition_penalty;  // >1 penalizes repeats (1.0 = disabled)
    float frequency_penalty;   // subtractive per-occurrence (0 = disabled)
    float presence_penalty;    // subtractive binary (0 = disabled)
    int repeat_last_n;         // penalty window: scan last N tokens (0 = all)
    float dry_multiplier;      // DRY penalty scale (0 = disabled)
    float dry_base;            // DRY exponential base (default 1.75)
    int dry_allowed_length;    // N-gram lengths ≤ this not penalized (default 2)
    int dry_penalty_last_n;    // How far back to scan (0 = all)
    int mirostat;              // 0 = off, 2 = Mirostat v2
    float mirostat_tau;        // target entropy (default 5.0)
    float mirostat_eta;        // learning rate (default 0.1)
    int apply_chat_template;   // 1 = yes (default), 0 = no
    int ignore_eos;            // 1 = don't stop on EOS (benchmark mode)
    int logprobs;              // 1 = return logprobs, 0 = off
    int top_logprobs;          // 0-20, number of top alternatives
    int json_mode;             // 1 = constrain output to valid JSON
} ImpGenerateParams;

ImpGenerateParams imp_generate_params_default(void);

// Callback for streaming token output. Return 0 to continue, non-zero to stop.
typedef int (*ImpTokenCallback)(const char* text, size_t len, void* user_data);

// Synchronous generation with streaming callback. Calls cb for each generated token.
ImpError imp_generate_streaming(ImpContext ctx, const char* prompt, const ImpGenerateParams* params,
                                ImpTokenCallback cb, void* user_data);

// Synchronous generation from a text prompt
ImpError imp_generate(ImpContext ctx, const char* prompt, const ImpGenerateParams* params, char* output_buf,
                      size_t output_buf_size, size_t* output_len);

// Token-level generation
ImpError imp_tokenize(ImpModel model, const char* text, int32_t* tokens, int* n_tokens, int max_tokens);

ImpError imp_detokenize(ImpModel model, const int32_t* tokens, int n_tokens, char* output_buf,
                        size_t output_buf_size);

// Prefill: process input tokens, populate KV cache.
// NOTE: prefill samples the FIRST output token; without `params` the request
// inherits Request struct defaults (top_p=1, top_k=0) and ignores caller
// sampling choices for that first token. Quantized MoE models with noisy
// logit tails (Gemma-4-NVFP4, Qwen3-Coder-NVFP4) can produce a degenerate
// first token under those defaults. Use imp_prefill_with_params instead and
// pass the same params you will hand to imp_decode_step.
ImpError imp_prefill(ImpContext ctx, const int32_t* tokens, int n_tokens);

// Prefill that applies caller-supplied sampling params to the first-token
// sample at end of the last prefill chunk. Strongly preferred over
// imp_prefill for any path that uses non-default temperature/top_p/top_k.
// `params` may be NULL (degrades to imp_prefill semantics).
ImpError imp_prefill_with_params(ImpContext ctx, const int32_t* tokens, int n_tokens,
                                 const ImpGenerateParams* params);

// Decode: generate one token
ImpError imp_decode_step(ImpContext ctx, const ImpGenerateParams* params, int32_t* out_token);

// Teacher-forced perplexity over tokens[0..n_tokens-1] (eval/bench).
// Resets context, runs a SINGLE-CHUNK prefill (so all-position hidden survives),
// applies the LM head to every position, and returns PPL in *out_ppl.
// Requires n_tokens <= the model's max prefill length. *out_ppl < 0 on failure.
ImpError imp_perplexity(ImpContext ctx, const int32_t* tokens, int n_tokens, double* out_ppl);

// Reset context state (clear KV cache etc.)
ImpError imp_context_reset(ImpContext ctx);

// Enable MTP-based speculative decoding (DeepSeek-V3-family models, e.g.
// Qwen3.6 with `model_mtp.safetensors`). k = draft length (1-4 typical).
// Returns IMP_ERROR_INVALID_ARGUMENT if model has no MTP head loaded.
// Phase 3 scaffolding: API in place, auto-invocation from decode loop is
// Phase 4 production work (currently Phase 5 smoke tests use the C++ API).
ImpError imp_enable_mtp_spec_decode(ImpContext ctx, int k);

// --- Vision (Multimodal) ---

// Set an image for the next generation. Must be called after imp_context_create
// and before imp_generate/imp_generate_streaming. Only valid when mmproj was
// loaded during context creation. Pass NULL to clear the image.
ImpError imp_set_image(ImpContext ctx, const char* image_path);

// Set image from raw memory (e.g. decoded base64). Pass NULL/0 to clear.
ImpError imp_set_image_from_memory(ImpContext ctx, const uint8_t* data, size_t len);

// --- Version ---
const char* imp_version(void);

#ifdef __cplusplus
}
#endif
