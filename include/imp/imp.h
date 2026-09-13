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
 *   from a single thread at a time.
 * - ONE context per PROCESS. The engine arena and the graph-slot pool are
 *   process-global; a second live context shares them and the first
 *   imp_context_free() releases them under the other one. A second
 *   imp_context_create() is refused with IMP_ERROR_INVALID_ARG rather than
 *   returning a handle that breaks later (#1629). Sequential
 *   create/free/create IS supported. For concurrent inference use the
 *   batching the engine already does, or a second process.
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

// Opts into loading the MTP head sidecar for SafeTensors models that ship one (DeepSeek-V3
// family, e.g. Qwen3.6): ~1.57 GiB BF16, useful only with imp_enable_mtp_spec_decode.
// imp_model_load() == load_mtp_head=0. No effect for GGUF or models without an MTP sidecar.
ImpError imp_model_load_ex(const char* path, ImpModelFormat format, int load_mtp_head,
                           ImpModel* out_model);

void imp_model_free(ImpModel model);

// Query model metadata
ImpModelArch imp_model_arch(ImpModel model);
int imp_model_n_layers(ImpModel model);
int imp_model_d_model(ImpModel model);
int imp_model_vocab_size(ImpModel model);
int imp_model_max_seq_len(ImpModel model);
/* Effective context window the engine actually allocated for this context
 * (VRAM-aware auto-sizing). May be smaller than imp_model_max_seq_len when VRAM
 * is tight; gate prompt length on this to reject over-long prompts cleanly. */
int imp_context_max_seq_len(ImpContext ctx);
/* BOS token id when the model's tokenizer prepends one (add_bos), else -1.
 * Teacher-forced eval (perplexity) must prepend it for BOS-dependent
 * families (Gemma, Llama) or the NLL measures an out-of-distribution
 * sequence. */
int32_t imp_model_bos_token(ImpModel model);

/* --- LoRA adapters (runtime low-rank deltas, no weight patching) ---
 * imp_lora_load: load a HuggingFace PEFT adapter directory (or a bare
 * adapter_model.safetensors). Returns an adapter id >= 1 via out_id. The
 * adapter's shapes are checked against the model; a mismatch answers
 * IMP_ERROR_INVALID_MODEL.
 * imp_lora_set: activate an adapter (0 = base model). Swapping re-captures
 * decode CUDA graphs on the next request: swap between requests, not
 * mid-generation. The prefix cache is keyed by the active adapter. Adapters
 * live until the context is freed. */
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

// Prefill samples the FIRST output token; without params it uses Request defaults (top_p=1,
// top_k=0), ignoring caller sampling. Quantized MoE models with noisy logit tails (Gemma-4-NVFP4,
// Qwen3-Coder-NVFP4) can degenerate on that token. Use imp_prefill_with_params instead.
ImpError imp_prefill(ImpContext ctx, const int32_t* tokens, int n_tokens);

// Applies caller sampling params to the first-token sample at end of the last prefill chunk.
// Strongly preferred over imp_prefill for non-default temperature/top_p/top_k.
// params may be NULL (degrades to imp_prefill semantics).
ImpError imp_prefill_with_params(ImpContext ctx, const int32_t* tokens, int n_tokens,
                                 const ImpGenerateParams* params);

// Returns IMP_ERROR_CANCELLED when the engine cancels mid-decode (e.g. KV pool exhausted,
// reject-newest; engine log names cause+remedy). IMP_ERROR_INTERNAL after natural FINISH is
// the end-of-stream signal for callers that keep stepping.
ImpError imp_decode_step(ImpContext ctx, const ImpGenerateParams* params, int32_t* out_token);

// Teacher-forced perplexity over tokens[0..n_tokens-1]: resets context, runs a SINGLE-CHUNK
// prefill (all-position hidden survives), applies the LM head to every position.
// Requires n_tokens <= the model's max prefill length; *out_ppl < 0 on failure.
ImpError imp_perplexity(ImpContext ctx, const int32_t* tokens, int n_tokens, double* out_ppl);

// Writes activation-calibration stats collected so far to `path` ([calibration] enabled),
// input to imp-quantize's AWQ scale search.
// IMP_ERROR_INVALID_ARG when calibration was never enabled or no forward pass ran.
ImpError imp_calibration_write(ImpContext ctx, const char* path);

// Reset context state (clear KV cache etc.)
ImpError imp_context_reset(ImpContext ctx);

// MTP-based speculative decoding (DeepSeek-V3-family, e.g. Qwen3.6 model_mtp.safetensors).
// k = draft length (1-4 typical). IMP_ERROR_INVALID_ARGUMENT if no MTP head loaded.
// API in place; auto-invocation from the decode loop is not wired (C++ API only today).
ImpError imp_enable_mtp_spec_decode(ImpContext ctx, int k);

// --- Vision (Multimodal) ---

// Set an image for the next generation. Must be called after imp_context_create
// and before imp_generate/imp_generate_streaming. Only valid when mmproj was
// loaded during context creation. Pass NULL to clear the image.
ImpError imp_set_image(ImpContext ctx, const char* image_path);

// Set image from raw memory (e.g. decoded base64). Pass NULL/0 to clear.
ImpError imp_set_image_from_memory(ImpContext ctx, const uint8_t* data, size_t len);

// Appends an image instead of replacing the pending one, so a prompt can carry several;
// consumed in order added, one per placeholder.
// IMP_ERROR_UNSUPPORTED on single-image models (mmproj path): refused, not silently truncated.
ImpError imp_add_image(ImpContext ctx, const char* image_path);
ImpError imp_add_image_from_memory(ImpContext ctx, const uint8_t* data, size_t len);

// Number of image tokens the pending image expands to, 0 if none. Dynamic-resolution
// encoders (Qwen3-VL) only know this after the image is set.
// Call between imp_set_image and tokenizing so the prompt reserves exactly this many placeholders.
int imp_pending_image_tokens(ImpContext ctx);

// Suspend to RAM (see imp-server /admin/suspend, /admin/resume): capture D2H-copies weight
// buffers to host RAM (imp_weights_snapshot_capture(model, headroom_mb, &snap); engine stays
// alive, failure tears down nothing); imp_context_free+imp_model_free free VRAM; imp_gpu_release(1)
// trims pools/resets the device. imp_weights_snapshot_arm(snap) + reload the SAME model file
// restores buffer bytes (a mismatch falls back to cold path); free with imp_weights_snapshot_free.
// Capture: UNSUPPORTED for weights transformed post-upload (native MXFP4 GGUF, gpt-oss, Gemma-4
// fused-expert split); OUT_OF_MEMORY if host MemAvailable < snapshot bytes + headroom_mb.
typedef struct ImpWeightSnapshot_T* ImpWeightSnapshot;

ImpError imp_weights_snapshot_capture(ImpModel model, size_t host_ram_headroom_mb,
                                      ImpWeightSnapshot* out_snap);
// Arm for the NEXT model load in this process (non-consuming: the snapshot
// stays owned by the caller and can be re-armed after a failed resume).
ImpError imp_weights_snapshot_arm(ImpWeightSnapshot snap);
void imp_weights_snapshot_free(ImpWeightSnapshot snap);
size_t imp_weights_snapshot_bytes(ImpWeightSnapshot snap);
// Number of uploads restored from this snapshot by the last armed consume.
int imp_weights_snapshot_hits(ImpWeightSnapshot snap);

// Releases process-held GPU resources after model/context teardown: syncs, trims the async
// mempool, and (device_reset != 0) resets the CUDA primary context so nvidia-smi shows ~0 MiB.
// After a reset, the next imp_model_load/imp_context_create reinitializes CUDA from scratch.
ImpError imp_gpu_release(int device_reset);

// --- Version ---
const char* imp_version(void);

#ifdef __cplusplus
}
#endif
