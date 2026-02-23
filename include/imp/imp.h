#pragma once

#include "imp/types.h"
#include "imp/config.h"
#include "imp/error.h"

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handles
typedef struct ImpModel_T* ImpModel;
typedef struct ImpContext_T* ImpContext;

// --- Model Loading ---

ImpError imp_model_load(
    const char* path,
    ImpModelFormat format,
    ImpModel* out_model
);

void imp_model_free(ImpModel model);

// Query model metadata
ImpModelArch imp_model_arch(ImpModel model);
int imp_model_n_layers(ImpModel model);
int imp_model_d_model(ImpModel model);
int imp_model_vocab_size(ImpModel model);
int imp_model_max_seq_len(ImpModel model);

// --- Context / Runtime ---

ImpError imp_context_create(
    ImpModel model,
    const ImpConfig* config,
    ImpContext* out_ctx
);

void imp_context_free(ImpContext ctx);

// --- Generation ---

typedef struct {
    float temperature;
    float top_p;
    int top_k;
    int max_tokens;
    int seed;                      // -1 = random
} ImpGenerateParams;

ImpGenerateParams imp_generate_params_default(void);

// Synchronous generation from a text prompt
ImpError imp_generate(
    ImpContext ctx,
    const char* prompt,
    const ImpGenerateParams* params,
    char* output_buf,
    size_t output_buf_size,
    size_t* output_len
);

// Token-level generation
ImpError imp_tokenize(
    ImpModel model,
    const char* text,
    int32_t* tokens,
    int* n_tokens,
    int max_tokens
);

ImpError imp_detokenize(
    ImpModel model,
    const int32_t* tokens,
    int n_tokens,
    char* output_buf,
    size_t output_buf_size
);

// Prefill: process input tokens, populate KV cache
ImpError imp_prefill(
    ImpContext ctx,
    const int32_t* tokens,
    int n_tokens
);

// Decode: generate one token
ImpError imp_decode_step(
    ImpContext ctx,
    const ImpGenerateParams* params,
    int32_t* out_token
);

// Reset context state (clear KV cache etc.)
ImpError imp_context_reset(ImpContext ctx);

// --- Speculative Decoding ---

// Set the draft model for speculative decoding. Must be called after
// imp_context_create and before imp_generate or imp_decode_step.
// The draft model should be a smaller/faster variant of the same architecture.
ImpError imp_set_draft_model(
    ImpContext ctx,
    const char* draft_model_path,
    ImpModelFormat format
);

// --- Version ---
const char* imp_version(void);

#ifdef __cplusplus
}
#endif
