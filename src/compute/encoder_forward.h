#pragma once
// =============================================================================
// encoder_forward.h — encoder-only embedder forward (#836, nomic-bert)
// =============================================================================
//
// Self-contained bidirectional encoder pass (post-LN BERT variant with rotary
// positions and SwiGLU FFN — the nomic-embed recipe). Deliberately independent
// of GraphExecutor: no KV cache, no CUDA graphs, no sampling. Reuses the
// shared primitives (embedding_lookup, gemm, rope_forward,
// attention_cublas_prefill, swiglu, layernorm_residual).
//
// Per input:
//   h = LayerNorm(tok_emb[t] + token_types[0])           (post-embedding LN)
//   12 x [ q,k,v = h @ Wq/Wk/Wv ; rope(q,k) ;
//          a = softmax(qk^T)v (bidirectional) @ Wo ;
//          h = LayerNorm(h + a) ;                        (attn_output_norm)
//          f = swiglu(h@Wg, h@Wu) @ Wd ;
//          h = LayerNorm(h + f) ]                        (layer_output_norm)
//   out = L2-normalize(mean(h, axis=0))
//
// Weights arrive Q8_0 from GGUF; encoder_workspace_init dequantizes them ONCE
// to FP16 side buffers (~230 MB for nomic's 137M params) so the forward is
// plain FP16 GEMMs.
// =============================================================================

#include "core/tensor.h"
#include <cuda_runtime.h>
#include <vector>
#include <span>

namespace imp {

class Model;

struct EncoderLayerWeights {
    void* wq = nullptr;  // [d, d] FP16
    void* wk = nullptr;
    void* wv = nullptr;
    void* wo = nullptr;
    void* wg = nullptr;  // [d_ff, d]
    void* wu = nullptr;  // [d_ff, d]
    void* wd = nullptr;  // [d, d_ff]
};

struct EncoderWorkspace {
    int max_tokens = 0;
    int d_model = 0;
    int d_ff = 0;
    int n_layers = 0;
    int n_heads = 0;
    int head_dim = 0;
    float rope_theta = 10000.0f;
    float ln_eps = 1e-12f;

    // Dequantized FP16 weights (owned).
    std::vector<EncoderLayerWeights> layers;

    // Activation scratch (owned), sized for max_tokens.
    int32_t* d_tokens = nullptr;     // [max_n]
    int32_t* d_positions = nullptr;  // [max_n]
    void* d_h = nullptr;             // [max_n, d] FP16 hidden
    void* d_q = nullptr;             // [max_n, d]
    void* d_k = nullptr;             // [max_n, d]
    void* d_v = nullptr;             // [max_n, d]
    void* d_attn = nullptr;          // [max_n, d] attention out
    void* d_proj = nullptr;          // [max_n, d] o/down proj out
    void* d_gate = nullptr;          // [max_n, d_ff]
    void* d_up = nullptr;            // [max_n, d_ff]
    void* d_act = nullptr;           // [max_n, d_ff]
    void* d_scores = nullptr;        // [n_heads, max_n, max_n] FP32 S-matrix
    float* d_pooled = nullptr;       // [d] pooled + normalized output
};

// Dequantize weights + allocate scratch. Call once after upload_weights_gpu.
bool encoder_workspace_init(EncoderWorkspace& ws, const Model& model, int max_tokens,
                            cudaStream_t stream);
void encoder_workspace_free(EncoderWorkspace& ws);

// Full pass: host tokens -> pooled, L2-normalized embedding (host float[d]).
// Returns false on precondition violation (n > max_tokens, ws not init).
bool encoder_embed(const Model& model, EncoderWorkspace& ws, std::span<const int32_t> tokens, float* out_host,
                   cudaStream_t stream);

}  // namespace imp
