#pragma once

#include "model/model.h"
#include "memory/kv_cache.h"
#include "memory/kv_cache_manager.h"
#include "graph/executor.h"
#include <memory>
#include <vector>
#include <string>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace imp {

// EAGLE-3 head weights (all FP16 on GPU after loading).
struct EagleHead {
    // Token embedding (shared with target if dimensions match)
    Tensor embed_tokens;  // [eagle_vocab, d_model]
    GGMLQuantType embed_qtype = GGMLQuantType::NONE;  // quantization type
    bool embed_shared = false;  // true if using target's tok_emb_

    // Feature fusion: FC [3*d_model → d_model] + bias
    Tensor fc_weight;  // [d_model, 3*d_model]
    Tensor fc_bias;    // [d_model] (optional)

    // Single transformer layer
    Tensor input_norm;      // [d_model]
    Tensor hidden_norm;     // [d_model] (optional, EAGLE-3 v3)
    Tensor wq, wk, wv, wo; // attention projections (may take 2*d_model input)
    Tensor q_bias, k_bias, v_bias;  // optional biases (Qwen3)
    Tensor post_attn_norm;  // [d_model]
    Tensor gate_proj, up_proj, down_proj;  // MLP

    // EAGLE LM head (may differ from target vocab — uses own draft vocab)
    Tensor lm_head;    // [eagle_vocab, d_model] (null if shared with target)
    Tensor lm_norm;    // [d_model] norm before lm_head (null if using target's)

    // Vocab mapping: EAGLE draft vocab ↔ target vocab
    int32_t* d_d2t = nullptr;  // [eagle_vocab] draft→target mapping (INT32 indices)
    int32_t* d_t2d = nullptr;  // [target_vocab] target→draft mapping (INT32 indices)
    bool has_own_vocab = false;  // true if eagle_vocab != target_vocab

    // Dimensions (inferred from weight shapes)
    int d_model = 0;
    int attn_in_dim = 0;  // input dim for attention (may be 2*d_model for EAGLE-3)
    int n_heads = 0;
    int n_kv_heads = 0;
    int head_dim = 0;
    int d_ff = 0;
    int vocab_size = 0;       // EAGLE vocab (may differ from target)
    int target_vocab = 0;     // target model vocab
    float rms_norm_eps = 1e-5f;
};

struct EagleConfig {
    int spec_k = 4;             // draft tokens per step
    int feat_layer_low = -1;    // -1 = auto (layer 0)
    int feat_layer_mid = -1;    // -1 = auto (n_layers/2)
    int feat_layer_high = -1;   // -1 = auto (n_layers-1)
};

class EagleDecoder {
public:
    EagleDecoder() = default;
    ~EagleDecoder();

    // Initialize with target model references and config.
    // Does NOT load the EAGLE head — call load_head() separately.
    bool init(GraphExecutor* target_executor, Model* target_model,
              KVCacheManager* target_kv_manager, KVCache* target_kv_cache,
              const EagleConfig& config, cudaStream_t stream);

    // Load EAGLE head weights from a SafeTensors file.
    bool load_head(const std::string& safetensors_path, cudaStream_t stream);

    // Full speculative step: draft K tokens → verify with target → accept/reject.
    // Returns all accepted tokens + the corrected next token (1 to K+1 tokens).
    std::vector<int32_t> step(int32_t last_token, int position, int seq_id,
                               float temperature, float top_p, int top_k, int seed,
                               cudaStream_t stream);

    int spec_k() const { return config_.spec_k; }
    bool is_initialized() const { return initialized_; }

    // Statistics
    int64_t total_drafted() const { return total_drafted_; }
    int64_t total_accepted() const { return total_accepted_; }
    float acceptance_rate() const {
        return total_drafted_ > 0
            ? static_cast<float>(total_accepted_) / total_drafted_ : 0.0f;
    }

    // Enable feature snapshots in the target executor.
    // Must be called AFTER executor is initialized.
    void enable_snapshots();
    void disable_snapshots();

private:
    // Run EAGLE head forward once: FC fusion + 1 transformer layer.
    // Writes result to d_eagle_hidden_.
    void eagle_forward(int32_t token, int position, cudaStream_t stream);

    // Run one target decode step to capture correct feature snapshots.
    void refresh_features(int32_t token, int position, int seq_id, cudaStream_t stream);

    // Draft K tokens using the EAGLE head.
    std::vector<int32_t> draft_tokens(int32_t last_token, int position,
                                       int seq_id, cudaStream_t stream);

    // Verify draft tokens using the target model (identical to SpeculativeDecoder::verify).
    struct VerifyResult {
        int n_accepted = 0;
        std::vector<int32_t> accepted;
        int32_t next_token = -1;
    };

    VerifyResult verify(const std::vector<int32_t>& draft, int32_t last_token,
                        int position, int seq_id,
                        float temperature, float top_p, int top_k, int seed,
                        cudaStream_t stream);

    EagleHead head_;
    EagleConfig config_;
    bool initialized_ = false;

    // Target model references (non-owning)
    GraphExecutor* target_executor_ = nullptr;
    Model* target_model_ = nullptr;
    KVCacheManager* target_kv_manager_ = nullptr;
    KVCache* target_kv_cache_ = nullptr;

    // EAGLE's own KV cache (1 layer, owned by eagle_kv_manager_)
    KVCache* eagle_kv_ = nullptr;
    std::unique_ptr<KVCacheManager> eagle_kv_manager_;

    // Feature snapshot buffers [3][d_model] — filled by target forward pass
    half* d_feat_bufs_[3] = {};  // low, mid, high
    half* eagle_feat_ptrs_[3] = {};  // same as d_feat_bufs_, passed to executor

    // Workspace buffers
    half* d_concat_ = nullptr;         // [3*d_model] concat features buffer
    half* d_fused_ = nullptr;          // [d_model] FC fusion output
    half* d_eagle_hidden_ = nullptr;   // [d_model] EAGLE hidden state
    half* d_embed_out_ = nullptr;      // [d_model] embedding output
    half* d_norm_out_ = nullptr;       // [d_model] RMSNorm output
    half* d_attn_in_ = nullptr;        // [attn_in_dim] attention input (may be 2*d_model)
    half* d_hidden_normed_ = nullptr;  // [d_model] hidden_norm output

    // Attention workspace
    half* d_q_ = nullptr;
    half* d_k_ = nullptr;
    half* d_v_ = nullptr;
    half* d_attn_out_ = nullptr;
    half* d_proj_out_ = nullptr;

    // MLP workspace
    half* d_gate_out_ = nullptr;
    half* d_up_out_ = nullptr;
    half* d_swiglu_out_ = nullptr;
    half* d_ffn_out_ = nullptr;

    // LM head output (FP16 from GEMV, then cast to FP32 for sampling)
    half* d_eagle_logits_fp16_ = nullptr;  // [vocab_size]
    float* d_eagle_logits_ = nullptr;      // [vocab_size]

    // Pre-allocated device buffers for eagle_forward (avoid per-call cudaMalloc)
    int32_t* d_token_buf_ = nullptr;   // [1] for embedding lookup
    int* d_pos_buf_ = nullptr;         // [1] for position
    int* d_bt_buf_ = nullptr;          // [max_eagle_blocks] for block table
    int* d_cl_buf_ = nullptr;          // [1] for context length
    int max_eagle_blocks_ = 0;         // allocated size of d_bt_buf_
    int32_t* d_argmax_scratch_ = nullptr;  // [ARGMAX_SCRATCH_BYTES] for sample_greedy

    // Pre-allocated device buffers for verify (avoid per-call cudaMalloc)
    int32_t* d_verify_tokens_ = nullptr;  // [max_verify]
    int* d_verify_positions_ = nullptr;   // [max_verify]
    int* d_verify_block_table_ = nullptr; // [max_verify * max_target_blocks]
    int* d_verify_ctx_lens_ = nullptr;    // [max_verify]
    int max_verify_alloc_ = 0;            // allocated n_verify capacity
    int max_verify_bt_alloc_ = 0;         // allocated block table capacity

    // Statistics
    int64_t total_drafted_ = 0;
    int64_t total_accepted_ = 0;

    // GPU allocations to free on destruction
    std::vector<void*> gpu_allocs_;

    void free_buffers();
    void* gpu_alloc(size_t bytes);  // allocate + track

    // Feature layer indices (resolved from config)
    int feat_low_ = 0;
    int feat_mid_ = 0;
    int feat_high_ = 0;

    // Host-side d2t mapping for fast CPU lookup (avoids sync cudaMemcpy per draft)
    std::vector<int32_t> h_d2t_;
};

} // namespace imp
