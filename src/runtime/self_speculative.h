#pragma once

#include "graph/executor.h"
#include "runtime/cuda_graph.h"
#include "memory/kv_cache.h"
#include "memory/kv_cache_manager.h"
#include <vector>
#include <cstdint>

namespace imp {

struct SelfSpecConfig {
    int spec_k = 4;           // number of draft tokens per step
    int exit_layer = -1;      // -1 = n_layers/2 (early exit mode)
    bool layer_skip = true;   // use layer skipping instead of early exit
    int skip_n = -1;          // layers to skip (-1 = n_layers/2, centered)
};

// Self-speculative decoding: uses early exit (first N layers) of the same
// model as a cheap draft, then verifies with all layers token-by-token.
// No separate draft model needed — shares executor and KV cache.
class SelfSpeculativeDecoder {
public:
    SelfSpeculativeDecoder() = default;
    ~SelfSpeculativeDecoder();

    bool init(GraphExecutor* executor, KVCacheManager* kv_manager,
              KVCache* kv_cache, int n_layers, const SelfSpecConfig& config = {});

    // Full step: draft K tokens with early exit, verify with full model.
    // Returns 1..K+1 accepted tokens.
    std::vector<int32_t> step(int32_t last_token, int position, int seq_id,
                               float temperature, float top_p, int top_k,
                               int seed, cudaStream_t stream = nullptr);

    int spec_k() const { return config_.spec_k; }
    int exit_layer() const { return exit_layer_; }

    // Stats
    int64_t total_drafted() const { return total_drafted_; }
    int64_t total_accepted() const { return total_accepted_; }

private:
    SelfSpecConfig config_;
    GraphExecutor* executor_ = nullptr;
    KVCacheManager* kv_manager_ = nullptr;
    KVCache* kv_cache_ = nullptr;
    int exit_layer_ = 0;         // early exit (when not using layer skip)
    int skip_start_ = -1;       // layer skip range [start, end)
    int skip_end_ = -1;
    int n_layers_ = 0;
    bool initialized_ = false;

    // Pre-allocated device buffers for up to K+1 tokens
    int32_t* d_tokens_ = nullptr;    // [max_n]
    int* d_positions_ = nullptr;     // [max_n]
    int* d_block_table_ = nullptr;   // [max_blocks] (single-seq) or [max_n * max_blocks_per_seq] (replicated)
    int d_block_table_cap_ = 0;
    int* d_ctx_len_ = nullptr;       // [max_n]

    // CUDA graph for draft forward pass (batch_size=1 decode with layer skip)
    CudaGraphRunner draft_graph_;
    int32_t* h_draft_sample_ = nullptr;   // mapped pinned memory for draft token readback
    int32_t* d_draft_sample_ = nullptr;   // device pointer to mapped pinned memory
    int draft_graph_max_blocks_ = -1;     // max_blocks_per_seq used for current graph

    // Stats
    int64_t total_drafted_ = 0;
    int64_t total_accepted_ = 0;

    void upload_block_table(int seq_id, cudaStream_t stream);
    void upload_block_table_replicated(int seq_id, int n_copies, int max_blocks_per_seq, cudaStream_t stream);
    void ensure_kv_blocks(int seq_id, int ctx_len);

    std::vector<int32_t> draft_tokens(int32_t last_token, int position,
                                       int seq_id, cudaStream_t stream);

    struct VerifyResult {
        int n_accepted = 0;
        std::vector<int32_t> accepted;
        int32_t next_token = -1;
    };

    VerifyResult verify(const std::vector<int32_t>& draft,
                        int32_t last_token, int position, int seq_id,
                        float temperature, float top_p, int top_k,
                        int seed, cudaStream_t stream);
};

} // namespace imp
