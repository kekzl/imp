#pragma once

#include "graph/executor.h"
#include "memory/kv_cache_manager.h"
#include "runtime/request.h"
#include <vector>
#include <memory>
#include <cuda_runtime.h>

namespace imp {

// N-gram speculative decoding: zero-cost draft via token history lookup.
// Searches the full token history (input + output) for matching n-grams
// and predicts the continuation. Verify uses a single batched forward pass.
//
// Effective for structured/repetitive output (JSON, tool calls, code)
// where patterns from the prompt recur in the output.
class NgramSpecDecoder {
public:
    NgramSpecDecoder() = default;
    ~NgramSpecDecoder();

    struct Config {
        int spec_k = 5;       // max draft tokens per step
        int ngram_n = 3;      // context window for matching
    };

    bool init(GraphExecutor* executor, KVCacheManager* kv_manager,
              KVCache* kv_cache, int n_layers, int spec_k = 5, int ngram_n = 3);

    struct StepResult {
        std::vector<int32_t> tokens;  // all accepted tokens (including corrected)
        int n_drafted = 0;
        int n_accepted = 0;
    };

    // Full step: draft via n-gram lookup, verify via batched forward pass.
    // Returns accepted tokens. If no n-gram match found, falls back to
    // normal single-token decode.
    StepResult step(std::shared_ptr<Request> req, int32_t last_token,
                    int position, int seq_id, cudaStream_t stream);

    // Statistics
    int64_t total_drafted() const { return total_drafted_; }
    int64_t total_accepted() const { return total_accepted_; }
    int64_t total_steps() const { return total_steps_; }
    float acceptance_rate() const {
        return total_drafted_ > 0 ? static_cast<float>(total_accepted_) / total_drafted_ : 0.0f;
    }

private:
    GraphExecutor* executor_ = nullptr;
    KVCacheManager* kv_manager_ = nullptr;
    KVCache* kv_cache_ = nullptr;
    int n_layers_ = 0;
    Config config_;

    // Pre-allocated device buffers for verify (avoids per-step cudaMalloc)
    int32_t* d_tokens_ = nullptr;
    int* d_positions_ = nullptr;
    int* d_block_table_ = nullptr;
    int* d_ctx_len_ = nullptr;
    int d_block_table_cap_ = 0;

    // Stats
    int64_t total_drafted_ = 0;
    int64_t total_accepted_ = 0;
    int64_t total_steps_ = 0;

    // Draft: search token history for matching n-gram, return continuation
    std::vector<int32_t> draft_tokens(const std::vector<int32_t>& input_tokens,
                                       const std::vector<int32_t>& output_tokens);

    // Verify: batched forward pass with all_logits, CPU acceptance check
    struct VerifyResult {
        int n_accepted;
        int32_t corrected_token;  // sampled from target at first rejection point
    };
    VerifyResult verify(const std::vector<int32_t>& draft,
                        std::shared_ptr<Request> req,
                        int position, int seq_id, cudaStream_t stream);
};

} // namespace imp
