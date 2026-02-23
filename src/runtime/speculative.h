#pragma once

#include "model/model.h"
#include "memory/kv_cache.h"
#include "memory/kv_cache_manager.h"
#include "graph/executor.h"
#include <memory>
#include <vector>
#include <cstdint>

namespace imp {

struct SpeculativeConfig {
    int spec_k = 4;                    // number of draft tokens to generate
    float acceptance_threshold = 0.0f; // 0 = standard stochastic acceptance
};

// Speculative decoding: use a small "draft" model to generate K candidate
// tokens, then verify them in a single forward pass of the "target" model.
// Accepted tokens are emitted immediately; rejected tokens cause a rollback.
class SpeculativeDecoder {
public:
    SpeculativeDecoder() = default;
    ~SpeculativeDecoder() = default;

    // Initialize with target and draft models.
    // target_executor: the main model's executor (already initialized)
    // draft_model: the small draft model
    // draft_executor: executor for the draft model
    bool init(GraphExecutor* target_executor,
              std::shared_ptr<Model> draft_model,
              std::unique_ptr<GraphExecutor> draft_executor,
              KVCacheManager* target_kv_manager,
              KVCacheManager* draft_kv_manager,
              const SpeculativeConfig& config = {});

    // Generate draft tokens from the draft model.
    // context_tokens: current sequence context (including any previously generated tokens)
    // last_token: the most recent token (input to draft model)
    // positions: starting position for the draft tokens
    // Returns: vector of K draft tokens
    std::vector<int32_t> draft_tokens(int32_t last_token, int position,
                                       int seq_id,
                                       cudaStream_t stream = nullptr);

    // Verify draft tokens using the target model.
    // Runs a pseudo-prefill of (K+1) tokens through the target model.
    // Returns: the number of accepted tokens (0 to K), and the corrected
    //          next token (either from accepted drafts or resampled).
    struct VerifyResult {
        int n_accepted = 0;              // how many draft tokens were accepted
        std::vector<int32_t> accepted;   // accepted token ids
        int32_t next_token = -1;         // the next token after accepted ones
    };

    VerifyResult verify(const std::vector<int32_t>& draft,
                        int32_t prompt_last_token,
                        int position,
                        int seq_id,
                        float temperature, float top_p, int top_k,
                        int seed,
                        cudaStream_t stream = nullptr);

    // Full speculative decode step: draft -> verify -> accept/reject.
    // Returns all newly generated tokens (1 to K+1).
    std::vector<int32_t> step(int32_t last_token, int position,
                               int seq_id,
                               float temperature, float top_p, int top_k,
                               int seed,
                               cudaStream_t stream = nullptr);

    int spec_k() const { return config_.spec_k; }

private:
    SpeculativeConfig config_;
    GraphExecutor* target_executor_ = nullptr;
    std::shared_ptr<Model> draft_model_;
    std::unique_ptr<GraphExecutor> draft_executor_;
    KVCacheManager* target_kv_manager_ = nullptr;
    KVCacheManager* draft_kv_manager_ = nullptr;
    bool initialized_ = false;

    // Stochastic acceptance for non-greedy sampling.
    // Accept draft token with probability min(1, p_target / p_draft).
    bool stochastic_accept(float p_target, float p_draft, unsigned int& rng_state);
};

} // namespace imp
