#pragma once

// Internal helpers shared across engine_*.cpp translation units.
// Not part of any public API; included only by src/runtime/engine*.cpp.
//
// Phase 4 of docs/superpowers/specs/2026-05-20-architecture-refactor-roadmap-design.md

#include "runtime/engine.h"
#include "runtime/request.h"
#include "compute/sampling.h"
#include "model/tokenizer.h"
#include "exec/executor.h"
#include "core/logging.h"

#include <chrono>
#include <cstdint>
#include <functional>

namespace imp::engine_internal {

// Free prefill metadata buffers when not using the pre-allocated pool.
inline void free_prefill_buffers(int32_t* d_token_ids, int* d_positions, int* d_block_tables, int* d_context_lens,
                                 cudaStream_t stream) {
    IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_token_ids, stream));
    IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_positions, stream));
    IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_block_tables, stream));
    IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_context_lens, stream));
}

// Compute a deterministic-but-varying seed for each decode step.
// Mixes the request seed (or a hash of the request ID + clock) with
// the current output token count so each step gets a unique RNG draw.
inline int compute_step_seed(const Request& req) {
    int base_seed = req.seed >= 0
                        ? req.seed
                        : static_cast<int>(std::hash<int>{}(req.id) ^
                                           std::chrono::steady_clock::now().time_since_epoch().count());
    int step = static_cast<int>(req.output_tokens.size());
    return base_seed + step;
}

// Build a TokenLogprobInfo from raw logits on the host.
inline TokenLogprobInfo build_logprob_info(const float* h_logits, int vocab_size, int32_t sampled_token,
                                           int top_logprobs, Tokenizer* tok) {
    LogprobResult lp_result;
    compute_logprobs_cpu(h_logits, vocab_size, sampled_token, top_logprobs, &lp_result);

    TokenLogprobInfo info;
    info.logprob = lp_result.sampled_logprob;
    info.text = tok->decode_token(sampled_token);
    info.top.reserve(lp_result.top.size());
    for (const auto& [tid, tlp] : lp_result.top) {
        info.top.push_back({tid, tlp, tok->decode_token(tid)});
    }
    return info;
}

// Ensure workspace 0 is active (used before prefill and after decode).
inline void ensure_prefill_workspace(GraphExecutor* executor) {
    if (executor->has_decode_workspace() && executor->active_workspace() != 0) {
        executor->use_workspace(0);
    }
}

}  // namespace imp::engine_internal
