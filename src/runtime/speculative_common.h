#pragma once

// Shared helpers for speculative and self-speculative decoders.

#include "core/logging.h"
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>
#include <vector>
#include <cstdint>

namespace imp {

// ---------------------------------------------------------------------------
// CUDA error checking (throws on failure for speculative.cpp, logs for
// self_speculative.cpp — callers choose which variant to use).
// ---------------------------------------------------------------------------

static inline void spec_check_cuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        IMP_LOG_ERROR("speculative %s: %s", msg, cudaGetErrorString(err));
    }
}

// ---------------------------------------------------------------------------
// CPU-side softmax with temperature scaling.
// Computes softmax in-place over a row of logits.
// ---------------------------------------------------------------------------

static inline void softmax_row(const float* row_logits, int vocab_size,
                                float temperature, std::vector<float>& probs) {
    probs.resize(vocab_size);

    // Find max for numerical stability
    float max_val = row_logits[0];
    for (int v = 1; v < vocab_size; ++v) {
        max_val = std::max(max_val, row_logits[v]);
    }

    // Apply temperature before softmax (greedy uses inv_temp=1)
    bool greedy = (temperature <= 1e-6f);
    float inv_temp = greedy ? 1.0f : (1.0f / temperature);
    float sum = 0.0f;
    for (int v = 0; v < vocab_size; ++v) {
        probs[v] = std::exp((row_logits[v] - max_val) * inv_temp);
        sum += probs[v];
    }
    float inv_sum = 1.0f / (sum + 1e-10f);
    for (int v = 0; v < vocab_size; ++v) {
        probs[v] *= inv_sum;
    }
}

// ---------------------------------------------------------------------------
// Argmax over a probability/logit vector.
// ---------------------------------------------------------------------------

static inline int32_t spec_argmax(const std::vector<float>& probs) {
    return static_cast<int32_t>(
        std::distance(probs.begin(), std::max_element(probs.begin(), probs.end())));
}

// ---------------------------------------------------------------------------
// Weighted random sample from a probability distribution using LCG RNG.
// ---------------------------------------------------------------------------

static inline int32_t spec_sample_from(const std::vector<float>& probs,
                                        unsigned int& rng_state) {
    // LCG step for random float in [0, 1)
    rng_state = rng_state * 1664525u + 1013904223u;
    float r = static_cast<float>(rng_state & 0x00FFFFFFu) /
              static_cast<float>(0x01000000u);

    float cumsum = 0.0f;
    for (int v = 0; v < static_cast<int>(probs.size()); ++v) {
        cumsum += probs[v];
        if (cumsum >= r) {
            return static_cast<int32_t>(v);
        }
    }
    return static_cast<int32_t>(probs.size() - 1);
}

// ---------------------------------------------------------------------------
// Stochastic acceptance: accept draft token with prob min(1, p_target/p_draft).
// ---------------------------------------------------------------------------

static inline bool spec_stochastic_accept(float p_target, float p_draft,
                                           unsigned int& rng_state) {
    if (p_draft <= 0.0f) return false;

    float ratio = p_target / p_draft;
    if (ratio >= 1.0f) return true;

    // LCG random number in [0, 1)
    rng_state = rng_state * 1664525u + 1013904223u;
    float r = static_cast<float>(rng_state & 0x00FFFFFFu) /
              static_cast<float>(0x01000000u);

    return r < ratio;
}

} // namespace imp
