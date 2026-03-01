#pragma once

#include <vector>
#include <cstdint>
#include <string>

namespace imp {

enum class RequestStatus {
    PENDING,
    PREFILLING,
    DECODING,
    FINISHED,
    CANCELLED
};

const char* request_status_name(RequestStatus status);

struct Request {
    int id = 0;
    RequestStatus status = RequestStatus::PENDING;

    std::vector<int32_t> input_tokens;
    std::vector<int32_t> output_tokens;

    int max_tokens = 256;
    float temperature = 1.0f;
    float top_p = 1.0f;
    int top_k = 0;
    int seed = -1;
    float min_p = 0.0f;               // Min probability threshold (0 = disabled)
    float typical_p = 1.0f;           // Locally typical sampling (1.0 = disabled)
    float repetition_penalty = 1.0f;   // >1 penalizes repeats (multiplicative)
    float frequency_penalty = 0.0f;    // Subtractive per-occurrence
    float presence_penalty = 0.0f;     // Subtractive binary (appeared or not)
    float dry_multiplier = 0.0f;      // DRY penalty scale (0 = disabled)
    float dry_base = 1.75f;           // DRY exponential base
    int dry_allowed_length = 2;       // N-grams at or below this not penalized
    int dry_penalty_last_n = 0;       // How far back to scan (0 = all)
    int mirostat = 0;                  // 0=off, 2=Mirostat v2
    float mirostat_tau = 5.0f;        // Target entropy
    float mirostat_eta = 0.1f;        // Learning rate
    float mirostat_mu = 0.0f;         // Running variable (persists across tokens, init = 2*tau)
    bool ignore_eos = false;   // Don't stop on EOS (benchmark mode)
    int prefill_offset = 0;    // Chunked prefill: tokens processed so far

    int context_len() const {
        return static_cast<int>(input_tokens.size() + output_tokens.size());
    }
};

} // namespace imp
