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
    float repetition_penalty = 1.0f;   // >1 penalizes repeats (multiplicative)
    float frequency_penalty = 0.0f;    // Subtractive per-occurrence
    float presence_penalty = 0.0f;     // Subtractive binary (appeared or not)
    bool ignore_eos = false;   // Don't stop on EOS (benchmark mode)
    int prefill_offset = 0;    // Chunked prefill: tokens processed so far

    int context_len() const {
        return static_cast<int>(input_tokens.size() + output_tokens.size());
    }
};

} // namespace imp
