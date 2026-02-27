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
    bool ignore_eos = false;   // Don't stop on EOS (benchmark mode)

    int context_len() const {
        return static_cast<int>(input_tokens.size() + output_tokens.size());
    }
};

} // namespace imp
