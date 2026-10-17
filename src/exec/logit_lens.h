#pragma once

#include <cstdint>
#include <vector>

namespace imp {

// Logit lens readout (roadmap row 115), index layer * rows + r; top_* hold top_n entries per index.
struct LogitLensResult {
    int layers = 0;
    int rows = 0;
    int top_n = 0;
    std::vector<float> lp;  // logprob of the row's sampled token
    std::vector<int32_t> rank;
    std::vector<float> entropy;  // nats
    std::vector<int32_t> top_ids;
    std::vector<float> top_lp;
};

}  // namespace imp
