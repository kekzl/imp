#pragma once

#include "model/model_arch.h"

namespace imp {

// Default sampling parameters per model family.
// These are user-preference defaults that cannot be auto-detected from model metadata.
struct SamplingDefaults {
    float temperature = 0.6f;
    float top_p = 0.95f;
    int top_k = 0;
};

// Get sampling defaults for a model architecture.
// Returns family-appropriate temperature/top_p/top_k values.
SamplingDefaults get_sampling_defaults(ModelArch arch);

} // namespace imp
