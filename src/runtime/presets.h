#pragma once

#include "model/generation_config.h"
#include "model/model_arch.h"

namespace imp {

// Default sampling parameters per model family.
// These are user-preference defaults that cannot be auto-detected from model metadata.
struct SamplingDefaults {
    float temperature = 0.6f;
    float top_p = 0.95f;
    int top_k = 0;
    float repetition_penalty = 1.0f;  // generation_config.json only; no arch preset sets one
};

// Get sampling defaults for a model architecture.
// Returns family-appropriate temperature/top_p/top_k values.
SamplingDefaults get_sampling_defaults(ModelArch arch);

// CLI and server defaults (#2462): generation_config.json (field >= 0) over the arch preset.
SamplingDefaults resolve_sampling_defaults(ModelArch arch, const GenerationConfig& gen);

}  // namespace imp
