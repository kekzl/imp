#include "runtime/presets.h"

namespace imp {

SamplingDefaults get_sampling_defaults(ModelArch arch) {
    SamplingDefaults d;
    d.temperature = 0.6f;
    d.top_p = 0.95f;
    d.top_k = 0;

    switch (arch) {
        case ModelArch::QWEN3:
        case ModelArch::QWEN3_MOE:
        case ModelArch::QWEN35:
        case ModelArch::QWEN35_MOE:
            d.top_k = 20;
            break;
        default:
            break;
    }

    return d;
}

} // namespace imp
