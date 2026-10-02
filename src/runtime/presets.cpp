#include "runtime/presets.h"

namespace imp {

SamplingDefaults get_sampling_defaults(ModelArch arch) {
    SamplingDefaults d;
    model_arch_sampling_defaults(arch, d.temperature, d.top_p, d.top_k);
    return d;
}

SamplingDefaults resolve_sampling_defaults(ModelArch arch, const HFConfigLoader::GenerationConfig& gen) {
    SamplingDefaults d = get_sampling_defaults(arch);
    if (gen.temperature >= 0.0f)
        d.temperature = gen.temperature;
    if (gen.top_p >= 0.0f)
        d.top_p = gen.top_p;
    if (gen.top_k >= 0)
        d.top_k = gen.top_k;
    return d;
}

}  // namespace imp
