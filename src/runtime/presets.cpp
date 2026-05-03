#include "runtime/presets.h"

namespace imp {

SamplingDefaults get_sampling_defaults(ModelArch arch) {
    SamplingDefaults d;
    model_arch_sampling_defaults(arch, d.temperature, d.top_p, d.top_k);
    return d;
}

}  // namespace imp
