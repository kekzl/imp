#pragma once

// Log lines of WeightMap::apply_weights for names no matcher claimed (kept out of weight_map.cpp:
// file-size and CCN ratchet).

#include "core/logging.h"
#include "model/model_arch.h"

#include <string>

namespace imp::weight_map_report {

// The first kWarnCap unclaimed names warn, the rest go to the debug log; total() reports them all.
// An unsupported 74B checkpoint logged 28 884 such WARN lines.
inline constexpr int kWarnCap = 16;

inline void name(const char* what, const std::string& tensor, int seen) {
    if (seen < kWarnCap)
        IMP_LOG_WARN("WeightMap: %s: %s", what, tensor.c_str());
    else
        IMP_LOG_DEBUG("WeightMap: %s: %s", what, tensor.c_str());
}

inline void total(int unrecognised) {
    if (unrecognised > kWarnCap)
        IMP_LOG_WARN(
            "WeightMap: %d unrecognised weight names in total, the first %d are listed above "
            "(all of them at log level debug)",
            unrecognised, kWarnCap);
}

// Not one tensor assigned. GENERIC = no registry row matched the checkpoint's class
// (hf_config_loader names it in its 'unknown HF architecture' line).
inline void nothing_assigned(ModelArch arch, int skipped) {
    if (arch == ModelArch::GENERIC)
        IMP_LOG_ERROR(
            "WeightMap: unsupported architecture: no loader matches this checkpoint's class "
            "(see the 'unknown HF architecture' line) and none of its %d tensors was "
            "recognised. Supported models: docs/MODELS.md",
            skipped);
    else
        IMP_LOG_ERROR("WeightMap: no tensors were assigned -- check weight names");
}

}  // namespace imp::weight_map_report
