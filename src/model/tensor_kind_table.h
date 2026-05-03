#pragma once

#include "core/tensor_kind.h"
#include "core/storage_tier.h"

namespace imp {

struct KindCapabilities {
    TierMask    supported;
    StorageTier required_floor;
    bool        fusable;
};

const KindCapabilities& capabilities_of(TensorKind k);

} // namespace imp
