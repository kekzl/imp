#pragma once

#include "core/qtype.h"
#include "core/tensor_kind.h"
#include "core/storage_tier.h"

namespace imp {

struct KindCapabilities {
    TierMask supported;
    StorageTier required_floor;
    bool fusable;
};

const KindCapabilities& capabilities_of(TensorKind k);

// Source-qtype-aware capabilities, mirroring nvfp4_beneficial() (pre_dequant_internal.h):
// NVFP4/CUTLASS_NVFP4/MXFP4 overlays only help when source has >=5.5 bits/elem (Q5_K, Q6_K,
// Q8_0, Q8_K); sub-5-bit sources (Q4_K, Q4_0, Q5_0, Q5_1, Q3_K, Q2_K, IQ*) drop those
// overlays and raise required_floor to FP16. Native NVFP4/MXFP4 sources keep it as the only
// tier; compute dtypes (F16/BF16/F32) get no compression-overlay. Used by StoragePlanner so
// plans don't propose NVFP4 for Q4_K weights (the 2026-05-24 Q4_K_M cache coverage gap).
KindCapabilities effective_capabilities(TensorKind k, QType source_qtype);

}  // namespace imp
