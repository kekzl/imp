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

// Source-qtype-aware capabilities. Refines the per-kind capabilities by what
// the source storage actually supports. Mirrors the runtime policy in
// `nvfp4_beneficial()` (pre_dequant_internal.h): NVFP4 / CUTLASS_NVFP4 / MXFP4
// overlays are only beneficial when the source qtype has ≥ 5.5 bits/element
// (Q5_K, Q6_K, Q8_0, Q8_K). For sub-5-bit sources (Q4_K, Q4_0, Q5_0, Q5_1,
// Q3_K, Q2_K, IQ*), NVFP4 is a representation change at the same bit-width —
// no compression win, possible quality risk. Those overlays are removed from
// `supported`, and `required_floor` is raised to FP16 if needed.
//
// Native NVFP4 / MXFP4 sources keep NVFP4 / MXFP4 as the only tier (the
// "overlay" IS the source storage). Compute dtypes (F16, BF16, F32) are
// treated as "no compression-overlay" (current LM-head / embeddings policy).
//
// Used by StoragePlanner so plans don't propose NVFP4 for Q4_K weights —
// the cause of the 2026-05-24 Q4_K_M cache coverage gap. See the
// plan-driven-cache design memo (archived: docs/archive/README.md).
KindCapabilities effective_capabilities(TensorKind k, QType source_qtype);

}  // namespace imp
