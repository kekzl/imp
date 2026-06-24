#pragma once

// Shared internal preamble for the GDN kernel translation units.
//
// gdn.cu was split (file-size gate) into:
//   - gdn.cu          : fused scan, RMSNorm-gated-SiLU, V-head reorder,
//                       reference scan, legacy per-token decode.
//   - gdn_scan.cu     : chunkwise (sequential) + WY-rep scalar scan variants.
//   - gdn_scan_tc.cu  : WY-rep Tensor-Core (WMMA) scan variants (Phase 2b/2c).
//
// All host launcher declarations live in compute/gdn.h (unchanged). This header
// only centralises the include set common to the split TUs. There are no shared
// __device__/__forceinline__ helpers, no __constant__ memory, and no
// anonymous-namespace constants/typedefs in the GDN kernels — each kernel
// inlines its own delta-rule math — so there is nothing further to share here.

#include "compute/gdn.h"
#include <cmath>
#include <type_traits>
