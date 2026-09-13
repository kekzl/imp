#pragma once

// Shared preamble for the GDN kernel TUs, split (file-size gate) into gdn.cu (fused scan,
// RMSNorm-gated-SiLU, V-head reorder, reference scan, legacy decode), gdn_scan.cu (chunkwise
// scalar variants), gdn_scan_tc.cu (WY-rep TC/WMMA variants). Host launchers in compute/gdn.h.
// No shared __device__ helpers/constants across the split kernels; each inlines its own math.

#include "compute/gdn.h"
#include <cmath>
#include <type_traits>
