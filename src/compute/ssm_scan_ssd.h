#pragma once

#include "compute/ssm_scan_reg.h"

namespace imp {

// Shortest chunk the SSD scan takes; shorter ones stay on ssm_scan_reg.
constexpr int kSsdMinTokens = 128;

// Chunked SSD (tensor-core) Mamba2 prefill scan, gdn.ssd_scan. Covers state_size 128,
// head_dim % 32 == 0, no snapshot row. Returns false (nothing launched) otherwise.
[[nodiscard]] bool ssm_scan_ssd_launch(const SsmScanArgs& a, bool fp16);

}  // namespace imp
