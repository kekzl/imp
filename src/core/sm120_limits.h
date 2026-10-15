#pragma once

#include <cstddef>

namespace imp {

// sm_120 opt-in dynamic smem per block: 99 KiB = 101376 B
// (cudaDevAttrMaxSharedMemoryPerBlockOptin on RTX 5090).
inline constexpr size_t kSm120SmemOptinBytes = 101376;

}  // namespace imp
