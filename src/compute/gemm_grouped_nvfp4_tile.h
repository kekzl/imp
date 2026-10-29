#pragma once

#include "compute/gemm_cutlass_grouped_3x.h"  // GroupedNvfp4DeviceArgs

#include <cuda_runtime.h>

namespace imp {

// Own grouped NVFP4 MoE GEMM on the CUTLASS 3.x grouped operands (same bytes, all state on the
// device). expanded: rows of the expert-sorted activation buffer (bound of its TMA map).
// K % 64 == 0, K >= 384, N % 8 == 0, <= 512 experts; false = refused (caller keeps CUTLASS).
// Two tensormap slots per CTA live in a static device array (one launch at a time).
[[nodiscard]] bool gemm_grouped_nvfp4_tile(int n_experts, int N, int K, int expanded,
                                           const GroupedNvfp4DeviceArgs& args, cudaStream_t stream);

// MoE prefill entry: the own kernel under gemm.moe_tile (ACTIVE logged once), else / on refusal
// gemm_grouped_cutlass_3x_nvfp4_device_args.
[[nodiscard]] bool gemm_grouped_nvfp4_dispatch(int n_experts, int N, int K, int expanded,
                                               const GroupedNvfp4DeviceArgs& args, cudaStream_t stream);

}  // namespace imp
