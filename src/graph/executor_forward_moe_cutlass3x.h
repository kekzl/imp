#pragma once
#include <cuda_runtime.h>

namespace imp {
// Fused gate+up grouped GEMM: one CUTLASS 3.x dispatch with 2*ne problems.
// Returns false on dispatch failure (caller should fall back to two-call form).
//
// Lives in a separate translation unit to prevent i-cache pressure in the
// 2500+ line executor_forward_moe.cu's run_moe_ffn function (which would
// regress decode by -7% per iteration2_findings.md).
//
// Currently flat-perf vs the GrpGemm-cached two-call form (commit 769effe
// already absorbs the per-dispatch savings the fusion was meant to deliver).
// Kept as opt-in (IMP_NVFP4_FUSED_GATEUP=1) for future scenarios where the
// per-call amortization may shift.
__attribute__((noinline))
bool dispatch_gate_up_grouped_fused(
    int ne_active,
    const int* host_M,
    int N, int K,
    const void* const* host_ptr_A,
    const void* const* host_ptr_SFA,
    const void* const* host_ptr_B_gate,
    const void* const* host_ptr_SFB_gate,
    const void* const* host_ptr_B_up,
    const void* const* host_ptr_SFB_up,
    void* const* host_ptr_D_gate,
    void* const* host_ptr_D_up,
    const float* host_alpha_gate,
    const float* host_alpha_up,
    cudaStream_t stream);
}
