// =============================================================================
// attention_fmha_mxf4nvf4_sm120.cu -- Stage 4 scaffolding for the MXFP4 FMHA
// hardware-blockscale MMA upgrade (kind::mxf4nvf4.block_scale.m16n8k64).
// =============================================================================
//
// This file is the landing pad for the Stage 4 integration. Upstream work:
//   - Raw MMA instruction: verified to compile + launch (b9ec21a / f50221e)
//   - A=0 → D=0 invariant: verified (7298175 / e7615f3)
//   - Quant math round-trip (linear layout): 9.5% RMSE (7a77063 / 4baf0d7)
//   - Quant math round-trip (HW scale layout): 9.5% RMSE (8148801 / 30e827f)
//   - Raw MMA throughput: 2.60× vs legacy (b623cda / b9fbb66)
//
// What's here:
//   - mxf4nvf4_blockscale_enabled(): checks IMP_FMHA_BLOCKSCALE env var
//   - fmha_sm120_mxf4nvf4_prefill(): entry point for the dispatcher
//     Currently delegates to the legacy f8f6f4.m16n8k32 kernel with a
//     diagnostic log — the actual kernel body is WIP.
//
// What's missing to land the 2.60× speedup:
//   - Per-thread CUTLASS ALayout / SFALayout translation to match the
//     (T32,V32)→(M16,K64) mapping expected by the HW MMA (see
//     sageattention3_blackwell/sageattn3/blackwell/cute_extension.h:157+).
//   - Per-16-element FP8 UE4M3 scale storage in SMEM (nvfp4_quant_hw
//     already produces this layout — needs integration here).
//   - Update the Q/K quantization loop in the kernel to emit the new
//     scale layout.
//   - Swap the MMA issue from kind::f8f6f4 to kind::mxf4nvf4.block_scale
//     (see mxf4nvf4_mma_bench.cu for the PTX template).
//
// Why deferred: the CUTLASS CuTe ALayout decomposition is not a one-pass
// translation — it distributes a thread's 32 FP4 values across 4 rows
// and 8 k-positions-per-row in a non-row-major pattern. Without running
// the MMA with reference data to cross-check, operand layout bugs are
// hard to catch. Future session should start with an end-to-end Q·K^T
// correctness harness against FP32 reference (see the rejected prototype
// attempt in the session transcript for context), then wire into the
// kernel.
// =============================================================================

#include "compute/attention_fmha_mxf4nvf4_sm120.h"
#include "compute/attention_fmha_mxfp4_sm120.h"
#include "core/logging.h"
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstring>

namespace imp {

// ---------------------------------------------------------------------------
// Env-var gate. IMP_FMHA_BLOCKSCALE=1 activates the new kernel path.
// Cached on first lookup.
// ---------------------------------------------------------------------------
bool mxf4nvf4_blockscale_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char* v = std::getenv("IMP_FMHA_BLOCKSCALE");
        cached = (v != nullptr && std::strcmp(v, "0") != 0) ? 1 : 0;
        if (cached) {
            IMP_LOG_INFO("IMP_FMHA_BLOCKSCALE set: routing MXFP4 prefill through "
                         "the Stage 4 landing pad (currently delegates to the legacy "
                         "kind::f8f6f4 kernel — the blockscale-MMA kernel body is WIP). "
                         "Expected raw MMA speedup: 2.60× once the CUTLASS SFALayout "
                         "translation is implemented.");
        }
    }
    return cached == 1;
}

// ---------------------------------------------------------------------------
// Entry point for the dispatcher.
// Returns the same contract as the legacy fmha_sm120_mxfp4_prefill:
//   true  → handled
//   false → config unsupported, caller should try next fallback
// ---------------------------------------------------------------------------
bool fmha_sm120_mxf4nvf4_prefill(
    const Tensor& Q, const Tensor& K, const Tensor& V, Tensor& O,
    float scale, bool causal, int sliding_window, float softcap,
    cudaStream_t stream)
{
    // TODO(Stage 4 completion): replace this delegate with a proper
    // kernel launch that uses kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.
    // See file-level comment for the outstanding work items.
    //
    // Until then, delegate to the legacy kernel so the code path is exercised
    // and regression tests still cover us under IMP_FMHA_BLOCKSCALE=1.
    return fmha_sm120_mxfp4_prefill(Q, K, V, O, scale, causal,
                                     sliding_window, softcap, stream);
}

} // namespace imp
