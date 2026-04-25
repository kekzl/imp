// =============================================================================
// attention_fmha_mxf4nvf4_sm120.cu -- Stage 4 blockscale FMHA entry point.
// =============================================================================
//
// The kernel body lives in attention_fmha_mxfp4_sm120.cu — template parameter
// UseBlockScaleMma switches the Phase 1 MMA between:
//   legacy: mma.sync.kind::f8f6f4.m16n8k32          (2× K-chunks per issue)
//   new:    mma.sync.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64
//           (merges 2 K-chunks → half the MMA issue count)
//
// Uniform scale=1.0 (sfa=sfb=0x38383838) keeps post-MMA manual per-row scaling
// unchanged, so output is bit-equivalent to the legacy path on m16n8k64 math.
//
// Operand layout for the new MMA is verified byte-exact in
// tests/test_mxf4nvf4_qkt_validate.cu (4 tests, 128/128 correct). The CuTe
// (T32,V32)→(M16,K64) layout is column-major — offset = k*M + m — which
// matches the legacy register distribution when 2 legacy chunks are merged
// into {a0,a1,a2,a3} = {chunk0_row_T1, chunk0_row_T1+8, chunk1_row_T1,
// chunk1_row_T1+8}. No SFA/SFB changes (uniform scale).
//
// Raw MMA bench: 2.50× throughput (254.62 vs 101.87 TOPS @ 170 warps × 1M
// iters). Actual kernel speedup is lower because Phase 1 is only a fraction
// of total kernel time (softmax, PV, load/store add overhead).
// =============================================================================

#include "compute/attention_fmha_mxf4nvf4_sm120.h"
#include "compute/attention_fmha_mxfp4_sm120.h"
#include "core/logging.h"
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstring>

namespace imp {

bool mxf4nvf4_blockscale_disabled() {
    static int cached = -1;
    if (cached < 0) {
        const char* v = std::getenv("IMP_FMHA_BLOCKSCALE");
        cached = (v != nullptr && std::strcmp(v, "0") == 0) ? 1 : 0;
        if (cached) {
            IMP_LOG_INFO("IMP_FMHA_BLOCKSCALE=0: MXFP4 attention forced to "
                         "legacy kind::f8f6f4.m16n8k32 path (A/B debug only).");
        }
    }
    return cached == 1;
}

bool mxf4nvf4_blockscale_enabled() {
    // Default ON — the per-16-element block_scale path is +1.8% vs legacy
    // at HD=128 and functionally equivalent. Can be disabled via
    // IMP_FMHA_BLOCKSCALE=0 for A/B testing.
    return !mxf4nvf4_blockscale_disabled();
}

bool fmha_sm120_mxf4nvf4_prefill(
    const Tensor& Q, const Tensor& K, const Tensor& V, Tensor& O,
    float scale, bool causal, int sliding_window, float softcap,
    cudaStream_t stream)
{
    return fmha_sm120_mxfp4_prefill(Q, K, V, O, scale, causal,
                                     sliding_window, softcap, stream,
                                     /*use_blockscale=*/true);
}

} // namespace imp
