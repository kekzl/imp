#include "compute/attention.h"
#include "compute/attention_tc.h"
#include "core/logging.h"
#include <cuda_runtime.h>
#include <cstdlib>

#include "compute/attention_fmha_sm120.h"
#include "compute/attention_fmha_mxfp4_sm120.h"
#include "compute/attention_fmha_mxf4nvf4_sm120.h"
#include "compute/attention_mxfp4_prefill.h"

namespace imp {

static int cached_sm_version = -1;

int get_device_sm_version() {
    if (cached_sm_version >= 0) return cached_sm_version;
    int device = 0;
    cudaGetDevice(&device);
    int major = 0, minor = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
    cached_sm_version = major * 10 + minor;
    IMP_LOG_INFO("Device SM version: %d.%d (sm_%d)", major, minor, cached_sm_version);
    return cached_sm_version;
}

void attention_prefill_dispatch(
    const Tensor& Q, const Tensor& K, const Tensor& V, Tensor& O,
    float scale, bool causal, int sliding_window, float softcap, cudaStream_t stream) {

    // MXFP4 Flash Attention: tiled FP4 E2M1 Q·K^T with online softmax.
    // O(n) memory, ~4x score throughput over FP16, ~2x over FP8.
    // Enabled with IMP_MXFP4_ATTENTION=1.
    //
    // By default uses kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64 with
    // per-16-element UE4M3 scales (+1.8% vs legacy at HD=128). head_dim=96
    // internally falls back to legacy kind::f8f6f4.m16n8k32 (not multiple of 64).
    // Set IMP_FMHA_BLOCKSCALE=0 to force the legacy path entirely (A/B testing).
    if (attention_mxfp4_available()) {
        static const bool use_blockscale = !mxf4nvf4_blockscale_disabled();
        if (use_blockscale) {
            if (fmha_sm120_mxf4nvf4_prefill(Q, K, V, O, scale, causal,
                                             sliding_window, softcap, stream)) {
                return;
            }
        } else {
            if (fmha_sm120_mxfp4_prefill(Q, K, V, O, scale, causal,
                                          sliding_window, softcap, stream)) {
                return;
            }
        }
        // Fall through: head_dim not supported (e.g. < 32), use FP8/FP16 path
    }

    // Native sm_120 FP8 FMHA: QK^T in FP8 E4M3 (m16n8k32) for 2x score throughput.
    // PV stays FP16. Set IMP_NO_FP8_FMHA=1 to force FP16 path.
    static bool use_fp8_fmha = !getenv("IMP_NO_FP8_FMHA");
    if (use_fp8_fmha) {
        bool fp8_ok = fmha_sm120_fp8_prefill(Q, K, V, O, scale, causal, sliding_window, softcap, stream);
        if (fp8_ok) {
            IMP_LOG_DEBUG("FMHA dispatch: using FP8 sm120 kernel (hd=%d)", static_cast<int>(Q.shape[3]));
            return;
        }
    }

    // Native sm_120 FP16 FMHA: WMMA for Blackwell with sliding window support.
    // Fallback when FP8 is disabled or unsupported config.
    // Set IMP_NO_FMHA_SM120=1 to skip and use WMMA fallback.
    static bool use_fmha_sm120 = !getenv("IMP_NO_FMHA_SM120");
    if (use_fmha_sm120) {
        if (fmha_sm120_prefill(Q, K, V, O, scale, causal, sliding_window, softcap, stream)) {
            return;
        }
    }

    // Final fallback: WMMA 128x64 tiles for Blackwell.
    flash_attention_blackwell(Q, K, V, O, scale, causal, sliding_window, softcap, stream);
}

} // namespace imp
