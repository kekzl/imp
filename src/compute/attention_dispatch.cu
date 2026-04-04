#include "compute/attention.h"
#include "compute/attention_tc.h"
#include "core/logging.h"
#include <cuda_runtime.h>
#include <cstdlib>

#include "compute/attention_cutlass_fmha.h"
#include "compute/attention_fmha_sm120.h"
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
    int sm = get_device_sm_version();
    // sliding_window that covers entire seq_kv doesn't restrict attention
    int seq_kv = static_cast<int>(K.shape[1]);
    bool sw_active = (sliding_window > 0 && sliding_window < seq_kv);

    // MXFP4 tensor core attention: block-scaled FP4 Q·K^T on sm_120+.
    // ~2x compute throughput over FP16 TC. Enabled with IMP_MXFP4_ATTENTION=1.
    // Uses O(seq²) memory — falls back for long sequences or unsupported configs.
    if (attention_mxfp4_available() && sm >= 120 && !sw_active) {
        if (attention_mxfp4_prefill(Q, K, V, O, scale, causal, softcap, stream)) {
            return;
        }
    }

    // Native sm_120 FP8 FMHA: QK^T in FP8 E4M3 (m16n8k32) for 2x score throughput.
    // PV stays FP16. Set IMP_NO_FP8_FMHA=1 to force FP16 path.
    static bool use_fp8_fmha = !getenv("IMP_NO_FP8_FMHA");
    if (use_fp8_fmha && sm >= 120) {
        bool fp8_ok = fmha_sm120_fp8_prefill(Q, K, V, O, scale, causal, sliding_window, softcap, stream);
        if (fp8_ok) {
            IMP_LOG_DEBUG("FMHA dispatch: using FP8 sm120 kernel (hd=%d)", static_cast<int>(Q.shape[3]));
            return;
        }
    }

    // Native sm_120 FP16 FMHA: WMMA for Blackwell with sliding window support.
    // Fallback when FP8 is disabled or unsupported config.
    // Set IMP_NO_FMHA_SM120=1 to skip and use CUTLASS/WMMA fallback.
    static bool use_fmha_sm120 = !getenv("IMP_NO_FMHA_SM120");
    if (use_fmha_sm120 && sm >= 120) {
        if (fmha_sm120_prefill(Q, K, V, O, scale, causal, sliding_window, softcap, stream)) {
            return;
        }
    }

    // CUTLASS FMHA: WGMMA + TMA on sm_90+. ~2x throughput vs WMMA.
    // Supports softcap (Gemma-2/3). Not supported: sliding window (Mistral).
    // Set IMP_NO_CUTLASS_FMHA=1 to force WMMA fallback (for benchmarking).
    static bool use_cutlass = !getenv("IMP_NO_CUTLASS_FMHA");
    if (use_cutlass && sm >= 90 && !sw_active) {
        if (cutlass_fmha_prefill(Q, K, V, O, scale, causal, softcap, stream)) {
            return;
        }
        // Fall through to hand-written kernels on failure
        int hd = static_cast<int>(Q.shape[3]);
        IMP_LOG_DEBUG("CUTLASS FMHA unavailable (hd=%d, softcap=%.1f), using WMMA fallback", hd, softcap);
    }

    if (sm >= 120) {
        // Optimized WMMA kernel with 128x64 tiles for Blackwell (sm_120+).
        flash_attention_blackwell(Q, K, V, O, scale, causal, sliding_window, softcap, stream);
    } else if (sm >= 90) {
        flash_attention_prefill_tc(Q, K, V, O, scale, causal, sliding_window, softcap, stream);
    } else {
        flash_attention_prefill(Q, K, V, O, scale, causal, sliding_window, softcap, stream);
    }
}

} // namespace imp
