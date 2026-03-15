#include "compute/attention.h"
#include "compute/attention_tc.h"
#include "core/logging.h"
#include <cuda_runtime.h>
#include <cstdlib>

#ifdef IMP_USE_CUTLASS
#include "compute/attention_cutlass_fmha.h"
#endif

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
#ifdef IMP_USE_CUTLASS
    // CUTLASS FMHA: WGMMA + TMA on sm_90+. ~2x throughput vs WMMA.
    // Supports softcap (Gemma-2/3). Not supported: sliding window (Mistral).
    // Set IMP_NO_CUTLASS_FMHA=1 to force WMMA fallback (for benchmarking).
    static bool use_cutlass = !getenv("IMP_NO_CUTLASS_FMHA");
    // sliding_window that covers entire seq_kv doesn't restrict attention
    int seq_kv = static_cast<int>(K.shape[1]);
    bool sw_active = (sliding_window > 0 && sliding_window < seq_kv);
    if (use_cutlass && sm >= 90 && !sw_active) {
        if (cutlass_fmha_prefill(Q, K, V, O, scale, causal, softcap, stream)) {
            return;
        }
        // Fall through to hand-written kernels on failure
        int hd = static_cast<int>(Q.shape[3]);
        IMP_LOG_DEBUG("CUTLASS FMHA unavailable (hd=%d, softcap=%.1f), using WMMA fallback", hd, softcap);
    }
#endif

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
