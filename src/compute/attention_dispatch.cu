#include "compute/attention.h"
#include "compute/attention_tc.h"
#include "core/logging.h"
#include "runtime/config.h"
#include <cuda_runtime.h>
#include <cstdlib>

#include "compute/attention_fmha_sm120.h"
#include "compute/attention_fmha_mxfp4_sm120.h"
#include "compute/attention_mxfp4_prefill.h"

namespace imp {

static int cached_sm_version = -1;

int get_device_sm_version() {
    if (cached_sm_version >= 0)
        return cached_sm_version;
    int device = 0;
    cudaGetDevice(&device);
    int major = 0, minor = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
    cached_sm_version = major * 10 + minor;
    IMP_LOG_INFO("Device SM version: %d.%d (sm_%d)", major, minor, cached_sm_version);
    return cached_sm_version;
}

// NOTE: the path-selection ORDER + config gates below are mirrored as a pure
// host function `select_attn_prefill_path` in attention_dispatch_decision.h and
// pinned by test_routing_decision.cpp (R2 / P1.4). The kernels are
// called lazily here (each launches on accept, falls through on decline), so
// this can't simply delegate to the pure function — but any reorder or gate
// change MUST be reflected in that header or the unit test will diverge.
void attention_prefill_dispatch(const Tensor& Q, const Tensor& K, const Tensor& V, Tensor& O, float scale,
                                bool causal, int sliding_window, float softcap, cudaStream_t stream,
                                const RuntimeConfig& rcfg, int q_offset) {
    // MXFP4 Flash Attention: tiled FP4 E2M1 Q·K^T with online softmax.
    // O(n) memory, ~4x score throughput over FP16, ~2x over FP8.
    // Enabled with IMP_MXFP4_ATTENTION=1.
    if (attention_mxfp4_available()) {
        if (fmha_sm120_mxfp4_prefill(Q, K, V, O, scale, causal, sliding_window, softcap, stream)) {
            return;
        }
        // Fall through: head_dim not supported (e.g. < 32), use FP8/FP16 path
    }

    // Register-resident FA2 ("echtes FA"): keeps S/P/O in registers, 1 barrier per
    // KV tile (vs the FP8 FMHA's smem-materialized S/P/O + 4 barriers). Opt-in via
    // [attention] fmha_fa2: "on" | "never" (default), env IMP_FMHA_FA2. head_dim=128.
    if (rcfg.attention.fmha_fa2 == "on") {
        if (fmha_sm120_fa2_prefill(Q, K, V, O, scale, causal, sliding_window, softcap, stream, q_offset)) {
            IMP_LOG_DEBUG("FMHA dispatch: using FA2 register-resident kernel (hd=%d)",
                          static_cast<int>(Q.shape[3]));
            return;
        }
        // unsupported config (hd!=128) → fall through to FP8/FP16 path
    }

    // Native sm_120 FP8 FMHA: QK^T in FP8 E4M3 (m16n8k32) for 2x score throughput.
    // PV stays FP16. [attention] fp8_fmha: "auto" (default ON) | "never"
    const bool use_fp8_fmha = rcfg.attention.fp8_fmha != "never";
    if (use_fp8_fmha) {
        bool fp8_ok = fmha_sm120_fp8_prefill(Q, K, V, O, scale, causal, sliding_window, softcap, stream,
                                              q_offset);
        if (fp8_ok) {
            IMP_LOG_DEBUG("FMHA dispatch: using FP8 sm120 kernel (hd=%d)", static_cast<int>(Q.shape[3]));
            return;
        }
    }

    // Native sm_120 FP16 FMHA: WMMA for Blackwell with sliding window support.
    // Fallback when FP8 is disabled or unsupported config.
    const bool use_fmha_sm120 = rcfg.attention.fmha_sm120 != "never";
    if (use_fmha_sm120) {
        if (fmha_sm120_prefill(Q, K, V, O, scale, causal, sliding_window, softcap, stream, q_offset)) {
            return;
        }
    }

    // Final fallback: WMMA 128x64 tiles for Blackwell.
    flash_attention_blackwell(Q, K, V, O, scale, causal, sliding_window, softcap, stream, q_offset);
}

}  // namespace imp
