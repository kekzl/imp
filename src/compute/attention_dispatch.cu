#include "compute/attention.h"
#include "compute/attention_tc.h"
#include "core/logging.h"
#include "runtime/config.h"
#include "runtime/process_diag.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

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
    // Enabled with [attention] mxfp4 = "always". Blockscale/ksmooth/pv_fp4
    // (#846 SageAttention3-recipe spike) are read from process_diag inside
    // the launcher.
    if (attention_mxfp4_available()) {
        if (fmha_sm120_mxfp4_prefill(Q, K, V, O, scale, causal, sliding_window, softcap, stream,
                                     process_diag_mxfp4_blockscale(), q_offset)) {
            return;
        }
        // Fall through: head_dim not supported (e.g. < 32), use FP8/FP16 path
    }

    // Register-resident FA2 ("echtes FA"): keeps S/P/O in registers, 1 barrier per
    // KV tile (vs the FP8 FMHA's smem-materialized S/P/O + 4 barriers). Default on
    // via [attention] fmha_fa2, env IMP_FMHA_FA2. head_dim=128. QK^T runs in f16
    // (same numerical class as cuBLAS) unless the user explicitly opts into the
    // e4m3 fp8-QK mode (fa2_fp16qk=never AND fp8_fmha=on) — raw-converted fp8
    // scores compound per layer into garbage on real activations (#511).
    if (rcfg.attention.fmha_fa2 == "on") {
        const bool fa2_fp8_optin =
            rcfg.attention.fa2_fp16qk == "never" && rcfg.attention.fp8_fmha == "on";
        if (fmha_sm120_fa2_prefill(Q, K, V, O, scale, causal, sliding_window, softcap, stream, q_offset,
                                   /*fp16_qk=*/!fa2_fp8_optin)) {
            IMP_LOG_DEBUG("FMHA dispatch: using FA2 register-resident kernel (hd=%d)",
                          static_cast<int>(Q.shape[3]));
            return;
        }
        // unsupported config (hd!=128) → fall through to FP16 WMMA path
    }

    // fp8-QK FMHA (smem-materializing): QK^T in raw-converted FP8 E4M3 for 2x
    // score throughput — but the unscaled e4m3 conversion carries ~10% relative
    // score error that compounds across layers (#511): teacher-forced PPL
    // gemma-3-12b 16.6 -> 549 / Qwen3-8B 40.5 -> 4506 when this kernel actually
    // serves prefill. Strictly opt-in: [attention] fp8_fmha = "on".
    if (rcfg.attention.fp8_fmha == "on") {
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

    // Final tier: WMMA 128x64 tiles for Blackwell. Declines (returns false)
    // for unsupported configs — hd ∉ {64,96,128,256} or smem over the device
    // opt-in (hd=256 at Br=64 needs ~176 KB vs 99 KB on sm_120).
    if (flash_attention_blackwell(Q, K, V, O, scale, causal, sliding_window, softcap, stream, q_offset)) {
        return;
    }

    // Chain exhausted. Fail loudly instead of leaving O unwritten — the old
    // silent blackwell→tc fallback at hd=256 swallowed the launch failure and
    // produced garbage logits (teacher-forced PPL ~1e10, #654). Reaching this
    // requires disabling the FP16 WMMA tier by config or an unsupported
    // head_dim; both deserve an error, not silent corruption.
    char msg[160];
    snprintf(msg, sizeof(msg),
             "attention_prefill_dispatch: no prefill kernel accepts head_dim=%d "
             "(check attention.fmha_sm120/fmha_fa2 config) (#654)",
             static_cast<int>(Q.shape[3]));
    throw std::runtime_error(msg);
}

}  // namespace imp
