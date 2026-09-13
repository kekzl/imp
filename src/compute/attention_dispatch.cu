#include "compute/attention.h"
#include "compute/attention_tc.h"
#include "core/logging.h"
#include "core/dispatch_policy.h"
#include "core/process_diag.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

#include "compute/attention_fmha_sm120.h"
#include "compute/attention_fmha_mxfp4_sm120.h"
#include "compute/attention_mxfp4_prefill.h"
#include "compute/dispatch_record.h"
#include "compute/attention_dispatch_decision.h"

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

// Path-selection order + gates here are mirrored as select_attn_prefill_path() in
// attention_dispatch_decision.h; each tier's outcome is replayed against that model and a
// divergence fires a one-shot log (not silence) so a reorder or gate drift is caught.
// KNOWN LIMIT: a tier reordered ahead of the winner that WOULD have accepted stays invisible,
// since the dispatch short-circuits before asking it.
static void verify_against_routing_model(const DispatchPolicy& rcfg, const AttnKernelSupport& sup,
                                         bool has_sinks, AttnPrefillPath chosen) {
    const AttnPrefillPath modeled = select_attn_prefill_path(rcfg, sup, has_sinks);
    if (modeled == chosen)
        return;
    static bool warned = false;
    if (warned)
        return;
    warned = true;
    IMP_LOG_ERROR(
        "attention routing model disagrees with the dispatch: dispatch ran %s, "
        "select_attn_prefill_path() says %s. attention_dispatch.cu and "
        "attention_dispatch_decision.h have drifted apart (F-3) — the routing unit "
        "test is now describing a dispatch that does not exist.",
        attn_prefill_path_name(chosen), attn_prefill_path_name(modeled));
}

void attention_prefill_dispatch(const Tensor& Q, const Tensor& K, const Tensor& V, Tensor& O, float scale,
                                bool causal, int sliding_window, float softcap, cudaStream_t stream,
                                const DispatchPolicy& rcfg, int q_offset, const half* attn_sinks) {
    // Observations, filled in as the chain is walked. A tier the chain never
    // reaches keeps its `false` — see the KNOWN LIMIT above.
    AttnKernelSupport sup{};
    const bool has_sinks = (attn_sinks != nullptr);
    // Learned attention sinks (#547/#992): only the FP16 WMMA FMHA tier folds them into online
    // softmax. Route straight there; fail loudly on decline instead of a sink-blind fallback.
    if (has_sinks) {
        if (rcfg.attention.fmha_sm120 != "never" &&
            (sup.fmha_sm120_accepts = fmha_sm120_prefill(Q, K, V, O, scale, causal, sliding_window,
                                                         softcap, stream, q_offset, attn_sinks))) {
            dispatch_record::set_attn_prefill_tier(AttnPrefillPath::FMHA_SM120);
            verify_against_routing_model(rcfg, sup, has_sinks, AttnPrefillPath::FMHA_SM120);
            return;
        }
        char msg[160];
        snprintf(msg, sizeof(msg),
                 "attention_prefill_dispatch: learned sinks set but the FP16 WMMA FMHA "
                 "declined head_dim=%d (or fmha_sm120=never) — no sink-capable kernel (#992)",
                 static_cast<int>(Q.shape[3]));
        throw std::runtime_error(msg);
    }

    // MXFP4 Flash Attention: tiled FP4 E2M1 QK^T with online softmax, O(n) memory, ~4x score
    // throughput over FP16 / ~2x over FP8. Enabled via [attention] mxfp4="always". Blockscale/
    // ksmooth/pv_fp4 (#846) read from process_diag inside the launcher.
    sup.mxfp4_available = attention_mxfp4_available();
    if (sup.mxfp4_available) {
        if ((sup.mxfp4_accepts = fmha_sm120_mxfp4_prefill(Q, K, V, O, scale, causal, sliding_window,
                                                          softcap, stream,
                                                          process_diag_mxfp4_blockscale(), q_offset))) {
            dispatch_record::set_attn_prefill_tier(AttnPrefillPath::MXFP4);
            verify_against_routing_model(rcfg, sup, has_sinks, AttnPrefillPath::MXFP4);
            return;
        }
        // Fall through: head_dim not supported (e.g. < 32), use FP8/FP16 path
    }

    // Register-resident FA2 ("echtes FA"): S/P/O in registers, 1 barrier/KV tile (vs FP8 FMHA's
    // 4-barrier smem path). Default on via [attention] fmha_fa2 / IMP_FMHA_FA2, head_dim=128.
    // QK^T is FP16 unless fa2_fp16qk="never" AND fp8_fmha="on" (raw fp8 scores compound, #511).
    // #1676: "never" only restores cuBLAS below fmha_prefill_threshold; above it FA2 still runs fp16.
    const bool fa2_fp8_optin = rcfg.attention.fa2_fp16qk == "never" && rcfg.attention.fp8_fmha == "on";
    const bool fa2_opted_out = rcfg.attention.fa2_fp16qk == "never" && !fa2_fp8_optin;
    if (rcfg.attention.fmha_fa2 == "on" && !fa2_opted_out) {
        if ((sup.fa2_accepts = fmha_sm120_fa2_prefill(Q, K, V, O, scale, causal, sliding_window, softcap,
                                                     stream, q_offset,
                                                     /*fp16_qk=*/!fa2_fp8_optin))) {
            dispatch_record::set_attn_prefill_tier(AttnPrefillPath::FA2);
            verify_against_routing_model(rcfg, sup, has_sinks, AttnPrefillPath::FA2);
            IMP_LOG_DEBUG("FMHA dispatch: using FA2 register-resident kernel (hd=%d)",
                          static_cast<int>(Q.shape[3]));
            return;
        }
        // unsupported config (hd!=128) → fall through to FP16 WMMA path
    }

    // fp8-QK FMHA (smem-materializing): QK^T in raw e4m3 for 2x score throughput, but conversion
    // error compounds across layers (#511). Strictly opt-in: [attention] fp8_fmha = "on".
    if (rcfg.attention.fp8_fmha == "on") {
        sup.fp8_accepts = fmha_sm120_fp8_prefill(Q, K, V, O, scale, causal, sliding_window, softcap,
                                                 stream, q_offset);
        if (sup.fp8_accepts) {
            dispatch_record::set_attn_prefill_tier(AttnPrefillPath::FP8);
            verify_against_routing_model(rcfg, sup, has_sinks, AttnPrefillPath::FP8);
            IMP_LOG_DEBUG("FMHA dispatch: using FP8 sm120 kernel (hd=%d)", static_cast<int>(Q.shape[3]));
            return;
        }
    }

    // Native sm_120 FP16 FMHA: WMMA for Blackwell with sliding window support.
    // Fallback when FP8 is disabled or unsupported config.
    const bool use_fmha_sm120 = rcfg.attention.fmha_sm120 != "never";
    if (use_fmha_sm120) {
        if ((sup.fmha_sm120_accepts =
                 fmha_sm120_prefill(Q, K, V, O, scale, causal, sliding_window, softcap, stream, q_offset))) {
            dispatch_record::set_attn_prefill_tier(AttnPrefillPath::FMHA_SM120);
            verify_against_routing_model(rcfg, sup, has_sinks, AttnPrefillPath::FMHA_SM120);
            return;
        }
    }

    // Final tier: WMMA 128x64 tiles for Blackwell. Declines (returns false)
    // for unsupported configs — hd ∉ {64,96,128,256} or smem over the device
    // opt-in (hd=256 at Br=64 needs ~176 KB vs 99 KB on sm_120).
    if ((sup.blackwell_accepts = flash_attention_blackwell(Q, K, V, O, scale, causal, sliding_window,
                                                           softcap, stream, q_offset))) {
        dispatch_record::set_attn_prefill_tier(AttnPrefillPath::BLACKWELL);
        verify_against_routing_model(rcfg, sup, has_sinks, AttnPrefillPath::BLACKWELL);
        return;
    }

    // Chain exhausted: fail loudly rather than leave O unwritten (the old silent fallback at hd=256
    // produced garbage logits, #654). Reachable only via a disabled FP16 WMMA tier or unsupported
    // head_dim; both deserve an error, not silent corruption.
    char msg[160];
    snprintf(msg, sizeof(msg),
             "attention_prefill_dispatch: no prefill kernel accepts head_dim=%d "
             "(check attention.fmha_sm120/fmha_fa2 config) (#654)",
             static_cast<int>(Q.shape[3]));
    throw std::runtime_error(msg);
}

}  // namespace imp
