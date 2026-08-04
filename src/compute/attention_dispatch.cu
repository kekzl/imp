#include "compute/attention.h"
#include "compute/attention_tc.h"
#include "core/logging.h"
#include "core/dispatch_policy.h"
#include "runtime/process_diag.h"
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

// The path-selection ORDER + config gates below are mirrored as a pure host
// function `select_attn_prefill_path` in attention_dispatch_decision.h.
//
// That mirror used to be TEST-ONLY: test_routing_decision.cpp was its sole
// includer, so a reorder here left the test green and its stated purpose ("any
// reorder or gate change shows up as a diff") unmet — audit finding F-3.
//
// This function now CONSULTS it. Each tier records the two booleans it already
// discovers — did the config gate pass, did the kernel accept — and once a tier
// wins, the model is replayed against those observations and must name the same
// winner. A gate that drifts apart, a tier added here and not there, or a
// reorder that moves a NON-accepting tier earlier now fires a loud one-shot
// divergence log instead of nothing.
//
// KNOWN LIMIT, stated rather than implied: a tier reordered ahead of the winner
// that WOULD have accepted is still invisible, because the dispatch short-
// circuits and never asks it. Closing that needs a real `*_supports()` predicate
// per kernel that the launcher itself consults — the kernels signal acceptance
// by executing, and two of FA2's seven decline points depend on the tile
// selection, so a predicate written beside them would be a THIRD copy of the
// rules. That is a five-TU refactor of the hottest prefill kernel and is not
// done here.
// One-shot so a divergence cannot flood a serving log; the first occurrence is
// the one that matters and it names both answers.
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
    // Learned attention sinks (gpt-oss #547/#992): only the FP16 WMMA FMHA
    // tier folds them into its online softmax. Route straight there and fail
    // loudly on decline — falling through to a sink-blind kernel produces
    // silently wrong output (the pre-#992 executor WARN case).
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

    // MXFP4 Flash Attention: tiled FP4 E2M1 Q·K^T with online softmax.
    // O(n) memory, ~4x score throughput over FP16, ~2x over FP8.
    // Enabled with [attention] mxfp4 = "always". Blockscale/ksmooth/pv_fp4
    // (#846 SageAttention3-recipe spike) are read from process_diag inside
    // the launcher.
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

    // Register-resident FA2 ("echtes FA"): keeps S/P/O in registers, 1 barrier per
    // KV tile (vs the FP8 FMHA's smem-materialized S/P/O + 4 barriers). Default on
    // via [attention] fmha_fa2, env IMP_FMHA_FA2. head_dim=128. QK^T runs in f16
    // (same numerical class as cuBLAS) unless the user explicitly opts into the
    // e4m3 fp8-QK mode (fa2_fp16qk=never AND fp8_fmha=on) — raw-converted fp8
    // scores compound per layer into garbage on real activations (#511).
    if (rcfg.attention.fmha_fa2 == "on") {
        const bool fa2_fp8_optin =
            rcfg.attention.fa2_fp16qk == "never" && rcfg.attention.fp8_fmha == "on";
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

    // fp8-QK FMHA (smem-materializing): QK^T in raw-converted FP8 E4M3 for 2x
    // score throughput — but the unscaled e4m3 conversion carries ~10% relative
    // score error that compounds across layers (#511): teacher-forced PPL
    // gemma-3-12b 16.6 -> 549 / Qwen3-8B 40.5 -> 4506 when this kernel actually
    // serves prefill. Strictly opt-in: [attention] fp8_fmha = "on".
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
