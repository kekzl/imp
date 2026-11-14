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
#include "compute/dispatch_record.h"
#include "compute/attention_dispatch_decision.h"

namespace imp {

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
        "select_attn_prefill_path() says %s. attention_dispatch.cpp and "
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
    // Learned attention sinks (#547/#992): fp16-qk FA2 and the FP16 WMMA FMHA fold them into online
    // softmax. Route only there; fail loudly on decline instead of a sink-blind fallback.
    if (has_sinks) {
        if (rcfg.attention.fmha_fa2 == "on") {
            sup.fa2_accepts = fmha_sm120_fa2_prefill(Q, K, V, O, scale, causal, sliding_window, softcap,
                                                     stream, q_offset, /*d_kv_len=*/nullptr, attn_sinks);
        }
        if (sup.fa2_accepts) {
            dispatch_record::set_attn_prefill_tier(AttnPrefillPath::FA2);
            verify_against_routing_model(rcfg, sup, has_sinks, AttnPrefillPath::FA2);
            return;
        }
        sup.fmha_sm120_accepts = fmha_sm120_prefill(Q, K, V, O, scale, causal, sliding_window, softcap,
                                                    stream, q_offset, attn_sinks);
        if (sup.fmha_sm120_accepts) {
            dispatch_record::set_attn_prefill_tier(AttnPrefillPath::FMHA_SM120);
            verify_against_routing_model(rcfg, sup, has_sinks, AttnPrefillPath::FMHA_SM120);
            return;
        }
        char msg[160];
        snprintf(msg, sizeof(msg),
                 "attention_prefill_dispatch: learned sinks set but FA2 and the FP16 WMMA FMHA "
                 "declined head_dim=%d (or both are off) - no sink-capable kernel (#992)",
                 static_cast<int>(Q.shape[3]));
        throw std::runtime_error(msg);
    }

    // Register-resident FA2 ("echtes FA"): S/P/O in registers, 1 barrier/KV tile, f16 QK^T.
    // Default on via [attention] fmha_fa2 / IMP_FMHA_FA2, head_dim 64/128/256.
    if (rcfg.attention.fmha_fa2 == "on") {
        sup.fa2_accepts = fmha_sm120_fa2_prefill(Q, K, V, O, scale, causal, sliding_window, softcap, stream,
                                                 q_offset);
        if (sup.fa2_accepts) {
            dispatch_record::set_attn_prefill_tier(AttnPrefillPath::FA2);
            verify_against_routing_model(rcfg, sup, has_sinks, AttnPrefillPath::FA2);
            IMP_LOG_DEBUG("FMHA dispatch: using FA2 register-resident kernel (hd=%d)",
                          static_cast<int>(Q.shape[3]));
            return;
        }
        // unsupported config (head_dim, smem) → fall through to FP16 WMMA path
    }

    // Native sm_120 FP16 FMHA: WMMA for Blackwell with sliding window support.
    sup.fmha_sm120_accepts = fmha_sm120_prefill(Q, K, V, O, scale, causal, sliding_window, softcap, stream,
                                                q_offset);
    if (sup.fmha_sm120_accepts) {
        dispatch_record::set_attn_prefill_tier(AttnPrefillPath::FMHA_SM120);
        verify_against_routing_model(rcfg, sup, has_sinks, AttnPrefillPath::FMHA_SM120);
        return;
    }

    // Final tier: WMMA 128x64 tiles for Blackwell. Declines (returns false)
    // for unsupported configs — hd ∉ {64,96,128,256} or smem over the device
    // opt-in (hd=256 at Br=64 needs ~176 KB vs 99 KB on sm_120).
    sup.blackwell_accepts = flash_attention_blackwell(Q, K, V, O, scale, causal, sliding_window, softcap,
                                                      stream, q_offset);
    if (sup.blackwell_accepts) {
        dispatch_record::set_attn_prefill_tier(AttnPrefillPath::BLACKWELL);
        verify_against_routing_model(rcfg, sup, has_sinks, AttnPrefillPath::BLACKWELL);
        return;
    }

    // Chain exhausted: fail loudly rather than leave O unwritten (the old silent fallback at hd=256
    // produced garbage logits, #654). Reachable only via an unsupported head_dim; an error, not
    // silent corruption.
    char msg[320];
    snprintf(msg, sizeof(msg),
             "attention_prefill_dispatch: no prefill kernel accepts head_dim=%d "
             "(check attention.fmha_fa2 config) (#654): Q [%d, %d, %d] K [%d, %d] q_offset %d "
             "qtype %d; fa2 %d (fmha_fa2=%s hd256=%d) fmha_sm120 %d blackwell %d",
             static_cast<int>(Q.shape[3]), static_cast<int>(Q.shape[0]), static_cast<int>(Q.shape[1]),
             static_cast<int>(Q.shape[2]), static_cast<int>(K.shape[1]), static_cast<int>(K.shape[2]),
             q_offset, static_cast<int>(Q.qtype), (int)sup.fa2_accepts, rcfg.attention.fmha_fa2.c_str(),
             (int)imp::process_diag_fa2_hd256(), (int)sup.fmha_sm120_accepts, (int)sup.blackwell_accepts);
    throw std::runtime_error(msg);
}

}  // namespace imp
