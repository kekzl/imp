#pragma once

#include "compute/dispatch_paths.h"  // AttnPrefillPath
#include "core/dispatch_policy.h"

// Pure host-side model of attention_dispatch.cu's prefill routing order, covered by a cheap CPU
// unit test (test_routing_decision.cpp) so a routing-order regression like #493 is caught there.
// Models config gates + kernel-accept short-circuit only; the actual launch is out of scope.
// Any reorder or gate change in attention_dispatch.cu must show up as a diff here.

namespace imp {

// AttnPrefillPath now lives in compute/dispatch_paths.h so the runtime recorder
// (compute/dispatch_record.h) names the tiers with the same vocabulary this
// model does — see #1205. Renaming a tier breaks both sides together.

// Per-kernel "would this kernel accept the config" flags, mirroring the bool each
// fmha_..._prefill() collapses to (mxfp4 also has an outer availability gate). Defaults reflect
// the common hd=128 F16 case where the specialized kernels accept.
struct AttnKernelSupport {
    bool mxfp4_available = false;   // attention_mxfp4_available()
    bool mxfp4_accepts = false;     // fmha_sm120_mxfp4_prefill(...) succeeded
    bool fa2_accepts = false;       // fmha_sm120_fa2_prefill(...) succeeded
    bool fp8_accepts = false;       // fmha_sm120_fp8_prefill(...) succeeded
    bool fmha_sm120_accepts = false;// fmha_sm120_prefill(...) succeeded
    bool blackwell_accepts = false; // flash_attention_blackwell(...) succeeded
                                    // (declines hd ∉ {64,96,128,256} and
                                    // smem-over-limit configs, e.g. hd=256)
};

// Reproduces attention_prefill_dispatch()'s path selection (config gates + kernel-decline
// fall-through). has_sinks mirrors the #992 pre-gate: learned sinks route straight to the FP16
// WMMA FMHA (only sink-capable tier); dispatch throws on decline instead of falling through.
inline AttnPrefillPath select_attn_prefill_path(const DispatchPolicy& rcfg, const AttnKernelSupport& sup,
                                                bool has_sinks = false) {
    // 0. Learned sinks (#992): FP16 WMMA FMHA or nothing.
    if (has_sinks) {
        if (rcfg.attention.fmha_sm120 != "never" && sup.fmha_sm120_accepts)
            return AttnPrefillPath::FMHA_SM120;
        return AttnPrefillPath::NONE;  // dispatcher throws (silent-wrong guard)
    }

    // 1. MXFP4 Flash Attention (opt-in, outer availability + per-config accept).
    if (sup.mxfp4_available && sup.mxfp4_accepts)
        return AttnPrefillPath::MXFP4;

    // 2. Register-resident FA2 — only when [attention] fmha_fa2 == "on".
    if (rcfg.attention.fmha_fa2 == "on" && sup.fa2_accepts)
        return AttnPrefillPath::FA2;

    // fp8-QK FMHA: strictly opt-in (== "on"). Raw e4m3 Q/K conversion compounds relative score
    // error per layer on real activations (#511). Default routes hd!=128 to the FP16 WMMA kernel.
    if (rcfg.attention.fp8_fmha == "on" && sup.fp8_accepts)
        return AttnPrefillPath::FP8;

    // 4. Native FP16 WMMA FMHA — ON unless fmha_sm120 == "never".
    if (rcfg.attention.fmha_sm120 != "never" && sup.fmha_sm120_accepts)
        return AttnPrefillPath::FMHA_SM120;

    // 5. Final tier: WMMA 128x64 Blackwell flash attention (no config gate,
    //    but declines unsupported configs — see AttnKernelSupport).
    if (sup.blackwell_accepts)
        return AttnPrefillPath::BLACKWELL;

    // 6. Chain exhausted: the dispatcher throws instead of producing garbage
    //    via an unchecked launch (#654).
    return AttnPrefillPath::NONE;
}

}  // namespace imp
