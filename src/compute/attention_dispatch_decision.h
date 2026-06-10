#pragma once

#include "runtime/config.h"

// Pure host-side model of the attention-prefill dispatch ordering in
// attention_dispatch.cu. Extracted so the routing table (which kernel serves a
// given config + head_dim) is covered by a cheap CPU unit test — the #493
// regression was exactly an unintended routing shift, and the 06-04 test audit
// flagged this as "cheap, catches path moves" (R2 / P1.4).
//
// The .cu dispatch interleaves three things:
//   1. config gates    (rcfg.attention.fmha_fa2 == "on", fp8_fmha != "never", …)
//   2. kernel support   (the kernel returns false for hd!=128, insufficient
//                        smem, non-F16, etc. and the dispatch falls through)
//   3. the actual kernel launch
//
// Only (1)+(2) decide *which* path runs; (3) is the device work. This header
// models (1)+(2) as a pure function: the caller supplies, per kernel, whether
// that kernel would *accept* the (Q,K,V,O) config (the boolean the real
// `fmha_..._prefill(...)` return value collapses to). The function then
// reproduces the exact short-circuit order of attention_dispatch.cu.
//
// Keeping this in lock-step with attention_dispatch.cu is the point: any reorder
// or gate change shows up as a diff in test_routing_decision.cpp.

namespace imp {

enum class AttnPrefillPath {
    MXFP4,        // fmha_sm120_mxfp4_prefill
    FA2,          // fmha_sm120_fa2_prefill (register-resident)
    FP8,          // fmha_sm120_fp8_prefill
    FMHA_SM120,   // fmha_sm120_prefill (WMMA)
    BLACKWELL,    // flash_attention_blackwell (final tier; declines unsupported configs)
    NONE,         // chain exhausted → attention_prefill_dispatch throws (#654)
};

// Per-kernel "would this kernel accept the config" flags, mirroring the bool
// each `fmha_..._prefill(...)` collapses to in the .cu (mxfp4 also has an outer
// availability gate). Defaults reflect the common hd=128 F16 case where the
// specialized kernels accept.
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

// Reproduces attention_prefill_dispatch()'s path selection (config gates +
// fall-through on kernel decline). Returns the path that would actually run.
inline AttnPrefillPath select_attn_prefill_path(const RuntimeConfig& rcfg,
                                                const AttnKernelSupport& sup) {
    // 1. MXFP4 Flash Attention (opt-in, outer availability + per-config accept).
    if (sup.mxfp4_available && sup.mxfp4_accepts)
        return AttnPrefillPath::MXFP4;

    // 2. Register-resident FA2 — only when [attention] fmha_fa2 == "on".
    if (rcfg.attention.fmha_fa2 == "on" && sup.fa2_accepts)
        return AttnPrefillPath::FA2;

    // 3. fp8-QK FMHA — strictly opt-in (== "on"). Raw e4m3 Q/K conversion
    //    compounds ~10% relative score error per layer on real activations
    //    (#511): teacher-forced PPL gemma-3-12b 16.6→549, Qwen3-8B 40.5→4506
    //    when this kernel actually serves prefill. Default routes hd!=128
    //    to the FP16 WMMA kernel below instead.
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
