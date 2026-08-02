#pragma once

#include "compute/dispatch_paths.h"  // MoePrefillPath
#include "model/model_arch.h"
#include "runtime/config.h"

// Pure host-side model of the MoE-prefill GEMM path selection in
// executor_forward_moe_cutlass.cu (and its precondition mirror
// moe_cutlass3x_will_use_device_args_ in executor_forward_moe.cu). Extracted so
// the grouped-GEMM-vs-fallback routing — added in PR #574 (#547, gpt-oss
// CUTLASS grouped prefill) — is covered by a cheap CPU unit test instead of
// E2E-only (R2 / P1.4).
//
// The .cu interleaves config gates + arch gates + workspace-readiness probes +
// the actual dispatch. Only the gates pick *which* path runs; this header
// models them as a pure function. The caller supplies, per tier, whether the
// device workspace for that tier is populated (the boolean the long
// `moe_.d_M_per && moe_.cutlass3x_packed && …` conjunction collapses to).

namespace imp {

// MoePrefillPath now lives in compute/dispatch_paths.h so the runtime recorder
// (compute/dispatch_record.h) names the tiers with the same vocabulary this
// model does — see #1205.

// Workspace-readiness flags, mirroring the device-pointer / packed-tensor
// conjunctions in the .cu. When false, that tier's preconditions are not met
// and selection falls through to the next tier.
struct MoePrefillWorkspace {
    // try_run_moe_cutlass3x_nvfp4_prefill_ returns false at entry (→ whole-
    // function LEGACY) unless the CUTLASS 3.x grouped path is available AND the
    // packed/scale workspace + per-expert NVFP4 tier coverage are all present.
    // This single flag collapses cutlass_grouped_3x_nvfp4_available() +
    // cutlass3x_packed/sf + covers_ids() — all checked BEFORE the device-args
    // block, so it gates device-args too.
    bool grouped_available = false;
    bool device_args_ready = false;  // cutlass3x device-args buffers all populated
    bool smallM_available = false;   // gemm_grouped_nvfp4_smallM_available()
    bool smallM_under_threshold = false;  // max active M <= nvfp4_smallM_threshold
    bool grouped_ready = false;      // host-args grouped path can run this layer
};

// Reproduces the path selection in try_run_moe_cutlass3x_nvfp4_prefill_.
// gpt-oss (#574/#547) is arch-gated OFF the device-args and smallM tiers (the
// fused act+quantize kernel has no GLU-clamp / per-expert-bias hooks) — it
// takes the GROUPED tier, which applies the bias seams + GPT_OSS_GLU activation.
inline MoePrefillPath select_moe_prefill_path(ModelArch arch, const RuntimeConfig& rcfg,
                                              const MoePrefillWorkspace& ws) {
    const bool is_gpt_oss = (arch == ModelArch::GPT_OSS);

    // Function-entry gate: moe.no_cutlass3x or an unavailable/uncovered CUTLASS
    // 3.x grouped path makes the whole try_run_... return false → per-expert
    // LEGACY fallback. Checked BEFORE device-args, so it gates every CUTLASS
    // tier including device-args.
    if (rcfg.moe.no_cutlass3x || !ws.grouped_available)
        return MoePrefillPath::LEGACY;

    // Tier 1: device-args fast path. Gated by moe.nvfp4_device_args, off for
    // gpt-oss, requires the full device-args workspace.
    if (rcfg.moe.nvfp4_device_args && !is_gpt_oss && ws.device_args_ready)
        return MoePrefillPath::DEVICE_ARGS;

    // Tier 2: smallM path. Opt-in (moe.nvfp4_smallM), off for gpt-oss, only
    // when the kernel is available and the active token count is under the
    // smallM threshold.
    if (rcfg.moe.nvfp4_smallM && !is_gpt_oss && ws.smallM_available && ws.smallM_under_threshold)
        return MoePrefillPath::SMALL_M;

    // Tier 3: host-args grouped GEMM (gpt-oss adds per-expert bias seams).
    if (ws.grouped_ready)
        return MoePrefillPath::GROUPED;

    // Tier 4: per-expert legacy fallback.
    return MoePrefillPath::LEGACY;
}

}  // namespace imp
