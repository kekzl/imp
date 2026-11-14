#pragma once

#include "compute/dispatch_paths.h"  // MoePrefillPath

// Pure host-side model of the MoE-prefill GEMM path selection in
// executor_forward_moe_cutlass.cpp (grouped-GEMM-vs-fallback routing, PR #574/#547), so
// it's covered by a cheap CPU unit test instead of E2E-only. Only the gates pick which
// path runs; the caller supplies, per tier, whether that tier's device workspace is populated.

namespace imp {

// MoePrefillPath now lives in compute/dispatch_paths.h so the runtime recorder
// (compute/dispatch_record.h) names the tiers with the same vocabulary this
// model does — see #1205.

// Workspace-readiness flags, mirroring the device-pointer / packed-tensor
// conjunctions in the .cpp. When false, that tier's preconditions are not met
// and selection falls through to the next tier.
struct MoePrefillWorkspace {
    // try_run_moe_cutlass3x_nvfp4_prefill_ returns false (whole-function LEGACY) unless the
    // grouped NVFP4 GEMM serves the layer shapes AND packed/scale workspace + per-expert NVFP4
    // tier coverage are all present. Collapses gemm_grouped_nvfp4_moe_supports() +
    // cutlass3x_packed/sf + covers_ids(), checked before device-args, gating it too.
    bool grouped_available = false;
    bool device_args_ready = false;  // cutlass3x device-args buffers all populated
    bool grouped_ready = false;      // host-args grouped path can run this layer
    bool staged_blocks = false;      // host-expert staged prefill (ctx.staged_blocks)
};

// Reproduces the path selection in try_run_moe_cutlass3x_nvfp4_prefill_. expert_bias_glu
// (ModelProfile::moe_expert_bias_glu, gpt-oss) runs device-args unstaged (bias + GPT_OSS_GLU + plain
// quantize, #2116); staged it stays off device-args.
inline MoePrefillPath select_moe_prefill_path(bool expert_bias_glu, const MoePrefillWorkspace& ws) {
    // Function-entry gate: an unavailable/uncovered grouped NVFP4 path makes try_run_...
    // return false -> per-expert LEGACY fallback. Checked before device-args, gating
    // every grouped tier including device-args.
    if (!ws.grouped_available)
        return MoePrefillPath::LEGACY;

    // Tier 1: device-args fast path. Off for staged gpt-oss, requires the full
    // device-args workspace.
    if (!(expert_bias_glu && ws.staged_blocks) && ws.device_args_ready)
        return MoePrefillPath::DEVICE_ARGS;

    // Tier 2: host-args grouped GEMM (gpt-oss adds per-expert bias seams).
    if (ws.grouped_ready)
        return MoePrefillPath::GROUPED;

    // Tier 3: per-expert legacy fallback.
    return MoePrefillPath::LEGACY;
}

}  // namespace imp
