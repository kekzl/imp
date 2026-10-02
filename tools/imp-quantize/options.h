#pragma once

// imp-quantize command line: what one run is asked to do.

#include "checkpoint_out.h"
#include "fp8_head.h"

#include <optional>
#include <string>

namespace imp::quantize {

struct Options {
    std::string in_dir, out_dir;
    std::string calib_file;  // --calib: activation statistics for AWQ scaling
    // --calib-groups: which AWQ scale groups run, for attributing a bad result.
    // Empty = awq::default_groups(n_rep, hybrid), resolved in build_plan.
    std::string calib_groups;
    // AWQ search error weight: "abs" (mean|x|/s)^2, "sq" E[x^2]/s^2 (calibration_stats.h).
    // sq: Qwen3-0.6B ABCD -1.10 % PPL, Qwen3-14B BD +0.15 PPL; default abs.
    bool calib_weight_sq = false;
    bool quantize_lm_head = false;  // lm_head == Nvfp4: quantized like any Linear
    // --lm-head: unset = default_lm_head_export(format), FP8 per-row for modelopt (#2479).
    LmHeadExport lm_head = LmHeadExport::Source;
    bool lm_head_set = false;
    KvHint kv_hint = KvHint::Auto;  // --kv-hint (#2480)
    // Keep a fused Q+gate q_proj out of NVFP4. OFF by default: the gate half carries the #1273
    // divergence, but excluding it measured ~1.5% WORSE perplexity end to end. Opt-in since the
    // trade may still favor models with a higher gate share than the one checkpoint measured.
    bool keep_attn_gate = false;
    // Keep GDN linear_attn projections at source precision ("all", or a list of in|gate|out; the
    // Qwen3.6-35B recipe keeps all three), so the runtime's F16 GDN prefill path and
    // gemm.nvfp4_gdn_proj_prefill apply to the export. Empty: quantize them like any Linear.
    std::string keep_gdn_proj;
    bool dry_run = false;
    OutputFormat format = OutputFormat::Modelopt;
};

// Command line into `opt`; a value = exit with it (usage error or --help), nullopt = run.
std::optional<int> parse_args(int argc, char** argv, Options& opt);

}  // namespace imp::quantize
