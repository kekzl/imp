#pragma once

// AWQ-class scale search for the NVFP4 quantizer: NVFP4 error is proportional to magnitude, so
// scaling an input channel's weights UP before quantizing and the activation DOWN after buys
// that channel precision at the expense of others. Exact identity: y = xW^T = (x/s)(W diag(s))^T.
// s is chosen by measurement (quantize with the real kernel per candidate, keep the winner);
// alpha=0 (s=1, plain RTN) is always in the grid.
// Objective: err = sum_ij a_j^2 (W_ij - Wq_ij/s_j)^2, a_j = mean|activation| of channel j - the
// diagonal approximation of output MSE (exact if channels were uncorrelated), same as the AWQ paper.

#include "awq_sites.h"
#include "quant_report.h"

#include "model/safetensors_raw.h"
#include "quant/awq_norm_fold.h"
#include "quant/calibration_stats.h"

#include <cstdint>
#include <expected>
#include <map>
#include <string>
#include <vector>

namespace imp::awq {

// BF16/F16 raw tensor -> host FP16 bits. Shared by the plan builder and the
// writer so both see bit-identical inputs.
std::vector<uint16_t> raw_to_fp16(const RawTensor& t);

// What the quantizer must do to each tensor before writing it. Empty maps mean
// "no transform", which is what a checkpoint gets when round-to-nearest wins
// every group.
struct Plan {
    // W[i][j] *= col_scale[j]  — the quantized matrices of a group.
    std::map<std::string, std::vector<float>> col_scale;
    // W[i][j] /= row_div[i]    — the producer whose output channels feed them.
    std::map<std::string, std::vector<float>> row_div;
    // v[j] /= vec_div[j]       — 1-D producers: RMSNorm weights and biases.
    std::map<std::string, std::vector<float>> vec_div;
    // Which convention a folded 1-D producer follows. A unit-offset RMSNorm
    // stores (1 + g)/s - 1, not g/s: dividing the delta of a (1 + g) norm
    // divides the wrong quantity. Absent means plain, which is what a bias is.
    std::map<std::string, NormOffset> vec_offset;

    // Channels whose divisor the storage bound pulled back toward 1 because the folded value
    // couldn't represent the intended gain in the tensor's own dtype (quant/awq_norm_fold.h).
    // Never a deleted channel, just a smaller scale where BF16 can't carry the fold.
    int channels_clamped = 0;
    // The search objective per group, which the tool computed and discarded
    // until the export report existed.
    std::vector<quantize::GroupError> group_errors;

    int groups_scaled = 0;   // a non-zero alpha won
    int groups_rtn = 0;      // alpha = 0 won, weights left alone
    int groups_skipped = 0;  // missing tensors or missing calibration
    // Switched off via the group selector (own line: a longer name here would
    // re-align the comment column of every field above it).
    int groups_disabled = 0;
    std::vector<std::string> notes;
};

// The four groups of awq_plan.cpp, selectable so a bad result can be attributed to one instead
// of "calibration". Default is all four (ABCD).
// --calib measurably HELPS Qwen3-0.6B/1.7B and HURTS wide-GQA Qwen3-14B (9.93->12.60 PPL): the
// harm is the ATTENTION groups (A, C) and mostly their interaction (A alone +0.65, C alone
// +0.02, AxC +1.36 at n_rep=5); FFN groups (BD) are clean (BD -0.13, BDxC +0.03). Prefer "BD"
// on wide-GQA models, the default on narrow-GQA ones.
// C's n_rep dependence: its statistic (a max, tied across query heads sharing a KV head) is the
// search's WEIGHT, so wide GQA makes the search minimise the wrong function.
// E and G are the GDN sites of qwen3_5 hybrids (awq_sites.h), absent elsewhere so they change
// nothing on a dense model; NOT yet measured on a hybrid - score any export using them against
// its uncalibrated twin first.
constexpr const char* kAwqAllGroups = "ABCDEG";

// Builds the transform from a calibration file and the checkpoint's own config.json. `groups`
// selects which of A/B/C/D run. Returns false with `err` set when the architecture's pre-norm
// layout isn't one this transform is valid for.
[[nodiscard]] std::expected<Plan, std::string> build_plan(
    const std::map<std::string, const RawTensor*>& index, const CalibrationStats& stats,
    const std::string& config_json_path, const std::string& groups, bool weight_sq = false);

// One matrix of a scale group, host-side FP16 bits, row-major [N, K].
struct GroupMatrix {
    const std::vector<uint16_t>* data = nullptr;
    int64_t N = 0;
};

struct SearchResult {
    std::vector<float> s;   // [K] per-input-channel scale, geometric-mean normalised
    float alpha = 0.0f;     // winning exponent; 0 means "round-to-nearest won"
    double err_rtn = 0.0;   // objective at alpha = 0
    double err_best = 0.0;  // objective at the winning alpha
};

// Searches alpha over [0,1], returns the winning scale vector. All matrices share inner
// dimension K; act_mean is [K]. act_sq is the per-channel second moment E[x^2], may be empty,
// in which case the search falls back to (act_mean/s)^2 (pre-IMPCAL02 behaviour).
[[nodiscard]] std::expected<SearchResult, std::string> search_group_scale(
    const std::vector<GroupMatrix>& mats, int64_t K, const std::vector<float>& act_mean,
    const std::vector<float>& act_sq = {});

}  // namespace imp::awq
