#pragma once

// AWQ-class scale search for the NVFP4 quantizer.
//
// The idea, in one line: NVFP4 error is proportional to the magnitude of what
// it quantizes, so multiplying an input channel's weights UP before quantizing
// and dividing the activation DOWN afterwards buys that channel precision at
// the expense of the others. Which channels deserve it is not visible in the
// weights — it takes a forward pass, which is what the calibration file holds.
//
// The transform is exact before quantization:
//
//     y = x W^T = (x / s) (W diag(s))^T
//
// so the only question is which s minimises the error AFTER quantizing
// W diag(s). We answer it by measurement, not by a closed form: quantize with
// the real kernel for each candidate and keep the winner. alpha = 0 (s = 1,
// plain round-to-nearest) is always in the grid, so a group where scaling does
// not pay keeps the untransformed weights.
//
// The objective is the activation-weighted reconstruction error
//
//     err = sum_ij  a_j^2 * (W_ij - Wq_ij / s_j)^2
//
// with a_j the mean |activation| of input channel j. That is the diagonal
// approximation of the output MSE — exact if input channels were uncorrelated,
// and the same approximation the AWQ paper's scale definition rests on.

#include "model/safetensors_raw.h"
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

    int groups_scaled = 0;   // a non-zero alpha won
    int groups_rtn = 0;      // alpha = 0 won, weights left alone
    int groups_skipped = 0;  // missing tensors or missing calibration
    // Switched off via the group selector (own line: a longer name here would
    // re-align the comment column of every field above it).
    int groups_disabled = 0;
    std::vector<std::string> notes;
};

// The four groups of awq_plan.cpp, selectable so a bad result can be attributed
// to one of them instead of to "calibration". Default is all four.
//
// This exists because --calib measurably HELPS Qwen3-0.6B/1.7B and measurably
// HURTS Qwen3-14B (9.93 -> 12.60 PPL). Measured 2026-08-05, the cause is the
// ATTENTION pair on wide GQA, and mostly their INTERACTION rather than either
// group alone: on the 14B (n_rep=5) C alone is +0.02 and A alone +0.65, but
// A x C adds a further +1.36. The FFN groups are clean — BD scores -0.13, i.e.
// better than round-to-nearest, and BD x C is +0.03. At n_rep=2 the same
// interaction is +0.05 and the full set wins.
//
// The n_rep dependence sits in C: its statistic must be constant across the
// query heads sharing a KV head, the tie is a max, and that statistic is the
// WEIGHT in the search objective — so a wide GQA group makes the search
// minimise the wrong function faithfully.
//
// So this is a production switch, not only a diagnostic: prefer "BD" on
// wide-GQA models and the default on narrow-GQA ones.
constexpr const char* kAwqAllGroups = "ABCD";

// Builds the transform from a calibration file and the checkpoint's own
// config.json. `groups` selects which of A/B/C/D run. Returns false with `err`
// set when the architecture is not one whose pre-norm layout this transform is
// valid for.
[[nodiscard]] std::expected<Plan, std::string> build_plan(
    const std::map<std::string, const RawTensor*>& index, const CalibrationStats& stats,
    const std::string& config_json_path, const std::string& groups);

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

// Searches alpha over [0, 1] and returns the winning scale vector.
// All matrices must share the inner dimension K. `act_mean` is [K].
// Returns the error text on GPU failure or bad shapes.
[[nodiscard]] std::expected<SearchResult, std::string> search_group_scale(
    const std::vector<GroupMatrix>& mats, int64_t K, const std::vector<float>& act_mean);

}  // namespace imp::awq
