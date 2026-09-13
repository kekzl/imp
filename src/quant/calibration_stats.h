#pragma once

// Activation-calibration statistics the engine writes and imp-quantize reads: one entry
// per (layer, weight kind), mean_abs[j] = (1/rows)*sum|x[row][j]| per input channel -
// the whole input AWQ-class calibration needs, and the only signal a forward pass gives an
// offline quantizer beyond the weights themselves. Deliberately a per-channel diagonal
// statistic, not a per-tensor Hessian: keeps a 14B-model calibration file in the low tens
// of MiB. Since IMPCAL02 also carries mean_sq[j] = (1/rows)*sum x[row][j]^2, because that
// (not mean_abs^2) weights the layer's output error via E[(sum dw_j x_j)^2] =
// sum dw_j^2 E[x_j^2]; using (mean_abs/s)^2 under-weighted heavy-tailed channels. An
// IMPCAL01 file reads back with mean_sq empty and the caller falls back to the old weight.

#include <cstdint>
#include <string>
#include <vector>

namespace imp {

struct CalibrationEntry {
    int layer = -1;               // transformer layer index
    std::string kind;             // tensor_kind_name(), e.g. "WQ" / "W_DOWN"
    uint64_t rows = 0;            // tokens accumulated into this entry
    std::vector<float> mean_abs;  // [K] per input channel
    std::vector<float> mean_sq;   // [K] per input channel, empty when read from an IMPCAL01 file
};

struct CalibrationStats {
    std::string model_id;  // free-form provenance string
    std::vector<CalibrationEntry> entries;

    const CalibrationEntry* find(int layer, const std::string& kind) const;
};

// Both return an empty string on success, a human-readable error otherwise.
std::string write_calibration_stats(const std::string& path, const CalibrationStats& stats);
std::string read_calibration_stats(const std::string& path, CalibrationStats& out);

}  // namespace imp
