#pragma once

// Activation-calibration statistics: the file the engine writes and
// imp-quantize reads.
//
// One entry per (layer, weight kind) that the forward pass fed through
// gemm_via_handle_, holding the mean absolute activation per INPUT channel:
//
//     mean_abs[j] = (1/rows) * sum_over_rows |x[row][j]|
//
// That vector is the whole input AWQ-class calibration needs. It is what
// says which input channels a quantizer must protect, and it is the only
// thing a forward pass can tell an offline quantizer that the weights
// cannot say by themselves.
//
// Deliberately NOT a per-tensor Hessian: a diagonal (per-channel) statistic
// is what the scale search below uses, and storing K floats per weight keeps
// a calibration file for a 14B model in the low tens of MiB.

#include <cstdint>
#include <string>
#include <vector>

namespace imp {

struct CalibrationEntry {
    int layer = -1;               // transformer layer index
    std::string kind;             // tensor_kind_name(), e.g. "WQ" / "W_DOWN"
    uint64_t rows = 0;            // tokens accumulated into this entry
    std::vector<float> mean_abs;  // [K] per input channel
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
