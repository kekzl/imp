#pragma once

// What an export cost per tensor, and the one line that says so: awq.cu computed
// SearchResult::err_rtn/err_best and nothing read them, so a 3x-better group and a 1.02x-better
// group both printed "scaled", and an uncalibrated export computed no error at all.
// Per-tensor figures are measured on the bytes actually written (packed nibbles + FP8
// micro-scales already on host after quantize_one), decoded against the FORMAT rather than
// imp's dequant kernel, so a paired quantize/dequant bug can't cancel out.

#include <cstdint>
#include <expected>
#include <string>
#include <vector>

namespace imp::quantize {

// One quantized tensor against its source.
struct TensorError {
    std::string name;
    double max_rel = 0.0;  // max |dq - src| / max(|src|) over the tensor
    double mse = 0.0;
};

// One AWQ group's objective, the number awq.cu computed and nobody read.
struct GroupError {
    int layer = 0;
    std::string kind;
    double err_rtn = 0.0;
    double err_best = 0.0;
};

// Decodes the written bytes and compares against the FP16 the quantizer was handed (packed
// [N,K/2], micro [N,K/16] FP8 E4M3). max_rel is normalised by the tensor's own absmax, not per
// element: dividing a near-zero weight's error by itself is how 4-bit reports 800% on a value
// nobody notices.
TensorError nvfp4_tensor_error(const std::string& name, const uint16_t* src_fp16, const unsigned char* packed,
                               const unsigned char* micro, float tensor_scale, int64_t N, int64_t K);

// The `limit` worst tensors by max_rel, largest first.
std::vector<TensorError> worst_tensors(std::vector<TensorError> rows, size_t limit);

// The summary line plus the worst `limit` tensors, one per line.
std::string format_error_summary(const std::vector<TensorError>& rows, size_t limit);

// The report an operator can diff between two exports.
std::string error_report_json(const std::vector<TensorError>& tensors, const std::vector<GroupError>& groups,
                              bool calibrated);

// Writes error_report_json() to <out_dir>/quant_report.json.
std::expected<void, std::string> write_error_report(const std::string& out_dir,
                                                    const std::vector<TensorError>& tensors,
                                                    const std::vector<GroupError>& groups, bool calibrated);

// The one line the tool prints before it does anything, so a checkpoint's
// provenance is in the operator's scrollback and not only in a doc.
std::string experimental_banner(bool calibrated, uint64_t samples);

}  // namespace imp::quantize
