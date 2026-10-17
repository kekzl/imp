#pragma once

// One weight to NVFP4: round-to-nearest (quantize_one), or GPTQ when a Hessian matches it
// (roadmap row 101, gptq.h).

#include "quant/calibration_hessian.h"

#include <cstdint>
#include <expected>
#include <map>
#include <string>
#include <vector>

namespace imp::quantize {

// Host-side packed data + scales of one quantized matrix.
struct Quantized {
    std::vector<unsigned char> packed;  // [N, K/2]
    std::vector<unsigned char> micro;   // [N, K/16] FP8 E4M3
    float tensor_scale = 1.0f;
};

// `forced_scale` > 0 quantizes against a scale decided elsewhere - the shared scale of a fused
// group. At 0 the tensor's own absmax decides.
std::expected<Quantized, std::string> quantize_one(const std::vector<uint16_t>& h_fp16, int64_t N, int64_t K,
                                                   float forced_scale);

// GPTQ over an IMPHES01 file: weights with a Hessian of their (layer, kind) and width go through
// gptq::quantize_nvfp4 with the AWQ column scales folded into H (x' = x / s), the rest through
// quantize_one.
class WeightQuantizer {
public:
    std::string open_gptq(const std::string& hess_path);
    std::expected<Quantized, std::string> quantize(
        const std::string& name, const std::vector<uint16_t>& h, int64_t N, int64_t K, float forced_scale,
        const std::map<std::string, std::vector<float>>& col_scale);
    // "" when GPTQ never ran.
    std::string summary() const;

private:
    std::string path_;
    std::vector<HessianIndexEntry> index_;
    double loss_rtn_ = 0.0, loss_gptq_ = 0.0;
    size_t n_ = 0;
};

}  // namespace imp::quantize
