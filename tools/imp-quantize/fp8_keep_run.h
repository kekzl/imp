#pragma once

// Device side of mixed NVFP4 + FP8 exports (fp8_keep.h): which units go FP8 (--fp8-keep,
// --fp8-budget + --fp8-score), the E4M3 codes, and the MIXED_PRECISION quantized_layers table.

#include "options.h"

#include "model/safetensors_raw.h"
#include "model/safetensors_writer.h"
#include "quant/calibration_stats.h"

#include <cstdint>
#include <deque>
#include <expected>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace imp::quantize {

struct Fp8Weight {
    std::vector<uint8_t> codes;  // [N, K] E4M3, code * scale = weight
    float scale = 1.0f;          // absmax / 448
    float input_scale = 0.0f;    // static activation scale from --calib max|x|, 0 = none written
};

// Per-tensor E4M3 through the runtime's quantize_fp16_to_fp8_e4m3_scaled.
[[nodiscard]] std::expected<Fp8Weight, std::string> quantize_fp8_tensor(const std::vector<uint16_t>& h_fp16,
                                                                        int64_t N, int64_t K);

class Fp8Keep {
public:
    using Fp16Of = std::function<std::expected<std::vector<uint16_t>, std::string>(const RawTensor&)>;
    using Pred = std::function<bool(const RawTensor&)>;
    using ForcedOf = std::function<float(const std::string&)>;

    // Picks the FP8 units. `nvfp4_target` = would this tensor be written as NVFP4; `fp16_of` = the
    // FP16 the writer quantizes (AWQ folded); `forced_of` = a fused group's shared NVFP4 scale.
    // "" on success or when no FP8 option is set.
    std::string plan(const Options& opt, const std::vector<std::unique_ptr<RawSafeTensors>>& opened,
                     const Pred& nvfp4_target, Fp16Of fp16_of, const ForcedOf& forced_of,
                     const std::map<std::string, std::vector<float>>& col_scale);
    bool active() const { return !units_.empty(); }
    bool selected(const std::string& weight_name) const;
    // Writes `t` as FP8 into `out`; the codes live until release_shard(). False = failed (printed).
    bool emit(const RawTensor& t, bool dry_run, std::vector<SafeTensorsOut>& out, size_t& bytes_out);
    void release_shard() { store_.clear(); }
    void note_nvfp4(const std::string& module) { nvfp4_modules_.push_back(module); }
    // The MIXED_PRECISION quantized_layers object, "" when no unit is FP8.
    std::string quantized_layers_json(const std::string& indent) const;

private:
    std::string score_units(const Options& opt,
                            const std::map<std::string, std::vector<const RawTensor*>>& members,
                            const ForcedOf& forced_of,
                            const std::map<std::string, std::vector<float>>& col_scale,
                            std::map<std::string, double>& scores);

    std::set<std::string> units_;
    Fp16Of fp16_of_;
    std::deque<Fp8Weight> store_;  // deque: push_back keeps earlier codes in place
    CalibrationStats calib_;       // --calib, for the static input_scale
    const std::map<std::string, std::vector<float>>* col_scale_ = nullptr;
    size_t n_dynamic_ = 0;  // FP8 weights written without an input_scale
    float static_input_scale(const std::string& weight_name, int64_t K) const;
    std::vector<std::string> fp8_modules_, nvfp4_modules_;
};

}  // namespace imp::quantize
