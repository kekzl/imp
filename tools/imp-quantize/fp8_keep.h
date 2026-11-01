#pragma once

// Mixed NVFP4 + FP8 exports (roadmap row 101): the most quantization-sensitive dense projections
// stay FP8 E4M3 with one F32 per-tensor scale (Modelopt FP8: weight * weight_scale), the rest is
// NVFP4. A unit is one layer's q/k/v, o, gate/up or down: fused siblings change format together.
// Host-only part (unit names, selection, byte math, output layout); the device side is
// fp8_keep_run.h.

#include "model/safetensors_writer.h"

#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace imp::quantize {

// "<layer>.<group>", group qkv | o | gate_up | down, for a dense attention / MLP projection weight
// ("model.layers.27.mlp.down_proj.weight" -> "27.down"); "" for anything else (MoE experts, GDN,
// norms, embeddings, lm_head).
std::string fp8_unit_of(const std::string& weight_name);

// --fp8-keep: comma list of units, the layer may be `*` ("*.down,27.gate_up"). False sets `err`.
bool parse_fp8_keep_list(const std::string& spec, std::vector<std::string>& out, std::string& err);
bool fp8_keep_list_matches(const std::vector<std::string>& list, const std::string& unit);

// --fp8-score as a text file: one "<unit> <score>" per line, '#' comments. False sets `err`.
bool parse_fp8_rank_text(const std::string& text, std::map<std::string, double>& out, std::string& err);

// Bytes of one FP8 tensor on disk: N*K codes + one F32 scale.
size_t fp8_output_bytes(int64_t N, int64_t K);
// What FP8 costs over NVFP4 for an [N, K] weight.
size_t fp8_extra_bytes(int64_t N, int64_t K);

struct Fp8Unit {
    std::string unit;
    double score = 0.0;      // larger = more worth FP8; scores <= 0 are never picked by a budget
    size_t extra_bytes = 0;  // sum of fp8_extra_bytes over its tensors
};

// Greedy by score per extra byte, largest first; a unit that would cross `budget_bytes` is
// skipped and smaller ones may still fit. Ties keep input order.
std::set<std::string> select_fp8_units(std::vector<Fp8Unit> units, size_t budget_bytes);

// Modelopt MIXED_PRECISION quantized_layers object: module -> {"quant_algo": "NVFP4", "group_size":
// 16} or {"quant_algo": "FP8"}; `indent` prefixes every line after the first.
std::string mixed_precision_layers_json(const std::vector<std::string>& nvfp4_modules,
                                        const std::vector<std::string>& fp8_modules,
                                        const std::string& indent);

// The two tensors an FP8 weight becomes: <base>.weight F8_E4M3 [N, K] and <base>.weight_scale
// F32 [1]. Pointers are borrowed until the shard is written.
std::vector<SafeTensorsOut> fp8_weight_outputs(const std::string& weight_name, int64_t N, int64_t K,
                                               const uint8_t* codes, const float* scale);

}  // namespace imp::quantize
