#pragma once

// What lands in the output directory besides the tensors, and in which layout.
// imp-quantize wrote one layout (Modelopt weight/weight_scale/weight_scale_2 +
// hf_quant_config.json); vLLM's NVFP4 path is compressed-tensors, differing silently in three
// ways: (1) different tensor names (weight_packed, weight_global_scale); (2) tensor scale
// stored as a DIVISOR (vLLM computes 1.0/weight_global_scale vs Modelopt's multiply by
// weight_scale_2 - swapping conventions dequantizes every weight by amax^2/36 and still loads);
// (3) fused layers must SHARE a tensor scale (vLLM merges q/k/v and gate/up, keeps
// weight_global_scale.max() - three different scales leaves two matrices dequantized wrong,
// and vLLM only warns).
// (3) is why this isn't a rename pass: scales must be decided across tensors before any is
// quantized. Host-only, CUDA-free, so the CPU test lane covers it.

#include <cstddef>
#include <cstdint>
#include <expected>
#include <string>
#include <vector>

namespace imp::quantize {

// Which layout the checkpoint is written in.
enum class OutputFormat {
    Modelopt,           // <mod>.weight / .weight_scale / .weight_scale_2 + hf_quant_config.json
    CompressedTensors,  // <mod>.weight_packed / .weight_scale / .weight_global_scale + config.json
};

// Parse a --format argument. Returns false on an unknown name.
bool parse_output_format(const std::string& s, OutputFormat& out);

const char* format_name(OutputFormat fmt);

// The three tensor names a quantized module produces, given `base` = the module
// name without its `.weight` suffix.
struct QuantTensorNames {
    std::string packed;        // U8      [N, K/2]
    std::string micro_scale;   // F8_E4M3 [N, K/16]
    std::string global_scale;  // F32     [1]
};
QuantTensorNames quant_tensor_names(const std::string& base, OutputFormat fmt);

// The number written into global_scale, given the quantizer's scale (val = fp4 * micro_scale
// * tensor_scale). Modelopt stores it as-is and multiplies; compressed-tensors stores the
// reciprocal and divides. tensor_scale of 0 (all-zero weight) is passed through as 0 rather
// than inverted to infinity - both readers already treat that as a null layer.
float global_scale_value(float tensor_scale, OutputFormat fmt);

// Fused-layer key: shared by exactly the weights an inference engine merges into one linear
// layer, "" otherwise. Derived from vLLM's packed_modules_mapping: qkv_proj<-q/k/v_proj,
// gate_up_proj<-gate/up_proj, in_proj_qkvz<-in_proj_qkv+in_proj_z, in_proj_ba<-in_proj_b+
// in_proj_a. Keyed on the module's parent prefix, so two layers (or two MoE experts) never
// share a scale.
std::string fusion_group_key(const std::string& weight_name);

// -- tensor scales ---------------------------------------------------------

// Largest finite magnitude in a host FP16 buffer, compared as bit patterns (FP16 orders
// magnitudes monotonically once sign is masked, so this is exact integer compares, not float
// accumulation). Infinities and NaNs are skipped so one bad value can't set the whole tensor's
// scale.
float fp16_absmax(const uint16_t* data, size_t n);

// Tensor scale to quantize with, given the tensor's largest magnitude: absmax/6 leaves
// micro-scales in [0,1]. NOT what published exports use (absmax/(6*448)): that convention
// measures worse (Qwen3-0.6B, ppl_corpus_45k, deterministic_gemm: 31.05 vs 29.47 for absmax/6,
// bit-identical repeat). Both conventions are just numbers vLLM/imp multiply back out, so the
// better-measuring one wins; do not "fix" to match the published convention without
// re-measuring.
float export_tensor_scale(float absmax);

// -- config.json -----------------------------------------------------------

// The quantization_config a compressed-tensors NVFP4 checkpoint declares. `ignore` lists
// modules left at source precision (vLLM builds an unquantized layer for each, else looks for
// scales never written). input_activations is null on purpose: this tool quantizes weights
// only, and vLLM reads null as NVFP4A16; declaring activation quantization not done would make
// vLLM quantize activations at runtime against scales never measured.
std::string compressed_tensors_quant_config(const std::vector<std::string>& ignore, bool calibrated);

// Inserts or replaces the top-level quantization_config in config.json, textually rather than
// via round-trip (json_util would turn bools into numbers and reformat every float). Replacing
// rather than prepending matters: a source may already carry
// "quantization_config": null (Qwen3.8-27B does), and two same-named keys resolve to the LAST
// one, so prepending would read as unquantized.
[[nodiscard]] std::expected<std::string, std::string> patch_config_json(const std::string& src,
                                                                        const std::string& quant_config_obj);

// A combination producing a checkpoint the target engine can't load, or nullptr if nothing to
// say, reported before conversion runs. Today: --lm-head with compressed-tensors output -
// vLLM's ParallelLMHead takes no scales, so the checkpoint fails to load there (loudly, but
// only after conversion). Defensible for an imp-only checkpoint, so this warns rather than
// refuses.
const char* portability_warning(OutputFormat fmt, bool quantize_lm_head);

// -- the files themselves --------------------------------------------------

// Whether this source can carry the declaration the chosen format needs. In
// compressed-tensors mode config.json is the ONLY place the checkpoint says it's quantized, and
// it's patched rather than written from nothing - without one the output is packed nibbles any
// reader takes for unquantized. Called BEFORE the first tensor is written, not at the copy
// step, so an undeclarable source costs a second, not a full conversion.
[[nodiscard]] std::expected<void, std::string> can_declare_quantization(const std::string& in_dir,
                                                                        OutputFormat fmt);

// Everything the checkpoint needs beside the weights. The named list is imp's own needs, NOT
// the whole requirement: an allowlist quietly breaks "stays loadable by whatever the source
// was loadable by" (measured: missing preprocessor_config.json broke vLLM load entirely,
// a file imp never reads). CompressedTensors mode PATCHES config.json with the new
// quantization_config; Modelopt mode copies config.json verbatim and declares in
// hf_quant_config.json.
[[nodiscard]] std::expected<void, std::string> copy_aux_files(
    const std::string& in_dir, const std::string& out_dir, OutputFormat fmt,
    const std::vector<std::string>& excluded_modules, bool calibrated);

// hf_quant_config.json — Modelopt's declaration. Not written for
// compressed-tensors, where a second declaration in a DIFFERENT format is a way
// for the two to disagree later.
[[nodiscard]] std::expected<void, std::string> write_modelopt_quant_config(
    const std::string& out_dir, const std::vector<std::string>& excluded, bool calibrated);

// recipe.yaml, llm-compressor's record of the run, written beside the compressed-tensors
// config block. Redundant on paper, worth it in practice: readers predating
// quantization_config detection look for this file only, and reading it as Modelopt inverts
// every tensor scale silently. States the same scheme as the config block from the same
// arguments so the two cannot disagree.
[[nodiscard]] std::expected<void, std::string> write_recipe_yaml(const std::string& out_dir,
                                                                 const std::vector<std::string>& excluded);

// A sharded checkpoint is only loadable with an index, and the index cannot be
// copied from the source: quantizing one weight turns it into three tensors, so
// the name->shard map has to be rebuilt from what was actually written.
[[nodiscard]] std::expected<void, std::string> write_shard_index(
    const std::string& out_dir, const std::vector<std::pair<std::string, std::string>>& tensor_to_shard,
    size_t total_bytes);

}  // namespace imp::quantize
