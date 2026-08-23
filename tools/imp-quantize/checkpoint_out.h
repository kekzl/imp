#pragma once

// What lands in the output directory besides the tensors, and in which layout.
//
// Writing a checkpoint OTHER engines can read.
//
// imp-quantize wrote one layout: Modelopt's `weight` / `weight_scale` /
// `weight_scale_2` plus an `hf_quant_config.json`. imp reads it, and nothing
// else does — vLLM's NVFP4 path is compressed-tensors, whose contract differs
// in three ways that are each silent when wrong:
//
//   1. Different tensor names (`weight_packed`, `weight_global_scale`).
//   2. The tensor scale is stored as a DIVISOR. vLLM computes
//      `1.0 / weight_global_scale` at load ("CT stores as divisors", see
//      compressed_tensors_w4a4_nvfp4.py) where Modelopt multiplies by
//      `weight_scale_2`. Writing one convention's number under the other's name
//      dequantizes every weight by amax²/36 and the checkpoint still loads.
//   3. Fused layers must SHARE a tensor scale. vLLM merges q/k/v into one
//      `qkv_proj` and gate/up into one `gate_up_proj`, then keeps
//      `weight_global_scale.max()` for the merged layer — so a checkpoint whose
//      q, k and v carry three different scales has two of them dequantized with
//      the wrong one. vLLM warns ("consider using a checkpoint with a shared
//      global NVFP4 scale for fused layers") and continues.
//
// (3) is why this is not a rename pass: the scales have to be decided across
// tensors before any of them is quantized, which is a change to the write
// order, not to the writer.
//
// Everything here is host-only and free of CUDA so the CPU test lane can cover
// it — these are exactly the rules that produce a checkpoint which loads and is
// quietly wrong.

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

// The number written into `global_scale`, given the scale the quantizer used
// (`val = fp4 * micro_scale * tensor_scale`).
//
// Modelopt stores it as-is and multiplies. compressed-tensors stores the
// reciprocal and divides. A tensor_scale of 0 (an all-zero weight) would make
// the reciprocal infinite, so it is passed through as 0 — which both readers
// already treat as a null layer (nvfp4_promote_weight_scale_2).
float global_scale_value(float tensor_scale, OutputFormat fmt);

// -- Fused layers ----------------------------------------------------------
//
// A key shared by exactly those weights an inference engine merges into one
// linear layer, or "" when the weight is not part of such a group. Derived from
// vLLM's `packed_modules_mapping`, which is the list that decides this:
//
//   qkv_proj      <- q_proj, k_proj, v_proj
//   gate_up_proj  <- gate_proj, up_proj
//   in_proj_qkvz  <- in_proj_qkv, in_proj_z      (GDN / linear attention)
//   in_proj_ba    <- in_proj_b, in_proj_a
//
// Keyed on the module's parent prefix, so two layers — or two experts of one
// MoE layer — never share a scale, and a checkpoint that stores its experts
// per-expert groups each expert's own gate/up pair.
std::string fusion_group_key(const std::string& weight_name);

// -- tensor scales ---------------------------------------------------------

// Largest finite magnitude in a host FP16 buffer.
//
// Compared as bit patterns: FP16 orders magnitudes monotonically once the sign
// is masked off, so this is one pass of integer compares and a single decode,
// and it is exact rather than a float accumulation. Infinities and NaNs are
// skipped — a BF16 source can carry values FP16 cannot hold, and one of them
// would otherwise set the scale for the whole tensor.
float fp16_absmax(const uint16_t* data, size_t n);

// The tensor scale to quantize with, given the largest magnitude in the tensor
// (or across a fused group). absmax / 6 leaves the micro-scales in [0, 1].
//
// MEASURED, and not what the theory predicted. Published exports scale by
// absmax / (6 * 448) instead, which puts the loudest micro-block's scale at FP8
// E4M3's largest normal and lifts every quiet block ~8 binades off the
// subnormal clamp in quantize_micro_block_nvfp4. That looks strictly better and
// is worse: Qwen3-0.6B over ppl_corpus_45k, deterministic_gemm, shared group
// scales in both arms, reads 31.05 against **29.47** for absmax / 6. Repeated
// across processes, bit-identical each time.
//
// The reader does not care either way — both conventions are just numbers vLLM
// and imp multiply back out — so the arm that measures better wins. Do not
// "fix" this to match the published convention without re-measuring.
float export_tensor_scale(float absmax);

// -- config.json -----------------------------------------------------------

// The `quantization_config` object a compressed-tensors NVFP4 checkpoint
// declares. `ignore` lists the modules left at source precision; vLLM builds an
// unquantized layer for each and would otherwise look for scales that were
// never written.
//
// `input_activations` is null on purpose: this tool quantizes weights only, and
// vLLM reads that as NVFP4A16 (compressed_tensors.py: "None for NVFP4A16").
// Declaring activation quantization we did not do would make vLLM quantize
// activations to FP4 at runtime against scales this checkpoint never measured.
std::string compressed_tensors_quant_config(const std::vector<std::string>& ignore, bool calibrated);

// Insert or replace the top-level `quantization_config` in a config.json.
//
// Textual on purpose. Round-tripping the document through a parser would be the
// obvious approach and is the wrong one here: json_util encodes booleans as
// numbers, so `"tie_word_embeddings": false` would come back as `0`, and every
// float in the file would be reformatted. This edits the one key and leaves
// every other byte identical.
//
// Replacing rather than prepending matters: a source config.json may already
// carry `"quantization_config": null` (Qwen3.8-27B does). Two keys of the same
// name is legal JSON that most parsers resolve to the LAST one, so prepending
// would produce a file that reads as unquantized.
[[nodiscard]] std::expected<std::string, std::string> patch_config_json(const std::string& src,
                                                                        const std::string& quant_config_obj);

// A combination that produces a checkpoint the target engine cannot load, or
// nullptr when there is nothing to say. Reported before the conversion runs.
//
// Today there is one: `--lm-head` with compressed-tensors output. vLLM's
// `ParallelLMHead` takes no scales ("There is no module or parameter named
// lm_head.weight_global_scale ... available: {'lm_head.weight'}"), so the
// checkpoint fails to load there — loudly, but only after the conversion. It is
// a defensible choice for an imp-only checkpoint, where it costs nothing on top
// of what the runtime already spends, so this warns rather than refuses.
const char* portability_warning(OutputFormat fmt, bool quantize_lm_head);

// -- the files themselves --------------------------------------------------

// Whether this source can carry the declaration the chosen format needs.
//
// In compressed-tensors mode config.json is the ONLY place the checkpoint says
// it is quantized, and it is patched rather than written from nothing. Without
// one the output is a directory of packed nibbles every reader takes for an
// unquantized model — the failure being that it loads.
//
// Called BEFORE the first tensor is written, not at the copy step where the
// file is actually needed: a source that cannot be declared should cost a
// second, not the 25 minutes of a 27B conversion that then refuses to finish.
[[nodiscard]] std::expected<void, std::string> can_declare_quantization(const std::string& in_dir,
                                                                        OutputFormat fmt);

// Everything the checkpoint needs beside the weights.
//
// The named list is what imp itself reads. It is deliberately NOT the whole
// requirement: a quantized checkpoint should stay loadable by whatever the
// source was loadable by, and an allowlist of imp's own needs quietly breaks
// that. Measured: the output of this tool would not load in vLLM at all,
// because `preprocessor_config.json` was missing, a file imp never reads.
//
// In CompressedTensors mode config.json is not copied but PATCHED with the
// quantization_config the writer just produced, since that is where vLLM reads
// the scheme from. In Modelopt mode it is copied verbatim and the declaration
// goes to hf_quant_config.json, which is where imp's loader looks.
[[nodiscard]] std::expected<void, std::string> copy_aux_files(
    const std::string& in_dir, const std::string& out_dir, OutputFormat fmt,
    const std::vector<std::string>& excluded_modules, bool calibrated);

// hf_quant_config.json — Modelopt's declaration. Not written for
// compressed-tensors, where a second declaration in a DIFFERENT format is a way
// for the two to disagree later.
[[nodiscard]] std::expected<void, std::string> write_modelopt_quant_config(
    const std::string& out_dir, const std::vector<std::string>& excluded, bool calibrated);

// recipe.yaml — llm-compressor's record of the run, written beside the
// config.json block for compressed-tensors output.
//
// Redundant on paper and worth it in practice: readers that predate
// `quantization_config` detection look for this file and nothing else, and one
// of them reading this checkpoint as Modelopt inverts every tensor scale in
// silence. It states the same scheme as the config block, from the same
// arguments, so the two cannot say different things.
[[nodiscard]] std::expected<void, std::string> write_recipe_yaml(const std::string& out_dir,
                                                                 const std::vector<std::string>& excluded);

// A sharded checkpoint is only loadable with an index, and the index cannot be
// copied from the source: quantizing one weight turns it into three tensors, so
// the name->shard map has to be rebuilt from what was actually written.
[[nodiscard]] std::expected<void, std::string> write_shard_index(
    const std::string& out_dir, const std::vector<std::pair<std::string, std::string>>& tensor_to_shard,
    size_t total_bytes);

}  // namespace imp::quantize
