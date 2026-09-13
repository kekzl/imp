#pragma once

#include "model/hf_config_loader.h"

#include <string>
#include <unordered_map>

namespace imp::llm_compressor {

// Counts of translation actions taken across one shard load. Used to emit
// a single summary log line at end of load instead of one log per tensor.
struct TranslationCounters {
    int suffix_renames = 0;  // .weight_packed → .weight, etc.
    int prefix_strips = 0;   // model.language_model. → model.
    int vision_skipped = 0;  // model.vision_tower.* / model.visual.* — dropped
    int vision_kept = 0;     // ...the same names, emitted verbatim (keep_vision)
    int gemma4_extras = 0;  // .layer_scalar / .per_expert_scale / router.scale (passed through to weight_map)
    int passed_through = 0;  // unknown patterns, returned unchanged
};

// Result of translating one tensor name. SKIP means do not emit this tensor.
struct NameTranslation {
    enum Action { EMIT, SKIP };
    Action action;
    std::string out_name;  // populated when action == EMIT
};

// Applies rename + prefix-strip + skip rules deterministically; increments the matching
// counter. keep_vision (probe_vision_tower) keeps vision tensors under their literal name,
// no strip/rename, matching the MTP head; default false is unchanged text-only behavior.
NameTranslation translate_name(const std::string& in, TranslationCounters& counters,
                               bool keep_vision = false);

// True if translate_name would SKIP this name. Lets the SafeTensors loader drop a whole
// shard whose contents are unused (e.g. model_visual.safetensors with no mmproj). Must
// pass the same keep_vision as translate_name or the tower's shard is dropped unseen.
bool name_is_skipped(const std::string& in, bool keep_vision = false);

// True if the name belongs to a vision tower / multimodal projector. Split out
// so the skip rule and the keep rule read from one list.
bool name_is_vision(const std::string& in);

// True if the name belongs to an embedded MTP draft head.
bool name_is_mtp(const std::string& in);

// True only when NOTHING in this load will read the tensor, distinct from name_is_skipped():
// translate_name() SKIPs mtp.* (not for the main tensor map) but load_shard() still diverts
// them into the MTP map, so "skipped" != "unused". Conflating the two lost the MTP head on
// sharded checkpoints. keep_mtp mirrors the caller's load_mtp_head, keep_vision the tower.
bool name_is_unused(const std::string& in, bool keep_vision, bool keep_mtp);

// Emit one INFO log summarizing what translate_name() did across a shard.
// Call once at the end of the enumerate-tensors loop in load_shard().
void log_summary(const TranslationCounters& counters);

// Parse a recipe.yaml file and populate cfg. Returns true if the file is
// a NVFP4 recipe in the expected QuantizationModifier shape; false on
// missing file, parse error, or unsupported scheme.
bool parse_recipe_yaml(const std::string& model_dir, imp::HFConfigLoader::NvFP4Config& cfg);

// quantization_config.ignore from config.json only, nothing else. recipe.yaml records the
// RUN's patterns, config.json the expanded module names (re:.*router matches ...router,
// not ...router.proj); the two are not interchangeable. False if file/block/list is absent.
bool read_config_ignore_list(const std::string& model_dir, std::vector<std::string>& out);

// recipe.yaml is llm-compressor's RUN record; config.json's quantization_config is the
// checkpoint's own declaration (read by HF/vLLM, guaranteed on any re-upload). Reading a
// compressed-tensors checkpoint as Modelopt scales weights by the reciprocal, wrong by
// amax^2/36. False when the scheme isn't served as NVFP4 (int4 pack-quantized, W8A8, ...).
bool parse_compressed_tensors_config(const std::string& model_dir, imp::HFConfigLoader::NvFP4Config& cfg);

}  // namespace imp::llm_compressor
