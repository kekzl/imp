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

// Apply rename + prefix-strip + skip rules deterministically. Increments
// the matching counter. Pure apart from counter mutation.
//
// `keep_vision` = the checkpoint declares a vision tower this build supports
// (HFConfigLoader::probe_vision_tower). Vision tensors are then EMITted under
// their original name — no prefix strip, no suffix rename — because the vision
// mapper dispatches on the literal `model.visual.<slot>` spelling, exactly like
// the MTP head does. Default false keeps the historical behaviour, so a
// text-only llm-compressor checkpoint translates identically to before.
NameTranslation translate_name(const std::string& in, TranslationCounters& counters,
                               bool keep_vision = false);

// True if the given tensor name would be SKIP'd by translate_name. Exposed so
// the SafeTensors loader can skip an entire shard whose contents are unused
// (e.g. model_visual.safetensors when no mmproj is configured). Must be called
// with the same `keep_vision` as translate_name — otherwise the shard holding
// the tower is dropped before translate_name ever sees a tensor from it.
bool name_is_skipped(const std::string& in, bool keep_vision = false);

// True if the name belongs to a vision tower / multimodal projector. Split out
// so the skip rule and the keep rule read from one list.
bool name_is_vision(const std::string& in);

// Emit one INFO log summarizing what translate_name() did across a shard.
// Call once at the end of the enumerate-tensors loop in load_shard().
void log_summary(const TranslationCounters& counters);

// Parse a recipe.yaml file and populate cfg. Returns true if the file is
// a NVFP4 recipe in the expected QuantizationModifier shape; false on
// missing file, parse error, or unsupported scheme.
bool parse_recipe_yaml(const std::string& model_dir, imp::HFConfigLoader::NvFP4Config& cfg);

}  // namespace imp::llm_compressor
