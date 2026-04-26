#include "model/llm_compressor_loader.h"

#include "core/logging.h"

#include <fstream>
#include <sstream>
#include <string_view>

namespace imp::llm_compressor {

namespace {

// Return true and update `name` in place if it ends with `from`. Replaces
// the suffix with `to`.
bool try_rename_suffix(std::string& name, std::string_view from, std::string_view to) {
    if (name.size() < from.size()) return false;
    if (name.compare(name.size() - from.size(), from.size(), from) != 0) return false;
    name.replace(name.size() - from.size(), from.size(), to);
    return true;
}

} // namespace

NameTranslation translate_name(const std::string& in, TranslationCounters& counters) {
    std::string out = in;

    // Step 1: suffix renames (mutually exclusive — try in order, stop at first match)
    if (try_rename_suffix(out, ".weight_packed", ".weight")) {
        counters.suffix_renames++;
        return {NameTranslation::EMIT, std::move(out)};
    }
    if (try_rename_suffix(out, ".weight_global_scale", ".weight_scale_2")) {
        counters.suffix_renames++;
        return {NameTranslation::EMIT, std::move(out)};
    }
    if (try_rename_suffix(out, ".input_global_scale", ".input_scale")) {
        counters.suffix_renames++;
        return {NameTranslation::EMIT, std::move(out)};
    }

    // No rule matched — pass through unchanged
    counters.passed_through++;
    return {NameTranslation::EMIT, std::move(out)};
}

void log_summary(const TranslationCounters& c) {
    IMP_LOG_INFO("llm-compressor format: %d suffix renames, %d prefix strips, "
                 "%d vision tensors skipped, %d Gemma-4 extras skipped, "
                 "%d pass-through",
                 c.suffix_renames, c.prefix_strips,
                 c.vision_skipped, c.gemma4_extra_skipped,
                 c.passed_through);
}

bool parse_recipe_yaml(const std::string& /*model_dir*/,
                       imp::HFConfigLoader::NvFP4Config& /*cfg*/) {
    // Implemented in a later task.
    return false;
}

} // namespace imp::llm_compressor
