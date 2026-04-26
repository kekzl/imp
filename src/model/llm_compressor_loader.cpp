#include "model/llm_compressor_loader.h"

#include "core/logging.h"

#include <fstream>
#include <sstream>
#include <string_view>
#include <vector>

namespace imp::llm_compressor {

namespace {

bool try_rename_suffix(std::string& name, std::string_view from, std::string_view to) {
    if (name.size() < from.size()) return false;
    if (name.compare(name.size() - from.size(), from.size(), from) != 0) return false;
    name.replace(name.size() - from.size(), from.size(), to);
    return true;
}

bool starts_with(std::string_view s, std::string_view prefix) {
    return s.size() >= prefix.size() && s.compare(0, prefix.size(), prefix) == 0;
}

bool ends_with(std::string_view s, std::string_view suffix) {
    return s.size() >= suffix.size() &&
           s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// Recognized projection names whose `.scale` is NOT a Gemma-4 extra.
// If a tensor name segment immediately before `.scale` matches one of these,
// the tensor passes through (handled later by weight_map).
bool is_proj_segment(std::string_view name_before_dot_scale) {
    // Last segment between dots — find last '.' in the substring.
    auto pos = name_before_dot_scale.rfind('.');
    std::string_view last = (pos == std::string_view::npos)
                                ? name_before_dot_scale
                                : name_before_dot_scale.substr(pos + 1);
    return last == "q_proj" || last == "k_proj" || last == "v_proj" ||
           last == "o_proj" || last == "gate_proj" || last == "up_proj" ||
           last == "down_proj";
}

// Strip leading/trailing whitespace and optionally a matching pair of quotes.
std::string trim_value(std::string_view sv) {
    while (!sv.empty() && (sv.front() == ' ' || sv.front() == '\t')) sv.remove_prefix(1);
    while (!sv.empty() && (sv.back() == ' ' || sv.back() == '\t' ||
                           sv.back() == '\r' || sv.back() == '\n')) sv.remove_suffix(1);
    if (sv.size() >= 2 && (sv.front() == '\'' || sv.front() == '"') &&
        sv.front() == sv.back()) {
        sv.remove_prefix(1);
        sv.remove_suffix(1);
    }
    return std::string(sv);
}

// Parse a bracket-array `[a, b, c]` (single-line). Returns vector of values.
std::vector<std::string> parse_bracket_array(std::string_view body) {
    std::vector<std::string> out;
    auto lb = body.find('[');
    auto rb = body.rfind(']');
    if (lb == std::string_view::npos || rb == std::string_view::npos || rb <= lb) return out;
    std::string_view inner = body.substr(lb + 1, rb - lb - 1);

    size_t start = 0;
    bool in_quote = false;
    char quote_char = 0;
    for (size_t i = 0; i <= inner.size(); i++) {
        bool at_end = (i == inner.size());
        char c = at_end ? ',' : inner[i];
        if (!at_end && in_quote) {
            if (c == quote_char) in_quote = false;
            continue;
        }
        if (!at_end && (c == '\'' || c == '"')) {
            in_quote = true;
            quote_char = c;
            continue;
        }
        if (c == ',') {
            std::string item = trim_value(inner.substr(start, i - start));
            if (!item.empty()) out.push_back(std::move(item));
            start = i + 1;
        }
    }
    return out;
}

} // namespace

NameTranslation translate_name(const std::string& in, TranslationCounters& counters) {
    std::string out = in;

    // Step 1: skip patterns (vision tower) — check raw input before any mutation.
    if (starts_with(out, "model.vision_tower.") || starts_with(out, "model.visual.")) {
        counters.vision_skipped++;
        return {NameTranslation::SKIP, ""};
    }

    // Step 2: skip Gemma-4 extras.
    if (ends_with(out, ".layer_scalar") || ends_with(out, ".per_expert_scale")) {
        counters.gemma4_extra_skipped++;
        return {NameTranslation::SKIP, ""};
    }
    if (ends_with(out, ".scale")) {
        // Only skip if the segment immediately before .scale is NOT a known proj.
        std::string_view before_scale(out.data(), out.size() - 6); // strip ".scale"
        if (!is_proj_segment(before_scale)) {
            counters.gemma4_extra_skipped++;
            return {NameTranslation::SKIP, ""};
        }
        // else fall through (pass-through handles it).
    }

    // Step 3: prefix strip (multimodal language_model wrapper).
    static constexpr const char kMultimodalPrefix[] = "model.language_model.";
    static constexpr size_t kMultimodalPrefixLen = sizeof(kMultimodalPrefix) - 1;
    if (starts_with(out, kMultimodalPrefix)) {
        out = "model." + out.substr(kMultimodalPrefixLen);
        counters.prefix_strips++;
        // Continue to suffix-rename step below.
    }

    // Step 4: suffix renames (mutually exclusive).
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

    // Step 5: pass through (still increments prefix_strips counter from above
    // if we did strip; suffix_renames stays 0 in that case).
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

bool parse_recipe_yaml(const std::string& model_dir,
                       imp::HFConfigLoader::NvFP4Config& cfg) {
    std::ifstream in(model_dir + "/recipe.yaml");
    if (!in.good()) return false;

    std::string scheme;
    std::vector<std::string> ignore_list;
    bool seen_quant_mod = false;

    std::string line;
    while (std::getline(in, line)) {
        std::string_view sv(line);
        while (!sv.empty() && (sv.front() == ' ' || sv.front() == '\t')) sv.remove_prefix(1);

        if (sv.find("QuantizationModifier:") == 0) { seen_quant_mod = true; continue; }
        if (!seen_quant_mod) continue;

        if (sv.find("scheme:") == 0) {
            scheme = trim_value(sv.substr(7));
        } else if (sv.find("ignore:") == 0) {
            ignore_list = parse_bracket_array(sv.substr(7));
        } else if (sv.find("group_size:") == 0) {
            try { cfg.group_size = std::stoi(trim_value(sv.substr(11))); }
            catch (...) { /* keep default */ }
        }
    }

    if (!seen_quant_mod) {
        IMP_LOG_ERROR("recipe.yaml has no QuantizationModifier block");
        return false;
    }
    if (scheme != "NVFP4" && scheme != "NVFP4_W4A16") {
        IMP_LOG_ERROR("recipe.yaml scheme '%s' not supported (need NVFP4 or NVFP4_W4A16)",
                      scheme.c_str());
        return false;
    }

    cfg.exclude_modules = std::move(ignore_list);
    cfg.format = imp::HFConfigLoader::NvFP4Format::LLM_COMPRESSOR;
    IMP_LOG_INFO("NVFP4 model (llm-compressor): scheme=%s, group_size=%d, exclude=%zu modules",
                 scheme.c_str(), cfg.group_size, cfg.exclude_modules.size());
    return true;
}

} // namespace imp::llm_compressor
