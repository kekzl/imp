#include "model/llm_compressor_loader.h"

#include "core/logging.h"

#include <fstream>
#include <sstream>
#include <string_view>
#include <vector>

namespace imp::llm_compressor {

namespace {

bool try_rename_suffix(std::string& name, std::string_view from, std::string_view to) {
    if (name.size() < from.size())
        return false;
    if (name.compare(name.size() - from.size(), from.size(), from) != 0)
        return false;
    name.replace(name.size() - from.size(), from.size(), to);
    return true;
}

bool starts_with(std::string_view s, std::string_view prefix) {
    return s.size() >= prefix.size() && s.compare(0, prefix.size(), prefix) == 0;
}

bool ends_with(std::string_view s, std::string_view suffix) {
    return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// Recognized projection names whose `.scale` is NOT a Gemma-4 extra.
// If a tensor name segment immediately before `.scale` matches one of these,
// the tensor passes through (handled later by weight_map).
bool is_proj_segment(std::string_view name_before_dot_scale) {
    // Last segment between dots — find last '.' in the substring.
    auto pos = name_before_dot_scale.rfind('.');
    std::string_view last = (pos == std::string_view::npos) ? name_before_dot_scale
                                                            : name_before_dot_scale.substr(pos + 1);
    return last == "q_proj" || last == "k_proj" || last == "v_proj" || last == "o_proj" ||
           last == "gate_proj" || last == "up_proj" || last == "down_proj";
}

// Strip leading/trailing whitespace and optionally a matching pair of quotes.
std::string trim_value(std::string_view sv) {
    while (!sv.empty() && (sv.front() == ' ' || sv.front() == '\t'))
        sv.remove_prefix(1);
    while (!sv.empty() && (sv.back() == ' ' || sv.back() == '\t' || sv.back() == '\r' || sv.back() == '\n'))
        sv.remove_suffix(1);
    if (sv.size() >= 2 && (sv.front() == '\'' || sv.front() == '"') && sv.front() == sv.back()) {
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
    if (lb == std::string_view::npos || rb == std::string_view::npos || rb <= lb)
        return out;
    std::string_view inner = body.substr(lb + 1, rb - lb - 1);

    size_t start = 0;
    bool in_quote = false;
    char quote_char = 0;
    for (size_t i = 0; i <= inner.size(); i++) {
        bool at_end = (i == inner.size());
        char c = at_end ? ',' : inner[i];
        if (!at_end && in_quote) {
            if (c == quote_char)
                in_quote = false;
            continue;
        }
        if (!at_end && (c == '\'' || c == '"')) {
            in_quote = true;
            quote_char = c;
            continue;
        }
        if (c == ',') {
            std::string item = trim_value(inner.substr(start, i - start));
            if (!item.empty())
                out.push_back(std::move(item));
            start = i + 1;
        }
    }
    return out;
}

}  // namespace

bool name_is_vision(const std::string& in) {
    return starts_with(in, "model.vision_tower.") || starts_with(in, "model.visual.") ||
           starts_with(in, "vision_tower.") || starts_with(in, "multi_modal_projector.");
}

bool name_is_skipped(const std::string& in, bool keep_vision) {
    if (name_is_vision(in))
        return !keep_vision;
    return starts_with(in, "mtp.") || starts_with(in, "model.mtp.");
}

NameTranslation translate_name(const std::string& in, TranslationCounters& counters, bool keep_vision) {
    std::string out = in;

    // Step 0: vision tensors the caller asked to keep leave untouched. They are
    // returned BEFORE the rename steps on purpose — the vision mapper matches
    // the literal checkpoint spelling, so a prefix strip or a `.weight_packed`
    // rename here would silently unmap a slot rather than fail.
    if (keep_vision && name_is_vision(out)) {
        counters.vision_kept++;
        return {NameTranslation::EMIT, std::move(out)};
    }

    // Step 1: skip patterns (vision tower / multimodal projector / MTP head).
    if (name_is_skipped(out, keep_vision)) {
        counters.vision_skipped++;
        return {NameTranslation::SKIP, ""};
    }

    // Step 2: count Gemma-4 extras (still emitted — weight_map routes them to
    // layer.{ffn_gate_inp_scale, expert_down_scale, layer_out_scale}, which the
    // forward path applies to make the model coherent). Phase 1 skipped these
    // because the runtime wiring wasn't validated yet; Phase 2 enables them.
    if (ends_with(out, ".layer_scalar") || ends_with(out, ".per_expert_scale")) {
        counters.gemma4_extras++;
    } else if (ends_with(out, ".scale")) {
        std::string_view before_scale(out.data(), out.size() - 6);  // strip ".scale"
        if (!is_proj_segment(before_scale)) {
            counters.gemma4_extras++;
        }
    }

    // Step 3: prefix strip. Two multimodal wrappers seen in the wild:
    //   (a) Gemma-4-style: `model.language_model.<rest>` → `model.<rest>`
    //   (b) Mistral3-style: `language_model.<rest>` → `<rest>` (so e.g.
    //       `language_model.model.layers.0.q.weight_packed` becomes
    //       `model.layers.0.q.weight_packed`, and `language_model.lm_head.weight`
    //       becomes `lm_head.weight`).
    // (a) wins when both could match because it is a strict superset prefix.
    static constexpr const char kGemma4Prefix[] = "model.language_model.";
    static constexpr size_t kGemma4PrefixLen = sizeof(kGemma4Prefix) - 1;
    static constexpr const char kMistral3Prefix[] = "language_model.";
    static constexpr size_t kMistral3PrefixLen = sizeof(kMistral3Prefix) - 1;
    if (starts_with(out, kGemma4Prefix)) {
        out = "model." + out.substr(kGemma4PrefixLen);
        counters.prefix_strips++;
    } else if (starts_with(out, kMistral3Prefix)) {
        out = out.substr(kMistral3PrefixLen);
        counters.prefix_strips++;
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
    IMP_LOG_INFO(
        "llm-compressor format: %d suffix renames, %d prefix strips, "
        "%d vision tensors skipped, %d vision tensors kept, %d Gemma-4 extras emitted, "
        "%d pass-through",
        c.suffix_renames, c.prefix_strips, c.vision_skipped, c.vision_kept, c.gemma4_extras,
        c.passed_through);
}

// Count leading spaces (tabs counted as 4) to track YAML block scope.
static int count_indent(const std::string& line) {
    int n = 0;
    for (char c : line) {
        if (c == ' ')
            n++;
        else if (c == '\t')
            n += 4;
        else
            break;
    }
    return n;
}

bool parse_recipe_yaml(const std::string& model_dir, imp::HFConfigLoader::NvFP4Config& cfg) {
    std::ifstream in(model_dir + "/recipe.yaml");
    if (!in.good())
        return false;

    std::string scheme;
    std::vector<std::string> ignore_list;
    bool seen_quant_mod = false;

    // Indent-aware tracking for the elaborate `config_groups: group_0: weights: {...}`
    // schema (Mistral3 / SmoothQuant pipelines). When weights_indent >= 0 we are
    // inside the `weights:` block of `config_groups.group_0` and capture the
    // numeric quantization signature.
    int config_groups_indent = -1;
    int weights_indent = -1;
    int weights_num_bits = -1;
    std::string weights_type;
    int weights_group_size = -1;

    // Multi-line bracket-array accumulator for `ignore: [..., ..., \n  ...]`.
    std::string ignore_buf;
    bool collecting_ignore = false;

    std::string line;
    while (std::getline(in, line)) {
        if (collecting_ignore) {
            ignore_buf.push_back(' ');
            ignore_buf.append(line);
            if (ignore_buf.find(']') != std::string::npos) {
                ignore_list = parse_bracket_array(ignore_buf);
                collecting_ignore = false;
                ignore_buf.clear();
            }
            continue;
        }

        int indent = count_indent(line);
        std::string_view sv(line);
        while (!sv.empty() && (sv.front() == ' ' || sv.front() == '\t'))
            sv.remove_prefix(1);

        // Indent-based exits — only fire on non-empty content lines so blank
        // lines don't bogusly close a block.
        if (!sv.empty()) {
            if (weights_indent >= 0 && indent <= weights_indent) {
                weights_indent = -1;
            }
            if (config_groups_indent >= 0 && indent <= config_groups_indent) {
                config_groups_indent = -1;
                weights_indent = -1;
            }
        }

        if (sv.find("QuantizationModifier:") == 0) {
            seen_quant_mod = true;
            // A new modifier block resets any prior config_groups state.
            config_groups_indent = -1;
            weights_indent = -1;
            continue;
        }
        if (!seen_quant_mod)
            continue;

        if (sv.find("config_groups:") == 0) {
            config_groups_indent = indent;
            continue;
        }
        if (config_groups_indent >= 0 && sv.find("weights:") == 0) {
            weights_indent = indent;
            continue;
        }

        if (weights_indent >= 0) {
            if (sv.find("num_bits:") == 0) {
                try {
                    weights_num_bits = std::stoi(trim_value(sv.substr(9)));
                } catch (...) {  // best-effort parse — keep default num_bits on malformed value
                }
            } else if (sv.find("type:") == 0) {
                weights_type = trim_value(sv.substr(5));
            } else if (sv.find("group_size:") == 0) {
                try {
                    weights_group_size = std::stoi(trim_value(sv.substr(11)));
                } catch (...) {  // best-effort parse — keep default group_size on malformed value
                }
            }
            continue;
        }

        if (sv.find("scheme:") == 0) {
            scheme = trim_value(sv.substr(7));
        } else if (sv.find("ignore:") == 0) {
            std::string body(sv.substr(7));
            if (body.find('[') != std::string::npos && body.find(']') == std::string::npos) {
                ignore_buf = std::move(body);
                collecting_ignore = true;
            } else {
                ignore_list = parse_bracket_array(body);
            }
        } else if (sv.find("group_size:") == 0) {
            try {
                cfg.group_size = std::stoi(trim_value(sv.substr(11)));
            } catch (...) {  // best-effort parse — keep default group_size on malformed value
            }
        }
    }

    if (!seen_quant_mod) {
        IMP_LOG_ERROR("recipe.yaml has no QuantizationModifier block");
        return false;
    }

    // If no simple `scheme:` line appeared, infer NVFP4 from the elaborate
    // `config_groups.group_0.weights.{num_bits: 4, type: float}` shape.
    if (scheme.empty() && weights_num_bits == 4 && weights_type == "float") {
        scheme = "NVFP4";
        if (weights_group_size > 0)
            cfg.group_size = weights_group_size;
    }

    if (scheme != "NVFP4" && scheme != "NVFP4_W4A16") {
        // Soft fail: returning false signals "no NVFP4 metadata to apply".
        // The SafeTensors loader treats that as "load with whatever the wire
        // dtype gives us" — typically FP16/BF16/FP8. That keeps inference
        // alive for unsupported llm-compressor schemes (W8A8-INT8, FP8, …)
        // instead of hard-blocking the load.
        IMP_LOG_WARN(
            "recipe.yaml scheme '%s' is not natively supported by imp "
            "(only NVFP4 / NVFP4_W4A16). Falling back to the on-wire dtype; "
            "weights will load but quantization metadata is ignored.",
            scheme.c_str());
        return false;
    }

    cfg.exclude_modules = std::move(ignore_list);
    cfg.format = imp::HFConfigLoader::NvFP4Format::LLM_COMPRESSOR;
    IMP_LOG_INFO("NVFP4 model (llm-compressor): scheme=%s, group_size=%d, exclude=%zu modules",
                 scheme.c_str(), cfg.group_size, cfg.exclude_modules.size());
    return true;
}

}  // namespace imp::llm_compressor
