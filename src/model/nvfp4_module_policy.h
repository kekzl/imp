#pragma once

// A compressed-tensors NVFP4 checkpoint states a COMPLETE partition of its Linears:
// config_groups[*].targets (always ["Linear"]) says what's quantized, quantization_config
// .ignore lists what stayed at source precision. Before #1960 the ignore list was parsed and
// read nowhere, so a Linear that lost its weight_scale (misnamed, wrong dtype, dropped shard)
// was indistinguishable from one the author deliberately left in BF16. Pure and header-only:
// the loader drives it with a real tensor map, the CPU lane with synthetic names.

#include <cstdint>
#include <regex>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace imp::nvfp4_policy {

inline bool ends_with_(std::string_view s, std::string_view suf) {
    return s.size() >= suf.size() && s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

inline bool starts_with_(std::string_view s, std::string_view pre) {
    return s.size() >= pre.size() && s.compare(0, pre.size(), pre) == 0;
}

inline bool contains_(std::string_view s, std::string_view what) {
    return s.find(what) != std::string_view::npos;
}

// Strips the multimodal wrapper prefix, exactly as llm_compressor::translate_name step 3.
// The ignore list is written against the ON-DISK spelling while the tensor map holds the
// translated one, so both sides are normalized here rather than each caller remembering which.
inline std::string normalize_module(std::string_view name) {
    static constexpr std::string_view kGemma4 = "model.language_model.";
    static constexpr std::string_view kMistral3 = "language_model.";
    if (starts_with_(name, kGemma4))
        return "model." + std::string(name.substr(kGemma4.size()));
    if (starts_with_(name, kMistral3))
        return std::string(name.substr(kMistral3.size()));
    return std::string(name);
}

// Drop the tensor-role suffix so a tensor name becomes its module name. Both
// spellings are accepted because the caller may hold either side of the
// `.weight_packed` -> `.weight` rename.
inline std::string module_of_tensor(std::string_view tensor_name) {
    static constexpr std::string_view kSuffixes[] = {
        ".weight_packed", ".weight_global_scale", ".weight_scale_inv", ".weight_scale_2",
        ".weight_scale",  ".input_global_scale",  ".input_scale",      ".weight",
        ".bias"};
    for (std::string_view suf : kSuffixes)
        if (ends_with_(tensor_name, suf))
            return normalize_module(tensor_name.substr(0, tensor_name.size() - suf.size()));
    return normalize_module(tensor_name);
}

// Glob with `*` only (Modelopt writes `*.output_layer`); no character classes,
// no `?`. Iterative, so a pathological pattern cannot blow the stack.
inline bool glob_match(std::string_view pat, std::string_view s) {
    size_t p = 0, i = 0, star = std::string_view::npos, mark = 0;
    while (i < s.size()) {
        if (p < pat.size() && pat[p] == '*') {
            star = p++;
            mark = i;
        } else if (p < pat.size() && pat[p] == s[i]) {
            ++p;
            ++i;
        } else if (star != std::string_view::npos) {
            p = star + 1;
            i = ++mark;
        } else {
            return false;
        }
    }
    while (p < pat.size() && pat[p] == '*')
        ++p;
    return p == pat.size();
}

// Wrapper-prefix strip applied to a REGEX, not a module name: an ignore entry written
// against on-disk names needs it on the re: arm too, or re:model\.language_model\..*
// matches nothing against already-stripped names (an unclassified-slot refusal). Both the
// raw and stripped pattern are tried, so this can only add a match.
inline std::string normalize_pattern(std::string_view p) {
    static constexpr std::string_view kForms[][2] = {{"model\\.language_model\\.", "model\\."},
                                                     {"model.language_model.", "model."},
                                                     {"language_model\\.", ""},
                                                     {"language_model.", ""}};
    for (const auto& f : kForms)
        if (starts_with_(p, f[0]))
            return std::string(f[1]) + std::string(p.substr(f[0].size()));
    return std::string(p);
}

// One ignore entry, parsed once. Compiling a regex per (entry, module) pair
// costs seconds on a 30k-module checkpoint; the attribution pass below walks
// every module for every entry.
struct CompiledIgnoreEntry {
    bool is_regex = false;
    bool regex_ok = false;
    bool has_norm = false;
    std::regex rx, rx_norm;
    std::string literal;  // normalized, for the non-regex forms
    bool has_glob = false;
};

inline CompiledIgnoreEntry compile_ignore_entry(std::string_view raw) {
    CompiledIgnoreEntry e;
    if (starts_with_(raw, "re:")) {
        e.is_regex = true;
        const std::string pat(raw.substr(3));
        const std::string norm = normalize_pattern(pat);
        try {
            e.rx = std::regex(pat);
            e.regex_ok = true;
            if (norm != pat) {
                e.rx_norm = std::regex(norm);
                e.has_norm = true;
            }
        } catch (const std::regex_error&) {
            e.regex_ok = false;  // an unparseable pattern matches nothing, never everything
        }
        return e;
    }
    e.literal = normalize_module(raw);
    e.has_glob = contains_(e.literal, "*");
    return e;
}

// One ignore entry against one module name, the three forms the two producers write:
//   re:<regex>   compressed-tensors/vLLM check_equal_or_regex_match
//   a.b.c        fully qualified module (imp-quantize: tensor name minus .weight)
//   *.suffix     Modelopt glob
// Plus a trailing-segment match (lm_head covers model.lm_head), matching vLLM's fused lookup.
inline bool entry_matches(const CompiledIgnoreEntry& e, const std::string& module) {
    if (e.is_regex) {
        if (!e.regex_ok)
            return false;
        return std::regex_match(module, e.rx) || (e.has_norm && std::regex_match(module, e.rx_norm));
    }
    if (e.literal == module)
        return true;
    if (e.has_glob)
        return glob_match(e.literal, module);
    return module.size() > e.literal.size() && ends_with_(module, e.literal) &&
           module[module.size() - e.literal.size() - 1] == '.';
}

inline bool entry_matches(std::string_view entry_raw, const std::string& module) {
    return entry_matches(compile_ignore_entry(entry_raw), module);
}

inline bool module_is_ignored(const std::string& module, const std::vector<std::string>& ignore) {
    for (const std::string& e : ignore)
        if (entry_matches(e, module))
            return true;
    return false;
}

// Roles that are 2-D, K-aligned, and still must NOT be NVFP4 (imp-quantize refuses to write
// them too, tensor_policy.cpp calls this same function so writer/reader cannot drift).
// Returns true and fills why_not when excluded. Each entry measured, not assumed:
//   embeddings   quantizing costs quality for no decode-bandwidth win
//   vision tower loader reads F16/BF16/F32 only; an NVFP4 tower is unreadable
//   MTP head     loader has no scale companions; quantized draft acceptance went 81%->0/24
//   MLA latent   kv_a_proj_with_mqa/kv_b_proj are sliced/reshaped by the runtime (bisected)
//   MoE router   FP4 across 16 shared scales flips the top-k pick (.gate.weight only)
//   K % 16       the kernel's hard-coded micro-block size
inline bool role_excluded(const std::string& name, int64_t K, bool allow_lm_head, std::string& why_not) {
    // `embeddings` covers Nemotron's `backbone.embeddings`, which is an
    // embedding table under a name the `embed_*` tests miss; it was the one
    // unclassified slot across the local checkpoint set.
    if (contains_(name, "embed_tokens") || contains_(name, "embed_positions") ||
        contains_(name, "embeddings")) {
        why_not = "embedding";
        return true;
    }
    if (contains_(name, "norm")) {
        why_not = "norm";
        return true;
    }
    if (contains_(name, "lm_head") && !allow_lm_head) {
        why_not = "lm_head (use --lm-head to include)";
        return true;
    }
    if (contains_(name, "visual.") || contains_(name, "vision_tower.")) {
        why_not = "vision tower (upload path takes source precision only)";
        return true;
    }
    if (starts_with_(name, "mtp.")) {
        why_not = "MTP draft head (its loader reads the weight by name, without scales)";
        return true;
    }
    if (contains_(name, "kv_a_proj") || contains_(name, "kv_b_proj")) {
        why_not = "MLA latent projection (runtime slices it - must stay full precision)";
        return true;
    }
    if (ends_with_(name, ".gate.weight") || ends_with_(name, ".router.weight")) {
        why_not = "MoE router (FP4 changes expert selection)";
        return true;
    }
    if (K % 16 != 0) {
        why_not = "K=" + std::to_string(K) + " is not a multiple of 16";
        return true;
    }
    return false;
}

// One `.weight` entry of the tensor map, reduced to what the partition needs.
struct SlotObservation {
    std::string name;  // translated tensor name, ending in `.weight`
    int ndim = 0;
    int64_t K = 0;                  // logical K; for a packed weight that is 2 * shape[1]
    bool has_micro_scale = false;   // a sibling `.weight_scale` exists
    bool has_global_scale = false;  // a sibling `.weight_scale_2` exists
};

// Two independent counts kept apart: SLOT counts partition the Linear-role .weight tensors
// the loader holds; IGNORE counts partition the checkpoint's own ignore list (what the
// operator sees). Most ignore entries land on non-Linear tensors (vision, conv1d, embedding,
// MTP), so reporting "N ignored" against the slot count misreads as "the rest were dropped".
struct Inventory {
    int quantized = 0;
    int ignored = 0;
    int unclassified = 0;
    int missing_global_scale = 0;
    // entries == on_linear_slot + outside_linear_set + unmatched
    int ignore_entries = 0;
    int ignore_on_linear_slot = 0;
    int ignore_outside_linear_set = 0;
    int ignore_unmatched = 0;
    std::string first_unclassified;
    std::string first_missing_global_scale;
};

// `lm_head` counts as a Linear role here even though imp-quantize refuses to
// write it by default: the checkpoint lists it in `ignore`, so leaving it out
// of the partition would drop a row the author explicitly declared.
inline bool is_linear_role(const SlotObservation& s) {
    if (s.ndim != 2 || s.K <= 0)
        return false;
    std::string why;
    return !role_excluded(s.name, s.K, /*allow_lm_head=*/true, why);
}

inline Inventory classify(const std::vector<SlotObservation>& slots, const std::vector<std::string>& ignore) {
    Inventory inv;
    // module name -> did it reach the classifier (i.e. is it a Linear role).
    std::vector<std::pair<std::string, bool>> modules;
    modules.reserve(slots.size());
    for (const SlotObservation& s : slots) {
        const std::string module = module_of_tensor(s.name);
        const bool linear = is_linear_role(s);
        modules.emplace_back(module, linear);
        if (!linear)
            continue;
        if (s.has_micro_scale) {
            inv.quantized++;
            if (!s.has_global_scale) {
                inv.missing_global_scale++;
                if (inv.first_missing_global_scale.empty())
                    inv.first_missing_global_scale = module;
            }
            continue;
        }
        if (module_is_ignored(module, ignore)) {
            inv.ignored++;
            continue;
        }
        inv.unclassified++;
        if (inv.first_unclassified.empty())
            inv.first_unclassified = module;
    }

    // Attributes the ignore list itself: an entry matching any Linear-role slot counts as
    // honoured even if that slot turned out packed. Question is what became of the operator's
    // ignore lines, not what became of the tensors.
    inv.ignore_entries = static_cast<int>(ignore.size());
    for (const std::string& raw : ignore) {
        const CompiledIgnoreEntry e = compile_ignore_entry(raw);
        bool on_linear = false, seen = false;
        for (const auto& [module, linear] : modules) {
            if (!entry_matches(e, module))
                continue;
            seen = true;
            if (linear) {
                on_linear = true;
                break;
            }
        }
        if (on_linear)
            inv.ignore_on_linear_slot++;
        else if (seen)
            inv.ignore_outside_linear_set++;
        else
            inv.ignore_unmatched++;
    }
    return inv;
}

// Both refusal rules are scoped to compressed-tensors; that scope is load-bearing. Modelopt's
// exclude_modules is a HINT (may list far less than what's actually unquantized) and its
// weight_scale_2 is genuinely optional, so applying either rule there refuses working
// checkpoints. compressed-tensors states a complete partition with a weight_global_scale per
// module; a gap there is an imp-side read defect, not a property of the export.
inline bool refuses(const Inventory& inv, bool is_compressed_tensors, std::string* why) {
    if (!is_compressed_tensors)
        return false;
    if (inv.unclassified > 0) {
        if (why) {
            *why = "compressed-tensors NVFP4 checkpoint: " + std::to_string(inv.unclassified) +
                   " Linear module(s) are neither packed nor in quantization_config.ignore (first: " +
                   inv.first_unclassified +
                   "). The checkpoint declares targets=[Linear] plus a complete ignore list, so a "
                   "leftover means imp lost the weight_scale on the way in and would serve that "
                   "module as if the author had left it in source precision.";
        }
        return true;
    }
    if (inv.missing_global_scale > 0) {
        if (why) {
            *why = "compressed-tensors NVFP4 checkpoint: " + std::to_string(inv.missing_global_scale) +
                   " packed module(s) carry no weight_global_scale (first: " +
                   inv.first_missing_global_scale +
                   "). The format divides by it, so imp would default the tensor scale to 1.0 and "
                   "serve that Linear off by the checkpoint's amax/6 - silently, at exit code 0.";
        }
        return true;
    }
    return false;
}

}  // namespace imp::nvfp4_policy
