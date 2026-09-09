#pragma once

// The checkpoint's quantization contract, as a pure function.
//
// A compressed-tensors NVFP4 checkpoint states two things about every module:
// `config_groups[*].targets` says which module class is quantized (always
// `["Linear"]` in the exports imp reads) and `quantization_config.ignore` lists
// the modules that stayed at source precision. Together those two are a
// COMPLETE partition of the checkpoint's Linears: a Linear is either packed or
// listed, never neither. Qwen3.8-27B-NVFP4-vllm: 496 packed modules, 170 ignore
// entries, 121 plain 2-D weights of which 121 are in `ignore`, 0 left over.
//
// Until #1960 imp parsed `ignore` into NvFP4Config::exclude_modules and read it
// nowhere. A Linear that lost its `weight_scale` on the way in (misnamed,
// wrong dtype, dropped shard) was therefore indistinguishable from a Linear the
// author deliberately left in BF16: both arrive as a plain `.weight` and both
// serve. The partition is what makes the difference checkable, so this header
// reconstructs it and the loader refuses a checkpoint that does not partition.
//
// Pure and header-only on purpose: the loader drives it with a real tensor map
// and the CPU lane drives it with synthetic names.

#include <cstdint>
#include <regex>
#include <string>
#include <string_view>
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

// Strip the multimodal wrapper prefix, exactly as llm_compressor::translate_name
// step 3 does. The ignore list is written against the ON-DISK spelling
// (`model.language_model.layers.0.linear_attn.conv1d`) while the tensor map
// holds the translated one (`model.layers.0...`), so one side has to move; both
// are normalized here so neither caller has to remember which it holds.
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

// One ignore entry against one module name. The three forms are the ones the
// two producers actually write:
//   - `re:<regex>`   compressed-tensors / vLLM `check_equal_or_regex_match`
//   - `a.b.c`        a fully qualified module (what imp-quantize writes:
//                    tools/imp-quantize/checkpoint_out.cpp feeds it the tensor
//                    name minus `.weight`)
//   - `*.suffix`     Modelopt glob
// Plus a trailing-segment match (`lm_head` covers `model.lm_head`), which is
// how vLLM's fused-module lookup resolves a short entry.
inline bool entry_matches(std::string_view entry_raw, const std::string& module) {
    if (starts_with_(entry_raw, "re:")) {
        try {
            return std::regex_match(module, std::regex(std::string(entry_raw.substr(3))));
        } catch (const std::regex_error&) {
            return false;  // an unparseable pattern matches nothing, never everything
        }
    }
    const std::string entry = normalize_module(entry_raw);
    if (entry == module)
        return true;
    if (contains_(entry, "*"))
        return glob_match(entry, module);
    return module.size() > entry.size() && ends_with_(module, entry) &&
           module[module.size() - entry.size() - 1] == '.';
}

inline bool module_is_ignored(const std::string& module, const std::vector<std::string>& ignore) {
    for (const std::string& e : ignore)
        if (entry_matches(e, module))
            return true;
    return false;
}

// The roles that are 2-D, K-aligned, and still must NOT be NVFP4. This is the
// list `imp-quantize` refuses to write (tools/imp-quantize/tensor_policy.cpp
// calls this function, so the writer and the reader cannot drift), because the
// question is the same one in both directions: is this module a Linear the
// NVFP4 GEMM path serves? Returns true and fills `why_not` when it is not.
//
// Each entry was measured, not assumed:
//   embeddings   quantizing them costs quality for no bandwidth win on decode.
//   vision tower qwen3vl_vision_upload.cpp takes F16/BF16/F32 only, so an NVFP4
//                tower is a tower no loader can read back; every shape check
//                waves it through.
//   MTP head     its loader reads `mtp.*.weight` by name and knows nothing about
//                the scale companions. Quantized on Qwen3.8-27B, draft
//                acceptance went 81% -> 0 of 24: costs speed, never
//                correctness, so there is no louder symptom.
//   MLA latent   `kv_a_proj_with_mqa` / `kv_b_proj` are sliced and reshaped by
//                the runtime. Found by bisection on DeepSeek-V2-Lite: leaving
//                only these two full precision made the checkpoint coherent.
//   MoE router   FP4 across 16 shared-scale values changes the top-k pick. With
//                the MLA pair already excluded, the router alone still produced
//                garbage. Modelopt and llm-compressor keep routers full
//                precision for the same reason. `.gate.weight` is the router;
//                expert `gate_proj.weight` is unaffected by that suffix test.
//   K % 16       the micro-block size the kernel hard-codes.
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

struct Inventory {
    int quantized = 0;
    int ignored = 0;
    int unclassified = 0;
    int missing_global_scale = 0;
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
    for (const SlotObservation& s : slots) {
        if (!is_linear_role(s))
            continue;
        const std::string module = module_of_tensor(s.name);
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
    return inv;
}

// Whether the load must be refused, and why.
//
// Both rules are scoped to compressed-tensors, and that scope is the load-
// bearing part. Modelopt's `exclude_modules` is a HINT (`["lm_head"]` on
// exports that leave far more than lm_head unquantized) and its
// `weight_scale_2` is genuinely optional, so applying either rule there would
// refuse working checkpoints. compressed-tensors states a complete partition
// and gives every quantized module its own `weight_global_scale`; a gap is a
// defect on the imp side of the read, not a property of the export.
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
