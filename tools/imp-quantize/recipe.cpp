#include "recipe.h"

#include "checkpoint_out.h"
#include "fp8_head.h"
#include "imp/imp.h"
#include "model/hf_fetch.h"
#include "options.h"

#ifndef IMP_TREE_ID
#define IMP_TREE_ID ""
#endif

namespace imp::quantize {

const char* build_tree_id() {
    constexpr const char* id = IMP_TREE_ID;
    return id[0] ? id : "unknown";
}

void fill_recipe(Recipe& r, const Options& opt, bool kv_cache_fp8) {
    r.imp_version = imp_version();
    r.imp_tree = build_tree_id();
    r.format = format_name(opt.format);
    r.lm_head = lm_head_export_name(opt.lm_head);
    r.kv_hint = kv_hint_name(opt.kv_hint);
    r.kv_cache_fp8 = kv_cache_fp8;
    r.keep_attn_gate = opt.keep_attn_gate;
    r.keep_gdn_proj = opt.keep_gdn_proj;
    r.calibrated = !opt.calib_file.empty();
    if (r.calibrated) {
        r.calib_sha256 = hf::sha256_file(opt.calib_file);
        r.calib_weight = opt.calib_weight_sq ? "sq" : "abs";
    }
}

std::string recipe_json(const Recipe& r, const std::string& indent) {
    const std::string in = indent + "  ";
    auto str = [](const std::string& s) { return '"' + json_escape(s) + '"'; };
    auto boolean = [](bool b) { return std::string(b ? "true" : "false"); };
    std::string j = "{\n";
    j += in + "\"recipe_version\": " + std::to_string(kRecipeVersion) + ",\n";
    j += in + "\"imp_version\": " + str(r.imp_version) + ",\n";
    j += in + "\"imp_tree\": " + str(r.imp_tree) + ",\n";
    j += in + "\"format\": " + str(r.format) + ",\n";
    j += in + "\"lm_head\": " + str(r.lm_head) + ",\n";
    j += in + "\"kv_hint\": " + str(r.kv_hint) + ",\n";
    j += in + "\"kv_cache_fp8\": " + boolean(r.kv_cache_fp8) + ",\n";
    j += in + "\"keep_attn_gate\": " + boolean(r.keep_attn_gate) + ",\n";
    j += in + "\"keep_gdn_proj\": " + str(r.keep_gdn_proj) + ",\n";
    j += in + "\"calibration\": ";
    if (!r.calibrated) {
        j += "null\n";
    } else {
        const std::string c = in + "  ";
        j += "{\n";
        j += c + "\"method\": \"awq\",\n";
        j += c + "\"file_sha256\": " + str(r.calib_sha256) + ",\n";
        j += c + "\"model_id\": " + str(r.calib_model_id) + ",\n";
        j += c + "\"samples\": " + std::to_string(r.calib_samples) + ",\n";
        j += c + "\"entries\": " + std::to_string(r.calib_entries) + ",\n";
        j += c + "\"groups\": " + str(r.calib_groups) + ",\n";
        j += c + "\"weight\": " + str(r.calib_weight) + ",\n";
        j += c + "\"n_rep\": " + std::to_string(r.n_rep) + ",\n";
        j += c + "\"hybrid\": " + boolean(r.hybrid) + "\n";
        j += in + "}\n";
    }
    j += indent + "}";
    return j;
}

}  // namespace imp::quantize
