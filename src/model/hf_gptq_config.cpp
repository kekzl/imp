#include "model/hf_config_loader.h"
#include "model/json_util.h"
#include "core/logging.h"

#include <algorithm>
#include <cctype>
#include <string>

namespace imp {

bool HFConfigLoader::load_gptq_config(const std::string& model_dir, GPTQConfig& cfg) {
    // quantize_config.json (AutoGPTQ/GPTQModel): quant_method optional. config.json
    // (HF/optimum): quantization_config.quant_method must be gptq.
    auto try_parse = [&](const std::string& path, bool nested_under_qc) -> bool {
        JValue root;
        if (!parse_json_file(path, root))
            return false;
        const JValue* base = &root;
        if (nested_under_qc) {
            base = jobj_find(root, "quantization_config");
            if (!base || base->type != JType::OBJECT)
                return false;
        }
        std::string method;
        const bool has_method = jobj_get_string(*base, "quant_method", method);
        std::transform(method.begin(), method.end(), method.begin(),
                       [](unsigned char c) { return std::tolower(c); });
        if ((nested_under_qc || has_method) && method != "gptq")
            return false;

        IMP_LOG_INFO("loading GPTQ config from %s", path.c_str());
        jobj_opt_int(*base, "bits", cfg.bits);
        jobj_opt_int(*base, "group_size", cfg.group_size);
        const JValue* da = jobj_find(*base, "desc_act");
        cfg.desc_act = da && da->type == JType::NUMBER && da->num_val != 0.0;
        if (!jobj_get_string(*base, "checkpoint_format", cfg.checkpoint_format))
            jobj_opt_string(*base, "format", cfg.checkpoint_format);
        const JValue* marlin = jobj_find(*base, "is_marlin_format");
        if (marlin && marlin->type == JType::NUMBER && marlin->num_val != 0.0)
            cfg.checkpoint_format = "marlin";
        IMP_LOG_INFO("  GPTQ: bits=%d group_size=%d desc_act=%s checkpoint_format=%s", cfg.bits,
                     cfg.group_size, cfg.desc_act ? "true" : "false",
                     cfg.checkpoint_format.empty() ? "unspecified" : cfg.checkpoint_format.c_str());
        return true;
    };
    return try_parse(model_dir + "/quantize_config.json", /*nested_under_qc=*/false) ||
           try_parse(model_dir + "/config.json", /*nested_under_qc=*/true);
}

}  // namespace imp
