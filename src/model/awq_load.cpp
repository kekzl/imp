#include "model/awq_load.h"
#include "model/hf_config_loader.h"
#include "quant/dequant_awq.h"
#include "core/logging.h"

#include <algorithm>
#include <cctype>
#include <string>

namespace imp {

// AWQ variants with a dequant (#2205): 4-bit GEMM layout with zero points. An empty version is
// AutoAWQ's default (GEMM). Every other variant keeps the #2196 refusal.
static bool awq_variant_supported(const HFConfigLoader::AWQConfig& c) {
    std::string v = c.version;
    std::transform(v.begin(), v.end(), v.begin(), [](unsigned char ch) { return std::tolower(ch); });
    return c.bits == 4 && c.zero_point && (v.empty() || v == "gemm");
}

static bool awq_projection_ok(const TransformerLayer::GPTQWeight& gw, int group_size, std::string* why) {
    const bool dtypes_ok = gw.qweight.qtype == QType::INT32 && gw.qweight.ndim == 2 && gw.qzeros.data &&
                           gw.qzeros.qtype == QType::INT32 && gw.qzeros.ndim == 2 && gw.scales.data &&
                           gw.scales.qtype == QType::F16 && gw.scales.ndim == 2;
    if (!dtypes_ok) {
        *why = "needs qweight INT32 2-D, qzeros INT32 2-D, scales F16 2-D";
        return false;
    }
    int K = 0, N = 0;
    return awq::check_shapes(gw.qweight.shape, gw.qzeros.shape, gw.scales.shape, group_size, &K, &N, why);
}

// Marks every AWQ projection for dequant_awq4 after checking its tensors. Refuses when a
// .qweight in the checkpoint has no projection slot (it would be dropped silently).
static bool apply_awq_layers(Model& model, const std::unordered_map<std::string, Tensor>& tensor_map,
                             int group_size, std::string* why) {
    size_t n_proj = 0;
    for (size_t li = 0; li < model.layers_.size(); ++li) {
        auto& L = model.layers_[li];
        for (auto* gw : {&L.gptq_q, &L.gptq_k, &L.gptq_v, &L.gptq_o, &L.gptq_gate, &L.gptq_up, &L.gptq_down}) {
            if (!gw->qweight.data)
                continue;
            if (!awq_projection_ok(*gw, group_size, why)) {
                *why = "layer " + std::to_string(li) + ": " + *why;
                return false;
            }
            gw->bits = 4;
            gw->group_size = group_size;
            gw->awq_gemm = true;
            gw->zero_offset = 0;  // AWQ stores z as-is: w = (q - z) * s, passes the upload gate
            ++n_proj;
        }
    }
    size_t n_qweight = 0;
    for (const auto& kv : tensor_map)
        if (kv.first.size() > 8 && kv.first.compare(kv.first.size() - 8, 8, ".qweight") == 0)
            ++n_qweight;
    if (n_proj == 0 || n_proj != n_qweight) {
        *why = std::to_string(n_qweight) + " .qweight tensors, " + std::to_string(n_proj) +
               " on a q/k/v/o/gate/up/down projection slot";
        return false;
    }
    IMP_LOG_INFO("AWQ GEMM 4-bit: %zu projections, group_size=%d, dequantized to FP16 at upload", n_proj,
                 group_size);
    return true;
}

// AWQ: supported variant -> projections marked for dequant; anything else refused (#2196).
bool awq_refuses(Model& model, ModelConfig& cfg, const std::unordered_map<std::string, Tensor>& tensor_map,
                        const std::string& model_dir) {
    HFConfigLoader::AWQConfig awq_cfg;
    if (!HFConfigLoader::load_awq_config(model_dir, awq_cfg))
        return false;
    cfg.is_awq_prequant = true;
    cfg.awq_group_size = awq_cfg.group_size;
    const std::string detected = "bits=" + std::to_string(awq_cfg.bits) + " group_size=" +
                                 std::to_string(awq_cfg.group_size) + " zero_point=" +
                                 (awq_cfg.zero_point ? "true" : "false") + " version=" +
                                 (awq_cfg.version.empty() ? "unspecified" : awq_cfg.version);
    if (!awq_variant_supported(awq_cfg)) {
        IMP_LOG_ERROR(
            "AWQ SafeTensors detected (%s): AWQ variant not supported. Only bits=4 zero_point=true "
            "version=gemm dequantizes (#2205); loading this one as wire dtype would give wrong output. "
            "Use a GPTQ or NVFP4 export instead.",
            detected.c_str());
        return true;
    }
    std::string why;
    if (!apply_awq_layers(model, tensor_map, awq_cfg.group_size, &why)) {
        IMP_LOG_ERROR("AWQ SafeTensors (%s) refused: %s", detected.c_str(), why.c_str());
        return true;
    }
    return false;
}

}  // namespace imp
