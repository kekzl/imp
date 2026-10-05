#include "model/lfm2_names.h"

#include <array>
#include <string_view>
#include <utility>

namespace imp {

namespace {

// LFM2 module path -> HF-canonical module path; the tensor suffix (.weight, NVFP4 .weight_scale,
// .weight_scale_2, .input_scale) is kept.
constexpr std::array<std::pair<std::string_view, std::string_view>, 12> kModuleRewrites{{
    {"operator_norm", "input_layernorm"},
    {"ffn_norm", "post_attention_layernorm"},
    {"self_attn.out_proj", "self_attn.o_proj"},
    {"self_attn.q_layernorm", "self_attn.q_norm"},
    {"self_attn.k_layernorm", "self_attn.k_norm"},
    {"conv.in_proj", "mamba.in_proj"},
    {"conv.conv", "mamba.conv1d"},
    {"conv.out_proj", "mamba.out_proj"},
    {"feed_forward.w1", "mlp.gate_proj"},
    {"feed_forward.w3", "mlp.up_proj"},
    {"feed_forward.w2", "mlp.down_proj"},
    {"feed_forward.gate", "mlp.gate"},
}};

constexpr std::array<std::pair<std::string_view, std::string_view>, 3> kExpertRewrites{{
    {"w1", "gate_proj"},
    {"w3", "up_proj"},
    {"w2", "down_proj"},
}};

// `sub` = from + "." + suffix -> to + "." + suffix.
bool rewrite_module(std::string_view sub, std::string_view from, std::string_view to, std::string& out) {
    if (sub.size() <= from.size() || sub.compare(0, from.size(), from) != 0 || sub[from.size()] != '.')
        return false;
    out = std::string(to) + std::string(sub.substr(from.size()));
    return true;
}

}  // namespace

std::string lfm2_canonical_name(const std::string& name) {
    if (name == "model.embedding_norm.weight")
        return "model.norm.weight";
    constexpr std::string_view kLayers = "model.layers.";
    if (name.compare(0, kLayers.size(), kLayers) != 0)
        return name;
    const size_t dot = name.find('.', kLayers.size());
    if (dot == std::string::npos)
        return name;
    const std::string head = name.substr(0, dot + 1);  // model.layers.N.
    const std::string_view sub = std::string_view(name).substr(dot + 1);
    if (sub == "feed_forward.expert_bias")
        return head + "mlp.gate.bias";
    std::string out;
    for (const auto& [from, to] : kModuleRewrites)
        if (rewrite_module(sub, from, to, out))
            return head + out;
    constexpr std::string_view kExperts = "feed_forward.experts.";
    if (sub.compare(0, kExperts.size(), kExperts) == 0) {
        const std::string_view rest = sub.substr(kExperts.size());  // E.w1.<suffix>
        const size_t edot = rest.find('.');
        if (edot != std::string_view::npos)
            for (const auto& [from, to] : kExpertRewrites)
                if (rewrite_module(rest.substr(edot + 1), from, to, out))
                    return head + "mlp.experts." + std::string(rest.substr(0, edot + 1)) + out;
    }
    return name;
}

}  // namespace imp
