#include "core/logging.h"
#include "model/model_limits.h"
#include "model/weight_map_arch.h"

#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>

namespace imp {

namespace {

// Separate q/k/v projections and short convs of one KDA layer, fused below into the GDN slots.
struct KdaPieces {
    const Tensor* proj[3] = {nullptr, nullptr, nullptr};  // q, k, v  [H*128, d_model]
    const Tensor* conv[3] = {nullptr, nullptr, nullptr};  // q, k, v  [H*128, 1, kernel]
};

bool dense(const Tensor& t) {
    return (t.qtype == QType::F32 || t.qtype == QType::F16 || t.qtype == QType::BF16) && !t.scales &&
           t.data && t.ndim >= 2 && t.shape[0] > 0;
}

// Row-concatenation of three dense tensors with equal trailing dims and dtype, bytes owned by `owned`.
Tensor concat_rows(const Tensor* const parts[3], std::vector<void*>& owned, const std::string& what) {
    size_t total = 0;
    for (int i = 0; i < 3; ++i) {
        const Tensor& p = *parts[i];
        const bool same = p.qtype == parts[0]->qtype && p.ndim == parts[0]->ndim &&
                          p.nbytes() / static_cast<size_t>(p.shape[0]) ==
                              parts[0]->nbytes() / static_cast<size_t>(parts[0]->shape[0]);
        if (!dense(p) || !same)
            throw std::runtime_error("bailing_hybrid: " + what +
                                     " pieces are not dense tensors of one shape");
        total += p.nbytes();
    }
    void* buf = std::malloc(total);
    if (!buf)
        throw std::runtime_error("bailing_hybrid: out of host memory fusing " + what);
    owned.push_back(buf);
    Tensor out = *parts[0];
    out.data = buf;
    out.shape[0] = 0;
    size_t off = 0;
    for (int i = 0; i < 3; ++i) {
        std::memcpy(static_cast<char*>(buf) + off, parts[i]->data, parts[i]->nbytes());
        off += parts[i]->nbytes();
        out.shape[0] += parts[i]->shape[0];
    }
    out.compute_strides();
    return out;
}

// model.layers.N.attention.<sub> -> the GDN (linear_attn) or MLA (self_attn) spelling; nullptr = keep.
const char* attention_target(std::string_view sub, const Tensor& t, int n_heads) {
    static constexpr std::pair<std::string_view, const char*> kMap[] = {
        {"f_proj.weight", "linear_attn.in_proj_a.weight"},
        {"b_proj.weight", "linear_attn.in_proj_b.weight"},
        {"o_proj.weight", "linear_attn.out_proj.weight"},
        {"o_norm.weight", "linear_attn.norm.weight"},
        {"A_log", "linear_attn.A_log"},
        {"dt_bias", "linear_attn.dt_bias"},
        {"q_a_proj.weight", "self_attn.q_a_proj.weight"},
        {"q_a_layernorm.weight", "self_attn.q_a_layernorm.weight"},
        {"q_b_proj.weight", "self_attn.q_proj.weight"},
        {"kv_a_proj_with_mqa.weight", "self_attn.kv_a_proj_with_mqa.weight"},
        {"kv_a_layernorm.weight", "self_attn.kv_a_layernorm.weight"},
        {"kv_b_proj.weight", "self_attn.kv_b_proj.weight"},
        {"dense.weight", "self_attn.o_proj.weight"},
    };
    for (const auto& [from, to] : kMap)
        if (sub == from)
            return to;
    // g_proj: MLA head-wise output gate [n_heads, d_model] or the KDA gated-norm gate [H*128, d_model].
    if (sub == "g_proj.weight")
        return t.shape[0] == n_heads ? "self_attn.head_gate.weight" : "linear_attn.in_proj_z.weight";
    return nullptr;
}

// Records q/k/v proj and conv pieces; true when `sub` was one of them.
bool take_kda_piece(std::string_view sub, const Tensor& t, KdaPieces& p) {
    static constexpr std::string_view kProj[3] = {"q_proj.weight", "k_proj.weight", "v_proj.weight"};
    static constexpr std::string_view kConv[3] = {"q_conv1d.weight", "k_conv1d.weight", "v_conv1d.weight"};
    for (int i = 0; i < 3; ++i) {
        if (sub == kProj[i]) {
            p.proj[i] = &t;
            return true;
        }
        if (sub == kConv[i]) {
            p.conv[i] = &t;
            return true;
        }
    }
    return false;
}

std::string canonical_top_level(const std::string& name) {
    if (name == "model.word_embeddings.weight")
        return "model.embed_tokens.weight";
    constexpr std::string_view kBias = ".mlp.gate.expert_bias";
    if (name.ends_with(kBias))
        return name.substr(0, name.size() - kBias.size()) + ".mlp.gate.bias";
    return name;
}

}  // namespace

void arch_prepare_tensor_map(const ModelConfig& cfg, std::unordered_map<std::string, Tensor>& tensors,
                             std::vector<void*>& owned) {
    if (cfg.arch != ModelArch::BAILING_HYBRID)
        return;
    constexpr std::string_view kLayers = "model.layers.";
    constexpr std::string_view kAttn = ".attention.";
    std::unordered_map<std::string, Tensor> out;
    std::map<int, KdaPieces> kda;
    out.reserve(tensors.size());
    for (const auto& [name, t] : tensors) {
        const size_t attn = name.find(kAttn);
        if (!name.starts_with(kLayers) || attn == std::string::npos) {
            out.emplace(canonical_top_level(name), t);
            continue;
        }
        const int layer = parse_index(name.substr(kLayers.size(), attn - kLayers.size()));
        if (layer < 0 || layer >= cfg.n_layers)
            throw std::runtime_error("bailing_hybrid: tensor " + name + " names no declared layer");
        const std::string prefix = name.substr(0, attn + 1);  // model.layers.N.
        const std::string_view sub = std::string_view(name).substr(attn + kAttn.size());
        if (take_kda_piece(sub, tensors.at(name), kda[layer]))
            continue;
        const char* target = attention_target(sub, t, cfg.n_heads);
        out.emplace(target ? prefix + target : name, t);
    }
    size_t fused = 0;
    for (auto& [layer, p] : kda) {
        const bool any = p.proj[0] || p.proj[1] || p.proj[2] || p.conv[0] || p.conv[1] || p.conv[2];
        const bool all = p.proj[0] && p.proj[1] && p.proj[2] && p.conv[0] && p.conv[1] && p.conv[2];
        if (!any)
            continue;
        if (!all)
            throw std::runtime_error("bailing_hybrid: layer " + std::to_string(layer) +
                                     " lacks one of the KDA q/k/v projections or convs");
        const std::string prefix = "model.layers." + std::to_string(layer) + ".linear_attn.";
        out.emplace(prefix + "in_proj_qkv.weight", concat_rows(p.proj, owned, "q/k/v projection"));
        out.emplace(prefix + "conv1d.weight", concat_rows(p.conv, owned, "q/k/v short conv"));
        ++fused;
    }
    IMP_LOG_INFO("bailing_hybrid: %zu KDA layers fused (q|k|v projections and short convs)", fused);
    tensors.swap(out);
}

}  // namespace imp
