#include "core/logging.h"
#include "model/model_limits.h"
#include "model/weight_map_arch.h"

#include <array>
#include <cstddef>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>

namespace imp {

namespace {

// One KDA layer's q/k/v projections (every suffix: .weight, NVFP4 .weight_scale/_2, .input_scale) and
// short convs. Dense projections fuse into in_proj_qkv like bailing_hybrid; NVFP4 ones keep three global
// scales, so they stay three tensors (in_proj_q / in_proj_k / in_proj_v, ssm_in + kda_k_proj + kda_v_proj).
struct KimiKdaLayer {
    std::array<std::map<std::string, const Tensor*>, 3> proj;  // suffix -> tensor, for q, k, v
    std::array<const Tensor*, 3> conv = {nullptr, nullptr, nullptr};
    int renamed = 0;  // the other KDA tensors, renamed one to one
    bool a_log = false;
};

constexpr std::string_view kQkv[3] = {"q", "k", "v"};

// self_attn.<module><suffix> of a KDA layer -> linear_attn.<target><suffix>; nullptr = not a 1:1 rename.
const char* kda_target(std::string_view module) {
    static constexpr std::pair<std::string_view, const char*> kMap[] = {
        {"f_a_proj", "f_a_proj"},  {"f_b_proj", "in_proj_a"}, {"g_a_proj", "g_a_proj"},
        {"g_b_proj", "in_proj_z"}, {"b_proj", "in_proj_b"},   {"o_proj", "out_proj"},
        {"o_norm", "norm"},        {"A_log", "A_log"},        {"dt_bias", "dt_bias"},
    };
    for (const auto& [from, to] : kMap)
        if (module == from)
            return to;
    return nullptr;
}

// block_sparse_moe.<rest> -> the DeepSeek mlp.* spelling the weight map routes (NVFP4 scales included).
std::string moe_target(std::string_view rest) {
    constexpr std::string_view kExperts = "experts.";
    if (rest == "gate.weight")
        return "mlp.gate.weight";
    if (rest == "gate.e_score_correction_bias")
        return "mlp.gate.bias";
    if (rest.starts_with("shared_experts."))
        return "mlp." + std::string(rest);
    if (rest.starts_with(kExperts)) {
        const size_t dot = rest.find('.', kExperts.size());
        if (dot != std::string_view::npos && rest.size() > dot + 3 && rest[dot + 1] == 'w' &&
            rest[dot + 3] == '.') {
            const char* proj = rest[dot + 2] == '1'   ? "gate_proj"
                               : rest[dot + 2] == '3' ? "up_proj"
                               : rest[dot + 2] == '2' ? "down_proj"
                                                      : nullptr;
            if (proj)
                return "mlp." + std::string(rest.substr(0, dot + 1)) + proj +
                       std::string(rest.substr(dot + 3));
        }
    }
    return "block_sparse_moe." + std::string(rest);
}

// [1, 1, H, 1] A_log (HF view) -> [H]: the scalar upload reads numel(), the scans index [H].
Tensor flatten(const Tensor& t) {
    Tensor out = t;
    const int64_t n = t.numel();
    out.ndim = 1;
    out.shape[0] = n;
    for (int d = 1; d < kMaxDims; ++d)
        out.shape[d] = 0;
    out.compute_strides();
    return out;
}

void emit_qkv(int layer, KimiKdaLayer& L, std::unordered_map<std::string, Tensor>& out,
              std::vector<void*>& owned, size_t& fused, size_t& split) {
    const std::string prefix = "model.layers." + std::to_string(layer) + ".linear_attn.";
    for (int i = 0; i < 3; ++i)
        if (!L.proj[i].contains(".weight") || !L.conv[i])
            throw std::runtime_error("kimi_linear: KDA layer " + std::to_string(layer) + " lacks the " +
                                     std::string(kQkv[i]) + " projection or short conv");
    const Tensor* convs[3] = {L.conv[0], L.conv[1], L.conv[2]};
    out.emplace(prefix + "conv1d.weight", kda_concat_rows(convs, owned, "kimi_linear: q/k/v short conv"));
    int n_dense = 0;
    for (int i = 0; i < 3; ++i)
        n_dense += (L.proj[i].size() == 1 && kda_dense_tensor(*L.proj[i].at(".weight"))) ? 1 : 0;
    if (n_dense == 3) {
        const Tensor* parts[3] = {L.proj[0].at(".weight"), L.proj[1].at(".weight"), L.proj[2].at(".weight")};
        out.emplace(prefix + "in_proj_qkv.weight",
                    kda_concat_rows(parts, owned, "kimi_linear: q/k/v projection"));
        ++fused;
        return;
    }
    if (n_dense != 0)
        throw std::runtime_error("kimi_linear: KDA layer " + std::to_string(layer) +
                                 " mixes dense and quantized q/k/v projections");
    for (int i = 0; i < 3; ++i)
        for (const auto& [suffix, t] : L.proj[i])
            out.emplace(prefix + "in_proj_" + std::string(kQkv[i]) + suffix, *t);
    ++split;
}

}  // namespace

void kimi_linear_prepare_tensor_map(const ModelConfig& cfg, std::unordered_map<std::string, Tensor>& tensors,
                                    std::vector<void*>& owned) {
    if (static_cast<int>(cfg.kda_layers.size()) != cfg.n_layers)
        throw std::runtime_error(
            "kimi_linear: config carries no KDA layer list (linear_attn_config.kda_layers)");
    constexpr std::string_view kLayers = "model.layers.";
    constexpr std::string_view kSelf = "self_attn.";
    constexpr std::string_view kMoe = "block_sparse_moe.";
    std::unordered_map<std::string, Tensor> out;
    std::map<int, KimiKdaLayer> kda;
    out.reserve(tensors.size());
    for (const auto& [name, t] : tensors) {
        const size_t dot = name.starts_with(kLayers) ? name.find('.', kLayers.size()) : std::string::npos;
        if (dot == std::string::npos) {
            out.emplace(name, t);
            continue;
        }
        const int layer = parse_index(name.substr(kLayers.size(), dot - kLayers.size()));
        if (layer < 0 || layer >= cfg.n_layers)
            throw std::runtime_error("kimi_linear: tensor " + name + " names no declared layer");
        const std::string prefix = name.substr(0, dot + 1);  // model.layers.N.
        const std::string_view rest = std::string_view(name).substr(dot + 1);
        if (rest.starts_with(kMoe)) {
            out.emplace(prefix + moe_target(rest.substr(kMoe.size())), t);
            continue;
        }
        if (!rest.starts_with(kSelf)) {
            out.emplace(name, t);
            continue;
        }
        const std::string_view sub = rest.substr(kSelf.size());
        const size_t mdot = sub.find('.');
        const std::string_view module = sub.substr(0, mdot);
        const std::string suffix(mdot == std::string_view::npos ? std::string_view{} : sub.substr(mdot));
        const bool is_kda_layer = cfg.kda_layers[static_cast<size_t>(layer)] != 0;
        if (!is_kda_layer) {
            if (kda_target(module) != nullptr && module != "o_proj")
                throw std::runtime_error("kimi_linear: " + name +
                                         " is a KDA tensor in full-attention layer " + std::to_string(layer));
            out.emplace(name, t);  // MLA: q_proj, kv_a_proj_with_mqa, kv_a_layernorm, kv_b_proj, o_proj
            continue;
        }
        KimiKdaLayer& L = kda[layer];
        bool taken = false;
        for (int i = 0; i < 3 && !taken; ++i) {
            if (module == std::string(kQkv[i]) + "_proj") {
                L.proj[static_cast<size_t>(i)][suffix] = &t;
                taken = true;
            } else if (module == std::string(kQkv[i]) + "_conv1d" && suffix == ".weight") {
                L.conv[static_cast<size_t>(i)] = &t;
                taken = true;
            }
        }
        if (taken)
            continue;
        const char* target = kda_target(module);
        if (!target)
            throw std::runtime_error("kimi_linear: unrecognised KDA tensor " + name);
        const std::string to = prefix + "linear_attn." + target + suffix;
        if (module == "A_log") {
            out.emplace(to, flatten(t));
            L.a_log = true;
        } else {
            out.emplace(to, t);
        }
        ++L.renamed;
    }
    size_t fused = 0, split = 0;
    for (int i = 0; i < cfg.n_layers; ++i) {
        if (!cfg.kda_layers[static_cast<size_t>(i)])
            continue;
        auto it = kda.find(i);
        if (it == kda.end() || !it->second.a_log)
            throw std::runtime_error("kimi_linear: KDA layer " + std::to_string(i) +
                                     " has no A_log in the checkpoint");
        emit_qkv(i, it->second, out, owned, fused, split);
    }
    IMP_LOG_INFO(
        "kimi_linear: %zu KDA layers (%zu q|k|v fused dense, %zu split quantized), block_sparse_moe -> mlp",
        fused + split, fused, split);
    tensors.swap(out);
}

}  // namespace imp
