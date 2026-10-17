// Sparse autoencoder host side (roadmap row 116): Gemma Scope 2 directory loader and the
// per-token feature summary.

#include "exec/sae.h"
#include "model/json_util.h"
#include "model/safetensors_raw.h"

#include <algorithm>
#include <cstring>
#include <numeric>

namespace imp {

SaeTokenInfo sae_token_info(const float* acts, int width, int top_n, float rel_err) {
    SaeTokenInfo info;
    info.rel_err = rel_err;
    std::vector<int32_t> ids;
    for (int j = 0; j < width; j++)
        if (acts[j] != 0.0f)
            ids.push_back(j);
    info.l0 = static_cast<int>(ids.size());
    const size_t keep = std::min(ids.size(), static_cast<size_t>(std::max(top_n, 0)));
    std::partial_sort(ids.begin(), ids.begin() + static_cast<std::ptrdiff_t>(keep), ids.end(),
                      [&](int32_t a, int32_t b) { return acts[a] != acts[b] ? acts[a] > acts[b] : a < b; });
    for (size_t i = 0; i < keep; i++)
        info.top.emplace_back(ids[i], acts[ids[i]]);
    return info;
}

bool load_sae_dir(const std::string& dir, int d_model, int n_layers, SaeHost& out, std::string* err) {
    auto fail = [&](const std::string& why) {
        if (err)
            *err = "sae: " + dir + ": " + why;
        return false;
    };
    JValue cfg;
    if (!parse_json_file(dir + "/config.json", cfg))
        return fail("config.json missing or not a JSON object");
    std::string hook, arch;
    int64_t width = 0;
    jobj_opt_string(cfg, "hf_hook_point_in", hook);
    jobj_opt_string(cfg, "architecture", arch);
    jobj_opt_int(cfg, "width", width);
    if (arch != "jump_relu")
        return fail("architecture \"" + arch + "\" is not jump_relu");
    // "model.layers.N.output": the residual after decoder layer N.
    constexpr const char* kPre = "model.layers.";
    constexpr const char* kPost = ".output";
    if (hook.rfind(kPre, 0) != 0 || hook.size() <= std::strlen(kPre) + std::strlen(kPost) ||
        hook.compare(hook.size() - std::strlen(kPost), std::strlen(kPost), kPost) != 0)
        return fail("hf_hook_point_in \"" + hook + "\" is not model.layers.N.output");
    const std::string num = hook.substr(std::strlen(kPre),
                                        hook.size() - std::strlen(kPre) - std::strlen(kPost));
    if (num.empty() || !std::all_of(num.begin(), num.end(), [](char c) { return c >= '0' && c <= '9'; }))
        return fail("hf_hook_point_in \"" + hook + "\" has no layer number");
    const int layer = std::stoi(num);
    if (layer < 0 || layer >= n_layers)
        return fail("layer " + num + " outside the model's " + std::to_string(n_layers) + " layers");
    if (width <= 0)
        return fail("config.json has no positive width");

    RawSafeTensors st;
    if (const std::string why = st.open(dir + "/params.safetensors"); !why.empty())
        return fail(why);
    const int64_t w = width, d = d_model;
    struct Want {
        const char* name;
        std::vector<int64_t> shape;
        std::vector<float>* dst;
    };
    const Want want[] = {{"w_enc", {d, w}, &out.w_enc},
                         {"w_dec", {w, d}, &out.w_dec},
                         {"b_enc", {w}, &out.b_enc},
                         {"threshold", {w}, &out.threshold},
                         {"b_dec", {d}, &out.b_dec}};
    for (const auto& x : want) {
        const auto it = std::find_if(st.tensors().begin(), st.tensors().end(),
                                     [&](const RawTensor& t) { return t.name == x.name; });
        if (it == st.tensors().end())
            return fail(std::string("tensor ") + x.name + " missing");
        if (it->dtype != "F32" || it->shape != x.shape)
            return fail(std::string("tensor ") + x.name + " is not F32 of the model's d_model " +
                        std::to_string(d_model) + " and width " + std::to_string(width));
        x.dst->resize(static_cast<size_t>(it->numel()));
        std::memcpy(x.dst->data(), it->data, it->nbytes);
    }
    out.layer = layer;
    out.d_model = d_model;
    out.width = static_cast<int>(width);
    return true;
}

}  // namespace imp
