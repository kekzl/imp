#include "act_prescale.h"

#include "expert_destack.h"
#include "tensor_policy.h"

#include <cstdio>
#include <filesystem>
#include <string_view>
#include <vector>

namespace imp::quantize {

namespace {

bool ends_with(std::string_view s, std::string_view suf) {
    return s.size() >= suf.size() && s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

// v[i] *= f for i in [lo, hi), creating the all-ones vector of length n first.
void multiply(std::map<std::string, std::vector<float>>& m, const std::string& name, int64_t n, int64_t lo,
              int64_t hi, float f) {
    auto& v = m[name];
    if (v.empty())
        v.assign(static_cast<size_t>(n), 1.0f);
    for (int64_t i = lo; i < hi; ++i)
        v[static_cast<size_t>(i)] *= f;
}

struct PairRule {
    std::string_view producer_suffix;
    std::string_view consumer_suffix;
    float factor;
    bool c_rows_only;  // conv.in_proj [B | C | x]: only the C third feeds out_proj
};

constexpr PairRule kRules[] = {
    {".w3.weight", ".w2.weight", 64.0f, false},
    {".conv.in_proj.weight", ".conv.out_proj.weight", 16.0f, true},
    {".self_attn.v_proj.weight", ".self_attn.out_proj.weight", 8.0f, false},
};

}  // namespace

int add_lfm2_act_prescale(awq::Plan& plan, const std::map<std::string, const RawTensor*>& index,
                          const std::function<bool(const RawTensor&)>& quantized) {
    int pairs = 0;
    for (const auto& [name, t] : index) {
        for (const auto& r : kRules) {
            if (!ends_with(name, r.producer_suffix) || t->shape.size() != 2)
                continue;
            const std::string stem = name.substr(0, name.size() - r.producer_suffix.size());
            const auto c = index.find(stem + std::string(r.consumer_suffix));
            if (c == index.end() || c->second->shape.size() != 2 || !quantized(*t) || !quantized(*c->second))
                continue;
            const int64_t rows = t->shape[0];
            const int64_t lo = r.c_rows_only ? rows / 3 : 0;
            const int64_t hi = r.c_rows_only ? 2 * (rows / 3) : rows;
            if (r.c_rows_only && rows % 3 != 0)
                continue;
            multiply(plan.row_div, name, rows, lo, hi, 1.0f / r.factor);  // producer rows * factor
            multiply(plan.col_scale, c->first, c->second->shape[1], 0, c->second->shape[1],
                     1.0f / r.factor);  // consumer cols / factor
            ++pairs;
        }
    }
    return pairs;
}

void apply_lfm2_prescale(const Options& opt, const std::vector<std::unique_ptr<RawSafeTensors>>& opened,
                         const std::set<std::string>& gated_q_proj, awq::Plan& plan) {
    const std::string mt = model_type_from_config(
        (std::filesystem::path(opt.in_dir) / "config.json").string());
    if ((mt != "lfm2" && mt != "lfm2_moe") || opt.format != OutputFormat::Modelopt)
        return;
    std::map<std::string, const RawTensor*> index;
    for (const auto& src : opened)
        for (const auto& t : src->tensors())
            index[t.name] = &t;
    const int pairs = add_lfm2_act_prescale(plan, index, [&](const RawTensor& t) {
        std::string why;
        return gated_q_proj.count(t.name) == 0 && !keep_gdn_projection(t.name, opt.keep_gdn_proj) &&
               should_quantize(t, opt.quantize_lm_head, why);
    });
    printf(
        "LFM2 activation pre-scale: %d producer/consumer pairs folded (w2 x1/64, conv.out_proj x1/16, "
        "out_proj x1/8)\n",
        pairs);
}

}  // namespace imp::quantize
