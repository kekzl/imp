#include "act_prescale.h"

#include "expert_destack.h"
#include "model/json_util.h"
#include "tensor_policy.h"

#include <cstdio>
#include <filesystem>
#include <span>
#include <string_view>
#include <vector>

namespace imp::quantize {

namespace {

bool ends_with(std::string_view s, std::string_view suf) {
    return s.size() >= suf.size() && s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

// v[i] *= f for every i with sel(i), creating the all-ones vector of length n first.
template <typename Sel>
void multiply(std::map<std::string, std::vector<float>>& m, const std::string& name, int64_t n, float f,
              Sel sel) {
    auto& v = m[name];
    if (v.empty())
        v.assign(static_cast<size_t>(n), 1.0f);
    for (int64_t i = 0; i < n; ++i)
        if (sel(i))
            v[static_cast<size_t>(i)] *= f;
}

enum class Rows {
    All,
    MiddleThird,  // conv.in_proj [B | C | x]: only C feeds out_proj
    MlaV,         // kv_b_proj per head [nope K | v V]: only V feeds o_proj
};

struct PairRule {
    std::string_view producer_suffix;
    std::string_view consumer_suffix;
    float factor;
    Rows rows;
    std::string_view must_contain;  // empty = any
    bool producer_kept_ok;          // the producer may stay full precision (copy path folds it)
};

constexpr PairRule kLfm2Rules[] = {
    {".w3.weight", ".w2.weight", 64.0f, Rows::All, "", false},
    {".conv.in_proj.weight", ".conv.out_proj.weight", 16.0f, Rows::MiddleThird, "", false},
    {".self_attn.v_proj.weight", ".self_attn.out_proj.weight", 8.0f, Rows::All, "", false},
};

// GLM-4.7-Flash: dense mlp.down_proj input 97 % flushed blocks (max 0.196), o_proj 13 % (max 1.234).
constexpr PairRule kGlmRules[] = {
    {".mlp.up_proj.weight", ".mlp.down_proj.weight", 64.0f, Rows::All, "", false},
    {".up_proj.weight", ".down_proj.weight", 16.0f, Rows::All, ".mlp.experts.", false},
    {".self_attn.kv_b_proj.weight", ".self_attn.o_proj.weight", 16.0f, Rows::MlaV, "", true},
};

// North-Mini-Code (cohere2_moe): dense mlp.down_proj input 11.74 % flushed (max 147, x16 < 448 x 6).
// PPL 11.9865 none, 11.8222 dense x16, 12.3546 with v_proj/o_proj x16 too (o_proj 3.30 % flushed).
constexpr PairRule kCohereRules[] = {
    {".mlp.up_proj.weight", ".mlp.down_proj.weight", 16.0f, Rows::All, "", false},
};

struct MlaDims {
    int64_t nope = 0, v = 0;
};

bool row_selected(Rows rows, int64_t i, int64_t n, const MlaDims& mla) {
    switch (rows) {
        case Rows::All:
            return true;
        case Rows::MiddleThird:
            return i >= n / 3 && i < 2 * (n / 3);
        case Rows::MlaV:
            return i % (mla.nope + mla.v) >= mla.nope;
    }
    return false;
}

bool rows_fit(Rows rows, int64_t n, const MlaDims& mla) {
    if (rows == Rows::MiddleThird)
        return n % 3 == 0;
    if (rows == Rows::MlaV)
        return mla.nope + mla.v > 0 && n % (mla.nope + mla.v) == 0;
    return true;
}

int fold_rules(awq::Plan& plan, const std::map<std::string, const RawTensor*>& index,
               std::span<const PairRule> rules, const MlaDims& mla,
               const std::function<bool(const RawTensor&)>& quantized) {
    int pairs = 0;
    for (const auto& [name, t] : index) {
        for (const auto& r : rules) {
            if (!ends_with(name, r.producer_suffix) || t->shape.size() != 2 ||
                (!r.must_contain.empty() && name.find(r.must_contain) == std::string::npos))
                continue;
            const std::string stem = name.substr(0, name.size() - r.producer_suffix.size());
            const auto c = index.find(stem + std::string(r.consumer_suffix));
            if (c == index.end() || c->second->shape.size() != 2 || !quantized(*c->second) ||
                (!r.producer_kept_ok && !quantized(*t)) || !rows_fit(r.rows, t->shape[0], mla))
                continue;
            const int64_t n = t->shape[0];
            multiply(plan.row_div, name, n, 1.0f / r.factor,  // producer rows * factor
                     [&](int64_t i) { return row_selected(r.rows, i, n, mla); });
            multiply(plan.col_scale, c->first, c->second->shape[1],
                     1.0f / r.factor,  // consumer cols / factor
                     [](int64_t) { return true; });
            ++pairs;
        }
    }
    return pairs;
}

}  // namespace

int add_lfm2_act_prescale(awq::Plan& plan, const std::map<std::string, const RawTensor*>& index,
                          const std::function<bool(const RawTensor&)>& quantized) {
    return fold_rules(plan, index, kLfm2Rules, {}, quantized);
}

void apply_act_prescale(const Options& opt, const std::vector<std::unique_ptr<RawSafeTensors>>& opened,
                        const std::set<std::string>& gated_q_proj, awq::Plan& plan) {
    const std::string cfg_path = (std::filesystem::path(opt.in_dir) / "config.json").string();
    const std::string mt = model_type_from_config(cfg_path);
    const bool lfm2 = mt == "lfm2" || mt == "lfm2_moe";
    const bool glm = mt == "glm4_moe_lite";
    std::span<const PairRule> rules = lfm2 ? std::span<const PairRule>(kLfm2Rules) : kGlmRules;
    if (mt == "cohere2_moe")
        rules = kCohereRules;
    else if (!lfm2 && !glm)
        return;
    if (opt.format != OutputFormat::Modelopt)
        return;
    MlaDims mla;
    JValue cfg;
    if (glm && parse_json_file(cfg_path, cfg) && cfg.type == JType::OBJECT) {
        jobj_opt_int(cfg, "qk_nope_head_dim", mla.nope);
        jobj_opt_int(cfg, "v_head_dim", mla.v);
    }
    std::map<std::string, const RawTensor*> index;
    for (const auto& src : opened)
        for (const auto& t : src->tensors())
            index[t.name] = &t;
    const auto quantized = [&](const RawTensor& t) {
        std::string why;
        return gated_q_proj.count(t.name) == 0 && !keep_gdn_projection(t.name, opt.keep_gdn_proj) &&
               should_quantize(t, opt.quantize_lm_head, why);
    };
    const int pairs = fold_rules(plan, index, rules, mla, quantized);
    printf("activation pre-scale (%s): %d producer/consumer pairs folded\n", mt.c_str(), pairs);
}

}  // namespace imp::quantize
