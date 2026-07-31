// Builds the AWQ transform for one checkpoint: which input channels each
// weight group protects, and where the compensating 1/s is folded so the
// exported checkpoint stays a plain NVFP4 checkpoint that needs no runtime
// support.
//
// Four groups per layer, defined by which activation they share:
//
//   A  q,k,v      <- input_layernorm            fold 1/s into the norm weight
//   B  gate,up    <- post_attention_layernorm   fold 1/s into the norm weight
//   C  o_proj     <- v_proj output channels     fold 1/s into v_proj's rows
//   D  down_proj  <- up_proj output channels    fold 1/s into up_proj's rows
//
// A and B are exact for any pre-norm RMSNorm of the form y = (x/rms(x)) * g:
// dividing g by s divides the norm's output by s, and rms(x) is computed
// before g so it does not move. That is exactly why Gemma-class models are
// refused here — their norm applies (1 + g), where dividing g is not dividing
// the output.
//
// C and D are exact because attention is linear in v and SwiGLU's product is
// elementwise: scaling one output channel of the producer scales precisely the
// matching input channel of the consumer. C additionally has to respect GQA —
// several query heads read one KV head, so every o_proj input channel mapping
// to the same v channel must get the same scale. Tying the ACTIVATION statistic
// before the search is what guarantees that: s is a pure function of a.
//
// Groups C and D are searched first because their fold modifies v_proj and
// up_proj, which are themselves members of groups A and B — those searches
// must see the weights they will actually be quantizing.

#include "awq.h"

#include "model/json_util.h"
#include "quant/awq_transform.h"

#include <cuda_fp16.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>

namespace imp::awq {

namespace {

const JValue* obj_get(const JValue& v, const char* key) {
    if (v.type != JType::OBJECT)
        return nullptr;
    for (const auto& [k, val] : v.obj)
        if (k == key)
            return &val;
    return nullptr;
}

int64_t obj_int(const JValue& v, const char* key, int64_t fallback) {
    const JValue* f = obj_get(v, key);
    return (f && f->type == JType::NUMBER) ? f->as_int() : fallback;
}

// The norm fold assumes RMSNorm applies its weight multiplicatively with no
// +1 offset. Refusing everything else is the whole safety story for group A/B:
// a Gemma-style (1 + g) norm would silently produce a different model.
bool arch_uses_plain_rmsnorm(const std::string& model_type) {
    static const char* kOk[] = {"qwen2", "qwen3", "llama", "mistral"};
    for (const char* a : kOk)
        if (model_type == a)
            return true;
    return false;
}

struct Shape2 {
    int64_t N = 0, K = 0;
};

bool tensor_shape(const std::map<std::string, const RawTensor*>& index, const std::string& name,
                  Shape2& out) {
    auto it = index.find(name);
    if (it == index.end() || it->second->shape.size() != 2)
        return false;
    out.N = it->second->shape[0];
    out.K = it->second->shape[1];
    return true;
}

// Fetches a matrix as FP16, with any already-decided row divisor applied so the
// search sees the values that will actually be quantized.
bool fetch_matrix(const std::map<std::string, const RawTensor*>& index, const Plan& plan,
                  const std::string& name, std::vector<uint16_t>& out, Shape2& shape) {
    auto it = index.find(name);
    if (it == index.end() || it->second->shape.size() != 2)
        return false;
    const RawTensor& t = *it->second;
    if (t.dtype != "BF16" && t.dtype != "F16")
        return false;
    shape.N = t.shape[0];
    shape.K = t.shape[1];
    out = raw_to_fp16(t);
    auto rd = plan.row_div.find(name);
    if (rd != plan.row_div.end())
        awq_apply_matrix(out, shape.N, shape.K, rd->second, {});
    return true;
}

// Runs one group: search, then record the column scales and the fold. Returns
// false only on a hard error; a skipped group is reported through `plan`.
bool run_group(const std::map<std::string, const RawTensor*>& index, const CalibrationStats& stats, int layer,
               const char* kind, const std::vector<std::string>& members,
               const std::vector<float>& act_override, Plan& plan, SearchResult& result, std::string& err) {
    // Cleared up front: callers read `result.alpha` to decide whether to fold,
    // and a skipped group must never leave the previous group's scale standing.
    result = SearchResult{};
    const CalibrationEntry* stat = stats.find(layer, kind);
    if (!stat && act_override.empty()) {
        plan.groups_skipped++;
        return true;
    }
    const std::vector<float>& act = act_override.empty() ? stat->mean_abs : act_override;

    std::vector<std::vector<uint16_t>> storage;
    std::vector<GroupMatrix> mats;
    int64_t K = -1;
    storage.reserve(members.size());
    for (const auto& name : members) {
        std::vector<uint16_t> bits;
        Shape2 sh;
        if (!fetch_matrix(index, plan, name, bits, sh)) {
            plan.groups_skipped++;
            plan.notes.push_back("layer " + std::to_string(layer) + " " + kind + ": missing or " +
                                 "unquantizable member " + name);
            return true;
        }
        if (K < 0)
            K = sh.K;
        else if (K != sh.K) {
            plan.groups_skipped++;
            plan.notes.push_back("layer " + std::to_string(layer) + " " + kind + ": members disagree on K");
            return true;
        }
        storage.push_back(std::move(bits));
        mats.push_back({&storage.back(), sh.N});
    }
    if (K <= 0 || K % 16 != 0 || static_cast<int64_t>(act.size()) != K) {
        plan.groups_skipped++;
        plan.notes.push_back("layer " + std::to_string(layer) + " " + kind +
                             ": K does not match the calibration vector");
        return true;
    }
    // `mats` holds pointers into `storage`; reserve() above keeps them stable.
    for (size_t i = 0; i < members.size(); i++)
        mats[i].data = &storage[i];

    if (!search_group_scale(mats, K, act, result, err))
        return false;

    if (result.alpha == 0.0f) {
        plan.groups_rtn++;
        return true;
    }
    plan.groups_scaled++;
    for (const auto& name : members)
        plan.col_scale[name] = result.s;
    return true;
}

}  // namespace

std::vector<uint16_t> raw_to_fp16(const RawTensor& t) {
    const int64_t n = t.numel();
    std::vector<uint16_t> out(static_cast<size_t>(n));
    const auto* src = static_cast<const uint16_t*>(t.data);
    if (t.dtype == "F16") {
        std::memcpy(out.data(), src, static_cast<size_t>(n) * 2);
        return out;
    }
    for (int64_t i = 0; i < n; i++) {
        uint32_t bits = static_cast<uint32_t>(src[i]) << 16;
        float f;
        std::memcpy(&f, &bits, sizeof(float));
        // __half_raw, not memcpy over __half — the latter is a class with a
        // protected member and the compiler is right to complain.
        const __half_raw h(__float2half(f));
        out[static_cast<size_t>(i)] = h.x;
    }
    return out;
}

bool build_plan(const std::map<std::string, const RawTensor*>& index, const CalibrationStats& stats,
                const std::string& config_json_path, Plan& plan, std::string& err) {
    std::ifstream f(config_json_path, std::ios::binary);
    if (!f) {
        err = "cannot read " + config_json_path + " (needed for the architecture check)";
        return false;
    }
    std::stringstream ss;
    ss << f.rdbuf();
    const std::string text = ss.str();
    JsonParser parser(text.data(), text.size());
    JValue cfg = parser.parse();
    if (!parser.ok() || cfg.type != JType::OBJECT) {
        err = "cannot parse " + config_json_path;
        return false;
    }

    const JValue* mt = obj_get(cfg, "model_type");
    const std::string model_type = (mt && mt->type == JType::STRING) ? mt->str_val : "";
    if (!arch_uses_plain_rmsnorm(model_type)) {
        err = "--calib does not support model_type '" + model_type +
              "': the norm fold is only valid for a plain multiplicative RMSNorm "
              "(qwen2/qwen3/llama/mistral). Quantize without --calib instead.";
        return false;
    }

    const int64_t n_layers = obj_int(cfg, "num_hidden_layers", 0);
    const int64_t n_heads = obj_int(cfg, "num_attention_heads", 0);
    const int64_t n_kv_heads = obj_int(cfg, "num_key_value_heads", n_heads);
    const int64_t hidden = obj_int(cfg, "hidden_size", 0);
    int64_t head_dim = obj_int(cfg, "head_dim", 0);
    if (head_dim <= 0 && n_heads > 0)
        head_dim = hidden / n_heads;
    if (n_layers <= 0 || n_heads <= 0 || n_kv_heads <= 0 || head_dim <= 0) {
        err = "config.json is missing the layer/head geometry --calib needs";
        return false;
    }
    const int64_t n_rep = n_heads / n_kv_heads;

    for (int64_t L = 0; L < n_layers; L++) {
        const std::string base = "model.layers." + std::to_string(L) + ".";
        const std::string v_name = base + "self_attn.v_proj.weight";
        const std::string o_name = base + "self_attn.o_proj.weight";
        const std::string up_name = base + "mlp.up_proj.weight";
        const std::string down_name = base + "mlp.down_proj.weight";
        SearchResult res;

        // ---- Group C: o_proj, folded into v_proj's output rows ----
        // This one folds into the tensor the KV CACHE stores, and imp's default
        // KV dtype on this model class is FP8_E4M3 — so the obvious worry is
        // that widening v's per-channel range costs more in the cache than the
        // scale wins in the weight. Measured on Qwen3-0.6B and REFUTED: the
        // FP8-vs-FP16-KV penalty is 0.300 PPL on the calibrated checkpoint and
        // 0.595 on the round-to-nearest one. The scaled v_proj is, if anything,
        // friendlier to FP8 KV than the unscaled one.
        Shape2 osh, vsh;
        const CalibrationEntry* o_stat = stats.find(static_cast<int>(L), "WO");
        if (o_stat && tensor_shape(index, o_name, osh) && tensor_shape(index, v_name, vsh) &&
            static_cast<int64_t>(o_stat->mean_abs.size()) == osh.K && osh.K == n_heads * head_dim &&
            vsh.N == n_kv_heads * head_dim) {
            // Tie the statistic across the query heads that share a KV head, so
            // the resulting scale is foldable into v_proj at all.
            std::vector<float> tied(static_cast<size_t>(osh.K), 0.0f);
            std::vector<float> per_v(static_cast<size_t>(vsh.N), 0.0f);
            for (int64_t i = 0; i < osh.K; i++) {
                const int64_t h = i / head_dim, d = i % head_dim;
                const int64_t vc = (h / n_rep) * head_dim + d;
                per_v[static_cast<size_t>(vc)] = std::max(per_v[static_cast<size_t>(vc)],
                                                          o_stat->mean_abs[static_cast<size_t>(i)]);
            }
            for (int64_t i = 0; i < osh.K; i++) {
                const int64_t h = i / head_dim, d = i % head_dim;
                tied[static_cast<size_t>(i)] = per_v[static_cast<size_t>((h / n_rep) * head_dim + d)];
            }
            if (!run_group(index, stats, static_cast<int>(L), "WO", {o_name}, tied, plan, res, err))
                return false;
            if (res.alpha != 0.0f && plan.col_scale.count(o_name)) {
                std::vector<float> row_div(static_cast<size_t>(vsh.N), 1.0f);
                for (int64_t i = 0; i < osh.K; i++) {
                    const int64_t h = i / head_dim, d = i % head_dim;
                    row_div[static_cast<size_t>((h / n_rep) * head_dim + d)] =
                        plan.col_scale[o_name][static_cast<size_t>(i)];
                }
                plan.row_div[v_name] = row_div;
                const std::string v_bias = base + "self_attn.v_proj.bias";
                if (index.count(v_bias))
                    plan.vec_div[v_bias] = row_div;
            }
        } else {
            plan.groups_skipped++;
        }

        // ---- Group D: down_proj, folded into up_proj's output rows ----
        Shape2 dsh, ush;
        if (tensor_shape(index, down_name, dsh) && tensor_shape(index, up_name, ush) && dsh.K == ush.N) {
            if (!run_group(index, stats, static_cast<int>(L), "W_DOWN", {down_name}, {}, plan, res, err))
                return false;
            if (res.alpha != 0.0f && plan.col_scale.count(down_name)) {
                plan.row_div[up_name] = plan.col_scale[down_name];
                const std::string up_bias = base + "mlp.up_proj.bias";
                if (index.count(up_bias))
                    plan.vec_div[up_bias] = plan.col_scale[down_name];
            }
        } else {
            plan.groups_skipped++;
        }

        // ---- Group A: q/k/v, folded into input_layernorm ----
        // Every member reads the same normed hidden state, so any member's
        // statistic describes the group; taking the max is just robustness
        // against one of them being absent from the calibration file.
        {
            std::vector<std::string> members;
            for (const char* p : {"q_proj", "k_proj", "v_proj"}) {
                const std::string n = base + "self_attn." + p + ".weight";
                if (index.count(n))
                    members.push_back(n);
            }
            std::vector<float> act;
            for (const char* k : {"WQ", "WK", "WV"}) {
                const CalibrationEntry* e = stats.find(static_cast<int>(L), k);
                if (!e)
                    continue;
                if (act.empty())
                    act = e->mean_abs;
                else if (act.size() == e->mean_abs.size())
                    for (size_t i = 0; i < act.size(); i++)
                        act[i] = std::max(act[i], e->mean_abs[i]);
            }
            const std::string norm = base + "input_layernorm.weight";
            if (!members.empty() && !act.empty() && index.count(norm)) {
                if (!run_group(index, stats, static_cast<int>(L), "WQ", members, act, plan, res, err))
                    return false;
                if (res.alpha != 0.0f)
                    plan.vec_div[norm] = res.s;
            } else {
                plan.groups_skipped++;
            }
        }

        // ---- Group B: gate/up, folded into post_attention_layernorm ----
        {
            std::vector<std::string> members;
            for (const char* p : {"gate_proj", "up_proj"}) {
                const std::string n = base + "mlp." + p + ".weight";
                if (index.count(n))
                    members.push_back(n);
            }
            std::vector<float> act;
            for (const char* k : {"W_GATE", "W_UP"}) {
                const CalibrationEntry* e = stats.find(static_cast<int>(L), k);
                if (!e)
                    continue;
                if (act.empty())
                    act = e->mean_abs;
                else if (act.size() == e->mean_abs.size())
                    for (size_t i = 0; i < act.size(); i++)
                        act[i] = std::max(act[i], e->mean_abs[i]);
            }
            const std::string norm = base + "post_attention_layernorm.weight";
            if (!members.empty() && !act.empty() && index.count(norm)) {
                if (!run_group(index, stats, static_cast<int>(L), "W_GATE", members, act, plan, res, err))
                    return false;
                if (res.alpha != 0.0f)
                    plan.vec_div[norm] = res.s;
            } else {
                plan.groups_skipped++;
            }
        }

        if ((L + 1) % 8 == 0 || L + 1 == n_layers)
            printf("  AWQ search: layer %lld/%lld\n", (long long)(L + 1), (long long)n_layers);
        fflush(stdout);
    }

    return true;
}

}  // namespace imp::awq
