// Builds the AWQ transform for one checkpoint: which input channels each
// weight group protects, and where the compensating 1/s is folded so the
// exported checkpoint stays a plain NVFP4 checkpoint that needs no runtime
// support.
//
// The site table (which consumers share an activation, which producer absorbs
// 1/s, whether that producer's norm carries a unit offset) lives in
// awq_sites.h and is decided from tensor names alone, so the CPU lane holds
// it. This file does the parts that need the checkpoint and the GPU: the
// activation statistic, the scale search, the safety scan over consumers this
// planner does not know by name, and the storage bound on the fold.
//
//   A  q,k,v            <- input_layernorm            fold into the norm
//   G  linear_attn.in_proj_{qkv,z,a,b} <- input_layernorm          same norm
//   B  gate,up          <- post_attention_layernorm    fold into the norm
//   C  o_proj           <- v_proj output channels      fold into v_proj rows
//   D  down_proj        <- up_proj output channels     fold into up_proj rows
//   E  linear_attn.out_proj <- linear_attn.norm        tied across value heads
//
// The norm folds are exact for BOTH conventions: for y = (x/rms(x)) * g the
// producer stores g/s, and for y = (x/rms(x)) * (1 + g) it stores
// (1 + g)/s - 1, so that (1 + g') = (1 + g)/s. rms(x) is computed before g in
// either case, so the norm's own statistic does not move. What is NOT exact is
// the STORAGE: near a gain of zero a BF16 half-ulp on a delta near -1 is an
// unbounded relative error on the gain, so every norm divisor is passed
// through awq_clamp_norm_divisors before it reaches either side of the pair
// (quant/awq_norm_fold.h).
//
// C, D and E are exact because attention is linear in v, SwiGLU's product is
// elementwise, and the GDN gate multiplies after the norm: scaling one output
// channel of the producer scales precisely the matching input channel of the
// consumer. C and E additionally tie the ACTIVATION statistic across the heads
// that share a producer channel, which is what guarantees the fold exists at
// all: s is then a pure function of a.
//
// The row folds are searched first because their producers (v_proj, up_proj)
// are themselves members of the norm groups, and those searches must see the
// weights they will actually be quantizing.

#include "awq.h"

#include "core/fp_bits.h"
#include "model/json_util.h"
#include "quant/awq_transform.h"

#include <cuda_fp16.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <set>
#include <sstream>

namespace imp::awq {

namespace {

// How far the gain a folded norm channel reproduces may sit from the gain the
// fold intended. Below this the fold is worth having; a channel that cannot
// hold it keeps a smaller scale instead of losing its gain.
constexpr float kNormFoldTol = 0.01f;

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

std::string obj_str(const JValue& v, const char* key) {
    const JValue* f = obj_get(v, key);
    return (f && f->type == JType::STRING) ? f->str_val : std::string();
}

struct Shape2 {
    int64_t N = 0, K = 0;
};

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

// A 1-D producer's values, in float. The clamp needs the numbers the tensor
// actually holds, not an approximation of them: the whole bound is about what
// this dtype can and cannot store.
bool raw_to_float(const RawTensor& t, std::vector<float>& out) {
    const int64_t n = t.numel();
    out.resize(static_cast<size_t>(n));
    if (t.dtype == "F32") {
        std::memcpy(out.data(), t.data, static_cast<size_t>(n) * 4);
        return true;
    }
    const auto* src = static_cast<const uint16_t*>(t.data);
    if (t.dtype == "F16") {
        for (int64_t i = 0; i < n; i++)
            out[static_cast<size_t>(i)] = half_to_float(src[i]);
        return true;
    }
    if (t.dtype == "BF16") {
        for (int64_t i = 0; i < n; i++)
            out[static_cast<size_t>(i)] = bf16_to_float(src[i]);
        return true;
    }
    return false;
}

// Every 2-D `.weight` under `prefix` that is not a member and not explicitly
// exempt. Used to catch consumers this file does not know about by name — a new
// architecture adding a second reader of a pre-norm must not silently break the
// fold.
//
// This is not hypothetical. Two weight roles are deliberately excluded from
// quantization (MLA latent projections, MoE router) and therefore never receive
// the compensating column scale, and the MoE router reads exactly the norm
// group B folds into. A MoE checkpoint whose expert names happened to match
// would have had its router silently fed inputs divided by s.
std::vector<std::string> unlisted_consumers(const std::map<std::string, const RawTensor*>& index,
                                            const std::string& prefix,
                                            const std::vector<std::string>& members,
                                            const std::vector<std::string>& exempt_suffixes) {
    std::vector<std::string> out;
    for (const auto& [name, t] : index) {
        if (name.rfind(prefix, 0) != 0 || t->shape.size() != 2)
            continue;
        if (name.size() < 7 || name.compare(name.size() - 7, 7, ".weight") != 0)
            continue;
        if (std::find(members.begin(), members.end(), name) != members.end())
            continue;
        bool skip = false;
        for (const auto& suf : exempt_suffixes)
            if (name.size() >= suf.size() &&
                name.compare(name.size() - suf.size(), suf.size(), suf) == 0)
                skip = true;
        if (!skip)
            out.push_back(name);
    }
    return out;
}

// The site's activation statistic: the max over the kinds that read this
// activation, then tied across the producer channels the fold shares. Taking
// the max is robustness against one kind being absent from the calibration
// file; tying BEFORE the search is what makes the resulting scale foldable.
bool site_statistic(const CalibrationStats& stats, int layer, const FoldSite& site, int64_t K,
                    const std::vector<int64_t>& tie, int64_t producer_len, std::vector<float>& out) {
    out.clear();
    for (const auto& key : site.calib_keys) {
        const CalibrationEntry* e = stats.find(layer, key);
        if (!e || static_cast<int64_t>(e->mean_abs.size()) != K)
            continue;
        if (out.empty())
            out = e->mean_abs;
        else
            for (size_t i = 0; i < out.size(); i++)
                out[i] = std::max(out[i], e->mean_abs[i]);
    }
    if (out.empty())
        return false;
    if (site.tie == TieMode::Identity)
        return true;
    std::vector<float> per_producer(static_cast<size_t>(producer_len), 0.0f);
    for (int64_t i = 0; i < K; i++)
        per_producer[static_cast<size_t>(tie[static_cast<size_t>(i)])] =
            std::max(per_producer[static_cast<size_t>(tie[static_cast<size_t>(i)])],
                     out[static_cast<size_t>(i)]);
    for (int64_t i = 0; i < K; i++)
        out[static_cast<size_t>(i)] = per_producer[static_cast<size_t>(tie[static_cast<size_t>(i)])];
    return true;
}

// Runs one site: search, then record the column scales and the fold. Returns
// the error text only on a hard error; a site that cannot run is reported
// through `plan` and leaves the checkpoint untransformed at that site.
std::expected<void, std::string> run_site(const std::map<std::string, const RawTensor*>& index,
                                          const CalibrationStats& stats, int layer, const FoldSite& site,
                                          Geometry geo, Plan& plan) {
    const std::string label = "layer " + std::to_string(layer) + " group " + std::string(1, site.group);
    auto skip = [&](const std::string& why) {
        plan.groups_skipped++;
        plan.notes.push_back(label + ": " + why);
    };

    std::vector<std::vector<uint16_t>> storage;
    std::vector<GroupMatrix> mats;
    int64_t K = -1;
    storage.reserve(site.members.size());
    for (const auto& name : site.members) {
        std::vector<uint16_t> bits;
        Shape2 sh;
        if (!fetch_matrix(index, plan, name, bits, sh)) {
            skip("missing or unquantizable member " + name);
            return {};
        }
        if (K < 0)
            K = sh.K;
        else if (K != sh.K) {
            skip("members disagree on K");
            return {};
        }
        storage.push_back(std::move(bits));
        mats.push_back({nullptr, sh.N});
    }
    // `mats` holds pointers into `storage`; reserve() above keeps them stable.
    for (size_t i = 0; i < mats.size(); i++)
        mats[i].data = &storage[i];
    if (K <= 0 || K % 16 != 0) {
        skip("K = " + std::to_string(K) + " is not an NVFP4 width");
        return {};
    }

    const auto pit = index.find(site.producer);
    if (pit == index.end()) {
        skip("producer " + site.producer + " is not in this checkpoint");
        return {};
    }
    const RawTensor& producer = *pit->second;
    // One producer, one fold. Two groups of the same layer folding into the
    // same tensor would leave the second one's divisor standing and the first
    // one's consumers scaled against a divisor nothing applied.
    if (plan.vec_div.count(site.producer) || plan.row_div.count(site.producer)) {
        skip(site.producer + " was already folded by another group of this layer");
        return {};
    }
    // A norm's own length IS the tie: the GDN gated norm is [head_dim] and
    // shared across the value heads, and reading its width off the tensor
    // avoids depending on which config key spells the linear-attention head.
    if (site.tie == TieMode::PerHeadDim)
        geo.head_dim = producer.numel();
    const int64_t plen = tie_producer_len(site.tie, K, geo);
    const std::vector<int64_t> tie = tie_map(site.tie, K, geo);
    if (plen <= 0 || tie.empty()) {
        skip("K = " + std::to_string(K) + " does not tie onto " + site.producer);
        return {};
    }
    const bool producer_fits = site.kind == FoldKind::MatrixRows
                                   ? (producer.shape.size() == 2 && producer.shape[0] == plen)
                                   : (producer.numel() == plen);
    if (!producer_fits) {
        skip(site.producer + " has the wrong shape for this fold");
        return {};
    }

    // Dividing a norm divides its output for EVERY consumer of it, not only the
    // group's members. A consumer this planner does not know by name is a
    // blocker, not a detail.
    if (!site.scan_prefix.empty()) {
        const auto extra = unlisted_consumers(index, site.scan_prefix, site.members, site.scan_exempt);
        if (!extra.empty()) {
            skip("no fold, " + extra.front() + " also reads " + site.producer + " but is not scaled");
            return {};
        }
    }

    std::vector<float> act;
    if (!site_statistic(stats, layer, site, K, tie, plen, act)) {
        skip("no calibration entry of width " + std::to_string(K));
        return {};
    }

    auto searched = search_group_scale(mats, K, act);
    if (!searched)
        return std::unexpected(searched.error());
    SearchResult res = std::move(*searched);
    plan.group_errors.push_back({layer, std::string(1, site.group), res.err_rtn, res.err_best});
    if (res.alpha == 0.0f) {
        plan.groups_rtn++;
        return {};
    }

    std::vector<float> div(static_cast<size_t>(plen), 1.0f);
    for (int64_t i = 0; i < K; i++)
        div[static_cast<size_t>(tie[static_cast<size_t>(i)])] = res.s[static_cast<size_t>(i)];

    if (site.kind == FoldKind::NormVector) {
        std::vector<float> g;
        if (!raw_to_float(producer, g)) {
            skip("producer dtype " + producer.dtype + " cannot carry a fold");
            return {};
        }
        NormFoldReport rep;
        awq_clamp_norm_divisors(g.data(), g.size(), site.offset, producer.dtype, kNormFoldTol, div, rep);
        plan.channels_clamped += static_cast<int>(rep.clamped);
        if (rep.clamped > 0) {
            char worst[96];
            std::snprintf(worst, sizeof(worst), "%.3f -> %.3f", double(rep.worst_clamp_from),
                          double(rep.worst_clamp_to));
            plan.notes.push_back(label + ": " + std::to_string(rep.clamped) + " of " +
                                 std::to_string(rep.channels) + " channels clamped so " +
                                 producer.dtype + " can carry the fold (worst " + worst + ")");
        }
        // The consumer must be scaled by what the producer can actually store,
        // or the pair stops being inverse on exactly the channels that needed
        // protecting most.
        for (int64_t i = 0; i < K; i++)
            res.s[static_cast<size_t>(i)] = div[static_cast<size_t>(tie[static_cast<size_t>(i)])];
        plan.vec_div[site.producer] = div;
        plan.vec_offset[site.producer] = site.offset;
    } else {
        plan.row_div[site.producer] = div;
        if (!site.producer_bias.empty() && index.count(site.producer_bias))
            plan.vec_div[site.producer_bias] = div;
    }

    plan.groups_scaled++;
    for (const auto& name : site.members)
        plan.col_scale[name] = res.s;
    return {};
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
        // __half_raw, not memcpy over __half — the latter is a class with a
        // protected member and the compiler is right to complain.
        const __half_raw h(__float2half(bf16_to_float(src[i])));
        out[static_cast<size_t>(i)] = h.x;
    }
    return out;
}

std::expected<Plan, std::string> build_plan(const std::map<std::string, const RawTensor*>& index,
                                            const CalibrationStats& stats,
                                            const std::string& config_json_path, const std::string& groups) {
    Plan plan;
    if (groups.empty())
        return std::unexpected("--calib-groups selects no group; use a subset of " +
                               std::string(kAwqAllGroups));
    if (groups.find_first_not_of(kAwqAllGroups) != std::string::npos)
        return std::unexpected("--calib-groups '" + groups + "' has a letter outside " +
                               std::string(kAwqAllGroups));
    if (groups != kAwqAllGroups)
        plan.notes.push_back("group selector: " + groups + " (default " + kAwqAllGroups +
                             ") - this is a diagnostic subset, not a normal checkpoint");

    std::ifstream f(config_json_path, std::ios::binary);
    if (!f) {
        return std::unexpected("cannot read " + config_json_path + " (needed for the architecture check)");
    }
    std::stringstream ss;
    ss << f.rdbuf();
    const std::string text = ss.str();
    JsonParser parser(text);
    JValue cfg = parser.parse();
    if (!parser.ok() || cfg.type != JType::OBJECT) {
        return std::unexpected("cannot parse " + config_json_path);
    }

    // The convention is looked up, not guessed: imp applies the +1 at load for
    // the qwen3_5 family and at kernel time for Gemma, and the fold has to know
    // which. A wrapper config states the text model's own type under
    // text_config, so both spellings are tried.
    const JValue* text_cfg = obj_get(cfg, "text_config");
    std::string model_type = obj_str(cfg, "model_type");
    auto conv = arch_norm_convention(model_type);
    if (!conv && text_cfg) {
        const std::string inner = obj_str(*text_cfg, "model_type");
        if (auto nested = arch_norm_convention(inner)) {
            conv = nested;
            model_type = inner;
        }
    }
    if (!conv) {
        return std::unexpected("--calib does not support model_type '" + model_type +
                               "': this tool folds a plain (g) or unit-offset (1 + g) RMSNorm and has "
                               "no block layout for that architecture. Quantize without --calib instead.");
    }

    auto geom_int = [&](const char* key, int64_t fallback) {
        const int64_t top = obj_int(cfg, key, -1);
        if (top > 0)
            return top;
        return text_cfg ? obj_int(*text_cfg, key, fallback) : fallback;
    };
    const int64_t n_layers = geom_int("num_hidden_layers", 0);
    const int64_t n_heads = geom_int("num_attention_heads", 0);
    const int64_t n_kv_heads = geom_int("num_key_value_heads", n_heads);
    const int64_t hidden = geom_int("hidden_size", 0);
    int64_t head_dim = geom_int("head_dim", 0);
    if (head_dim <= 0 && n_heads > 0)
        head_dim = hidden / n_heads;
    if (n_layers <= 0 || n_heads <= 0 || n_kv_heads <= 0 || head_dim <= 0) {
        return std::unexpected("config.json is missing the layer/head geometry --calib needs");
    }
    const Geometry geo{head_dim, n_heads / n_kv_heads};

    // The layer prefix comes off the checkpoint. Hardcoding "model.layers."
    // cost nothing visible on a hybrid that names them
    // model.language_model.layers.N: every group found zero members and the
    // export was labelled calibrated.
    std::set<std::string> names;
    for (const auto& [name, t] : index)
        names.insert(name);
    const std::string prefix = resolve_layer_prefix(names);
    if (prefix.empty()) {
        return std::unexpected("cannot find the per-layer tensor prefix in this checkpoint "
                               "(neither model.layers.N nor a nested equivalent)");
    }
    printf("AWQ: model_type %s, %s norm, layers under %s\n", model_type.c_str(),
           *conv == NormConvention::UnitOffset ? "unit-offset (1 + g)" : "plain", prefix.c_str());

    for (int64_t L = 0; L < n_layers; L++) {
        const std::string base = prefix + std::to_string(L) + ".";
        const auto selected = layer_fold_sites(names, base, *conv, groups);
        const auto all = layer_fold_sites(names, base, *conv, kAwqAllGroups);
        plan.groups_disabled += static_cast<int>(all.size() - selected.size());
        for (const auto& site : selected) {
            const auto ran = run_site(index, stats, static_cast<int>(L), site, geo, plan);
            if (!ran)
                return std::unexpected(ran.error());
        }
        // A MoE layer's FFN weight is in per-expert tensors this planner does
        // not group, so the experts - the bulk of the model - stay at
        // round-to-nearest. Say so; a silent absence reads like a technicality
        // rather than "most of the weights were not calibrated".
        if (all.empty() && index.count(base + "mlp.experts.0.gate_proj.weight")) {
            plan.notes.push_back("layer " + std::to_string(L) +
                                 ": MoE experts NOT calibrated (per-expert groups are not "
                                 "modelled yet) - they stay round-to-nearest");
        }

        if ((L + 1) % 8 == 0 || L + 1 == n_layers)
            printf("  AWQ search: layer %lld/%lld\n", (long long)(L + 1), (long long)n_layers);
        fflush(stdout);
    }

    return plan;
}

}  // namespace imp::awq
