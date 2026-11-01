#include "fp8_keep.h"

#include "checkpoint_out.h"
#include "tensor_policy.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <sstream>

namespace imp::quantize {

namespace {

const char* group_of_module(const std::string& parent, const std::string& module) {
    if (parent == "self_attn") {
        if (module == "q_proj" || module == "k_proj" || module == "v_proj")
            return "qkv";
        if (module == "o_proj")
            return "o";
    } else if (parent == "mlp") {
        if (module == "gate_proj" || module == "up_proj")
            return "gate_up";
        if (module == "down_proj")
            return "down";
    }
    return nullptr;
}

bool valid_group(const std::string& g) { return g == "qkv" || g == "o" || g == "gate_up" || g == "down"; }

}  // namespace

std::string fp8_unit_of(const std::string& weight_name) {
    static const std::string kLayers = ".layers.", kW = ".weight";
    const size_t p = weight_name.find(kLayers);
    if (p == std::string::npos || !weight_name.ends_with(kW))
        return "";
    // <prefix>.layers.<L>.<parent>.<module>.weight, nothing between parent and module.
    const std::string rest = weight_name.substr(p + kLayers.size(),
                                                weight_name.size() - p - kLayers.size() - kW.size());
    std::vector<std::string> parts;
    std::stringstream ss(rest);
    for (std::string s; std::getline(ss, s, '.');)
        parts.push_back(s);
    if (parts.size() != 3 || parts[0].empty() ||
        parts[0].find_first_not_of("0123456789") != std::string::npos)
        return "";
    const char* g = group_of_module(parts[1], parts[2]);
    return g ? parts[0] + "." + g : "";
}

bool parse_fp8_keep_list(const std::string& spec, std::vector<std::string>& out, std::string& err) {
    // Split by hand: getline drops a trailing empty entry ("27.down," would pass).
    for (size_t b = 0; !spec.empty() && b <= spec.size();) {
        const size_t c = std::min(spec.find(',', b), spec.size());
        const std::string e = spec.substr(b, c - b);
        b = c + 1;
        const size_t dot = e.find('.');
        const std::string layer = dot == std::string::npos ? "" : e.substr(0, dot);
        const std::string group = dot == std::string::npos ? "" : e.substr(dot + 1);
        const bool layer_ok = layer == "*" ||
                              (!layer.empty() && layer.find_first_not_of("0123456789") == std::string::npos);
        if (!layer_ok || !valid_group(group)) {
            err = "'" + e + "': expected <layer|*>.<qkv|o|gate_up|down>";
            return false;
        }
        out.push_back(e);
    }
    if (out.empty()) {
        err = "empty unit list";
        return false;
    }
    return true;
}

bool fp8_keep_list_matches(const std::vector<std::string>& list, const std::string& unit) {
    const size_t dot = unit.find('.');
    if (unit.empty() || dot == std::string::npos)
        return false;
    const std::string any = "*" + unit.substr(dot);
    return std::ranges::any_of(list, [&](const std::string& e) { return e == unit || e == any; });
}

bool parse_fp8_rank_text(const std::string& text, std::map<std::string, double>& out, std::string& err) {
    std::stringstream ss(text);
    int line_no = 0;
    for (std::string line; std::getline(ss, line);) {
        line_no++;
        if (const size_t h = line.find('#'); h != std::string::npos)
            line.resize(h);
        std::stringstream ls(line);
        std::string unit, score_s, extra;
        if (!(ls >> unit))
            continue;
        char* end = nullptr;
        const bool has_score = static_cast<bool>(ls >> score_s);
        const double score = has_score ? std::strtod(score_s.c_str(), &end) : 0.0;
        std::vector<std::string> one;
        std::string unit_err;
        if (!has_score || end == score_s.c_str() || *end != '\0' || !std::isfinite(score) || (ls >> extra) ||
            !parse_fp8_keep_list(unit, one, unit_err) || unit.starts_with("*")) {
            err = "line " + std::to_string(line_no) + ": expected '<layer>.<qkv|o|gate_up|down> <score>'";
            return false;
        }
        out[unit] = score;
    }
    if (out.empty()) {
        err = "no unit in the rank file";
        return false;
    }
    return true;
}

size_t fp8_output_bytes(int64_t N, int64_t K) {
    if (N <= 0 || K <= 0)
        return 0;
    return (static_cast<size_t>(N) * static_cast<size_t>(K)) + sizeof(float);
}

size_t fp8_extra_bytes(int64_t N, int64_t K) {
    const size_t fp8 = fp8_output_bytes(N, K), nv = nvfp4_output_bytes(N, K);
    return fp8 > nv ? fp8 - nv : 0;
}

std::set<std::string> select_fp8_units(std::vector<Fp8Unit> units, size_t budget_bytes) {
    auto density = [](const Fp8Unit& u) {
        return u.extra_bytes ? u.score / static_cast<double>(u.extra_bytes) : 0.0;
    };
    std::stable_sort(units.begin(), units.end(),
                     [&](const Fp8Unit& a, const Fp8Unit& b) { return density(a) > density(b); });
    std::set<std::string> picked;
    size_t used = 0;
    for (const Fp8Unit& u : units) {
        if (u.score <= 0.0 || used + u.extra_bytes > budget_bytes)
            continue;
        used += u.extra_bytes;
        picked.insert(u.unit);
    }
    return picked;
}

std::string mixed_precision_layers_json(const std::vector<std::string>& nvfp4_modules,
                                        const std::vector<std::string>& fp8_modules,
                                        const std::string& indent) {
    std::string s = "{";
    bool first = true;
    auto add = [&](const std::string& m, const char* spec) {
        s += (first ? "\n" : ",\n") + indent + "  \"" + json_escape(m) + "\": " + spec;
        first = false;
    };
    for (const auto& m : nvfp4_modules)
        add(m, "{\"quant_algo\": \"NVFP4\", \"group_size\": 16}");
    for (const auto& m : fp8_modules)
        add(m, "{\"quant_algo\": \"FP8\"}");
    return s + "\n" + indent + "}";
}

std::vector<SafeTensorsOut> fp8_weight_outputs(const std::string& weight_name, int64_t N, int64_t K,
                                               const uint8_t* codes, const float* scale) {
    const std::string base = weight_name.substr(0, weight_name.size() - std::string(".weight").size());
    return {{weight_name, "F8_E4M3", {N, K}, codes, static_cast<size_t>(N) * static_cast<size_t>(K)},
            {base + ".weight_scale", "F32", {1}, scale, sizeof(float)}};
}

}  // namespace imp::quantize
