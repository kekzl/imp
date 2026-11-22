#pragma once
// reasoning_effort choices as a chat template's Jinja source spells them: the literal list after
// `in` (Qwen3.8: ('xhigh', 'medium', 'low')) and default('x') or `set reasoning_effort = "x"` (gpt-oss),
// or a name -> number map (Inkling2: effort_map = {"none": 0.0, ..., "max": 0.99}, default `else 0.9`).
// All outputs stay empty when the source names none; nothing is inferred from the model family.

#include <algorithm>
#include <charconv>
#include <cstdio>
#include <regex>
#include <string>
#include <vector>

namespace imp {

// s as a number, when all of it is one ("0.5", "0", not "0.5x" or "").
[[nodiscard]] inline bool reasoning_effort_number(const std::string& s, double& out) {
    const char* end = s.data() + s.size();
    const auto r = std::from_chars(s.data(), end, out);
    return !s.empty() && r.ec == std::errc() && r.ptr == end;
}

// levels (optional): the numbers of a name -> number map, parallel to values; empty for a plain list.
inline void parse_reasoning_effort_spec(const std::string& jinja_src, std::vector<std::string>& values,
                                        std::string& default_value, std::vector<double>* levels = nullptr) {
    values.clear();
    default_value.clear();
    if (levels)
        levels->clear();
    std::smatch m;
    static const std::regex map_re(R"(effort_map\s*=\s*\{([^}]*)\})");
    if (std::regex_search(jinja_src, m, map_re)) {
        static const std::regex pair_re(R"(['"]([^'"]+)['"]\s*:\s*([0-9]*\.?[0-9]+))");
        std::vector<double> nums;
        const std::string items = m[1].str();
        for (std::sregex_iterator it(items.begin(), items.end(), pair_re), end; it != end; ++it) {
            double v = 0.0;
            if (!reasoning_effort_number((*it)[2].str(), v))
                continue;
            values.push_back((*it)[1].str());
            nums.push_back(v);
        }
        // The default is a number: named by the first level that carries it.
        static const std::regex else_re(R"(reasoning_effort\s+is\s+not\s+none\s+else\s+([0-9]*\.?[0-9]+))");
        double d = 0.0;
        if (std::regex_search(jinja_src, m, else_re) && reasoning_effort_number(m[1].str(), d))
            if (const auto at = std::find(nums.begin(), nums.end(), d); at != nums.end())
                default_value = values[static_cast<size_t>(at - nums.begin())];
        if (levels)
            *levels = std::move(nums);
        if (!values.empty())
            return;
    }
    static const std::regex list_re(R"(reasoning_effort\s+(?:not\s+)?in\s*[(\[]([^)\]]*)[)\]])");
    if (std::regex_search(jinja_src, m, list_re)) {
        static const std::regex item_re(R"(['"]([^'"]+)['"])");
        const std::string items = m[1].str();
        for (std::sregex_iterator it(items.begin(), items.end(), item_re), end; it != end; ++it)
            values.push_back((*it)[1].str());
    }
    static const std::regex default_re(
        R"(reasoning_effort\s*\|\s*default\(\s*['"]([^'"]+)['"]|set\s+reasoning_effort\s*=\s*['"]([^'"]+)['"])");
    if (std::regex_search(jinja_src, m, default_re))
        default_value = m[1].matched ? m[1].str() : m[2].str();
}

// A request value against a name -> number map: "" when it is a listed name or a number inside the
// map's range, else the reason. A template without a map takes any value (its own branch decides).
inline std::string reasoning_effort_map_error(const std::string& value,
                                              const std::vector<std::string>& values,
                                              const std::vector<double>& levels) {
    if (value.empty() || levels.empty() || std::find(values.begin(), values.end(), value) != values.end())
        return "";
    const auto [lo, hi] = std::minmax_element(levels.begin(), levels.end());
    double v = 0.0;
    if (reasoning_effort_number(value, v) && v >= *lo && v <= *hi)
        return "";
    std::string names;
    for (const auto& n : values)
        names += (names.empty() ? "" : ", ") + n;
    char range[64];
    std::snprintf(range, sizeof(range), "%g to %g", *lo, *hi);
    return "reasoning_effort '" + value + "' is not one of " + names + " or a number from " + range;
}

}  // namespace imp
