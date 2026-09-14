#pragma once
// reasoning_effort choices as a chat template's Jinja source spells them: the literal list after
// `in` (Qwen3.8: ('xhigh', 'medium', 'low')) and default('x') or `set reasoning_effort = "x"` (gpt-oss).
// Both outputs stay empty when the source names neither; nothing is inferred from the model family.

#include <regex>
#include <string>
#include <vector>

namespace imp {

inline void parse_reasoning_effort_spec(const std::string& jinja_src, std::vector<std::string>& values,
                                        std::string& default_value) {
    values.clear();
    default_value.clear();
    std::smatch m;
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

}  // namespace imp
