#pragma once

#include <nlohmann/json.hpp>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <string>
#include <vector>

// /v1/completions "prompt": a string, a token-id list, or a one-element list of either (OpenAI
// spec; lm-eval local-completions sends token ids). Exactly one of `text` / `ids` is filled.
// Returns "" or the 400 message: a batch of prompts is refused like n>1.
inline std::string parse_completion_prompt(const nlohmann::json& body, std::string& text, std::vector<int32_t>& ids) {
    using nlohmann::json;
    const auto is_ids = [](const json& a) {
        return a.is_array() && !a.empty() && std::all_of(a.begin(), a.end(), [](const json& t) {
                   return t.is_number_integer() && t.get<int64_t>() >= 0 && t.get<int64_t>() <= INT32_MAX;
               });
    };
    const json* p = body.contains("prompt") ? &body["prompt"] : nullptr;
    if (p && p->is_array() && p->size() == 1 && ((*p)[0].is_string() || is_ids((*p)[0])))
        p = &(*p)[0];
    if (p && p->is_string() && !p->get<std::string>().empty()) {
        text = p->get<std::string>();
        return "";
    }
    if (p && is_ids(*p)) {
        for (const auto& t : *p)
            ids.push_back(static_cast<int32_t>(t.get<int64_t>()));
        return "";
    }
    if (p && p->is_array() && p->size() > 1 &&
        std::all_of(p->begin(), p->end(), [&](const json& e) { return e.is_string() || is_ids(e); }))
        return "a batch of prompts is not supported on /v1/completions; send one prompt per call";
    return "\"prompt\" is required: a non-empty string, a list of token ids, or a one-element list of either";
}
