#pragma once

// Vendor-prefixed speculation usage keys, defined in ONE place: OpenAI writes them
// (add_spec_usage_), Anthropic/Responses shims lift them via this same list instead of their own
// hand-copied ones - a 4th key used to reach only 1 of 3 dialects silently.
// tests/test_spec_usage.cpp asserts this table matches exactly what add_spec_usage_ writes.

#include <nlohmann/json.hpp>

namespace imp_server {

inline constexpr const char* kSpecUsageKeys[] = {
    "imp_spec_drafted",  "imp_spec_accepted",         "imp_spec_emitted",
    "imp_spec_verify_steps", "imp_spec_declined", "imp_spec_declined_detail",
};

// Copy whichever of the keys `from` carries into `to`. Absent keys stay absent:
// the block only appears when speculation had something to say.
inline void copy_spec_usage_keys(const nlohmann::json& from, nlohmann::json& to) {
    if (!from.is_object())
        return;
    for (const char* k : kSpecUsageKeys)
        if (from.contains(k))
            to[k] = from[k];
}

}  // namespace imp_server
