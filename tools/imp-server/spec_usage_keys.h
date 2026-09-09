#pragma once

// The vendor-prefixed speculation keys that ride in `usage`, in ONE place
// because three surfaces carry them and only one of them writes them.
//
// OpenAI chat writes them into `usage.completion_tokens_details`
// (add_spec_usage_, handlers_internal.h). The Anthropic shim lifts them into
// top-level `usage`, and the Responses shim into `usage.output_tokens_details`,
// and both did it through a hand-copied list of three names. So the moment a
// fourth key appeared, `/v1/chat/completions` reported it and the other two
// dialects silently did not - which is how `imp_spec_emitted` and the decline
// reason reached one surface out of three while the docs claimed all of them.
//
// A CPU test (tests/test_spec_usage.cpp) asserts that this table is exactly
// the set of keys add_spec_usage_ writes, so adding a key without adding it
// here is red rather than invisible.

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
