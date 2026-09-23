#include "logit_bias.h"

#include "utils.h"

#include <charconv>
#include <cmath>

std::string parse_logit_bias(const nlohmann::json& body, int cap,
                             std::vector<std::pair<int32_t, float>>& out) {
    if (!body.contains("logit_bias") || body["logit_bias"].is_null())
        return "";
    const nlohmann::json& lb = body["logit_bias"];
    if (!lb.is_object())
        return "\"logit_bias\" must be an object mapping token ids to biases";
    // Every entry costs a blocking device-to-host copy per decode step (#1617): refuse, not truncate.
    if (cap > 0 && static_cast<int>(lb.size()) > cap)
        return "\"logit_bias\" has " + std::to_string(lb.size()) + " entries, above the server limit of " +
               std::to_string(cap) + " (--max-logit-bias)";
    for (const auto& [key, val] : lb.items()) {
        int32_t id = -1;
        const char* end = key.data() + key.size();
        const auto [p, ec] = std::from_chars(key.data(), end, id);
        if (key.empty() || key[0] == '-' || ec != std::errc() || p != end)
            return "\"logit_bias\" key '" + sanitize_for_echo(key, 64) +
                   "' is not a token id (a non-negative integer)";
        // is_number() excludes booleans; NaN fails the range test.
        const double b = val.is_number() ? val.get<double>() : std::nan("");
        if (!(b >= -100.0 && b <= 100.0))
            return "\"logit_bias\" value for token " + key + " must be a number in [-100, 100]";
        out.emplace_back(id, static_cast<float>(b));
    }
    return "";
}

std::string logit_bias_vocab_error(const std::vector<std::pair<int32_t, float>>& bias, int vocab) {
    for (const auto& [id, b] : bias)
        if (id >= vocab)
            return "\"logit_bias\" token id " + std::to_string(id) + " is outside the vocabulary (" +
                   std::to_string(vocab) + " tokens)";
    return "";
}
