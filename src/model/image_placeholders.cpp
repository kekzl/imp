#include "model/image_placeholders.h"

#include <algorithm>

namespace imp {

std::expected<void, std::string> expand_image_placeholders(std::vector<int32_t>& tokens, int32_t pad_id,
                                                           const std::vector<int>& counts) {
    size_t found = 0;
    for (int32_t t : tokens)
        if (t == pad_id)
            ++found;
    if (found != counts.size())
        return std::unexpected("prompt holds " + std::to_string(found) + " image placeholder(s) but " +
                               std::to_string(counts.size()) + " image(s) were encoded");
    for (size_t k = 0; k < counts.size(); ++k)
        if (counts[k] <= 0)
            return std::unexpected("image " + std::to_string(k) + " produced no tokens");
    if (found == 0)
        return {};

    size_t total = tokens.size();
    for (int c : counts)
        total += static_cast<size_t>(c) - 1;

    std::vector<int32_t> out;
    out.reserve(total);
    size_t k = 0;
    for (int32_t t : tokens) {
        if (t != pad_id) {
            out.push_back(t);
            continue;
        }
        out.insert(out.end(), static_cast<size_t>(counts[k]), pad_id);
        ++k;
    }
    tokens = std::move(out);
    return {};
}

size_t image_content_hash(std::span<const uint8_t> data) {
    size_t h = 0xcbf29ce484222325ULL;
    for (const uint8_t b : data) {
        h ^= b;
        h *= 0x100000001b3ULL;
    }
    return h ? h : 1;  // 0 is the cache's "no image" sentinel
}

size_t combine_image_hash(size_t running, size_t next) {
    if (running == 0)
        return next;
    // FNV-style mix, so swapping two images changes the result. Never returns
    // 0: that value means "no image" to the prefix cache.
    const size_t h = (running * 0x100000001b3ULL) ^ next;
    return h ? h : 1;
}

int image_tokens_before(const std::vector<int32_t>& tokens, int32_t pad_id, int upto) {
    if (upto <= 0)
        return 0;
    const size_t end = std::min(static_cast<size_t>(upto), tokens.size());
    int n = 0;
    for (size_t i = 0; i < end; ++i)
        n += (tokens[i] == pad_id);
    return n;
}

}  // namespace imp
