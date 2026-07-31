#include "model/image_placeholders.h"

#include <algorithm>

namespace imp {

bool expand_image_placeholders(std::vector<int32_t>& tokens, int32_t pad_id, const std::vector<int>& counts,
                               std::string& err) {
    size_t found = 0;
    for (int32_t t : tokens)
        if (t == pad_id)
            ++found;
    if (found != counts.size()) {
        err = "prompt holds " + std::to_string(found) + " image placeholder(s) but " +
              std::to_string(counts.size()) + " image(s) were encoded";
        return false;
    }
    for (size_t k = 0; k < counts.size(); ++k) {
        if (counts[k] <= 0) {
            err = "image " + std::to_string(k) + " produced no tokens";
            return false;
        }
    }
    if (found == 0)
        return true;

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
    return true;
}

size_t image_content_hash(const uint8_t* data, size_t len) {
    size_t h = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < len; ++i) {
        h ^= data[i];
        h *= 0x100000001b3ULL;
    }
    return h ? h : 1;  // 0 is the cache's "no image" sentinel
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
