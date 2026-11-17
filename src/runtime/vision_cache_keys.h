#pragma once

// Prefix-cache key tokens for a prompt with images: every placeholder (one shared id) is replaced
// by a value derived from the picture it stands for, as vLLM keys each block on the images in it.
// Text before the first image keys like plain text; a new screenshot invalidates only its blocks on.

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace imp {

// Key of placeholder i of a picture with content hash h. Negative: never equal to a token id.
constexpr int32_t vision_key_token(size_t h, size_t i) {
    uint64_t z = static_cast<uint64_t>(h) + (static_cast<uint64_t>(i) + 1) * 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z ^= z >> 31;
    return static_cast<int32_t>(static_cast<uint32_t>(z) | 0x80000000u);
}

// Key tokens for `tokens` whose placeholders are `pad_id`. Maximal runs of pad_id map to
// part_hashes in prompt order when the counts match; otherwise every run keys on whole_hash
// (the text before the first picture still matches). Empty when the prompt has no placeholder.
inline std::vector<int32_t> vision_cache_key_tokens(std::span<const int32_t> tokens, int32_t pad_id,
                                                    std::span<const size_t> part_hashes, size_t whole_hash) {
    size_t runs = 0;
    for (size_t i = 0; i < tokens.size(); ++i)
        runs += tokens[i] == pad_id && (i == 0 || tokens[i - 1] != pad_id) ? 1 : 0;
    if (runs == 0 || pad_id < 0)
        return {};
    const bool per_part = runs == part_hashes.size();
    std::vector<int32_t> keys(tokens.begin(), tokens.end());
    size_t next_run = 0, run = 0, pos = 0;
    for (size_t i = 0; i < keys.size(); ++i) {
        if (tokens[i] != pad_id)
            continue;
        if (i == 0 || tokens[i - 1] != pad_id) {
            run = next_run++;
            pos = 0;
        }
        keys[i] = vision_key_token(per_part ? part_hashes[run] : whole_hash, pos++);
    }
    return keys;
}

}  // namespace imp
