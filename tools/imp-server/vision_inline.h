#pragma once

// Where a picture sits in the prompt: the request parser leaves kVisionPartMark at each image or
// video part, the prompt builder swaps mark k for the rendered block of part k. Pictures in the
// first user turn instead moved every later one's tokens when a screenshot was added (prefix
// cache hit 0 of 17312 shared tokens) and showed the model a tool result's picture before the call.

#include "model/chat_template.h"

#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace imp::server {

// Control bytes no chat text carries; a prompt holding them by accident fails the count check.
inline constexpr std::string_view kVisionPartMark = "\x1f\x1dimp-vision-part\x1d\x1f";

inline size_t count_vision_marks(const std::vector<imp::ChatMessage>& msgs) {
    size_t n = 0;
    for (const auto& m : msgs)
        for (size_t at = m.content.find(kVisionPartMark); at != std::string::npos;
             at = m.content.find(kVisionPartMark, at + kVisionPartMark.size()))
            ++n;
    return n;
}

inline void strip_vision_marks(std::vector<imp::ChatMessage>& msgs) {
    for (auto& m : msgs)
        for (size_t at = m.content.find(kVisionPartMark); at != std::string::npos;
             at = m.content.find(kVisionPartMark, at))
            m.content.erase(at, kVisionPartMark.size());
}

// Mark k becomes blocks[k], in message order. False when the counts differ: the marks are
// stripped and the caller places the blocks itself.
inline bool place_vision_blocks(std::vector<imp::ChatMessage>& msgs, std::span<const std::string> blocks) {
    if (blocks.empty() || count_vision_marks(msgs) != blocks.size()) {
        strip_vision_marks(msgs);
        return false;
    }
    size_t k = 0;
    for (auto& m : msgs)
        for (size_t at = m.content.find(kVisionPartMark); at != std::string::npos;
             at = m.content.find(kVisionPartMark, at)) {
            m.content.replace(at, kVisionPartMark.size(), blocks[k]);
            at += blocks[k++].size();
        }
    return true;
}

}  // namespace imp::server
