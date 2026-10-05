#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace imp {

// Jump-ahead draft from the request's own output (roadmap row 46): the most recent run of
// `out` tokens that starts at a token boundary and spells a prefix of `text`, so the draft
// follows the model's own split of a repeated key instead of the canonical encode (which
// exits at 1/3 on `_the` vs `_`). Looks back at most `window` tokens; empty when no
// occurrence covers at least `min_tokens` tokens.
[[nodiscard]] std::vector<int32_t> jump_draft_from_history(const std::vector<int32_t>& out,
                                                           const std::function<std::string(int32_t)>& piece,
                                                           const std::string& text, int window,
                                                           int min_tokens);

}  // namespace imp
