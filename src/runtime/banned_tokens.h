#pragma once
// Engine-wide sampler ban list (Engine::build_banned_token_list): template delimiters and CONTROL
// tokens, minus the markers a model is trained to emit. One device copy serves every sampling site
// (graphs, batched decode, spec verify), so the list cannot differ per request.
#include <cstdint>
#include <vector>

namespace imp {

class Tokenizer;

// Ids never sampled: `delimiters` (start-of-turn / header ids, negative = absent) and every CONTROL
// token (heuristic `<|..|>` scan without token types), except eos, `stop_ids` and the trained
// markers. Sorted, unique.
[[nodiscard]] std::vector<int32_t> build_banned_token_ids(const Tokenizer* tok,
                                                          const std::vector<int32_t>& stop_ids,
                                                          const std::vector<int32_t>& delimiters);

}  // namespace imp
