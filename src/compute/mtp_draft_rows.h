// LM-head rows an MTP draft pass scores (speculative.mtp_draft_vocab, speculative.mtp_nvfp4_head).
#pragma once

#include <cstdint>

namespace imp {

// Rows the NVFP4 draft copy holds: head rows [0, draft_vocab), every row at draft_vocab <= 0 or >= vocab.
[[nodiscard]] constexpr int mtp_draft_copy_rows(int vocab, int draft_vocab) {
    return draft_vocab > 0 && draft_vocab < vocab ? draft_vocab : vocab;
}

// Rows one draft pass scores. copy_rows = rows the NVFP4 draft copy holds (0 = no copy), fp8 = the pass
// reads the FP8 head. A pass never reads past the copy; draft_vocab caps a copy or the FP8 head only.
[[nodiscard]] constexpr int mtp_draft_pass_rows(int vocab, int64_t copy_rows, bool fp8, int draft_vocab) {
    const bool copy = copy_rows > 0;
    const int rows = copy && copy_rows < vocab ? static_cast<int>(copy_rows) : vocab;
    return (copy || fp8) && draft_vocab > 0 && draft_vocab < rows ? draft_vocab : rows;
}

}  // namespace imp
