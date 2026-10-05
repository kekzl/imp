#pragma once

#include "runtime/suffix_draft.h"

#include <cstddef>
#include <string>
#include <unordered_map>

namespace imp {

class Tokenizer;

// speculative.ngram_corpus (#2421): text files under `path` (a file or a directory, recursive)
// tokenized into `idx`, one negative separator token between files. Files with a NUL byte are
// skipped as binary; reading stops at `max_bytes`. Returns the files indexed.
int load_ngram_corpus(const std::string& path, const Tokenizer& tok, SuffixDraftIndex& idx, size_t max_bytes);

// Engine init: builds the corpus index into `indexes` when `path` is set (64 MiB cap).
void install_ngram_corpus(const std::string& path, const Tokenizer* tok, int min_match, int max_match,
                          std::unordered_map<int, SuffixDraftIndex>& indexes);

}  // namespace imp
