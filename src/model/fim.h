#pragma once

// Fill-in-the-middle (#2201): FIM token discovery and PSM prompt assembly for /infill and the
// /v1/completions `suffix` field. Ids come from GGUF metadata or the vocab, never a model table.

#include <cstdint>
#include <string>
#include <vector>

namespace imp {

class Tokenizer;

// Index into Tokenizer::fim_meta_id().
enum FimRole : int { kFimPre = 0, kFimSuf, kFimMid, kFimPad, kFimRep, kFimSep };

struct FimTokens {
    int32_t pre = -1;
    int32_t suf = -1;
    int32_t mid = -1;
    int32_t pad = -1;  // optional
    int32_t rep = -1;  // optional: repo-name marker (Qwen2.5-Coder <|repo_name|>)
    int32_t sep = -1;  // optional: file separator (<|file_sep|>)
    bool supported() const { return pre >= 0 && suf >= 0 && mid >= 0; }
};

// GGUF metadata ids first, then a lookup of the known FIM token texts; a text hit counts only
// when the vocab marks it special/added (a plain BPE piece spelling "<PRE>" is not a FIM token).
FimTokens find_fim_tokens(const Tokenizer& tok);

struct FimExtraChunk {
    std::string filename;
    std::string text;
};

struct FimInput {
    std::string prefix;
    std::string suffix;
    std::string prompt;  // llama.cpp /infill: appended to the prefix, before the suffix marker
    std::vector<FimExtraChunk> extra;
};

// PSM: [BOS] PRE prefix+prompt SUF suffix MID. With `extra`, llama.cpp repo layout in front:
// [REP "myproject\n"] ([SEP name"\n"] text)* [SEP "filename\n"], REP/SEP only when the vocab has
// them, else a text divider per chunk. Precondition: fim.supported().
std::vector<int32_t> build_fim_prompt(const Tokenizer& tok, const FimTokens& fim, const FimInput& in);

// Texts of the FIM marker tokens present in the vocab: stop strings for a FIM generation.
std::vector<std::string> fim_stop_texts(const Tokenizer& tok, const FimTokens& fim);

}  // namespace imp
