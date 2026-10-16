#pragma once
// tokenizer.json pre-tokenizer run step by step on PCRE2, as HF tokenizers runs it on Oniguruma:
// each step re-splits every piece of the previous one. Covers Split (Regex/String, all five
// behaviors, invert), ByteLevel (use_regex), Digits, Punctuation, Whitespace, WhitespaceSplit.

#include <memory>
#include <string>
#include <vector>

namespace imp {

struct JValue;

class RegexPretokenizer {
public:
    enum class Behavior { kRemoved, kIsolated, kMergedWithPrevious, kMergedWithNext, kContiguous };
    struct Step;

    // nullptr when a step is not covered (Metaspace, unknown type) or a regex does not compile;
    // *why names it.
    static std::shared_ptr<const RegexPretokenizer> from_json(const JValue& pre_tok, std::string* why);
    // One Split(Isolated) per regex, in order: GGUF tokenizer.ggml.pre names.
    static std::shared_ptr<const RegexPretokenizer> from_regexes(const std::vector<std::string>& regexes,
                                                                 std::string* why);

    RegexPretokenizer();
    ~RegexPretokenizer();
    RegexPretokenizer(const RegexPretokenizer&) = delete;
    RegexPretokenizer& operator=(const RegexPretokenizer&) = delete;

    std::vector<std::string> split(const std::string& text) const;
    size_t n_steps() const;

private:
    std::vector<Step> steps_;
};

// Oniguruma (Ruby syntax) regex text as PCRE2 reads it the same way: \h -> hex digit, \s/\S with
// U+0085, \w/\W/\b/\B over L, M, N, Pc. Exposed for unit tests.
std::string onig_to_pcre2(const std::string& rx);

// The regex a GGUF tokenizer.ggml.pre name stands for, "" when the name is unknown.
std::vector<std::string> gguf_pre_regexes(const std::string& name);

// GPT-2 ByteLevel(use_regex=true) regex.
extern const char* const kGpt2PretokRegex;

}  // namespace imp
