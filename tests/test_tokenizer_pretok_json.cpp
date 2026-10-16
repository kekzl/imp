// tokenizer.json pre-tokenizer and BPE flags read at load, checked against HF tokenizers behaviour.
#include "model/tokenizer.h"

#include <gtest/gtest.h>

#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

namespace imp {
namespace {

using Chunks = std::vector<std::string>;

std::string write_temp_json(const std::string& body) {
    static int seq = 0;
    std::string path = "/tmp/imp_tok_pretok_" + std::to_string(::getpid()) + "_" + std::to_string(seq++) +
                       ".json";
    std::ofstream(path) << body;
    return path;
}

// MiniCPM5 tokenizer.json: Split \p{N}{1,3} then a cl100k-like Split with \p{N}+. HF: "12345" ->
// "123" "45"; the qwen2 family split it into five single digits (13180 vs 13909 corpus tokens).
TEST(TokenizerJsonPretok, DigitTriplesLeftThenDigitRunFamily) {
    EXPECT_EQ(digit_triples_left_split("ab12345 6"), (Chunks{"ab", "123", "45", " ", "6"}));
    EXPECT_EQ(digit_triples_left_split("\xD9\xA1\xD9\xA2x"),
              (Chunks{"\xD9\xA1\xD9\xA2", "x"}));  // Arabic-Indic
    EXPECT_EQ(digit_run_pre_tokenize("v12345 a"), (Chunks{"v", "12345", " a"}));
    std::string path = write_temp_json(R"JSON({
      "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
        {"type": "Split", "pattern": {"Regex": "\\p{N}{1,3}"}, "behavior": "Isolated", "invert": false},
        {"type": "Split", "pattern": {"Regex": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}+| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"},
         "behavior": "Isolated", "invert": false},
        {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": false}]},
      "model": {"type": "BPE", "vocab": {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4, "12": 5, "123": 6, "45": 7, "34": 8},
                "merges": ["3 4", "1 2", "12 3", "4 5"]}
    })JSON");
    Tokenizer tok;
    ASSERT_TRUE(tok.load(path));
    std::remove(path.c_str());
    tok.set_add_bos(false);
    EXPECT_EQ(tok.pre_tokenizer(), "digitrun");
    EXPECT_EQ(tok.encode("12345"), (std::vector<int32_t>{6, 7}));
}

// LFM2.5 tokenizer.json: BPE ignore_merges=true takes a pre-token found whole in the vocab as one id
// ("|---|---|---|\n" is one token there; merges alone built four).
TEST(TokenizerJsonPretok, IgnoreMergesTakesWholeVocabWord) {
    for (const bool ignore : {true, false}) {
        std::string path = write_temp_json(std::string(R"JSON({
          "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "use_regex": true},
          "model": {"type": "BPE", "ignore_merges": )JSON") +
                                           (ignore ? "true" : "false") + R"JSON(,
                    "vocab": {"a": 0, "b": 1, "c": 2, "ab": 3, "abc": 4}, "merges": ["a b"]}
        })JSON");
        Tokenizer tok;
        ASSERT_TRUE(tok.load(path));
        std::remove(path.c_str());
        tok.set_add_bos(false);
        EXPECT_EQ(tok.encode("abc"), ignore ? (std::vector<int32_t>{4}) : (std::vector<int32_t>{3, 2}))
            << ignore;
    }
}

}  // namespace
}  // namespace imp
