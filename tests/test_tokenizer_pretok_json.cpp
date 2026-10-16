// tokenizer.json pre-tokenizer and BPE flags read at load, checked against HF tokenizers behaviour.
#include "model/json_util.h"
#include "model/tokenizer.h"
#include "model/tokenizer_regex.h"

#include <gtest/gtest.h>

#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <memory>
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
    auto left = RegexPretokenizer::from_regexes({"\\p{N}{1,3}"}, nullptr);
    ASSERT_NE(left, nullptr);
    EXPECT_EQ(left->split("ab12345 6"), (Chunks{"ab", "123", "45", " ", "6"}));
    EXPECT_EQ(left->split("\xD9\xA1\xD9\xA2x"), (Chunks{"\xD9\xA1\xD9\xA2", "x"}));  // Arabic-Indic
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
    EXPECT_EQ(tok.encode("12345"), (std::vector<int32_t>{6, 7}));
}

std::shared_ptr<const RegexPretokenizer> from_json_text(const std::string& json) {
    JsonParser parser(json);
    JValue v = parser.parse();
    std::string why;
    auto rp = parser.ok() ? RegexPretokenizer::from_json(v, &why) : nullptr;
    EXPECT_NE(rp, nullptr) << why;
    return rp;
}

// normalizer.rs split behaviors; truths: HF tokenizers pre_tokenize_str, Split(pattern, behavior).
TEST(TokenizerJsonPretok, SplitBehaviorsMatchHf) {
    auto split = [](const char* behavior, bool invert, const std::string& s) {
        auto rp = from_json_text(
            std::string(R"({"type": "Split", "pattern": {"String": "-"}, "behavior": ")") + behavior +
            R"(", "invert": )" + (invert ? "true" : "false") + "}");
        return rp ? rp->split(s) : Chunks{};
    };
    const std::string s = "a-b--c-";
    EXPECT_EQ(split("Removed", false, s), (Chunks{"a", "b", "c"}));
    EXPECT_EQ(split("Isolated", false, s), (Chunks{"a", "-", "b", "-", "-", "c", "-"}));
    EXPECT_EQ(split("MergedWithPrevious", false, s), (Chunks{"a-", "b-", "-", "c-"}));
    EXPECT_EQ(split("MergedWithNext", false, s), (Chunks{"a", "-b", "-", "-c", "-"}));
    EXPECT_EQ(split("Contiguous", false, s), (Chunks{"a", "-", "b", "--", "c", "-"}));
    EXPECT_EQ(split("Removed", true, s), (Chunks{"-", "-", "-", "-"}));  // each match its own piece
    auto empty = RegexPretokenizer::from_regexes({"(?=b)"}, nullptr);    // empty matches split too
    ASSERT_NE(empty, nullptr);
    EXPECT_EQ(empty->split("abab"), (Chunks{"a", "ba", "b"}));
}

// Digits, Punctuation, Whitespace, WhitespaceSplit and a Sequence of them, as HF pre-tokenizes.
TEST(TokenizerJsonPretok, CharPreTokenizersMatchHf) {
    auto seq = from_json_text(R"({"type": "Sequence", "pretokenizers": [
        {"type": "WhitespaceSplit"}, {"type": "Punctuation", "behavior": "Isolated"},
        {"type": "Digits", "individual_digits": true}]})");
    ASSERT_NE(seq, nullptr);
    EXPECT_EQ(seq->split("Hi,  you 42!$"), (Chunks{"Hi", ",", "you", "4", "2", "!", "$"}));
    auto runs = from_json_text(R"({"type": "Digits", "individual_digits": false})");
    ASSERT_NE(runs, nullptr);
    EXPECT_EQ(runs->split("a123b45"), (Chunks{"a", "123", "b", "45"}));
    auto ws = from_json_text(R"({"type": "Whitespace"})");
    ASSERT_NE(ws, nullptr);
    EXPECT_EQ(ws->split("Hey friend!  How?"), (Chunks{"Hey", "friend", "!", "How", "?"}));
    EXPECT_EQ(from_json_text(R"({"type": "ByteLevel", "use_regex": false})")->n_steps(), 0U);
}

// Oniguruma reads \s with U+0085 (NEL) and \w with every N and M; PCRE2 UCP does not.
TEST(TokenizerJsonPretok, OnigClassesSurviveTheRewrite) {
    auto ws = RegexPretokenizer::from_regexes({"\\s+"}, nullptr);
    ASSERT_NE(ws, nullptr);
    EXPECT_EQ(ws->split("a\xC2\x85"
                        "b"),
              (Chunks{"a", "\xC2\x85", "b"}));
    auto word = RegexPretokenizer::from_regexes({"\\w+"}, nullptr);
    ASSERT_NE(word, nullptr);
    EXPECT_EQ(word->split("x\xE0\xA4\x83\xC2\xBD-"), (Chunks{"x\xE0\xA4\x83\xC2\xBD", "-"}));  // Mc, No
    auto hex = RegexPretokenizer::from_regexes({"\\h+"}, nullptr);  // Ruby \h: hex digit
    ASSERT_NE(hex, nullptr);
    EXPECT_EQ(hex->split("zz0aFgg"), (Chunks{"zz", "0aF", "gg"}));
    auto boundary = RegexPretokenizer::from_regexes({"\\b"}, nullptr);  // Me (U+20DD) is \w
    ASSERT_NE(boundary, nullptr);
    EXPECT_EQ(boundary->split("a\xE2\x83\x9D b"), (Chunks{"a\xE2\x83\x9D", " ", "b"}));
    EXPECT_EQ(onig_to_pcre2("[^\\s\\w]"), "[^\\s\\x{85}\\p{L}\\p{M}\\p{N}\\p{Pc}]");
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

// DeepSeek-V2-shaped Split sequence. Truths: HF tokenizers 0.23.2 pre_tokenize_str on
// Sequence([Split(r, "isolated") for r in the steps] + [Digits(individual_digits=True)]).
TEST(PreTokenizeSplitSequenceTest, MatchesHfOnDeepSeekShapedSteps) {
    // Digits(individual_digits=True) is the single-char \p{N} split.
    auto seq = RegexPretokenizer::from_regexes({"[\r\n]", "\\s?[A-Za-z\xc3\x80-\xc3\xbf]+",
                                                "\\s?[!-/:-~\xe3\x80\x80-\xe3\x80\x82]+", "\\s+$",
                                                "[\xe4\xb8\x80-\xe9\xbe\xa5]+", "\\p{N}"},
                                               nullptr);
    ASSERT_NE(seq, nullptr);
    auto split = [&](const std::string& s) { return seq->split(s); };
    EXPECT_EQ(split("  two leading"), (Chunks{" ", " two", " leading"}));
    EXPECT_EQ(split("Hello, world!\n"), (Chunks{"Hello", ",", " world", "!", "\n"}));
    EXPECT_EQ(split("x = 12345;  "), (Chunks{"x", " =", " ", "1", "2", "3", "4", "5", ";", "  "}));
    EXPECT_EQ(split("\xe4\xb8\xad\xe6\x96\x87"
                    "abc\xe3\x80\x82"),
              (Chunks{"\xe4\xb8\xad\xe6\x96\x87", "abc", "\xe3\x80\x82"}));
    EXPECT_EQ(split("caf\xc3\xa9  \xc3\xa9t\xc3\xa9  "),
              (Chunks{"caf", "\xc3\xa9", " ", " \xc3\xa9", "t", "\xc3\xa9", "  "}));
    EXPECT_EQ(split("a\xe3\x80\x80"
                    "b"),
              (Chunks{"a",
                      "\xe3\x80\x80"
                      "b"}));  // U+3000 is the \s of \s?
    EXPECT_EQ(split("ab  \n  cd"), (Chunks{"ab", "  ", "\n", " ", " cd"}));
    EXPECT_EQ(split(" \xc2\xb2x"), (Chunks{" ", "\xc2\xb2", "x"}));  // U+00B2 is a Digits piece
}

// Onig (Ruby syntax) $ also matches before any "\n", not only at the end.
TEST(PreTokenizeSplitSequenceTest, TrailingSpaceDollarIsEndOfLine) {
    auto seq = RegexPretokenizer::from_regexes({"\\s+$"}, nullptr);
    ASSERT_NE(seq, nullptr);
    EXPECT_EQ(seq->split("ab  \ncd"), (Chunks{"ab", "  ", "\ncd"}));
    EXPECT_EQ(seq->split("ab \t\n\ncd  "), (Chunks{"ab", " \t\n", "\ncd", "  "}));
    EXPECT_EQ(seq->split("a  b"), (Chunks{"a  b"}));
}

// Oniguruma forms PCRE2 has no spelling for are refused, never compiled to something else.
TEST(PreTokenizeSplitSequenceTest, RefusesRegexItCannotMatchExactly) {
    EXPECT_EQ(onig_to_pcre2("[a[b]]"), "");       // nested class
    EXPECT_EQ(onig_to_pcre2("[a-z&&[^b]]"), "");  // intersection
    EXPECT_EQ(onig_to_pcre2("[\\S]"), "");        // negated shorthand inside a class
    EXPECT_EQ(RegexPretokenizer::from_regexes({"[a[b]]"}, nullptr), nullptr);
    EXPECT_EQ(RegexPretokenizer::from_regexes({"(unclosed"}, nullptr), nullptr);
    EXPECT_EQ(RegexPretokenizer::from_regexes({}, nullptr), nullptr);
}

}  // namespace
}  // namespace imp
