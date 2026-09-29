// Fill-in-the-middle (#2201) against real tokenizer files, no weights/GPU. test-e2e, outside the
// unit lane: IMP_TEST_FIM_TOKENIZER = Qwen3-Coder-30B-A3B-Instruct dir, IMP_TEST_NOFIM_TOKENIZER =
// a dir without FIM tokens (Gemma-4-26B-A4B-it). A set-but-wrong path fails.

#include "model/fim.h"
#include "model/tokenizer.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

namespace {

using imp::FimInput;
using imp::FimTokens;
using imp::Tokenizer;

bool load_from_env(const char* env_name, Tokenizer& tok) {
    const char* env = std::getenv(env_name);
    if (!env)
        return false;
    std::filesystem::path p(env);
    if (std::filesystem::is_directory(p))
        p /= "tokenizer.json";
    EXPECT_TRUE(std::filesystem::exists(p)) << env_name << "=" << env << ": " << p << " missing";
    EXPECT_TRUE(tok.load(p.string())) << "Tokenizer::load failed for " << p;
    return true;
}

// Expected ids: HF `tokenizers` Tokenizer.from_file(<Qwen3-Coder-30B-A3B-Instruct>/tokenizer.json)
// .encode(<marker-joined string>, add_special_tokens=False).ids, run 2026-09-29.
const std::string kPrefix = "def add(a, b):\n    ";
const std::string kSuffix = "\n\nprint(add(1, 2))\n";

TEST(FimRealTokenizer, QwenCoderIdsAndPsmPrompt) {
    Tokenizer tok;
    if (!load_from_env("IMP_TEST_FIM_TOKENIZER", tok))
        GTEST_SKIP() << "Set IMP_TEST_FIM_TOKENIZER to a Qwen3-Coder model dir";
    // tokenizer_config.json add_bos_token=false: safetensors_loader applies it, Tokenizer::load does not.
    tok.set_add_bos(false);
    const FimTokens f = imp::find_fim_tokens(tok);
    std::printf("FIM ids: pre=%d suf=%d mid=%d pad=%d rep=%d sep=%d\n", f.pre, f.suf, f.mid, f.pad, f.rep,
                f.sep);
    EXPECT_EQ(f.pre, 151659);
    EXPECT_EQ(f.suf, 151661);
    EXPECT_EQ(f.mid, 151660);
    EXPECT_EQ(f.pad, 151662);
    EXPECT_EQ(f.rep, 151663);
    EXPECT_EQ(f.sep, 151664);
    ASSERT_TRUE(f.supported());

    FimInput in;
    in.prefix = kPrefix;
    in.suffix = kSuffix;
    EXPECT_EQ(imp::build_fim_prompt(tok, f, in),
              (std::vector<int32_t>{151659, 750, 912, 2877, 11, 293, 982, 257, 151661, 271, 1350, 25906, 7,
                                    16, 11, 220, 17, 1171, 151660}));

    in.extra = {{"util.py", "X = 1\n"}};
    EXPECT_EQ(imp::build_fim_prompt(tok, f, in),
              (std::vector<int32_t>{151663, 2408, 4987, 198, 151664, 1314, 7197,   198,    55,
                                    284,    220,  16,   198, 151664, 8404, 198,    151659, 750,
                                    912,    2877, 11,   293, 982,    257,  151661, 271,    1350,
                                    25906,  7,    16,   11,  220,    17,   1171,   151660}));

    const auto stops = imp::fim_stop_texts(tok, f);
    EXPECT_NE(std::find(stops.begin(), stops.end(), "<|file_sep|>"), stops.end());
}

TEST(FimRealTokenizer, NonFimTokenizerNotSupported) {
    Tokenizer tok;
    if (!load_from_env("IMP_TEST_NOFIM_TOKENIZER", tok))
        GTEST_SKIP() << "Set IMP_TEST_NOFIM_TOKENIZER to a model dir without FIM tokens";
    const FimTokens f = imp::find_fim_tokens(tok);
    std::printf("non-FIM ids: pre=%d suf=%d mid=%d\n", f.pre, f.suf, f.mid);
    EXPECT_FALSE(f.supported());
}

}  // namespace
