// Fill-in-the-middle (#2201): FIM token discovery and PSM prompt assembly (model/fim.h).
// Synthetic vocab, CPU lane. Real tokenizer files: tests/test_fim_tokenizer.cpp (test-e2e).

#include "model/fim.h"
#include "model/tokenizer.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace {

using imp::FimInput;
using imp::FimTokens;
using imp::Tokenizer;

// Single-byte vocab plus markers: every id below is literal, not computed by the code under test.
// ids: 0 <unk> 1 <s> 2 </s> 3 <|fim_prefix|> 4 <|fim_suffix|> 5 <|fim_middle|> 6 <PRE> 7 a 8 b 9 c
Tokenizer synthetic(bool with_fim_types) {
    Tokenizer t;
    const std::vector<std::string> v = {
        "<unk>", "<s>", "</s>", "<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>", "<PRE>", "a", "b", "c"};
    EXPECT_TRUE(t.load_vocab(v, std::vector<float>(v.size(), 0.0f), 1, 2));
    // CONTROL=3, NORMAL=1. <PRE> at id 6 stays NORMAL: a plain piece, not a marker.
    const int m = with_fim_types ? 3 : 1;
    t.load_token_types({3, 3, 3, m, m, m, 1, 1, 1, 1});
    t.set_add_bos(false);
    return t;
}

TEST(Fim, SyntheticDiscoversMarkerIds) {
    const Tokenizer t = synthetic(true);
    const FimTokens f = imp::find_fim_tokens(t);
    EXPECT_EQ(f.pre, 3);
    EXPECT_EQ(f.suf, 4);
    EXPECT_EQ(f.mid, 5);
    EXPECT_EQ(f.pad, -1);
    EXPECT_TRUE(f.supported());
}

TEST(Fim, SyntheticPsmOrderExact) {
    const Tokenizer t = synthetic(true);
    const FimTokens f = imp::find_fim_tokens(t);
    FimInput in;
    in.prefix = "a";
    in.suffix = "b";
    EXPECT_EQ(imp::build_fim_prompt(t, f, in), (std::vector<int32_t>{3, 7, 4, 8, 5}));
    in.prompt = "c";  // llama.cpp /infill: prompt continues the prefix
    EXPECT_EQ(imp::build_fim_prompt(t, f, in), (std::vector<int32_t>{3, 7, 9, 4, 8, 5}));
}

TEST(Fim, SyntheticNormalPiecesAreNotMarkers) {
    // Same texts, typed NORMAL: no FIM support, and "<PRE>" (NORMAL) never counts either.
    const FimTokens f = imp::find_fim_tokens(synthetic(false));
    EXPECT_FALSE(f.supported());
    EXPECT_EQ(f.pre, -1);
}

TEST(Fim, GgufMetadataIdsWin) {
    Tokenizer t = synthetic(false);
    t.set_fim_meta_id(imp::kFimPre, 7);
    t.set_fim_meta_id(imp::kFimSuf, 8);
    t.set_fim_meta_id(imp::kFimMid, 9);
    const FimTokens f = imp::find_fim_tokens(t);
    EXPECT_EQ(f.pre, 7);
    EXPECT_EQ(f.suf, 8);
    EXPECT_EQ(f.mid, 9);
}

TEST(Fim, EmptyTokenizerNotSupported) {
    const Tokenizer t;
    EXPECT_FALSE(imp::find_fim_tokens(t).supported());
}

}  // namespace
