#include "model/tokenizer.h"
#include <gtest/gtest.h>

namespace imp {
namespace {

// ---- Helpers to build synthetic vocabularies ----

// Build a minimal SPM tokenizer with proper BPE merge chain and byte fallback.
// SPM BPE merges pairs of adjacent symbols bottom-up, so the vocab must contain
// all intermediate merge tokens (not just final words).
//
// Merge chains:
//   H + e → He → He + llo → Hello → ▁ + Hello → ▁Hello
//   l + l → ll → ll + o → llo
//   o + r → or → w + or → wor → wor + ld → world → ▁ + world → ▁world
//   l + d → ld
//   t + h → th → ▁ + th → ▁th → ▁th + e → ▁the
//   c + a → ca → ▁ + ca → ▁ca → ▁ca + t → ▁cat
static Tokenizer make_spm_tokenizer() {
    // Token indices:
    //  0: <unk>  1: <s>  2: </s>  3: ▁  (U+2581 space symbol)
    //  4-14: individual ASCII chars (H, e, l, o, w, r, d, t, h, c, a)
    //  15-20: pair merges (He, ll, or, ld, th, ca)
    //  21-22: triple merges (llo, wor)
    //  23-24: ▁-prefixed pairs (▁th, ▁ca)
    //  25-26: full words (Hello, world)
    //  27-28: ▁-prefixed words (▁the, ▁cat)
    //  29-30: ▁-prefixed full words (▁Hello, ▁world)
    //  31-286: byte fallback <0x00>..<0xFF>
    std::vector<std::string> tokens = {
        "<unk>", "<s>", "</s>",
        "\xe2\x96\x81",  // ▁ (bare space)
        "H", "e", "l", "o", "w", "r", "d", "t", "h", "c", "a",  // 4-14
        "He", "ll", "or", "ld", "th", "ca",                       // 15-20
        "llo", "wor",                                               // 21-22
        "\xe2\x96\x81" "th", "\xe2\x96\x81" "ca",                 // 23-24 (▁th, ▁ca)
        "Hello", "world",                                           // 25-26
        "\xe2\x96\x81" "the", "\xe2\x96\x81" "cat",               // 27-28 (▁the, ▁cat)
        "\xe2\x96\x81" "Hello", "\xe2\x96\x81" "world",           // 29-30 (▁Hello, ▁world)
    };
    // Scores: higher (less negative) = merge first. The BPE algorithm picks the
    // highest-scoring pair at each step, so longer tokens need higher scores.
    std::vector<float> scores = {
        0.0f, 0.0f, 0.0f,
        -8.0f,  // ▁
        -8.0f, -8.0f, -8.0f, -8.0f, -8.0f, -8.0f, -8.0f, -8.0f, -8.0f, -8.0f, -8.0f, // chars
        -6.0f, -6.0f, -6.0f, -6.0f, -6.0f, -6.0f, // pair merges
        -5.0f, -5.0f,  // triple merges (llo, wor)
        -5.0f, -5.0f,  // ▁th, ▁ca
        -4.0f, -4.0f,  // Hello, world
        -3.0f, -3.0f,  // ▁the, ▁cat
        -2.0f, -2.0f,  // ▁Hello, ▁world
    };

    // Add byte fallback tokens <0x00>..<0xFF>
    for (int b = 0; b < 256; b++) {
        char buf[8];
        std::snprintf(buf, sizeof(buf), "<0x%02X>", b);
        tokens.push_back(buf);
        scores.push_back(-10.0f);
    }

    Tokenizer tok;
    tok.load_vocab(tokens, scores, /*bos_id=*/1, /*eos_id=*/2);
    tok.set_type("spm");
    tok.set_add_bos(true);
    tok.set_add_space_prefix(true);
    return tok;
}

// Build a minimal GPT2/BPE tokenizer
static Tokenizer make_gpt2_tokenizer() {
    std::vector<std::string> tokens;
    std::vector<float> scores;

    // Token 0: <unk>, 1: <s> (BOS), 2: </s> (EOS)
    tokens.push_back("<unk>"); scores.push_back(0.0f);
    tokens.push_back("<s>");   scores.push_back(0.0f);
    tokens.push_back("</s>"); scores.push_back(0.0f);

    auto codepoint_to_utf8 = [](uint32_t cp) -> std::string {
        std::string s;
        if (cp < 0x80) {
            s += static_cast<char>(cp);
        } else if (cp < 0x800) {
            s += static_cast<char>(0xC0 | (cp >> 6));
            s += static_cast<char>(0x80 | (cp & 0x3F));
        } else if (cp < 0x10000) {
            s += static_cast<char>(0xE0 | (cp >> 12));
            s += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            s += static_cast<char>(0x80 | (cp & 0x3F));
        }
        return s;
    };

    // GPT2 byte-to-codepoint table (matches tokenizer.cpp)
    static const uint32_t BYTE_TO_CP[256] = {
        256, 257, 258, 259, 260, 261, 262, 263, 264, 265,
        266, 267, 268, 269, 270, 271, 272, 273, 274, 275,
        276, 277, 278, 279, 280, 281, 282, 283, 284, 285,
        286, 287, 288,
        33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
        48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
        63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77,
        78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92,
        93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106,
        107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
        120, 121, 122, 123, 124, 125, 126,
        289, 290, 291, 292, 293, 294, 295, 296, 297, 298,
        299, 300, 301, 302, 303, 304, 305, 306, 307, 308,
        309, 310, 311, 312, 313, 314, 315, 316, 317, 318,
        319, 320, 321, 322,
        161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172,
        323,
        174, 175, 176, 177, 178, 179, 180, 181, 182, 183,
        184, 185, 186, 187, 188, 189, 190, 191, 192, 193,
        194, 195, 196, 197, 198, 199, 200, 201, 202, 203,
        204, 205, 206, 207, 208, 209, 210, 211, 212, 213,
        214, 215, 216, 217, 218, 219, 220, 221, 222, 223,
        224, 225, 226, 227, 228, 229, 230, 231, 232, 233,
        234, 235, 236, 237, 238, 239, 240, 241, 242, 243,
        244, 245, 246, 247, 248, 249, 250, 251, 252, 253,
        254, 255,
    };

    // Tokens 3-258: individual byte tokens in GPT2 encoding
    for (int b = 0; b < 256; b++) {
        tokens.push_back(codepoint_to_utf8(BYTE_TO_CP[b]));
        scores.push_back(0.0f);
    }
    // byte_token_id(b) = 3 + b

    // Add merged tokens (id 259+)
    std::string H_tok = codepoint_to_utf8(BYTE_TO_CP['H']);
    std::string e_tok = codepoint_to_utf8(BYTE_TO_CP['e']);
    std::string l_tok = codepoint_to_utf8(BYTE_TO_CP['l']);
    std::string o_tok = codepoint_to_utf8(BYTE_TO_CP['o']);

    std::string He_tok = H_tok + e_tok;
    std::string ll_tok = l_tok + l_tok;
    std::string llo_tok = ll_tok + o_tok;
    std::string Hello_tok = He_tok + llo_tok;

    int He_id = static_cast<int>(tokens.size());
    tokens.push_back(He_tok); scores.push_back(0.0f);

    int ll_id = static_cast<int>(tokens.size());
    tokens.push_back(ll_tok); scores.push_back(0.0f);

    int llo_id = static_cast<int>(tokens.size());
    tokens.push_back(llo_tok); scores.push_back(0.0f);

    int Hello_id = static_cast<int>(tokens.size());
    tokens.push_back(Hello_tok); scores.push_back(0.0f);
    (void)He_id; (void)ll_id; (void)llo_id; (void)Hello_id;

    Tokenizer tok;
    tok.load_vocab(tokens, scores, /*bos_id=*/1, /*eos_id=*/2);
    tok.set_type("gpt2");
    tok.set_add_bos(false);

    // Merge rules (applied in order: lower rank = higher priority)
    std::vector<std::string> merges = {
        H_tok + " " + e_tok,     // H + e -> He (rank 0)
        l_tok + " " + l_tok,     // l + l -> ll (rank 1)
        ll_tok + " " + o_tok,    // ll + o -> llo (rank 2)
        He_tok + " " + llo_tok,  // He + llo -> Hello (rank 3)
    };
    tok.load_merges(merges);

    return tok;
}

// ---- SPM Tokenizer Tests ----

TEST(TokenizerSPMTest, LoadVocab) {
    Tokenizer tok = make_spm_tokenizer();
    EXPECT_GT(tok.vocab_size(), 30);
    EXPECT_EQ(tok.bos_id(), 1);
    EXPECT_EQ(tok.eos_id(), 2);
}

TEST(TokenizerSPMTest, EmptyVocabFails) {
    Tokenizer tok;
    EXPECT_FALSE(tok.load_vocab({}, {}, 0, 0));
    EXPECT_EQ(tok.vocab_size(), 0);
}

TEST(TokenizerSPMTest, EmptyTextReturnsEmpty) {
    Tokenizer tok = make_spm_tokenizer();
    auto ids = tok.encode("");
    EXPECT_TRUE(ids.empty());
}

TEST(TokenizerSPMTest, EncodeWholeWord) {
    Tokenizer tok = make_spm_tokenizer();
    // "Hello" with space prefix -> merges to ▁Hello -> token 29
    auto ids = tok.encode("Hello");
    ASSERT_EQ(ids.size(), 1u);
    EXPECT_EQ(ids[0], 29);  // ▁Hello
}

TEST(TokenizerSPMTest, EncodeNoPrefix) {
    Tokenizer tok = make_spm_tokenizer();
    // "Hello" with no_prefix=true -> merges to Hello -> token 25
    auto ids = tok.encode("Hello", /*no_prefix=*/true);
    ASSERT_EQ(ids.size(), 1u);
    EXPECT_EQ(ids[0], 25);  // Hello
}

TEST(TokenizerSPMTest, EncodeMultipleWords) {
    Tokenizer tok = make_spm_tokenizer();
    // "Hello world" -> ▁Hello ▁world -> [29, 30]
    auto ids = tok.encode("Hello world");
    ASSERT_EQ(ids.size(), 2u);
    EXPECT_EQ(ids[0], 29);  // ▁Hello
    EXPECT_EQ(ids[1], 30);  // ▁world
}

TEST(TokenizerSPMTest, EncodeSubwordMerge) {
    Tokenizer tok = make_spm_tokenizer();
    // "the cat" -> ▁the ▁cat -> [27, 28]
    auto ids = tok.encode("the cat");
    ASSERT_EQ(ids.size(), 2u);
    EXPECT_EQ(ids[0], 27);  // ▁the
    EXPECT_EQ(ids[1], 28);  // ▁cat
}

TEST(TokenizerSPMTest, SpacePrefixDisabled) {
    Tokenizer tok = make_spm_tokenizer();
    tok.set_add_space_prefix(false);
    // "Hello" without space prefix -> merges to Hello -> token 25
    auto ids = tok.encode("Hello");
    ASSERT_EQ(ids.size(), 1u);
    EXPECT_EQ(ids[0], 25);  // Hello
}

TEST(TokenizerSPMTest, ByteFallback) {
    Tokenizer tok = make_spm_tokenizer();
    tok.set_add_space_prefix(false);
    // Tab character (0x09) not in vocab -> byte fallback <0x09>
    // Byte fallback tokens start at index 31
    auto ids = tok.encode("\t");
    ASSERT_FALSE(ids.empty());
    EXPECT_EQ(ids[0], 31 + 0x09);
}

TEST(TokenizerSPMTest, DecodeRoundtrip) {
    Tokenizer tok = make_spm_tokenizer();
    std::string original = "Hello world";
    auto ids = tok.encode(original);
    std::string decoded = tok.decode(ids);
    // SPM decode replaces ▁ with space, so decoded starts with space
    EXPECT_EQ(decoded, " Hello world");
}

TEST(TokenizerSPMTest, DecodeByteToken) {
    Tokenizer tok = make_spm_tokenizer();
    // Decode a byte fallback token: <0x41> = 'A' (byte fallback at index 31 + 0x41)
    int32_t byte_a_token = 31 + 0x41;
    std::string decoded = tok.decode_token(byte_a_token);
    EXPECT_EQ(decoded, "A");
}

TEST(TokenizerSPMTest, DecodeOutOfRange) {
    Tokenizer tok = make_spm_tokenizer();
    EXPECT_EQ(tok.decode_token(-1), "");
    EXPECT_EQ(tok.decode_token(99999), "");
}

TEST(TokenizerSPMTest, FindToken) {
    Tokenizer tok = make_spm_tokenizer();
    EXPECT_EQ(tok.find_token("<s>"), 1);
    EXPECT_EQ(tok.find_token("</s>"), 2);
    EXPECT_EQ(tok.find_token("<nonexistent>"), -1);
}

// ---- GPT2 Tokenizer Tests ----

TEST(TokenizerGPT2Test, EncodeSimple) {
    Tokenizer tok = make_gpt2_tokenizer();
    // "Hi" -> pre-tokenize: ["Hi"] -> bytes: H(72), i(105)
    // No merge for H+i, so individual byte tokens
    auto ids = tok.encode("Hi");
    ASSERT_EQ(ids.size(), 2u);
    EXPECT_EQ(ids[0], 3 + 'H');  // byte token for H
    EXPECT_EQ(ids[1], 3 + 'i');  // byte token for i
}

TEST(TokenizerGPT2Test, EncodeMerge) {
    Tokenizer tok = make_gpt2_tokenizer();
    // "Hello" -> bytes: H, e, l, l, o
    // Merges: H+e->He, l+l->ll, ll+o->llo, He+llo->Hello
    auto ids = tok.encode("Hello");
    ASSERT_EQ(ids.size(), 1u);
}

TEST(TokenizerGPT2Test, DecodeRoundtrip) {
    Tokenizer tok = make_gpt2_tokenizer();
    std::string original = "Hello";
    auto ids = tok.encode(original);
    std::string decoded = tok.decode(ids);
    EXPECT_EQ(decoded, original);
}

TEST(TokenizerGPT2Test, DecodeRoundtripASCII) {
    Tokenizer tok = make_gpt2_tokenizer();
    std::string original = "AB";
    auto ids = tok.encode(original);
    std::string decoded = tok.decode(ids);
    EXPECT_EQ(decoded, original);
}

TEST(TokenizerGPT2Test, EmptyText) {
    Tokenizer tok = make_gpt2_tokenizer();
    auto ids = tok.encode("");
    EXPECT_TRUE(ids.empty());
}

TEST(TokenizerGPT2Test, PreTokenizeDigits) {
    Tokenizer tok = make_gpt2_tokenizer();
    auto ids = tok.encode("1234");
    // 4 individual byte tokens (no merges for digits)
    EXPECT_EQ(ids.size(), 4u);
}

TEST(TokenizerGPT2Test, PreTokenizeSpaces) {
    Tokenizer tok = make_gpt2_tokenizer();
    // " A" -> 2 byte tokens
    auto ids = tok.encode(" A");
    EXPECT_EQ(ids.size(), 2u);
}

// ---- Type dispatch ----

TEST(TokenizerDispatchTest, SPMDefault) {
    Tokenizer tok;
    EXPECT_EQ(tok.type(), "spm");
}

TEST(TokenizerDispatchTest, SetType) {
    Tokenizer tok;
    tok.set_type("gpt2");
    EXPECT_EQ(tok.type(), "gpt2");
}

} // namespace
} // namespace imp
