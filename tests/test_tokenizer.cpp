#include "model/tokenizer.h"
#include <gtest/gtest.h>

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

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
        "<unk>",
        "<s>",
        "</s>",
        "\xe2\x96\x81",  // ▁ (bare space)
        "H",
        "e",
        "l",
        "o",
        "w",
        "r",
        "d",
        "t",
        "h",
        "c",
        "a",  // 4-14
        "He",
        "ll",
        "or",
        "ld",
        "th",
        "ca",  // 15-20
        "llo",
        "wor",  // 21-22
        "\xe2\x96\x81"
        "th",
        "\xe2\x96\x81"
        "ca",  // 23-24 (▁th, ▁ca)
        "Hello",
        "world",  // 25-26
        "\xe2\x96\x81"
        "the",
        "\xe2\x96\x81"
        "cat",  // 27-28 (▁the, ▁cat)
        "\xe2\x96\x81"
        "Hello",
        "\xe2\x96\x81"
        "world",  // 29-30 (▁Hello, ▁world)
    };
    // Scores: higher (less negative) = merge first. The BPE algorithm picks the
    // highest-scoring pair at each step, so longer tokens need higher scores.
    std::vector<float> scores = {
        0.0f,  0.0f,  0.0f,
        -8.0f,                                                                        // ▁
        -8.0f, -8.0f, -8.0f, -8.0f, -8.0f, -8.0f, -8.0f, -8.0f, -8.0f, -8.0f, -8.0f,  // chars
        -6.0f, -6.0f, -6.0f, -6.0f, -6.0f, -6.0f,                                     // pair merges
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
    tokens.push_back("<unk>");
    scores.push_back(0.0f);
    tokens.push_back("<s>");
    scores.push_back(0.0f);
    tokens.push_back("</s>");
    scores.push_back(0.0f);

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
        256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275,
        276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 33,  34,  35,  36,  37,  38,  39,
        40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,
        60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,
        80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,  97,  98,  99,
        100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
        120, 121, 122, 123, 124, 125, 126, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301,
        302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321,
        322, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 323, 174, 175, 176, 177, 178, 179,
        180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199,
        200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219,
        220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239,
        240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255,
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
    tokens.push_back(He_tok);
    scores.push_back(0.0f);

    int ll_id = static_cast<int>(tokens.size());
    tokens.push_back(ll_tok);
    scores.push_back(0.0f);

    int llo_id = static_cast<int>(tokens.size());
    tokens.push_back(llo_tok);
    scores.push_back(0.0f);

    int Hello_id = static_cast<int>(tokens.size());
    tokens.push_back(Hello_tok);
    scores.push_back(0.0f);
    (void)He_id;
    (void)ll_id;
    (void)llo_id;
    (void)Hello_id;

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

// ---- SPM UTF-8 / Edge Case Tests ----

TEST(TokenizerSPMTest, EncodeUTF8Multibyte) {
    Tokenizer tok = make_spm_tokenizer();
    // "café" contains multibyte UTF-8 (é = 0xC3 0xA9). Small vocab will use byte
    // fallback, but must not crash.
    auto ids = tok.encode("caf\xc3\xa9");
    EXPECT_GT(ids.size(), 0u);
    // Round-trip: decode should produce something (byte fallback reconstructs bytes)
    std::string decoded = tok.decode(ids);
    EXPECT_FALSE(decoded.empty());
}

TEST(TokenizerSPMTest, EncodeEmoji) {
    Tokenizer tok = make_spm_tokenizer();
    // "Hello 🌍" — emoji is 4-byte UTF-8 (F0 9F 8C 8D), falls back to byte tokens
    auto ids = tok.encode("Hello \xf0\x9f\x8c\x8d");
    EXPECT_GT(ids.size(), 0u);
    std::string decoded = tok.decode(ids);
    EXPECT_FALSE(decoded.empty());
}

TEST(TokenizerSPMTest, EncodeCJK) {
    Tokenizer tok = make_spm_tokenizer();
    // "你好世界" — each CJK char is 3-byte UTF-8, will byte-fallback
    auto ids = tok.encode("\xe4\xbd\xa0\xe5\xa5\xbd\xe4\xb8\x96\xe7\x95\x8c");
    EXPECT_GT(ids.size(), 0u);
    std::string decoded = tok.decode(ids);
    EXPECT_FALSE(decoded.empty());
}

// ---- GPT2 UTF-8 / Edge Case Tests ----

TEST(TokenizerGPT2Test, EncodeUTF8) {
    Tokenizer tok = make_gpt2_tokenizer();
    // "café" — GPT2 byte-level BPE encodes all bytes, should round-trip
    auto ids = tok.encode("caf\xc3\xa9");
    EXPECT_GT(ids.size(), 0u);
    std::string decoded = tok.decode(ids);
    EXPECT_EQ(decoded, "caf\xc3\xa9");
}

TEST(TokenizerGPT2Test, LongString) {
    Tokenizer tok = make_gpt2_tokenizer();
    // 1000-char string of repeating ASCII
    std::string long_str(1000, 'x');
    auto ids = tok.encode(long_str);
    EXPECT_GT(ids.size(), 0u);
    std::string decoded = tok.decode(ids);
    EXPECT_EQ(decoded, long_str);
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

TEST(TokenizerDispatchTest, MaxLength) {
    // Test both tokenizer types with a 10K-char string — must not crash
    std::string huge(10000, 'A');

    {
        Tokenizer tok = make_spm_tokenizer();
        auto ids = tok.encode(huge);
        EXPECT_GT(ids.size(), 0u);
        EXPECT_LE(ids.size(), 10001u);  // at most 1 token per char + space prefix
    }
    {
        Tokenizer tok = make_gpt2_tokenizer();
        auto ids = tok.encode(huge);
        EXPECT_GT(ids.size(), 0u);
        EXPECT_LE(ids.size(), 10001u);
    }
}

// ---- Gemma-4 Tokenizer Tests ----

// Synthetic Gemma-4 tokenizer reproducing the byte-fallback bug on "Linus".
// merge_ranks contains "Lin u" → "Linu" as an intermediate rule whose output
// is NOT in the vocabulary. Buggy code applies the merge unconditionally,
// producing an unknown symbol that byte-falls-back the entire word.
//
// Layout:
//   0: <unk>  1: <s>  2: </s>
//   3: ▁
//   4-14: individual ASCII chars (L, i, n, u, s, ...)
//   15: "Li"  16: "Lin"  17: "us"
//   18-273: byte fallback <0x00>..<0xFF>
static Tokenizer make_gemma4_tokenizer() {
    std::vector<std::string> tokens = {
        "<unk>",
        "<s>",
        "</s>",
        "\xe2\x96\x81",  // ▁ (id 3)
        "L",
        "i",
        "n",
        "u",
        "s",  // 4-8
        " ",
        "\t",
        "\n",  // 9-11 (raw whitespace, rarely used)
    };
    std::vector<float> scores(tokens.size(), 0.0f);

    int Li_id = static_cast<int>(tokens.size());
    tokens.push_back("Li");
    scores.push_back(0.0f);

    int Lin_id = static_cast<int>(tokens.size());
    tokens.push_back("Lin");
    scores.push_back(0.0f);

    int us_id = static_cast<int>(tokens.size());
    tokens.push_back("us");
    scores.push_back(0.0f);
    (void)Li_id;

    // Byte fallback <0x00>..<0xFF>
    int byte_base = static_cast<int>(tokens.size());
    for (int b = 0; b < 256; b++) {
        char buf[8];
        std::snprintf(buf, sizeof(buf), "<0x%02X>", b);
        tokens.push_back(buf);
        scores.push_back(0.0f);
    }

    Tokenizer tok;
    tok.load_vocab(tokens, scores, /*bos_id=*/1, /*eos_id=*/2);
    tok.set_type("gemma4");
    tok.set_add_bos(false);
    tok.set_add_space_prefix(false);  // Gemma handles ▁ internally

    // Merge rules (lower rank = higher priority). Key format: "a b".
    // "Lin u" → "Linu" is an intermediate merge; "Linu" is NOT in vocab.
    // A correct BPE implementation (like llama.cpp) skips this merge.
    std::vector<std::string> merges = {
        "L i",    // rank 0 → Li   (in vocab)
        "Li n",   // rank 1 → Lin  (in vocab)
        "Lin u",  // rank 2 → Linu (NOT in vocab — trigger for bug)
        "u s",    // rank 3 → us   (in vocab)
    };
    tok.load_merges(merges);
    (void)Lin_id;
    (void)us_id;
    (void)byte_base;
    return tok;
}

TEST(TokenizerGemma4Test, MergeSkippedWhenResultNotInVocab) {
    Tokenizer tok = make_gemma4_tokenizer();
    int Lin_id = tok.find_token("Lin");
    int us_id = tok.find_token("us");
    ASSERT_GE(Lin_id, 0);
    ASSERT_GE(us_id, 0);
    ASSERT_LT(tok.find_token("Linu"), 0);  // sanity: "Linu" must NOT be in vocab

    // "Linus": buggy code merges all the way to "Linu" (intermediate) then
    // byte-fallbacks the unknown symbol, producing 5 byte-fallback tokens.
    // Correct behavior: skip the "Lin u" merge, apply "u s → us" instead,
    // yielding [Lin, us] — 2 tokens.
    auto ids = tok.encode("Linus");
    ASSERT_EQ(ids.size(), 2u) << "expected [Lin, us], got byte-fallback";
    EXPECT_EQ(ids[0], Lin_id);
    EXPECT_EQ(ids[1], us_id);
}

TEST(TokenizerGemma4Test, KnownSingleTokenMergesStillWork) {
    // Regression guard: even with the merge-guard in place, a string whose
    // full merge chain stays in-vocab must still produce a single token.
    Tokenizer tok = make_gemma4_tokenizer();
    auto ids = tok.encode("Lin");
    ASSERT_EQ(ids.size(), 1u);
    EXPECT_EQ(ids[0], tok.find_token("Lin"));
}

TEST(TokenizerGemma4Test, TruncatedUTF8AtEndDoesNotCrash) {
    // Regression: nfc_decode_utf8 used to advance pos past end-of-string
    // on a truncated multi-byte sequence, producing a partial codepoint.
    // Ensure encoding a truncated UTF-8 tail terminates cleanly.
    Tokenizer tok = make_gemma4_tokenizer();
    std::string truncated = "Lin\xe2\x96";  // UTF-8 ▁ missing its 3rd byte
    auto ids = tok.encode(truncated);
    EXPECT_GT(ids.size(), 0u);
}

TEST(TokenizerGemma4Test, DecodeByteFallbackFormsValidUTF8) {
    // Bytes 0xE2 0x96 0x81 in sequence form the UTF-8 of ▁ (U+2581). After
    // Gemma-4 decode, ▁ must be replaced by ASCII space — even when the
    // three bytes arrive as three separate byte-fallback tokens.
    //
    // Before fix: decode_spm_token runs SPIECE_SPACE replacement per-token,
    // fails to stitch the 3 bytes together, returns literal "▁Linus".
    Tokenizer tok = make_gemma4_tokenizer();
    int byte_base = tok.find_token("<0x00>");
    ASSERT_GE(byte_base, 0);

    std::vector<int32_t> ids = {
        byte_base + 0xE2, byte_base + 0x96, byte_base + 0x81,  // ▁
        byte_base + 'L',  byte_base + 'i',  byte_base + 'n',  byte_base + 'u', byte_base + 's',
    };
    std::string decoded = tok.decode(ids);
    EXPECT_EQ(decoded, " Linus");
}

// mark_as_control: defensive overlay used by the SafeTensors loader to patch
// missing CONTROL flags from special_tokens_map.json's authoritative list.
TEST(TokenizerControlTest, MarkAsControlAllocatesAndSets) {
    Tokenizer tok = make_spm_tokenizer();
    // Fixture has no token_types yet → has_token_types() is false.
    EXPECT_FALSE(tok.has_token_types());

    int32_t id_unk = 0;                          // <unk> in the fixture
    EXPECT_FALSE(tok.is_control_token(id_unk));  // empty types → false

    tok.mark_as_control(id_unk);
    EXPECT_TRUE(tok.has_token_types());  // lazy alloc
    EXPECT_TRUE(tok.is_control_token(id_unk));

    // Other vocab entries default to NORMAL=1 (not CONTROL).
    EXPECT_FALSE(tok.is_control_token(4));   // 'H'
    EXPECT_FALSE(tok.is_control_token(25));  // 'Hello'
}

TEST(TokenizerControlTest, MarkAsControlIsIdempotent) {
    Tokenizer tok = make_spm_tokenizer();
    tok.mark_as_control(2);  // </s>
    tok.mark_as_control(2);  // again
    EXPECT_TRUE(tok.is_control_token(2));
}

TEST(TokenizerControlTest, MarkAsControlIgnoresInvalidIds) {
    Tokenizer tok = make_spm_tokenizer();
    tok.mark_as_control(-1);                // invalid
    tok.mark_as_control(tok.vocab_size());  // out of range
    EXPECT_FALSE(tok.has_token_types());    // no allocation triggered
}

TEST(TokenizerControlTest, PreservesExistingTypes) {
    Tokenizer tok = make_spm_tokenizer();
    // Pre-populate with a custom type vector (e.g. simulating GGUF metadata).
    std::vector<int32_t> types(tok.vocab_size(), 1);  // all NORMAL
    types[1] = 3;                                     // <s> = CONTROL
    types[2] = 3;                                     // </s> = CONTROL
    tok.load_token_types(types);

    // Patching a previously-NORMAL token must not clobber the others.
    tok.mark_as_control(0);  // <unk>
    EXPECT_TRUE(tok.is_control_token(0));
    EXPECT_TRUE(tok.is_control_token(1));
    EXPECT_TRUE(tok.is_control_token(2));
    EXPECT_FALSE(tok.is_control_token(4));  // 'H' still normal
}

// ---- Qwen2 pre-tokenizer (#657) ----
//
// Expected segmentations derived from the canonical Qwen2 regex and verified
// against llama.cpp `llama-tokenize` on Qwen3-8B (the listed chunks correspond
// 1:1 to the canonical token ids on these probes). The old gpt2 fallback
// split every punctuation char individually, making canonical merges
// ("->", "://", "(x", ".com") impossible.

using Chunks = std::vector<std::string>;

TEST(Qwen2PreTokenizeTest, SymbolRunsMerge) {
    EXPECT_EQ(qwen2_pre_tokenize("x->bar"), (Chunks{"x", "->", "bar"}));
    EXPECT_EQ(qwen2_pre_tokenize("https://example.com"),
              (Chunks{"https", "://", "example", ".com"}));
}

TEST(Qwen2PreTokenizeTest, PrefixCharGluesToLetterRun) {
    // [^\r\n\p{L}\p{N}]?\p{L}+ — '(' and '.' attach to the following word.
    EXPECT_EQ(qwen2_pre_tokenize("def foo(x):\n"),
              (Chunks{"def", " foo", "(x", "):\n"}));
}

TEST(Qwen2PreTokenizeTest, SingleDigits) {
    // Qwen2 splits digits individually (\p{N}), unlike the 3-digit gpt2 rule.
    EXPECT_EQ(qwen2_pre_tokenize("q=1&r=42"),
              (Chunks{"q", "=", "1", "&r", "=", "4", "2"}));
}

TEST(Qwen2PreTokenizeTest, Contractions) {
    EXPECT_EQ(qwen2_pre_tokenize("don't stop"), (Chunks{"don", "'t", " stop"}));
}

TEST(Qwen2PreTokenizeTest, IndentationKeepsOneSpaceForWord) {
    // \s+(?!\S) backtracks one position: 4-space indent → "   " + " return".
    EXPECT_EQ(qwen2_pre_tokenize("    return x"),
              (Chunks{"   ", " return", " x"}));
}

TEST(Qwen2PreTokenizeTest, NewlineRuns) {
    // \s*[\r\n]+ groups whitespace up to the LAST newline; the remaining
    // indent feeds the next chunk.
    EXPECT_EQ(qwen2_pre_tokenize("a\n\n  b"), (Chunks{"a", "\n\n", " ", " b"}));
    // Symbol run absorbs trailing newlines ([\r\n]*).
    EXPECT_EQ(qwen2_pre_tokenize("};\n\nint"), (Chunks{"};\n\n", "int"}));
}

TEST(Qwen2PreTokenizeTest, TrailingWhitespace) {
    EXPECT_EQ(qwen2_pre_tokenize("abc   "), (Chunks{"abc", "   "}));
}

// ---- o200k pre-tokenizer (gpt-oss / GPT-4o, #657) ----
//
// Expected chunks verified against HF tokenizers `pre_tokenize_str` on the
// gpt-oss-20b tokenizer.json (Ġ/Ċ rendered as plain space/newline here).

TEST(O200kPreTokenizeTest, CodeLine) {
    EXPECT_EQ(o200k_pre_tokenize("def foo(x): return x"),
              (Chunks{"def", " foo", "(x", "):", " return", " x"}));
}

TEST(O200kPreTokenizeTest, DigitTriples) {
    // \p{N}{1,3}: greedy groups of three, and a bare space stays standalone
    // before digits (no ' ?' option on the digit rule).
    EXPECT_EQ(o200k_pre_tokenize("17 + 25 = 12345"),
              (Chunks{"17", " +", " ", "25", " =", " ", "123", "45"}));
}

TEST(O200kPreTokenizeTest, UrlPrefixAndSlashes) {
    EXPECT_EQ(o200k_pre_tokenize("https://a.com/x?q=1"),
              (Chunks{"https", "://", "a", ".com", "/x", "?q", "=", "1"}));
}

TEST(O200kPreTokenizeTest, ContractionIsSuffix) {
    // Unlike qwen2 (standalone leading alternative), o200k attaches the
    // contraction to the letter run: "don't" is ONE chunk.
    EXPECT_EQ(o200k_pre_tokenize("don't stop"), (Chunks{"don't", " stop"}));
}

TEST(O200kPreTokenizeTest, CaseAwareLetterRuns) {
    EXPECT_EQ(o200k_pre_tokenize("camelCase HTTPSession ALLCAPS"),
              (Chunks{"camel", "Case", " HTTPSession", " ALLCAPS"}));
}

TEST(O200kPreTokenizeTest, WhitespaceBacktracking) {
    EXPECT_EQ(o200k_pre_tokenize("    return x"), (Chunks{"   ", " return", " x"}));
    EXPECT_EQ(o200k_pre_tokenize("a  b"), (Chunks{"a", " ", " b"}));
}

TEST(O200kPreTokenizeTest, SymbolRunTrailingNewlinesAndSlashes) {
    EXPECT_EQ(o200k_pre_tokenize("};\n\nint x"), (Chunks{"};\n\n", "int", " x"}));
    // The trailing class is [\r\n/]*: a slash absorbs a following newline.
    EXPECT_EQ(o200k_pre_tokenize("x\n/\ny"), (Chunks{"x", "\n", "/\n", "y"}));
}

TEST(O200kPreTokenizeTest, NonAsciiSymbolsAreSymbolsNotLetters) {
    // → (U+2192) and — (U+2014) are \p{S}/\p{P}, NOT letters: they take the
    // symbol-run rule (which absorbs trailing newlines). With the old
    // ≥0x80=letter approximation, " →\n" became [" →", "\n"] and diverged
    // from canonical (the 4 residual corpus diffs in #657).
    EXPECT_EQ(o200k_pre_tokenize("a \xe2\x86\x92\nb"), (Chunks{"a", " \xe2\x86\x92\n", "b"}));
    EXPECT_EQ(o200k_pre_tokenize("x \xe2\x80\x94 y"), (Chunks{"x", " \xe2\x80\x94", " y"}));
    // Non-ASCII LETTERS keep working: "Käse" stays one chunk.
    EXPECT_EQ(o200k_pre_tokenize("K\xc3\xa4se"), (Chunks{"K\xc3\xa4se"}));
}

TEST(Qwen2PreTokenizeTest, NonAsciiSymbolsAreSymbolsNotLetters) {
    EXPECT_EQ(qwen2_pre_tokenize("a \xe2\x86\x92\nb"), (Chunks{"a", " \xe2\x86\x92\n", "b"}));
    EXPECT_EQ(qwen2_pre_tokenize("K\xc3\xa4se"), (Chunks{"K\xc3\xa4se"}));
}

// ---- cl100k pre-tokenizer (GPT-4/tiktoken lineage, Phi-4; #657) ----
//
// qwen2 rules with digit triples. Chunk truths verified against HF
// tokenizers pre_tokenize_str on Phi-4-reasoning-plus.

TEST(Cl100kPreTokenizeTest, DigitTriplesCaseBlindStandaloneContractions) {
    EXPECT_EQ(cl100k_pre_tokenize("17 + 12345"),
              (Chunks{"17", " +", " ", "123", "45"}));
    // Case-blind letter runs: camelCase stays ONE chunk (unlike o200k).
    EXPECT_EQ(cl100k_pre_tokenize("camelCase HTTPSession"),
              (Chunks{"camelCase", " HTTPSession"}));
    // Contractions split standalone (like qwen2, unlike o200k's suffix).
    EXPECT_EQ(cl100k_pre_tokenize("don't stop"), (Chunks{"don", "'t", " stop"}));
    EXPECT_EQ(cl100k_pre_tokenize("a->b https://x.com/y"),
              (Chunks{"a", "->", "b", " https", "://", "x", ".com", "/y"}));
}

// ---- Added-token atomic matching: `normalized=false` regardless of `special` ----
//
// HF matches an added token atomically against the raw input iff normalized=false
// (special only governs decode-skipping). Qwen3 ships <think>/</think> and
// <tool_call> as special=false, normalized=false — they MUST tokenize to their
// single atomic id, not BPE-split into "<","think",">". A regression here breaks
// the <think>-as-stop-token guard and no-think suppression.

static std::string write_temp_tokenizer_json(const std::string& body) {
    std::string path = std::string("/tmp/imp_tok_test_") + std::to_string(::getpid()) +
                       "_" + std::to_string(reinterpret_cast<uintptr_t>(&body)) + ".json";
    std::ofstream(path) << body;
    return path;
}

// id 200 = <think> (special=false, normalized=false) -> atomic
// id 201 = <plain> (special=false, normalized=true)  -> NOT atomic (BPE-split)
// id 202 = <|ctrl|> (special=true)                   -> atomic (control)
static const char* kAddedTokenJson = R"JSON({
  "model": { "type": "BPE", "vocab": { "a": 0, "b": 1, "c": 2 }, "merges": [] },
  "added_tokens": [
    { "id": 200, "content": "<think>",  "special": false, "normalized": false },
    { "id": 201, "content": "<plain>",  "special": false, "normalized": true  },
    { "id": 202, "content": "<|ctrl|>", "special": true,  "normalized": false }
  ]
})JSON";

static bool contains_id(const std::vector<int32_t>& v, int32_t id) {
    for (int32_t x : v)
        if (x == id)
            return true;
    return false;
}

TEST(TokenizerAddedTokens, NormalizedFalseNonSpecialIsAtomic) {
    std::string path = write_temp_tokenizer_json(kAddedTokenJson);
    Tokenizer tok;
    ASSERT_TRUE(tok.load(path));
    std::remove(path.c_str());

    // <think>: special=false but normalized=false -> single atomic id 200.
    auto think = tok.encode("<think>");
    EXPECT_TRUE(contains_id(think, 200)) << "<think> not matched atomically (BPE-split)";

    // It must NOT decode-skip (special=false stays visible).
    EXPECT_NE(tok.decode({200}).find("<think>"), std::string::npos);

    // Control token (special=true) is atomic as before.
    EXPECT_TRUE(contains_id(tok.encode("<|ctrl|>"), 202));
}

TEST(TokenizerAddedTokens, NormalizedTrueIsNotPromoted) {
    std::string path = write_temp_tokenizer_json(kAddedTokenJson);
    Tokenizer tok;
    ASSERT_TRUE(tok.load(path));
    std::remove(path.c_str());

    // <plain>: normalized=true -> must NOT be matched as the atomic id 201.
    EXPECT_FALSE(contains_id(tok.encode("<plain>"), 201))
        << "normalized=true token should not be promoted to atomic matching";
}

}  // namespace
}  // namespace imp
