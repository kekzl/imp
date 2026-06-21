// Tokenizer robustness / round-trip tests — TEST_AUDIT.md Phase 2, risk #10.
//
// The #510 class (NUL string terminators leaking into SSE deltas) lived in the
// special-token / byte rendering paths. These tests pin the contract that
// matters there: for a byte-level (GPT2) tokenizer, encode∘decode is the
// IDENTITY on arbitrary byte content — including embedded NUL, lone surrogate
// bytes (invalid UTF-8), 2/3/4-byte UTF-8, emoji + ZWJ, and long runs — and
// that boundary token ids (negative, 0, vocab_size, vocab_size-1) decode to ""
// without faulting.
//
// Why GPT2 for the identity assertions: byte-level BPE has a total byte→token
// mapping (every one of 256 bytes is a token), so the round-trip is exact by
// construction. SPM with a tiny synthetic vocab only round-trips via byte
// fallback and applies documented normalization (▁→space), so for SPM we assert
// the WEAKER, documented contract (no crash + non-empty decode), matching the
// existing suite's stance. Special-token-as-text behavior is asserted as the
// actual contract (CONTROL pre-split), not a wished-for one.

#include "model/tokenizer.h"
#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

namespace imp {
namespace {

// ---- Full byte-level GPT2 fixture (256 byte tokens, no merges) ----
//
// With every byte mapped to its own token and no merge rules, encode produces
// one token per byte and decode reverses it exactly. This makes the round-trip
// the cleanest possible identity oracle for "any bytes in → same bytes out".

static std::string codepoint_to_utf8(uint32_t cp) {
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
}

// GPT2 byte→codepoint table (must match tokenizer.cpp's byte_to_gpt2 mapping).
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

static Tokenizer make_byte_gpt2_tokenizer() {
    std::vector<std::string> tokens;
    std::vector<float> scores;
    tokens.push_back("<unk>");
    scores.push_back(0.0f);
    tokens.push_back("<s>");
    scores.push_back(0.0f);
    tokens.push_back("</s>");
    scores.push_back(0.0f);
    // ids 3..258: one token per byte value
    for (int b = 0; b < 256; b++) {
        tokens.push_back(codepoint_to_utf8(BYTE_TO_CP[b]));
        scores.push_back(0.0f);
    }
    Tokenizer tok;
    tok.load_vocab(tokens, scores, /*bos_id=*/1, /*eos_id=*/2);
    tok.set_type("gpt2");
    tok.set_add_bos(false);
    tok.load_merges({});  // no merges → exactly one token per byte
    return tok;
}

// Encode then decode; assert byte-exact identity.
static void expect_roundtrip(const Tokenizer& tok, const std::string& s, const char* what) {
    auto ids = tok.encode(s);
    std::string back = tok.decode(ids);
    EXPECT_EQ(back, s) << "round-trip mismatch for: " << what;
}

// ---- Byte-exact round-trip identity (GPT2 byte-level) ----

TEST(TokenizerRobustness, RoundtripASCII) {
    Tokenizer tok = make_byte_gpt2_tokenizer();
    expect_roundtrip(tok, "Hello, World! 0123456789", "ascii");
}

TEST(TokenizerRobustness, RoundtripUtf8TwoByte) {
    Tokenizer tok = make_byte_gpt2_tokenizer();
    // German umlauts + sharp s: ä ö ü ß — all 2-byte UTF-8.
    // Split after \x9f so the following 'e' is not swallowed into the hex escape
    // (\x is greedy: "\x9fe" would parse as one out-of-range escape, not ß + 'e').
    expect_roundtrip(tok, "Gr\xc3\xbc\xc3\x9f"
                          "e \xc3\xa4\xc3\xb6\xc3\xbc",
                     "utf8 2-byte");
}

TEST(TokenizerRobustness, RoundtripUtf8ThreeByte) {
    Tokenizer tok = make_byte_gpt2_tokenizer();
    // CJK "你好世界" — each char 3-byte UTF-8.
    expect_roundtrip(tok, "\xe4\xbd\xa0\xe5\xa5\xbd\xe4\xb8\x96\xe7\x95\x8c", "utf8 3-byte");
}

TEST(TokenizerRobustness, RoundtripUtf8FourByteEmoji) {
    Tokenizer tok = make_byte_gpt2_tokenizer();
    // 🌍 (U+1F30D) — 4-byte UTF-8.
    expect_roundtrip(tok, "earth \xf0\x9f\x8c\x8d done", "utf8 4-byte emoji");
}

TEST(TokenizerRobustness, RoundtripEmojiZWJSequence) {
    Tokenizer tok = make_byte_gpt2_tokenizer();
    // 👩‍💻 = woman (U+1F469) + ZWJ (U+200D) + laptop (U+1F4BB). The ZWJ
    // (E2 80 8D) is the exact kind of invisible joiner that special-token
    // rendering paths mishandle. Byte-level must reproduce every byte.
    expect_roundtrip(tok, "\xf0\x9f\x91\xa9\xe2\x80\x8d\xf0\x9f\x92\xbb", "emoji ZWJ sequence");
}

TEST(TokenizerRobustness, RoundtripEmbeddedNUL) {
    // THE #510 class: a NUL byte in the middle of the text. std::string holds
    // it fine; the tokenizer must encode it as the byte-0 token and decode it
    // back — not truncate at the NUL nor leak a stray terminator.
    Tokenizer tok = make_byte_gpt2_tokenizer();
    std::string s("a\0b", 3);  // explicit length: "a", NUL, "b"
    ASSERT_EQ(s.size(), 3u);
    auto ids = tok.encode(s);
    std::string back = tok.decode(ids);
    ASSERT_EQ(back.size(), 3u) << "NUL byte was dropped or string truncated";
    EXPECT_EQ(back, s);
    EXPECT_EQ(back[1], '\0');
}

TEST(TokenizerRobustness, RoundtripAllNULRun) {
    // A run of pure NUL bytes — stresses the "empty-looking but non-empty"
    // edge where C-string logic would see length 0.
    Tokenizer tok = make_byte_gpt2_tokenizer();
    std::string s(8, '\0');
    auto ids = tok.encode(s);
    std::string back = tok.decode(ids);
    EXPECT_EQ(back, s);
    EXPECT_EQ(back.size(), 8u);
}

TEST(TokenizerRobustness, RoundtripLoneSurrogateBytes) {
    // Invalid UTF-8: bytes 0xED 0xA0 0x80 are the encoding of a lone high
    // surrogate (U+D800), which is NOT valid UTF-8. A byte-level tokenizer has
    // no notion of validity — it must pass the raw bytes through unchanged and
    // not crash on the malformed sequence. Defined behavior = identity.
    Tokenizer tok = make_byte_gpt2_tokenizer();
    std::string s("x\xed\xa0\x80y", 5);
    auto ids = tok.encode(s);
    std::string back = tok.decode(ids);
    EXPECT_EQ(back, s);
}

TEST(TokenizerRobustness, RoundtripAllSingleBytes) {
    // Every one of the 256 byte values, individually, round-trips to itself.
    // This is the exhaustive form of the NUL test and the strongest possible
    // identity assertion for the byte path.
    Tokenizer tok = make_byte_gpt2_tokenizer();
    for (int b = 0; b < 256; b++) {
        std::string s(1, static_cast<char>(b));
        auto ids = tok.encode(s);
        std::string back = tok.decode(ids);
        EXPECT_EQ(back, s) << "byte 0x" << std::hex << b << " did not round-trip";
    }
}

TEST(TokenizerRobustness, RoundtripLongWhitespaceRun) {
    Tokenizer tok = make_byte_gpt2_tokenizer();
    // Long run of spaces + tabs + newlines. Whitespace handling in the
    // pre-tokenizer is a common place to lose or duplicate bytes.
    std::string s;
    for (int i = 0; i < 200; i++)
        s += " \t\n";
    expect_roundtrip(tok, s, "long whitespace run");
}

TEST(TokenizerRobustness, RoundtripVeryLongToken) {
    Tokenizer tok = make_byte_gpt2_tokenizer();
    // A 5000-byte run of a single non-space byte (one "word"): exercises the
    // BPE inner loop's allocation/linked-list path at scale without merges.
    std::string s(5000, 'Z');
    expect_roundtrip(tok, s, "very long single token");
}

// ---- Boundary token-id decode (must never fault) ----

TEST(TokenizerRobustness, DecodeBoundaryIdsNoCrash) {
    Tokenizer tok = make_byte_gpt2_tokenizer();
    const int vs = tok.vocab_size();
    ASSERT_GT(vs, 0);
    // Out-of-range ids must decode to "" (the documented guard), never index
    // out of bounds. -1, vocab_size, and a far-out id are all invalid.
    EXPECT_EQ(tok.decode_token(-1), "");
    EXPECT_EQ(tok.decode_token(vs), "");
    EXPECT_EQ(tok.decode_token(vs + 1000), "");
    EXPECT_EQ(tok.decode_token(INT32_MIN), "");
    EXPECT_EQ(tok.decode_token(INT32_MAX), "");
    // Valid boundary ids must decode without faulting (content unimportant).
    (void)tok.decode_token(0);
    (void)tok.decode_token(vs - 1);
}

TEST(TokenizerRobustness, DecodeMixedValidInvalidIds) {
    // A decode of a sequence containing out-of-range ids must skip them (each
    // contributes "") and still emit the valid ones — no crash, no desync.
    Tokenizer tok = make_byte_gpt2_tokenizer();
    int a = tok.find_token(codepoint_to_utf8(BYTE_TO_CP['A']));
    int b = tok.find_token(codepoint_to_utf8(BYTE_TO_CP['B']));
    ASSERT_GE(a, 0);
    ASSERT_GE(b, 0);
    std::vector<int32_t> ids = {a, -5, b, tok.vocab_size() + 7, a};
    std::string back = tok.decode(ids);
    EXPECT_EQ(back, "ABA");
}

// ---- Special-token strings as TEXT input (contract pinning) ----
//
// These pin the ACTUAL behavior (audit §4: characterize, don't wish). The
// contract differs by whether token_types_ marks a literal as CONTROL.

TEST(TokenizerRobustness, LiteralMarkerWithoutControlIsPlainText) {
    // No token_types loaded → special_pieces_ is empty → no pre-split. A
    // literal "<think>" typed by a user is encoded as ordinary bytes and
    // round-trips as text. CONTRACT: without CONTROL metadata, markers are
    // NOT promoted to control tokens.
    Tokenizer tok = make_byte_gpt2_tokenizer();
    EXPECT_FALSE(tok.has_token_types());
    std::string s = "say <think> literally";
    auto ids = tok.encode(s);
    std::string back = tok.decode(ids);
    EXPECT_EQ(back, s);
}

TEST(TokenizerRobustness, LiteralMarkerWithControlIsPreSplit) {
    // When "<think>" exists in the vocab AND is tagged CONTROL, the encoder
    // pre-splits on it and emits its single control-token id for the literal
    // substring. CONTRACT: this IS the documented behavior — a user who types
    // the literal marker gets the control token. (Whether that is desirable is
    // a policy question above the tokenizer; we pin the mechanism.)
    std::vector<std::string> tokens;
    std::vector<float> scores;
    tokens.push_back("<unk>");
    scores.push_back(0.0f);
    tokens.push_back("<s>");
    scores.push_back(0.0f);
    tokens.push_back("</s>");
    scores.push_back(0.0f);
    for (int b = 0; b < 256; b++) {
        tokens.push_back(codepoint_to_utf8(BYTE_TO_CP[b]));
        scores.push_back(0.0f);
    }
    int think_id = static_cast<int>(tokens.size());
    tokens.push_back("<think>");
    scores.push_back(0.0f);

    Tokenizer tok;
    tok.load_vocab(tokens, scores, /*bos_id=*/1, /*eos_id=*/2);
    tok.set_type("gpt2");
    tok.set_add_bos(false);
    tok.load_merges({});
    // Mark <think> as CONTROL (type 3) — this builds special_pieces_.
    std::vector<int32_t> types(tokens.size(), 1);
    types[think_id] = 3;
    tok.load_token_types(types);

    auto ids = tok.encode("a<think>b");
    // Expect: byte 'a', the single <think> control token, byte 'b'.
    bool saw_think = false;
    for (int32_t id : ids)
        if (id == think_id)
            saw_think = true;
    EXPECT_TRUE(saw_think) << "CONTROL-tagged literal marker must encode as its control token";
    // And decoding that control token renders the literal text back (the #510
    // rendering surface): no stray NUL, exactly "<think>".
    EXPECT_EQ(tok.decode_token(think_id), "<think>");
}

// ---- SPM weaker contract (documented normalization, no crash) ----

TEST(TokenizerRobustness, SpmByteFallbackNoCrashOnArbitraryBytes) {
    // SPM with a byte-fallback vocab: arbitrary bytes (incl. NUL and invalid
    // UTF-8) must encode without crashing. We assert the documented weaker
    // contract: round-trip reproduces the original bytes via byte fallback
    // (SPM applies ▁→space normalization only to the space marker, absent here
    // because add_space_prefix is off and the input has no ▁).
    std::vector<std::string> tokens = {"<unk>", "<s>", "</s>"};
    std::vector<float> scores = {0.0f, 0.0f, 0.0f};
    int byte_base = static_cast<int>(tokens.size());
    for (int b = 0; b < 256; b++) {
        char buf[8];
        std::snprintf(buf, sizeof(buf), "<0x%02X>", b);
        tokens.push_back(buf);
        scores.push_back(-10.0f);
    }
    (void)byte_base;
    Tokenizer tok;
    tok.load_vocab(tokens, scores, 1, 2);
    tok.set_type("spm");
    tok.set_add_bos(false);
    tok.set_add_space_prefix(false);

    std::string s("\x01\x02\xff\x00\x7f", 5);
    auto ids = tok.encode(s);
    ASSERT_FALSE(ids.empty());
    std::string back = tok.decode(ids);
    EXPECT_EQ(back, s) << "SPM byte fallback must reconstruct raw bytes";
}

}  // namespace
}  // namespace imp
