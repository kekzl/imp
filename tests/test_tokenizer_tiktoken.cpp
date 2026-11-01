// tiktoken loader (Tokenizer::load_tiktoken) on a synthetic rank file: 256 byte ranks plus a few
// merged tokens, so tiktoken's lowest-rank-concatenation rule is checked without the 2.8 MB Kimi file.
// Full HF parity on the real file: test_tokenizer_kimi.cpp (env-gated).
#include "model/tokenizer.h"

#include <gtest/gtest.h>

#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

namespace imp {
namespace {

using Ids = std::vector<int32_t>;

std::string b64(const std::string& in) {
    static const char* kAlpha = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    size_t i = 0;
    for (; i + 2 < in.size(); i += 3) {
        const uint32_t v = (static_cast<uint8_t>(in[i]) << 16) | (static_cast<uint8_t>(in[i + 1]) << 8) |
                           static_cast<uint8_t>(in[i + 2]);
        for (int s = 18; s >= 0; s -= 6)
            out += kAlpha[(v >> s) & 63];
    }
    if (i + 1 == in.size()) {
        const uint32_t v = static_cast<uint8_t>(in[i]) << 16;
        out += std::string{kAlpha[(v >> 18) & 63], kAlpha[(v >> 12) & 63], '=', '='};
    } else if (i + 2 == in.size()) {
        const uint32_t v = (static_cast<uint8_t>(in[i]) << 16) | (static_cast<uint8_t>(in[i + 1]) << 8);
        out += std::string{kAlpha[(v >> 18) & 63], kAlpha[(v >> 12) & 63], kAlpha[(v >> 6) & 63], '='};
    }
    return out;
}

std::string temp_path(const char* ext) {
    static int seq = 0;
    return "/tmp/imp_tiktoken_" + std::to_string(::getpid()) + "_" + std::to_string(seq++) + ext;
}

// Ranks 0-255 = the bytes, then 256 "ab", 257 "bc", 258 "abc", 259 "yz", 260 "xy", 261 "pqr".
std::string rank_file(const std::vector<std::string>& extra, int skip_byte = -1) {
    std::string body;
    int rank = 0;
    for (int b = 0; b < 256; b++, rank++)
        if (b != skip_byte)
            body += b64(std::string(1, static_cast<char>(b))) + " " + std::to_string(rank) + "\n";
    for (const auto& t : extra)
        body += b64(t) + " " + std::to_string(rank++) + "\n";
    return body;
}

const std::vector<std::string> kExtra = {"ab", "bc", "abc", "yz", "xy", "pqr"};

constexpr const char* kConfig = R"({"added_tokens_decoder": {
    "262": {"content": "[BOS]", "special": true},
    "264": {"content": "<|im_end|>", "special": true}}})";

struct Files {
    std::string model, config;
    ~Files() {
        std::remove(model.c_str());
        std::remove(config.c_str());
    }
};

bool load(Tokenizer& tok, const std::string& ranks, const std::string& config = kConfig) {
    Files f{temp_path(".model"), temp_path(".json")};
    std::ofstream(f.model) << ranks;
    std::ofstream(f.config) << config;
    return tok.load_tiktoken(f.model, f.config);
}

TEST(TiktokenLoader, MergesTheLowestRankConcatenation) {
    Tokenizer tok;
    ASSERT_TRUE(load(tok, rank_file(kExtra)));
    EXPECT_EQ(tok.type(), "gpt2");
    EXPECT_FALSE(tok.add_bos());
    // "ab" (256) before "bc" (257), then "ab"+"c" is "abc" (258): every split is a merge pair.
    EXPECT_EQ(tok.encode("abcx"), (Ids{258, 'x'}));
    // "yz" (259) outranks "xy" (260): x + yz, though "xy" comes first in the text.
    EXPECT_EQ(tok.encode("xyz"), (Ids{'x', 259}));
    EXPECT_EQ(tok.encode("abc"), (Ids{258}));  // whole pre-token in the ranks
    EXPECT_EQ(tok.encode("bcd"), (Ids{257, 'd'}));
    // A pre-token found whole in the ranks is one token: "pqr" has no merge path ("pq", "qr" absent).
    EXPECT_EQ(tok.encode("pqr"), (Ids{261}));
    EXPECT_EQ(tok.encode("pqrs"), (Ids{'p', 'q', 'r', 's'}));
}

TEST(TiktokenLoader, SpecialsAreAtomicAndReservedSlotsNamed) {
    Tokenizer tok;
    ASSERT_TRUE(load(tok, rank_file(kExtra)));
    EXPECT_EQ(tok.vocab_size(), 262 + 258);  // base + tokenization_kimi.py's 256 + 2 special slots
    EXPECT_EQ(tok.find_token("[BOS]"), 262);
    EXPECT_EQ(tok.find_token("<|reserved_token_263|>"), 263);
    EXPECT_EQ(tok.find_token("<|im_end|>"), 264);
    EXPECT_EQ(tok.find_token("<|reserved_token_519|>"), 519);
    EXPECT_EQ(tok.encode("ab<|im_end|>xyz"), (Ids{256, 264, 'x', 259}));
    EXPECT_EQ(tok.encode("<|reserved_token_263|>[BOS]"), (Ids{263, 262}));
    EXPECT_EQ(tok.decode({264, 256, 263}), "<|im_end|>ab<|reserved_token_263|>");
}

TEST(TiktokenLoader, RawBytesRoundTrip) {
    Tokenizer tok;
    ASSERT_TRUE(load(tok, rank_file(kExtra)));
    for (const std::string s : {"Gr\xc3\xbc\xc3\x9f"
                                "e \xe6\x9d\xb1\xe4\xba\xac abcx",
                                "  \t\n\n xyz 12345 \xf0\x9f\x91\x8d", "\xc3\xa9\x01z"}) {
        const Ids ids = tok.encode(s);
        EXPECT_EQ(tok.decode(ids), s);
    }
}

TEST(TiktokenLoader, RefusesMalformedRankFiles) {
    Tokenizer tok;
    EXPECT_FALSE(load(tok, rank_file(kExtra, /*skip_byte=*/'q')));              // byte without a token
    EXPECT_FALSE(load(tok, rank_file(kExtra) + b64("zz") + " 300\n"));          // rank gap
    EXPECT_FALSE(load(tok, rank_file(kExtra) + b64("ab") + " 262\n"));          // repeated token
    EXPECT_FALSE(load(tok, rank_file(kExtra) + "!!notbase64 262\n"));           // bad base64
    EXPECT_FALSE(load(tok, rank_file(kExtra) + b64("zz") + "\n"));              // no rank
    EXPECT_FALSE(load(tok, rank_file(kExtra) + b64("zz") + " 99999999999\n"));  // rank past kMaxTokenId
    EXPECT_TRUE(load(tok, rank_file(kExtra), "not json"));  // specials optional: reserved names only
    EXPECT_EQ(tok.find_token("<|reserved_token_262|>"), 262);
}

}  // namespace
}  // namespace imp
