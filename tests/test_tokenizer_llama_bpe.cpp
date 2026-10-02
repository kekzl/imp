// GGUF tokenizer.ggml.pre aliases: llama-bpe (Llama 3) uses the cl100k regex (#2520), tekken the nemotron
// scan (#2411).
#include "model/tokenizer.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace imp {
namespace {

// GPT-2 bytes_to_unicode: printable Latin-1 bytes map to themselves, the rest to U+0100 + n.
std::string gpt2_byte(int b) {
    static int remap[256] = {};
    static bool init = false;
    if (!init) {
        int n = 0;
        for (int i = 0; i < 256; ++i) {
            const bool keep = (i >= '!' && i <= '~') || (i >= 0xA1 && i <= 0xAC) || (i >= 0xAE && i <= 0xFF);
            remap[i] = keep ? i : 256 + n++;
        }
        init = true;
    }
    const int cp = remap[b];
    if (cp < 0x80)
        return std::string(1, static_cast<char>(cp));
    return {static_cast<char>(0xC0 | (cp >> 6)), static_cast<char>(0x80 | (cp & 0x3F))};
}

// Byte tokens at ids 0..255, then ' '+'[' (256), ' ['+'`' (257), '.'+'s' (258).
Tokenizer make_tokenizer() {
    std::vector<std::string> tokens;
    for (int b = 0; b < 256; ++b)
        tokens.push_back(gpt2_byte(b));
    const std::string sp = gpt2_byte(' ');
    tokens.insert(tokens.end(), {sp + "[", sp + "[`", ".s"});
    Tokenizer tok;
    EXPECT_TRUE(tok.load_vocab(tokens, std::vector<float>(tokens.size(), 0.0f), 0, 0));
    tok.set_type("gpt2");
    tok.set_add_bos(false);
    tok.load_merges({sp + " [", sp + "[ `", ". s"});
    return tok;
}

TEST(TokenizerLlamaBpeTest, UsesCl100kChunks) {
    Tokenizer tok = make_tokenizer();
    for (const char* pre : {"llama-bpe", "llama3", "cl100k"}) {
        tok.set_pre_tokenizer(pre);
        // cl100k " ?[^\s\p{L}\p{N}]+": " [`" is one chunk; "[^\r\n\p{L}\p{N}]?\p{L}+": ".s" is one.
        EXPECT_EQ(tok.encode(" [`"), (std::vector<int32_t>{257})) << pre;
        EXPECT_EQ(tok.encode(".s"), (std::vector<int32_t>{258})) << pre;
    }
}

// GGUF pre "tekken" (Devstral-Small-2, #2411) is the nemotron scan; the gpt2 fallback split "_case"
// and "/," apart (592 of 1202 parity records differed). Chunks: HF tokenizers on tokenizer.json.
TEST(TokenizerLlamaBpeTest, TekkenUsesNemotronChunks) {
    Tokenizer tok;
    tok.set_pre_tokenizer("tekken");
    EXPECT_EQ(tok.pre_tokenizer(), "nemotron");
    using Chunks = std::vector<std::string>;
    EXPECT_EQ(nemotron_pre_tokenize("snake_case and camelCase"),
              (Chunks{"snake", "_case", " and", " camel", "Case"}));
    EXPECT_EQ(nemotron_pre_tokenize("X/,c"), (Chunks{"X", "/,", "c"}));
}

}  // namespace
}  // namespace imp
