#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "compute/json_constrain.h"
#include "compute/constrain_common.h"
#include "compute/preamble_gate.h"
#include "model/tokenizer.h"

#include <string>
#include <cfloat>

#include "test_cuda_skip.h"

namespace imp {
namespace {

// ===========================================================================
// Test 1: classify_token("0") includes NUMBER category
// ===========================================================================
TEST(JsonConstrainTest, ClassifyDigit) {
    uint16_t cat = classify_token("0");
    EXPECT_TRUE(cat & CAT_NUMBER_START) << "Digit '0' should be classified as NUMBER_START";
    EXPECT_TRUE(cat & CAT_NUMBER_CONT) << "Digit '0' should be classified as NUMBER_CONT";
}

// ===========================================================================
// Test 2: classify_token("{") returns OPEN_BRACE
// ===========================================================================
TEST(JsonConstrainTest, ClassifyBrace) {
    uint16_t cat = classify_token("{");
    EXPECT_TRUE(cat & CAT_OPEN_BRACE) << "'{' should be classified as OPEN_BRACE";
    EXPECT_FALSE(cat & CAT_CLOSE_BRACE) << "'{' should NOT be CLOSE_BRACE";

    uint16_t cat2 = classify_token("}");
    EXPECT_TRUE(cat2 & CAT_CLOSE_BRACE) << "'}' should be classified as CLOSE_BRACE";
}

// ===========================================================================
// Test 3: classify_token('"') returns QUOTE
// ===========================================================================
TEST(JsonConstrainTest, ClassifyQuote) {
    uint16_t cat = classify_token("\"");
    EXPECT_TRUE(cat & CAT_QUOTE) << "'\"' should be classified as QUOTE";
}

// ===========================================================================
// Test 4: classify_token for multi-char string content
// ===========================================================================
TEST(JsonConstrainTest, ClassifyStringContent) {
    // "hello" is all lowercase printable — should be STRING_CHAR
    uint16_t cat = classify_token("hello");
    EXPECT_TRUE(cat & CAT_STRING_CHAR) << "'hello' should be classified as STRING_CHAR";
    // Also a literal continuation (all lowercase)
    EXPECT_TRUE(cat & CAT_LITERAL_CONT) << "'hello' (all lowercase) should also be LITERAL_CONT";
}

// ===========================================================================
// Test 5: classify_token for structural tokens
// ===========================================================================
TEST(JsonConstrainTest, ClassifyStructural) {
    EXPECT_TRUE(classify_token("[") & CAT_OPEN_BRACKET);
    EXPECT_TRUE(classify_token("]") & CAT_CLOSE_BRACKET);
    EXPECT_TRUE(classify_token(":") & CAT_COLON);
    EXPECT_TRUE(classify_token(",") & CAT_COMMA);
}

// ===========================================================================
// Test 6: classify_token for literal starts
// ===========================================================================
TEST(JsonConstrainTest, ClassifyLiteralStarts) {
    EXPECT_TRUE(classify_token("t") & CAT_TRUE_START);
    EXPECT_TRUE(classify_token("f") & CAT_FALSE_START);
    EXPECT_TRUE(classify_token("n") & CAT_NULL_START);
    EXPECT_TRUE(classify_token("true") & CAT_TRUE_START);
    EXPECT_TRUE(classify_token("false") & CAT_FALSE_START);
    EXPECT_TRUE(classify_token("null") & CAT_NULL_START);
}

// ===========================================================================
// Test 7: classify_token for whitespace
// ===========================================================================
TEST(JsonConstrainTest, ClassifyWhitespace) {
    EXPECT_TRUE(classify_token(" ") & CAT_WHITESPACE);
    EXPECT_TRUE(classify_token("\n") & CAT_WHITESPACE);
    EXPECT_TRUE(classify_token("") & CAT_WHITESPACE);  // empty = whitespace
}

// ===========================================================================
// Test 8: classify_token for number patterns
// ===========================================================================
TEST(JsonConstrainTest, ClassifyNumbers) {
    EXPECT_TRUE(classify_token("-") & CAT_NUMBER_START);
    EXPECT_TRUE(classify_token("123") & CAT_NUMBER_START);
    EXPECT_TRUE(classify_token("123") & CAT_NUMBER_CONT);
    EXPECT_TRUE(classify_token(".") & CAT_NUMBER_CONT);
}

// ===========================================================================
// Test 9: GPU mask kernel - constrain_mask_allow_kernel (category mask only) masks invalid tokens
// ===========================================================================
TEST(JsonConstrainTest, MaskAllowsValidTokens) {
    SKIP_IF_NO_CUDA();

    // Simulate 4 tokens: "{", "hello", "0", "}"
    // with allowed_mask = CAT_OPEN_BRACE | CAT_OPEN_BRACKET (START state)
    constexpr int vocab = 4;
    uint16_t h_cats[vocab] = {
        CAT_OPEN_BRACE,                                             // "{"
        CAT_STRING_CHAR,                                            // "hello"
        static_cast<uint16_t>(CAT_NUMBER_START | CAT_NUMBER_CONT),  // "0"
        CAT_CLOSE_BRACE                                             // "}"
    };
    uint16_t h_mask = CAT_OPEN_BRACE | CAT_OPEN_BRACKET | CAT_WHITESPACE;

    float h_logits[vocab] = {1.0f, 2.0f, 3.0f, 4.0f};

    // Upload to device
    uint16_t *d_cats, *d_mask;
    float* d_logits;
    cudaMalloc(&d_cats, vocab * sizeof(uint16_t));
    cudaMalloc(&d_mask, sizeof(uint16_t));
    cudaMalloc(&d_logits, vocab * sizeof(float));

    cudaMemcpy(d_cats, h_cats, vocab * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, &h_mask, sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_logits, h_logits, vocab * sizeof(float), cudaMemcpyHostToDevice);

    constrain_mask_allow_kernel<<<1, vocab>>>(d_logits, d_cats, nullptr, d_mask, vocab, vocab, false);
    cudaDeviceSynchronize();

    float h_out[vocab];
    cudaMemcpy(h_out, d_logits, vocab * sizeof(float), cudaMemcpyDeviceToHost);

    // Token 0 ("{") matches CAT_OPEN_BRACE -> should be untouched
    EXPECT_FLOAT_EQ(h_out[0], 1.0f);
    // Token 1 ("hello"), 2 ("0"), 3 ("}") don't match -> should be -FLT_MAX
    EXPECT_FLOAT_EQ(h_out[1], -FLT_MAX);
    EXPECT_FLOAT_EQ(h_out[2], -FLT_MAX);
    EXPECT_FLOAT_EQ(h_out[3], -FLT_MAX);

    cudaFree(d_cats);
    cudaFree(d_mask);
    cudaFree(d_logits);
}

TEST(JsonConstrainTest, RawControlCharInStringMasked) {
    SKIP_IF_NO_CUDA();

    //          0        1      2       3    4     5      6
    std::vector<std::string> toks = {"<unk>", "<s>", "</s>", "{", "\"", "\"\n", "\"ok"};
    std::vector<float> scores(toks.size(), 0.0f);
    Tokenizer tok;
    EXPECT_TRUE(tok.load_vocab(toks, scores, /*bos_id=*/1, /*eos_id=*/2));

    JsonConstrainer jc;
    ASSERT_TRUE(jc.init(tok));
    jc.update(3);  // '{' → OBJECT_START, next token may open a key string

    const int vocab = static_cast<int>(toks.size());
    std::vector<float> h(vocab, 1.0f);
    float* d = nullptr;
    cudaMalloc(&d, vocab * sizeof(float));
    cudaMemcpy(d, h.data(), vocab * sizeof(float), cudaMemcpyHostToDevice);
    jc.apply_mask(d, vocab, 0);
    cudaDeviceSynchronize();
    cudaMemcpy(h.data(), d, vocab * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d);

    EXPECT_GT(h[6], -1e30f) << "'\"ok' (quote + printable key chars) must be allowed";
    EXPECT_FLOAT_EQ(h[5], -FLT_MAX) << "'\"<newline>' must be masked — raw control char in string";
}

TEST(JsonConstrainTest, ModelVocabLargerThanTokenizerMasksPadding) {
    SKIP_IF_NO_CUDA();

    //          0        1      2       3    4     5    6
    std::vector<std::string> toks = {"<unk>", "<s>", "</s>", "{", "\"", "}", "0"};
    std::vector<float> scores(toks.size(), 0.0f);
    Tokenizer tok;
    EXPECT_TRUE(tok.load_vocab(toks, scores, /*bos_id=*/1, /*eos_id=*/2));

    JsonConstrainer jc;
    ASSERT_TRUE(jc.init(tok));

    const int tok_vocab = static_cast<int>(toks.size());
    const int model_vocab = tok_vocab + 9;  // simulated lm_head padding rows

    std::vector<float> h(model_vocab, 1.0f);
    float* d = nullptr;
    cudaMalloc(&d, model_vocab * sizeof(float));
    cudaMemcpy(d, h.data(), model_vocab * sizeof(float), cudaMemcpyHostToDevice);
    jc.apply_mask(d, model_vocab, 0);
    cudaDeviceSynchronize();
    cudaMemcpy(h.data(), d, model_vocab * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d);

    // '{' starts a JSON document — must stay alive.
    EXPECT_GT(h[3], -1e30f) << "'{' must be allowed at document start";
    // Every padding id (no tokenizer entry) must be masked.
    for (int i = tok_vocab; i < model_vocab; i++) {
        EXPECT_FLOAT_EQ(h[i], -FLT_MAX) << "padding id " << i << " leaked through the json mask";
    }
}

// #1104: a constrainer that cannot mask must never decode SILENTLY unconstrained. The device
// allow list used to allocate lazily inside apply_mask() mid-decode; on a model loading with
// no free VRAM (#1103) that allocation failed and apply_mask returned without masking or
// logging, so the first constrained request per process came back as prose - deterministic,
// byte-identical across restarts, and invisible in imp_constrained_eager_fallback_total since
// no fallback was taken. Pins that the allow list is already resident right after init()
// reports success, making the failure impossible rather than unlikely.
TEST(JsonConstrainTest, InitReservesTheAllowListSoApplyMaskNeverAllocates) {
    SKIP_IF_NO_CUDA();

    std::vector<std::string> toks = {"<unk>", "<s>", "</s>", "{", "\"", "}", "0"};
    std::vector<float> scores(toks.size(), 0.0f);
    Tokenizer tok;
    EXPECT_TRUE(tok.load_vocab(toks, scores, /*bos_id=*/1, /*eos_id=*/2));

    JsonConstrainer jc;
    ASSERT_TRUE(jc.init(tok));
    EXPECT_TRUE(jc.has_device_allow_list())
        << "init() succeeded without reserving the allow list — apply_mask would allocate on the "
           "serving path, and a failure there masks nothing at all (issue #1104)";

    // And the very first apply_mask already constrains: no warm-up call needed.
    const int vocab = static_cast<int>(toks.size());
    std::vector<float> h(vocab, 1.0f);
    float* d = nullptr;
    ASSERT_EQ(cudaMalloc(&d, vocab * sizeof(float)), cudaSuccess);
    cudaMemcpy(d, h.data(), vocab * sizeof(float), cudaMemcpyHostToDevice);
    jc.apply_mask(d, vocab, 0);
    cudaDeviceSynchronize();
    cudaMemcpy(h.data(), d, vocab * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d);

    EXPECT_GT(h[3], -1e30f) << "'{' must open the document on the FIRST masked step";
    EXPECT_FLOAT_EQ(h[5], -FLT_MAX) << "'}' cannot open a document — the first step must already mask";
}

}  // namespace
}  // namespace imp
