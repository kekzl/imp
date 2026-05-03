#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "compute/json_constrain.h"
#include "compute/constrain_common.h"

#include <string>
#include <cfloat>

#define SKIP_IF_NO_CUDA()                       \
    do {                                        \
        int dev;                                \
        if (cudaGetDevice(&dev) != cudaSuccess) \
            GTEST_SKIP();                       \
    } while (0)

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
// Test 9: GPU mask kernel — constrain_mask_kernel masks invalid tokens
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

    constrain_mask_kernel<<<1, vocab>>>(d_logits, d_cats, d_mask, vocab);
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

}  // namespace
}  // namespace imp
