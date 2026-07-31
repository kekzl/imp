// Expanding a chat template's single image placeholder to the encoder's token
// count.
//
// A miscount here does not crash: the embedding replacement fills whatever
// placeholders it finds, and every position after the image is shifted, so the
// model reads a coherent prompt that says something else.

#include "model/image_placeholders.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace imp {
namespace {

constexpr int32_t kPad = 151655;  // <|image_pad|>

TEST(ImagePlaceholders, ExpandsTheSinglePlaceholder) {
    std::vector<int32_t> t = {1, 2, kPad, 3, 4};
    std::string err;
    ASSERT_TRUE(expand_image_placeholders(t, kPad, {3}, err)) << err;
    EXPECT_EQ(t, (std::vector<int32_t>{1, 2, kPad, kPad, kPad, 3, 4}));
}

TEST(ImagePlaceholders, ExpandsEachImageToItsOwnCount) {
    std::vector<int32_t> t = {kPad, 7, kPad};
    std::string err;
    ASSERT_TRUE(expand_image_placeholders(t, kPad, {2, 3}, err)) << err;
    EXPECT_EQ(t, (std::vector<int32_t>{kPad, kPad, 7, kPad, kPad, kPad}));
}

TEST(ImagePlaceholders, ACountOfOneLeavesTheSequenceUnchanged) {
    const std::vector<int32_t> before = {5, kPad, 6};
    std::vector<int32_t> t = before;
    std::string err;
    ASSERT_TRUE(expand_image_placeholders(t, kPad, {1}, err)) << err;
    EXPECT_EQ(t, before);
}

TEST(ImagePlaceholders, NoImagesAndNoPlaceholdersIsANoOp) {
    const std::vector<int32_t> before = {1, 2, 3};
    std::vector<int32_t> t = before;
    std::string err;
    ASSERT_TRUE(expand_image_placeholders(t, kPad, {}, err)) << err;
    EXPECT_EQ(t, before);
}

// The two mismatch directions. Either one means the prompt and the encoder
// describe different inputs.
TEST(ImagePlaceholders, RefusesACountMismatch) {
    std::string err;
    std::vector<int32_t> a = {kPad};
    EXPECT_FALSE(expand_image_placeholders(a, kPad, {4, 4}, err));
    EXPECT_NE(err.find("1 image placeholder"), std::string::npos) << err;

    std::vector<int32_t> b = {kPad, kPad};
    err.clear();
    EXPECT_FALSE(expand_image_placeholders(b, kPad, {4}, err));
    EXPECT_NE(err.find("2 image placeholder"), std::string::npos) << err;

    std::vector<int32_t> c = {1, 2};
    err.clear();
    EXPECT_FALSE(expand_image_placeholders(c, kPad, {4}, err));
}

TEST(ImagePlaceholders, RefusesAnImageThatProducedNothing) {
    std::vector<int32_t> t = {kPad};
    std::string err;
    EXPECT_FALSE(expand_image_placeholders(t, kPad, {0}, err));
    EXPECT_NE(err.find("no tokens"), std::string::npos) << err;
    EXPECT_EQ(t.size(), 1u) << "a rejection must leave the sequence alone";
}

// A rejection must not have half-expanded the sequence first.
TEST(ImagePlaceholders, RejectionLeavesTheSequenceUntouched) {
    const std::vector<int32_t> before = {kPad, 9, kPad, 8};
    std::vector<int32_t> t = before;
    std::string err;
    EXPECT_FALSE(expand_image_placeholders(t, kPad, {3}, err));
    EXPECT_EQ(t, before);
}

TEST(ImagePlaceholders, HandlesAdjacentPlaceholders) {
    std::vector<int32_t> t = {kPad, kPad};
    std::string err;
    ASSERT_TRUE(expand_image_placeholders(t, kPad, {2, 2}, err)) << err;
    EXPECT_EQ(t.size(), 4u);
}

// `image_tokens_before` is where a chunk learns which embedding to resume from.
// Off by one here is not a crash — it shifts the whole rest of the image by one
// position, which reads as a slightly wrong answer.
TEST(ImageTokensBefore, CountsOnlyWhatPrecedesTheChunk) {
    //                             0     1     2     3     4     5
    const std::vector<int32_t> t = {9, kPad, kPad, 8, kPad, 7};
    EXPECT_EQ(image_tokens_before(t, kPad, 0), 0) << "the first chunk resumes from nothing";
    EXPECT_EQ(image_tokens_before(t, kPad, 1), 0) << "the boundary token itself is not yet placed";
    EXPECT_EQ(image_tokens_before(t, kPad, 2), 1);
    EXPECT_EQ(image_tokens_before(t, kPad, 3), 2);
    EXPECT_EQ(image_tokens_before(t, kPad, 5), 3);
    EXPECT_EQ(image_tokens_before(t, kPad, 6), 3) << "whole prompt";
}

TEST(ImageTokensBefore, ClampsRatherThanReadingPastTheEnd) {
    const std::vector<int32_t> t = {kPad, 5};
    EXPECT_EQ(image_tokens_before(t, kPad, 99), 1);
    EXPECT_EQ(image_tokens_before(t, kPad, -3), 0);
    EXPECT_EQ(image_tokens_before({}, kPad, 4), 0);
}

TEST(ImageTokensBefore, TextOnlyPromptsResumeFromZero) {
    const std::vector<int32_t> t = {1, 2, 3, 4};
    EXPECT_EQ(image_tokens_before(t, kPad, 4), 0);
}

}  // namespace
}  // namespace imp
