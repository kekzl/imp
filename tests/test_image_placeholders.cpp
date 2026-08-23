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
    const auto r = expand_image_placeholders(t, kPad, {3});
    ASSERT_TRUE(r) << r.error();
    EXPECT_EQ(t, (std::vector<int32_t>{1, 2, kPad, kPad, kPad, 3, 4}));
}

TEST(ImagePlaceholders, ExpandsEachImageToItsOwnCount) {
    std::vector<int32_t> t = {kPad, 7, kPad};
    const auto r = expand_image_placeholders(t, kPad, {2, 3});
    ASSERT_TRUE(r) << r.error();
    EXPECT_EQ(t, (std::vector<int32_t>{kPad, kPad, 7, kPad, kPad, kPad}));
}

TEST(ImagePlaceholders, ACountOfOneLeavesTheSequenceUnchanged) {
    const std::vector<int32_t> before = {5, kPad, 6};
    std::vector<int32_t> t = before;
    const auto r = expand_image_placeholders(t, kPad, {1});
    ASSERT_TRUE(r) << r.error();
    EXPECT_EQ(t, before);
}

TEST(ImagePlaceholders, NoImagesAndNoPlaceholdersIsANoOp) {
    const std::vector<int32_t> before = {1, 2, 3};
    std::vector<int32_t> t = before;
    const auto r = expand_image_placeholders(t, kPad, {});
    ASSERT_TRUE(r) << r.error();
    EXPECT_EQ(t, before);
}

// The two mismatch directions. Either one means the prompt and the encoder
// describe different inputs.
TEST(ImagePlaceholders, RefusesACountMismatch) {
    std::vector<int32_t> a = {kPad};
    const auto ra = expand_image_placeholders(a, kPad, {4, 4});
    ASSERT_FALSE(ra.has_value());
    EXPECT_NE(ra.error().find("1 image placeholder"), std::string::npos) << ra.error();

    std::vector<int32_t> b = {kPad, kPad};
    const auto rb = expand_image_placeholders(b, kPad, {4});
    ASSERT_FALSE(rb.has_value());
    EXPECT_NE(rb.error().find("2 image placeholder"), std::string::npos) << rb.error();

    std::vector<int32_t> c = {1, 2};
    EXPECT_FALSE(expand_image_placeholders(c, kPad, {4}).has_value());
}

TEST(ImagePlaceholders, RefusesAnImageThatProducedNothing) {
    std::vector<int32_t> t = {kPad};
    const auto r = expand_image_placeholders(t, kPad, {0});
    ASSERT_FALSE(r.has_value());
    EXPECT_NE(r.error().find("no tokens"), std::string::npos) << r.error();
    EXPECT_EQ(t.size(), 1u) << "a rejection must leave the sequence alone";
}

// A rejection must not have half-expanded the sequence first.
TEST(ImagePlaceholders, RejectionLeavesTheSequenceUntouched) {
    const std::vector<int32_t> before = {kPad, 9, kPad, 8};
    std::vector<int32_t> t = before;
    EXPECT_FALSE(expand_image_placeholders(t, kPad, {3}).has_value());
    EXPECT_EQ(t, before);
}

TEST(ImagePlaceholders, HandlesAdjacentPlaceholders) {
    std::vector<int32_t> t = {kPad, kPad};
    const auto r = expand_image_placeholders(t, kPad, {2, 2});
    ASSERT_TRUE(r) << r.error();
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

// The prefix cache asks one question — "same tokens and same pictures?" — so a
// request carrying several images folds them into one salt. Order has to matter:
// two prompts differing only in which picture comes first are different prompts.
TEST(CombineImageHash, IsOrderSensitiveAndNeverZero) {
    const size_t a = 0x1111, b = 0x2222;
    const size_t ab = combine_image_hash(combine_image_hash(0, a), b);
    const size_t ba = combine_image_hash(combine_image_hash(0, b), a);
    EXPECT_NE(ab, ba) << "swapping two images must change the cache key";
    EXPECT_NE(ab, 0u) << "0 is the cache's 'no image' sentinel";

    EXPECT_EQ(combine_image_hash(0, a), a) << "one image hashes to itself";
    EXPECT_NE(combine_image_hash(combine_image_hash(0, a), a), a) << "the same picture twice is not once";
}

TEST(ImageTokensBefore, TextOnlyPromptsResumeFromZero) {
    const std::vector<int32_t> t = {1, 2, 3, 4};
    EXPECT_EQ(image_tokens_before(t, kPad, 4), 0);
}

}  // namespace
}  // namespace imp
