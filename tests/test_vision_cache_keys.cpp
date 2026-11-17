#include "runtime/vision_cache_keys.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

using imp::vision_cache_key_tokens;

namespace {

constexpr int32_t kPad = 151655;  // Qwen <|image_pad|>
constexpr int32_t kStart = 151652, kEnd = 151653;

// text(4) [start pad*3 end] text(2) [start pad*2 end] text(1)
std::vector<int32_t> two_image_prompt() {
    return {10, 11, 12, 13, kStart, kPad, kPad, kPad, kEnd, 20, 21, kStart, kPad, kPad, kEnd, 30};
}

}  // namespace

TEST(VisionCacheKeys, TextBeforeTheFirstPictureKeysAsPlainText) {
    const auto tokens = two_image_prompt();
    const std::vector<size_t> parts = {0xA, 0xB};
    const auto keys = vision_cache_key_tokens(tokens, kPad, parts, 0xAB);
    ASSERT_EQ(keys.size(), tokens.size());
    for (size_t i = 0; i < 5; ++i)
        EXPECT_EQ(keys[i], tokens[i]) << i;
    for (size_t i : {5u, 6u, 7u, 12u, 13u})
        EXPECT_LT(keys[i], 0) << "a placeholder key must never equal a token id, at " << i;
    EXPECT_NE(keys[5], keys[6]) << "placeholders of one picture key by position";
}

TEST(VisionCacheKeys, NewSecondPictureLeavesTheFirstPicturesKeys) {
    const auto tokens = two_image_prompt();
    const std::vector<size_t> old_parts = {0xA, 0xB}, new_parts = {0xA, 0xC};
    const auto a = vision_cache_key_tokens(tokens, kPad, old_parts, 0xAB);
    const auto b = vision_cache_key_tokens(tokens, kPad, new_parts, 0xAC);
    for (size_t i = 0; i < 12; ++i)
        EXPECT_EQ(a[i], b[i]) << "keys up to the changed picture must match, at " << i;
    EXPECT_NE(a[12], b[12]) << "the changed picture must change its keys";
}

TEST(VisionCacheKeys, OtherFirstPictureChangesKeysFromItsPlaceholders) {
    const auto tokens = two_image_prompt();
    const std::vector<size_t> p1 = {0xA, 0xB}, p2 = {0xD, 0xB};
    const auto a = vision_cache_key_tokens(tokens, kPad, p1, 1);
    const auto b = vision_cache_key_tokens(tokens, kPad, p2, 2);
    EXPECT_NE(a[5], b[5]);
}

TEST(VisionCacheKeys, CountMismatchKeysEveryRunOnTheWholeHash) {
    const auto tokens = two_image_prompt();
    const std::vector<size_t> one_part = {0xA};  // two runs, one part: mapping unknown
    const auto keys = vision_cache_key_tokens(tokens, kPad, one_part, 0xAB);
    const std::vector<size_t> none;
    const auto whole = vision_cache_key_tokens(tokens, kPad, none, 0xAB);
    EXPECT_EQ(keys, whole);
    const auto other_whole = vision_cache_key_tokens(tokens, kPad, none, 0xAC);
    EXPECT_NE(keys[5], other_whole[5]);
    for (size_t i = 0; i < 5; ++i)
        EXPECT_EQ(keys[i], tokens[i]);
}

TEST(VisionCacheKeys, NoPlaceholderMeansNoKeyTokens) {
    const std::vector<int32_t> tokens = {1, 2, 3};
    const std::vector<size_t> parts = {0xA};
    EXPECT_TRUE(vision_cache_key_tokens(tokens, kPad, parts, 0xA).empty());
    EXPECT_TRUE(vision_cache_key_tokens(two_image_prompt(), -1, parts, 0xA).empty());
}
