// Pictures keep their place in the prompt: the parser's mark k becomes block k where the part sat.

#include "vision_inline.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

using imp::ChatMessage;
using imp::server::count_vision_marks;
using imp::server::kVisionPartMark;
using imp::server::place_vision_blocks;
using imp::server::strip_vision_marks;

namespace {

std::string mark() { return std::string(kVisionPartMark); }

}  // namespace

TEST(ServerVisionInline, BlocksLandWhereThePartsSat) {
    std::vector<ChatMessage> msgs = {
        {"system", "sys"},
        {"user", "look: " + mark() + " and " + mark()},
        {"assistant", "ok"},
        {"user", mark()},
    };
    const std::vector<std::string> blocks = {"<A>", "<B>", "<C>"};
    ASSERT_EQ(count_vision_marks(msgs), 3u);
    ASSERT_TRUE(place_vision_blocks(msgs, blocks));
    EXPECT_EQ(msgs[1].content, "look: <A> and <B>");
    EXPECT_EQ(msgs[3].content, "<C>");
    EXPECT_EQ(count_vision_marks(msgs), 0u);
}

// A later screenshot leaves every earlier turn's text as it was: the prefix cache keys on it.
TEST(ServerVisionInline, AppendedPictureKeepsEarlierTurns) {
    std::vector<ChatMessage> one = {{"user", "first"}, {"user", mark()}};
    std::vector<ChatMessage> two = {{"user", "first"},
                                    {"user", mark()},
                                    {"assistant", "call"},
                                    {"user", mark()}};
    ASSERT_TRUE(place_vision_blocks(one, std::vector<std::string>{"<P>"}));
    ASSERT_TRUE(place_vision_blocks(two, std::vector<std::string>{"<P>", "<P>"}));
    EXPECT_EQ(one[0].content, two[0].content);
    EXPECT_EQ(one[1].content, two[1].content);
}

TEST(ServerVisionInline, CountMismatchStripsAndReportsFalse) {
    std::vector<ChatMessage> msgs = {{"user", "a" + mark() + "b" + mark() + "c"}};
    EXPECT_FALSE(place_vision_blocks(msgs, std::vector<std::string>{"<A>"}));
    EXPECT_EQ(msgs[0].content, "abc");
    std::vector<ChatMessage> none = {{"user", "plain"}};
    EXPECT_FALSE(place_vision_blocks(none, std::vector<std::string>{}));
    EXPECT_EQ(none[0].content, "plain");
}

TEST(ServerVisionInline, StripRemovesEveryMark) {
    std::vector<ChatMessage> msgs = {{"user", mark() + mark() + "x" + mark()}, {"tool", "t"}};
    strip_vision_marks(msgs);
    EXPECT_EQ(msgs[0].content, "x");
    EXPECT_EQ(msgs[1].content, "t");
}

// A block that itself contains the mark bytes cannot loop the replace.
TEST(ServerVisionInline, BlockContainingTheMarkTerminates) {
    std::vector<ChatMessage> msgs = {{"user", mark()}};
    ASSERT_TRUE(place_vision_blocks(msgs, std::vector<std::string>{"[" + mark() + "]"}));
    EXPECT_EQ(msgs[0].content, "[" + mark() + "]");
}
