// OpenAI wire: pictures inside role "tool" messages move to the user turn behind them.

#include "tool_result_images.h"

#include <gtest/gtest.h>

#include <string>

using json = nlohmann::ordered_json;
using imp::server::rehome_tool_images;

namespace {

json img(const char* url) { return {{"type", "image_url"}, {"image_url", {{"url", url}}}}; }
json txt(const char* t) { return {{"type", "text"}, {"text", t}}; }
const char* kNote1 = "[1 image(s) from this tool result follow in the next user message]";

}  // namespace

TEST(ToolResultImages, APictureMovesToANewUserTurnBehindTheToolRun) {
    const json in = json::array({
        {{"role", "user"}, {"content", "take a screenshot"}},
        {{"role", "assistant"}, {"content", ""}},
        {{"role", "tool"}, {"tool_call_id", "a"}, {"content", json::array({txt("done"), img("data:x")})}},
        {{"role", "tool"}, {"tool_call_id", "b"}, {"content", "plain"}},
    });
    int moved = 0;
    const json out = rehome_tool_images(in, &moved);
    ASSERT_EQ(out.size(), 5u);
    EXPECT_EQ(moved, 1);
    EXPECT_EQ(out[2]["content"], std::string("done\n") + kNote1);
    EXPECT_EQ(out[2]["tool_call_id"], "a");
    EXPECT_EQ(out[3], in[3]);  // the run of tool messages stays together
    EXPECT_EQ(out[4]["role"], "user");
    EXPECT_EQ(out[4]["content"], json::array({img("data:x")}));
}

TEST(ToolResultImages, PicturesGoInFrontOfTheNextUserTurn) {
    const json in = json::array({
        {{"role", "tool"}, {"tool_call_id", "a"}, {"content", json::array({img("data:1"), img("data:2")})}},
        {{"role", "user"}, {"content", "what do you see?"}},
        {{"role", "tool"}, {"tool_call_id", "b"}, {"content", json::array({img("data:3"), txt("t")})}},
        {{"role", "user"}, {"content", json::array({txt("and now?")})}},
    });
    int moved = 0;
    const json out = rehome_tool_images(in, &moved);
    ASSERT_EQ(out.size(), 4u);
    EXPECT_EQ(moved, 3);
    EXPECT_EQ(out[0]["content"], "[2 image(s) from this tool result follow in the next user message]");
    EXPECT_EQ(out[1]["content"], json::array({img("data:1"), img("data:2"), txt("what do you see?")}));
    EXPECT_EQ(out[2]["content"], std::string("t\n") + kNote1);
    EXPECT_EQ(out[3]["content"], json::array({img("data:3"), txt("and now?")}));
}

TEST(ToolResultImages, AnAssistantTurnBehindTheToolRunGetsTheUserTurnInFront) {
    const json in = json::array({
        {{"role", "tool"}, {"tool_call_id", "a"}, {"content", json::array({img("data:1")})}},
        {{"role", "assistant"}, {"content", "I see it"}},
    });
    const json out = rehome_tool_images(in);
    ASSERT_EQ(out.size(), 3u);
    EXPECT_EQ(out[1]["role"], "user");
    EXPECT_EQ(out[1]["content"], json::array({img("data:1")}));
    EXPECT_EQ(out[2], in[1]);
}

TEST(ToolResultImages, MessagesWithoutToolPicturesComeBackUnchanged) {
    const json in = json::array({
        {{"role", "system"}, {"content", "s"}},
        {{"role", "user"}, {"content", json::array({img("data:u"), txt("mine")})}},
        {{"role", "tool"}, {"tool_call_id", "a"}, {"content", json::array({txt("a"), txt("b")})}},
        {{"role", "tool"}, {"tool_call_id", "b"}, {"content", {{"k", 1}}}},
        {{"role", "assistant"}, {"content", nullptr}},
    });
    int moved = -1;
    EXPECT_EQ(rehome_tool_images(in, &moved), in);
    EXPECT_EQ(moved, 0);
    EXPECT_EQ(rehome_tool_images(json("not an array")), json("not an array"));
}
