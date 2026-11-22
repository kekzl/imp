// Thinking blocks of a streamed /v1/messages response (tools/imp-server/anthropic.h).

#include "anthropic.h"

#include <gtest/gtest.h>

#include <string>

// A response with two thinking blocks (thinking, tool calls, thinking): each block's signature is the
// digest of its own text, or the server refuses its own second block on the next request (HTTP 400).
TEST(AnthropicThinking, EveryStreamedBlockIsSignedOverItsOwnText) {
    using imp_server::anthropic::thinking_signature;
    imp_server::anthropic::ThinkingBlockText block;
    block.add("Let me look ");
    block.add("around first.");
    EXPECT_EQ(block.close(), thinking_signature("Let me look around first."));
    EXPECT_TRUE(block.empty());
    block.add("The second call was a find.");
    EXPECT_EQ(block.close(), thinking_signature("The second call was a find."));
}
