// CPU unit test for split_prompt_usage (anthropic.h): the three prompt fields are disjoint (#3055).

#include "anthropic.h"

#include <gtest/gtest.h>

namespace {

TEST(AnthropicCacheUsage, SplitSumsToPromptOnEveryTurnShape) {
    struct Case {
        int prompt, cached, creation, input, read, written;
    };
    const Case cases[] = {
        {1550, 0, 1536, 14, 0, 1536},    // cold pinned turn: block-rounded write, tail stays input
        {1600, 1536, 48, 16, 1536, 48},  // warm turn
        {100, 0, 0, 100, 0, 0},          // nothing pinned
        {64, 64, 0, 0, 64, 0},           // full hit
        {40, 500, 0, 0, 40, 0},          // cached past the prompt
        {100, 90, 50, 0, 90, 10},        // creation past what is left
        {100, -1, -1, 100, 0, 0},        // negative counters
    };
    for (const Case& c : cases) {
        const auto s = imp_server::anthropic::split_prompt_usage(c.prompt, c.cached, c.creation);
        EXPECT_EQ(s.input, c.input) << c.prompt << "/" << c.cached << "/" << c.creation;
        EXPECT_EQ(s.cache_read, c.read) << c.prompt << "/" << c.cached << "/" << c.creation;
        EXPECT_EQ(s.cache_creation, c.written) << c.prompt << "/" << c.cached << "/" << c.creation;
        EXPECT_EQ(s.input + s.cache_read + s.cache_creation, c.prompt);
    }
}

}  // namespace
