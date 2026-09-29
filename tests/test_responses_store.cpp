// Responses store (#2206): TTL, LRU entry cap, byte cap (responses_store.h), and the
// previous_response_id merge (responses.h continue_conversation). Fake clock, no sleeps.

#include <gtest/gtest.h>
#include "responses.h"
#include "responses_store.h"

#include <cstdint>
#include <string>

using imp_server::responses::continue_conversation;
using imp_server::responses::json;
using imp_server::responses::normalize_input_items;
using imp_server::responses::responses_to_openai_body;
using imp_server::responses::ResponseStore;
using imp_server::responses::ResponseStoreLimits;
using imp_server::responses::StoredResponse;

namespace {

struct FakeClock {
    int64_t ms = 1000;
};

StoredResponse entry(const std::string& text) {
    StoredResponse e;
    e.input_items = json::array({{{"role", "user"}, {"content", text}}});
    e.output_items = json::array({{{"type", "message"},
                                   {"role", "assistant"},
                                   {"content", json::array({{{"type", "output_text"}, {"text", text}}})}}});
    e.response = {{"id", "r"}, {"output", e.output_items}};
    e.model = "m";
    return e;
}

ResponseStore make_store(FakeClock& clk, int64_t ttl_s, size_t entries, size_t bytes) {
    ResponseStoreLimits lim;
    lim.ttl_seconds = ttl_s;
    lim.max_entries = entries;
    lim.max_bytes = bytes;
    return ResponseStore(lim, [&clk] { return clk.ms; });
}

}  // namespace

TEST(ResponseStore, PutGetRoundTrip) {
    FakeClock clk;
    auto s = make_store(clk, 60, 10, 1 << 20);
    ASSERT_TRUE(s.put("a", entry("hello")));
    auto got = s.get("a");
    ASSERT_NE(got, nullptr);
    EXPECT_EQ(got->output_items[0]["content"][0]["text"], "hello");
    EXPECT_EQ(s.get("missing"), nullptr);
    EXPECT_EQ(s.size(), 1u);
    EXPECT_EQ(s.bytes(), ResponseStore::entry_bytes("a", entry("hello")));
}

TEST(ResponseStore, TtlExpiresFromInsertion) {
    FakeClock clk;
    auto s = make_store(clk, 10, 10, 1 << 20);
    ASSERT_TRUE(s.put("a", entry("x")));
    clk.ms += 9999;
    ASSERT_NE(s.get("a"), nullptr);  // a hit does not extend the TTL
    clk.ms += 1;
    EXPECT_EQ(s.get("a"), nullptr);
    EXPECT_EQ(s.size(), 0u);
    EXPECT_EQ(s.bytes(), 0u);
    EXPECT_EQ(s.expirations(), 1u);
    EXPECT_EQ(s.evictions(), 0u);
}

TEST(ResponseStore, PurgeExpiredDropsWithoutAccess) {
    FakeClock clk;
    auto s = make_store(clk, 5, 10, 1 << 20);
    ASSERT_TRUE(s.put("a", entry("x")));
    ASSERT_TRUE(s.put("b", entry("y")));
    clk.ms += 5000;
    s.purge_expired();
    EXPECT_EQ(s.size(), 0u);
    EXPECT_EQ(s.expirations(), 2u);
}

TEST(ResponseStore, EntryCapEvictsLeastRecentlyUsed) {
    FakeClock clk;
    auto s = make_store(clk, 60, 2, 1 << 20);
    ASSERT_TRUE(s.put("a", entry("1")));
    ASSERT_TRUE(s.put("b", entry("2")));
    ASSERT_NE(s.get("a"), nullptr);  // a is now most recent, b least
    ASSERT_TRUE(s.put("c", entry("3")));
    EXPECT_EQ(s.size(), 2u);
    EXPECT_EQ(s.evictions(), 1u);
    EXPECT_NE(s.get("a"), nullptr);
    EXPECT_EQ(s.get("b"), nullptr);
    EXPECT_NE(s.get("c"), nullptr);
}

TEST(ResponseStore, ByteCapEvictsUntilUnderCap) {
    FakeClock clk;
    const size_t one = ResponseStore::entry_bytes("a", entry("payload"));
    auto s = make_store(clk, 60, 100, one * 2 + one / 2);  // room for two
    ASSERT_TRUE(s.put("a", entry("payload")));
    ASSERT_TRUE(s.put("b", entry("payload")));
    ASSERT_TRUE(s.put("c", entry("payload")));
    EXPECT_EQ(s.size(), 2u);
    EXPECT_LE(s.bytes(), one * 2 + one / 2);
    EXPECT_EQ(s.evictions(), 1u);
    EXPECT_EQ(s.get("a"), nullptr);
}

TEST(ResponseStore, OversizeEntryIsRefusedAndStoresNothing) {
    FakeClock clk;
    auto s = make_store(clk, 60, 10, 16);
    EXPECT_FALSE(s.put("a", entry("far more than sixteen bytes once serialized")));
    EXPECT_EQ(s.size(), 0u);
    EXPECT_EQ(s.bytes(), 0u);
}

TEST(ResponseStore, AnyZeroLimitDisables) {
    FakeClock clk;
    for (auto lim : {ResponseStoreLimits{0, 10, 1 << 20}, ResponseStoreLimits{60, 0, 1 << 20},
                     ResponseStoreLimits{60, 10, 0}}) {
        ResponseStore s(lim, [&clk] { return clk.ms; });
        EXPECT_FALSE(s.enabled());
        EXPECT_FALSE(s.put("a", entry("x")));
        EXPECT_EQ(s.size(), 0u);
    }
    EXPECT_TRUE(ResponseStoreLimits{}.enabled());
    EXPECT_EQ(ResponseStoreLimits{}.ttl_seconds, 3600);
    EXPECT_EQ(ResponseStoreLimits{}.max_entries, 1000u);
    EXPECT_EQ(ResponseStoreLimits{}.max_bytes, size_t{256} << 20);
}

TEST(ResponseStore, ReplaceAndEraseKeepByteAccounting) {
    FakeClock clk;
    auto s = make_store(clk, 60, 10, 1 << 20);
    ASSERT_TRUE(s.put("a", entry("short")));
    ASSERT_TRUE(s.put("a", entry("a longer payload")));
    EXPECT_EQ(s.size(), 1u);
    EXPECT_EQ(s.bytes(), ResponseStore::entry_bytes("a", entry("a longer payload")));
    EXPECT_TRUE(s.erase("a"));
    EXPECT_FALSE(s.erase("a"));
    EXPECT_EQ(s.bytes(), 0u);
    EXPECT_EQ(s.evictions(), 0u);
}

// previous_response_id must produce the messages a stateless client gets by resending the
// transcript (input + output items + new input); this is what makes the tokens identical.
TEST(ResponsesContinue, MergedInputEqualsStatelessResend) {
    const json turn1_input = "What is 2+2?";
    const json turn1_output = json::array(
        {{{"type", "reasoning"},
          {"id", "rs_1"},
          {"summary", json::array({{{"type", "summary_text"}, {"text", "add"}}})}},
         {{"type", "message"},
          {"id", "msg_1"},
          {"role", "assistant"},
          {"content", json::array({{{"type", "output_text"}, {"text", "4"}}})}},
         {{"type", "function_call"},
          {"id", "fc_1"},
          {"call_id", "call_1"},
          {"name", "calc"},
          {"arguments", "{\"x\":1}"}}});
    const json turn2_input = json::array(
        {{{"type", "function_call_output"}, {"call_id", "call_1"}, {"output", "ok"}},
         {{"role", "user"}, {"content", "And 3+3?"}}});

    const json merged = continue_conversation(normalize_input_items(turn1_input), turn1_output, turn2_input);

    json stateless_input = json::array({{{"role", "user"}, {"content", "What is 2+2?"}}});
    for (const auto& it : turn1_output)
        stateless_input.push_back(it);
    for (const auto& it : turn2_input)
        stateless_input.push_back(it);

    const json a = responses_to_openai_body(json{{"instructions", "be brief"}, {"input", merged}});
    const json b = responses_to_openai_body(json{{"instructions", "be brief"}, {"input", stateless_input}});
    EXPECT_EQ(a["messages"], b["messages"]);
    ASSERT_EQ(a["messages"].size(), 6u);  // system, user, assistant, assistant(tool_calls), tool, user
    EXPECT_EQ(a["messages"][2]["content"], "4");
    EXPECT_EQ(a["messages"][5]["content"], "And 3+3?");
}

TEST(ResponsesContinue, ChainsAndHandlesStringAndAbsentInput) {
    const json t1 = continue_conversation(json::array(), json::array(), "a");
    ASSERT_EQ(t1.size(), 1u);
    const json out1 = json::array({{{"role", "assistant"}, {"content", "b"}}});
    const json t2 = continue_conversation(t1, out1, "c");
    ASSERT_EQ(t2.size(), 3u);
    EXPECT_EQ(t2[2]["content"], "c");
    const json t3 = continue_conversation(t2, json::array(), json());  // no new input
    EXPECT_EQ(t3, t2);
    EXPECT_EQ(normalize_input_items(json()), json::array());
}
