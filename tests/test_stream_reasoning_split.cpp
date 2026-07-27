// ===========================================================================
// StreamReasoningSplitter: the streaming half of the think/content demux.
//
// WHY THIS EXISTS: the non-streaming path splits reasoning offline, where the
// whole text is available and a `</think>` anywhere proves the prefix was
// reasoning. Streaming has to decide per token, and the two paths disagreeing
// is a bug the caller sees directly — the same request returns the chain of
// thought in `reasoning_content` without `stream:true` and as the visible
// answer with it.
//
// The case that matters in practice (found by running Claude Code against
// imp-server): a tool request suppresses thinking, the chat template renders a
// PRE-CLOSED think block, and the model reasons anyway. Its output then carries
// no `<think>` opener — only the closer — so scanning for an opener can never
// succeed, and the reasoning is streamed to the user as the answer.
//
// Header-only dependency, so this runs in the CPU lane.
// ===========================================================================

#include <gtest/gtest.h>

#include "reasoning_split.h"

#include <string>
#include <vector>

using imp::server::StreamReasoningSplitter;
using imp::server::ThinkPhase;

namespace {

struct Split {
    std::string reasoning;
    std::string content;
    int reasoning_tokens = 0;
};

// Feed pieces as if they were decoded tokens. `close_id` is the special-token
// id for `</think>` (-1 when the model spells it as text).
Split drive(StreamReasoningSplitter& s, const std::vector<std::string>& pieces, int close_id = -1,
            const std::string& close_piece = "</think>") {
    Split out;
    for (const auto& p : pieces) {
        const int id = (close_id >= 0 && p == close_piece) ? close_id : 1;
        auto r = s.feed(p, id);
        out.reasoning += r.reasoning;
        out.content += r.content;
        out.reasoning_tokens += r.reasoning_tokens;
    }
    auto fin = s.finish();
    out.reasoning += fin.reasoning;
    out.content += fin.content;
    return out;
}

// Mirrors imp::server::kAgentScanLimit in the stream driver.
constexpr int kAgentScan = 256;

// A chain of thought long enough to outrun the default 8-token scan budget —
// real ones are hundreds of tokens.
std::vector<std::string> long_cot() {
    std::vector<std::string> v;
    for (int i = 0; i < 40; i++)
        v.push_back("reason" + std::to_string(i) + " ");
    return v;
}

}  // namespace

// The baseline the streaming path already got right: an explicit opener.
TEST(StreamReasoningSplit, OpenerInOutputRoutesToReasoning) {
    StreamReasoningSplitter s(ThinkPhase::SCAN, -1, -1);
    auto r = drive(s, {"<think>", "thinking ", "more", "</think>", "the answer"});
    EXPECT_NE(r.reasoning.find("thinking"), std::string::npos) << r.reasoning;
    EXPECT_EQ(r.content, "the answer");
}

// Generation that starts inside a template-opened block: no opener arrives, and
// the phase says so up front.
TEST(StreamReasoningSplit, PromptOpenedBlockStartsInReasoning) {
    StreamReasoningSplitter s(ThinkPhase::REASONING, -1, -1);
    auto r = drive(s, {"thinking ", "more ", "</think>", "the answer"});
    EXPECT_NE(r.reasoning.find("thinking"), std::string::npos) << r.reasoning;
    EXPECT_EQ(r.content, "the answer");
}

// THE REGRESSION. Suppressed thinking on a model that reasons anyway: no
// opener, a long chain of thought, then the closer. Everything before the
// closer is reasoning — scanning for an opener that will never come must not
// dump it into the user-visible channel.
//
// kAgentScan is the budget the stream driver uses when the request carries
// tools: the hold has to outlast a real chain of thought, or the closer that
// proves what the prefix was arrives after the leak.
TEST(StreamReasoningSplit, CloserWithoutOpenerReclassifiesTheWholePrefix) {
    StreamReasoningSplitter s(ThinkPhase::SCAN, -1, -1, kAgentScan);
    auto pieces = long_cot();
    pieces.push_back("</think>");
    pieces.push_back("the answer");
    auto r = drive(s, pieces);
    EXPECT_EQ(r.content, "the answer") << "chain of thought leaked into content: " << r.content;
    EXPECT_NE(r.reasoning.find("reason0"), std::string::npos) << "prefix not classified as reasoning";
    EXPECT_NE(r.reasoning.find("reason39"), std::string::npos);
    EXPECT_EQ(r.reasoning.find("the answer"), std::string::npos) << "answer trapped in reasoning";
}

// Same, with `</think>` arriving as a single special token rather than text —
// which is how Qwen3 emits it.
TEST(StreamReasoningSplit, CloserAsSpecialTokenAlsoReclassifies) {
    StreamReasoningSplitter s(ThinkPhase::SCAN, -1, /*think_end_id=*/42, kAgentScan);
    auto pieces = long_cot();
    pieces.push_back("</think>");
    pieces.push_back("the answer");
    auto r = drive(s, pieces, /*close_id=*/42);
    EXPECT_EQ(r.content, "the answer") << "leaked: " << r.content;
    EXPECT_NE(r.reasoning.find("reason0"), std::string::npos);
}

// The other direction must not regress: a model that does NOT reason has to
// reach the user. Holding output while waiting for a closer that never comes is
// only acceptable if it is bounded and then flushed as content.
TEST(StreamReasoningSplit, NoReasoningStillFlushesAsContent) {
    StreamReasoningSplitter s(ThinkPhase::SCAN, -1, -1, kAgentScan);
    std::vector<std::string> pieces;
    for (int i = 0; i < 40; i++)
        pieces.push_back("word" + std::to_string(i) + " ");
    auto r = drive(s, pieces);
    EXPECT_TRUE(r.reasoning.empty()) << "plain answer misclassified as reasoning: " << r.reasoning;
    EXPECT_NE(r.content.find("word0"), std::string::npos);
    EXPECT_NE(r.content.find("word39"), std::string::npos);
}

// CONTENT phase is a pass-through — a request that asked for no extraction must
// not have its text reshuffled by either mechanism.
TEST(StreamReasoningSplit, ContentPhaseIsPassThrough) {
    StreamReasoningSplitter s(ThinkPhase::CONTENT, -1, -1);
    auto r = drive(s, {"plain ", "answer ", "</think>", "tail"});
    EXPECT_TRUE(r.reasoning.empty()) << r.reasoning;
    EXPECT_NE(r.content.find("plain"), std::string::npos);
}

// The hold must not swallow a tool call. Releasing it early is the caller's
// job (the splitter knows nothing about tool tags), so it needs the buffer and
// a way to give it up — without that, the fix for the leak above breaks
// streamed tool-call argument deltas, which is how it was caught.
TEST(StreamReasoningSplit, FlushScanReleasesTheHeldBufferAsContent) {
    StreamReasoningSplitter s(ThinkPhase::SCAN, -1, -1, kAgentScan);
    for (const char* p : {"Sure", ", ", "<tool_", "call>"})
        s.feed(p, 1);
    EXPECT_EQ(s.phase(), ThinkPhase::SCAN);
    EXPECT_NE(s.held().find("<tool_call>"), std::string::npos) << s.held();

    auto r = s.flush_scan();
    EXPECT_EQ(r.content, "Sure, <tool_call>");
    EXPECT_TRUE(r.reasoning.empty());
    EXPECT_EQ(s.phase(), ThinkPhase::CONTENT);

    // Everything after the release streams through untouched.
    auto after = s.feed("{\"name\":", 1);
    EXPECT_EQ(after.content, "{\"name\":");
}

// Releasing when nothing is held, or outside SCAN, must be a no-op.
TEST(StreamReasoningSplit, FlushScanIsANoOpOutsideScan) {
    StreamReasoningSplitter s(ThinkPhase::CONTENT, -1, -1);
    auto r = s.flush_scan();
    EXPECT_TRUE(r.content.empty());
    EXPECT_TRUE(r.reasoning.empty());
    EXPECT_EQ(s.phase(), ThinkPhase::CONTENT);
}
