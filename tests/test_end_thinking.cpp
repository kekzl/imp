// POST /v1/requests/{id}/end_thinking (#2420): BatchingEngine id registry and status decision,
// on a BatchingEngine that never starts (no engine, no GPU). The forced closer itself:
// EndThinking.* in test_think_stop_logic.cpp.

#include "batching_engine.h"

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

namespace {
using ET = BatchingEngine::EndThinking;

std::shared_ptr<ServerRequest> make_sr(std::vector<std::string> ids) {
    auto sr = std::make_shared<ServerRequest>();
    sr->request = std::make_shared<imp::Request>();
    sr->public_ids = std::move(ids);
    return sr;
}
}  // namespace

TEST(EndThinkingRegistry, UnknownIdIsNotFound) {
    BatchingEngine be;
    EXPECT_EQ(be.end_thinking("chatcmpl-none"), ET::NotFound);
    EXPECT_EQ(be.end_thinking(""), ET::NotFound);
}

TEST(EndThinkingRegistry, PendingRequestIsFlaggedOnceAndIdempotent) {
    BatchingEngine be;
    auto sr = make_sr({"chatcmpl-1", "client-trace-7"});
    be.submit(sr);
    bool first = false;
    EXPECT_EQ(be.end_thinking("chatcmpl-1", &first), ET::Ending);
    EXPECT_TRUE(first);
    EXPECT_TRUE(sr->end_thinking.load());
    EXPECT_EQ(be.end_thinking("chatcmpl-1", &first), ET::Ending);
    EXPECT_FALSE(first);
    EXPECT_EQ(be.end_thinking("client-trace-7", &first), ET::Ending);
    EXPECT_FALSE(first);
}

TEST(EndThinkingRegistry, PhaseDecidesStatus) {
    BatchingEngine be;
    auto sr = make_sr({"msg_imp_1"});
    be.submit(sr);
    sr->think_phase.store(ServerRequest::kThinkOff);
    EXPECT_EQ(be.end_thinking("msg_imp_1"), ET::AlreadyClosed);
    EXPECT_FALSE(sr->end_thinking.load());
    sr->think_phase.store(ServerRequest::kThinkNoCloser);
    EXPECT_EQ(be.end_thinking("msg_imp_1"), ET::NoCloser);
    EXPECT_FALSE(sr->end_thinking.load());
    sr->think_phase.store(ServerRequest::kThinkOn);
    EXPECT_EQ(be.end_thinking("msg_imp_1"), ET::Ending);
    EXPECT_TRUE(sr->end_thinking.load());
}

TEST(EndThinkingRegistry, FinishedRequestIsNotFound) {
    BatchingEngine be;
    auto sr = make_sr({"resp_imp1"});
    be.submit(sr);
    sr->push_finish("stop");
    EXPECT_EQ(be.end_thinking("resp_imp1"), ET::NotFound);
    EXPECT_FALSE(sr->end_thinking.load());
}

TEST(EndThinkingRegistry, RequestWithoutIdsIsNotRegistered) {
    BatchingEngine be;
    auto sr = make_sr({});
    be.submit(sr);
    EXPECT_EQ(be.end_thinking(""), ET::NotFound);
}
