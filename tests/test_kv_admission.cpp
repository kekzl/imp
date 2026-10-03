// #2486: KV admission by expected decode length, SWAPPED requests come back before new ones.

#include <gtest/gtest.h>

#include "memory/kv_cache.h"
#include "memory/kv_cache_manager.h"
#include "runtime/decode_length_estimate.h"
#include "runtime/request.h"
#include "runtime/scheduler.h"

#include <memory>
#include <vector>

namespace imp {
namespace {

std::unique_ptr<KVCacheManager> make_mgr(int blocks) {
    return std::make_unique<KVCacheManager>(KVCache::for_accounting(2, 4, 64, QType::F16, blocks));
}

std::shared_ptr<Request> make_req(int id, int prompt, int max_tokens) {
    auto r = std::make_shared<Request>();
    r->id = id;
    r->input_tokens.resize(static_cast<size_t>(prompt), id + 1);
    r->max_tokens = max_tokens;
    return r;
}

TEST(KvAdmission, EstimateIsTheP90OfFinishedOutputsWithFloorAndWarmup) {
    DecodeLengthEstimate e;
    EXPECT_EQ(e.tokens(), DecodeLengthEstimate::kInitialTokens);
    for (int i = 1; i <= 10; ++i)
        e.record(i * 100);        // 100..1000
    EXPECT_EQ(e.tokens(), 1000);  // index 9 of 10 sorted
    DecodeLengthEstimate small;
    for (int i = 0; i < DecodeLengthEstimate::kMinSamples; ++i)
        small.record(10);
    EXPECT_EQ(small.tokens(), DecodeLengthEstimate::kFloorTokens);
    EXPECT_EQ(admission_decode_tokens(-1, 8192, e), 8192);
    EXPECT_EQ(admission_decode_tokens(0, 8192, e), 1000);
    EXPECT_EQ(admission_decode_tokens(0, 300, e), 300);
    EXPECT_EQ(admission_decode_tokens(512, 8192, e), 512);
}

// 32 requests, 32-token prompts, max_tokens 8192 (server default), 2000-block pool.
TEST(KvAdmission, ExpectedLengthAdmitsMoreThanMaxTokens) {
    for (const int mode : {-1, 0}) {
        auto mgr = make_mgr(2000);
        Scheduler sched(64);
        sched.set_kv_manager(mgr.get());
        sched.set_admission_decode_mode(mode);
        for (int i = 0; i < 32; ++i)
            sched.add_request(make_req(i, 32, 8192));
        std::vector<std::shared_ptr<Request>> prefill, decode;
        sched.schedule(prefill, decode);
        // mode -1: 2 + (8192/16 + 1) = 515 blocks each, 3 x 515 = 1545 <= 2000 < 4 x 515.
        // mode 0: 2 + (1024/16 + 1) = 67 blocks each, 29 x 67 = 1943 <= 2000 < 30 x 67.
        EXPECT_EQ(prefill.size(), mode < 0 ? 3u : 29u) << "mode " << mode;
    }
}

TEST(KvAdmission, SwappedRequestComesBackFirstAndHoldsTheQueue) {
    auto mgr = make_mgr(64);
    Scheduler sched(16);
    sched.set_kv_manager(mgr.get());
    sched.set_admission_decode_mode(0);
    bool allow = false;
    int swap_ins = 0;
    sched.set_swap_in([&](Request& r) {
        if (!allow || !mgr->allocate_blocks(r.id, 4))
            return false;
        ++swap_ins;
        return true;
    });

    auto swapped = make_req(0, 32, 64);
    sched.add_request(swapped);
    std::vector<std::shared_ptr<Request>> prefill, decode;
    sched.schedule(prefill, decode);
    ASSERT_EQ(prefill.size(), 1u);
    // The engine moved its KV to host (engine_kv_swap.cpp) and freed the blocks.
    mgr->free_sequence(swapped->id);
    swapped->status = RequestStatus::SWAPPED;

    auto fresh = make_req(1, 32, 64);
    sched.add_request(fresh);
    sched.schedule(prefill, decode);
    EXPECT_TRUE(prefill.empty()) << "a new request must not take the blocks a swapped one waits for";
    EXPECT_TRUE(decode.empty());
    EXPECT_EQ(sched.swapped_count(), 1);

    allow = true;
    sched.schedule(prefill, decode);
    EXPECT_EQ(swap_ins, 1);
    EXPECT_EQ(swapped->status, RequestStatus::DECODING);
    ASSERT_EQ(decode.size(), 1u);
    EXPECT_EQ(decode[0], swapped);
    ASSERT_EQ(prefill.size(), 1u);
    EXPECT_EQ(prefill[0], fresh);
    EXPECT_GT(mgr->outstanding_reserved_blocks(), 0) << "swap-in re-sets the decode reservation";
}

TEST(KvAdmission, FinishedRequestsFeedTheEstimate) {
    auto mgr = make_mgr(4000);
    Scheduler sched(64);
    sched.set_kv_manager(mgr.get());
    sched.set_admission_decode_mode(0);
    std::vector<std::shared_ptr<Request>> reqs, prefill, decode;
    for (int i = 0; i < DecodeLengthEstimate::kMinSamples; ++i) {
        reqs.push_back(make_req(i, 16, 8192));
        sched.add_request(reqs.back());
    }
    sched.schedule(prefill, decode);
    for (auto& r : reqs) {
        r->output_tokens.assign(400, 7);
        r->status = RequestStatus::FINISHED;
        mgr->free_sequence(r->id);
    }
    sched.schedule(prefill, decode);
    EXPECT_EQ(sched.decode_estimate().samples(), DecodeLengthEstimate::kMinSamples);
    EXPECT_EQ(sched.reserve_decode_tokens(*make_req(99, 16, 8192)), 400);
}

}  // namespace
}  // namespace imp
