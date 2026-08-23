// Scheduler tests, split out of test_continuous_batching.cpp when that file
// crossed the 800-line hard threshold (tools/check_filesize.py).
//
// The split is by suite, not by size: BatchBuilderTest builds GPU batches,
// GPUBatchTest owns their device memory, and SchedulerTest decides which
// requests exist at all. They shared a file because they share a subject, not
// because they are one unit.
//
// Both files stay in test-e2e, and both are covered by the unit lane's
// gtest_filter, so scripts/check_e2e_lane_split.sh needs no change: it lists
// test NAMES, not files.

#include <gtest/gtest.h>

#include <algorithm>
#include <cuda_runtime.h>

#include "memory/kv_cache.h"
#include "memory/kv_cache_manager.h"
#include "runtime/batch.h"
#include "runtime/request.h"
#include "runtime/scheduler.h"
#include "test_cuda_skip.h"

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace imp {
namespace {

// 7. Scheduler basic: prefill then decode
TEST(SchedulerTest, BasicPrefillThenDecode) {
    Scheduler sched(4);  // max batch = 4

    auto req1 = std::make_shared<Request>();
    req1->input_tokens = {1, 2, 3, 4, 5};

    auto req2 = std::make_shared<Request>();
    req2->input_tokens = {10, 11, 12};

    sched.add_request(req1);
    sched.add_request(req2);

    // First schedule: both should go to prefill
    std::vector<std::shared_ptr<Request>> prefill, decode;
    sched.schedule(prefill, decode);

    EXPECT_EQ(prefill.size(), 2u);
    EXPECT_EQ(decode.size(), 0u);
    EXPECT_EQ(sched.active_count(), 2);

    // Simulate prefill completion -> DECODING
    req1->status = RequestStatus::DECODING;
    req2->status = RequestStatus::DECODING;

    // Second schedule: both should be in decode
    sched.schedule(prefill, decode);
    EXPECT_EQ(prefill.size(), 0u);
    EXPECT_EQ(decode.size(), 2u);
}
// 8. Scheduler respects max_batch_size
TEST(SchedulerTest, MaxBatchSizeLimit) {
    Scheduler sched(2);  // max batch = 2

    for (int i = 0; i < 5; i++) {
        auto req = std::make_shared<Request>();
        req->input_tokens = {1, 2, 3};
        sched.add_request(req);
    }

    std::vector<std::shared_ptr<Request>> prefill, decode;
    sched.schedule(prefill, decode);

    // Only 2 should be admitted
    EXPECT_EQ(prefill.size(), 2u);
    EXPECT_TRUE(sched.has_pending());
    EXPECT_EQ(sched.active_count(), 2);
}
// 9. Scheduler removes finished requests
TEST(SchedulerTest, RemovesFinishedRequests) {
    Scheduler sched(4);

    auto req1 = std::make_shared<Request>();
    req1->input_tokens = {1};
    auto req2 = std::make_shared<Request>();
    req2->input_tokens = {2};

    sched.add_request(req1);
    sched.add_request(req2);

    std::vector<std::shared_ptr<Request>> prefill, decode;
    sched.schedule(prefill, decode);
    EXPECT_EQ(sched.active_count(), 2);

    // Mark req1 as finished
    req1->status = RequestStatus::FINISHED;

    // Next schedule should clean up
    sched.schedule(prefill, decode);
    EXPECT_EQ(sched.active_count(), 1);
}
// 10. Memory-aware scheduling
TEST(SchedulerTest, MemoryAwareScheduling) {
    SKIP_IF_NO_CUDA();

    // Pool of 8 blocks = 128 tokens. Each request is a 32-token prompt
    // (2 blocks) plus max_tokens=16 (1 block + the partial-block spare), so
    // one reservation is 4 blocks and exactly two fit.
    auto cache = std::make_unique<KVCache>(
        /*n_layers=*/2, /*n_kv_heads=*/4, /*head_dim=*/64, QType::F16, /*max_blocks=*/8);

    auto mgr = std::make_unique<KVCacheManager>(std::move(cache));

    Scheduler sched(16);  // high batch size, but limited by memory
    sched.set_kv_manager(mgr.get());

    for (int i = 0; i < 5; i++) {
        auto req = std::make_shared<Request>();
        req->id = i;                      // distinct KV sequences
        req->input_tokens.resize(32, i);  // 32 tokens = 2 blocks
        req->max_tokens = 16;             // + 1 block + 1 spare
        sched.add_request(req);
    }

    std::vector<std::shared_ptr<Request>> prefill, decode;
    sched.schedule(prefill, decode);

    EXPECT_EQ(prefill.size(), 2u);
    EXPECT_TRUE(sched.has_pending());
}

// 10b. Admission reserves the generation, not just the prompt (#1635).
//
// Before the fix this admitted both requests: the test is `prompt fits`, and
// four 2-block prompts fit an 8-block pool. The generation then ran the pool
// dry and the loser was cancelled mid-stream, after the client had already
// received part of the answer.
TEST(SchedulerTest, AdmissionReservesGeneration) {
    SKIP_IF_NO_CUDA();

    // 16 blocks = 256 tokens. Each request: 32-token prompt (2 blocks) +
    // max_tokens=64 (4 blocks + 1 spare) = 7 blocks reserved, so two fit and
    // the third must queue.
    //
    // Three requests, not two: with two, the first request's reservation
    // alone already starves the second, and the test would pass with the
    // admission quantity mutated back to the prompt. The third is what the
    // admission test has to see - free blocks say yes (10 left), the
    // outstanding reservations say no.
    auto cache = std::make_unique<KVCache>(
        /*n_layers=*/2, /*n_kv_heads=*/4, /*head_dim=*/64, QType::F16, /*max_blocks=*/16);
    auto mgr = std::make_unique<KVCacheManager>(std::move(cache));

    Scheduler sched(16);
    sched.set_kv_manager(mgr.get());

    std::vector<std::shared_ptr<Request>> reqs;
    for (int i = 0; i < 3; i++) {
        auto req = std::make_shared<Request>();
        req->id = i;
        req->input_tokens.resize(32, i);
        req->max_tokens = 64;
        reqs.push_back(req);
        sched.add_request(req);
    }

    std::vector<std::shared_ptr<Request>> prefill, decode;
    sched.schedule(prefill, decode);

    // Two admitted, one still queued - and none cancelled.
    EXPECT_EQ(prefill.size(), 2u);
    EXPECT_TRUE(sched.has_pending());
    for (const auto& r : reqs)
        EXPECT_NE(r->status, RequestStatus::CANCELLED);

    // 12 blocks are free and 10 are promised: the free count on its own
    // would have admitted the third.
    EXPECT_EQ(mgr->num_free_blocks(), 12);
    EXPECT_EQ(mgr->outstanding_reserved_blocks(), 10);

    // When one finishes, its reservation goes with it.
    mgr->free_sequence(reqs[0]->id);
    reqs[0]->status = RequestStatus::FINISHED;
    EXPECT_EQ(mgr->outstanding_reserved_blocks(), 5);

    sched.schedule(prefill, decode);
    // Two, not one: the third is admitted now, and reqs[1] is still PREFILLING
    // at offset 0 (nothing stepped it here) so it is re-queued with it. Before
    // #1643 the refill required `prefill_offset > 0` and dropped it - in the
    // engine that never showed, because every promoted request was served in
    // the same tick it was promoted.
    EXPECT_EQ(prefill.size(), 2u);
    EXPECT_NE(std::find(prefill.begin(), prefill.end(), reqs[1]), prefill.end())
        << "an admitted request that has not been stepped yet must stay schedulable";
    EXPECT_NE(std::find(prefill.begin(), prefill.end(), reqs[2]), prefill.end());
    EXPECT_FALSE(sched.has_pending());
}

// 10c. A pool too small to ever hold prompt + max_tokens degrades to
// prompt-only admission instead of queueing the request forever (#1635).
TEST(SchedulerTest, AdmissionClampsReserveToPoolSize) {
    SKIP_IF_NO_CUDA();

    // 4 blocks = 64 tokens, against a 32-token prompt + max_tokens=256.
    // The full reserve (2 + 17) never fits, so the clamp is what keeps this
    // request servable at all.
    auto cache = std::make_unique<KVCache>(
        /*n_layers=*/2, /*n_kv_heads=*/4, /*head_dim=*/64, QType::F16, /*max_blocks=*/4);
    auto mgr = std::make_unique<KVCacheManager>(std::move(cache));

    Scheduler sched(16);
    sched.set_kv_manager(mgr.get());

    auto req = std::make_shared<Request>();
    req->id = 0;
    req->input_tokens.resize(32, 7);
    req->max_tokens = 256;
    sched.add_request(req);

    std::vector<std::shared_ptr<Request>> prefill, decode;
    sched.schedule(prefill, decode);

    EXPECT_EQ(prefill.size(), 1u);
    EXPECT_NE(req->status, RequestStatus::CANCELLED);
}
// 11. Continuous batching: prefill priority over decode
TEST(SchedulerTest, PrefillPriorityOverDecode) {
    Scheduler sched(4);

    // Add first request and schedule it (prefill)
    auto req1 = std::make_shared<Request>();
    req1->input_tokens = {1, 2};
    sched.add_request(req1);

    std::vector<std::shared_ptr<Request>> prefill, decode;
    sched.schedule(prefill, decode);
    ASSERT_EQ(prefill.size(), 1u);

    // Mark as decoding
    req1->status = RequestStatus::DECODING;

    // Add a new request while req1 is decoding
    auto req2 = std::make_shared<Request>();
    req2->input_tokens = {3, 4};
    sched.add_request(req2);

    // Schedule: req2 should go to prefill, req1 to decode
    sched.schedule(prefill, decode);
    EXPECT_EQ(prefill.size(), 1u);
    EXPECT_EQ(decode.size(), 1u);
    EXPECT_EQ(prefill[0], req2);
    EXPECT_EQ(decode[0], req1);
}
// 12. Scheduler handles cancelled requests
TEST(SchedulerTest, HandlesCancel) {
    Scheduler sched(4);

    auto req1 = std::make_shared<Request>();
    req1->input_tokens = {1};
    auto req2 = std::make_shared<Request>();
    req2->input_tokens = {2};

    sched.add_request(req1);
    sched.add_request(req2);

    std::vector<std::shared_ptr<Request>> prefill, decode;
    sched.schedule(prefill, decode);
    EXPECT_EQ(sched.active_count(), 2);

    // Cancel req1
    req1->status = RequestStatus::CANCELLED;

    sched.schedule(prefill, decode);
    EXPECT_EQ(sched.active_count(), 1);
}
// A request cancelled while it is still QUEUED must not be promoted (#1633).
//
// HandlesCancel above covers the other half: cancelled while already active.
// That one passed throughout, because `active_` was filtered and `pending_`
// was not - so the server's own disconnect path, which cancels before the
// request is ever scheduled, ran a full generation for a client that was gone.
TEST(SchedulerTest, DoesNotPromoteARequestCancelledWhileQueued) {
    Scheduler sched(4);

    auto queued = std::make_shared<Request>();
    queued->input_tokens = {1, 2, 3};
    auto live = std::make_shared<Request>();
    live->input_tokens = {4, 5, 6};

    sched.add_request(queued);
    sched.add_request(live);

    // The client disconnects before the first schedule() call.
    queued->status = RequestStatus::CANCELLED;

    std::vector<std::shared_ptr<Request>> prefill, decode;
    sched.schedule(prefill, decode);

    ASSERT_EQ(prefill.size(), 1u);
    EXPECT_EQ(prefill[0], live);
    EXPECT_EQ(sched.active_count(), 1);
    // And the promotion must not have overwritten the status, which is what
    // hid this downstream: PREFILLING says "a client is waiting".
    EXPECT_EQ(queued->status, RequestStatus::CANCELLED);
    EXPECT_FALSE(sched.has_pending());
}
// The whole queue cancelled is not a batch of work.
TEST(SchedulerTest, ACancelledQueueSchedulesNothing) {
    Scheduler sched(8);
    std::vector<std::shared_ptr<Request>> reqs;
    for (int i = 0; i < 5; i++) {
        auto r = std::make_shared<Request>();
        r->input_tokens = {i};
        r->status = RequestStatus::CANCELLED;
        reqs.push_back(r);
        sched.add_request(r);
    }

    std::vector<std::shared_ptr<Request>> prefill, decode;
    sched.schedule(prefill, decode);

    EXPECT_TRUE(prefill.empty());
    EXPECT_TRUE(decode.empty());
    EXPECT_EQ(sched.active_count(), 0);
    EXPECT_FALSE(sched.has_pending());
}
// 17. Multiple requests through scheduler lifecycle
TEST(SchedulerTest, FullLifecycle) {
    Scheduler sched(4);

    // Add 4 requests
    std::vector<std::shared_ptr<Request>> reqs(4);
    for (int i = 0; i < 4; i++) {
        reqs[i] = std::make_shared<Request>();
        reqs[i]->input_tokens = {1, 2, 3};
        sched.add_request(reqs[i]);
    }

    std::vector<std::shared_ptr<Request>> prefill, decode;

    // Step 1: All 4 go to prefill
    sched.schedule(prefill, decode);
    EXPECT_EQ(prefill.size(), 4u);
    EXPECT_EQ(decode.size(), 0u);
    EXPECT_EQ(sched.active_count(), 4);
    EXPECT_FALSE(sched.has_pending());

    // Simulate: all transition to DECODING
    for (auto& r : reqs)
        r->status = RequestStatus::DECODING;

    // Step 2: All 4 in decode batch
    sched.schedule(prefill, decode);
    EXPECT_EQ(prefill.size(), 0u);
    EXPECT_EQ(decode.size(), 4u);

    // Simulate: reqs[0] and reqs[2] finish
    reqs[0]->status = RequestStatus::FINISHED;
    reqs[2]->status = RequestStatus::FINISHED;

    // Step 3: Only reqs[1] and reqs[3] remain
    sched.schedule(prefill, decode);
    EXPECT_EQ(prefill.size(), 0u);
    EXPECT_EQ(decode.size(), 2u);
    EXPECT_EQ(sched.active_count(), 2);

    // Add 2 new requests
    auto new1 = std::make_shared<Request>();
    new1->input_tokens = {10};
    auto new2 = std::make_shared<Request>();
    new2->input_tokens = {20};
    sched.add_request(new1);
    sched.add_request(new2);

    // Step 4: New requests go to prefill, existing to decode
    sched.schedule(prefill, decode);
    EXPECT_EQ(prefill.size(), 2u);
    EXPECT_EQ(decode.size(), 2u);
    EXPECT_EQ(sched.active_count(), 4);

    // Simulate: all finish
    for (auto& r : reqs)
        r->status = RequestStatus::FINISHED;
    new1->status = RequestStatus::FINISHED;
    new2->status = RequestStatus::FINISHED;

    sched.schedule(prefill, decode);
    EXPECT_EQ(sched.active_count(), 0);
    EXPECT_FALSE(sched.has_pending());
}
// 21. Scheduler: batched decode with mid-batch completion
TEST(SchedulerTest, BatchedDecodeWithMidBatchCompletion) {
    Scheduler sched(8);

    // Create 6 requests, prefill all
    std::vector<std::shared_ptr<Request>> reqs(6);
    for (int i = 0; i < 6; i++) {
        reqs[i] = std::make_shared<Request>();
        reqs[i]->input_tokens = {1, 2, 3, 4};
        sched.add_request(reqs[i]);
    }

    std::vector<std::shared_ptr<Request>> prefill, decode;
    sched.schedule(prefill, decode);
    EXPECT_EQ(prefill.size(), 6u);
    EXPECT_EQ(decode.size(), 0u);

    // All transition to DECODING
    for (auto& r : reqs)
        r->status = RequestStatus::DECODING;

    // Step 1: All 6 in batched decode
    sched.schedule(prefill, decode);
    EXPECT_EQ(decode.size(), 6u);

    // Simulate: reqs[1] and reqs[4] finish mid-batch
    reqs[1]->status = RequestStatus::FINISHED;
    reqs[4]->status = RequestStatus::FINISHED;

    // Step 2: Only 4 remain in decode
    sched.schedule(prefill, decode);
    EXPECT_EQ(decode.size(), 4u);
    EXPECT_EQ(sched.active_count(), 4);

    // Add 3 new requests while 4 are decoding
    for (int i = 0; i < 3; i++) {
        auto req = std::make_shared<Request>();
        req->input_tokens = {10, 20, 30};
        sched.add_request(req);
    }

    // Step 3: 3 new prefill + 4 decode (total 7 within max_batch=8)
    sched.schedule(prefill, decode);
    EXPECT_EQ(prefill.size(), 3u);
    EXPECT_EQ(decode.size(), 4u);
    EXPECT_EQ(sched.active_count(), 7);
}
// 23. Scheduler: decode batch size respects max_batch_size
TEST(SchedulerTest, DecodeBatchSizeLimit) {
    Scheduler sched(4);  // max batch = 4

    // Create 6 requests, prefill 4 (max)
    std::vector<std::shared_ptr<Request>> reqs(6);
    for (int i = 0; i < 6; i++) {
        reqs[i] = std::make_shared<Request>();
        reqs[i]->input_tokens = {1};
        sched.add_request(reqs[i]);
    }

    std::vector<std::shared_ptr<Request>> prefill, decode;
    sched.schedule(prefill, decode);
    EXPECT_EQ(prefill.size(), 4u);  // max_batch=4
    EXPECT_TRUE(sched.has_pending());

    // Transition first 4 to DECODING
    for (int i = 0; i < 4; i++)
        reqs[i]->status = RequestStatus::DECODING;

    // Schedule: 4 decoding, 2 pending — pending cannot enter because batch is full
    sched.schedule(prefill, decode);
    EXPECT_EQ(decode.size(), 4u);
    // Pending requests admitted depends on scheduler policy (some schedulers
    // reserve slots for prefill). Check total active <= max_batch.
    EXPECT_LE(sched.active_count(), 4);

    // Finish 2, freeing slots
    reqs[0]->status = RequestStatus::FINISHED;
    reqs[1]->status = RequestStatus::FINISHED;

    // Now pending requests should be admitted
    sched.schedule(prefill, decode);
    EXPECT_EQ(decode.size(), 2u);   // reqs[2] + reqs[3]
    EXPECT_GE(prefill.size(), 1u);  // at least 1 pending admitted
    EXPECT_LE(sched.active_count(), 4);
}
// 24. Shortest-input-first (SIF) ordering
TEST(SchedulerTest, ShortestInputFirst) {
    Scheduler sched(2);  // admit only 2 at a time

    // Add requests in descending size order
    auto long_req = std::make_shared<Request>();
    long_req->id = 1;
    long_req->input_tokens.resize(100, 0);  // 100 tokens

    auto medium_req = std::make_shared<Request>();
    medium_req->id = 2;
    medium_req->input_tokens.resize(50, 0);  // 50 tokens

    auto short_req = std::make_shared<Request>();
    short_req->id = 3;
    short_req->input_tokens.resize(10, 0);  // 10 tokens

    sched.add_request(long_req);
    sched.add_request(medium_req);
    sched.add_request(short_req);

    std::vector<std::shared_ptr<Request>> prefill, decode;
    sched.schedule(prefill, decode);

    // SIF: shortest two should be admitted first
    ASSERT_EQ(prefill.size(), 2u);
    EXPECT_EQ(prefill[0]->id, 3);      // 10 tokens (shortest)
    EXPECT_EQ(prefill[1]->id, 2);      // 50 tokens (second shortest)
    EXPECT_TRUE(sched.has_pending());  // 100-token request still pending
}
// 25. Chunked prefill re-scheduling
TEST(SchedulerTest, ChunkedPrefillRescheduling) {
    Scheduler sched(4);

    auto req = std::make_shared<Request>();
    req->id = 1;
    req->input_tokens.resize(64, 0);
    sched.add_request(req);

    std::vector<std::shared_ptr<Request>> prefill, decode;

    // First schedule: promotes to prefill
    sched.schedule(prefill, decode);
    ASSERT_EQ(prefill.size(), 1u);
    EXPECT_EQ(prefill[0]->status, RequestStatus::PREFILLING);

    // Simulate partial prefill: only processed first 32 tokens
    req->prefill_offset = 32;

    // Second schedule: should re-appear in prefill batch for remaining chunk
    sched.schedule(prefill, decode);
    ASSERT_EQ(prefill.size(), 1u);
    EXPECT_EQ(prefill[0]->id, 1);
    EXPECT_EQ(prefill[0]->prefill_offset, 32);
}
// 26. Chunked prefill completes — transitions to decode
TEST(SchedulerTest, ChunkedPrefillCompleteThenDecode) {
    Scheduler sched(4);

    auto req = std::make_shared<Request>();
    req->input_tokens.resize(64, 0);
    sched.add_request(req);

    std::vector<std::shared_ptr<Request>> prefill, decode;
    sched.schedule(prefill, decode);
    ASSERT_EQ(prefill.size(), 1u);

    // Simulate: prefill fully completed, transition to decoding
    req->prefill_offset = 64;
    req->status = RequestStatus::DECODING;

    sched.schedule(prefill, decode);
    EXPECT_EQ(prefill.size(), 0u);
    EXPECT_EQ(decode.size(), 1u);
}
// 27. Empty scheduler returns empty batches
TEST(SchedulerTest, EmptyScheduler) {
    Scheduler sched(4);

    std::vector<std::shared_ptr<Request>> prefill, decode;
    sched.schedule(prefill, decode);

    EXPECT_EQ(prefill.size(), 0u);
    EXPECT_EQ(decode.size(), 0u);
    EXPECT_EQ(sched.active_count(), 0);
    EXPECT_FALSE(sched.has_pending());
}
// 28. Memory-aware scheduling skips large requests, admits smaller ones
TEST(SchedulerTest, MemoryAwareSkipsLargeAdmitsSmall) {
    SKIP_IF_NO_CUDA();

    // 4 blocks total, block_size=16
    auto cache = std::make_unique<KVCache>(
        /*n_layers=*/1, /*n_kv_heads=*/1, /*head_dim=*/64, QType::F16, /*max_blocks=*/4);
    auto mgr = std::make_unique<KVCacheManager>(std::move(cache));

    Scheduler sched(16);
    sched.set_kv_manager(mgr.get());

    // Large request: 80 tokens = 5 blocks (exceeds 4 total — infeasible, cancelled)
    auto large = std::make_shared<Request>();
    large->id = 1;
    large->input_tokens.resize(80, 0);
    sched.add_request(large);

    // Small request: 16 tokens = 1 block (fits)
    auto small_req = std::make_shared<Request>();
    small_req->id = 2;
    small_req->input_tokens.resize(16, 0);
    sched.add_request(small_req);

    std::vector<std::shared_ptr<Request>> prefill, decode;
    sched.schedule(prefill, decode);

    // Small admitted; large is infeasible (exceeds total cache capacity) so the
    // scheduler cancels it up-front rather than leaving it pending — leaving
    // a never-admittable request in pending_ would busy-loop the worker
    // (Nemotron-H regression that prompted the cancel-on-infeasible path).
    ASSERT_EQ(prefill.size(), 1u);
    EXPECT_EQ(prefill[0]->id, 2);
    EXPECT_EQ(large->status, RequestStatus::CANCELLED);
    EXPECT_FALSE(sched.has_pending());
}
// 29. All requests too large for memory — all cancelled (none feasible)
TEST(SchedulerTest, AllRequestsTooLargeForMemory) {
    SKIP_IF_NO_CUDA();

    auto cache = std::make_unique<KVCache>(
        /*n_layers=*/1, /*n_kv_heads=*/1, /*head_dim=*/64, QType::F16, /*max_blocks=*/2);
    auto mgr = std::make_unique<KVCacheManager>(std::move(cache));

    Scheduler sched(16);
    sched.set_kv_manager(mgr.get());

    // 3 requests each needing 3 blocks but only 2 available — all infeasible
    std::vector<std::shared_ptr<Request>> reqs;
    for (int i = 0; i < 3; i++) {
        auto req = std::make_shared<Request>();
        req->input_tokens.resize(48, 0);  // 48 tokens = 3 blocks
        sched.add_request(req);
        reqs.push_back(req);
    }

    std::vector<std::shared_ptr<Request>> prefill, decode;
    sched.schedule(prefill, decode);

    EXPECT_EQ(prefill.size(), 0u);
    EXPECT_EQ(decode.size(), 0u);
    EXPECT_FALSE(sched.has_pending());
    for (const auto& r : reqs)
        EXPECT_EQ(r->status, RequestStatus::CANCELLED);
}
// 30. Concurrent new prefill while others decoding
TEST(SchedulerTest, NewPrefillWhileDecoding) {
    Scheduler sched(8);

    // Start 3 requests decoding
    std::vector<std::shared_ptr<Request>> existing(3);
    for (int i = 0; i < 3; i++) {
        existing[i] = std::make_shared<Request>();
        existing[i]->id = i;
        existing[i]->input_tokens = {1, 2};
        sched.add_request(existing[i]);
    }

    std::vector<std::shared_ptr<Request>> prefill, decode;
    sched.schedule(prefill, decode);
    for (auto& r : existing)
        r->status = RequestStatus::DECODING;

    // Add 2 new requests while 3 are decoding
    auto new1 = std::make_shared<Request>();
    new1->id = 10;
    new1->input_tokens = {5, 6, 7};
    auto new2 = std::make_shared<Request>();
    new2->id = 11;
    new2->input_tokens = {8};

    sched.add_request(new1);
    sched.add_request(new2);

    sched.schedule(prefill, decode);

    // SIF: new2 (1 token) before new1 (3 tokens)
    ASSERT_EQ(prefill.size(), 2u);
    EXPECT_EQ(prefill[0]->id, 11);  // shorter first
    EXPECT_EQ(prefill[1]->id, 10);
    EXPECT_EQ(decode.size(), 3u);
    EXPECT_EQ(sched.active_count(), 5);
}
// 31. Add 10 requests, cancel 5 immediately, add 5 more — no crash, remaining schedulable
TEST(SchedulerTest, AddRemoveRapidly) {
    Scheduler sched(16);

    // Add 10 requests
    std::vector<std::shared_ptr<Request>> reqs(10);
    for (int i = 0; i < 10; i++) {
        reqs[i] = std::make_shared<Request>();
        reqs[i]->id = i;
        reqs[i]->input_tokens = {1, 2, 3};
        sched.add_request(reqs[i]);
    }

    // Schedule to promote all to active/prefill
    std::vector<std::shared_ptr<Request>> prefill, decode;
    sched.schedule(prefill, decode);
    EXPECT_EQ(prefill.size(), 10u);

    // Cancel 5 of them immediately
    for (int i = 0; i < 5; i++) {
        reqs[i]->status = RequestStatus::CANCELLED;
    }

    // Add 5 more requests
    std::vector<std::shared_ptr<Request>> new_reqs(5);
    for (int i = 0; i < 5; i++) {
        new_reqs[i] = std::make_shared<Request>();
        new_reqs[i]->id = 100 + i;
        new_reqs[i]->input_tokens = {4, 5};
        sched.add_request(new_reqs[i]);
    }

    // Transition surviving original requests to DECODING
    for (int i = 5; i < 10; i++) {
        reqs[i]->status = RequestStatus::DECODING;
    }

    // Schedule: cancelled removed, new ones prefill, survivors decode
    sched.schedule(prefill, decode);
    EXPECT_EQ(decode.size(), 5u);   // reqs[5..9] decoding
    EXPECT_EQ(prefill.size(), 5u);  // new_reqs[0..4] prefilling
    EXPECT_EQ(sched.active_count(), 10);
    EXPECT_FALSE(sched.has_pending());
}
// 32. Empty scheduler: get_prefill_batch and get_decode_batch return empty
TEST(SchedulerTest, EmptyBatch) {
    Scheduler sched(8);

    std::vector<std::shared_ptr<Request>> prefill, decode;

    // Multiple calls with no requests — all empty, no crash
    for (int i = 0; i < 3; i++) {
        sched.schedule(prefill, decode);
        EXPECT_EQ(prefill.size(), 0u);
        EXPECT_EQ(decode.size(), 0u);
        EXPECT_EQ(sched.active_count(), 0);
        EXPECT_FALSE(sched.has_pending());
    }
}
// 33. Adding more requests than max_batch_size caps the batch, doesn't crash
TEST(SchedulerTest, MaxBatchSize) {
    Scheduler sched(3);  // small max batch

    // Add 20 requests
    for (int i = 0; i < 20; i++) {
        auto req = std::make_shared<Request>();
        req->id = i;
        req->input_tokens = {1};
        sched.add_request(req);
    }

    std::vector<std::shared_ptr<Request>> prefill, decode;
    sched.schedule(prefill, decode);

    // Only max_batch_size admitted
    EXPECT_EQ(prefill.size(), 3u);
    EXPECT_EQ(sched.active_count(), 3);
    EXPECT_TRUE(sched.has_pending());

    // Drain remaining: finish current, schedule again repeatedly
    int total_admitted = 3;
    for (auto& r : prefill)
        r->status = RequestStatus::FINISHED;

    while (sched.has_pending()) {
        sched.schedule(prefill, decode);
        EXPECT_LE(static_cast<int>(prefill.size()), 3);
        total_admitted += static_cast<int>(prefill.size());
        for (auto& r : prefill)
            r->status = RequestStatus::FINISHED;
    }

    EXPECT_EQ(total_admitted, 20);
}

// A long prompt must not be passed over forever (#1634).
//
// Shortest-first is the policy and stays. What it lacked was a bound: the
// queue is re-sorted on every arrival, so under sustained short traffic a long
// prompt is overtaken every round, with nothing that ever makes it its turn.
TEST(SchedulerTest, AgingStopsALongPromptFromStarving) {
    Scheduler sched(1);  // one slot, so every round admits exactly one

    auto long_req = std::make_shared<Request>();
    long_req->input_tokens.assign(500, 1);
    sched.add_request(long_req);

    std::vector<std::shared_ptr<Request>> prefill, decode;
    bool long_admitted = false;

    // A short request arrives before every scheduling round, which is the
    // traffic pattern that starves the long one.
    for (int round = 0; round < Scheduler::kAgingRounds + 4 && !long_admitted; round++) {
        auto shorty = std::make_shared<Request>();
        shorty->input_tokens.assign(3, 1);
        sched.add_request(shorty);

        sched.schedule(prefill, decode);
        for (auto& r : prefill) {
            if (r == long_req)
                long_admitted = true;
            r->status = RequestStatus::FINISHED;  // free the slot for the next round
        }
    }

    EXPECT_TRUE(long_admitted) << "the long prompt was never admitted within "
                               << (Scheduler::kAgingRounds + 4) << " rounds";
}

// The property aging must not cost: among requests of the same age, the
// shorter one still goes first.
TEST(SchedulerTest, ShortestFirstStillHoldsAmongPeers) {
    Scheduler sched(1);

    auto long_req = std::make_shared<Request>();
    long_req->input_tokens.assign(500, 1);
    auto short_req = std::make_shared<Request>();
    short_req->input_tokens.assign(3, 1);
    sched.add_request(long_req);
    sched.add_request(short_req);

    std::vector<std::shared_ptr<Request>> prefill, decode;
    sched.schedule(prefill, decode);
    ASSERT_EQ(prefill.size(), 1u);
    EXPECT_EQ(prefill[0], short_req) << "same age, so length decides";
}

}  // namespace
}  // namespace imp
