// Split out of test_continuous_batching.cpp at the 800-line hard threshold
// (tools/check_filesize.py), by suite not size: BatchBuilderTest/GPUBatchTest/SchedulerTest
// share a subject, not a unit. Both files stay in test-e2e and the unit lane's gtest_filter,
// so scripts/check_e2e_lane_split.sh (lists test NAMES) needs no change.

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
// The admission gate (lazy recurrent-slot commit): a false answer seats
// nobody this round and takes nothing; the request is retried next round.
TEST(SchedulerTest, AdmissionGateHoldsTheWholeRound) {
    Scheduler sched(4);
    bool open = false;
    int asked = 0;
    sched.set_admission_gate([&] {
        ++asked;
        return open;
    });
    auto req1 = std::make_shared<Request>();
    req1->input_tokens = {1, 2, 3};
    auto req2 = std::make_shared<Request>();
    req2->input_tokens = {4, 5};
    sched.add_request(req1);
    sched.add_request(req2);

    std::vector<std::shared_ptr<Request>> prefill, decode;
    sched.schedule(prefill, decode);
    EXPECT_EQ(prefill.size(), 0u);
    EXPECT_EQ(sched.active_count(), 0);
    EXPECT_EQ(sched.pending_count(), 2);
    EXPECT_EQ(asked, 1) << "one refusal ends the round; the second request is not asked for";
    EXPECT_EQ(req1->status, RequestStatus::PENDING) << "held, not cancelled";

    open = true;
    sched.schedule(prefill, decode);
    EXPECT_EQ(prefill.size(), 2u);
    EXPECT_EQ(asked, 3);
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
    // Admission is bookkeeping: memory-aware tests build an accounting cache (block ids/counts,
    // no VRAM) and run in the CI lane; only the growable-pool test needs a device. Pool of 8
    // blocks=128 tokens; each request (32-token prompt=2 blocks + max_tokens=16=1 block+spare)
    // reserves 4 blocks, so exactly two fit.
    auto cache = KVCache::for_accounting(
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

// #1635: admission must reserve the generation, not just the prompt. Before the fix, four
// 2-block prompts admitted into an 8-block pool (all fit), then generation ran the pool dry
// and the loser was cancelled mid-stream after the client had already received part of the answer.
TEST(SchedulerTest, AdmissionReservesGeneration) {
    // 16 blocks=256 tokens; each request (32-token prompt=2 blocks + max_tokens=64=4 blocks+1
    // spare)=7 blocks, so two fit and the third queues. Three requests, not two: with two, the
    // first reservation alone already starves the second, so a mutated (prompt-only) admission
    // quantity would still pass.
    auto cache = KVCache::for_accounting(
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
    // Two, not one admitted: the third fits, reqs[1] is still PREFILLING at offset 0 and gets
    // re-queued with it. Before #1643 the refill required prefill_offset > 0 and dropped it - never
    // visible in the engine since every promoted request was served the same tick it was promoted.
    EXPECT_EQ(prefill.size(), 2u);
    EXPECT_NE(std::find(prefill.begin(), prefill.end(), reqs[1]), prefill.end())
        << "an admitted request that has not been stepped yet must stay schedulable";
    EXPECT_NE(std::find(prefill.begin(), prefill.end(), reqs[2]), prefill.end());
    EXPECT_FALSE(sched.has_pending());
}

// Aggregate admission pressure must grow a growable pool toward its ceiling: before this,
// fitting requests queued while the pool sat at its initial commit (32x8k concurrent served
// effectively ~8-wide, 4437 of 6483 ceiling blocks never committed).
TEST(SchedulerGpuTest, GrowsPoolUnderAggregatePressure) {
    SKIP_IF_NO_CUDA();

    // 8 committed blocks, 16-block ceiling. Each request reserves 4 blocks
    // (32-token prompt = 2 blocks, max_tokens=16 = 1 block + spare): two fit
    // the commit, four fit the ceiling.
    auto cache = std::make_unique<KVCache>(
        /*n_layers=*/2, /*n_kv_heads=*/4, /*head_dim=*/64, QType::F16, /*max_blocks=*/8,
        /*block_size=*/16, /*alloc=*/nullptr, /*ceiling_blocks=*/16);
    if (!cache->growable())
        GTEST_SKIP() << "no VMM backend on this device";
    KVCache* kvc = cache.get();
    auto mgr = std::make_unique<KVCacheManager>(std::move(cache));

    Scheduler sched(16);
    sched.set_kv_manager(mgr.get());

    for (int i = 0; i < 5; i++) {
        auto req = std::make_shared<Request>();
        req->id = i;
        req->input_tokens.resize(32, i);
        req->max_tokens = 16;
        sched.add_request(req);
    }

    std::vector<std::shared_ptr<Request>> prefill, decode;
    sched.schedule(prefill, decode);

    // Four admitted (the ceiling's worth), the fifth queued, pool at ceiling.
    EXPECT_EQ(prefill.size(), 4u);
    EXPECT_TRUE(sched.has_pending());
    EXPECT_EQ(kvc->total_blocks(), 16);
    EXPECT_GE(kvc->growths(), 1u);
}

// 10c. A pool too small to ever hold prompt + max_tokens degrades to
// prompt-only admission instead of queueing the request forever (#1635).
TEST(SchedulerTest, AdmissionClampsReserveToPoolSize) {
    // 4 blocks = 64 tokens, against a 32-token prompt + max_tokens=256.
    // The full reserve (2 + 17) never fits, so the clamp is what keeps this
    // request servable at all.
    auto cache = KVCache::for_accounting(
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
// #1633: a request cancelled while still QUEUED must not be promoted. HandlesCancel covers
// the active-cancel half only, because active_ was filtered and pending_ was not - the
// server's own disconnect path (cancels before scheduling) ran a full generation for a gone client.
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
    // 4 blocks total, block_size=16
    auto cache = KVCache::for_accounting(
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

    // A request exceeding total cache capacity is cancelled up-front rather than left pending:
    // a never-admittable request in pending_ would busy-loop the worker (Nemotron-H regression).
    ASSERT_EQ(prefill.size(), 1u);
    EXPECT_EQ(prefill[0]->id, 2);
    EXPECT_EQ(large->status, RequestStatus::CANCELLED);
    EXPECT_FALSE(sched.has_pending());
}
// 29. All requests too large for memory — all cancelled (none feasible)
TEST(SchedulerTest, AllRequestsTooLargeForMemory) {
    auto cache = KVCache::for_accounting(
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

// #1634: shortest-first has no bound - the queue re-sorts on every arrival, so under
// sustained short traffic a long prompt is overtaken every round and never gets its turn.
// Aging fixes that.
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

// Priority (lower value first) is the primary admission key: a long
// high-priority request beats a short default-priority one.
TEST(SchedulerTest, PriorityBeatsShortestFirst) {
    Scheduler sched(1);

    auto short_default = std::make_shared<Request>();
    short_default->input_tokens.assign(3, 1);
    auto long_urgent = std::make_shared<Request>();
    long_urgent->input_tokens.assign(500, 1);
    long_urgent->priority = -1;
    sched.add_request(short_default);
    sched.add_request(long_urgent);

    std::vector<std::shared_ptr<Request>> prefill, decode;
    sched.schedule(prefill, decode);
    ASSERT_EQ(prefill.size(), 1u);
    EXPECT_EQ(prefill[0], long_urgent) << "lower priority value must admit first";
}

// Within one priority class the pre-priority order is unchanged:
// shortest-first among peers.
TEST(SchedulerTest, EqualPriorityFallsBackToShortestFirst) {
    Scheduler sched(1);

    auto long_req = std::make_shared<Request>();
    long_req->input_tokens.assign(500, 1);
    long_req->priority = 5;
    auto short_req = std::make_shared<Request>();
    short_req->input_tokens.assign(3, 1);
    short_req->priority = 5;
    sched.add_request(long_req);
    sched.add_request(short_req);

    std::vector<std::shared_ptr<Request>> prefill, decode;
    sched.schedule(prefill, decode);
    ASSERT_EQ(prefill.size(), 1u);
    EXPECT_EQ(prefill[0], short_req);
}

// Strict dominance: aging never lifts a request over a higher-priority class; a starved
// low-priority request is admitted only once no higher-priority work is pending (caller's
// contract, documented in scheduler.cpp).
TEST(SchedulerTest, AgingDoesNotCrossPriorityClasses) {
    Scheduler sched(1);

    auto low = std::make_shared<Request>();
    low->input_tokens.assign(3, 1);
    low->priority = 1;
    sched.add_request(low);

    std::vector<std::shared_ptr<Request>> prefill, decode;

    // Sustained higher-priority traffic across the aging boundary: low must
    // never be picked while a priority-0 request is pending.
    for (int round = 0; round < Scheduler::kAgingRounds + 4; round++) {
        auto urgent = std::make_shared<Request>();
        urgent->input_tokens.assign(3, 1);
        sched.add_request(urgent);

        sched.schedule(prefill, decode);
        ASSERT_EQ(prefill.size(), 1u);
        EXPECT_NE(prefill[0], low) << "aged low-priority overtook class 0 in round " << round;
        prefill[0]->status = RequestStatus::FINISHED;
    }

    // Traffic stops: the low-priority request drains.
    sched.schedule(prefill, decode);
    ASSERT_EQ(prefill.size(), 1u);
    EXPECT_EQ(prefill[0], low);
}

// Aging still bounds starvation WITHIN a priority class.
TEST(SchedulerTest, AgingStillBoundsStarvationWithinAClass) {
    Scheduler sched(1);

    auto long_req = std::make_shared<Request>();
    long_req->input_tokens.assign(500, 1);
    long_req->priority = 2;
    sched.add_request(long_req);

    std::vector<std::shared_ptr<Request>> prefill, decode;
    bool long_admitted = false;
    for (int round = 0; round < Scheduler::kAgingRounds + 4 && !long_admitted; round++) {
        auto shorty = std::make_shared<Request>();
        shorty->input_tokens.assign(3, 1);
        shorty->priority = 2;
        sched.add_request(shorty);

        sched.schedule(prefill, decode);
        for (auto& r : prefill) {
            if (r == long_req)
                long_admitted = true;
            r->status = RequestStatus::FINISHED;
        }
    }
    EXPECT_TRUE(long_admitted);
}

// AUDIT_arch_2026 C-2: aging fixed sort order, not the allocator. A 4-block pool holds one
// decoder (1+1 reserved); a 48-token request needs 4 and can't fit while it runs, a 16-token
// one needs 2 and can - a fresh short request every round starved the long one indefinitely.
// Once aged, the long request holds the queue: no short request passes it, it goes first
// when blocks free up.
TEST(SchedulerTest, AnAgedRequestHoldsTheQueueUntilItsBlocksAreFree) {
    auto cache = KVCache::for_accounting(
        /*n_layers=*/1, /*n_kv_heads=*/1, /*head_dim=*/64, QType::F16, /*max_blocks=*/4);
    auto mgr = std::make_unique<KVCacheManager>(std::move(cache));
    Scheduler sched(4);
    sched.set_kv_manager(mgr.get());
    std::vector<std::shared_ptr<Request>> prefill, decode;

    auto running = std::make_shared<Request>();
    running->id = 1;
    running->input_tokens.assign(16, 1);
    running->max_tokens = 0;
    sched.add_request(running);
    sched.schedule(prefill, decode);
    ASSERT_EQ(prefill.size(), 1u);
    running->status = RequestStatus::DECODING;  // holds 1 block + 1 reserved from here on

    auto long_req = std::make_shared<Request>();
    long_req->id = 2;
    long_req->input_tokens.assign(48, 1);  // 3 blocks + 1 reserved = 4: the whole pool
    long_req->max_tokens = 0;
    sched.add_request(long_req);

    int next_id = 100;
    int shorts_admitted_before_aging = 0;
    std::shared_ptr<Request> pending_short;
    for (int round = 0; round < Scheduler::kAgingRounds; round++) {
        pending_short = std::make_shared<Request>();
        pending_short->id = next_id++;
        pending_short->input_tokens.assign(16, 1);  // 1 block + 1 reserved = 2: fits beside `running`
        pending_short->max_tokens = 0;
        sched.add_request(pending_short);
        sched.schedule(prefill, decode);
        ASSERT_EQ(std::find(prefill.begin(), prefill.end(), long_req), prefill.end())
            << "the long request cannot fit while `running` holds the pool";
        for (auto& r : prefill) {
            shorts_admitted_before_aging++;
            mgr->free_sequence(r->id);
            r->status = RequestStatus::FINISHED;
        }
    }
    EXPECT_GT(shorts_admitted_before_aging, 0) << "the shorts must have been passing the long request";

    // Aged now. The short that arrives this round fits, and is held anyway.
    pending_short = std::make_shared<Request>();
    pending_short->id = next_id++;
    pending_short->input_tokens.assign(16, 1);
    pending_short->max_tokens = 0;
    sched.add_request(pending_short);
    sched.schedule(prefill, decode);
    EXPECT_TRUE(prefill.empty()) << "an aged head that cannot get its blocks holds the queue";
    EXPECT_EQ(long_req->status, RequestStatus::PENDING) << "held, not cancelled: the pool can hold it";

    // The blocks come back: the aged request goes first, the short waits.
    mgr->free_sequence(running->id);
    running->status = RequestStatus::FINISHED;
    sched.schedule(prefill, decode);
    ASSERT_EQ(prefill.size(), 1u);
    EXPECT_EQ(prefill[0], long_req);
    EXPECT_TRUE(sched.has_pending()) << "the short is still queued behind it";
}

}  // namespace
}  // namespace imp
