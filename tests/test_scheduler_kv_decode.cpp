// #2361: decode-time KV for admitted sequences. Accounting cache, no GPU.

#include <gtest/gtest.h>

#include "memory/kv_cache.h"
#include "memory/kv_cache_manager.h"
#include "runtime/request.h"
#include "runtime/scheduler.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace imp {
namespace {

// #2361: 8 slots, 32 x 30k prompts + 64 tokens, prefix cache on: no admitted sequence may find
// the pool dry at decode. Engine steps simulated: prefill completes in one step and registers its
// hashes, decode appends a block per 16 tokens, a finished sequence frees into the prefix cache.
TEST(SchedulerTest, AdmittedSequencesNeverRunThePoolDryAtDecode) {
    auto cache = KVCache::for_accounting(/*n_layers=*/2, /*n_kv_heads=*/4, /*head_dim=*/64, QType::F16,
                                         /*max_blocks=*/16384);
    auto mgr = std::make_unique<KVCacheManager>(std::move(cache));
    mgr->set_prefix_caching_enabled(true);
    Scheduler sched(8);
    sched.set_kv_manager(mgr.get());
    const int bs = mgr->kv_cache()->block_size();

    std::vector<std::shared_ptr<Request>> reqs;
    for (int i = 0; i < 32; i++) {
        auto req = std::make_shared<Request>();
        req->id = i;
        req->input_tokens.resize(29900 + 11 * i);
        for (size_t t = 0; t < req->input_tokens.size(); ++t)
            req->input_tokens[t] = static_cast<int32_t>((t * 7919 + static_cast<size_t>(i) * 104729) %
                                                        151000);
        req->max_tokens = 64;
        reqs.push_back(req);
        sched.add_request(req);
    }
    int dry = 0, finished = 0;
    std::vector<std::shared_ptr<Request>> prefill, decode;
    for (int step = 0; step < 20000 && finished < 32; ++step) {
        sched.schedule(prefill, decode);
        for (auto& r : prefill) {
            r->prefill_offset = static_cast<int>(r->input_tokens.size());
            mgr->register_block_hashes(r->id, r->input_tokens);
            r->output_tokens.push_back(1);
            r->status = RequestStatus::DECODING;
        }
        for (auto& r : decode) {
            const int need = (r->context_len() + bs - 1) / bs;
            if (need > static_cast<int>(mgr->block_table(r->id).size()) && mgr->append_block(r->id) < 0) {
                ++dry;
                mgr->free_sequence(r->id);
                r->status = RequestStatus::CANCELLED;
                ++finished;
                continue;
            }
            r->output_tokens.push_back(1);
            if (static_cast<int>(r->output_tokens.size()) >= r->max_tokens) {
                mgr->free_sequence(r->id);
                r->status = RequestStatus::FINISHED;
                ++finished;
            }
        }
    }
    EXPECT_EQ(finished, 32);
    EXPECT_EQ(dry, 0) << "admission promised blocks the pool could not hand out at decode";
}

// #2361: a row the pipeline drain finished after the schedule is still in the step's decode
// batch; its KV is freed, so append_block returns -1 and the engine cancelled it as "exhausted".
TEST(SchedulerTest, DecodeRowRetiredSkipsDrainedRows) {
    auto cache = KVCache::for_accounting(/*n_layers=*/2, /*n_kv_heads=*/4, /*head_dim=*/64, QType::F16,
                                         /*max_blocks=*/64);
    KVCacheManager mgr(std::move(cache));
    ASSERT_TRUE(mgr.allocate_blocks(7, 4));
    mgr.free_sequence(7);
    EXPECT_LT(mgr.append_block(7), 0) << "a freed sequence gets no block, with 64 free";
    EXPECT_EQ(mgr.num_free_blocks(), 64);

    EXPECT_TRUE(decode_row_retired(RequestStatus::FINISHED, 64, 64));
    EXPECT_TRUE(decode_row_retired(RequestStatus::CANCELLED, 3, 64));
    EXPECT_TRUE(decode_row_retired(RequestStatus::DECODING, 64, 64)) << "generation complete";
    EXPECT_FALSE(decode_row_retired(RequestStatus::DECODING, 63, 64));
    EXPECT_FALSE(decode_row_retired(RequestStatus::DECODING, 5000, 0)) << "0 = no max_tokens cap";
}

}  // namespace
}  // namespace imp
