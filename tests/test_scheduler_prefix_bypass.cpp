// #2198 direct mode: Request::bypass_prefix_cache at scheduler admission, on the plain
// prefix-reuse path and on the hybrid/SWA snapshot-callback path. Accounting cache, no GPU.

#include <gtest/gtest.h>

#include "memory/kv_cache.h"
#include "memory/kv_cache_manager.h"
#include "runtime/request.h"
#include "runtime/scheduler.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

namespace imp {
namespace {

// #2198 direct mode: bypass_prefix_cache admits with a full prefill even when the prompt's
// blocks are cached; the same prompt without the flag reuses them (control arm).
TEST(SchedulerTest, BypassPrefixCacheSkipsReuse) {
    auto cache = KVCache::for_accounting(
        /*n_layers=*/2, /*n_kv_heads=*/4, /*head_dim=*/64, QType::F16, /*max_blocks=*/64);
    auto mgr = std::make_unique<KVCacheManager>(std::move(cache));
    mgr->set_prefix_caching_enabled(true);

    std::vector<int32_t> prompt(128);
    for (int i = 0; i < 128; i++)
        prompt[static_cast<size_t>(i)] = 1000 + i;
    ASSERT_EQ(mgr->allocate_blocks_with_prefix(/*seq_id=*/0, prompt), 0);
    mgr->register_block_hashes(0, prompt);
    mgr->free_sequence(0);
    ASSERT_GT(mgr->num_cached_blocks(), 0);

    Scheduler sched(8);
    sched.set_kv_manager(mgr.get());
    auto cached = std::make_shared<Request>();
    cached->id = 1;
    cached->input_tokens = prompt;
    cached->max_tokens = 1;
    auto direct = std::make_shared<Request>();
    direct->id = 2;
    direct->input_tokens = prompt;
    direct->max_tokens = 1;
    direct->bypass_prefix_cache = true;
    sched.add_request(cached);
    sched.add_request(direct);

    std::vector<std::shared_ptr<Request>> prefill, decode;
    sched.schedule(prefill, decode);
    ASSERT_EQ(prefill.size(), 2u);
    EXPECT_GT(cached->cached_tokens, 0) << "control: without the flag the cached prefix is reused";
    EXPECT_EQ(direct->cached_tokens, 0);
    EXPECT_EQ(direct->prefill_offset, 0);
}

// Hybrid/SWA admission: the snapshot-boundary callback (Engine::hybrid_prefix_reuse_limit_ in
// production) must never be consulted for a bypass request; the control arm still reuses.
TEST(SchedulerTest, BypassPrefixCacheSkipsSnapshotReuseCallback) {
    auto cache = KVCache::for_accounting(
        /*n_layers=*/2, /*n_kv_heads=*/4, /*head_dim=*/64, QType::F16, /*max_blocks=*/64);
    auto mgr = std::make_unique<KVCacheManager>(std::move(cache));
    mgr->set_prefix_caching_enabled(true);

    std::vector<int32_t> prompt(128);
    for (int i = 0; i < 128; i++)
        prompt[static_cast<size_t>(i)] = 2000 + i;
    ASSERT_EQ(mgr->allocate_blocks_with_prefix(/*seq_id=*/0, prompt), 0);
    mgr->register_block_hashes(0, prompt);
    mgr->free_sequence(0);

    Scheduler sched(8);
    sched.set_kv_manager(mgr.get());
    std::vector<int> consulted;
    KVCacheManager* m = mgr.get();
    sched.set_prefix_reuse_limit([&consulted, m](Request& r) {
        consulted.push_back(r.id);
        std::vector<size_t> hashes;
        const int bs = kKVBlockSize;
        return std::min(m->longest_cached_prefix_blocks(r.input_tokens, hashes),
                        (static_cast<int>(r.input_tokens.size()) - 1) / bs);
    });
    auto cached = std::make_shared<Request>();
    cached->id = 1;
    cached->input_tokens = prompt;
    cached->max_tokens = 1;
    auto direct = std::make_shared<Request>();
    direct->id = 2;
    direct->input_tokens = prompt;
    direct->max_tokens = 1;
    direct->bypass_prefix_cache = true;
    sched.add_request(cached);
    sched.add_request(direct);

    std::vector<std::shared_ptr<Request>> prefill, decode;
    sched.schedule(prefill, decode);
    ASSERT_EQ(prefill.size(), 2u);
    EXPECT_EQ(consulted, std::vector<int>{1}) << "the snapshot callback ran for the bypass request";
    EXPECT_GT(cached->cached_tokens, 0) << "control: the snapshot path reuses the cached prefix";
    EXPECT_EQ(direct->cached_tokens, 0);
    EXPECT_EQ(direct->prefill_offset, 0);
}

// The one predicate both admission sites (scheduler, engine prefill fallback) read.
TEST(RequestTest, PrefixReuseAllowed) {
    Request plain;
    EXPECT_TRUE(plain.prefix_reuse_allowed());
    Request direct;
    direct.bypass_prefix_cache = true;
    EXPECT_FALSE(direct.prefix_reuse_allowed());
    Request image_no_hash;
    image_no_hash.n_vision_tokens = 4;
    EXPECT_FALSE(image_no_hash.prefix_reuse_allowed()) << "an unhashed image would reuse another picture";
    Request image_hashed;
    image_hashed.n_vision_tokens = 4;
    image_hashed.vision_content_hash = 0x1234;
    EXPECT_TRUE(image_hashed.prefix_reuse_allowed());
    image_hashed.bypass_prefix_cache = true;
    EXPECT_FALSE(image_hashed.prefix_reuse_allowed());
}

}  // namespace
}  // namespace imp
