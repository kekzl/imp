// #2245: embedding requests mean-pool every input row, so scheduler admission must never
// reuse cached prefix blocks for them (prefill_offset 0). Accounting cache, no GPU.

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

// Same cached prompt: a chat request reuses the prefix (control arm), an embedding request does not.
TEST(SchedulerTest, EmbeddingRequestSkipsPrefixReuse) {
    auto cache = KVCache::for_accounting(
        /*n_layers=*/2, /*n_kv_heads=*/4, /*head_dim=*/64, QType::F16, /*max_blocks=*/64);
    auto mgr = std::make_unique<KVCacheManager>(std::move(cache));
    mgr->set_prefix_caching_enabled(true);

    std::vector<int32_t> prompt(128);
    for (int i = 0; i < 128; i++)
        prompt[static_cast<size_t>(i)] = 3000 + i;
    ASSERT_EQ(mgr->allocate_blocks_with_prefix(/*seq_id=*/0, prompt), 0);
    mgr->register_block_hashes(0, prompt);
    mgr->free_sequence(0);
    ASSERT_GT(mgr->num_cached_blocks(), 0);

    Scheduler sched(8);
    sched.set_kv_manager(mgr.get());
    auto chat = std::make_shared<Request>();
    chat->id = 1;
    chat->input_tokens = prompt;
    chat->max_tokens = 1;
    auto embed = std::make_shared<Request>();
    embed->id = 2;
    embed->input_tokens = prompt;
    embed->max_tokens = 1;
    embed->embedding_request = true;
    sched.add_request(chat);
    sched.add_request(embed);

    std::vector<std::shared_ptr<Request>> prefill, decode;
    sched.schedule(prefill, decode);
    ASSERT_EQ(prefill.size(), 2u);
    EXPECT_GT(chat->cached_tokens, 0) << "control: a chat request reuses the cached prefix";
    EXPECT_EQ(embed->cached_tokens, 0);
    EXPECT_EQ(embed->prefill_offset, 0);
}

}  // namespace
}  // namespace imp
