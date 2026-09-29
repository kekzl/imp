// #2198 direct mode: Request::bypass_prefix_cache at scheduler admission, on the plain
// prefix-reuse path and on the hybrid/SWA snapshot-callback path; shared-mode submission waves.
// Accounting cache, no GPU.

#include <gtest/gtest.h>

#include "memory/kv_cache.h"
#include "memory/kv_cache_manager.h"
#include "runtime/request.h"
#include "runtime/scheduler.h"
#include "score_waves.h"

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

// #2198 shared mode: runs the handler's submission waves (score_waves.h) through the scheduler.
// Returns per item {wave index, prefill_offset}; finish emulates Engine::finish_request_release_.
struct WaveRun {
    std::vector<int> wave_of;
    std::vector<int> offset;
    std::vector<size_t> admitted_per_schedule;
};

WaveRun run_waves(const std::vector<std::vector<size_t>>& waves,
                  const std::vector<std::vector<int32_t>>& prompts) {
    auto cache = KVCache::for_accounting(
        /*n_layers=*/2, /*n_kv_heads=*/4, /*head_dim=*/64, QType::F16, /*max_blocks=*/512);
    auto mgr = std::make_unique<KVCacheManager>(std::move(cache));
    mgr->set_prefix_caching_enabled(true);
    Scheduler sched(64);
    sched.set_kv_manager(mgr.get());
    WaveRun out;
    out.wave_of.assign(prompts.size(), -1);
    out.offset.assign(prompts.size(), -1);
    for (size_t w = 0; w < waves.size(); w++) {
        std::vector<std::shared_ptr<Request>> reqs;
        for (const size_t i : waves[w]) {
            auto r = std::make_shared<Request>();
            r->id = static_cast<int>(i) + 1;
            r->input_tokens = prompts[i];
            r->max_tokens = 1;
            r->score_token_ids = {65, 66};
            sched.add_request(r);
            reqs.push_back(r);
        }
        std::vector<std::shared_ptr<Request>> prefill, decode;
        sched.schedule(prefill, decode);
        out.admitted_per_schedule.push_back(prefill.size());
        for (const auto& r : reqs) {
            const auto i = static_cast<size_t>(r->id - 1);
            out.wave_of[i] = static_cast<int>(w);
            out.offset[i] = r->prefill_offset;
            r->status = RequestStatus::FINISHED;
            mgr->register_block_hashes(r->id, r->input_tokens);
            mgr->free_sequence(r->id);
        }
    }
    return out;
}

// 8 items: 160 evidence tokens (10 blocks) + a distinct 40-token criterion/options suffix.
std::vector<std::vector<int32_t>> decide_prompts(size_t n) {
    std::vector<std::vector<int32_t>> prompts(n);
    for (size_t i = 0; i < n; i++) {
        for (int t = 0; t < 160; t++)
            prompts[i].push_back(3000 + t);
        for (int t = 0; t < 40; t++)
            prompts[i].push_back(10000 + static_cast<int32_t>(i) * 100 + t);
    }
    return prompts;
}

TEST(SchedulerTest, DecideSharedWavesReuseEvidenceInOneStep) {
    const size_t n = 8;
    const auto prompts = decide_prompts(n);
    const auto waves = server::score_waves(server::ScoreMode::Shared, n);
    ASSERT_EQ(waves.size(), 2u) << "shared: item 0 alone, then items 1..n-1";
    const WaveRun run = run_waves(waves, prompts);
    EXPECT_EQ(run.offset[0], 0) << "item 0 prefills the evidence";
    ASSERT_EQ(run.admitted_per_schedule.size(), 2u);
    EXPECT_EQ(run.admitted_per_schedule[1], n - 1) << "items 2..n are admitted in one scheduling step";
    for (size_t i = 1; i < n; i++) {
        EXPECT_EQ(run.wave_of[i], 1) << "item " << i;
        EXPECT_EQ(run.offset[i], 160) << "item " << i << " reuses the 10 evidence blocks";
    }
}

// Control: the same items as ONE wave (no bypass) reuse nothing; blocks publish at finish.
TEST(SchedulerTest, DecideOneWaveCannotReuseUnpublishedEvidence) {
    const size_t n = 8;
    const auto prompts = decide_prompts(n);
    const WaveRun run = run_waves(server::score_waves(server::ScoreMode::Direct, n), prompts);
    ASSERT_EQ(run.admitted_per_schedule.size(), 1u);
    EXPECT_EQ(run.admitted_per_schedule[0], n);
    for (size_t i = 0; i < n; i++)
        EXPECT_EQ(run.offset[i], 0) << "item " << i;
}

TEST(SchedulerTest, ScoreWavesPerMode) {
    using W = std::vector<std::vector<size_t>>;
    EXPECT_EQ(server::score_waves(server::ScoreMode::Serial, 3), (W{{0}, {1}, {2}}));
    EXPECT_EQ(server::score_waves(server::ScoreMode::Direct, 3), (W{{0, 1, 2}}));
    EXPECT_EQ(server::score_waves(server::ScoreMode::Shared, 3), (W{{0}, {1, 2}}));
    EXPECT_EQ(server::score_waves(server::ScoreMode::Shared, 1), (W{{0}}));
    EXPECT_TRUE(server::score_waves(server::ScoreMode::Shared, 0).empty());
}

// Score rows ride the ragged prefill (#2198 shared); the other exclusions stay.
TEST(RequestTest, RaggedPrefillAllowed) {
    Request plain;
    EXPECT_TRUE(plain.ragged_prefill_allowed());
    Request score;
    score.score_token_ids = {65, 66};
    EXPECT_TRUE(score.ragged_prefill_allowed()) << "score rows must batch in shared mode";
    Request embed;
    embed.embedding_request = true;
    EXPECT_FALSE(embed.ragged_prefill_allowed());
    Request lp;
    lp.logprobs = true;
    EXPECT_FALSE(lp.ragged_prefill_allowed());
    Request plp;
    plp.prompt_logprobs = 1;
    EXPECT_FALSE(plp.ragged_prefill_allowed());
    Request constrained;
    constrained.regex_pattern = "(A|B)";
    EXPECT_FALSE(constrained.ragged_prefill_allowed());
    Request vision;
    vision.n_vision_tokens = 4;
    EXPECT_FALSE(vision.ragged_prefill_allowed());
}

}  // namespace
}  // namespace imp
