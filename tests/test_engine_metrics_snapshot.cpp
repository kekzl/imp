// AUDIT_ARCH_2026_11_24 C-13: /metrics read engine counters on a handler thread while the worker
// mutated them under no common lock. The worker now publishes one EngineMetricsSnapshot per step;
// these tests drive the cell with a writer and a reader thread and check no read is torn.

#include "runtime/engine_metrics_snapshot.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <gtest/gtest.h>
#include <thread>
#include <type_traits>

namespace {

using imp::EngineMetricsCell;
using imp::EngineMetricsSnapshot;

// Every field written from one counter i: a read mixing two publishes breaks one of the relations.
EngineMetricsSnapshot make_snapshot(uint64_t i) {
    EngineMetricsSnapshot s;
    s.kv_pressure_rejections = i;  // first field after `publishes`
    s.kv_swaps_out = i;
    s.has_kv = true;
    s.kv_total_blocks = 4096;
    s.kv_cached_blocks = static_cast<int>(i % 1000);
    s.kv_free_blocks = s.kv_total_blocks - s.kv_cached_blocks - static_cast<int>(i % 97);  // live = i % 97
    s.snap_unused_tokens = i;
    s.moe_imb_layers = static_cast<int>(i % 61);
    s.spec.mtp.drafted = static_cast<long long>(i);
    s.spec.other.drafted = static_cast<long long>(i) * 2;
    s.spec.drafted = s.spec.mtp.drafted + s.spec.other.drafted;
    s.spec.other.verify_wall_ms = static_cast<double>(i);  // last field
    return s;
}

// The relations make_snapshot() writes, checked on one read.
bool consistent(const EngineMetricsSnapshot& s) {
    const uint64_t i = s.kv_pressure_rejections;
    const int live = s.kv_total_blocks - s.kv_free_blocks - s.kv_cached_blocks;
    return s.publishes == i + 1 && s.kv_swaps_out == i && s.snap_unused_tokens == i &&
           s.kv_cached_blocks == static_cast<int>(i % 1000) && live == static_cast<int>(i % 97) &&
           s.moe_imb_layers == static_cast<int>(i % 61) && s.spec.drafted == static_cast<long long>(i) * 3 &&
           s.spec.other.verify_wall_ms == static_cast<double>(i);
}

}  // namespace

// One memcpy per side, no allocation on the per-step publish.
static_assert(std::is_trivially_copyable_v<EngineMetricsSnapshot>);

TEST(EngineMetricsCell, ReadBeforePublishIsZero) {
    EngineMetricsCell cell;
    const EngineMetricsSnapshot s = cell.read();
    EXPECT_EQ(s.publishes, 0u);
    EXPECT_FALSE(s.has_kv);
    EXPECT_EQ(s.kv_total_blocks, 0);
    EXPECT_EQ(s.spec.drafted, 0);
}

TEST(EngineMetricsCell, PublishCountsAndCopies) {
    EngineMetricsCell cell;
    cell.publish(make_snapshot(0));
    cell.publish(make_snapshot(1));
    const EngineMetricsSnapshot s = cell.read();
    EXPECT_EQ(s.publishes, 2u);
    EXPECT_TRUE(consistent(s));
}

// Writer publishes as fast as it can, reader scrapes concurrently: every read is one whole publish,
// never two halves, and publishes never go backwards. A publish without the lock fails here.
TEST(EngineMetricsCell, ConcurrentReadsAreNeverTorn) {
    constexpr uint64_t kPublishes = 200000;
    EngineMetricsCell cell;
    cell.publish(make_snapshot(0));
    std::atomic<bool> done{false};
    std::thread writer([&] {
        for (uint64_t i = 1; i < kPublishes; ++i)
            cell.publish(make_snapshot(i));
        done.store(true, std::memory_order_release);
    });
    uint64_t reads = 0, torn = 0, last = 0, backwards = 0, distinct = 0;
    while (!done.load(std::memory_order_acquire)) {
        const EngineMetricsSnapshot s = cell.read();
        ++reads;
        if (!consistent(s))
            ++torn;
        if (s.publishes < last)
            ++backwards;
        if (s.publishes != last)
            ++distinct;
        last = s.publishes;
    }
    writer.join();
    std::printf("[ metrics ] %llu reads, %llu distinct publishes seen, %llu torn\n",
                static_cast<unsigned long long>(reads), static_cast<unsigned long long>(distinct),
                static_cast<unsigned long long>(torn));
    EXPECT_EQ(torn, 0u);
    EXPECT_EQ(backwards, 0u);
    EXPECT_GT(distinct, 1u) << "reader never overlapped the writer: the test proved nothing";
    const EngineMetricsSnapshot final_snap = cell.read();
    EXPECT_EQ(final_snap.publishes, kPublishes);
    EXPECT_TRUE(consistent(final_snap));
}

// The per-step cost: one uncontended lock and one struct copy.
TEST(EngineMetricsCell, PublishCostIsBounded) {
    constexpr int kN = 1000000;
    EngineMetricsCell cell;
    const EngineMetricsSnapshot s = make_snapshot(7);
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < kN; ++i)
        cell.publish(s);
    const double ns =
        std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count() / kN;
    std::printf("[ metrics ] publish %.1f ns, snapshot %zu bytes\n", ns, sizeof(EngineMetricsSnapshot));
    EXPECT_EQ(cell.read().publishes, static_cast<uint64_t>(kN));
    EXPECT_LT(ns, 5000.0);  // a 5 ms decode step: < 0.1 %; loose for sanitizer builds
}
