// AUDIT_ARCH_2026_11_24 C-2: a model swap, suspend or unload freed the tokenizer while other
// handlers still held a raw pointer, and submit never checked which load a request was built for.
// The protocol lives in ModelLifetimeGate so it runs here without a model.

#include "model_lifetime.h"

#include <atomic>
#include <chrono>
#include <gtest/gtest.h>
#include <mutex>
#include <thread>
#include <vector>

using namespace std::chrono_literals;
using Exclusive = ModelLifetimeGate::Exclusive;

namespace {

// Spins until `pred` holds or 5 s passed; the tests assert on the result, never on a sleep.
template <class Pred>
bool eventually(Pred pred) {
    const auto deadline = std::chrono::steady_clock::now() + 5s;
    while (!pred()) {
        if (std::chrono::steady_clock::now() > deadline)
            return false;
        std::this_thread::sleep_for(1ms);
    }
    return true;
}

}  // namespace

TEST(ModelLifetimeGate, ExclusiveIsImmediateWithoutReaders) {
    ModelLifetimeGate g;
    EXPECT_EQ(g.begin_exclusive(0ms), Exclusive::Acquired);
    EXPECT_TRUE(g.exclusive_pending());
    g.end_exclusive();
    EXPECT_FALSE(g.exclusive_pending());
    EXPECT_TRUE(g.try_enter());
    g.leave();
}

// (a) The freeing side gets the model only after the last reader left.
TEST(ModelLifetimeGate, ExclusiveWaitsUntilReadersLeave) {
    ModelLifetimeGate g;
    auto a = acquire_model_lease(g);
    auto b = acquire_model_lease(g);
    ASSERT_TRUE(a && b);
    ASSERT_EQ(g.readers(), 2);

    std::atomic<bool> acquired{false};
    std::atomic<int> readers_at_acquire{-1};
    std::thread swapper([&] {
        const Exclusive r = g.begin_exclusive(10s);
        readers_at_acquire.store(g.readers());
        acquired.store(r == Exclusive::Acquired);
    });
    ASSERT_TRUE(eventually([&] { return g.exclusive_pending(); }));
    a.reset();
    std::this_thread::sleep_for(50ms);
    EXPECT_EQ(readers_at_acquire.load(), -1) << "exclusive was granted with one reader still inside";
    b.reset();
    swapper.join();
    EXPECT_TRUE(acquired.load());
    EXPECT_EQ(readers_at_acquire.load(), 0);
    g.end_exclusive();
}

// (b) Readers that outlive the budget: the swap is refused and the gate serves on as before.
TEST(ModelLifetimeGate, TimesOutWithReadersInsideAndStaysUsable) {
    ModelLifetimeGate g;
    auto stuck = acquire_model_lease(g);
    ASSERT_TRUE(stuck);
    const uint64_t epoch = g.epoch();

    EXPECT_EQ(g.begin_exclusive(30ms), Exclusive::TimedOut);
    EXPECT_FALSE(g.exclusive_pending()) << "a timed-out swap must not keep refusing requests";
    EXPECT_EQ(g.epoch(), epoch) << "nothing was freed, so the epoch stands";
    EXPECT_TRUE(stuck->current());

    auto next = acquire_model_lease(g);
    ASSERT_TRUE(next) << "the old model keeps serving after the refused swap";
    EXPECT_EQ(g.readers(), 2);
    next.reset();
    stuck.reset();
    EXPECT_EQ(g.begin_exclusive(0ms), Exclusive::Acquired) << "a retry succeeds once the readers left";
    g.end_exclusive();
}

// (c) A request built against load N is refused at submit once the load is N+1.
TEST(ModelLifetimeGate, SnapshotOfAnOlderEpochIsRefusedAtSubmit) {
    ModelLifetimeGate g;
    const uint64_t snapshot_epoch = g.epoch();
    EXPECT_TRUE(g.is_current(snapshot_epoch));

    ASSERT_EQ(g.begin_exclusive(0ms), Exclusive::Acquired);
    g.bump_epoch();
    g.end_exclusive();

    EXPECT_FALSE(g.is_current(snapshot_epoch)) << "tokens of load N must not reach load N+1";
    auto lease = acquire_model_lease(g);
    ASSERT_TRUE(lease);
    EXPECT_EQ(lease->epoch(), snapshot_epoch + 1);
    EXPECT_TRUE(lease->current());
}

// (d) From "swap pending" to the end of the swap no reader gets in, also while the swap still waits.
TEST(ModelLifetimeGate, NoReaderIsAdmittedBetweenPendingAndTheBump) {
    ModelLifetimeGate g;
    auto inside = acquire_model_lease(g);
    ASSERT_TRUE(inside);

    std::atomic<bool> done{false};
    std::thread swapper([&] {
        ASSERT_EQ(g.begin_exclusive(10s), Exclusive::Acquired);
        done.store(true);
    });
    ASSERT_TRUE(eventually([&] { return g.exclusive_pending(); }));
    EXPECT_FALSE(acquire_model_lease(g)) << "admitted while the swap waits for its readers";
    EXPECT_EQ(g.readers(), 1) << "a refused reader must not be counted";
    EXPECT_FALSE(done.load());

    inside.reset();
    swapper.join();
    EXPECT_FALSE(acquire_model_lease(g)) << "admitted between the grant and the epoch bump";
    g.bump_epoch();
    EXPECT_FALSE(acquire_model_lease(g)) << "admitted before the swap ended";
    g.end_exclusive();

    auto after = acquire_model_lease(g);
    ASSERT_TRUE(after);
    EXPECT_EQ(after->epoch(), 1u);
}

TEST(ModelLifetimeGate, SecondExclusiveCallerIsBusyNotQueued) {
    ModelLifetimeGate g;
    ASSERT_EQ(g.begin_exclusive(0ms), Exclusive::Acquired);
    EXPECT_EQ(g.begin_exclusive(10s), Exclusive::Busy) << "two swaps must not wait on each other";
    EXPECT_TRUE(g.exclusive_pending()) << "the refused caller must not clear the holder's mark";
    g.end_exclusive();
}

// The server's shape: the swapper owns state.mtx, and a reader needs that mutex once more (its
// submit) before it can leave. Waiting with the mutex held would run into the timeout every time.
TEST(ModelLifetimeGate, ReleasesTheCallersLockWhileWaitingAndOwnsItAfter) {
    ModelLifetimeGate g;
    std::timed_mutex state_mtx;
    auto lease = acquire_model_lease(g);
    ASSERT_TRUE(lease);

    std::atomic<bool> reader_submitted{false};
    std::thread reader([&] {
        ASSERT_TRUE(eventually([&] { return g.exclusive_pending(); }));
        {
            std::lock_guard<std::timed_mutex> lock(state_mtx);  // the submit
            reader_submitted.store(lease->current());
        }
        lease.reset();
    });

    Exclusive r = Exclusive::Busy;
    {
        std::lock_guard<std::timed_mutex> lock(state_mtx);
        r = g.begin_exclusive(10s, &state_mtx);
        bool owned = false;
        std::thread([&] {
            owned = !state_mtx.try_lock();
            if (!owned)
                state_mtx.unlock();
        }).join();
        EXPECT_TRUE(owned) << "the caller's lock must be owned again on return";
    }
    reader.join();
    EXPECT_EQ(r, Exclusive::Acquired);
    EXPECT_TRUE(reader_submitted.load()) << "an admitted request still submits during the wait";
    EXPECT_EQ(g.readers(), 0);
    g.end_exclusive();
}

TEST(ModelLifetimeGate, ConcurrentReadersNeverOverlapAnExclusiveHolder) {
    // 8 reader threads against 200 swaps: a reader inside while a swap holds the model would be
    // the use-after-free. `holder` is the model: 1 while loaded, 0 while "freed".
    ModelLifetimeGate g;
    std::atomic<int> holder{1};
    std::atomic<int> violations{0};
    std::atomic<bool> stop{false};
    std::vector<std::thread> readers;
    for (int i = 0; i < 8; i++) {
        readers.emplace_back([&] {
            while (!stop.load()) {
                if (auto lease = acquire_model_lease(g)) {
                    if (holder.load() != 1 || !lease->current())
                        violations.fetch_add(1);
                    std::this_thread::yield();
                    if (holder.load() != 1)
                        violations.fetch_add(1);
                }
            }
        });
    }
    int swaps = 0;
    for (int i = 0; i < 200; i++) {
        if (g.begin_exclusive(5s) != Exclusive::Acquired)
            continue;
        holder.store(0);
        g.bump_epoch();
        std::this_thread::yield();
        holder.store(1);
        g.end_exclusive();
        swaps++;
    }
    stop.store(true);
    for (auto& t : readers)
        t.join();
    EXPECT_EQ(swaps, 200);
    EXPECT_EQ(violations.load(), 0);
    EXPECT_EQ(g.readers(), 0);
}
