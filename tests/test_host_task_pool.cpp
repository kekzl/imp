// HostTaskPool (memory/host_task_pool.h): every submitted task runs, wait() and the
// destructor join all of them. CPU lane: plain memcpy on threads, no CUDA.

#include <gtest/gtest.h>

#include "memory/host_task_pool.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <vector>

using namespace imp;

namespace {

std::vector<uint8_t> pattern(size_t n, uint8_t seed) {
    std::vector<uint8_t> v(n);
    for (size_t i = 0; i < n; ++i)
        v[i] = static_cast<uint8_t>(seed + i * 31);
    return v;
}

void submit_copies(HostTaskPool& pool, std::vector<uint8_t>& dst, const std::vector<uint8_t>& src,
                   size_t chunk) {
    for (size_t off = 0; off < src.size(); off += chunk) {
        const size_t n = std::min(chunk, src.size() - off);
        pool.submit([d = dst.data() + off, s = src.data() + off, n] { std::memcpy(d, s, n); });
    }
}

}  // namespace

TEST(HostTaskPool, WaitSeesEveryTaskRun) {
    const std::vector<uint8_t> src = pattern(257 * 4099, 7);
    std::vector<uint8_t> dst(src.size(), 0);
    HostTaskPool pool(4);
    submit_copies(pool, dst, src, 4099);
    pool.wait();
    EXPECT_EQ(dst, src);

    // Usable after wait(): a second round lands too.
    const std::vector<uint8_t> src2 = pattern(src.size(), 99);
    submit_copies(pool, dst, src2, 4099);
    pool.wait();
    EXPECT_EQ(dst, src2);
}

TEST(HostTaskPool, DestructorJoinsPendingTasks) {
    const std::vector<uint8_t> src = pattern(1 << 20, 3);
    std::vector<uint8_t> dst(src.size(), 0);
    std::atomic<int> ran{0};
    {
        HostTaskPool pool(3);
        submit_copies(pool, dst, src, 1000);
        for (int i = 0; i < 100; ++i)
            pool.submit([&ran] { ran.fetch_add(1); });
    }
    EXPECT_EQ(dst, src);
    EXPECT_EQ(ran.load(), 100);
}

TEST(HostTaskPool, ZeroThreadsStillRuns) {
    const std::vector<uint8_t> src = pattern(64, 1);
    std::vector<uint8_t> dst(64, 0);
    HostTaskPool pool(0);
    submit_copies(pool, dst, src, 64);
    pool.wait();
    EXPECT_EQ(dst, src);
}
