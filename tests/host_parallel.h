#pragma once
// Host reference loops spread over independent outputs: index i runs the serial body on one
// thread, so every output keeps its own summation order (bit-identical to the serial loop).

#include <algorithm>
#include <atomic>
#include <thread>
#include <vector>

namespace imp_test {

template <class Fn>
void parallel_for(int n, Fn&& fn) {
    const int hw = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    const int nt = std::min(n, hw);
    if (nt <= 1) {
        for (int i = 0; i < n; ++i)
            fn(i);
        return;
    }
    std::atomic<int> next{0};
    std::vector<std::thread> pool;
    pool.reserve(static_cast<size_t>(nt));
    for (int t = 0; t < nt; ++t)
        pool.emplace_back([&] {
            for (int i = next.fetch_add(1); i < n; i = next.fetch_add(1))
                fn(i);
        });
    for (auto& th : pool)
        th.join();
}

}  // namespace imp_test
