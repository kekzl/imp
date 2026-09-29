#pragma once

// Submission order of /v1/decide and /v1/score items (#2198). A wave is submitted at once; the next
// wave starts after every request of the previous one finished. Prefix blocks are published only
// at finish (Engine::finish_request_release_), so reuse needs the sharer in an earlier wave.

#include <cstddef>
#include <numeric>
#include <vector>

namespace imp::server {

enum class ScoreMode { Serial, Direct, Shared };

// serial: one item per wave. direct: one wave, prefix cache bypassed.
// shared: item 0 alone (publishes the evidence prefix), then items 1..n-1 as one wave.
inline std::vector<std::vector<size_t>> score_waves(ScoreMode mode, size_t n) {
    std::vector<std::vector<size_t>> waves;
    if (n == 0)
        return waves;
    if (mode == ScoreMode::Serial) {
        for (size_t i = 0; i < n; i++)
            waves.push_back({i});
        return waves;
    }
    const size_t first = mode == ScoreMode::Shared ? 1 : n;
    waves.emplace_back(first);
    std::iota(waves.back().begin(), waves.back().end(), size_t{0});
    if (first < n) {
        waves.emplace_back(n - first);
        std::iota(waves.back().begin(), waves.back().end(), first);
    }
    return waves;
}

}  // namespace imp::server
