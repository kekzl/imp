#pragma once

#include "core/log_once.h"
#include "core/logging.h"

#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

namespace imp {

// A designed fallback (ROADMAP B-108): the caller survives on a named alternative path.
// Every hit counts into /metrics `imp_fallback_total{site=...}`; the first hit per model load
// logs one WARN (the LogOnce member re-arms with every other latch). Static storage only.
class FallbackSite {
public:
    explicit FallbackSite(const char* name) noexcept;
    FallbackSite(const FallbackSite&) = delete;
    FallbackSite& operator=(const FallbackSite&) = delete;

    // Counts one hit; true on the first hit since the last model load (the caller logs then).
    [[nodiscard]] bool hit() noexcept {
        count_.fetch_add(1, std::memory_order_relaxed);
        return once_.first();
    }
    [[nodiscard]] const char* name() const noexcept { return name_; }
    [[nodiscard]] uint64_t count() const noexcept { return count_.load(std::memory_order_relaxed); }

private:
    friend struct FallbackRegistry;
    const char* name_;
    std::atomic<uint64_t> count_{0};
    LogOnce once_;
    FallbackSite* next_ = nullptr;
};

struct FallbackCount {
    std::string site;
    uint64_t count = 0;
};

// Every site constructed so far, summed per name and sorted by it (sites may share a name).
[[nodiscard]] std::vector<FallbackCount> fallback_counts();

}  // namespace imp

// Counts the fallback at `site` (a static imp::FallbackSite) and logs its WARN once per model
// load. The message names what is skipped, why, and what runs instead.
#define IMP_FALLBACK_WARN(site, ...)   \
    do {                               \
        if ((site).hit())              \
            IMP_LOG_WARN(__VA_ARGS__); \
    } while (0)
