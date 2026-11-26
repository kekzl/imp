#include "core/fallback_site.h"

#include <algorithm>
#include <cstring>

namespace imp {

namespace {

// constinit: set before any dynamic init, so a file-scope site of another TU can register.
constinit std::atomic<FallbackSite*> g_head{nullptr};

}  // namespace

struct FallbackRegistry {
    static FallbackSite* head() noexcept { return g_head.load(std::memory_order_acquire); }
    static FallbackSite* next(const FallbackSite* s) noexcept { return s->next_; }
    static void link(FallbackSite* s) noexcept {
        FallbackSite* h = g_head.load(std::memory_order_relaxed);
        do {
            s->next_ = h;
        } while (!g_head.compare_exchange_weak(h, s, std::memory_order_release, std::memory_order_relaxed));
    }
};

FallbackSite::FallbackSite(const char* name) noexcept : name_(name) { FallbackRegistry::link(this); }

std::vector<FallbackCount> fallback_counts() {
    std::vector<FallbackCount> out;
    // Sites have static storage and a trivial destructor: the list never shrinks or dangles.
    for (const FallbackSite* s = FallbackRegistry::head(); s; s = FallbackRegistry::next(s)) {
        auto it = std::find_if(out.begin(), out.end(), [s](const FallbackCount& c) {
            return std::strcmp(c.site.c_str(), s->name()) == 0;
        });
        if (it == out.end())
            out.push_back({s->name(), s->count()});
        else
            it->count += s->count();
    }
    std::sort(out.begin(), out.end(),
              [](const FallbackCount& a, const FallbackCount& b) { return a.site < b.site; });
    return out;
}

}  // namespace imp
