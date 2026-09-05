#include "exec/gemm_kernel_registry.h"
#include <utility>
#include "core/logging.h"

namespace imp {

// ---------------------------------------------------------------------------
// Singleton storage
// ---------------------------------------------------------------------------
GemmKernelRegistry& GemmKernelRegistry::instance() {
    static GemmKernelRegistry registry;
    return registry;
}

void GemmKernelRegistry::register_kernel(GemmStrategy strategy, GemmKernelFn fn) {
    IMP_CHECK(fn != nullptr, "GemmKernelRegistry::register_kernel: fn is null");
    // Check for existing entry (slice 1 expects at most one per strategy).
    for (std::size_t i = 0; i < count_; ++i) {
        if (entries_[i].strategy == strategy) {
            IMP_LOG_WARN(
                "GemmKernelRegistry: overriding existing entry (tier=%d qtype=%d m_is_one=%d)",
                std::to_underlying(strategy.tier), std::to_underlying(strategy.weight_qtype),
                strategy.m_is_one ? 1 : 0);
            entries_[i].fn = fn;
            return;
        }
    }
    IMP_CHECK(count_ < (sizeof(entries_) / sizeof(entries_[0])),
              "GemmKernelRegistry: capacity exceeded (max=%zu)",
              sizeof(entries_) / sizeof(entries_[0]));
    entries_[count_].strategy = strategy;
    entries_[count_].fn = fn;
    ++count_;
}

GemmDispatchResult GemmKernelRegistry::dispatch(const GemmStrategy& strategy,
                                                const GemmKernelArgs& args) const {
    for (std::size_t i = 0; i < count_; ++i) {
        if (entries_[i].strategy == strategy)
            return entries_[i].fn(args);
    }
    return GemmDispatchResult::NoMatch;
}

std::size_t GemmKernelRegistry::size() const noexcept { return count_; }
}  // namespace imp
