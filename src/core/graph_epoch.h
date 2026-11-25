#pragma once

#include <atomic>
#include <cstdint>

namespace imp {

// Invariant I3 (docs/internals/MEMORY.md): one process-wide counter for "an address or a topology a
// captured CUDA graph baked changed". An owner bumps it, every graph runner stores it at capture
// and compares it before replay: 1 load + 1 compare per launch, no lock, no allocation.
namespace detail {
inline std::atomic<uint64_t> g_graph_epoch{1};  // 0 is reserved: "captured under no epoch"
}  // namespace detail

[[nodiscard]] inline uint64_t graph_epoch() noexcept {
    return detail::g_graph_epoch.load(std::memory_order_relaxed);
}

// Returns the new epoch. Called by the owner of a moved buffer and by Engine::invalidate_forward_graphs.
inline uint64_t bump_graph_epoch() noexcept {
    return detail::g_graph_epoch.fetch_add(1, std::memory_order_relaxed) + 1;
}

// I3 regrow rule: the owner passes the buffer address before and after. Bumps only when a live
// address went away (before non-null and != after): 0 bumps on a first allocation or a reuse.
[[nodiscard]] inline bool bump_graph_epoch_if_moved(const void* before, const void* after) noexcept {
    if (before == nullptr || before == after)
        return false;
    (void)bump_graph_epoch();
    return true;
}

}  // namespace imp
