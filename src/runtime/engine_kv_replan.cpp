#include "runtime/engine.h"
#include "core/logging.h"

#include <algorithm>

namespace imp {

// Re-size the KV pool at the library reserve warmup just measured, to what a start that found
// it cached builds (#2436). Runs before graph prewarm: raise_ceiling moves every KV pointer.
// Growth goes through try_grow_to, so it never commits past free VRAM above the headroom.
void Engine::apply_measured_library_reserve_() {
    if (measured_library_reserve_ == SIZE_MAX || kv_cache_raw_ == nullptr || !kv_manager_)
        return;
    // Whole MiB, as the cache stores it and the next start charges it.
    const size_t measured = (measured_library_reserve_ >> 20) << 20;
    kv_replan_.initial_pct = runtime_config_.kv_cache.growable_initial_pct;
    const KvReplan r = replan_kv_for_measured_reserve(kv_replan_, measured);
    const size_t charged = kv_replan_.probe.library_reserve_bytes;
    kv_replan_.armed = false;  // once per start
    if (!r.apply)
        return;
    const int ceiling_before = kv_cache_raw_->ceiling_blocks();
    const int commit_before = kv_cache_raw_->total_blocks();
    if (r.ceiling_blocks > ceiling_before && kv_cache_raw_->raise_ceiling(r.ceiling_blocks)) {
        prefill_graph_runner_.invalidate();
        (void)kv_cache_raw_->probe_residency();
    }
    const int commit_after = kv_cache_raw_->try_grow_to(r.commit_blocks);
    if (config_.use_prefix_caching) {
        const int pin_pct = std::clamp(config_.prefix_pin_budget_pct, 0, 100);
        if (pin_pct > 0)
            kv_manager_->set_pin_budget_blocks(std::max(1, commit_after * pin_pct / 100));
    }
    IMP_LOG_INFO(
        "library reserve: applied this start's %.0f MiB measurement in place of the %.0f MiB charge: "
        "KV plan %d -> %d blocks, committed %d -> %d (target %d), ceiling %d -> %d",
        measured / (1024.0 * 1024.0), charged / (1024.0 * 1024.0), kv_replan_.plan_blocks, r.plan_blocks,
        commit_before, commit_after, r.commit_blocks, ceiling_before, kv_cache_raw_->ceiling_blocks());
}

}  // namespace imp
