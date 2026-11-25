// engine_metrics_snapshot.cpp: the one place the stepping thread copies what /metrics reports (C-13).

#include "runtime/engine.h"

#include "exec/executor.h"
#include "memory/kv_host_spill.h"
#include "memory/recurrent_snapshot_store.h"

namespace imp {

void Engine::publish_metrics_snapshot() {
    EngineMetricsSnapshot s;
    s.kv_pressure_rejections = kv_pressure_rejections();
    s.kv_pool_growths = kv_pool_growths();
    s.streaming_kv_auto_enables = streaming_kv_auto_enables();
    s.prefix_cache_evictions = prefix_cache_evictions();
    s.mixed_decode_steps = mixed_decode_steps();
    s.kv_swaps_out = kv_swaps_out();
    if (kv_cache_raw_) {
        s.has_kv = true;
        s.kv_total_blocks = kv_cache_raw_->total_blocks();
        s.kv_free_blocks = kv_cache_raw_->num_free_blocks();
        s.kv_bandwidth_gbps = kv_cache_raw_->residency_gbps();
    }
    if (kv_manager_) {
        s.has_kv_manager = true;
        s.kv_cached_blocks = kv_manager_->num_cached_blocks();
        s.kv_reclaimable_blocks = kv_manager_->num_reclaimable_cached_blocks();
        s.kv_pinned_blocks = kv_manager_->num_pinned_blocks();
        s.kv_session_pinned_blocks = kv_manager_->num_session_pinned_blocks();
        s.kv_sessions = kv_manager_->num_sessions();
        s.kv_reserved_blocks = kv_manager_->outstanding_reserved_blocks();
        if (const KVHostSpill* spill = kv_manager_->host_spill()) {
            s.has_host_spill = true;
            s.spill_saves = spill->saves();
            s.spill_restores = spill->restores();
            s.spill_replacements = spill->replacements();
            s.spill_pinned_bytes = static_cast<uint64_t>(spill->pinned_slots()) * spill->slot_bytes();
        }
    }
    if (const RecurrentSnapshotStore* snaps = recurrent_snapshots_.get()) {
        s.has_recurrent_snapshots = true;
        s.snap_saves = snaps->saves();
        s.snap_demotions = snaps->demotions();
        s.snap_evictions = snaps->evictions();
        const auto& h = snaps->reuse();
        s.snap_restores_device = h.restores_device.load(std::memory_order_relaxed);
        s.snap_restores_host = h.restores_host.load(std::memory_order_relaxed);
        s.snap_misses_absent = h.misses_absent.load(std::memory_order_relaxed);
        s.snap_misses_stale = h.misses_stale.load(std::memory_order_relaxed);
        s.snap_unused_tokens = h.unused_tokens.load(std::memory_order_relaxed);
    }
    if (executor_) {
        const GraphExecutor::MoeImbalanceSource m = executor_->moe_imbalance_source();
        s.moe_imb_acc = m.acc;
        s.moe_imb_layers = m.layers;
        s.moe_imb_experts = m.experts;
    }
    s.spec = spec_stats_;
    metrics_cell_.publish(s);
}

}  // namespace imp
