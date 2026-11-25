#pragma once

// What /metrics reads of an engine (AUDIT_ARCH_2026_11_24 C-13): the stepping thread copies the
// counters it mutates into one value struct per step; a scrape copies that struct and touches no
// live engine field. Header-only, no CUDA, so the CPU lane drives the cell with threads.

#include <cstdint>
#include <mutex>

namespace imp {

// Speculative-decode counters per draft source (#1321): `mtp` is the trained head, `other` every
// remaining drafter (n-gram/suffix matcher, prediction, recycling).
struct SpecSourceStats {
    long long verify_steps = 0;
    long long drafted = 0;
    long long accepted = 0;
    long long emitted = 0;
    double verify_wall_ms = 0;
};
struct SpecStats {
    long long verify_steps = 0;  // verify forwards run
    long long miss_steps = 0;    // decode steps with no usable draft
    long long drafted = 0;       // draft tokens proposed
    long long accepted = 0;      // draft tokens accepted
    long long emitted = 0;       // tokens emitted by verify steps
    long long sampled = 0;       // verify steps that drew rows (speculative.sampled)
    double verify_wall_ms = 0;   // host wall inside step_spec_verify_ (verify steps only)
    SpecSourceStats mtp{};       // verify steps the MTP head drafted
    SpecSourceStats other{};     // every non-MTP drafter
};

// Trivially copyable: one memcpy under the cell lock on each side.
struct EngineMetricsSnapshot {
    uint64_t publishes = 0;  // set by the cell, 1 after the first publish
    uint64_t kv_pressure_rejections = 0;
    uint64_t kv_pool_growths = 0;
    uint64_t streaming_kv_auto_enables = 0;
    uint64_t prefix_cache_evictions = 0;
    uint64_t mixed_decode_steps = 0;
    uint64_t kv_swaps_out = 0;

    bool has_kv = false;  // KV pool present
    int kv_total_blocks = 0;
    int kv_free_blocks = 0;
    double kv_bandwidth_gbps = 0.0;
    bool has_kv_manager = false;
    int kv_cached_blocks = 0;
    int kv_reclaimable_blocks = 0;
    int kv_pinned_blocks = 0;
    int kv_session_pinned_blocks = 0;
    int kv_sessions = 0;
    int kv_reserved_blocks = 0;

    bool has_host_spill = false;  // kv_cache.host_spill_mb tier present
    uint64_t spill_saves = 0;
    uint64_t spill_restores = 0;
    uint64_t spill_replacements = 0;
    uint64_t spill_pinned_bytes = 0;

    bool has_recurrent_snapshots = false;
    uint64_t snap_saves = 0;
    uint64_t snap_demotions = 0;
    uint64_t snap_evictions = 0;
    uint64_t snap_restores_device = 0;
    uint64_t snap_restores_host = 0;
    uint64_t snap_misses_absent = 0;
    uint64_t snap_misses_stale = 0;
    uint64_t snap_unused_tokens = 0;

    // MoE imbalance accumulator (#1548): device pointer, stable from its lazy allocation until
    // ~GraphExecutor. The scrape copies it under a model lease, which keeps the executor alive.
    const unsigned int* moe_imb_acc = nullptr;
    int moe_imb_layers = 0;
    int moe_imb_experts = 0;

    SpecStats spec{};
};

// One writer at a time (the thread that steps the engine), any number of readers. Each side holds
// the mutex for one struct copy only, so a scrape never stalls a step for longer than that copy.
class EngineMetricsCell {
public:
    void publish(const EngineMetricsSnapshot& s) {
        std::lock_guard<std::mutex> lock(mu_);
        snap_ = s;
        snap_.publishes = ++publishes_;
    }
    EngineMetricsSnapshot read() const {
        std::lock_guard<std::mutex> lock(mu_);
        return snap_;
    }

private:
    mutable std::mutex mu_;
    EngineMetricsSnapshot snap_{};
    uint64_t publishes_ = 0;
};

}  // namespace imp
