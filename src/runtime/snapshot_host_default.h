#pragma once

// Recurrent snapshot host tier sized with the KV host spill tier: on a GDN hybrid a spilled
// prefix restores only with its snapshot, so KV kept without one prefills again anyway.

#include "core/logging.h"
#include "runtime/config.h"

#include <algorithm>
#include <cstddef>

namespace imp {

// kv_cache.host_spill_mb=auto: the spill tier takes two thirds of the host RAM above the pin headroom
// and the snapshot tier (spill / 2) the rest; whole GiB, 0 below 1 GiB or when the host figure is unknown.
inline constexpr int kHostSpillAutoMaxMb = 16384;
[[nodiscard]] constexpr int host_spill_auto_mb(size_t avail_bytes, size_t headroom_bytes) {
    if (avail_bytes <= headroom_bytes)
        return 0;
    const size_t spill_gib = ((avail_bytes - headroom_bytes) >> 30) * 2 / 3;
    return static_cast<int>(std::min(spill_gib << 10, static_cast<size_t>(kHostSpillAutoMaxMb)));
}

// Resolves kv_cache.host_spill_mb < 0 in place; true when it was auto.
[[nodiscard]] inline bool apply_host_spill_auto(RuntimeConfig& rc, bool prefix_caching, size_t avail_bytes,
                                                size_t headroom_bytes) {
    if (rc.kv_cache.host_spill_mb >= 0)
        return false;
    rc.kv_cache.host_spill_mb = prefix_caching ? host_spill_auto_mb(avail_bytes, headroom_bytes) : 0;
    IMP_LOG_INFO("kv_cache.host_spill_mb: auto -> %d (host RAM available %.1f GiB, %.1f GiB kept free%s)",
                 rc.kv_cache.host_spill_mb, avail_bytes / (1024.0 * 1024.0 * 1024.0),
                 headroom_bytes / (1024.0 * 1024.0 * 1024.0), prefix_caching ? "" : ", prefix caching off");
    return true;
}

// Snapshot host MiB per KV host spill MiB: 16384 spill -> 8192 (103 slabs on Qwen3.8-27B).
inline constexpr int kSnapshotHostPerSpillDivisor = 2;

// server.recurrent_snapshot_host_mb follows kv_cache.host_spill_mb / 2 unless the operator set it.
inline void apply_snapshot_host_spill_default(RuntimeConfig& rc) {
    if (rc.kv_cache.host_spill_mb <= 0 || rc.was_set("server.recurrent_snapshot_host_mb"))
        return;
    const int want = rc.kv_cache.host_spill_mb / kSnapshotHostPerSpillDivisor;
    if (want <= rc.server.recurrent_snapshot_host_mb)
        return;
    rc.server.recurrent_snapshot_host_mb = want;
    IMP_LOG_INFO("server.recurrent_snapshot_host_mb: auto -> %d (kv_cache.host_spill_mb=%d)", want,
                 rc.kv_cache.host_spill_mb);
}

// Host tier bytes for RecurrentSnapshotStore::init, the default above applied first.
inline size_t snapshot_host_budget_bytes(RuntimeConfig& rc) {
    apply_snapshot_host_spill_default(rc);
    return static_cast<size_t>(std::max(rc.server.recurrent_snapshot_host_mb, 0)) << 20;
}

}  // namespace imp
