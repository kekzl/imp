#pragma once

// Recurrent snapshot host tier sized with the KV host spill tier: on a GDN hybrid a spilled
// prefix restores only with its snapshot, so KV kept without one prefills again anyway.

#include "core/logging.h"
#include "runtime/config.h"

#include <algorithm>
#include <cstddef>

namespace imp {

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
