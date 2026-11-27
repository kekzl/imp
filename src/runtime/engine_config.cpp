#include "runtime/engine_config.h"

#include "runtime/config.h"

#include <algorithm>

namespace imp {

// imp.conf [server]/[paths]/[vram]/[kv_cache] keys reach EngineConfig here (#541). Every bridged
// field is optional and takes imp.conf only when unset; use_prefix_caching is not bridged
// (#758, #1299/#1314). A negative kv_cache key stays auto (0), as before.
void bridge_runtime_config(EngineConfig& ec, const RuntimeConfig& rc) {
    if (!ec.prefix_pin_budget_pct)
        ec.prefix_pin_budget_pct = rc.server.prefix_pin_budget_pct;
    if (!ec.mmproj_path)
        ec.mmproj_path = rc.paths.mmproj;
    if (!ec.vram_budget_mb)
        ec.vram_budget_mb = static_cast<size_t>(std::max(rc.runtime.vram_budget_mb, 0));
    if (!ec.kv_fraction)
        ec.kv_fraction = rc.vram.kv_fraction;
    if (!ec.vram_reserve_floor_pct)
        ec.vram_reserve_floor_pct = rc.vram.reserve_floor_pct;
    if (!ec.library_reserve_mb)
        ec.library_reserve_mb = rc.vram.library_reserve_mb;
    if (!ec.kv_cache_max_blocks)
        ec.kv_cache_max_blocks = std::max(rc.kv_cache.max_blocks, 0);
    if (!ec.kv_block_size)
        ec.kv_block_size = std::max(rc.kv_cache.block_size, 0);
}

EngineRuntimeState seed_runtime_state(const EngineConfig& ec) {
    return {.cuda_graphs = ec.use_cuda_graphs,
            .streaming_kv = ec.streaming_kv_enabled,
            .streaming_kv_window = ec.streaming_kv_window,
            .max_batch_size = ec.max_batch_size,
            .max_seq_len = ec.max_seq_len,
            .kv_cache_dtype = ec.kv_cache_dtype,
            .streaming_kv_auto = ec.streaming_kv_auto};
}

EngineConfig resolved_engine_config(const EngineConfig& config, const EngineRuntimeState& rt) {
    EngineConfig c = config;
    c.max_batch_size = rt.max_batch_size;
    c.max_seq_len = rt.max_seq_len;
    c.kv_cache_dtype = rt.kv_cache_dtype;
    c.streaming_kv_auto = rt.streaming_kv_auto;
    return c;
}

}  // namespace imp
