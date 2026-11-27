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
            .streaming_kv_auto = ec.streaming_kv_auto,
            .use_fp8_prefill = ec.use_fp8_prefill,
            .use_nvfp4_decode = ec.use_nvfp4_decode,
            .dual_path_quant = ec.dual_path_quant,
            .ssm_state_dtype = ec.ssm_state_dtype,
            .kv_block_size = ec.kv_block_size.value_or(0),
            .library_reserve_mb = ec.library_reserve_mb.value_or(-1),
            .use_prefix_caching = ec.use_prefix_caching,
            .swa_shared_tokens = ec.swa_shared_tokens,
            .kv_ceiling_seqs = ec.kv_ceiling_seqs,
            .nvfp4_decode_all = ec.nvfp4_decode_all,
            .fp8_lm_head = ec.fp8_lm_head,
            .lm_head_source_only = ec.lm_head_source_only,
            .q8_drop_source = ec.q8_drop_source,
            .q4k_in_place = ec.q4k_in_place};
}

EngineConfig resolved_engine_config(const EngineConfig& config, const EngineRuntimeState& rt) {
    EngineConfig c = config;
    c.max_batch_size = rt.max_batch_size;
    c.max_seq_len = rt.max_seq_len;
    c.kv_cache_dtype = rt.kv_cache_dtype;
    c.streaming_kv_auto = rt.streaming_kv_auto;
    c.use_fp8_prefill = rt.use_fp8_prefill;
    c.use_nvfp4_decode = rt.use_nvfp4_decode;
    c.dual_path_quant = rt.dual_path_quant;
    c.ssm_state_dtype = rt.ssm_state_dtype;
    c.kv_block_size = rt.kv_block_size;
    c.library_reserve_mb = rt.library_reserve_mb;
    c.use_prefix_caching = rt.use_prefix_caching;
    c.swa_shared_tokens = rt.swa_shared_tokens;
    c.kv_ceiling_seqs = rt.kv_ceiling_seqs;
    c.nvfp4_decode_all = rt.nvfp4_decode_all;
    c.fp8_lm_head = rt.fp8_lm_head;
    c.lm_head_source_only = rt.lm_head_source_only;
    c.q8_drop_source = rt.q8_drop_source;
    c.q4k_in_place = rt.q4k_in_place;
    return c;
}

}  // namespace imp
