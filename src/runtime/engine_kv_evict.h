#pragma once

// KV eviction proof harness, engine half (kv_cache.evict_keep_ratio, engine_kv_evict.cpp).
// Design: docs/plans/2026-11-01-kv-eviction.md.

#include "core/config/kv_cache.h"
#include "core/qtype.h"
#include <cuda_runtime.h>

namespace imp {

struct ModelConfig;
struct RuntimeConfig;
class GraphExecutor;
class KVCacheManager;
class KVCache;

// Init gate, after the sparse resolver: zeroes evict_keep_ratio with a WARN where the harness cannot run.
void kv_evict_resolve(RuntimeConfig& rc, const ModelConfig& m, QType kv_dtype, bool streaming_kv);

// After a request's last prefill chunk: score, select and free its prompt blocks (no-op when off).
void kv_evict_after_prefill(GraphExecutor& ex, KVCacheManager& mgr, KVCache* cache, const cfg::KVCache& kc,
                            int req_id, int prompt_len, bool mtp_active, cudaStream_t stream);

}  // namespace imp
