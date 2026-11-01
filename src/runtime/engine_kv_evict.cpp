// KV eviction proof harness, engine half (kv_cache.evict_keep_ratio): the init gate and the
// post-prefill score -> select -> free step. Design: docs/plans/2026-11-01-kv-eviction.md.

#include "runtime/engine_kv_evict.h"
#include "core/logging.h"
#include "exec/executor.h"
#include "memory/kv_cache.h"
#include "memory/kv_cache_manager.h"
#include "memory/kv_evict_select.h"
#include "model/model_config.h"
#include "runtime/config.h"
#include <vector>

namespace imp {

namespace {

// Model geometry the harness cannot serve, nullptr = fine.
const char* kv_evict_model_refusal(const ModelConfig& m) {
    if (m.ssm_inner_size > 0 || m.ssm_state_size > 0)
        return "hybrid / recurrent layers";
    if (m.sliding_window > 0)
        return "sliding-window layers";
    if (m.is_mla())
        return "MLA";
    const int hd = m.head_dim > 0 ? m.head_dim : m.d_model / m.n_heads;
    if (m.has_mrope() || m.yarn_ext_factor > 0.0f || m.rope_freq_scale != 1.0f || !m.rope_neox ||
        (m.rope_dim > 0 && m.rope_dim != hd))
        return "RoPE variant (needs plain NeoX RoPE over the full head_dim)";
    if (!m.head_dim_per_layer.empty())
        return "per-layer KV geometry";
    return nullptr;
}

// Runtime settings whose paths would see the -1 holes or break the query statistics.
const char* kv_evict_runtime_refusal(const RuntimeConfig& rc, QType dt, bool streaming_kv) {
    if (dt != QType::F16 && dt != QType::FP8_E4M3 && dt != QType::NVFP4)
        return "KV dtype (F16, FP8 or NVFP4 only)";
    if (rc.speculative.ngram || rc.speculative.token_recycling || rc.speculative.mtp_k > 0)
        return "speculation (set speculative.ngram=false, speculative.mtp_k=0)";
    if (streaming_kv)
        return "StreamingLLM";
    if (rc.kv_cache.bitdecoding_residual_tokens > 0)
        return "residual KV";
    if (rc.attention.sparse_topk_tokens > 0)
        return "sparse decode attention (set attention.sparse_topk_tokens=0)";
    if (rc.runtime.prefill_batch)
        return "ragged prefill (set runtime.prefill_batch=false)";
    return nullptr;
}

}  // namespace

void kv_evict_resolve(RuntimeConfig& rc, const ModelConfig& m, QType kv_dtype, bool streaming_kv) {
    auto& kc = rc.kv_cache;
    if (kc.evict_keep_ratio <= 0.0f)
        return;
    const char* refuse = kc.evict_keep_ratio >= 1.0f ? "keep ratio >= 1 keeps every block"
                                                     : kv_evict_model_refusal(m);
    if (!refuse)
        refuse = kv_evict_runtime_refusal(rc, kv_dtype, streaming_kv);
    if (refuse) {
        IMP_LOG_WARN("kv_cache.evict_keep_ratio=%.2f ignored: %s", static_cast<double>(kc.evict_keep_ratio),
                     refuse);
        kc.evict_keep_ratio = 0.0f;
        return;
    }
    IMP_LOG_INFO("kv_cache.evict_keep_ratio=%.2f: sink %d, recent %d, min prompt %d tokens",
                 static_cast<double>(kc.evict_keep_ratio), kc.evict_sink_tokens, kc.evict_recent_tokens,
                 kc.evict_min_tokens);
}

void kv_evict_after_prefill(GraphExecutor& ex, KVCacheManager& mgr, KVCache* cache, const cfg::KVCache& kc,
                            int req_id, int prompt_len, bool mtp_active, cudaStream_t stream) {
    if (!ex.kv_evict_enabled() || prompt_len < kc.evict_min_tokens || cache == nullptr)
        return;
    if (mtp_active) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            IMP_LOG_WARN("kv evict: MTP speculation active - not evicting (set speculative.mtp_k=0)");
        }
        return;
    }
    const KvEvictParams p{kc.evict_keep_ratio, kc.evict_sink_tokens, kc.evict_recent_tokens,
                          cache->block_size()};
    const KvEvictRange r = kv_evict_candidates(prompt_len, p);
    if (r.end <= r.begin)
        return;
    const std::vector<int>& bt = mgr.block_table(req_id);
    std::vector<float> scores;
    if (!ex.kv_evict_score_prompt(cache, bt, prompt_len, r.begin, r.end, scores, stream))
        return;
    const std::vector<int> evict = kv_evict_select(scores, prompt_len, p);
    const int freed = mgr.evict_blocks(req_id, evict);
    const int n_blocks = (prompt_len + p.block_size - 1) / p.block_size;
    IMP_LOG_INFO("kv evict: req %d prompt %d tokens, freed %d of %d blocks (selected %zu, keep %.2f)", req_id,
                 prompt_len, freed, n_blocks, evict.size(), static_cast<double>(kc.evict_keep_ratio));
}

}  // namespace imp
