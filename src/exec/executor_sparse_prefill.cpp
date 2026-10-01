// Sparse prefill attention (attention.sparse_prefill_topk_tokens): scratch at init and the
// per-chunk past-page pick the chunked prefill path calls. Kernels: sparse_attn_select.cu.

#include "core/dispatch_policy.h"
#include "core/logging.h"
#include "exec/executor.h"
#include "exec/sparse_attn_geometry.h"
#include "exec/sparse_attn_select.h"
#include "memory/engine_arena.h"
#include "memory/kv_cache.h"
#include <algorithm>

namespace imp {

namespace {
constexpr int kPrefillRows = 64;  // kMaxPrefillRows in sparse_attn_select.cu
}  // namespace

void GraphExecutor::allocate_sparse_prefill_scratch_() {
    const auto& acfg = dispatch_policy().attention;
    if (acfg.sparse_prefill_topk_tokens <= 0)
        return;
    const int max_ctx_tokens = (mla_absorb_max_seq_ > 0) ? mla_absorb_max_seq_ : max_tokens_;
    const SparseGeometry geo = sparse_geometry(acfg.sparse_prefill_topk_tokens, acfg.sparse_sink_tokens,
                                               acfg.sparse_prefill_recent_tokens, 0, max_ctx_tokens,
                                               kv_block_size_);
    const int cap = geo.max_ctx_blocks;
    auto sc = engine_arena().take_bytes((size_t)kPrefillRows * cap * sizeof(float));
    auto agg = engine_arena().take_bytes((size_t)cap * sizeof(float));
    auto tbl = engine_arena().take_bytes((size_t)cap * sizeof(int));
    auto ctx = engine_arena().take_bytes(2 * sizeof(int));  // [past len in, selected len out]
    if (sc.empty() || agg.empty() || tbl.empty() || ctx.empty()) {
        IMP_LOG_WARN("sparse prefill scratch unavailable from the T2 arena - feature disabled");
        return;
    }
    qscratch_.sp_prefill_scores = reinterpret_cast<float*>(sc.data());
    qscratch_.sp_prefill_agg = reinterpret_cast<float*>(agg.data());
    qscratch_.sp_prefill_table = reinterpret_cast<int*>(tbl.data());
    qscratch_.sp_prefill_ctx = reinterpret_cast<int*>(ctx.data());
    qscratch_.sp_prefill_budget_blocks = geo.budget_blocks;
    qscratch_.sp_prefill_sink_blocks = geo.sink_blocks;
    qscratch_.sp_prefill_recent_blocks = geo.recent_blocks;
    qscratch_.sp_prefill_cap_blocks = cap;
    qscratch_.sp_prefill_rows = std::clamp(acfg.sparse_prefill_rows, 1, kPrefillRows);
    // The metadata pass reads this flag too; set it when only the prefill side is on.
    qscratch_.sparse_score_meanstd = acfg.sparse_score_meanstd;
    qscratch_.sparse_score_std_coef = acfg.sparse_score_std_coef;
    IMP_LOG_INFO(
        "Sparse prefill attention: budget %d blocks (%d tokens), sink %d + recent %d blocks, "
        "%d query rows, score %s",
        geo.budget_blocks, geo.budget_blocks * kv_block_size_, geo.sink_blocks, geo.recent_blocks,
        qscratch_.sp_prefill_rows, acfg.sparse_score_meanstd ? "mean+std" : "minmax");
}

GraphExecutor::SparsePrefillPast GraphExecutor::sparse_prefill_pick_(const half* q, int n, KVCache* cache,
                                                                     int kv_layer, const int* table,
                                                                     int q_offset, int nh, int nkv, int hd,
                                                                     bool cap_replay, bool paged_kv_written,
                                                                     int sliding_window, const void* sinks,
                                                                     cudaStream_t stream) {
    SparsePrefillPast dense{table, q_offset};
    const auto& s = qscratch_;
    const int kv_bs = cache->block_size();
    // Graph-replayed verify chunks, the paged-FP4 branch (KV already appended), SWA and learned
    // sinks keep the dense past.
    if (cap_replay || paged_kv_written || sliding_window != 0 || sinks != nullptr ||
        s.sp_prefill_budget_blocks <= 0 || !cache->key_minmax_enabled() || nkv <= 0 || nh / nkv > 16 ||
        (q_offset + kv_bs - 1) / kv_bs <= s.sp_prefill_budget_blocks)
        return dense;
    const int past = sparse_prefill_select_past(q, n, s.sp_prefill_rows, cache->key_minmax_ptr(kv_layer, 0),
                                                table, q_offset, nh, nkv, hd, kv_bs, s.sp_prefill_cap_blocks,
                                                s.sp_prefill_budget_blocks, s.sp_prefill_sink_blocks,
                                                s.sp_prefill_recent_blocks, s.sparse_score_meanstd,
                                                s.sparse_score_std_coef, s.sp_prefill_scores,
                                                s.sp_prefill_agg, s.sp_prefill_table, s.sp_prefill_ctx,
                                                stream);
    return past > 0 ? SparsePrefillPast{s.sp_prefill_table, past} : dense;
}

}  // namespace imp
