// Sparse prefill attention (attention.sparse_prefill_topk_tokens): scratch at init and the
// per-chunk past-page pick the chunked prefill path calls. Kernels: sparse_attn_select.cu.

#include "core/dispatch_policy.h"
#include "core/fallback_site.h"
#include "core/log_once.h"
#include "core/logging.h"
#include "exec/executor.h"
#include "exec/sparse_attn_geometry.h"
#include "exec/sparse_attn_select.h"
#include "memory/engine_arena.h"
#include "memory/kv_cache.h"

namespace imp {

namespace {
constexpr int kPrefillRows = 64;  // kMaxPrefillRows in sparse_attn_select.cu
FallbackSite s_fb_sparse_prefill{"sparse_prefill_scratch"};
}  // namespace

void GraphExecutor::allocate_sparse_prefill_scratch_() {
    const auto& acfg = dispatch_policy().attention;
    if (acfg.sparse_prefill_topk_tokens <= 0)
        return;
    const int max_ctx_tokens = (mla_absorb_max_seq_ > 0) ? mla_absorb_max_seq_ : max_tokens_;
    const SparseGeometry geo = sparse_geometry(acfg.sparse_prefill_topk_tokens, cfg::kSparseSinkTokens,
                                               cfg::kSparsePrefillRecentTokens, 0, max_ctx_tokens,
                                               kv_block_size_);
    const int cap = geo.max_ctx_blocks;
    StableSpan<std::byte> sc = engine_arena().take_bytes((size_t)kPrefillRows * cap * sizeof(float));
    StableSpan<std::byte> agg = engine_arena().take_bytes((size_t)cap * sizeof(float));
    StableSpan<std::byte> tbl = engine_arena().take_bytes((size_t)cap * sizeof(int));
    StableSpan<std::byte> ctx = engine_arena().take_bytes(2 *
                                                          sizeof(int));  // [past len in, selected len out]
    if (sc.empty() || agg.empty() || tbl.empty() || ctx.empty()) {
        IMP_FALLBACK_WARN(
            s_fb_sparse_prefill,
            "sparse prefill scratch (%.2f MiB) not taken from the T2 arena: sparse prefill off, "
            "dense prefill attention instead",
            static_cast<double>((static_cast<size_t>(kPrefillRows + 1) * cap * sizeof(float)) +
                                (static_cast<size_t>(cap) * sizeof(int)) + (2 * sizeof(int))) /
                (1024.0 * 1024.0));
        return;
    }
    qscratch_.sp_prefill_scores = sc.as<float>();
    qscratch_.sp_prefill_agg = agg.as<float>();
    qscratch_.sp_prefill_table = tbl.as<int>();
    qscratch_.sp_prefill_ctx = ctx.as<int>();
    qscratch_.sp_prefill_budget_blocks = geo.budget_blocks;
    qscratch_.sp_prefill_sink_blocks = geo.sink_blocks;
    qscratch_.sp_prefill_recent_blocks = geo.recent_blocks;
    qscratch_.sp_prefill_cap_blocks = cap;
    static_assert(cfg::kSparsePrefillRows >= 1 && cfg::kSparsePrefillRows <= kPrefillRows);
    qscratch_.sp_prefill_rows = cfg::kSparsePrefillRows;
    IMP_LOG_INFO(
        "Sparse prefill attention: budget %d blocks (%d tokens), sink %d + recent %d blocks, "
        "%d query rows, score mean+std",
        geo.budget_blocks, geo.budget_blocks * kv_block_size_, geo.sink_blocks, geo.recent_blocks,
        qscratch_.sp_prefill_rows);
}

GraphExecutor::SparsePrefillPast GraphExecutor::sparse_prefill_pick_(const half* q, int n, KVCache* cache,
                                                                     int kv_layer, const int* table,
                                                                     int q_offset, int nh, int nkv, int hd,
                                                                     bool cap_replay, int sliding_window,
                                                                     const void* sinks, cudaStream_t stream) {
    SparsePrefillPast dense{table, q_offset};
    const auto& s = qscratch_;
    const int kv_bs = cache->block_size();
    // Graph-replayed verify chunks, SWA and learned sinks keep the dense past.
    if (cap_replay || sliding_window != 0 || sinks != nullptr || s.sp_prefill_budget_blocks <= 0 ||
        !cache->key_minmax_enabled() || nkv <= 0 || nh / nkv > 16 ||
        (q_offset + kv_bs - 1) / kv_bs <= s.sp_prefill_budget_blocks)
        return dense;
    const int past = sparse_prefill_select_past({.q = q,
                                                 .n_rows = n,
                                                 .sample_rows = s.sp_prefill_rows,
                                                 .minmax_layer = cache->key_minmax_ptr(kv_layer, 0),
                                                 .past_block_table = table,
                                                 .past_len = q_offset,
                                                 .n_heads = nh,
                                                 .n_kv_heads = nkv,
                                                 .head_dim = hd,
                                                 .block_size = kv_bs,
                                                 .capacity_blocks = s.sp_prefill_cap_blocks,
                                                 .budget_blocks = s.sp_prefill_budget_blocks,
                                                 .sink_blocks = s.sp_prefill_sink_blocks,
                                                 .recent_blocks = s.sp_prefill_recent_blocks,
                                                 .std_coef = cfg::kSparseScoreStdCoef,
                                                 .scores_scratch = s.sp_prefill_scores.data(),
                                                 .agg_scratch = s.sp_prefill_agg.data(),
                                                 .out_block_table = s.sp_prefill_table.data(),
                                                 .out_ctx_scratch = s.sp_prefill_ctx.data(),
                                                 .stream = stream});
    if (past <= 0)
        return dense;
    // Activity proof (prefill is not graph-captured, so this fires where the selection runs).
    static LogOnce logged_active;
    if (logged_active.first()) {
        IMP_LOG_INFO("sparse prefill attention ACTIVE: past %d -> %d tokens at L%d", q_offset, past,
                     kv_layer);
    }
    return SparsePrefillPast{s.sp_prefill_table.data(), past};
}

}  // namespace imp
