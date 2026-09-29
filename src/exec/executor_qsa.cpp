// Qwen4Exp QSA indexer orchestration: index_qk_proj GEMM, raw/block key caches, block
// top-k selection, gather of the selected K/V rows into a scratch paged cache and the
// existing paged decode kernel over it. Math and kernels: compute/qsa_indexer.h.
//
// Decode (one sequence): the whole step runs on the device from state.positions, so the
// captured graph replays correctly; below 512 complete blocks the selection is every
// token in order, i.e. the same bytes the dense kernel reads. Prefill: dense FA2 runs for
// the chunk, then the rows whose position reaches budget + ratio - 1 (2051) are recomputed
// through the selection (attention.qsa_force: every row).
#include "compute/attention_paged.h"
#include "compute/gemm.h"
#include "compute/qsa_indexer.h"
#include "core/logging.h"
#include "exec/executor.h"
#include "exec/executor_helpers.h"
#include "memory/kv_cache.h"

#include <algorithm>

namespace imp {

namespace {

}

bool GraphExecutor::qsa_layer_(int layer) const {
    return dispatch_policy().attention.qsa && model_->layer(layer).qsa_index_qk.data != nullptr;
}

QsaGeom GraphExecutor::qsa_geom_(int layer) const {
    const auto& cfg = model_->config();
    QsaGeom g{};
    g.n_heads = static_cast<int>(model_->layer(layer).qsa_index_qk.shape[0]) / kQsaDim - 1;
    g.ratio = cfg.qsa_ratio;
    g.budget = cfg.qsa_budget;
    g.rope_dim = std::min(cfg.rope_dim > 0 ? cfg.rope_dim : cfg.head_dim, kQsaDim);
    g.theta = cfg.rope_theta;
    g.eps = cfg.rms_norm_eps;
    return g;
}

bool GraphExecutor::qsa_alloc_(int max_tokens) {
    const auto& cfg = model_->config();
    int layer = -1;
    for (int i = 0; i < cfg.n_layers; ++i)
        if (model_->layer(i).qsa_index_qk.data != nullptr)
            layer = i;
    if (layer < 0)
        return true;
    const int cols = static_cast<int>(model_->layer(layer).qsa_index_qk.shape[0]);
    const int n_heads = cols / kQsaDim - 1;
    if (cols % kQsaDim != 0 || n_heads < 1 || n_heads > 4 || cfg.qsa_ratio <= 0 || cfg.qsa_budget <= 0 ||
        compute_dtype_ != QType::F16) {
        IMP_LOG_ERROR("QSA indexer: unsupported shape (index_qk_proj rows %d, ratio %d, budget %d) or "
                      "compute dtype",
                      cols, cfg.qsa_ratio, cfg.qsa_budget);
        return false;
    }
    const int cap = cfg.qsa_budget + cfg.qsa_ratio - 1;
    qsa_rows_ = std::clamp(dispatch_policy().attention.qsa_rows, 1, 256);
    auto alloc_f16 = [&](Tensor& t, int64_t r, int64_t c, const char* tag) {
        void* p = vram_alloc(vram_alloc_, align256(static_cast<size_t>(r) * c * 2), tag);
        if (!p)
            return false;
        const int64_t shape[2] = {r, c};
        t = Tensor(p, QType::F16, 2, shape, true);
        return true;
    };
    if (!alloc_f16(qsa_qk_, max_tokens, cols, "qsa_qk") ||
        !alloc_f16(qsa_q_, max_tokens, static_cast<int64_t>(n_heads) * kQsaDim, "qsa_q"))
        return false;
    qsa_sel_tokens_ = static_cast<int32_t*>(
        vram_alloc(vram_alloc_, align256(static_cast<size_t>(qsa_rows_) * cap * sizeof(int32_t)), "qsa_sel"));
    qsa_sel_count_ = static_cast<int32_t*>(vram_alloc(vram_alloc_, 256, "qsa_sel_count"));
    qsa_scratch_ctx_ = static_cast<int32_t*>(vram_alloc(vram_alloc_, 256, "qsa_scratch_ctx"));
    if (!qsa_sel_tokens_ || !qsa_sel_count_ || !qsa_scratch_ctx_)
        return false;
    IMP_LOG_INFO("QSA indexer: %d query heads x %d, block %d, budget %d (top %d blocks), rows/pass %d",
                 n_heads, kQsaDim, cfg.qsa_ratio, cfg.qsa_budget, cfg.qsa_budget / cfg.qsa_ratio, qsa_rows_);
    return true;
}

// Context-sized state, sized from the KV cache on the first forward (the cache is built after
// the workspaces): raw and block keys per QSA layer, score scratch, scratch paged K/V.
bool GraphExecutor::qsa_ensure_ctx_(const InferenceState& state, cudaStream_t stream) {
    if (qsa_max_ctx_ > 0)
        return true;
    const auto& cfg = model_->config();
    KVCache* cache = state.kv_cache;
    if (!cache)
        return false;
    const int bs = cache->block_size();
    const int max_ctx = cache->ceiling_blocks() * bs;
    const int nb_max = max_ctx / cfg.qsa_ratio + 1;
    const int cap = cfg.qsa_budget + cfg.qsa_ratio - 1;
    const int bpr = (cap + bs - 1) / bs;
    qsa_layers_.assign(cfg.n_layers, QsaLayerState{});
    size_t bytes = 0;
    for (int i = 0; i < cfg.n_layers; ++i) {
        if (model_->layer(i).qsa_index_qk.data == nullptr)
            continue;
        QsaLayerState& L = qsa_layers_[i];
        const size_t rk = align256(static_cast<size_t>(max_ctx) * kQsaDim * 2);
        const size_t bk = align256(static_cast<size_t>(nb_max) * kQsaDim * 2);
        L.raw_keys = vram_alloc(vram_alloc_, rk, "qsa_raw_keys");
        L.block_keys = vram_alloc(vram_alloc_, bk, "qsa_block_keys");
        if (!L.raw_keys || !L.block_keys)
            return false;
        bytes += rk + bk;
    }
    const size_t sc = align256(static_cast<size_t>(qsa_rows_) * nb_max * sizeof(float));
    const size_t kv = align256(static_cast<size_t>(qsa_rows_) * bpr * bs * cfg.n_kv_heads * cfg.head_dim * 2);
    const size_t bt = align256(static_cast<size_t>(qsa_rows_) * bpr * sizeof(int32_t));
    qsa_scores_ = static_cast<float*>(vram_alloc(vram_alloc_, sc, "qsa_scores"));
    qsa_k_scratch_ = vram_alloc(vram_alloc_, kv, "qsa_k_scratch");
    qsa_v_scratch_ = vram_alloc(vram_alloc_, kv, "qsa_v_scratch");
    qsa_scratch_bt_ = static_cast<int32_t*>(vram_alloc(vram_alloc_, bt, "qsa_scratch_bt"));
    if (!qsa_scores_ || !qsa_k_scratch_ || !qsa_v_scratch_ || !qsa_scratch_bt_)
        return false;
    qsa_init_scratch_bt(qsa_scratch_bt_, qsa_rows_, bpr, stream);
    qsa_blocks_per_row_ = bpr;
    qsa_nb_max_ = nb_max;
    qsa_max_ctx_ = max_ctx;
    IMP_LOG_INFO("QSA indexer: context %d, key caches %.1f MiB, scratch K/V %.1f MiB (%d rows x %d blocks)",
                 max_ctx, bytes / (1024.0 * 1024.0), 2.0 * kv / (1024.0 * 1024.0), qsa_rows_, bpr);
    return true;
}

void GraphExecutor::qsa_free_() {
    auto fr = [&](void*& p) {
        if (p)
            vram_free(vram_alloc_, p);
        p = nullptr;
    };
    for (QsaLayerState& L : qsa_layers_) {
        fr(L.raw_keys);
        fr(L.block_keys);
    }
    qsa_layers_.clear();
    fr(qsa_qk_.data);
    fr(qsa_q_.data);
    void* p;
    p = qsa_sel_tokens_;   fr(p); qsa_sel_tokens_ = nullptr;
    p = qsa_sel_count_;    fr(p); qsa_sel_count_ = nullptr;
    p = qsa_scratch_ctx_;  fr(p); qsa_scratch_ctx_ = nullptr;
    p = qsa_scores_;       fr(p); qsa_scores_ = nullptr;
    p = qsa_scratch_bt_;   fr(p); qsa_scratch_bt_ = nullptr;
    fr(qsa_k_scratch_);
    fr(qsa_v_scratch_);
    qsa_max_ctx_ = 0;
}

// Selected-token attention for `rows` query rows (q_rows/o_rows: [rows, nh*hd] FP16) whose
// token lists sit in qsa_sel_tokens_[0..rows).
void GraphExecutor::qsa_attend_rows_(int layer, const InferenceState& state, const void* q_rows, void* o_rows,
                                     int rows, const int* bt, float scale, int ctx_hint,
                                     cudaStream_t stream) {
    const auto& cfg = model_->config();
    KVCache* cache = state.kv_cache;
    const int bs = cache->block_size();
    const int nkv = cfg.n_kv_heads, hd = cfg.head_dim, nh = cfg.n_heads;
    const int cap = cfg.qsa_budget + cfg.qsa_ratio - 1;
    const int kv_layer = get_kv_layer(kv_layer_map_, layer);
    qsa_gather_kv(static_cast<const half*>(cache->k_ptr(kv_layer, 0)),
                  static_cast<const half*>(cache->v_ptr(kv_layer, 0)), bt, bs, nkv, hd, qsa_sel_tokens_,
                  qsa_sel_count_, cap, static_cast<half*>(qsa_k_scratch_), static_cast<half*>(qsa_v_scratch_),
                  qsa_blocks_per_row_, qsa_scratch_ctx_, rows, stream);
    const int64_t qd[4] = {rows, 1, nh, hd};
    const int64_t cs[4] = {static_cast<int64_t>(rows) * qsa_blocks_per_row_, bs, nkv, hd};
    Tensor Q(const_cast<void*>(q_rows), QType::F16, 4, qd, true);
    Tensor O(o_rows, QType::F16, 4, qd, true);
    Tensor Ks(qsa_k_scratch_, QType::F16, 4, cs, true);
    Tensor Vs(qsa_v_scratch_, QType::F16, 4, cs, true);
    // Split-K scratch as the dense decode uses it (the kernel re-checks the size against
    // rows x heads x splits and drops to fewer splits or none): without it one CTA per head
    // walks the 2051 tokens serially, 31.7 vs 54.1 tok/s at 4.5k context (2026-09-20).
    paged_attention_set_splitk_scratch(qscratch_.splitk, qscratch_.splitk_size);
    // The selection holds at most cap tokens, and below cap it is every token in order:
    // the dense path splits on the real context, so this must too or the same bytes
    // reduce in a different order.
    const int max_ctx = std::max(1, std::min(cap, ctx_hint > 0 ? ctx_hint : cap));
    paged_attention_decode(Q, Ks, Vs, O, qsa_scratch_bt_, qsa_scratch_ctx_, bs, scale, max_ctx, 0,
                           cfg.attn_logit_softcap, stream, qsa_blocks_per_row_, 0, nullptr, hd);
}

bool GraphExecutor::qsa_decode_(int layer, const InferenceState& state, const Tensor& no, Tensor& qv,
                                Tensor& ao, const int* bt, float scale, cudaStream_t stream) {
    if (!qsa_seq_ok_ || !qsa_ensure_ctx_(state, stream))
        return false;
    // Proof-of-activity for A/B arms: below budget + ratio - 1 the selection is every
    // token in order, so an arm that never logs this ran bit-identically to dense.
    static bool logged_qsa_active = false;
    if (!logged_qsa_active && state.max_context_len > model_->config().qsa_budget) {
        logged_qsa_active = true;
        IMP_LOG_INFO("QSA indexer ACTIVE: ctx %d > budget %d tokens", state.max_context_len,
                     model_->config().qsa_budget);
    }
    const auto& ly = model_->layer(layer);
    const QsaGeom g = qsa_geom_(layer);
    const QsaLayerState& L = qsa_layers_[layer];
    Tensor qk = view_tokens(qsa_qk_, 1);
    gemm(no, ly.qsa_index_qk, qk, 1.0f, 0.0f, stream);
    qsa_prep_queries(static_cast<const half*>(qk.data), state.positions,
                     static_cast<const half*>(ly.qsa_index_q_norm.data), static_cast<half*>(qsa_q_.data),
                     static_cast<half*>(L.raw_keys), 1, g, stream);
    qsa_pool_blocks(static_cast<const half*>(L.raw_keys), static_cast<const half*>(ly.qsa_index_k_norm.data),
                    static_cast<half*>(L.block_keys), 0, 1, state.positions, g, stream);
    qsa_select(static_cast<const half*>(qsa_q_.data), state.positions, static_cast<const half*>(L.block_keys),
               qsa_scores_, qsa_nb_max_, qsa_sel_tokens_, qsa_sel_count_, 1, g, stream);
    qsa_attend_rows_(layer, state, qv.data, ao.data, 1, bt, scale, state.max_context_len, stream);
    return true;
}

void GraphExecutor::qsa_prefill_(int layer, const InferenceState& state, int n, const Tensor& no, Tensor& qv,
                                 Tensor& ao, const int* bt, float scale, cudaStream_t stream) {
    const int pos0 = state.prefill_offset;
    if (pos0 == 0) {
        qsa_valid_ = 0;
        qsa_seq_ok_ = true;
    }
    if (!qsa_seq_ok_)
        return;
    if (!qsa_ensure_ctx_(state, stream)) {
        IMP_LOG_WARN("QSA indexer: context state allocation failed; attention stays dense for this sequence");
        qsa_seq_ok_ = false;
        return;
    }
    if (pos0 > qsa_valid_ || pos0 + n > qsa_max_ctx_) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            IMP_LOG_WARN("QSA indexer: chunk at %d with keys valid to %d (context %d): a resumed prefix has "
                         "no indexer keys; attention stays dense for this sequence",
                         pos0, qsa_valid_, qsa_max_ctx_);
        }
        qsa_seq_ok_ = false;
        return;
    }
    const auto& ly = model_->layer(layer);
    const auto& cfg = model_->config();
    const QsaGeom g = qsa_geom_(layer);
    const QsaLayerState& L = qsa_layers_[layer];
    Tensor qk = view_tokens(qsa_qk_, n);
    gemm(no, ly.qsa_index_qk, qk, 1.0f, 0.0f, stream);
    qsa_prep_queries(static_cast<const half*>(qk.data), state.positions,
                     static_cast<const half*>(ly.qsa_index_q_norm.data), static_cast<half*>(qsa_q_.data),
                     static_cast<half*>(L.raw_keys), n, g, stream);
    const int b0 = pos0 / g.ratio;
    const int nb = (pos0 + n) / g.ratio - b0;
    qsa_pool_blocks(static_cast<const half*>(L.raw_keys), static_cast<const half*>(ly.qsa_index_k_norm.data),
                    static_cast<half*>(L.block_keys), b0, nb, nullptr, g, stream);
    qsa_valid_ = pos0 + n;

    const int cap = g.budget + g.ratio - 1;
    const int r0 = dispatch_policy().attention.qsa_force ? 0 : std::max(0, cap - pos0);
    const size_t q_row = static_cast<size_t>(cfg.n_heads) * cfg.head_dim * 2;
    const bool dbg = dispatch_policy().attention.qsa_debug && layer == 3;
    for (int r = r0; r < n; r += qsa_rows_) {
        const int rows = std::min(qsa_rows_, n - r);
        qsa_select(static_cast<const half*>(qsa_q_.data) + static_cast<size_t>(r) * g.n_heads * kQsaDim,
                   state.positions + r, static_cast<const half*>(L.block_keys), qsa_scores_, qsa_nb_max_,
                   qsa_sel_tokens_, qsa_sel_count_, rows, g, stream);
        if (dbg)
            qsa_debug_rows_(layer, state, static_cast<const char*>(qv.data) + r * q_row,
                            static_cast<const char*>(ao.data) + r * q_row, rows, pos0 + r, bt, scale, stream);
        qsa_attend_rows_(layer, state, static_cast<const char*>(qv.data) + r * q_row,
                         static_cast<char*>(ao.data) + r * q_row, rows, bt, scale, pos0 + r + rows, stream);
    }
}

// Diagnostics (attention.qsa_debug): for `rows` query rows (positions p0..), the paged kernel
// on the REAL cache (ctx = p + 1, the sequence's block table replicated per row) against the
// dense output already in `o_dense` (FA2) and against the selected path. Logs max |diff|.
void GraphExecutor::qsa_debug_rows_(int layer, const InferenceState& state, const void* q_rows,
                                    const void* o_dense, int rows, int p0, const int* bt, float scale,
                                    cudaStream_t stream) {
    const auto& cfg = model_->config();
    KVCache* cache = state.kv_cache;
    const int bs = cache->block_size(), nkv = cfg.n_kv_heads, hd = cfg.head_dim, nh = cfg.n_heads;
    // Prefill hands a flat block table (max_blocks_per_seq = 0): replicate the blocks in use.
    const int mb = (p0 + rows + bs - 1) / bs;
    const size_t o_elems = static_cast<size_t>(rows) * nh * hd;
    std::vector<int> h_bt(static_cast<size_t>(rows) * mb), h_ctx(rows);
    std::vector<int> row0(mb);
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(row0.data(), bt, mb * sizeof(int), cudaMemcpyDeviceToHost, stream));
    IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream));
    for (int r = 0; r < rows; ++r) {
        std::copy(row0.begin(), row0.end(), h_bt.begin() + static_cast<size_t>(r) * mb);
        h_ctx[r] = p0 + r + 1;
    }
    int* d_bt = static_cast<int*>(vram_alloc(vram_alloc_, h_bt.size() * sizeof(int), "qsa_debug"));
    int* d_ctx = static_cast<int*>(vram_alloc(vram_alloc_, h_ctx.size() * sizeof(int), "qsa_debug"));
    half* d_o = static_cast<half*>(vram_alloc(vram_alloc_, o_elems * sizeof(half), "qsa_debug"));
    half* d_o2 = static_cast<half*>(vram_alloc(vram_alloc_, o_elems * sizeof(half), "qsa_debug"));
    if (!d_bt || !d_ctx || !d_o || !d_o2) {
        IMP_LOG_WARN("[qsa-debug] scratch allocation failed, skipping the comparison");
        return;
    }
    IMP_CUDA_CHECK_LOG(cudaMemcpy(d_bt, h_bt.data(), h_bt.size() * sizeof(int), cudaMemcpyHostToDevice));
    IMP_CUDA_CHECK_LOG(cudaMemcpy(d_ctx, h_ctx.data(), h_ctx.size() * sizeof(int), cudaMemcpyHostToDevice));
    const int kv_layer = get_kv_layer(kv_layer_map_, layer);
    const int64_t qd[4] = {rows, 1, nh, hd};
    const int64_t cs[4] = {static_cast<int64_t>(cache->total_blocks()), bs, nkv, hd};
    Tensor Q(const_cast<void*>(q_rows), QType::F16, 4, qd, true);
    Tensor O(d_o, QType::F16, 4, qd, true);
    Tensor Kc(cache->k_ptr(kv_layer, 0), QType::F16, 4, cs, true);
    Tensor Vc(cache->v_ptr(kv_layer, 0), QType::F16, 4, cs, true);
    paged_attention_set_splitk_scratch(nullptr, 0);
    paged_attention_decode(Q, Kc, Vc, O, d_bt, d_ctx, bs, scale, p0 + rows, 0, cfg.attn_logit_softcap, stream,
                           mb, 0, nullptr, hd);
    // Selected path into a private output so ao stays FA2's for the comparison.
    qsa_attend_rows_(layer, state, q_rows, d_o2, rows, bt, scale, p0 + rows, stream);
    IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream));
    std::vector<half> h_fa2(o_elems), h_paged(o_elems), h_sel(o_elems);
    IMP_CUDA_CHECK_LOG(cudaMemcpy(h_fa2.data(), o_dense, o_elems * sizeof(half), cudaMemcpyDeviceToHost));
    IMP_CUDA_CHECK_LOG(cudaMemcpy(h_paged.data(), d_o, o_elems * sizeof(half), cudaMemcpyDeviceToHost));
    IMP_CUDA_CHECK_LOG(cudaMemcpy(h_sel.data(), d_o2, o_elems * sizeof(half), cudaMemcpyDeviceToHost));
    double d_fp = 0, d_ps = 0, mx = 0;
    for (size_t i = 0; i < o_elems; ++i) {
        const double f = __half2float(h_fa2[i]), p = __half2float(h_paged[i]), s = __half2float(h_sel[i]);
        d_fp = std::max(d_fp, std::fabs(f - p));
        d_ps = std::max(d_ps, std::fabs(p - s));
        mx = std::max(mx, std::fabs(f));
    }
    IMP_LOG_INFO("[qsa-debug] layer %d rows %d..%d: max|fa2 - paged_cache| %.4g, max|paged_cache - selected| %.4g, "
                 "max|fa2| %.3g",
                 layer, p0, p0 + rows - 1, d_fp, d_ps, mx);
    vram_free(vram_alloc_, d_bt);
    vram_free(vram_alloc_, d_ctx);
    vram_free(vram_alloc_, d_o);
    vram_free(vram_alloc_, d_o2);
}

}  // namespace imp
