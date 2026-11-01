// KV eviction proof harness, executor half (kv_cache.evict_keep_ratio): scratch, the prefill query
// statistics hook, and the prompt scoring the engine runs after the last prefill chunk.
// Design: docs/plans/2026-11-01-kv-eviction.md.

#include "core/dispatch_policy.h"
#include "core/logging.h"
#include "exec/executor.h"
#include "exec/executor_helpers.h"
#include "exec/kv_evict.h"
#include "memory/engine_arena.h"
#include "memory/kv_cache.h"
#include <algorithm>
#include <cmath>
#include <vector>

namespace imp {

namespace {

int resolved_head_dim(const ModelConfig& cfg) {
    return cfg.head_dim > 0 ? cfg.head_dim : (cfg.n_heads > 0 ? cfg.d_model / cfg.n_heads : 0);
}

// TriAttention future offsets D = {1, 2, 4, ..., 2^16} (arXiv 2604.04921 Eq. 11).
constexpr int kEvictOffsets = 17;

}  // namespace

void GraphExecutor::allocate_kv_evict_scratch_() {
    kv_evict_ = KvEvictScratch{};
    if (dispatch_policy().kv_cache.evict_keep_ratio <= 0.0f || model_ == nullptr)
        return;
    const auto& cfg = model_->config();
    const int hd = resolved_head_dim(cfg);
    int n_kv = 0;
    for (int idx : kv_layer_map_)
        n_kv = std::max(n_kv, idx + 1);
    if (kv_layer_map_.empty())
        n_kv = cfg.n_layers;
    const int bs = kv_block_size_;
    const int cap_blocks = ((std::max(mla_absorb_max_seq_, max_tokens_) + bs - 1) / bs) + 16;
    const int cap_rows = std::max(max_logit_tokens_, 33);
    const size_t qst = static_cast<size_t>(n_kv) * kv_evict_qstats_floats(cfg.n_heads, hd) * sizeof(float);
    const size_t uw = static_cast<size_t>(n_kv) * cfg.n_heads * (hd + (hd / 2)) * sizeof(float);
    const size_t hs = static_cast<size_t>(cfg.n_heads) * 2 * sizeof(double);
    const size_t toks = static_cast<size_t>(cap_blocks) * bs;
    auto s_q = engine_arena().take_bytes(qst);
    auto s_uw = engine_arena().take_bytes(uw);
    auto s_hs = engine_arena().take_bytes(hs);
    auto s_tm = engine_arena().take_bytes(toks * sizeof(unsigned));
    auto s_ta = engine_arena().take_bytes(toks * sizeof(float));
    auto s_bt = engine_arena().take_bytes(static_cast<size_t>(cap_blocks) * sizeof(int));
    auto s_cb = engine_arena().take_bytes(static_cast<size_t>(cap_rows) * cap_blocks * sizeof(int));
    auto s_cc = engine_arena().take_bytes(static_cast<size_t>(cap_rows) * sizeof(int));
    if (s_q.empty() || s_uw.empty() || s_hs.empty() || s_tm.empty() || s_ta.empty() || s_bt.empty() ||
        s_cb.empty() || s_cc.empty()) {
        IMP_LOG_WARN("kv_cache.evict_keep_ratio: scratch unavailable from the T2 arena - eviction disabled");
        return;
    }
    kv_evict_.qstats = reinterpret_cast<float*>(s_q.data());
    kv_evict_.uw = reinterpret_cast<float*>(s_uw.data());
    kv_evict_.head_stats = reinterpret_cast<double*>(s_hs.data());
    kv_evict_.tok_max = reinterpret_cast<unsigned*>(s_tm.data());
    kv_evict_.tok_acc = reinterpret_cast<float*>(s_ta.data());
    kv_evict_.bt = reinterpret_cast<int*>(s_bt.data());
    kv_evict_.compact_bt = reinterpret_cast<int*>(s_cb.data());
    kv_evict_.compact_ctx = reinterpret_cast<int*>(s_cc.data());
    kv_evict_.n_kv_layers = n_kv;
    kv_evict_.cap_blocks = cap_blocks;
    kv_evict_.cap_rows = cap_rows;
    kv_evict_.enabled = true;
    const size_t tables = (static_cast<size_t>(cap_rows) + 1) * cap_blocks * sizeof(int);
    IMP_LOG_INFO("KV eviction harness: keep %.2f of prompt blocks, %d KV layers, scratch %.1f MiB",
                 static_cast<double>(dispatch_policy().kv_cache.evict_keep_ratio), n_kv,
                 static_cast<double>(qst + uw + hs + (toks * 8) + tables) / (1024.0 * 1024.0));
}

void GraphExecutor::kv_evict_tables_(const InferenceState& st, const int* bt, int sub_n, int n_seq, int kv_bs,
                                     const int*& attn_bt, const int*& attn_ctx, cudaStream_t stream) {
    if (!kv_evict_.enabled)
        return;
    const int mbps = st.max_blocks_per_seq;
    if (sub_n != n_seq || n_seq > kv_evict_.cap_rows || mbps > kv_evict_.cap_blocks ||
        (mbps <= 0 && n_seq != 1)) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            IMP_LOG_ERROR(
                "kv evict: decode shape outside the compaction scratch (rows %d/%d, stride %d/%d) - "
                "attention reads the raw table",
                n_seq, kv_evict_.cap_rows, mbps, kv_evict_.cap_blocks);
        }
        return;
    }
    kv_evict_compact_tables(bt, st.context_lens, n_seq, mbps > 0 ? mbps : kv_evict_.cap_blocks, kv_bs,
                            kv_evict_.compact_bt, kv_evict_.compact_ctx, stream);
    attn_bt = kv_evict_.compact_bt;
    attn_ctx = kv_evict_.compact_ctx;
}

void GraphExecutor::kv_evict_prefill_qstats_(int layer, const InferenceState& state, const Tensor& q, int n,
                                             cudaStream_t stream) {
    if (!kv_evict_.enabled || state.spec_verify_chunk)
        return;
    const int kv_layer = kv_layer_map_.empty() ? layer : get_kv_layer(kv_layer_map_, layer);
    if (kv_layer < 0 || kv_layer >= kv_evict_.n_kv_layers || q.qtype != QType::F16 ||
        state.positions == nullptr)
        return;
    const auto& cfg = model_->config();
    const int hd = resolved_head_dim(cfg);
    float* st = kv_evict_.qstats + (static_cast<int64_t>(kv_layer) * kv_evict_qstats_floats(cfg.n_heads, hd));
    // A sequence prefilled from position 0 starts a fresh statistic; continuation chunks add to it.
    if (state.prefill_offset == 0)
        IMP_CUDA_CHECK_LOG(
            cudaMemsetAsync(st, 0, kv_evict_qstats_floats(cfg.n_heads, hd) * sizeof(float), stream));
    const float inv_scaling = cfg.rope_freq_scale > 0.0f ? 1.0f / cfg.rope_freq_scale : 1.0f;
    (void)kv_evict_qstats_accumulate(static_cast<const half*>(q.data), state.positions, n, cfg.n_heads, hd,
                                     cfg.rope_neox, cfg.rope_theta, inv_scaling, st, stream);
}

bool GraphExecutor::kv_evict_score_prompt(KVCache* cache, std::span<const int> h_block_table, int ctx_len,
                                          int tok_begin, int tok_end, std::vector<float>& token_scores,
                                          cudaStream_t stream) {
    if (!kv_evict_.enabled || cache == nullptr || model_ == nullptr || ctx_len <= 0)
        return false;
    const auto& cfg = model_->config();
    const int hd = resolved_head_dim(cfg);
    const int nh = cfg.n_heads;
    const int pairs = hd / 2;
    const int bs = cache->block_size();
    const int n_blocks = (ctx_len + bs - 1) / bs;
    if (n_blocks > kv_evict_.cap_blocks || static_cast<int>(h_block_table.size()) < n_blocks ||
        tok_begin < 0 || tok_end > ctx_len || tok_end <= tok_begin)
        return false;
    const int L = kv_evict_.n_kv_layers;
    const int64_t qsf = kv_evict_qstats_floats(nh, hd);

    std::vector<float> qs(static_cast<size_t>(L) * qsf);
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(qs.data(), kv_evict_.qstats, qs.size() * sizeof(float),
                                       cudaMemcpyDeviceToHost, stream));
    IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream));
    for (int l = 0; l < L; ++l) {
        const float cnt = qs[(static_cast<size_t>(l) * qsf) + (static_cast<size_t>(nh) * pairs * 3)];
        if (static_cast<int>(cnt) != ctx_len) {
            IMP_LOG_WARN("kv evict: layer %d query statistics cover %d rows, prompt has %d - not evicting", l,
                         static_cast<int>(cnt), ctx_len);
            return false;
        }
    }

    // u = mean over offsets of the query centre rotated to position ctx_len + delta; w = E|q_f| - |E q_f|.
    const float inv_scaling = cfg.rope_freq_scale > 0.0f ? 1.0f / cfg.rope_freq_scale : 1.0f;
    std::vector<double> cmean(static_cast<size_t>(pairs)), smean(static_cast<size_t>(pairs));
    for (int f = 0; f < pairs; ++f) {
        float freq = 1.0f / std::pow(cfg.rope_theta, (2.0f * f) / static_cast<float>(2 * pairs));
        freq *= inv_scaling;
        double c = 0.0, s = 0.0;
        for (int j = 0; j < kEvictOffsets; ++j) {
            const double phi = static_cast<double>(ctx_len + (1 << j)) * static_cast<double>(freq);
            c += std::cos(phi);
            s += std::sin(phi);
        }
        cmean[f] = c / kEvictOffsets;
        smean[f] = s / kEvictOffsets;
    }
    // Per layer: u [nh][hd], then w [nh][pairs].
    const int uw_row = hd + pairs;
    std::vector<float> packed(static_cast<size_t>(L) * nh * uw_row);
    for (int l = 0; l < L; ++l) {
        const float* st = qs.data() + (static_cast<size_t>(l) * qsf);
        float* lu = packed.data() + (static_cast<size_t>(l) * nh * uw_row);
        float* lw = lu + (static_cast<size_t>(nh) * hd);
        for (int h = 0; h < nh; ++h) {
            float* u = lu + (static_cast<size_t>(h) * hd);
            float* w = lw + (static_cast<size_t>(h) * pairs);
            for (int f = 0; f < pairs; ++f) {
                const float* a = st + ((static_cast<size_t>(h) * pairs + f) * 3);
                const double A = a[0] / ctx_len, B = a[1] / ctx_len, M = a[2] / ctx_len;
                u[f] = static_cast<float>((A * cmean[f]) - (B * smean[f]));
                u[f + pairs] = static_cast<float>((A * smean[f]) + (B * cmean[f]));
                w[f] = static_cast<float>(M - std::sqrt((A * A) + (B * B)));
            }
        }
    }
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(kv_evict_.uw, packed.data(), packed.size() * sizeof(float),
                                       cudaMemcpyHostToDevice, stream));
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(kv_evict_.bt, h_block_table.data(),
                                       static_cast<size_t>(n_blocks) * sizeof(int), cudaMemcpyHostToDevice,
                                       stream));
    IMP_CUDA_CHECK_LOG(
        cudaMemsetAsync(kv_evict_.tok_max, 0, static_cast<size_t>(ctx_len) * sizeof(unsigned), stream));
    IMP_CUDA_CHECK_LOG(
        cudaMemsetAsync(kv_evict_.tok_acc, 0, static_cast<size_t>(ctx_len) * sizeof(float), stream));

    const QType dt = cache->qtype();
    for (int l = 0; l < L; ++l) {
        KvEvictKeyView kv;
        kv.dtype = dt;
        kv.k_base = cache->k_ptr(l, 0);
        kv.block_stride = static_cast<int64_t>(cache->block_bytes(l));
        if (dt == QType::NVFP4) {
            kv.scale_base = static_cast<const uint8_t*>(cache->k_scale_ptr(l, 0));
            kv.scale_block_stride = static_cast<int64_t>(cache->scale_block_bytes(l));
        }
        kv.n_kv_heads = cfg.n_kv_heads;
        kv.head_dim = hd;
        kv.block_size = bs;
        const float* u = kv_evict_.uw + (static_cast<int64_t>(l) * nh * uw_row);
        const float* w = u + (static_cast<int64_t>(nh) * hd);
        IMP_CUDA_CHECK_LOG(
            cudaMemsetAsync(kv_evict_.head_stats, 0, static_cast<size_t>(nh) * 2 * sizeof(double), stream));
        kv_evict_score_pass(0, kv, u, w, kv_evict_.bt, tok_begin, tok_end, nh, kv_evict_.head_stats,
                            kv_evict_.tok_max, stream);
        kv_evict_score_pass(1, kv, u, w, kv_evict_.bt, tok_begin, tok_end, nh, kv_evict_.head_stats,
                            kv_evict_.tok_max, stream);
        kv_evict_accumulate_layer(kv_evict_.tok_max, kv_evict_.tok_acc, tok_begin, tok_end, stream);
    }
    token_scores.assign(static_cast<size_t>(ctx_len), NAN);
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(token_scores.data() + tok_begin, kv_evict_.tok_acc + tok_begin,
                                       static_cast<size_t>(tok_end - tok_begin) * sizeof(float),
                                       cudaMemcpyDeviceToHost, stream));
    IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream));
    for (int t = tok_begin; t < tok_end; ++t)
        token_scores[static_cast<size_t>(t)] /= static_cast<float>(L);
    return true;
}

}  // namespace imp
