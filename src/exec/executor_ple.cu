// Qwen4Exp PLE (layer 1): host n-gram hash + gather from the mmapped F8 table, then key/value
// projections, the stream gate and the dilated depthwise conv on the device. Math in compute/ple.h.
//
// Scratch: everything between the previous block's hc_write_ and this layer's hc_read_ is free,
// so the projections and gates live in the hc_* buffers (ple_embed_dim == d_model is checked at
// alloc). Only the pinned staging row block is PLE-owned.
//
// Sequence state is per sequence: the 9 conv rows are the SSM slab tail of the sequence's slot
// (SSMState::extra_state; reset, snapshots and spec rollback move them with the slab), the
// n-gram context is the sequence's tokens at pos0-2, pos0-1 (InferenceState::ple_hist_*).

#include "compute/gated_residual.h"
#include "compute/gemm.h"
#include "compute/ple.h"
#include "core/logging.h"
#include "exec/executor.h"
#include "exec/executor_helpers.h"
#include "model/ngram_table.h"

namespace imp {

bool GraphExecutor::ple_alloc_(int max_tokens) {
    const NGramTable* tab = model_->ngram_table();
    if (tab == nullptr)
        return true;
    const auto& cfg = model_->config();
    const int hc = cfg.hc_count;
    const int d = cfg.d_model;
    if (tab->embed_dim() != d || compute_dtype_ != QType::F16) {
        IMP_LOG_ERROR("PLE: embed_dim %d must equal d_model %d and compute dtype must be FP16",
                      tab->embed_dim(), d);
        return false;
    }
    int layer = -1;
    for (int i = 0; i < cfg.n_layers; i++) {
        if (model_->layer(i).ple_key_proj.data != nullptr)
            layer = i;
    }
    const Tensor& conv = model_->layer(layer).ple_conv1d;
    const int channels = hc * d;
    const int kernel = static_cast<int>(conv.numel() / channels);
    const int state_len = (kernel - 1) * tab->ngram_size();
    ple_host_ = PinnedBuffer::acquire(cuda_host_pinned_allocator(), static_cast<size_t>(max_tokens) * d * 2);
    if (!ple_host_) {
        IMP_LOG_ERROR("PLE: pinned staging for %d tokens failed", max_tokens);
        return false;
    }
    const size_t emb_bytes = align256(static_cast<size_t>(max_tokens) * d * 2);
    void* pe = vram_alloc(vram_alloc_, emb_bytes, "ple_emb");
    if (pe == nullptr)
        return false;
    const int64_t emb_shape[2] = {max_tokens, d};
    ple_emb_dev_ = Tensor(pe, QType::F16, 2, emb_shape, true);
    IMP_CUDA_CHECK_LOG(cudaEventCreateWithFlags(&ple_h2d_done_, cudaEventDisableTiming));
    ple_ids_.resize(static_cast<size_t>(max_tokens) * tab->n_heads());
    IMP_LOG_INFO(
        "PLE: layer %d, conv kernel %d dilation %d (state %d rows x %d per SSM slot), "
        "%.1f MiB pinned staging",
        layer, kernel, tab->ngram_size(), state_len, channels,
        static_cast<double>(max_tokens) * d * 2 / (1024.0 * 1024.0));
    return true;
}

void GraphExecutor::ple_free_() {
    ple_host_.reset();
    if (ple_emb_dev_.data != nullptr) {
        vram_free(vram_alloc_, ple_emb_dev_.data);
        ple_emb_dev_.data = nullptr;
    }
    if (ple_h2d_done_ != nullptr) {
        cudaEventDestroy(ple_h2d_done_);
        ple_h2d_done_ = nullptr;
    }
}

int GraphExecutor::ple_context_len() const {
    const NGramTable* tab = model_->ngram_table();
    return tab ? tab->context_len() : 0;
}

void GraphExecutor::ple_run_(const InferenceState& state, int layer, int n, cudaStream_t stream) {
    const NGramTable* tab = model_->ngram_table();
    const auto& cfg = model_->config();
    const auto& L = model_->layer(layer);
    const int hc = cfg.hc_count;
    const int d = cfg.d_model;
    const int ctx_len = tab->context_len();

    // Batched decode: one row per sequence, conv rows addressed through the slot table.
    const bool batched = !state.is_prefill && state.ssm_n_seq > 1 && state.ssm_seq_slots != nullptr;
    const int n_seq = batched ? state.ssm_n_seq : 1;
    SSMState* ss = state.ssm_state;
    void* conv_state = ss ? ss->extra_state(batched ? 0 : state.ssm_seq_id) : nullptr;
    // A prefill-mode chunk is one sequence even when attention splits its rows (#964).
    const bool served = conv_state != nullptr && !state.ssm_grouped_chunk() && !state.ragged_prefill() &&
                        (batched ? (n == n_seq && state.ple_host_ready)
                                 : (state.is_prefill || state.n_sequences == 1));
    if (!served) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            IMP_LOG_ERROR(
                "PLE: forward shape not served (rows %d, sequences %d, slots %d, prefill %d, "
                "staged %d, slab tail %d); PLE block skipped, output is wrong",
                n, state.n_sequences, state.ssm_n_seq, (int)state.is_prefill, (int)state.ple_host_ready,
                (int)(conv_state != nullptr));
        }
        return;
    }

    if (!state.ple_host_ready) {
        // Eager path (prefill, or a decode step the engine did not stage): read the ids and
        // the chunk's first position back, take the context from the token history. Pinned
        // landing zone: a D2H into pageable memory is a staged copy (245 us on WSL2).
        const size_t rb_bytes = static_cast<size_t>(n + 1) * sizeof(int32_t);
        if (ple_readback_.bytes() < rb_bytes)
            ple_readback_ = PinnedBuffer::acquire(cuda_host_pinned_allocator(), rb_bytes);
        std::vector<int32_t> rb_fallback;
        int32_t* rb = ple_readback_.as<int32_t>();
        if (!rb) {
            rb_fallback.resize(static_cast<size_t>(n) + 1);
            rb = rb_fallback.data();
        }
        IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(rb, state.token_ids, n * sizeof(int32_t),
                                           cudaMemcpyDeviceToHost, stream));
        IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(rb + n, state.positions, sizeof(int),
                                           cudaMemcpyDeviceToHost, stream));
        IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream));
        const int pos0 = rb[n];
        int32_t ctx[8];
        const bool known = ctx_len <= 8 &&
                           ngram_context_at(state.ple_hist_in, state.ple_hist_in_n, state.ple_hist_out,
                                            state.ple_hist_out_n, pos0, ctx_len, cfg.ple_eos_token_id, ctx);
        // Position == history index (R3): the history must hold ids[0] at pos0 when it reaches it.
        int32_t at_pos0 = 0;
        const bool in_hist = ngram_context_at(state.ple_hist_in, state.ple_hist_in_n, state.ple_hist_out,
                                              state.ple_hist_out_n, pos0 + 1, 1, cfg.ple_eos_token_id,
                                              &at_pos0);
        if (!known || (in_hist && at_pos0 != rb[0])) {
            static bool warned_ctx = false;
            if (!warned_ctx) {
                warned_ctx = true;
                IMP_LOG_ERROR(
                    "PLE: token history does not cover position %d (history %d+%d, token %d vs %d); "
                    "n-gram context is wrong",
                    pos0, state.ple_hist_in_n, state.ple_hist_out_n, in_hist ? at_pos0 : -1, rb[0]);
            }
        }
        if (pos0 == 0)
            IMP_CUDA_CHECK_LOG(cudaMemsetAsync(conv_state, 0, model_->ple_state_bytes(), stream));
        ple_prepare_host_(rb, n, 1, ctx, stream);
    }

    Tensor emb = view_tokens(ple_emb_dev_, n);  // [n, d] the gathered n-gram embedding
    Tensor key = view_tokens(hc_mixw_, n);     // [n, hc*d]: key_proj, then q, then gv (in place)
    Tensor keyn = view_tokens(hc_normed_, n);  // [n, hc*d]: normed key, then normed gv
    Tensor value = view_tokens(hc_mixed_, n);  // [n, d]
    Tensor hw = view_tokens(hc_hidden_, n);
    gemm(emb, L.ple_key_proj, key, 1.0f, 0.0f, stream);  // [n, d] x [hc*d, d]^T
    hc_grouped_rmsnorm(key, L.ple_norm_key, keyn, hc, d, cfg.rms_norm_eps, stream);
    gemm(emb, L.ple_value_proj, value, 1.0f, 0.0f, stream);  // [n, d] x [d, d]^T
    hc_grouped_rmsnorm(hw, L.ple_norm_query, key, hc, d, cfg.rms_norm_eps, stream);
    ple_gate_value(keyn, key, value, hc, d, stream);
    hc_grouped_rmsnorm(key, L.ple_norm_conv, keyn, hc, d, cfg.rms_norm_eps, stream);
    const int channels = hc * d;
    const int kernel = static_cast<int>(L.ple_conv1d.numel() / channels);
    const int64_t slot_stride = batched ? static_cast<int64_t>(ss->slot_stride_bytes() / sizeof(uint16_t))
                                        : 0;
    // Verify chunk: row-0 snapshot with the GDN state (spec_snap_slab), commit only the real rows.
    void* snap = (!batched && state.spec_snap_slab) ? ss->extra_state_in(state.spec_snap_slab) : nullptr;
    ple_conv_add(key, keyn, L.ple_conv1d, conv_state, slot_stride, batched ? state.ssm_seq_slots : nullptr,
                 n_seq, hw, channels, kernel, tab->ngram_size(), stream, snap, state.d_snap_n,
                 batched ? nullptr : state.d_chunk_len);
}

void GraphExecutor::ple_prepare_host_(const int32_t* ids, int n, int n_seq, const int32_t* ctx,
                                      cudaStream_t stream) {
    const NGramTable* tab = model_->ngram_table();
    const int d = model_->config().d_model;
    const int ctx_len = tab->context_len();
    const int rows = n / n_seq;
    for (int s = 0; s < n_seq; s++)
        tab->hash(ctx + static_cast<size_t>(s) * ctx_len, ids + static_cast<size_t>(s) * rows, rows,
                  ple_ids_.data() + static_cast<size_t>(s) * rows * tab->n_heads());
    // The previous chunk's H2D must have drained before the staging rows are rewritten.
    IMP_CUDA_CHECK_LOG(cudaEventSynchronize(ple_h2d_done_));
    tab->gather(ple_ids_.data(), n, ple_host_.as<uint16_t>());
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(ple_emb_dev_.data, ple_host_.data(),
                                       static_cast<size_t>(n) * d * 2, cudaMemcpyHostToDevice, stream));
    IMP_CUDA_CHECK_LOG(cudaEventRecord(ple_h2d_done_, stream));
}

bool GraphExecutor::prepare_decode_step_host(const int32_t* ids, int n, int n_seq, const int32_t* ple_ctx,
                                             cudaStream_t stream) {
    dev_expert_cache_.take_over(stream);
    if (model_->ngram_table() == nullptr || n <= 0 || n_seq <= 0 || n % n_seq != 0 || ple_ctx == nullptr ||
        n > static_cast<int>(ple_emb_dev_.shape[0]))
        return false;
    ple_prepare_host_(ids, n, n_seq, ple_ctx, stream);
    return true;
}

}  // namespace imp
