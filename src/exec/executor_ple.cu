// Qwen4Exp PLE (layer 1): host n-gram hash + gather from the mmapped F8 table, then key/value
// projections, the stream gate and the dilated depthwise conv on the device. Math in compute/ple.h.
//
// Scratch: everything between the previous block's hc_write_ and this layer's hc_read_ is free,
// so the projections and gates live in the hc_* buffers (ple_embed_dim == d_model is checked at
// alloc). Only the pinned staging row block and the conv state are PLE-owned.
//
// Sequence state (the 2 context tokens and the 9 conv rows) is single-sequence: reset when the
// chunk starts at position 0, carried otherwise. Batched decode and prefix-cache resumes are not
// modelled yet (logged once).

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
    const size_t st_bytes = align256(static_cast<size_t>(state_len) * channels * 2);
    void* p = vram_alloc(vram_alloc_, st_bytes, "ple_conv_state");
    if (p == nullptr)
        return false;
    const int64_t shape[2] = {state_len, channels};
    ple_conv_state_ = Tensor(p, QType::F16, 2, shape, true);
    IMP_CUDA_CHECK_LOG(cudaMemset(p, 0, st_bytes));
    IMP_CUDA_CHECK_LOG(cudaEventCreateWithFlags(&ple_h2d_done_, cudaEventDisableTiming));
    ple_ctx_.assign(static_cast<size_t>(tab->context_len()), cfg.ple_eos_token_id);
    ple_ids_.resize(static_cast<size_t>(max_tokens) * tab->n_heads());
    IMP_LOG_INFO("PLE: layer %d, conv kernel %d dilation %d (state %d rows x %d), %.1f MiB pinned staging",
                 layer, kernel, tab->ngram_size(), state_len, channels,
                 static_cast<double>(max_tokens) * d * 2 / (1024.0 * 1024.0));
    return true;
}

void GraphExecutor::ple_free_() {
    ple_host_.reset();
    if (ple_conv_state_.data != nullptr) {
        vram_free(vram_alloc_, ple_conv_state_.data);
        ple_conv_state_.data = nullptr;
    }
    if (ple_h2d_done_ != nullptr) {
        cudaEventDestroy(ple_h2d_done_);
        ple_h2d_done_ = nullptr;
    }
}

void GraphExecutor::ple_run_(const InferenceState& state, int layer, int n, cudaStream_t stream) {
    const NGramTable* tab = model_->ngram_table();
    const auto& cfg = model_->config();
    const auto& L = model_->layer(layer);
    const int hc = cfg.hc_count;
    const int d = cfg.d_model;
    const int ctx_len = tab->context_len();

    // Host side: token ids and the chunk's first position (sequence start = reset).
    // Pinned landing zone: a D2H into pageable memory is a staged copy (245 us on WSL2).
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
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(rb + n, state.positions, sizeof(int), cudaMemcpyDeviceToHost,
                                       stream));
    IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream));
    const int32_t* ids = rb;
    const int pos0 = rb[n];
    static bool warned_batch = false;
    if (state.ssm_n_seq > 1 && !warned_batch) {
        warned_batch = true;
        IMP_LOG_ERROR("PLE: batched decode (%d sequences) shares one n-gram context; output is wrong",
                      state.ssm_n_seq);
    }
    if (pos0 == 0) {
        std::fill(ple_ctx_.begin(), ple_ctx_.end(), cfg.ple_eos_token_id);
        IMP_CUDA_CHECK_LOG(cudaMemsetAsync(ple_conv_state_.data, 0, ple_conv_state_.nbytes(), stream));
    }
    tab->hash(ple_ctx_.data(), ids, n, ple_ids_.data());
    // The previous chunk's H2D must have drained before the staging rows are rewritten.
    IMP_CUDA_CHECK_LOG(cudaEventSynchronize(ple_h2d_done_));
    tab->gather(ple_ids_.data(), n, ple_host_.as<uint16_t>());
    for (int i = 0; i < ctx_len; i++) {
        const int src = n - ctx_len + i;
        ple_ctx_[i] = (src >= 0) ? ids[src] : ple_ctx_[src + ctx_len];
    }

    Tensor emb = view_tokens(hc_out_, n);      // [n, d] the gathered n-gram embedding
    Tensor key = view_tokens(hc_mixw_, n);     // [n, hc*d]: key_proj, then q, then gv (in place)
    Tensor keyn = view_tokens(hc_normed_, n);  // [n, hc*d]: normed key, then normed gv
    Tensor value = view_tokens(hc_mixed_, n);  // [n, d]
    Tensor hw = view_tokens(hc_hidden_, n);
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(emb.data, ple_host_.data(), static_cast<size_t>(n) * d * 2,
                                       cudaMemcpyHostToDevice, stream));
    IMP_CUDA_CHECK_LOG(cudaEventRecord(ple_h2d_done_, stream));
    gemm(emb, L.ple_key_proj, key, 1.0f, 0.0f, stream);  // [n, d] x [hc*d, d]^T
    hc_grouped_rmsnorm(key, L.ple_norm_key, keyn, hc, d, cfg.rms_norm_eps, stream);
    gemm(emb, L.ple_value_proj, value, 1.0f, 0.0f, stream);  // [n, d] x [d, d]^T
    hc_grouped_rmsnorm(hw, L.ple_norm_query, key, hc, d, cfg.rms_norm_eps, stream);
    ple_gate_value(keyn, key, value, hc, d, stream);
    hc_grouped_rmsnorm(key, L.ple_norm_conv, keyn, hc, d, cfg.rms_norm_eps, stream);
    const int channels = hc * d;
    const int kernel = static_cast<int>(L.ple_conv1d.numel() / channels);
    ple_conv_add(key, keyn, L.ple_conv1d, ple_conv_state_, hw, channels, kernel, tab->ngram_size(), stream);
}

}  // namespace imp
