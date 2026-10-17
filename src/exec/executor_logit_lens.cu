// Logit lens (roadmap row 115): the forward copies the lens rows of hidden_ after every layer into a
// stash; the readout runs the tier-aware LM head once over [layers x rows] stashed rows and reports
// the sampled token's logprob and rank, the top-N and the entropy per (layer, row).

#include "exec/executor.h"
#include "exec/logit_lens.h"
#include "compute/prompt_logprobs_rows.h"
#include "core/logging.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace imp {

namespace {

// Bytes per stashed row: the FP16 row, then target, lp, rank, entropy and a kPlpMaxTopN top list.
size_t lens_row_bytes(int d_model) {
    return static_cast<size_t>(d_model) * sizeof(half) + 4 * sizeof(float) +
           static_cast<size_t>(kPlpMaxTopN) * (sizeof(int32_t) + sizeof(float));
}

}  // namespace

Tensor GraphExecutor::lm_head_rows_(int n) const { return lm_rows_ ? *lm_rows_ : view_tokens(hidden_, n); }

void GraphExecutor::set_lens_rows(std::vector<int> rows) {
    lens_rows_ = std::move(rows);
    lens_layers_ = 0;
    if (lens_rows_.empty() || vram_alloc_ == nullptr)
        return;
    const size_t need = static_cast<size_t>(model_->n_layers()) * lens_rows_.size() *
                            lens_row_bytes(model_->config().d_model) +
                        6 * size_t{256};  // alignment of the six output arrays
    if (need > lens_scratch_bytes_) {
        lens_scratch_bytes_ = 0;
        lens_scratch_ = VramOwned<char>(*vram_alloc_, need, "logit_lens");
        if (lens_scratch_)
            lens_scratch_bytes_ = need;
    }
    if (!lens_scratch_) {
        IMP_LOG_ERROR("logit lens: %zu bytes of scratch not allocatable, lens off", need);
        lens_rows_.clear();
    }
}

void GraphExecutor::lens_stash_(int layer, int n, cudaStream_t stream) {
    if (lens_rows_.empty() || layer >= model_->n_layers())
        return;
    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &cap) != cudaSuccess || cap != cudaStreamCaptureStatusNone)
        return;  // the engine runs lens steps eager; a capture must not bake the copies
    const int d = model_->config().d_model;
    const int r_n = static_cast<int>(lens_rows_.size());
    const size_t row_b = static_cast<size_t>(d) * sizeof(half);
    char* dst = lens_scratch_.get() + static_cast<size_t>(layer) * r_n * row_b;
    const char* src = static_cast<const char*>(hidden_.data);
    for (int r = 0; r < r_n; ++r) {
        if (lens_rows_[r] < 0 || lens_rows_[r] >= n)
            continue;
        IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(dst + static_cast<size_t>(r) * row_b,
                                           src + static_cast<size_t>(lens_rows_[r]) * row_b, row_b,
                                           cudaMemcpyDeviceToDevice, stream));
    }
    lens_layers_ = std::max(lens_layers_, layer + 1);
}

bool GraphExecutor::logit_lens_readout(const std::vector<int32_t>& targets, int top_n, LogitLensResult& out,
                                       cudaStream_t stream) {
    const int r_n = static_cast<int>(lens_rows_.size());
    if (!initialized_ || r_n == 0 || lens_layers_ == 0 || static_cast<int>(targets.size()) != r_n)
        return false;
    top_n = std::clamp(top_n, 0, kPlpMaxTopN);
    const auto& cfg = model_->config();
    const int d = cfg.d_model;
    const int n = lens_layers_ * r_n;
    // Stash first (sized for every layer), then the per-row outputs behind it.
    char* p = lens_scratch_.get() + static_cast<size_t>(model_->n_layers()) * r_n * d * sizeof(half);
    auto take = [&](size_t bytes) {
        char* q = p;
        p += (bytes + 255) & ~size_t{255};
        return q;
    };
    auto* d_tgt = reinterpret_cast<int32_t*>(take(n * sizeof(int32_t)));
    auto* d_lp = reinterpret_cast<float*>(take(n * sizeof(float)));
    auto* d_rank = reinterpret_cast<int32_t*>(take(n * sizeof(int32_t)));
    auto* d_ent = reinterpret_cast<float*>(take(n * sizeof(float)));
    auto* d_top_ids = reinterpret_cast<int32_t*>(take(static_cast<size_t>(n) * top_n * sizeof(int32_t)));
    auto* d_top_lp = reinterpret_cast<float*>(take(static_cast<size_t>(n) * top_n * sizeof(float)));
    if (p > lens_scratch_.get() + lens_scratch_bytes_)
        return false;

    std::vector<int32_t> h_tgt(n);
    for (int i = 0; i < n; ++i)
        h_tgt[i] = targets[i % r_n];
    IMP_CUDA_CHECK_LOG(
        cudaMemcpyAsync(d_tgt, h_tgt.data(), n * sizeof(int32_t), cudaMemcpyHostToDevice, stream));

    const int64_t shape[2] = {n, d};
    const Tensor rows(lens_scratch_.get(), QType::F16, 2, shape, true);
    // allow_cutlass=false: the FP16-activation head of batch-1 decode, so the last layer's row
    // reproduces the logprob the sampler saw.
    lm_rows_ = &rows;
    for_each_lm_head_batch_(n, stream, /*allow_cutlass=*/false, [&](const Tensor& lg, int row0, int csz) {
        prompt_logprobs_rows(static_cast<const float*>(lg.data), csz, cfg.vocab_size, cfg.vocab_size,
                             d_tgt + row0, top_n, d_lp + row0, d_rank + row0,
                             top_n > 0 ? d_top_ids + static_cast<int64_t>(row0) * top_n : nullptr,
                             top_n > 0 ? d_top_lp + static_cast<int64_t>(row0) * top_n : nullptr, stream,
                             d_ent + row0);
    });
    lm_rows_ = nullptr;

    out.layers = lens_layers_;
    out.rows = r_n;
    out.top_n = top_n;
    out.lp.resize(n);
    out.rank.resize(n);
    out.entropy.resize(n);
    out.top_ids.resize(static_cast<size_t>(n) * top_n);
    out.top_lp.resize(static_cast<size_t>(n) * top_n);
    auto d2h = [&](void* dst, const void* src, size_t bytes) {
        if (bytes > 0)
            IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, stream));
    };
    d2h(out.lp.data(), d_lp, n * sizeof(float));
    d2h(out.rank.data(), d_rank, n * sizeof(int32_t));
    d2h(out.entropy.data(), d_ent, n * sizeof(float));
    d2h(out.top_ids.data(), d_top_ids, out.top_ids.size() * sizeof(int32_t));
    d2h(out.top_lp.data(), d_top_lp, out.top_lp.size() * sizeof(float));
    return cudaStreamSynchronize(stream) == cudaSuccess;
}

}  // namespace imp
