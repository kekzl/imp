// Sparse autoencoder (roadmap row 116, exec/sae.h): load one SAE per engine, arm the rows of a
// forward (every position of a request with clamps, the sampled position of one with sae_top),
// read the features out after the step.

#include "runtime/engine.h"
#include "core/logging.h"
#include "exec/executor.h"
#include "exec/sae.h"
#include "model/model.h"
#include "runtime/request.h"

#include <cuda_runtime.h>

namespace imp {

int Engine::sae_load(const std::string& dir) {
    if (!model_ || !executor_)
        return 0;
    if (model_->profile().gated_residual) {  // the stream is hc_count x d_model, not what the SAE read
        IMP_LOG_ERROR("sae: %s refused, gated-residual models are not supported", dir.c_str());
        return 0;
    }
    SaeHost host;
    std::string why;
    if (!load_sae_dir(dir, model_->config().d_model, model_->n_layers(), host, &why)) {
        IMP_LOG_ERROR("%s", why.c_str());
        return 0;
    }
    auto dev = std::make_unique<SaeDevice>();
    dev->layer = host.layer;
    dev->d_model = host.d_model;
    dev->width = host.width;
    cudaStream_t s = decode_stream();
    auto up = [&](VramOwned<float>& dst, const std::vector<float>& src, const char* tag) {
        dst = VramOwned<float>(vram_alloc_, src.size(), tag);
        return dst && cudaMemcpyAsync(dst.get(), src.data(), src.size() * sizeof(float),
                                      cudaMemcpyHostToDevice, s) == cudaSuccess;
    };
    const bool ok = up(dev->w_enc, host.w_enc, "sae_w_enc") && up(dev->w_dec, host.w_dec, "sae_w_dec") &&
                    up(dev->b_enc, host.b_enc, "sae_b_enc") &&
                    up(dev->threshold, host.threshold, "sae_thr") &&
                    up(dev->b_dec, host.b_dec, "sae_b_dec") && cudaStreamSynchronize(s) == cudaSuccess;
    if (!ok) {
        IMP_LOG_ERROR("sae: %s refused, %.1f MiB not allocatable or upload failed", dir.c_str(),
                      static_cast<double>(2 * host.w_enc.size() * sizeof(float)) / (1024.0 * 1024.0));
        return 0;
    }
    sae_ = std::move(dev);
    executor_->set_sae(sae_.get());
    IMP_LOG_INFO("sae: %s loaded, layer %d, width %d", dir.c_str(), sae_->layer, sae_->width);
    return sae_->width;
}

int Engine::sae_width() const { return sae_ ? sae_->width : 0; }

namespace {

bool wants_sae(const Request& r) { return r.sae_top > 0 || !r.sae_clamp.empty(); }

}  // namespace

bool Engine::sae_arm_decode_(const std::vector<std::shared_ptr<Request>>& rows) {
    if (!sae_)
        return false;
    std::vector<SaeRowSpec> spec;  // decode: hidden_ row i is rows[i]
    for (int i = 0; i < static_cast<int>(rows.size()); ++i)
        if (wants_sae(*rows[i]))
            spec.push_back({i, rows[i]->sae_top > 0, rows[i]->sae_clamp});
    const bool on = !spec.empty();
    if (on || executor_->sae_armed())  // never leave an earlier arm on a foreign forward
        executor_->set_sae_rows(std::move(spec), decode_stream());
    return on;
}

void Engine::sae_arm_prefill_(const Request& req, int n_tokens, bool last_chunk, cudaStream_t stream) {
    if (!sae_)
        return;
    std::vector<SaeRowSpec> spec;
    if (wants_sae(req) && n_tokens > 0) {
        if (req.sae_clamp.empty()) {  // features only: the last row, it predicts the first output token
            if (last_chunk)
                spec.push_back({n_tokens - 1, true, {}});
        } else {
            for (int i = 0; i < n_tokens; ++i)
                spec.push_back({i, last_chunk && req.sae_top > 0 && i == n_tokens - 1, req.sae_clamp});
        }
    }
    if (!spec.empty() || executor_->sae_armed())
        executor_->set_sae_rows(std::move(spec), stream);
}

void Engine::sae_collect_(const std::vector<std::shared_ptr<Request>>& rows, cudaStream_t stream) {
    if (!sae_ || !executor_->sae_armed())
        return;
    SaeReadout out;
    const bool ok = executor_->sae_readout(out, stream);
    executor_->set_sae_rows({}, stream);
    int r = 0;  // readout rows come in row order: one per request with sae_top
    for (const auto& req : rows) {
        if (req->sae_top <= 0)
            continue;
        // One entry per output token even on a failed readout: output_sae stays parallel to output_tokens.
        if (ok && r < out.rows)
            req->output_sae.push_back(sae_token_info(out.acts.data() + (static_cast<size_t>(r) * out.width),
                                                     out.width, req->sae_top, out.rel_err[r]));
        else
            req->output_sae.emplace_back();
        ++r;
    }
    if (!ok)
        IMP_LOG_WARN("sae: readout failed, %d entr%s left empty", r, r == 1 ? "y" : "ies");
}

}  // namespace imp
