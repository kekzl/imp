// DSpark block drafter (speculative.dspark_model): load, context feed from the executor feature tap,
// draft per verify step. Drafter: runtime/dspark_drafter.h; reference: tools/analysis/dspark_reference.py.
#include "runtime/engine.h"
#include "core/logging.h"
#include "exec/executor.h"
#include "model/model.h"
#include "runtime/dspark_drafter.h"
#include "runtime/request.h"

#include <cuda_fp16.h>

#include <algorithm>
#include <chrono>
#include <vector>

namespace imp {

bool Engine::init_dspark_() {
    const std::string& dir = runtime_config_.speculative.dspark_model;
    if (dir.empty())
        return true;
    const auto& mc = model_->config();
    if (model_->profile().is_moe || mc.is_mla() || offload_mgr_ != nullptr) {
        IMP_LOG_ERROR(
            "speculative.dspark_model: needs a dense, fully resident target (MoE, MLA and "
            "--gpu-layers are not supported)");
        return false;
    }
    auto dr = std::make_unique<DSparkDrafter>();
    const int max_ctx = config_.max_seq_len > 0 ? config_.max_seq_len : mc.max_seq_len;
    const DSparkOptions opt{.fp8 = runtime_config_.speculative.dspark_fp8, .graph = config_.use_cuda_graphs};
    if (std::string err = dr->load(vram_alloc_, dir, mc.d_model, mc.n_layers, max_ctx, stream_, opt);
        !err.empty()) {
        IMP_LOG_ERROR("speculative.dspark_model %s: %s", dir.c_str(), err.c_str());
        return false;
    }
    const int rows = executor_->max_tokens();
    const size_t feat_bytes = static_cast<size_t>(rows) * dr->feature_dim() * sizeof(half);
    dspark_feat_ = VramOwned<char>(vram_alloc_, feat_bytes, "dspark_features");
    if (!dspark_feat_) {
        IMP_LOG_ERROR("speculative.dspark_model: feature buffer (%zu MiB) allocation failed",
                      feat_bytes >> 20);
        return false;
    }
    feature_tap_configure(executor_->feature_tap, dr->target_layer_ids(), mc.n_layers, mc.d_model,
                          dspark_feat_.get(), rows);
    // One sequence's context; before the KV pool and the decode graphs are sized.
    if (config_.max_batch_size != 1) {
        IMP_LOG_INFO("max_batch_size %d -> 1: speculative.dspark_model drafts for one sequence",
                     config_.max_batch_size);
        config_.max_batch_size = 1;
        runtime_config_.runtime.max_batch_size = 1;
        if (scheduler_)
            scheduler_->clamp_max_batch_size(1);
    }
    IMP_LOG_INFO("dspark: %s, block %d, %zu target layers, %zu MiB drafter + %zu MiB feature rows (%d)",
                 dir.c_str(), dr->block_size(), dr->target_layer_ids().size(), dr->device_bytes() >> 20,
                 feat_bytes >> 20, rows);
    dspark_ = std::move(dr);
    return true;
}

void Engine::dspark_prefill_feed_(const Request& req, int offset, int n) {
    if (!dspark_)
        return;
    if (req.id != dspark_req_) {
        dspark_->reset();
        dspark_req_ = req.id;
    }
    dspark_->append_ctx(reinterpret_cast<const half*>(dspark_feat_.get()), dspark_->feature_dim(), n, offset,
                        stream_);
}

namespace {

// Host wall per draft step, logged every kWindow drafts: propose (syncs) and the post-verify append.
struct DSparkTiming {
    static constexpr int kWindow = 64;
    double propose_ms = 0.0, append_ms = 0.0;
    int n = 0;
};
DSparkTiming g_dspark_timing;

double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

std::vector<int32_t> Engine::dspark_draft_(const Request& req) {
    if (!dspark_ || req.id != dspark_req_ || req.output_tokens.empty())
        return {};
    if (runtime_config_.diagnostics.spec_trace)
        IMP_LOG_INFO("[dspark] draft anchor=%d pos=%d ctx_rows=%d next_pos=%d", req.output_tokens.back(),
                     req.context_len() - 1, dspark_->ctx_rows(), dspark_->next_pos());
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<int32_t> draft = dspark_->propose(req.output_tokens.back(), req.context_len() - 1, stream_);
    DSparkTiming& t = g_dspark_timing;
    t.propose_ms += ms_since(t0);
    if (++t.n == DSparkTiming::kWindow) {
        IMP_LOG_INFO("[dspark] %d drafts: propose %.2f ms, append %.2f ms per draft (host wall)", t.n,
                     t.propose_ms / t.n, t.append_ms / t.n);
        t = DSparkTiming{};
    }
    return draft;
}

void Engine::dspark_post_verify_(int rows, int pos0) {
    if (!dspark_)
        return;
    const auto t0 = std::chrono::steady_clock::now();
    dspark_->append_ctx(reinterpret_cast<const half*>(dspark_feat_.get()), dspark_->feature_dim(), rows, pos0,
                        stream_);
    g_dspark_timing.append_ms += ms_since(t0);
}

}  // namespace imp
