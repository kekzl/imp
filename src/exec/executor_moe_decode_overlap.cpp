// Decode side streams of the device expert cache path (executor_forward_moe_nvfp4_host.cu): the shared
// expert FFN (moe.shared_expert_overlap) and the miss gather (moe.expert_gather_overlap) run beside the
// routed GEMVs; joins keep the add order, so output bits match the serial path.
#include "exec/executor.h"
#include "exec/executor_helpers.h"
#include "exec/executor_kernels.h"
#include "core/logging.h"
#include "exec/expert_batch_policy.h"
#include "compute/gemm.h"
#include "compute/gated_residual.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>

namespace imp {

static_assert(kHostExpertAutoBatch <= kDevExpertMaxRows, "auto batch rows must fit one device resolve");

void GraphExecutor::init_shexp_overlap_() {
    free_shexp_overlap_();
    const auto& prof = model_->profile();
    if (!dispatch_policy().moe.shared_expert_overlap || dispatch_policy().moe.no_shared_mlp ||
        compute_dtype_ != QType::F16 || prof.sandwich_norms || prof.sanitize_ffn_fp16)
        return;
    const auto& cfg = model_->config();
    int eff = 0;
    for (int i = 0; i < cfg.n_layers; ++i) {
        const auto& ly = model_->layer(i);
        if (dev_expert_cache_.layer_ready(i) && ly.w_up_shared.data != nullptr)
            eff = std::max(eff, static_cast<int>(ly.w_up_shared.shape[0]));
    }
    if (eff == 0)
        return;
    auto& o = shexp_overlap_;
    const size_t bytes = kDevExpertMaxRows * ((3 * static_cast<size_t>(eff)) + cfg.d_model) * sizeof(half);
    o.buf = static_cast<char*>(vram_alloc(vram_alloc_, bytes, "shexp_overlap"));
    if (o.buf == nullptr || !o.stream.create() || !o.fork.create() || !o.join.create()) {
        IMP_LOG_WARN("moe.shared_expert_overlap: side stream setup failed, shared expert stays serial");
        free_shexp_overlap_();
        return;
    }
    o.eff = eff;
    o.d = cfg.d_model;
    IMP_LOG_INFO(
        "moe.shared_expert_overlap: decode shared expert (eff %d) runs beside the expert miss gather", eff);
}

void GraphExecutor::free_shexp_overlap_() {
    auto& o = shexp_overlap_;
    o.stream.reset();
    o.fork.reset();
    o.join.reset();
    vram_free(vram_alloc_, o.buf);
    o.buf = nullptr;
    o.eff = o.d = 0;
    o.pending = false;
}

void GraphExecutor::init_gather_overlap_() {
    auto& g = gather_overlap_;
    g.stream.reset();
    g.resolved.reset();
    g.gathered.reset();
    g.pf_stream.reset();
    g.pf_layer = -1;
    if (!dispatch_policy().moe.expert_gather_overlap || moe_.entry_miss() == nullptr)
        return;
    if (!g.stream.create() || !g.resolved.create() || !g.gathered.create()) {
        IMP_LOG_WARN("moe.expert_gather_overlap: side stream setup failed, the gather stays in line");
        g.stream.reset();
        return;
    }
    IMP_LOG_INFO("moe.expert_gather_overlap: routed experts that hit run beside the expert miss gather");
    // moe.expert_prefetch issues prefetches; diagnostics.moe_expert_trace alone records the predictions
    // (main stream, in trace order) without prefetching.
    const int budget = std::clamp(dispatch_policy().moe.expert_prefetch, 0, kDevExpertPrefetchMax);
    const bool trace = !dispatch_policy().diagnostics.moe_expert_trace.empty();
    if ((budget > 0 || trace) &&
        !dev_expert_cache_.enable_prefetch(budget > 0 ? budget : 1,
                                           dispatch_policy().moe.expert_prefetch_blocks))
        IMP_LOG_WARN("moe.expert_prefetch: tables not allocated, no next-layer prediction");
    if (budget > 0 && dev_expert_cache_.prefetch_on() &&
        (!g.pf_stream.create() || !g.pf_fork.create() || !g.pf_read.create() || !g.pf_ready.create() ||
         !g.pf_gathered.create())) {
        IMP_LOG_WARN("moe.expert_prefetch: side stream setup failed, no prefetch");
        g.pf_stream.reset();
    }
    g.pf_x = {};
    if (dev_expert_cache_.prefetch_on() && dispatch_policy().moe.expert_prefetch_hc_norm &&
        model_->profile().gated_residual && vram_alloc_ != nullptr)
        g.pf_x = VramOwned<char>(*vram_alloc_, model_->config().d_model * sizeof(half), "expert_prefetch_x");
}

bool GraphExecutor::predict_next_experts_(int layer, const void* x, cudaStream_t stream) {
    auto& g = gather_overlap_;
    const int next = layer + 1;
    const auto& cfg = model_->config();
    if (!dev_expert_cache_.prefetch_on() || x == nullptr || cur_n_tokens_ != 1 || next >= cfg.n_layers ||
        !layer_has_moe(next) || !dev_expert_cache_.layer_ready(next) || dev_expert_cache_.layer_tiered(next))
        return false;
    const Tensor& gate = model_->layer(next).moe_gate;
    if (gate.data == nullptr || gate.qtype != QType::F16 || !gate.on_device || compute_dtype_ != QType::F16 ||
        gate.shape[0] != cfg.n_experts || gate.shape[1] != cfg.d_model)
        return false;
    const bool issue = static_cast<bool>(g.pf_stream);
    cudaStream_t ps = issue ? static_cast<cudaStream_t>(g.pf_stream) : stream;
    if (issue) {
        dev_expert_cache_.take_over(stream);  // a table reset must precede the fork, not race the resolve
        IMP_CUDA_CHECK_LOG(cudaEventRecord(g.pf_fork, stream));
        IMP_CUDA_CHECK_LOG(cudaStreamWaitEvent(ps, g.pf_fork, 0));
    }
    // Qwen4Exp: the hc streams this MoE block read, normed with the next block's mlp weights and summed.
    const auto& nly = model_->layer(next);
    if (g.pf_x && nly.hc_mlp_norm.data != nullptr &&
        hc_norm_sum(static_cast<const half*>(hc_hidden_.data), static_cast<const half*>(nly.hc_mlp_norm.data),
                    reinterpret_cast<half*>(g.pf_x.get()), cfg.hc_count, cfg.d_model, cfg.rms_norm_eps, ps))
        x = g.pf_x.get();
    gemv_gate_fp32(static_cast<const half*>(gate.data), static_cast<const half*>(x),
                   dev_expert_cache_.pf_logits(), cfg.n_experts, cfg.d_model, ps);
    if (issue)  // x (norm_out_) and hc_hidden_ read: main may overwrite after (it waits on the gather)
        IMP_CUDA_CHECK_LOG(cudaEventRecord(g.pf_read, ps));
    DeviceExpertCache::PredTrace tr;
    if (moe_.expert_trace != nullptr && moe_.trace_top_k == cfg.n_experts_active)
        tr = {moe_.expert_trace, moe_.trace_weights, moe_.trace_cursor,
              static_cast<unsigned long long>(moe_.trace_capacity)};
    dev_expert_cache_.predict_topk(dev_expert_cache_.pf_logits(), cfg.n_experts, cfg.n_experts_active, next,
                                   tr, ps);
    if (!issue)
        return false;
    dev_expert_cache_.prefetch_resolve(next, cfg.n_experts_active, ps);
    IMP_CUDA_CHECK_LOG(cudaEventRecord(g.pf_ready, ps));
    return true;
}

void GraphExecutor::join_expert_prefetch_(cudaStream_t stream) {
    auto& g = gather_overlap_;
    if (g.pf_layer < 0)
        return;
    IMP_CUDA_CHECK_LOG(cudaStreamWaitEvent(stream, g.pf_ready, 0));
    IMP_CUDA_CHECK_LOG(cudaStreamWaitEvent(stream, g.pf_gathered, 0));
    g.pf_layer = -1;
}

bool GraphExecutor::fork_shared_expert_(int layer, cudaStream_t stream, int d, const Tensor& no) {
    auto& o = shexp_overlap_;
    const auto& ly = model_->layer(layer);
    if (!o.stream || o.pending || ly.w_up_shared.data == nullptr || no.shape[0] > kDevExpertMaxRows ||
        d != o.d || ly.w_up_shared.shape[0] > o.eff)
        return false;
    const int n = static_cast<int>(no.shape[0]);
    const size_t e = static_cast<size_t>(kDevExpertMaxRows) * o.eff * sizeof(half);  // one [rows, eff] block
    IMP_CUDA_CHECK_LOG(cudaEventRecord(o.fork, stream));
    IMP_CUDA_CHECK_LOG(cudaStreamWaitEvent(o.stream, o.fork, 0));
    (void)shared_expert_core_(layer, o.stream, n, d, model_->config().rms_norm_eps, no, o.buf, o.buf + e,
                              o.buf + (2 * e), o.buf + (3 * e));
    IMP_CUDA_CHECK_LOG(cudaEventRecord(o.join, o.stream));
    o.pending = true;
    return true;
}

bool GraphExecutor::join_shared_expert_(cudaStream_t stream, int n, int d, Tensor& h) {
    auto& o = shexp_overlap_;
    if (!o.pending)
        return false;
    o.pending = false;
    IMP_CUDA_CHECK_LOG(cudaStreamWaitEvent(stream, o.join, 0));
    const int64_t shape[2] = {n, d};
    Tensor sh_down(o.buf + (3 * static_cast<size_t>(kDevExpertMaxRows) * o.eff * sizeof(half)),
                   compute_dtype_, 2, shape, true);
    elementwise_add(h, sh_down, stream);
    return true;
}

bool GraphExecutor::stage_dev_experts_(int layer, const int32_t* expert_indices, int top_k, int n_rows,
                                       bool non_gated_experts, size_t* ms_off, const void* x,
                                       cudaStream_t stream) {
    for (int p = 0; p < kExpertProjCount; ++p)
        ms_off[p] = dev_expert_cache_.ms_off(layer, p);
    // Gated only: relu_sqr runs in place, a second pass would square the hit rows twice.
    auto& g = gather_overlap_;
    const bool pf_here = g.pf_layer == layer && n_rows == 1;
    if (!g.stream || non_gated_experts || dev_expert_cache_.layer_tiered(layer) || !pf_here)
        join_expert_prefetch_(stream);  // no-op without a prefetch in flight
    if (!g.stream || non_gated_experts || dev_expert_cache_.layer_tiered(layer)) {
        dev_expert_cache_.resolve_and_stage(layer, expert_indices, top_k, n_rows, moe_.d_slot_idx, stream);
        return false;
    }
    if (pf_here)  // this layer's tables are settled once its prefetch resolve ran
        IMP_CUDA_CHECK_LOG(cudaStreamWaitEvent(stream, g.pf_ready, 0));
    g.pf_layer = -1;
    const bool pf_next = !non_gated_experts &&
                         predict_next_experts_(layer, n_rows == 1 ? x : nullptr, stream);
    dev_expert_cache_.resolve(layer, expert_indices, top_k, n_rows, moe_.d_slot_idx, moe_.entry_miss(),
                              stream);
    IMP_CUDA_CHECK_LOG(cudaEventRecord(g.resolved, stream));
    IMP_CUDA_CHECK_LOG(cudaStreamWaitEvent(g.stream, g.resolved, 0));
    dev_expert_cache_.gather(layer, top_k, n_rows, g.stream);
    if (pf_next)  // main overwrites the predictor's inputs only after `gathered`; the copy does not wait
        IMP_CUDA_CHECK_LOG(cudaStreamWaitEvent(g.stream, g.pf_read, 0));
    IMP_CUDA_CHECK_LOG(cudaEventRecord(g.gathered, g.stream));
    if (pf_next) {  // behind this layer's misses on the same stream: the next layer's gather follows it
        IMP_CUDA_CHECK_LOG(cudaStreamWaitEvent(g.stream, g.pf_ready, 0));
        dev_expert_cache_.prefetch_gather(layer + 1, g.stream);
        IMP_CUDA_CHECK_LOG(cudaEventRecord(g.pf_gathered, g.stream));
        g.pf_layer = layer + 1;
    }
    return true;
}

}  // namespace imp
