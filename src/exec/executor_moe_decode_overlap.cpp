// Decode side streams of the device expert cache path (executor_forward_moe_nvfp4_host.cu): the shared
// expert FFN (moe.shared_expert_overlap) and the miss gather (moe.expert_gather_overlap) run beside the
// routed GEMVs; joins keep the add order, so output bits match the serial path.
#include "exec/executor.h"
#include "exec/executor_helpers.h"
#include "exec/executor_kernels.h"
#include "core/logging.h"
#include "exec/expert_batch_policy.h"

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
    if (!dispatch_policy().moe.expert_gather_overlap || moe_.entry_miss() == nullptr)
        return;
    if (!g.stream.create() || !g.resolved.create() || !g.gathered.create()) {
        IMP_LOG_WARN("moe.expert_gather_overlap: side stream setup failed, the gather stays in line");
        g.stream.reset();
        return;
    }
    IMP_LOG_INFO("moe.expert_gather_overlap: routed experts that hit run beside the expert miss gather");
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
                                       bool non_gated_experts, size_t* ms_off, cudaStream_t stream) {
    for (int p = 0; p < kExpertProjCount; ++p)
        ms_off[p] = dev_expert_cache_.ms_off(layer, p);
    // Gated only: relu_sqr runs in place, a second pass would square the hit rows twice.
    auto& g = gather_overlap_;
    if (!g.stream || non_gated_experts || dev_expert_cache_.layer_tiered(layer)) {
        dev_expert_cache_.resolve_and_stage(layer, expert_indices, top_k, n_rows, moe_.d_slot_idx, stream);
        return false;
    }
    dev_expert_cache_.resolve(layer, expert_indices, top_k, n_rows, moe_.d_slot_idx, moe_.entry_miss(),
                              stream);
    IMP_CUDA_CHECK_LOG(cudaEventRecord(g.resolved, stream));
    IMP_CUDA_CHECK_LOG(cudaStreamWaitEvent(g.stream, g.resolved, 0));
    dev_expert_cache_.gather(layer, top_k, n_rows, g.stream);
    IMP_CUDA_CHECK_LOG(cudaEventRecord(g.gathered, g.stream));
    return true;
}

}  // namespace imp
