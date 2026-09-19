// Qwen4Exp gated residual (hyper-connections) around every block. Math in compute/gated_residual.h.
//
// The blocks keep their convention (hidden_ in, hidden_ + block_out out, pre-norm inside). Here the
// pre-norm weight is null, so rmsnorm_for_smallm_ is the identity and the block sees exactly the
// mixed input the model expects. hc_write_ then recovers block_out = hidden_ - mixed and injects it
// into the hc streams, which is the only state that carries across layers.

#include "compute/gated_residual.h"
#include "compute/gemm.h"
#include "core/logging.h"
#include "exec/executor.h"
#include "exec/executor_helpers.h"

namespace imp {

void GraphExecutor::hc_read_(const Tensor& norm_w, const Tensor& down, const Tensor& up, const Tensor* inject,
                             int n, cudaStream_t stream) {
    const auto& cfg = model_->config();
    const int hc = cfg.hc_count;
    const int d = cfg.d_model;
    Tensor hw = view_tokens(hc_hidden_, n);
    Tensor normed = view_tokens(hc_normed_, n);
    Tensor mixw = view_tokens(hc_mixw_, n);
    Tensor low = view_tokens(hc_low_, n);
    Tensor h = view_tokens(hidden_, n);
    Tensor mixed = view_tokens(hc_mixed_, n);

    // TEMP DEBUG (qwen4-exp bring-up): per-step sync checkpoints for the first three reads.
    static int dbg_hc = 0;
    const bool dbg = (dbg_hc < 3);
    if (dbg)
        ++dbg_hc;
    auto ck = [&](const char* what) {
        if (!dbg)
            return;
        const cudaError_t e = cudaDeviceSynchronize();
        IMP_LOG_WARN("hc dbg #%d n=%d: after %s: %s", dbg_hc, n, what, cudaGetErrorString(e));
    };
    hc_grouped_rmsnorm(hw, norm_w, normed, hc, d, cfg.rms_norm_eps, stream);
    ck("grouped_rmsnorm");
    gemm(normed, down, low, 1.0f, 0.0f, stream);  // [n, hc*d] x [lowrank, hc*d]^T
    ck("down gemm");
    hc_silu_div(low, hc, stream);
    gemm(low, up, mixw, 1.0f, 0.0f, stream);  // [n, lowrank] x [hc*d, lowrank]^T
    ck("up gemm");
    hc_mix(mixw, normed, h, hc, d, stream);
    ck("mix");
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(mixed.data, h.data, h.nbytes(), cudaMemcpyDeviceToDevice, stream));
    if (inject != nullptr) {
        Tensor inj = view_tokens(hc_inj_, n);
        gemm(normed, *inject, inj, 1.0f, 0.0f, stream);  // [n, hc*d] x [hc, hc*d]^T
        ck("inject gemm (N=hc)");
        hc_inject_weights(inj, hc, stream);
        ck("inject weights");
    }
}

bool GraphExecutor::hc_alloc_(int max_tokens) {
    if (!model_->profile().gated_residual)
        return true;
    const auto& cfg = model_->config();
    if (cfg.hc_count <= 0 || cfg.hc_lowrank <= 0) {
        IMP_LOG_ERROR("gated residual: hc_attn_norm present but hc_count=%d hc_lowrank=%d in the config",
                      cfg.hc_count, cfg.hc_lowrank);
        return false;
    }
    const int hc = cfg.hc_count;
    const int d = cfg.d_model;
    const size_t es = dtype_size(compute_dtype_);
    auto mk = [&](Tensor& t, int cols, const char* tag) -> bool {
        const size_t bytes = align256(static_cast<size_t>(max_tokens) * cols * es);
        void* p = vram_alloc(vram_alloc_, bytes, tag);
        if (p == nullptr)
            return false;
        const int64_t shape[2] = {max_tokens, cols};
        t = Tensor(p, compute_dtype_, 2, shape, true);
        return true;
    };
    const bool ok = mk(hc_hidden_, hc * d, "hc_hidden") && mk(hc_normed_, hc * d, "hc_normed") &&
                    mk(hc_mixw_, hc * d, "hc_mixw") && mk(hc_low_, cfg.hc_lowrank, "hc_low") &&
                    mk(hc_inj_, hc, "hc_inj") && mk(hc_mixed_, d, "hc_mixed") && mk(hc_out_, d, "hc_out");
    if (ok)
        IMP_LOG_INFO("gated residual: hc=%d lowrank=%d, %d streams x %d wide, %.1f MiB for %d tokens", hc,
                     cfg.hc_lowrank, hc, d,
                     (3.0 * hc * d + cfg.hc_lowrank + hc + 2.0 * d) * max_tokens * es / (1024.0 * 1024.0),
                     max_tokens);
    return ok;
}

void GraphExecutor::hc_free_() {
    for (Tensor* t : {&hc_hidden_, &hc_normed_, &hc_mixw_, &hc_low_, &hc_inj_, &hc_mixed_, &hc_out_}) {
        if (t->data != nullptr) {
            vram_free(vram_alloc_, t->data);
            t->data = nullptr;
        }
    }
}

void GraphExecutor::hc_write_(int n, cudaStream_t stream) {
    const auto& cfg = model_->config();
    Tensor h = view_tokens(hidden_, n);
    Tensor mixed = view_tokens(hc_mixed_, n);
    Tensor out = view_tokens(hc_out_, n);
    Tensor inj = view_tokens(hc_inj_, n);
    Tensor hw = view_tokens(hc_hidden_, n);
    hc_sub(h, mixed, out, stream);
    hc_inject_add(hw, out, inj, cfg.hc_count, cfg.d_model, stream);
}

}  // namespace imp
