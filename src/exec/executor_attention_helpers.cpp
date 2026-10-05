// GraphExecutor attention helpers split from executor_attention.cpp (file-size gate): the separate
// QK RMSNorm, the n == 1 dp4a QKV route, the wo Q8_1 input, the MLA q_lora Q projection, and the
// Cohere2 parallel-block residual pair.
#include "core/logging.h"
#include "exec/executor.h"
#include "exec/executor_kernels.h"
#include "exec/gemm_context.h"
#include "memory/engine_arena.h"
#include "compute/attention_paged.h"
#include "compute/gemm.h"
#include "compute/layernorm.h"
#include "lora/lora_adapter.h"

namespace imp {

void GraphExecutor::qk_norm_separate_(const TransformerLayer& ly, Tensor& qv, Tensor& kk, int n, int nh,
                                      int nkv, int hd, float eps, cudaStream_t stream) {
    // Norm width from the weight's element count: heads*hd, a divisor of hd, else hd.
    auto norm_dim = [hd](const Tensor& w, int heads) -> int {
        const int wd = static_cast<int>(w.shape[0]);
        if (wd == heads * hd)
            return wd;
        return (wd > 0 && wd < hd && hd % wd == 0) ? wd : hd;
    };
    auto apply = [&](Tensor& x, const Tensor& w, int heads) {
        if (w.data == nullptr)
            return;
        const int d = norm_dim(w, heads);
        int64_t flat[2] = {static_cast<int64_t>(n) * heads * hd / d, d};
        Tensor view = x.reshape(2, flat);
        rmsnorm(view, w, view, eps, stream, norm_w_off_);
    };
    apply(qv, ly.attn_q_norm, nh);
    apply(kk, ly.attn_k_norm, nkv);
}

void GraphExecutor::mla_q_projection_(const TransformerLayer& ly, const Tensor& no, Tensor& qv, int n,
                                      float eps, const GemmContext& ctx) {
    if (ly.q_a_proj.data == nullptr) {
        gemm_via_handle_(ly.wq_id, no, qv, ctx);
        return;
    }
    const int64_t shape[2] = {n, model_->config().q_lora_rank};
    Tensor q_a(mla_q_a_buf_, QType::F16, 2, shape, true);
    gemm_via_handle_(ly.q_a_proj_id, no, q_a, ctx);
    rmsnorm(q_a, ly.q_a_layernorm, q_a, eps, ctx.stream, 0.0f);  // plain gamma, like kv_a_layernorm
    gemm_via_handle_(ly.wq_id, q_a, qv, ctx);
}

void GraphExecutor::quantize_attn_out_q8_(const Tensor& ao, int K, cudaStream_t stream) {
    if (paged_attention_take_q8_epilogue())
        return;
    quantize_fp16_to_q8_1(static_cast<const half*>(ao.data), static_cast<block_q8_1*>(qscratch_.q8_1_buf),
                          qscratch_.d8_buf, K, stream);
}

bool GraphExecutor::dp4a_qkv_route_(const TransformerLayer& ly, int n, const Tensor& no) const {
    return n == 1 && qscratch_.q8_1_buf != nullptr && qscratch_.d8_buf != nullptr && no.qtype == QType::F16 &&
           ly.wq.qtype == ly.wk.qtype && is_dp4a_qtype(ly.wq.qtype) && ly.wv.data != nullptr &&
           is_dp4a_qtype(ly.wv.qtype);
}

// Cohere2 parallel block: two [max_tokens, d_model] FP16 residual copies, charged as
// ExecT2Demand::parallel_block. Dereferenced unconditionally: a short arena refuses the load.
bool GraphExecutor::take_parallel_block_buffers_() {
    const auto& cfg = model_->config();
    if (!cfg.parallel_block || par_in_buf_ != nullptr)
        return true;
    if (compute_dtype_ != QType::F16) {
        IMP_LOG_ERROR("parallel_block needs an FP16 residual stream");
        return false;
    }
    const size_t bytes = static_cast<size_t>(max_tokens_) * static_cast<size_t>(cfg.d_model) * sizeof(half);
    auto in = engine_arena().take_bytes(bytes);
    auto attn = engine_arena().take_bytes(bytes);
    if (in.empty() || attn.empty()) {
        IMP_LOG_ERROR("parallel-block residual scratch (2 x %.1f MiB) unavailable from the T2 arena",
                      bytes / (1024.0 * 1024.0));
        return false;
    }
    par_in_buf_ = in.data();
    par_attn_buf_ = attn.data();
    return true;
}

void GraphExecutor::parallel_block_step_(int stage, int n, cudaStream_t stream) {
    if (!model_->config().parallel_block)
        return;
    const int64_t count = static_cast<int64_t>(n) * model_->config().d_model;
    const size_t bytes = static_cast<size_t>(count) * sizeof(half);
    auto* h = static_cast<half*>(hidden_.data);
    if (stage == 0) {
        device_copy_async(par_in_buf_, h, bytes, stream);
    } else if (stage == 1) {
        // FFN reads norm(layer input): park input + attn, restore the input.
        device_copy_async(par_attn_buf_, h, bytes, stream);
        device_copy_async(h, par_in_buf_, bytes, stream);
    } else {
        parallel_residual_merge(h, static_cast<const half*>(par_attn_buf_),
                                static_cast<const half*>(par_in_buf_), count, stream);
    }
}

}  // namespace imp
