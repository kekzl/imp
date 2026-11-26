// LoRA runtime delta (#522): y += scale*(x.A^T).B^T, A[r,K], B[N,r],
// x[n,K], y[n,N]. n==1 (decode): the two purpose-built GEMV kernels of
// compute/lora_rows.h (cuBLAS is NOT safe under CUDA-graph capture, status 14,
// same class as #556); skinny shapes (r<=64) make custom kernels sufficient.
// n>1 (prefill): two cuBLAS gemm() calls (prefill is never captured). Rank
// intermediate lives in a persistent scratch sized when the adapter loads
// (reserve_lora_scratch); the engine invalidates decode graphs on adapter swap.

#include "exec/executor.h"
#include "memory/device_alloc_all.h"
#include "compute/gemm.h"
#include "compute/lora_rows.h"
#include "core/graph_epoch.h"
#include "core/logging.h"
#include "exec/inkling_registry.h"
#include "lora/lora_adapter.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace imp {

bool GraphExecutor::reserve_lora_scratch(int max_rank) {
    // Generic: fp32 [max_rank] for decode + fp16 [max_tokens x max_rank] for prefill. Inkling: fp32
    // [rows][max_rank] per q / k / v stream (inkling_lora_scratch_bytes). Grow-only.
    const size_t need = inkling_ ? inkling_lora_scratch_bytes(inkling_->rows_cap, max_rank)
                                 : (sizeof(float) * static_cast<size_t>(max_rank)) +
                                       (sizeof(half) * static_cast<size_t>(max_tokens_) * max_rank);
    if (need <= lora_scratch_sz_)
        return true;
    // The new buffer first: a failed grow keeps the active adapter's scratch.
    DeviceOwned<void> grown = DeviceOwned<void>::allocate_plain(need);
    if (!grown) {
        IMP_LOG_ERROR("LoRA: scratch alloc failed (%zu B)", need);
        return false;
    }
    const void* const before = lora_scratch_.get();
    lora_scratch_ = std::move(grown);
    lora_scratch_sz_ = need;
    // I3: lora_delta_ runs inside captured forwards (7 call sites); a larger adapter moves the scratch.
    (void)bump_graph_epoch_if_moved(before, lora_scratch_.get());
    wire_inkling_lora_();  // the active adapter's Inkling scratch moved with it
    return true;
}

void GraphExecutor::set_lora(const LoraAdapter* adapter) {
    lora_ = adapter;
    if (adapter && !reserve_lora_scratch(adapter->max_rank())) {
        IMP_LOG_ERROR("LoRA: adapter disabled");
        lora_ = nullptr;
    }
    wire_inkling_lora_();
}

// Inkling's forward applies the active adapter to q / k / v / o itself (InklingForward::delta_).
void GraphExecutor::wire_inkling_lora_() {
    if (!inkling_)
        return;
    InklingLora& l = inkling_->lora;
    l = InklingLora{};
    if (!lora_ || !lora_scratch_)
        return;
    l.a = lora_;
    const size_t plane = static_cast<size_t>(inkling_->rows_cap) * static_cast<size_t>(lora_->max_rank());
    for (int i = 0; i < 3; ++i)
        l.t[i] = static_cast<float*>(lora_scratch_.get()) + (plane * static_cast<size_t>(i));
}

// Apply one projection's delta. x must be the projection INPUT in F16
// ([n, K] row-major, contiguous), y the projection OUTPUT ([n, N] F16);
// the delta accumulates into y.
void GraphExecutor::lora_delta_(const LoraWeights& w, const void* x, void* y, int n,
                                cudaStream_t stream) {
    const float s = lora_->scale();
    if (n == 1) {
        LoraRowsArgs a;
        a.A = static_cast<const half*>(w.A), a.B = static_cast<const half*>(w.B);
        a.r = w.r, a.K = w.K, a.N = w.N, a.scale = s;
        a.x = static_cast<const half*>(x), a.ldx = w.K;
        a.y = static_cast<half*>(y), a.ldy = w.N;
        a.rows = 1;
        a.t = static_cast<float*>(lora_scratch_.get());
        lora_delta_rows(a, stream);
        return;
    }
    // Prefill: t = x · A^T  (gemm computes C = alpha · A_in @ B_w^T + beta · C
    // with B_w row-major [N, K] — exactly A's [r, K] layout).
    half* t_buf = reinterpret_cast<half*>(static_cast<float*>(lora_scratch_.get()) + lora_->max_rank());
    int64_t x_shape[2] = {n, w.K};
    int64_t a_shape[2] = {w.r, w.K};
    int64_t t_shape[2] = {n, w.r};
    int64_t b_shape[2] = {w.N, w.r};
    int64_t y_shape[2] = {n, w.N};
    Tensor xt(const_cast<void*>(x), QType::F16, 2, x_shape, true);
    Tensor at(w.A, QType::F16, 2, a_shape, true);
    Tensor tt(t_buf, QType::F16, 2, t_shape, true);
    Tensor bt(w.B, QType::F16, 2, b_shape, true);
    Tensor yt(y, QType::F16, 2, y_shape, true);
    gemm(xt, at, tt, 1.0f, 0.0f, stream);
    gemm(tt, bt, yt, s, 1.0f, stream);
}

}  // namespace imp
