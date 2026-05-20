// Pre-dequant Phase 1: FP16 cache.
// Converts all GGUF Q*_K-quantized weights to an FP16 device cache,
// gated by attention.mxfp4_fp16_cache_policy (legacy/pruned).
//
// Extracted from executor_pre_dequant.cu in Phase 3 of the architecture
// refactor roadmap. See pre_dequant_internal.h for shared helpers.

#include "exec/executor.h"
#include "exec/pre_dequant_internal.h"
#include "quant/dequant_gpu.h"
#include "core/logging.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

using imp::pre_dequant_internal::create_fused_weight_pair;
using imp::pre_dequant_internal::deduct_budget;
using imp::pre_dequant_internal::nvfp4_beneficial;

namespace imp {

void GraphExecutor::pre_dequant_phase1_fp16_cache_(
    const ModelConfig& cfg, const VRAMBudget& budget,
    size_t& remaining_budget, cudaStream_t stream) {
    size_t total_cache_bytes = 0;
    int cached_count = 0;
    bool budget_exhausted = false;

    if (wcache_.use_fp8) {
        IMP_LOG_INFO(
            "FP8 prefill: skipping FP16 cache (Phase 1), "
            "all dense weights → FP8 cache (Phase 2)");
        return;
    }
    if (budget.strategy == VRAMBudget::NVFP4_DECODE_ONLY) {
        IMP_LOG_INFO(
            "NVFP4 decode only: skipping FP16 cache (Phase 1), "
            "VRAM reserved for NVFP4 decode cache");
        return;
    }

    // --- Phase 1: FP16 weight cache + fused KV + fused gate+up ---
    auto cache_weight = [&](const Tensor& w, QType qtype) {
        if (!w.data || !dequant_gpu_supported(qtype))
            return;
        if (wcache_.fp16.count(w.data))
            return;  // already cached
        if (budget_exhausted)
            return;

        int rows = static_cast<int>(w.shape[0]);
        int cols = static_cast<int>(w.shape[1]);
        size_t fp16_bytes = static_cast<size_t>(rows) * cols * sizeof(half);

        if (total_cache_bytes + fp16_bytes > remaining_budget) {
            budget_exhausted = true;
            IMP_LOG_INFO(
                "FP16 cache: VRAM budget reached after %d tensors (%.1f / %.1f MiB), "
                "remaining weights will use on-the-fly dequant",
                cached_count, total_cache_bytes / (1024.0 * 1024.0),
                remaining_budget / (1024.0 * 1024.0));
            return;
        }

        void* fp16_buf = vram_alloc(vram_alloc_, fp16_bytes, "fp16_weight_cache");
        if (!fp16_buf) {
            budget_exhausted = true;
            IMP_LOG_WARN("FP16 cache: allocation failed after %d tensors (%.1f MiB)", cached_count,
                         total_cache_bytes / (1024.0 * 1024.0));
            return;
        }

        dequant_gpu(w.data, fp16_buf, qtype, rows, cols, stream);

        Tensor fp16_tensor(fp16_buf, QType::F16, w.ndim, w.shape, true);
        wcache_.fp16[w.data] = fp16_tensor;
        total_cache_bytes += fp16_bytes;
        cached_count++;
    };

    // Priority order: attention weights first (critical for cuBLAS prefill),
    // then SSM, shared experts, and dense FFN.  This ensures hybrid models
    // like Nemotron (23 SSM + 6 attention layers) cache all attention weights
    // before SSM weights exhaust the VRAM budget.
    for (int i = 0; i < cfg.n_layers; i++) {
        const auto& L = model_->layer(i);
        cache_weight(L.wq, L.wq.qtype);
        cache_weight(L.wk, L.wk.qtype);
        cache_weight(L.wv, L.wv.qtype);
        cache_weight(L.wo, L.wo.qtype);
    }
    for (int i = 0; i < cfg.n_layers; i++) {
        const auto& L = model_->layer(i);
        cache_weight(L.ssm_in, L.ssm_in.qtype);
        cache_weight(L.ssm_out, L.ssm_out.qtype);
        cache_weight(L.w_gate_shared, L.w_gate_shared.qtype);
        cache_weight(L.w_up_shared, L.w_up_shared.qtype);
        cache_weight(L.w_down_shared, L.w_down_shared.qtype);
        // When NVFP4 decode is active, skip dense FFN FP16 cache for eligible
        // weights.  Decode benefits more from NVFP4 (~47% BW reduction) than
        // prefill loses from on-the-fly dequant.  NVFP4 is also ~3.5x smaller
        // per tensor, so skipping FFN FP16 frees massive VRAM for full NVFP4.
        if (wcache_.nvfp4_decode_mode == 0 || !nvfp4_beneficial(L.w_gate.qtype))
            cache_weight(L.w_gate, L.w_gate.qtype);
        if (wcache_.nvfp4_decode_mode == 0 || !nvfp4_beneficial(L.w_up.qtype))
            cache_weight(L.w_up, L.w_up.qtype);
        if (wcache_.nvfp4_decode_mode == 0 || !nvfp4_beneficial(L.w_down.qtype))
            cache_weight(L.w_down, L.w_down.qtype);
    }

    // Create fused KV weights for strided batched prefill GEMM.
    // Each entry concatenates [wk; wv] as [2*nkv*hd, d_model] FP16 for one layer.
    int fused_kv_count = 0;
    for (int i = 0; i < cfg.n_layers; i++) {
        const auto& L = model_->layer(i);
        bool stop = false;
        if (create_fused_weight_pair(L.wk, L.wv, wcache_.fp16, vram_alloc_, total_cache_bytes,
                                     remaining_budget, stream, wcache_.fused_kv, i, stop))
            fused_kv_count++;
        else if (stop)
            break;
    }

    // Create fused gate+up weights for strided batched prefill GEMM.
    // Each entry concatenates [w_gate; w_up] as [2*d_ff, d_model] FP16 for one layer.
    int fused_gu_count = 0;
    for (int i = 0; i < cfg.n_layers; i++) {
        const auto& L = model_->layer(i);
        // Both must be the same shape (d_ff x d_model)
        if (L.w_gate.data && L.w_up.data &&
            (L.w_gate.shape[0] != L.w_up.shape[0] || L.w_gate.shape[1] != L.w_up.shape[1]))
            continue;
        bool stop = false;
        if (create_fused_weight_pair(L.w_gate, L.w_up, wcache_.fp16, vram_alloc_, total_cache_bytes,
                                     remaining_budget, stream, wcache_.fused_gate_up, i, stop))
            fused_gu_count++;
        else if (stop)
            break;
    }

    if (cached_count > 0) {
        IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream));
        wcache_.fp16_bytes = total_cache_bytes;
        IMP_LOG_INFO("FP16 weight cache: %d tensors, %.2f MiB (incl. %d fused KV, %d fused gate+up)",
                     cached_count, total_cache_bytes / (1024.0 * 1024.0), fused_kv_count, fused_gu_count);
    }

    // Deduct Phase 1 allocation from shared budget
    deduct_budget(remaining_budget, total_cache_bytes);
}

}  // namespace imp
