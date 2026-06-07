// Pre-dequant Phase 1: FP16 cache.
// Converts GGUF Q*_K-quantized weights to an FP16 device cache when the
// planner's per-weight tier decision selects FP16. Phase 5 PR #1 Commit 5.1.2
// replaces the old NVFP4_DECODE_ONLY all-or-nothing early-exit + per-FFN
// nvfp4_decode_mode heuristic with a centralised `effective_capabilities(kind,
// qtype)` check — mirrors the StoragePlanner's policy so Phase 1 and Phase 3
// agree on which weights belong where. Closes the 2026-05-24 Q4_K_M coverage
// gap (Gemma-3-12B Q4_K weights got zero cache because Phase 1 early-exited
// and Phase 3 only caches nvfp4_beneficial weights).
//
// Extracted from executor_pre_dequant.cu in Phase 3 of the architecture
// refactor roadmap. See pre_dequant_internal.h for shared helpers.

#include "exec/executor.h"
#include "exec/pre_dequant_internal.h"
#include "model/tensor_kind_table.h"
#include "quant/dequant_gpu.h"
#include "core/logging.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

using imp::pre_dequant_internal::create_fused_weight_pair;
using imp::pre_dequant_internal::deduct_budget;

namespace imp {

void GraphExecutor::pre_dequant_phase1_fp16_cache_(
    const ModelConfig& cfg, const VRAMBudget& budget,
    size_t& remaining_budget, cudaStream_t stream) {
    (void)budget;  // strategy gate retired — plan-driven now
    size_t total_cache_bytes = 0;
    int cached_count = 0;
    bool budget_exhausted = false;

    if (wcache_.use_fp8) {
        IMP_LOG_INFO(
            "FP8 prefill: skipping FP16 cache (Phase 1), "
            "all dense weights → FP8 cache (Phase 2)");
        return;
    }

    // --- Phase 1: FP16 weight cache + fused KV + fused gate+up ---
    // Plan-driven gate (Stage 1.3): FP16-cache a weight iff the StoragePlan
    // routes its source to the FP16 tier, OR the plan flagged it as needing an
    // FP16 companion. The plan (built in pre_dequant_weights + apply_arch_rules_)
    // already folds in every rule this gate used to compute inline:
    //   - Phase 3 owns nvfp4_beneficial weights (Q8_0/Q6_K/Q5_K) → tier NVFP4,
    //     not FP16 (the #428×#434 starvation guard, now structural).
    //   - FP8 unavailable → FP8-floor kinds fall back to FP16.
    //   - gemma-3: NVFP4-tier weights get fp16_companion so the Phase-3 decode
    //     cache is built FROM the FP16 copy, not from scratch (the <pad>/IMA
    //     corruption guard) — without a scattered arch check here.
    auto cache_weight = [&](const Tensor& w, QType qtype, TensorKind kind) {
        (void)kind;
        if (!w.data || !dequant_gpu_supported(qtype))
            return;
        if (wcache_.fp16.count(w.data))
            return;  // already cached
        if (budget_exhausted)
            return;
        const auto* e = storage_plan_.entry_of(w.data);
        if (!e || (e->tier != StorageTier::FP16 && !e->fp16_companion))
            return;  // plan routes this to NVFP4 / FP8 / native, or not in plan

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
    // before SSM weights exhaust the VRAM budget. Tier-decision is plan-driven
    // (storage_plan_); per-tensor budget pressure still applies.
    for (int i = 0; i < cfg.n_layers; i++) {
        const auto& L = model_->layer(i);
        cache_weight(L.wq, L.wq.qtype, TensorKind::WQ);
        cache_weight(L.wk, L.wk.qtype, TensorKind::WK);
        cache_weight(L.wv, L.wv.qtype, TensorKind::WV);
        cache_weight(L.wo, L.wo.qtype, TensorKind::WO);
    }
    for (int i = 0; i < cfg.n_layers; i++) {
        const auto& L = model_->layer(i);
        cache_weight(L.ssm_in, L.ssm_in.qtype, TensorKind::SSM_IN);
        cache_weight(L.ssm_out, L.ssm_out.qtype, TensorKind::SSM_OUT);
        cache_weight(L.w_gate_shared, L.w_gate_shared.qtype, TensorKind::W_GATE);
        cache_weight(L.w_up_shared, L.w_up_shared.qtype, TensorKind::W_UP);
        cache_weight(L.w_down_shared, L.w_down_shared.qtype, TensorKind::W_DOWN);
        cache_weight(L.w_gate, L.w_gate.qtype, TensorKind::W_GATE);
        cache_weight(L.w_up, L.w_up.qtype, TensorKind::W_UP);
        cache_weight(L.w_down, L.w_down.qtype, TensorKind::W_DOWN);
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
