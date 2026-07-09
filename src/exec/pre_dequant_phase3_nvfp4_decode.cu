// Pre-dequant Phase 3: NVFP4 decode-cache quantization.
// Multi-step quantization of decode-side weights to NVFP4, including
// candidate collection, two-pass mode-1/2 quantize, FP8 migration of
// failed candidates, CUTLASS conversion, MXFP4-source conversion, and
// MoE expert caching.
//
// Extracted from executor_pre_dequant.cu in Phase 3 of the architecture
// refactor roadmap. This file keeps the entry point + the NVFP4 quantize
// helpers (collect / mode-1 / mode-2 / second-pass / LM-head / projections);
// the FP8 migration, CUTLASS/MXFP4 conversion, and MoE expert caching live in
// pre_dequant_phase3_fp8.cu, pre_dequant_phase3_cutlass.cu, and
// pre_dequant_phase3_moe.cu respectively.
//
// See pre_dequant_internal.h for shared helpers.

#include "exec/executor.h"
#include "memory/vram_query.h"
#include "exec/quant_pipeline.h"
#include "exec/pre_dequant_internal.h"
#include "compute/gemm_cutlass_sm120.h"
#include "compute/gemm_cutlass_mxfp4_sm120.h"
#include "quant/dequant_gpu.h"
#include "quant/gpt_oss_mxfp4_convert.h"
#include "quant/fp8_quant.h"
#include "quant/nvfp4_quant.h"
#include "quant/nvfp4_gemm.h"
#include "core/logging.h"
#include "memory/vram_allocator.h"
#include "runtime/config.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <algorithm>
#include <cstdlib>
#include <unordered_set>
#include <cstring>
#include <vector>

namespace imp {

using imp::pre_dequant_internal::deduct_budget;
using imp::pre_dequant_internal::for_each_dense_weight;
using imp::pre_dequant_internal::nvfp4_beneficial;

void QuantPipeline::nvfp4_decode_collect_candidates_(const ModelConfig& cfg,
                                                     Nvfp4DecodeContext& dctx) {
    // Dual-path mode: attention weights stay at FP8 for quality.
    if (wcache_->dual_path_quant) {
        for (int i = 0; i < cfg.n_layers; i++) {
            const auto& L = model_->layer(i);
            if (L.wq.data)
                dctx.exclude_ptrs.insert(L.wq.data);
            if (L.wk.data)
                dctx.exclude_ptrs.insert(L.wk.data);
            if (L.wv.data)
                dctx.exclude_ptrs.insert(L.wv.data);
            if (L.wo.data)
                dctx.exclude_ptrs.insert(L.wo.data);
        }
        IMP_LOG_INFO("Dual-path quant: excluding %zu attention weights from NVFP4 cache",
                     dctx.exclude_ptrs.size());
    }

    // GDN/SSM models: exclude ssm_in/ssm_out projections from NVFP4.
    // These feed the recurrent scan which accumulates quantization error
    // in state H across tokens. 4-bit degrades quality on 9B+ models.
    // Opt-in gemm.nvfp4_ssm_proj keeps them IN the cache to measure the
    // speed/quality tradeoff (mirrors nvfp4_lm_head_gdn). This gate covers the
    // GGUF-source hybrids (Qwen3.6-35B-A3B Q4_K_M); native-NVFP4 SSM weights
    // are handled identically by the phase0b register gate.
    if (!runtime_config().gemm.nvfp4_ssm_proj) {
        int n_ssm_excluded = 0;
        for (int i = 0; i < cfg.n_layers; i++) {
            const auto& L = model_->layer(i);
            if (L.ssm_in.data) {
                dctx.exclude_ptrs.insert(L.ssm_in.data);
                n_ssm_excluded++;
            }
            if (L.ssm_out.data) {
                dctx.exclude_ptrs.insert(L.ssm_out.data);
                n_ssm_excluded++;
            }
        }
        if (n_ssm_excluded > 0)
            IMP_LOG_INFO("GDN/SSM: excluding %d recurrent projections from NVFP4 cache", n_ssm_excluded);
    } else {
        IMP_LOG_INFO("GDN/SSM: nvfp4_ssm_proj ON — recurrent projections eligible for NVFP4 cache");
    }

    const bool decode_all = runtime_config().gemm.nvfp4_decode_all;
    // force_beneficial bypasses the nvfp4_beneficial(qtype) gate — used by the
    // opt-in SSM path so a Q4_K/Q5_K recurrent projection (normally only
    // eligible under nvfp4_decode_all) can be cached on its own merit.
    auto collect_weight_nvfp4 = [&](const Tensor& w, QType qtype, bool force_beneficial) {
        if (!w.data)
            return;
        if (!force_beneficial && !nvfp4_beneficial(qtype, decode_all))
            return;
        if (wcache_->nvfp4.count(w.data))
            return;
        // Skip excluded weights (dual-path attention, GDN/SSM recurrent projections)
        if (dctx.exclude_ptrs.count(w.data))
            return;

        int cols = static_cast<int>(w.shape[1]);
        if (cols % 16 != 0)
            return;

        bool from_scratch = (wcache_->fp16.find(w.data) == wcache_->fp16.end());
        if (from_scratch && (!dequant_gpu_supported(qtype) || !qscratch_->dequant))
            return;
        dctx.entries.push_back({w.data, w, qtype, from_scratch});
    };

    // LM head first: largest single weight (vocab × d_model), biggest bandwidth win.
    collect_weight_nvfp4(model_->output_proj(), model_->out_proj_.qtype, false);

    // Dense attention + FFN: every tensor benefits every decode step.
    for_each_dense_weight(*model_, cfg, [&](const Tensor& w, QType qtype) {
        collect_weight_nvfp4(w, qtype, false);
    });

    // GGUF-source GDN/SSM recurrent projections (opt-in). For native-NVFP4
    // hybrids the SSM weights are already cached in phase0b (no FP16/quant
    // source), so this loop is a no-op there; it only fires for quantized-
    // source hybrids (e.g. Qwen3.6-35B-A3B Q4_K_M), where it forces the Q4_K
    // recurrent projections into the NVFP4 decode cache to remove the FP16
    // dequant→cuBLAS SSM tax. Quality risk is highest here (recurrent scan).
    if (runtime_config().gemm.nvfp4_ssm_proj) {
        for (int i = 0; i < cfg.n_layers; i++) {
            const auto& L = model_->layer(i);
            if (L.ssm_in.data)
                collect_weight_nvfp4(L.ssm_in, L.ssm_in.qtype, true);
            if (L.ssm_out.data)
                collect_weight_nvfp4(L.ssm_out, L.ssm_out.qtype, true);
        }
    }
}

void QuantPipeline::nvfp4_decode_cache_fp16_lm_head_(const ModelConfig& cfg, cudaStream_t stream) {
    if (!runtime_config().gemm.nvfp4_lm_head)
        return;

    const Tensor& lm = model_->output_proj();
    // Only handle a native-precision (FP16/BF16) LM head that lives on device.
    // A quantized-source LM head (Q*_K/Q8_0) is already routed through
    // collect_candidates → nvfp4_beneficial, so skip it here.
    if (!lm.data || !lm.on_device)
        return;
    if (lm.qtype != QType::F16 && lm.qtype != QType::BF16)
        return;
    if (lm.ndim != 2)
        return;
    const int rows = static_cast<int>(lm.shape[0]);  // vocab_size
    const int cols = static_cast<int>(lm.shape[1]);  // d_model
    if (cols % 16 != 0)
        return;
    // Already cached (e.g. tied embeddings already promoted, or re-entry).
    if (wcache_->nvfp4.count(lm.data))
        return;

    // GDN/SSM-hybrid models: the LM head is quality-load-bearing for the
    // recurrent state; NVFP4 there degrades coherence (memory
    // lm_head_only_nvfp4_qwen3_6_refuted). Detect via any GDN/SSM layer.
    // Opt-in override (gemm.nvfp4_lm_head_gdn) to re-measure the tradeoff.
    if (!runtime_config().gemm.nvfp4_lm_head_gdn) {
        for (int i = 0; i < cfg.n_layers; i++) {
            const auto& L = model_->layer(i);
            if (L.ssm_in.data || L.ssm_out.data || L.gdn_gate.data) {
                IMP_LOG_INFO("NVFP4 LM head: skipped (GDN/SSM-hybrid model)");
                return;
            }
        }
    }

    float* d_absmax_buf = nullptr;
    float* d_tscale_buf = nullptr;
    IMP_CUDA_CHECK_LOG(cudaMalloc(&d_absmax_buf, sizeof(float)));
    IMP_CUDA_CHECK_LOG(cudaMalloc(&d_tscale_buf, sizeof(float)));

    Tensor fp16_view(lm.data, QType::F16, 2, lm.shape, /*on_device=*/true);
    NvFP4QuantResult result;
    quantize_fp16_to_nvfp4_async(fp16_view, result, d_absmax_buf, d_tscale_buf, stream);
    IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream));

    float h_tscale = 1.0f;
    IMP_CUDA_CHECK_LOG(cudaMemcpy(&h_tscale, d_tscale_buf, sizeof(float), cudaMemcpyDeviceToHost));
    result.tensor_scale = h_tscale;
    result.N = rows;
    result.K = cols;
    wcache_->nvfp4[lm.data] = result;

    IMP_CUDA_CHECK_LOG(cudaFree(d_absmax_buf));
    IMP_CUDA_CHECK_LOG(cudaFree(d_tscale_buf));

    const double nvfp4_mib =
        (static_cast<size_t>(rows) * cols / 2 + static_cast<size_t>(rows) * cols / 16) /
        (1024.0 * 1024.0);
    IMP_LOG_INFO("NVFP4 LM head: quantized FP16 [%d x %d] → NVFP4 (%.1f MiB), decode GEMV fast path",
                 rows, cols, nvfp4_mib);
}

// Quantize the recipe-excluded BF16/FP16 GDN + attention projections of a
// native-NVFP4 hybrid model into NVFP4 decode-cache entries. Mirrors
// nvfp4_decode_cache_fp16_lm_head_ exactly (same quantize call, same wcache
// insertion, same guards: weight on device, qtype F16/BF16, ndim 2, cols%16==0,
// not already cached) — phase 4 then auto-routes M=1 decode through gemv_nvfp4
// because the weight lands in wcache_->nvfp4 (decode_tier → NVFP4). Prefill is
// untouched: the unfused originals stay BF16 and the prefill tier still uses the
// full-precision GEMM path. Opt-in via gemm.nvfp4_attn_proj → the recipe-excluded
// BF16 attention q/k/v/o (stateless within a step, low quality risk).
//
// The analogous lever for the BF16 GDN/Mamba in_proj/out_proj was built and
// measured to REGRESS decode (−9% Nemotron, −20% Qwen3.6) — the tuned FP16 GEMV
// (70-81% HBM) beats the NVFP4 GEMV for the wide GDN-output shapes — so it was
// removed; keeping those projections FP16 is correct for speed, not just quality.
void QuantPipeline::nvfp4_decode_cache_fp16_projections_(const ModelConfig& cfg,
                                                         cudaStream_t stream) {
    const bool do_attn = runtime_config().gemm.nvfp4_attn_proj;
    if (!do_attn)
        return;

    float* d_absmax_buf = nullptr;
    float* d_tscale_buf = nullptr;
    IMP_CUDA_CHECK_LOG(cudaMalloc(&d_absmax_buf, sizeof(float)));
    IMP_CUDA_CHECK_LOG(cudaMalloc(&d_tscale_buf, sizeof(float)));

    int n_attn = 0;
    size_t bytes_attn = 0;

    // Quantize one native-precision (F16/BF16) device weight into wcache_->nvfp4.
    // Returns the NVFP4 byte cost on success, 0 if skipped. Same guard set as the
    // LM-head path; idempotent (skips weights already cached).
    auto quantize_one = [&](const Tensor& w) -> size_t {
        if (!w.data || !w.on_device)
            return 0;
        if (w.qtype != QType::F16 && w.qtype != QType::BF16)
            return 0;  // already NVFP4/quantized, or a non-2-byte source
        if (w.ndim != 2)
            return 0;
        const int rows = static_cast<int>(w.shape[0]);
        const int cols = static_cast<int>(w.shape[1]);
        if (cols % 16 != 0)
            return 0;
        if (wcache_->nvfp4.count(w.data))
            return 0;  // already cached (e.g. re-entry)

        Tensor fp16_view(w.data, QType::F16, 2, w.shape, /*on_device=*/true);
        NvFP4QuantResult result;
        quantize_fp16_to_nvfp4_async(fp16_view, result, d_absmax_buf, d_tscale_buf, stream);
        IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream));

        float h_tscale = 1.0f;
        IMP_CUDA_CHECK_LOG(cudaMemcpy(&h_tscale, d_tscale_buf, sizeof(float), cudaMemcpyDeviceToHost));
        result.tensor_scale = h_tscale;
        result.N = rows;
        result.K = cols;
        wcache_->nvfp4[w.data] = result;
        return static_cast<size_t>(rows) * cols / 2 + static_cast<size_t>(rows) * cols / 16;
    };

    for (int i = 0; i < cfg.n_layers; i++) {
        const auto& L = model_->layer(i);
        for (const Tensor* w : {&L.wq, &L.wk, &L.wv, &L.wo}) {
            size_t b = quantize_one(*w);
            if (b) {
                n_attn++;
                bytes_attn += b;
            }
        }
    }

    IMP_CUDA_CHECK_LOG(cudaFree(d_absmax_buf));
    IMP_CUDA_CHECK_LOG(cudaFree(d_tscale_buf));

    if (n_attn > 0)
        IMP_LOG_INFO("NVFP4 attn proj: quantized %d BF16 q/k/v/o weights -> NVFP4 (%.1f MiB), decode GEMV fast path",
                     n_attn, bytes_attn / (1024.0 * 1024.0));
    else
        IMP_LOG_INFO("NVFP4 attn proj: no eligible BF16 q/k/v/o (flag on but model has none)");
}

void QuantPipeline::pre_dequant_phase3_nvfp4_decode_(
    const ModelConfig& cfg, const VRAMBudget& budget,
    size_t& remaining_budget, cudaStream_t stream) {
    if (wcache_->nvfp4_decode_mode <= 0)
        return;
    if (runtime_config().diagnostics.no_nvfp4_decode_cache) {
        IMP_LOG_INFO("NVFP4 decode cache DISABLED (diagnostics.no_nvfp4_decode_cache) — "
                     "decode runs on source-precision paths");
        return;
    }

    Nvfp4DecodeContext dctx;
    dctx.mode_str = (wcache_->nvfp4_decode_mode == 1) ? "additive" : "only";

    // Compute the shared mode-2 safety reserve once. Mode 1 keeps the upfront
    // 10% headroom (see vram_budget.cpp:50), so its budget arithmetic already
    // protects against shared/system-memory fallback; the in-loop safety is a
    // backstop only. Mode 2 omits the upfront 10% to fit larger weight caches
    // and previously paid for it with a 10% in-loop safety (3.2 GiB on a 32 GiB
    // 5090) that starved the dense NVFP4 cache. Replace with the same formula
    // the MoE expert path already uses: a KV-headroom estimate at 16 K tokens
    // plus a 256 MiB workspace cushion, clamped to [256 MiB, 1 GiB].
    if (wcache_->nvfp4_decode_mode == 2) {
        int n_attn_layers = 0;
        for (int i = 0; i < cfg.n_layers; i++) {
            if (model_->layer(i).wq.data != nullptr &&
                model_->layer(i).gdn_gate.data == nullptr)
                n_attn_layers++;
        }
        if (n_attn_layers == 0)
            n_attn_layers = cfg.n_layers;
        int hd = cfg.head_dim > 0 ? cfg.head_dim : (cfg.d_model / cfg.n_heads);
        int kv_heads = cfg.n_kv_heads > 0 ? cfg.n_kv_heads : cfg.n_heads;
        constexpr int kKvFloorTokens = 16384;
        size_t per_token_kv = static_cast<size_t>(n_attn_layers) * 2 *
                              static_cast<size_t>(kv_heads) * static_cast<size_t>(hd) * 2;
        size_t kv_reserve = static_cast<size_t>(kKvFloorTokens) * per_token_kv;
        constexpr size_t kWorkspaceSafety = 256ULL * 1024 * 1024;
        constexpr size_t kReserveCap = 1024ULL * 1024 * 1024;
        constexpr size_t kReserveFloor = 256ULL * 1024 * 1024;
        dctx.safety_reserve = std::clamp(kv_reserve + kWorkspaceSafety, kReserveFloor, kReserveCap);
    }

    nvfp4_decode_collect_candidates_(cfg, dctx);

    // Aliases keep the body that hasn't been extracted yet readable.
    const char* mode_str = dctx.mode_str;
    using NvFP4Entry = Nvfp4DecodeContext::Entry;
    std::vector<NvFP4Entry>& nvfp4_entries = dctx.entries;

    if (wcache_->nvfp4_decode_mode == 2 && !nvfp4_entries.empty()) {
        nvfp4_decode_quantize_mode2_(stream, dctx);
    } else if (!nvfp4_entries.empty()) {
        nvfp4_decode_quantize_mode1_(remaining_budget, stream, dctx);
    }

    if (wcache_->nvfp4_decode_mode == 2 && !wcache_->fp16.empty()) {
        nvfp4_decode_free_fp16_and_migrate_fp8_(remaining_budget, stream, dctx);
    }

    if (budget.nvfp4_second_pass && !nvfp4_entries.empty()) {
        nvfp4_decode_second_pass_(budget, stream, dctx);
    }

    // Native-NVFP4 models store the LM head in FP16/BF16 — quantize it to an
    // NVFP4 decode-cache entry so decode uses the fast GEMV instead of a cuBLAS
    // FP16 GEMV over vocab×d_model (~0.78 ms/token, ~19% of decode on Qwen3-8B).
    // Run after dense quantize so the entry is committed before CUTLASS convert.
    nvfp4_decode_cache_fp16_lm_head_(cfg, stream);

    // Native-NVFP4 hybrids store some attention projections BF16 (recipe
    // exclusion). Opt-in (gemm.nvfp4_attn_proj) quantizes the q/k/v/o into the
    // same NVFP4 decode cache. Run after the LM head, before CUTLASS convert so
    // these entries get the same block-scaled treatment.
    nvfp4_decode_cache_fp16_projections_(cfg, stream);

    if (!wcache_->nvfp4.empty() && cutlass_sm120_nvfp4_available()) {
        nvfp4_decode_convert_cutlass_(cfg, budget, remaining_budget, stream);
    }

    nvfp4_decode_convert_mxfp4_and_native_(cfg, stream);

    if (qscratch_->mxfp4_act_sf != nullptr && cutlass_sm120_mxfp4_available()) {
        nvfp4_decode_mxfp4_fp16_fallback_(cfg, stream);
    }

    if (model_->profile().is_gpt_oss)
        gpt_oss_convert_moe_experts_(cfg, dctx);
    nvfp4_decode_cache_moe_experts_(cfg, budget, remaining_budget, stream, dctx);
}

// Mode 2 ("only") incremental NVFP4 quantize. Process FP16-cached entries
// first (each conversion nets VRAM since NVFP4 ≈ 28% of FP16), then the
// from-scratch entries until VRAM is exhausted. Frees each source FP16
// entry immediately after the corresponding NVFP4 result is committed.
void QuantPipeline::nvfp4_decode_quantize_mode2_(cudaStream_t stream, Nvfp4DecodeContext& dctx) {
    using NvFP4Entry = Nvfp4DecodeContext::Entry;
    auto& nvfp4_entries = dctx.entries;

    // Sort: FP16-cached first (smallest first to bootstrap), then from-scratch.
    std::stable_sort(nvfp4_entries.begin(), nvfp4_entries.end(),
                     [](const NvFP4Entry& a, const NvFP4Entry& b) {
                         if (a.from_scratch != b.from_scratch)
                             return !a.from_scratch;
                         size_t a_sz = static_cast<size_t>(a.weight.shape[0]) * a.weight.shape[1];
                         size_t b_sz = static_cast<size_t>(b.weight.shape[0]) * b.weight.shape[1];
                         return a_sz < b_sz;
                     });

    float* d_absmax_buf = nullptr;
    float* d_tscale_buf = nullptr;
    IMP_CUDA_CHECK_LOG(cudaMalloc(&d_absmax_buf, sizeof(float)));
    IMP_CUDA_CHECK_LOG(cudaMalloc(&d_tscale_buf, sizeof(float)));

    int actual_count = 0;
    size_t actual_bytes = 0;
    int actual_from_fp16 = 0;
    int actual_from_scratch = 0;

    for (auto& e : nvfp4_entries) {
        int rows = static_cast<int>(e.weight.shape[0]);
        int cols = static_cast<int>(e.weight.shape[1]);
        size_t nvfp4_bytes = static_cast<size_t>(rows) * cols / 2 +
                             static_cast<size_t>(rows) * cols / 16 + 4;

        // Check actual free VRAM against the per-call safety reserve computed
        // in pre_dequant_phase3_nvfp4_decode_ (see dctx.safety_reserve).
        size_t free_mem = 0, total_mem = 0;
        vram_budget_mem_get_info(&free_mem, &total_mem);
        size_t nvfp4_safety = std::max(dctx.safety_reserve, static_cast<size_t>(1024 * 1024));
        if (free_mem < nvfp4_bytes + nvfp4_safety) {
            IMP_LOG_INFO(
                "NVFP4 incremental: VRAM exhausted after %d tensors "
                "(%.1f MiB, %.1f MiB free)",
                actual_count, actual_bytes / (1024.0 * 1024.0), free_mem / (1024.0 * 1024.0));
            break;
        }

        const half* fp16_ptr = nullptr;
        void* tmp_buf = nullptr;

        if (e.from_scratch) {
            size_t need = static_cast<size_t>(rows) * cols * sizeof(half);
            void* dq_buf = qscratch_->dequant;
            if (need > qscratch_->dequant_size) {
                if (cudaMalloc(&tmp_buf, need) != cudaSuccess || !tmp_buf)
                    continue;
                dq_buf = tmp_buf;
            }
            dequant_gpu(e.weight.data, dq_buf, e.qtype, rows, cols, stream);
            fp16_ptr = reinterpret_cast<const half*>(dq_buf);
        } else {
            auto it = wcache_->fp16.find(e.orig_ptr);
            fp16_ptr = reinterpret_cast<const half*>(it->second.data);
        }

        Tensor fp16_view(const_cast<half*>(fp16_ptr), QType::F16, 2, e.weight.shape, true);

        NvFP4QuantResult result;
        quantize_fp16_to_nvfp4_async(fp16_view, result, d_absmax_buf, d_tscale_buf, stream);

        // Sync immediately so we can read tensor_scale and free FP16
        IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream));

        float h_tscale;
        IMP_CUDA_CHECK_LOG(
            cudaMemcpy(&h_tscale, d_tscale_buf, sizeof(float), cudaMemcpyDeviceToHost));
        result.tensor_scale = h_tscale;
        wcache_->nvfp4[e.orig_ptr] = result;
        actual_bytes += nvfp4_bytes;
        actual_count++;

        if (tmp_buf)
            IMP_CUDA_CHECK_LOG(cudaFree(tmp_buf));

        // Free FP16 cache entry to reclaim VRAM for next weight
        if (!e.from_scratch) {
            auto it = wcache_->fp16.find(e.orig_ptr);
            if (it != wcache_->fp16.end()) {
                size_t freed = it->second.nbytes();
                vram_free(vram_alloc_, it->second.data);
                wcache_->fp16.erase(it);
                wcache_->fp16_bytes -= freed;
                actual_from_fp16++;
            }
        } else {
            actual_from_scratch++;
        }
    }

    IMP_CUDA_CHECK_LOG(cudaFree(d_absmax_buf));
    IMP_CUDA_CHECK_LOG(cudaFree(d_tscale_buf));

    wcache_->nvfp4_bytes = actual_bytes;
    IMP_LOG_INFO(
        "NVFP4 decode cache: %d tensors, %.2f MiB "
        "(%d from FP16, %d from scratch, mode: %s)",
        actual_count, actual_bytes / (1024.0 * 1024.0), actual_from_fp16, actual_from_scratch,
        dctx.mode_str);
}

// Mode 1 ("additive") batch NVFP4 quantize. Pick entries fitting the
// remaining VRAM budget, quantize them via a single batched
// quantize_fp16_to_nvfp4_async pass, then commit tensor_scales after one
// stream sync.
void QuantPipeline::nvfp4_decode_quantize_mode1_(size_t& remaining_budget, cudaStream_t stream,
                                                 Nvfp4DecodeContext& dctx) {
    using NvFP4Entry = Nvfp4DecodeContext::Entry;
    auto& nvfp4_entries = dctx.entries;
    (void)remaining_budget;  // read-only here; budget bookkeeping done after MoE phase

    size_t budget_used = 0;
    int nvfp4_count = 0;
    int nvfp4_from_scratch = 0;
    bool budget_exhausted = false;

    std::vector<NvFP4Entry> budgeted;
    for (auto& e : nvfp4_entries) {
        size_t rows = e.weight.shape[0], cols = e.weight.shape[1];
        size_t nvfp4_bytes = rows * cols / 2 + rows * cols / 16 + 4;
        if (budget_used + nvfp4_bytes > remaining_budget) {
            if (!budget_exhausted) {
                budget_exhausted = true;
                IMP_LOG_INFO(
                    "NVFP4 cache: VRAM budget reached after %d/%zu tensors "
                    "(%.1f / %.1f MiB)",
                    nvfp4_count, nvfp4_entries.size(), budget_used / (1024.0 * 1024.0),
                    remaining_budget / (1024.0 * 1024.0));
            }
            continue;
        }
        budget_used += nvfp4_bytes;
        nvfp4_count++;
        if (e.from_scratch)
            nvfp4_from_scratch++;
        budgeted.push_back(e);
    }

    float* d_absmax_buf = nullptr;
    IMP_CUDA_CHECK_LOG(cudaMalloc(&d_absmax_buf, sizeof(float)));

    float* d_tscales_all = nullptr;
    IMP_CUDA_CHECK_LOG(cudaMalloc(&d_tscales_all, budgeted.size() * sizeof(float)));

    std::vector<void*> tmp_bufs;
    for (size_t i = 0; i < budgeted.size(); i++) {
        auto& e = budgeted[i];
        const half* fp16_ptr = nullptr;
        int rows = static_cast<int>(e.weight.shape[0]);
        int cols = static_cast<int>(e.weight.shape[1]);

        if (e.from_scratch) {
            size_t need = static_cast<size_t>(rows) * cols * sizeof(half);
            void* dq_buf = qscratch_->dequant;
            if (need > qscratch_->dequant_size) {
                void* tmp = nullptr;
                if (cudaMalloc(&tmp, need) != cudaSuccess || !tmp)
                    continue;
                dq_buf = tmp;
                tmp_bufs.push_back(tmp);
            }
            dequant_gpu(e.weight.data, dq_buf, e.qtype, rows, cols, stream);
            fp16_ptr = reinterpret_cast<const half*>(dq_buf);
        } else {
            auto it = wcache_->fp16.find(e.orig_ptr);
            fp16_ptr = reinterpret_cast<const half*>(it->second.data);
        }

        Tensor fp16_view(const_cast<half*>(fp16_ptr), QType::F16, 2, e.weight.shape, true);

        NvFP4QuantResult result;
        quantize_fp16_to_nvfp4_async(fp16_view, result, d_absmax_buf, d_tscales_all + i, stream);
        wcache_->nvfp4[e.orig_ptr] = result;
    }

    IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream));
    for (void* p : tmp_bufs)
        IMP_CUDA_CHECK_LOG(cudaFree(p));

    std::vector<float> h_tscales(budgeted.size());
    IMP_CUDA_CHECK_LOG(cudaMemcpy(h_tscales.data(), d_tscales_all, budgeted.size() * sizeof(float),
                                  cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < budgeted.size(); i++) {
        auto it = wcache_->nvfp4.find(budgeted[i].orig_ptr);
        if (it != wcache_->nvfp4.end()) {
            it->second.tensor_scale = h_tscales[i];
        }
    }

    IMP_CUDA_CHECK_LOG(cudaFree(d_absmax_buf));
    IMP_CUDA_CHECK_LOG(cudaFree(d_tscales_all));

    wcache_->nvfp4_bytes = budget_used;
    if (nvfp4_from_scratch > 0) {
        IMP_LOG_INFO(
            "NVFP4 decode cache: %d tensors, %.2f MiB (%d from FP16 cache, %d via dequant scratch, "
            "mode: %s)",
            nvfp4_count, budget_used / (1024.0 * 1024.0), nvfp4_count - nvfp4_from_scratch,
            nvfp4_from_scratch, dctx.mode_str);
    } else {
        IMP_LOG_INFO("NVFP4 decode cache: %d tensors, %.2f MiB (mode: %s)", nvfp4_count,
                     budget_used / (1024.0 * 1024.0), dctx.mode_str);
    }
}

// NVFP4 second pass: after the FP16-free + FP8 migration phase frees VRAM,
// re-attempt NVFP4 quantization for entries skipped earlier due to budget
// pressure. Same per-tensor cudaMemGetInfo gate as mode 2.
void QuantPipeline::nvfp4_decode_second_pass_(const VRAMBudget& budget, cudaStream_t stream,
                                              Nvfp4DecodeContext& dctx) {
    (void)budget;
    auto& nvfp4_entries = dctx.entries;

    float* d_absmax_buf2 = nullptr;
    float* d_tscale_buf2 = nullptr;
    IMP_CUDA_CHECK_LOG(cudaMalloc(&d_absmax_buf2, sizeof(float)));
    IMP_CUDA_CHECK_LOG(cudaMalloc(&d_tscale_buf2, sizeof(float)));

    int second_count = 0;
    size_t second_bytes = 0;

    for (auto& e : nvfp4_entries) {
        if (wcache_->nvfp4.count(e.orig_ptr))
            continue;  // already cached
        int rows = static_cast<int>(e.weight.shape[0]);
        int cols = static_cast<int>(e.weight.shape[1]);
        size_t nvfp4_bytes = static_cast<size_t>(rows) * cols / 2 +
                             static_cast<size_t>(rows) * cols / 16 + 4;

        size_t free_mem2 = 0, total_mem2 = 0;
        vram_budget_mem_get_info(&free_mem2, &total_mem2);
        size_t nvfp4_safety2 = vram_reserve_floor(total_mem2);
        if (free_mem2 < nvfp4_bytes + nvfp4_safety2)
            break;

        // Dequant from quantized weights via scratch buffer
        size_t need = static_cast<size_t>(rows) * cols * sizeof(half);
        void* dq_buf = qscratch_->dequant;
        void* tmp_buf = nullptr;
        if (!dequant_gpu_supported(e.qtype) || !qscratch_->dequant)
            continue;
        if (need > qscratch_->dequant_size) {
            if (cudaMalloc(&tmp_buf, need) != cudaSuccess || !tmp_buf)
                continue;
            dq_buf = tmp_buf;
        }
        dequant_gpu(e.weight.data, dq_buf, e.qtype, rows, cols, stream);

        Tensor fp16_view(reinterpret_cast<half*>(dq_buf), QType::F16, 2, e.weight.shape, true);
        NvFP4QuantResult result;
        quantize_fp16_to_nvfp4_async(fp16_view, result, d_absmax_buf2, d_tscale_buf2, stream);
        IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream));

        float h_tscale;
        IMP_CUDA_CHECK_LOG(
            cudaMemcpy(&h_tscale, d_tscale_buf2, sizeof(float), cudaMemcpyDeviceToHost));
        result.tensor_scale = h_tscale;
        wcache_->nvfp4[e.orig_ptr] = result;
        second_bytes += nvfp4_bytes;
        second_count++;

        if (tmp_buf)
            IMP_CUDA_CHECK_LOG(cudaFree(tmp_buf));
    }

    IMP_CUDA_CHECK_LOG(cudaFree(d_absmax_buf2));
    IMP_CUDA_CHECK_LOG(cudaFree(d_tscale_buf2));

    if (second_count > 0) {
        wcache_->nvfp4_bytes += second_bytes;
        IMP_LOG_INFO("NVFP4 second pass: %d additional tensors, %.2f MiB", second_count,
                     second_bytes / (1024.0 * 1024.0));
    }
}

}  // namespace imp
