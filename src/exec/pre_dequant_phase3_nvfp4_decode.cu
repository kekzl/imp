// Pre-dequant Phase 3: NVFP4 decode-cache quantization.
// Multi-step quantization of decode-side weights to NVFP4, including
// candidate collection, two-pass mode-1/2 quantize, FP8 migration of
// failed candidates, CUTLASS conversion, MXFP4-source conversion, and
// MoE expert caching.
//
// Extracted from executor_pre_dequant.cu in Phase 3 of the architecture
// refactor roadmap. This is the bulk of the pre-dequant file's LOC —
// ~1325 LOC across the entry point and 10 helpers.
//
// See pre_dequant_internal.h for shared helpers.

#include "exec/executor.h"
#include "exec/pre_dequant_internal.h"
#include "compute/gemm_cutlass_sm120.h"
#include "compute/gemm_cutlass_mxfp4_sm120.h"
#include "quant/dequant_gpu.h"
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
#include <vector>

namespace imp {

using imp::pre_dequant_internal::deduct_budget;
using imp::pre_dequant_internal::for_each_dense_weight;
using imp::pre_dequant_internal::nvfp4_beneficial;

void GraphExecutor::nvfp4_decode_collect_candidates_(const ModelConfig& cfg,
                                                     Nvfp4DecodeContext& dctx) {
    // Dual-path mode: attention weights stay at FP8 for quality.
    if (wcache_.dual_path_quant) {
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
    {
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
    }

    auto collect_weight_nvfp4 = [&](const Tensor& w, QType qtype) {
        if (!w.data)
            return;
        if (!nvfp4_beneficial(qtype))
            return;
        if (wcache_.nvfp4.count(w.data))
            return;
        // Skip excluded weights (dual-path attention, GDN/SSM recurrent projections)
        if (dctx.exclude_ptrs.count(w.data))
            return;

        int cols = static_cast<int>(w.shape[1]);
        if (cols % 16 != 0)
            return;

        bool from_scratch = (wcache_.fp16.find(w.data) == wcache_.fp16.end());
        if (from_scratch && (!dequant_gpu_supported(qtype) || !qscratch_.dequant))
            return;
        dctx.entries.push_back({w.data, w, qtype, from_scratch});
    };

    // LM head first: largest single weight (vocab × d_model), biggest bandwidth win.
    collect_weight_nvfp4(model_->output_proj(), model_->out_proj_.qtype);

    // Dense attention + FFN: every tensor benefits every decode step.
    for_each_dense_weight(*model_, cfg, collect_weight_nvfp4);
}

void GraphExecutor::pre_dequant_phase3_nvfp4_decode_(
    const ModelConfig& cfg, const VRAMBudget& budget,
    size_t& remaining_budget, cudaStream_t stream) {
    if (wcache_.nvfp4_decode_mode <= 0)
        return;

    Nvfp4DecodeContext dctx;
    dctx.mode_str = (wcache_.nvfp4_decode_mode == 1) ? "additive" : "only";

    // Compute the shared mode-2 safety reserve once. Mode 1 keeps the upfront
    // 10% headroom (see vram_budget.cpp:50), so its budget arithmetic already
    // protects against shared/system-memory fallback; the in-loop safety is a
    // backstop only. Mode 2 omits the upfront 10% to fit larger weight caches
    // and previously paid for it with a 10% in-loop safety (3.2 GiB on a 32 GiB
    // 5090) that starved the dense NVFP4 cache. Replace with the same formula
    // the MoE expert path already uses: a KV-headroom estimate at 16 K tokens
    // plus a 256 MiB workspace cushion, clamped to [256 MiB, 1 GiB].
    if (wcache_.nvfp4_decode_mode == 2) {
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

    if (wcache_.nvfp4_decode_mode == 2 && !nvfp4_entries.empty()) {
        nvfp4_decode_quantize_mode2_(stream, dctx);
    } else if (!nvfp4_entries.empty()) {
        nvfp4_decode_quantize_mode1_(remaining_budget, stream, dctx);
    }

    if (wcache_.nvfp4_decode_mode == 2 && !wcache_.fp16.empty()) {
        nvfp4_decode_free_fp16_and_migrate_fp8_(remaining_budget, stream, dctx);
    }

    if (budget.nvfp4_second_pass && !nvfp4_entries.empty()) {
        nvfp4_decode_second_pass_(budget, stream, dctx);
    }

    if (!wcache_.nvfp4.empty() && cutlass_sm120_nvfp4_available()) {
        nvfp4_decode_convert_cutlass_(remaining_budget, stream);
    }

    nvfp4_decode_convert_mxfp4_and_native_(cfg, stream);

    if (qscratch_.mxfp4_act_sf != nullptr && cutlass_sm120_mxfp4_available()) {
        nvfp4_decode_mxfp4_fp16_fallback_(cfg, stream);
    }

    nvfp4_decode_cache_moe_experts_(cfg, remaining_budget, stream, dctx);
}

// Mode 2 ("only") incremental NVFP4 quantize. Process FP16-cached entries
// first (each conversion nets VRAM since NVFP4 ≈ 28% of FP16), then the
// from-scratch entries until VRAM is exhausted. Frees each source FP16
// entry immediately after the corresponding NVFP4 result is committed.
void GraphExecutor::nvfp4_decode_quantize_mode2_(cudaStream_t stream, Nvfp4DecodeContext& dctx) {
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
        IMP_CUDA_CHECK_LOG(cudaMemGetInfo(&free_mem, &total_mem));
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
            void* dq_buf = qscratch_.dequant;
            if (need > qscratch_.dequant_size) {
                if (cudaMalloc(&tmp_buf, need) != cudaSuccess || !tmp_buf)
                    continue;
                dq_buf = tmp_buf;
            }
            dequant_gpu(e.weight.data, dq_buf, e.qtype, rows, cols, stream);
            fp16_ptr = reinterpret_cast<const half*>(dq_buf);
        } else {
            auto it = wcache_.fp16.find(e.orig_ptr);
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
        wcache_.nvfp4[e.orig_ptr] = result;
        actual_bytes += nvfp4_bytes;
        actual_count++;

        if (tmp_buf)
            IMP_CUDA_CHECK_LOG(cudaFree(tmp_buf));

        // Free FP16 cache entry to reclaim VRAM for next weight
        if (!e.from_scratch) {
            auto it = wcache_.fp16.find(e.orig_ptr);
            if (it != wcache_.fp16.end()) {
                size_t freed = it->second.nbytes();
                vram_free(vram_alloc_, it->second.data);
                wcache_.fp16.erase(it);
                wcache_.fp16_bytes -= freed;
                actual_from_fp16++;
            }
        } else {
            actual_from_scratch++;
        }
    }

    IMP_CUDA_CHECK_LOG(cudaFree(d_absmax_buf));
    IMP_CUDA_CHECK_LOG(cudaFree(d_tscale_buf));

    wcache_.nvfp4_bytes = actual_bytes;
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
void GraphExecutor::nvfp4_decode_quantize_mode1_(size_t& remaining_budget, cudaStream_t stream,
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
            void* dq_buf = qscratch_.dequant;
            if (need > qscratch_.dequant_size) {
                void* tmp = nullptr;
                if (cudaMalloc(&tmp, need) != cudaSuccess || !tmp)
                    continue;
                dq_buf = tmp;
                tmp_bufs.push_back(tmp);
            }
            dequant_gpu(e.weight.data, dq_buf, e.qtype, rows, cols, stream);
            fp16_ptr = reinterpret_cast<const half*>(dq_buf);
        } else {
            auto it = wcache_.fp16.find(e.orig_ptr);
            fp16_ptr = reinterpret_cast<const half*>(it->second.data);
        }

        Tensor fp16_view(const_cast<half*>(fp16_ptr), QType::F16, 2, e.weight.shape, true);

        NvFP4QuantResult result;
        quantize_fp16_to_nvfp4_async(fp16_view, result, d_absmax_buf, d_tscales_all + i, stream);
        wcache_.nvfp4[e.orig_ptr] = result;
    }

    IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream));
    for (void* p : tmp_bufs)
        IMP_CUDA_CHECK_LOG(cudaFree(p));

    std::vector<float> h_tscales(budgeted.size());
    IMP_CUDA_CHECK_LOG(cudaMemcpy(h_tscales.data(), d_tscales_all, budgeted.size() * sizeof(float),
                                  cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < budgeted.size(); i++) {
        auto it = wcache_.nvfp4.find(budgeted[i].orig_ptr);
        if (it != wcache_.nvfp4.end()) {
            it->second.tensor_scale = h_tscales[i];
        }
    }

    IMP_CUDA_CHECK_LOG(cudaFree(d_absmax_buf));
    IMP_CUDA_CHECK_LOG(cudaFree(d_tscales_all));

    wcache_.nvfp4_bytes = budget_used;
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

// Mode 2 ("only") FP16-cache release with FP8 migration. Migrate every
// FP16 entry not already FP8-cached into a contiguous FP8 buffer
// (calibrate + per-tensor scale), then free the FP16 cache except entries
// that have no NVFP4/FP8 alternative (GDN ssm_in/ssm_out on hybrids).
// Also frees the fused KV / gate-up prefill caches.
void GraphExecutor::nvfp4_decode_free_fp16_and_migrate_fp8_(size_t& remaining_budget,
                                                            cudaStream_t stream,
                                                            Nvfp4DecodeContext& dctx) {
    (void)dctx;
    int migrated = 0;
    size_t migrated_bytes = 0;
    if (wcache_.use_fp8) {
        struct MigrateEntry {
            const void* orig_ptr;
            Tensor fp16_tensor;
            size_t n_elems;
        };
        std::vector<MigrateEntry> to_migrate;
        for (auto& [orig_ptr, fp16_tensor] : wcache_.fp16) {
            if (wcache_.fp8.count(orig_ptr))
                continue;
            size_t n = static_cast<size_t>(fp16_tensor.shape[0]) * fp16_tensor.shape[1];
            to_migrate.push_back({orig_ptr, fp16_tensor, n});
        }

        if (!to_migrate.empty()) {
            int max_grid = 0;
            size_t total_fp8_bytes = 0;
            for (auto& e : to_migrate) {
                int threads_needed = (static_cast<int>(e.n_elems) + 3) / 4;
                int grid = (threads_needed + 255) / 256;
                if (grid > max_grid)
                    max_grid = grid;
                total_fp8_bytes += e.n_elems;
            }

            float* d_block_maxes = nullptr;
            float* d_absmax = nullptr;
            IMP_CUDA_CHECK_LOG(cudaMalloc(&d_block_maxes, (size_t)max_grid * sizeof(float)));
            IMP_CUDA_CHECK_LOG(cudaMalloc(&d_absmax, sizeof(float)));

            float* d_scales_all = nullptr;
            IMP_CUDA_CHECK_LOG(cudaMalloc(&d_scales_all, to_migrate.size() * sizeof(float)));

            uint8_t* d_fp8_bulk = nullptr;
            d_fp8_bulk = static_cast<uint8_t*>(
                vram_alloc(vram_alloc_, total_fp8_bytes, "fp8_migration_cache"));
            if (!d_fp8_bulk) {
                cudaError_t e = cudaGetLastError();
                IMP_LOG_WARN("FP8 migration cache alloc failed (%.1f MiB): %s",
                             total_fp8_bytes / (1024.0 * 1024.0), cudaGetErrorString(e));
            }

            size_t fp8_offset = 0;
            for (size_t i = 0; i < to_migrate.size() && d_fp8_bulk; i++) {
                auto& e = to_migrate[i];
                void* fp8_buf = d_fp8_bulk + fp8_offset;
                fp8_offset += e.n_elems;

                calibrate_and_quantize_fp8_async(e.fp16_tensor.data, fp8_buf,
                                                 static_cast<int>(e.n_elems), d_block_maxes, max_grid,
                                                 d_absmax, d_scales_all + i, stream);

                Tensor fp8_t(fp8_buf, QType::FP8_E4M3, e.fp16_tensor.ndim, e.fp16_tensor.shape, true);
                wcache_.fp8[e.orig_ptr] = {fp8_t, 0.0f, d_scales_all + static_cast<ptrdiff_t>(i)};
                migrated++;
                migrated_bytes += e.n_elems + sizeof(float);
            }

            wcache_.fp8_migrated_data = d_fp8_bulk;
            wcache_.fp8_migrated_data_size = total_fp8_bytes;

            if (migrated > 0) {
                IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream));
                std::vector<float> h_scales(migrated);
                IMP_CUDA_CHECK_LOG(cudaMemcpy(h_scales.data(), d_scales_all, migrated * sizeof(float),
                                              cudaMemcpyDeviceToHost));
                int idx = 0;
                for (size_t i = 0; i < to_migrate.size() && idx < migrated; i++, idx++) {
                    auto it = wcache_.fp8.find(to_migrate[i].orig_ptr);
                    if (it != wcache_.fp8.end()) {
                        it->second.host_scale = h_scales[idx];
                    }
                }
            }

            IMP_CUDA_CHECK_LOG(cudaFree(d_block_maxes));
            IMP_CUDA_CHECK_LOG(cudaFree(d_absmax));
            wcache_.fp8_migrated_scales = d_scales_all;
            wcache_.fp8_migrated_count = migrated;
        }
    }

    // Free remaining FP16 cache — but KEEP entries that have no NVFP4
    // or FP8 alternative (e.g. GDN `ssm_in`/`ssm_out` on hybrid models
    // like Qwen 3.5/3.6). Without this, run_gdn falls back to on-the-fly
    // dequant which produces ~5% per-element drift at L0 and cascades
    // to sign-flips at the shared MLP → garbage output.
    size_t freed = 0;
    size_t kept_bytes = 0;
    int kept_count = 0;
    std::vector<const void*> to_erase;
    for (auto& [ptr, tensor] : wcache_.fp16) {
        const bool has_nvfp4 = (wcache_.nvfp4.find(ptr) != wcache_.nvfp4.end());
        const bool has_fp8 = (wcache_.fp8.find(ptr) != wcache_.fp8.end());
        if (has_nvfp4 || has_fp8) {
            vram_free(vram_alloc_, tensor.data);
            freed += static_cast<size_t>(tensor.shape[0]) * tensor.shape[1] * sizeof(half);
            to_erase.push_back(ptr);
        } else {
            kept_bytes += static_cast<size_t>(tensor.shape[0]) * tensor.shape[1] * sizeof(half);
            kept_count++;
        }
    }
    for (auto p : to_erase)
        wcache_.fp16.erase(p);
    wcache_.fp16_bytes = kept_bytes;
    if (kept_count > 0) {
        IMP_LOG_INFO(
            "NVFP4 only mode: preserved %d FP16 entries (%.2f MiB) "
            "with no NVFP4/FP8 alternative (GDN/hybrid weights)",
            kept_count, kept_bytes / (1024.0 * 1024.0));
    }

    // Free fused caches (prefill uses individual FP8 weights)
    for (auto& [idx, tensor] : wcache_.fused_kv) {
        if (tensor.data)
            vram_free(vram_alloc_, tensor.data);
    }
    wcache_.fused_kv.clear();
    for (auto& [idx, tensor] : wcache_.fused_gate_up) {
        if (tensor.data)
            vram_free(vram_alloc_, tensor.data);
    }
    wcache_.fused_gate_up.clear();

    remaining_budget += freed;
    wcache_.fp8_bytes += migrated_bytes;
    IMP_LOG_INFO(
        "NVFP4 only mode: freed FP16 cache (%.2f MiB), migrated %d weights to FP8 (%.2f MiB)",
        freed / (1024.0 * 1024.0), migrated, migrated_bytes / (1024.0 * 1024.0));
}

// NVFP4 second pass: after the FP16-free + FP8 migration phase frees VRAM,
// re-attempt NVFP4 quantization for entries skipped earlier due to budget
// pressure. Same per-tensor cudaMemGetInfo gate as mode 2.
void GraphExecutor::nvfp4_decode_second_pass_(const VRAMBudget& budget, cudaStream_t stream,
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
        if (wcache_.nvfp4.count(e.orig_ptr))
            continue;  // already cached
        int rows = static_cast<int>(e.weight.shape[0]);
        int cols = static_cast<int>(e.weight.shape[1]);
        size_t nvfp4_bytes = static_cast<size_t>(rows) * cols / 2 +
                             static_cast<size_t>(rows) * cols / 16 + 4;

        size_t free_mem2 = 0, total_mem2 = 0;
        IMP_CUDA_CHECK_LOG(cudaMemGetInfo(&free_mem2, &total_mem2));
        size_t nvfp4_safety2 = std::max(total_mem2 / 10, static_cast<size_t>(1024 * 1024));
        if (free_mem2 < nvfp4_bytes + nvfp4_safety2)
            break;

        // Dequant from quantized weights via scratch buffer
        size_t need = static_cast<size_t>(rows) * cols * sizeof(half);
        void* dq_buf = qscratch_.dequant;
        void* tmp_buf = nullptr;
        if (!dequant_gpu_supported(e.qtype) || !qscratch_.dequant)
            continue;
        if (need > qscratch_.dequant_size) {
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
        wcache_.nvfp4[e.orig_ptr] = result;
        second_bytes += nvfp4_bytes;
        second_count++;

        if (tmp_buf)
            IMP_CUDA_CHECK_LOG(cudaFree(tmp_buf));
    }

    IMP_CUDA_CHECK_LOG(cudaFree(d_absmax_buf2));
    IMP_CUDA_CHECK_LOG(cudaFree(d_tscale_buf2));

    if (second_count > 0) {
        wcache_.nvfp4_bytes += second_bytes;
        IMP_LOG_INFO("NVFP4 second pass: %d additional tensors, %.2f MiB", second_count,
                     second_bytes / (1024.0 * 1024.0));
    }
}

// Phase 3b: convert NVFP4 weights into CUTLASS sm_120 block-scaled format.
// Must run after FP16-free; the CUTLASS cache approximately doubles NVFP4
// VRAM (repacked data + SfAtom scales).  Budget-aware: stops if VRAM
// budget runs out and emits an info line.
void GraphExecutor::nvfp4_decode_convert_cutlass_(size_t& remaining_budget, cudaStream_t stream) {
    // After incremental mode, remaining_budget is stale.  Use actual free VRAM.
    size_t ct_budget;
    if (wcache_.nvfp4_decode_mode == 2) {
        IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream));
        size_t free_mem = 0, total_mem = 0;
        IMP_CUDA_CHECK_LOG(cudaMemGetInfo(&free_mem, &total_mem));
        // Intentionally NOT using dctx.safety_reserve here: populating
        // cutlass_nvfp4 in mode 2 destabilised CUDA-graph capture on
        // Qwen3-14B Q6_K (bimodal 97 vs 145 tok/s decode across trials).
        // The dense in-loop safety relaxation already delivers the +15%
        // decode win; the CUTLASS path stays conservative until the
        // capture-failure root cause is understood.
        size_t kCtReserve = std::max(total_mem / 10, static_cast<size_t>(256ULL * 1024 * 1024));
        ct_budget = (free_mem > kCtReserve) ? (free_mem - kCtReserve) : 0;
    } else {
        ct_budget = (remaining_budget > wcache_.nvfp4_bytes)
                        ? (remaining_budget - wcache_.nvfp4_bytes)
                        : 0;
    }
    int ct_count = 0;
    size_t ct_total = 0;
    bool ct_exhausted = false;
    for (auto& [ptr, nvfp4] : wcache_.nvfp4) {
        if (ct_exhausted)
            break;
        // Estimate CUTLASS allocation (only scale factors — data is borrowed)
        size_t est = cutlass_nvfp4_sf_size(static_cast<int>(nvfp4.N), static_cast<int>(nvfp4.K));
        if (ct_total + est > ct_budget) {
            ct_exhausted = true;
            IMP_LOG_INFO(
                "CUTLASS NVFP4 cache: VRAM budget reached after %d tensors "
                "(%.1f / %.1f MiB)",
                ct_count, ct_total / (1024.0 * 1024.0), ct_budget / (1024.0 * 1024.0));
            break;
        }
        CutlassNvFP4Weight cw;
        convert_nvfp4_to_cutlass(nvfp4, cw, stream);
        if (cw.data) {
            wcache_.cutlass_nvfp4[ptr] = cw;
            ct_total += cw.sf_bytes;
            ct_count++;
        }
    }
    if (ct_count > 0) {
        IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream));
        wcache_.cutlass_nvfp4_bytes = ct_total;
        deduct_budget(remaining_budget, ct_total + wcache_.nvfp4_bytes);
        IMP_LOG_INFO("CUTLASS sm_120 NVFP4 weight cache: %d tensors, %.2f MiB", ct_count,
                     ct_total / (1024.0 * 1024.0));
    }
}

// Phase 3c-native: register MXFP4 GGUF weights directly in CUTLASS cache.
// Bypasses NVFP4 — the GGUF data is unpacked into E2M1 + SfAtom UE8M0 on
// GPU. Allocates the MXFP4 activation scratch once if any layer carries
// MXFP4 weights, then runs an optional NVFP4->MXFP4 conversion pass for
// models with `use_mxfp4`.
void GraphExecutor::nvfp4_decode_convert_mxfp4_and_native_(const ModelConfig& cfg, cudaStream_t stream) {
    // These bypass NVFP4 entirely — the GGUF data is unpacked into
    // separate E2M1 data + SfAtom UE8M0 scales on GPU.
    // For native MXFP4, allocate activation buffers if not already done.
    if (cutlass_sm120_mxfp4_available()) {
        // Check if any layer has MXFP4 weights
        bool has_mxfp4 = false;
        auto check_mxfp4 = [&](const Tensor&, QType qt) {
            if (qt == QType::MXFP4)
                has_mxfp4 = true;
        };
        for (int i = 0; i < cfg.n_layers && !has_mxfp4; i++) {
            const auto& L = model_->layer(i);
            check_mxfp4(L.wq, L.wq.qtype);
            check_mxfp4(L.wk, L.wk.qtype);
            check_mxfp4(L.w_gate, L.w_gate.qtype);
            check_mxfp4(L.ssm_in, L.ssm_in.qtype);
            check_mxfp4(L.ssm_out, L.ssm_out.qtype);
        }

        // Allocate MXFP4 scratch if needed and not already allocated
        if (has_mxfp4 && !qscratch_.mxfp4_act_sf) {
            int max_k = 0, max_n = 0;
            for (int i = 0; i < cfg.n_layers; i++) {
                const auto& L = model_->layer(i);
                if (L.wq.data && L.wq.ndim >= 2) {
                    max_n = std::max(max_n, (int)L.wq.shape[0]);
                    max_k = std::max(max_k, (int)L.wq.shape[1]);
                }
                if (L.w_gate.data && L.w_gate.ndim >= 2) {
                    max_n = std::max(max_n, (int)L.w_gate.shape[0]);
                    max_k = std::max(max_k, (int)L.w_gate.shape[1]);
                }
                if (L.w_down.data && L.w_down.ndim >= 2) {
                    max_n = std::max(max_n, (int)L.w_down.shape[0]);
                    max_k = std::max(max_k, (int)L.w_down.shape[1]);
                }
                if (L.ssm_in.data && L.ssm_in.ndim >= 2) {
                    max_n = std::max(max_n, (int)L.ssm_in.shape[0]);
                    max_k = std::max(max_k, (int)L.ssm_in.shape[1]);
                }
                if (L.ssm_out.data && L.ssm_out.ndim >= 2) {
                    max_n = std::max(max_n, (int)L.ssm_out.shape[0]);
                    max_k = std::max(max_k, (int)L.ssm_out.shape[1]);
                }
            }
            if (max_k > 0) {
                qscratch_.mxfp4_act_sf_size = cutlass_mxfp4_sf_size(max_tokens_, max_k);
                qscratch_.mxfp4_workspace_size = gemm_mxfp4_cutlass_sm120_workspace(max_tokens_,
                                                                                    max_n, max_k);
                qscratch_.mxfp4_act_sf = vram_alloc(vram_alloc_, qscratch_.mxfp4_act_sf_size,
                                                    "mxfp4_act_sf");
                qscratch_.mxfp4_workspace = (qscratch_.mxfp4_workspace_size > 0)
                                                ? vram_alloc(vram_alloc_,
                                                             qscratch_.mxfp4_workspace_size,
                                                             "mxfp4_workspace")
                                                : nullptr;
                // Also need CUTLASS activation data buffer
                if (!qscratch_.cutlass_act_data) {
                    qscratch_.cutlass_act_data_size = static_cast<size_t>(max_tokens_) * (max_k / 2);
                    qscratch_.cutlass_act_data = vram_alloc(vram_alloc_,
                                                            qscratch_.cutlass_act_data_size,
                                                            "cutlass_act_data");
                }
                IMP_LOG_INFO("Native MXFP4: allocated activation scratch (sf=%.2f MiB)",
                             qscratch_.mxfp4_act_sf_size / (1024.0 * 1024.0));
            }
        }
    }

    // Convert NVFP4 weights to MXFP4 (UE8M0 scales) if MXFP4 prefill is enabled.
    // Same packed FP4 data (borrowed), only allocates new scale factor buffers.
    // Note: Hadamard rotation requires MR-GPTQ pre-rotated weights (SafeTensors).
    // For GGUF models, we use direct scale conversion (no rotation).
    if (wcache_.use_mxfp4 && qscratch_.mxfp4_act_sf != nullptr && cutlass_sm120_mxfp4_available()) {
        int mx_count = 0;
        size_t mx_total = 0;
        for (auto& [ptr, nvfp4] : wcache_.nvfp4) {
            // Only convert weights where K is multiple of 32 (MXFP4 requirement)
            if (nvfp4.K % 32 != 0)
                continue;
            CutlassMxFP4Weight mw;
            convert_nvfp4_to_mxfp4_cutlass(nvfp4, mw, stream);
            if (mw.data) {
                wcache_.cutlass_mxfp4[ptr] = mw;
                mx_total += mw.sf_bytes;
                mx_count++;
            }
        }
        if (mx_count > 0) {
            IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream));
            wcache_.cutlass_mxfp4_bytes = mx_total;
            IMP_LOG_INFO("CUTLASS sm_120 MXFP4 weight cache: %d tensors, %.2f MiB", mx_count,
                         mx_total / (1024.0 * 1024.0));
        }
    }
}

// Native MXFP4 GGUF unpack + FP16 fallback dequant. Registers MXFP4 weights
// directly in the CUTLASS cache, then for GDN / forced-fallback models
// dequants them into a bulk FP16 buffer and rewrites model weight pointers
// so the dispatch path sees FP16 instead of raw MXFP4 blocks.
void GraphExecutor::nvfp4_decode_mxfp4_fp16_fallback_(const ModelConfig& cfg, cudaStream_t stream) {
int mx_native = 0;
size_t mx_native_bytes = 0;
auto register_if_mxfp4 = [&](const Tensor& w, QType qt, bool is_attn = true) {
    if (qt != QType::MXFP4 || !w.data || !w.on_device)
        return;
    if (w.ndim < 2 || w.shape[1] % 32 != 0)
        return;
    if (wcache_.cutlass_mxfp4.count(w.data))
        return;  // already registered
    CutlassMxFP4Weight mw;
    if (unpack_mxfp4_gguf(w.data, w.shape[0], w.shape[1], mw, stream)) {
        mw.hadamard_bs = is_attn ? cfg.mxfp4_hadamard_attn : cfg.mxfp4_hadamard_ffn;
        wcache_.cutlass_mxfp4[w.data] = mw;
        mx_native_bytes += mw.sf_bytes + static_cast<size_t>(w.shape[0]) * (w.shape[1] / 2);
        mx_native++;
    }
};
for (int i = 0; i < cfg.n_layers; i++) {
    const auto& L = model_->layer(i);
    register_if_mxfp4(L.wq, L.wq.qtype, true);
    register_if_mxfp4(L.wk, L.wk.qtype, true);
    register_if_mxfp4(L.wv, L.wv.qtype, true);
    register_if_mxfp4(L.wo, L.wo.qtype, true);
    register_if_mxfp4(L.w_up, L.w_up.qtype, false);
    register_if_mxfp4(L.w_gate, L.w_gate.qtype, false);
    register_if_mxfp4(L.w_down, L.w_down.qtype, false);
    // GDN-specific weights (Qwen3.5)
    register_if_mxfp4(L.ssm_in, L.ssm_in.qtype, true);
    register_if_mxfp4(L.ssm_out, L.ssm_out.qtype, true);
    register_if_mxfp4(L.gdn_gate, L.gdn_gate.qtype, true);
    register_if_mxfp4(L.gdn_alpha, L.gdn_alpha.qtype, true);
    register_if_mxfp4(L.gdn_beta, L.gdn_beta.qtype, true);
}
register_if_mxfp4(model_->output_proj(), model_->out_proj_.qtype);
if (mx_native > 0) {
    IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream));
    wcache_.cutlass_mxfp4_bytes += mx_native_bytes;
    wcache_.use_mxfp4 = true;
    IMP_LOG_INFO("Native MXFP4 GGUF: %d tensors, %.2f MiB (direct → CUTLASS)", mx_native,
                 mx_native_bytes / (1024.0 * 1024.0));

    // Sync and check for errors from unpack kernels
    IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream));
    {
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess)
            IMP_LOG_ERROR("MXFP4 unpack error: %s", cudaGetErrorString(e));
    }

    // Check if MXFP4 GEMV is available (linear_scales populated).
    // GDN models force the FP16 fallback because — although every
    // linear projection in executor_ssm_gdn.cu *does* now go through
    // gemm_dispatch — the MXFP4 prefill dispatch path for GDN-shape
    // weights (notably Qwen3.5-27B's ssm_out at K=6144 N=5120, and
    // FFN at K=17408 N=5120) hits a cuBLAS-INTERNAL_ERROR (status
    // 14) cascade we have not yet root-caused. Tracking in
    // qwen35_27b_mxfp4_ima_2026_04_25.md. Until that's resolved,
    // honor the historical fallback path.
    bool force_fallback = runtime_config().attention.mxfp4_fp16_fallback;
    bool has_gdn = (cfg.ssm_inner_size > 0);
    bool mxfp4_gemv_available = !force_fallback && !has_gdn;
    for (auto& [p, m] : wcache_.cutlass_mxfp4)
        if (!m.linear_scales) {
            mxfp4_gemv_available = false;
            break;
        }

    if (mxfp4_gemv_available) {
        IMP_LOG_INFO("MXFP4 GEMV: all %d weights have linear_scales, skipping FP16 fallback",
                     mx_native);
    }

    // Dequant MXFP4 → FP16 for decode (only when MXFP4 GEMV not available).
    // Single bulk allocation to avoid CUDA heap fragmentation.
    //
    // Phase A2 (`docs/plans/qwen35_27b_mxfp4_host_dequant_design_2026_05_17.md`):
    // when attention.mxfp4_fp16_cache_policy == "pruned", skip
    // tensor slots that aren't read on the dispatch hot path —
    // MoE expert_*_packed (consumed only by executor_forward_moe.cu's
    // pre-cached FP16 path, which the MXFP4 batch-dequant route
    // bypasses) and the LM head out_proj_ (routed through
    // generic-dequant). For Qwen3.5-27B MXFP4 this shrinks
    // the FP16 fallback from ~48 GiB to ~8-12 GiB and unblocks
    // load on 32 GiB VRAM.
    std::unordered_set<const void*> pruned_skip_ptrs;
    const bool pruned_policy =
        (runtime_config().attention.mxfp4_fp16_cache_policy == "pruned");
    if (pruned_policy) {
        for (int li = 0; li < cfg.n_layers; ++li) {
            const auto& L = model_->layer(li);
            if (L.expert_gate_packed.data)
                pruned_skip_ptrs.insert(L.expert_gate_packed.data);
            if (L.expert_up_packed.data)
                pruned_skip_ptrs.insert(L.expert_up_packed.data);
            if (L.expert_down_packed.data)
                pruned_skip_ptrs.insert(L.expert_down_packed.data);
        }
        if (model_->output_proj().data)
            pruned_skip_ptrs.insert(model_->output_proj().data);
    }
    size_t pruned_skipped_bytes = 0;
    int pruned_skipped_count = 0;

    size_t fp16_total = 0;
    if (!mxfp4_gemv_available) {
        for (auto& [p, m] : wcache_.cutlass_mxfp4) {
            if (wcache_.fp16.count(p))
                continue;
            size_t b = static_cast<size_t>(m.N) * m.K * sizeof(half);
            if (pruned_policy && pruned_skip_ptrs.count(p)) {
                pruned_skipped_bytes += b;
                pruned_skipped_count++;
                continue;
            }
            fp16_total += b;
        }
        if (pruned_policy && pruned_skipped_count > 0) {
            IMP_LOG_INFO(
                "MXFP4 FP16 cache pruning: skipping %d MoE/LM-head tensors "
                "(%.2f GiB saved)",
                pruned_skipped_count,
                pruned_skipped_bytes / (1024.0 * 1024.0 * 1024.0));
        }
    }

    void* d_fp16_bulk = nullptr;
    if (fp16_total > 0) {
        // Pre-flight VRAM check: WSL2/WDDM cudaMalloc happily pages over
        // the device boundary into host RAM. cuBLASLt then fails at
        // runtime when it can't allocate its internal workspace → status
        // 14 (INVALID_VALUE) followed by a confusing downstream illegal
        // memory access (observed on Qwen3.5-27B-mxfp4 GDN where the
        // 12 GiB MXFP4 raw + 48 GiB FP16 fallback exceed 32 GiB VRAM).
        // Refuse the alloc instead of paging — keeps the failure mode
        // legible.
        size_t free_mem = 0, total_mem = 0;
        cudaMemGetInfo(&free_mem, &total_mem);
        constexpr size_t kRuntimeHeadroom = static_cast<size_t>(2) * 1024 * 1024 * 1024;
        bool oversubscribe = (free_mem <= kRuntimeHeadroom ||
                              fp16_total + kRuntimeHeadroom > free_mem);
        // The "force anyway despite oversubscription" path is gone —
        // attention.mxfp4_fp16_fallback is a plain bool now. If the
        // user explicitly opts in via imp.conf the oversubscribe
        // check still gates them; this matches the previous
        // IMP_MXFP4_FP16_FALLBACK=1 semantics. The legacy
        // =force escape hatch is obsolete.
        bool allow_force = false;
        if (oversubscribe && !allow_force) {
            IMP_LOG_ERROR(
                "MXFP4 FP16 fallback would oversubscribe VRAM "
                "(need %.1f GiB + %.1f GiB runtime headroom, %.1f GiB free). "
                "Model is too large for this GPU with the FP16 decode "
                "fallback. Use a smaller quant or a smaller model. "
                "Set IMP_MXFP4_FP16_FALLBACK=force to attempt anyway "
                "(may IMA at first decode forward).",
                fp16_total / (1024.0 * 1024.0 * 1024.0),
                kRuntimeHeadroom / (1024.0 * 1024.0 * 1024.0),
                free_mem / (1024.0 * 1024.0 * 1024.0));
            // Skip the alloc — wcache_.fp16 stays empty for these weights.
            // Downstream code will detect the missing entries and bail
            // with its own diagnostic instead of silently corrupting state.
            fp16_total = 0;
        } else if (oversubscribe) {
            IMP_LOG_WARN(
                "MXFP4 FP16 fallback: forcing oversubscribed alloc "
                "(IMP_MXFP4_FP16_FALLBACK=force, %.1f GiB > %.1f GiB free)",
                fp16_total / (1024.0 * 1024.0 * 1024.0), free_mem / (1024.0 * 1024.0 * 1024.0));
        }
    }
    if (fp16_total > 0) {
        cudaError_t ae = cudaMalloc(&d_fp16_bulk, fp16_total);
        if (ae != cudaSuccess) {
            IMP_LOG_ERROR("MXFP4 FP16 bulk alloc failed: %s (%.1f MiB)", cudaGetErrorString(ae),
                          fp16_total / (1024.0 * 1024.0));
            d_fp16_bulk = nullptr;
        }
    }

    if (d_fp16_bulk) {
        size_t offset = 0;
        for (auto& [ptr, mw] : wcache_.cutlass_mxfp4) {
            if (wcache_.fp16.count(ptr))
                continue;
            // Honor the same pruning filter the fp16_total compute
            // pass used; otherwise the offset accounting drifts.
            if (pruned_policy && pruned_skip_ptrs.count(ptr))
                continue;
            size_t fp16_bytes = static_cast<size_t>(mw.N) * mw.K * sizeof(half);
            void* d_fp16 = static_cast<char*>(d_fp16_bulk) + offset;
            offset += fp16_bytes;

            // GPU-side dequant via dequant_mxfp4_to_fp16. The previous CPU
            // fallback indexed `(r*bpr+b)*17` + `blk[16]`, which assumed
            // GGUF interleaved 17-byte block layout — but weight_upload.cu
            // splits to [data(N*bpr*16) | scales(N*bpr)] before GPU upload,
            // so the CPU path read scale bytes from inside data and produced
            // garbage FP16. The GPU kernel below reads the split layout
            // correctly (data first, scales at offset N*K/2).
            dequant_mxfp4_to_fp16(ptr, mw.N, mw.K, d_fp16, stream);
            int64_t shape[2] = {mw.N, mw.K};
            wcache_.fp16[ptr] = Tensor(d_fp16, QType::F16, 2, shape, true);
        }
    }  // end if (d_fp16_bulk)

    if (fp16_total > 0) {
        IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream));
        {
            cudaError_t e = cudaGetLastError();
            if (e != cudaSuccess)
                IMP_LOG_ERROR("MXFP4 dequant kernel error: %s", cudaGetErrorString(e));
        }
        IMP_LOG_INFO("MXFP4 decode fallback: dequant → FP16 cache %.2f MiB",
                     fp16_total / (1024.0 * 1024.0));

        // Replace model weight tensor pointers with FP16 data.
        // This ensures ALL code paths (GEMV, direct gemm, etc.) see
        // valid FP16 data instead of raw MXFP4 blocks.
        auto replace_weight = [&](Tensor& w, QType& qt) {
            auto it = wcache_.fp16.find(w.data);
            if (it != wcache_.fp16.end() && qt == QType::MXFP4) {
                w = it->second;
                qt = QType::F16;
            }
        };
        for (int i = 0; i < cfg.n_layers; i++) {
            TransformerLayer& L = const_cast<Model*>(model_)->layer(i);
            replace_weight(L.wq, L.wq.qtype);
            replace_weight(L.wk, L.wk.qtype);
            replace_weight(L.wv, L.wv.qtype);
            replace_weight(L.wo, L.wo.qtype);
            replace_weight(L.w_up, L.w_up.qtype);
            replace_weight(L.w_gate, L.w_gate.qtype);
            replace_weight(L.w_down, L.w_down.qtype);
            // GDN-specific weights (Qwen3.5)
            replace_weight(L.ssm_in, L.ssm_in.qtype);
            replace_weight(L.ssm_out, L.ssm_out.qtype);
            replace_weight(L.gdn_gate, L.gdn_gate.qtype);
            replace_weight(L.gdn_alpha, L.gdn_alpha.qtype);
            replace_weight(L.gdn_beta, L.gdn_beta.qtype);
        }
        replace_weight(const_cast<Model*>(model_)->out_proj_,
                       const_cast<Model*>(model_)->out_proj_.qtype);
        IMP_LOG_INFO("MXFP4 → FP16: replaced %d weight tensor pointers",
                     (int)wcache_.fp16.size());
    }
}
}

// Cache MoE expert weights — done after FP16 free so mode 2 has full budget.
// Handles two sub-paths:
//  - cache_moe_native_nvfp4: NVFP4-prequant SafeTensors (per-expert tensors)
//    consolidated into one contiguous packed_data + scales buffer per layer
//    per projection.
//  - cache_moe_expert_nvfp4: GGUF / re-quant path, expert_*_packed is the
//    3-D contiguous tensor.
void GraphExecutor::nvfp4_decode_cache_moe_experts_(const ModelConfig& cfg,
                                                    size_t& remaining_budget,
                                                    cudaStream_t stream,
                                                    Nvfp4DecodeContext& dctx) {
    (void)remaining_budget;
    size_t moe_budget;
    // Cache MoE expert weights — done after FP16 free so mode 2 has full budget
    if (wcache_.nvfp4_decode_mode == 2) {
        size_t free_mem = 0, total_mem = 0;
        IMP_CUDA_CHECK_LOG(cudaMemGetInfo(&free_mem, &total_mem));
        // Reserve VRAM so the KV cache (sized after this in init_kv_cache)
        // can fit `min_kv_tokens` (default 16K) + workspaces. Computed from
        // the model's actual attention layout — the previous 1 GiB constant
        // was over-cautious for hybrid models (Nemotron-H: 6/52 attn layers,
        // <100 MiB KV at 16K) where it starved the NVFP4 MoE cache and
        // forced decode through the legacy D2H-sync fallback.
        //
        // Capped at 1 GiB (the previous static value) so this can only
        // RELEASE budget, never tighten it vs the previous behavior. Floor
        // at 256 MiB to keep workspace + scratch room.
        //
        // IMP_MOE_RESERVE_MIB still overrides for manual tuning (range
        // 128-4096 MiB).
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
        // 16K tokens × n_attn × 2 (K+V) × kv_heads × hd × FP16
        constexpr int kKvFloorTokens = 16384;
        size_t per_token_kv = static_cast<size_t>(n_attn_layers) * 2 *
                              static_cast<size_t>(kv_heads) * static_cast<size_t>(hd) * 2;
        size_t kv_reserve = static_cast<size_t>(kKvFloorTokens) * per_token_kv;
        constexpr size_t kWorkspaceSafety = 256ULL * 1024 * 1024;
        constexpr size_t kReserveCap = 1024ULL * 1024 * 1024;
        constexpr size_t kReserveFloor = 256ULL * 1024 * 1024;
        size_t kMoeReserve = std::clamp(kv_reserve + kWorkspaceSafety, kReserveFloor, kReserveCap);
        {
            const int v = runtime_config().moe.reserve_mib;
            if (v >= 128 && v <= 4096)
                kMoeReserve = static_cast<size_t>(v) * 1024ULL * 1024ULL;
        }
        IMP_LOG_DEBUG("MoE reserve: %.0f MiB (n_attn=%d, kv_heads=%d, hd=%d → %.0f MiB KV at 16K + 256 MiB workspace)",
                      kMoeReserve / (1024.0 * 1024.0), n_attn_layers, kv_heads, hd,
                      kv_reserve / (1024.0 * 1024.0));
        moe_budget = (free_mem > kMoeReserve) ? (free_mem - kMoeReserve) : 0;
    } else {
        moe_budget = (remaining_budget > wcache_.nvfp4_bytes) ? (remaining_budget - wcache_.nvfp4_bytes)
                                                              : 0;
    }
    bool moe_budget_exhausted = false;
    // Self-tracked logical budget for cache_moe_native_nvfp4 (NVFP4 prequant
    // SafeTensors). cudaMemGetInfo doesn't reflect the per-expert cudaFree's
    // promptly on this driver, so we track allocations and frees logically.
    // Initial value is moe_budget plus the per-expert weights that the
    // function will swap out — those sum to the cached size, so net per
    // call is zero and all 40 layers fit if the initial budget covers one
    // layer's worth of overhead.
    size_t moe_logical_avail = moe_budget;

    auto cache_moe_expert_nvfp4 = [&](const Tensor& packed, QType qtype) {
        if (!packed.data)
            return;
        if (!nvfp4_beneficial(qtype))
            return;
        if (wcache_.nvfp4_moe.count(packed.data))
            return;
        if (moe_budget_exhausted)
            return;
        if (!packed.on_device)
            return;
        if (packed.ndim < 3)
            return;

        int ne = static_cast<int>(packed.shape[0]);
        int rows = static_cast<int>(packed.shape[1]);
        int cols = static_cast<int>(packed.shape[2]);
        if (cols % 16 != 0)
            return;
        if (!dequant_gpu_supported(qtype) || !qscratch_.dequant)
            return;

        size_t nvfp4_bytes = static_cast<size_t>(ne) * rows * cols / 2 +
                             static_cast<size_t>(ne) * rows * cols / 16 +
                             static_cast<size_t>(ne) * sizeof(float);

        if (dctx.nvfp4_moe_total + nvfp4_bytes > moe_budget) {
            moe_budget_exhausted = true;
            IMP_LOG_INFO(
                "NVFP4 MoE cache: VRAM budget reached after %d MoE tensors "
                "(%.1f / %.1f MiB)",
                dctx.nvfp4_moe_count, dctx.nvfp4_moe_total / (1024.0 * 1024.0), moe_budget / (1024.0 * 1024.0));
            return;
        }

        NvFP4MoEQuantResult result;
        quantize_packed_experts_to_nvfp4(packed.data, qtype, ne, rows, cols, qscratch_.dequant, result,
                                         stream);

        wcache_.nvfp4_moe[packed.data] = result;
        dctx.nvfp4_moe_total += nvfp4_bytes;
        dctx.nvfp4_moe_count++;
    };

    // NVFP4-prequant SafeTensors path: experts arrive as per-expert tensors
    // (expert_w_gate[e] / expert_w_up[e] / expert_w_down[e]) with NVFP4
    // qtype + .scales / .tensor_scale sidecars promoted in Phase 0. The 3D
    // expert_*_packed tensors are NULL (the loader only stamps them for
    // GGUF and Gemma-4). Without this branch, cache_moe_expert_nvfp4 would
    // early-return at `!packed.data` and the legacy FP16 dequant + cuBLAS
    // sm_80 WMMA fallback fires per layer per token, killing CUDA Graphs.
    //
    // We allocate one contiguous packed_data + micro_scales + tensor_scales
    // buffer per layer per projection, copy the per-expert pointers in,
    // and stamp `packed.data` so wcache lookups (line below the layer loop)
    // and the consumer dispatch in executor_forward_moe.cu (lookup via
    // expert_*_packed.data) wire up automatically. After a successful copy
    // for a layer the per-expert allocations are freed inline — at 35B-A3B
    // the duplicate (per-expert + contiguous) would peak at ~30 GiB which
    // doesn't fit in 32 GiB, and the legacy fallback can't fire for layers
    // where nvfp4_moe_*_ptr is non-null anyway.
    auto cache_moe_native_nvfp4 = [&](Tensor& packed, std::vector<Tensor>& experts) -> bool {
        if (experts.empty() || !experts[0].data)
            return false;
        if (experts[0].qtype != QType::NVFP4 || experts[0].scales == nullptr)
            return false;
        if (packed.data && wcache_.nvfp4_moe.count(packed.data))
            return false;
        if (moe_budget_exhausted)
            return false;

        int ne = static_cast<int>(experts.size());
        // SafeTensors NVFP4 prequant: per-expert weight tensor on-disk
        // dtype is U8 (loader → INT8 → Phase-0 promote → NVFP4) and shape
        // is [N, K_packed] where K_packed = K_logical/2 (two FP4 nibbles
        // per byte). The same packed-shape convention is what the
        // existing executor_attention.cu / executor_ffn.cu NVFP4 dispatch
        // expects when computing `tmp.K = hw->shape[1] * 2`. Match that.
        int64_t N = experts[0].shape[0];
        int64_t K_packed = experts[0].shape[1];
        int64_t K = K_packed * 2;  // logical inner dim
        if (K % 16 != 0)
            return false;

        size_t expert_packed_bytes = static_cast<size_t>(N) * K_packed;
        size_t expert_ms_bytes = static_cast<size_t>(N) * (K / 16);
        size_t total_packed = static_cast<size_t>(ne) * expert_packed_bytes;
        size_t total_ms = static_cast<size_t>(ne) * expert_ms_bytes;
        size_t total_ts = static_cast<size_t>(ne) * sizeof(float);
        size_t add_bytes = total_packed + total_ms + total_ts;

        // Self-tracked logical budget. cudaMemGetInfo does NOT reflect
        // cudaFree's of upload-time per-expert weights in time on this
        // driver — after ~5 layers it reports free=0 even though the
        // heap has ~5 GiB freed but not yet reclaimed. The previous
        // per-call cudaMemGetInfo gate aborted at ~7 layers (21/120
        // entries) and left layers 7-39 on the legacy fallback path
        // with D2H expert_offsets sync, killing CUDA graph capture and
        // pinning decode at ~30 tok/s. Track the budget logically:
        // initialised once from cudaMemGetInfo, decremented on alloc,
        // incremented after per-expert frees below — net per-call
        // change is zero so all 40 layers fit.
        if (add_bytes > moe_logical_avail) {
            moe_budget_exhausted = true;
            IMP_LOG_INFO(
                "NVFP4 MoE native cache: logical budget reached after %d "
                "tensors (%.1f MiB cached, %.1f MiB logical avail, need %.1f MiB)",
                dctx.nvfp4_moe_count, dctx.nvfp4_moe_total / (1024.0 * 1024.0),
                moe_logical_avail / (1024.0 * 1024.0), add_bytes / (1024.0 * 1024.0));
            return false;
        }

        void* d_packed = vram_alloc_force(vram_alloc_, total_packed, "nvfp4_moe_packed_native");
        void* d_ms = vram_alloc_force(vram_alloc_, total_ms, "nvfp4_moe_ms_native");
        void* d_ts_raw = vram_alloc_force(vram_alloc_, total_ts, "nvfp4_moe_ts_native");
        if (!d_packed || !d_ms || !d_ts_raw) {
            if (d_packed)
                vram_free(vram_alloc_, d_packed);
            if (d_ms)
                vram_free(vram_alloc_, d_ms);
            if (d_ts_raw)
                vram_free(vram_alloc_, d_ts_raw);
            moe_budget_exhausted = true;
            IMP_LOG_WARN(
                "NVFP4 MoE native cache: cudaMalloc failed at %d "
                "tensors (%.1f MiB cached) — driver heap exhausted",
                dctx.nvfp4_moe_count, dctx.nvfp4_moe_total / (1024.0 * 1024.0));
            return false;
        }
        moe_logical_avail = (moe_logical_avail > add_bytes) ? (moe_logical_avail - add_bytes) : 0;
        float* d_ts = static_cast<float*>(d_ts_raw);

        std::vector<float> h_ts(ne);
        for (int e = 0; e < ne; ++e) {
            const auto& w = experts[e];
            if (w.shape[0] != N || w.shape[1] != K_packed || !w.data || !w.scales) {
                IMP_LOG_WARN(
                    "NVFP4 MoE native: expert %d shape/data mismatch, "
                    "rolling back layer",
                    e);
                vram_free(vram_alloc_, d_packed);
                vram_free(vram_alloc_, d_ms);
                vram_free(vram_alloc_, d_ts_raw);
                return false;
            }
            IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(static_cast<char*>(d_packed) +
                                                   static_cast<size_t>(e) * expert_packed_bytes,
                                               w.data, expert_packed_bytes, cudaMemcpyDeviceToDevice,
                                               stream));
            IMP_CUDA_CHECK_LOG(
                cudaMemcpyAsync(static_cast<char*>(d_ms) + static_cast<size_t>(e) * expert_ms_bytes,
                                w.scales, expert_ms_bytes, cudaMemcpyDeviceToDevice, stream));
            h_ts[e] = w.tensor_scale;
        }
        IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(d_ts, h_ts.data(), total_ts, cudaMemcpyHostToDevice, stream));

        NvFP4MoEQuantResult r;
        r.packed_data = d_packed;
        r.micro_scales = d_ms;
        r.tensor_scales = d_ts;
        r.n_experts = ne;
        r.N = N;
        r.K = K;
        r.expert_stride_packed = expert_packed_bytes;
        r.expert_stride_ms = expert_ms_bytes;

        // Stamp the packed Tensor so wcache_.nvfp4_moe key + consumer
        // wiring (expert_*_packed.data lookup) work uniformly with the
        // GGUF path. Logical K (NOT K/2) per cache_moe_expert_nvfp4
        // convention at shape[2].
        int64_t shape[3] = {static_cast<int64_t>(ne), N, K};
        packed = Tensor(d_packed, QType::NVFP4, 3, shape, /*on_device=*/true);

        wcache_.nvfp4_moe[d_packed] = r;
        dctx.nvfp4_moe_total += add_bytes;
        dctx.nvfp4_moe_count++;

        // Free per-expert GPU allocations now — the legacy fallback path
        // (executor_forward_moe.cu:expert_gemm + chunked_dequant_gemm) can
        // no longer fire for this layer because nvfp4_moe_*_ptr is non-null
        // after the cache populates and stamps `packed`. Without freeing,
        // we hold the same NVFP4 weights twice (per-expert + contiguous);
        // the duplicate exhausts VRAM around layer 33 of Qwen3.6-35B-A3B
        // and breaks layers 33-39's fast path. Per-layer free keeps total
        // overhead bounded — only the just-copied 384 expert pointers are
        // released, and only after the contiguous copy succeeded.
        //
        // Sync the stream so the in-flight D2D copies (which read from
        // experts[e].data / .scales) finish before we cudaFree the source.
        IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream));
        auto* mut_model = const_cast<Model*>(model_);
        size_t freed_bytes = 0;
        for (int e = 0; e < ne; ++e) {
            auto& w = experts[e];
            if (w.data) {
                mut_model->release_gpu_allocation(w.data);
                IMP_CUDA_CHECK_LOG(cudaFree(w.data));
                freed_bytes += expert_packed_bytes;
                w.data = nullptr;
                w.on_device = false;
            }
            if (w.scales) {
                mut_model->release_gpu_allocation(w.scales);
                IMP_CUDA_CHECK_LOG(cudaFree(w.scales));
                freed_bytes += expert_ms_bytes;
                w.scales = nullptr;
            }
        }
        moe_logical_avail += freed_bytes;

        // Re-stamp per-expert Tensors to slice into the contiguous packed +
        // micro-scale buffers and register CUTLASS_NVFP4 entries so the MoE
        // prefill fast path (executor_forward_moe.cu CUTLASS 3.x grouped
        // branch) can fire instead of dequant→FP16→cuBLAS. The cleanup loop
        // above nulled experts[e].data because the original per-expert
        // source allocs were freed; the executor needs valid slice pointers
        // for register_tensor() and the per-expert wcache_.cutlass_nvfp4
        // lookup. Without this block, expert_*_ids[e] = kInvalidTensorID
        // (because t.data == nullptr) and covers_ids() rejects the fast
        // path → 88% of prefill time is spent in dequantize_nvfp4_moe_kernel.
        if (cutlass_sm120_nvfp4_available()) {
            size_t sf_per_expert = cutlass_nvfp4_sf_size(static_cast<int>(N), static_cast<int>(K));
            size_t sfatom_total = static_cast<size_t>(ne) * sf_per_expert;
            void* d_sfatom = vram_alloc_force(vram_alloc_, sfatom_total, "nvfp4_moe_sfatom");
            if (d_sfatom) {
                convert_nvfp4_moe_scales_to_sfatom(d_ms, d_sfatom, ne, static_cast<int>(N),
                                                   static_cast<int>(K), stream);
                for (int e = 0; e < ne; ++e) {
                    auto& w = experts[e];
                    void* data_slice = static_cast<char*>(d_packed) +
                                       static_cast<size_t>(e) * expert_packed_bytes;
                    void* sf_slice = static_cast<char*>(d_sfatom) +
                                     static_cast<size_t>(e) * sf_per_expert;
                    w.data = data_slice;
                    w.scales = static_cast<char*>(d_ms) + static_cast<size_t>(e) * expert_ms_bytes;
                    w.on_device = true;
                    w.tensor_scale = h_ts[e];
                    CutlassNvFP4Weight cw;
                    cw.data = data_slice;
                    cw.scale_factors = sf_slice;
                    cw.tensor_scale = h_ts[e];
                    cw.N = N;
                    cw.K = K;
                    cw.sf_bytes = sf_per_expert;
                    cw.sf_borrowed = true;  // shared layer-projection SfAtom buffer
                    wcache_.cutlass_nvfp4[data_slice] = cw;
                }
                IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream));
                wcache_.cutlass_nvfp4_bytes += sfatom_total;
                moe_logical_avail = (moe_logical_avail > sfatom_total)
                                        ? (moe_logical_avail - sfatom_total)
                                        : 0;
            } else {
                IMP_LOG_WARN(
                    "MoE NVFP4 SfAtom alloc failed (%.1f MiB for %d experts) "
                    "— prefill stays on dequant→cuBLAS fallback",
                    sfatom_total / (1024.0 * 1024.0), ne);
            }
        }

        return true;
    };

    for (int i = 0; i < cfg.n_layers; i++) {
        // Need mutable access to expert_*_packed for cache_moe_native_nvfp4
        // to stamp the contiguous buffer pointer. const_cast follows the
        // existing pattern at e.g. lines 1517 / 1598 of weight_upload.cu.
        auto& L = const_cast<Model*>(model_)->layer(i);

        bool g = false, u = false, d = false;
        if (cfg.is_nvfp4_prequant) {
            g = cache_moe_native_nvfp4(L.expert_gate_packed, L.expert_w_gate);
            u = cache_moe_native_nvfp4(L.expert_up_packed, L.expert_w_up);
            d = cache_moe_native_nvfp4(L.expert_down_packed, L.expert_w_down);
            // Non-gated MoE (e.g. Nemotron-H NemotronHForCausalLM): no gate
            // projection exists, so g=0 is expected when up and down cached.
            // Suppress the misleading warning in that case; expert_gemm's
            // wcache_.nvfp4_moe lookup handles the missing-gate path.
            bool non_gated = (L.expert_gate_packed.data == nullptr &&
                              (L.expert_w_gate.empty() ||
                               L.expert_w_gate[0].data == nullptr));
            if ((g || u || d) && !(g && u && d) && !(non_gated && u && d)) {
                IMP_LOG_WARN(
                    "Layer %d: partial NVFP4 MoE native cache "
                    "(g=%d u=%d d=%d) — fast path may not engage",
                    i, (int)g, (int)u, (int)d);
            }
        }

        // GGUF / re-quant path: only run when native didn't populate.
        // For GGUF NVFP4-target models the source qtype is Q*_K/Q8_0 and
        // packed.data is non-null; for prequant SafeTensors all three
        // native calls succeeded above and these are no-ops because
        // packed.data now points into wcache_.nvfp4_moe.
        if (!g)
            cache_moe_expert_nvfp4(L.expert_gate_packed, L.expert_gate_packed.qtype);
        if (!u)
            cache_moe_expert_nvfp4(L.expert_up_packed, L.expert_up_packed.qtype);
        if (!d)
            cache_moe_expert_nvfp4(L.expert_down_packed, L.expert_down_packed.qtype);
    }

    if (dctx.nvfp4_moe_count > 0) {
        wcache_.nvfp4_moe_bytes = dctx.nvfp4_moe_total;
        IMP_LOG_INFO("NVFP4 MoE cache: %d tensors, %.2f MiB", dctx.nvfp4_moe_count,
                     dctx.nvfp4_moe_total / (1024.0 * 1024.0));
    } else if (wcache_.nvfp4.empty()) {
        IMP_LOG_INFO("NVFP4 decode: no eligible weights found (all ≤ 4.5 bits/elem)");
    }
}

}  // namespace imp
