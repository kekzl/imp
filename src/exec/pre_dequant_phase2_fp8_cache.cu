// Pre-dequant Phase 2: FP8 cache.
// Converts weights to FP8 device tensors for the fp8_prefill path,
// gated by attention.fp8_prefill / runtime FP8 state.
//
// Mixed precision: attention weights (WQ/WK/WV/WO) are cached in FP16
// instead of FP8 to avoid precision loss that compounds across layers
// and shifts argmax at large vocab sizes (NVFP4 degeneration root cause).
// FFN/SSM weights tolerate 8-bit and go to FP8 for +53% prefill speed.
//
// Extracted from executor_pre_dequant.cu in Phase 3 of the architecture
// refactor roadmap. See pre_dequant_internal.h for shared helpers.

#include "exec/executor.h"
#include "exec/quant_pipeline.h"
#include "exec/pre_dequant_internal.h"
#include "quant/dequant_gpu.h"
#include "quant/nvfp4_quant.h"
#include "quant/fp8_quant.h"
#include "core/logging.h"
#include "memory/vram_allocator.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <algorithm>
#include <vector>

using imp::pre_dequant_internal::deduct_budget;
using imp::pre_dequant_internal::for_each_dense_weight;

namespace imp {

void QuantPipeline::pre_dequant_phase2_fp8_cache_(
    const ModelConfig& cfg, const VRAMBudget& budget,
    size_t& remaining_budget, cudaStream_t stream) {
    size_t fp8_budget = std::min(remaining_budget, budget.fp8_cache_bytes);
    size_t phase2_fp16_bytes = 0;
    if (wcache_->use_fp8) {
        // --- FP16 weight cache for native NVFP4 ---
        // FP8 quantization error (~0.5%/layer) compounds over 36 layers and
        // shifts argmax in 152K vocab. vLLM avoids this by dequanting NVFP4→FP16
        // fully at load and using FP16 cuBLAS for everything. We do the same:
        // dequantize all NVFP4 dense weights to FP16 for prefill, keep original
        // NVFP4 data for decode GEMV (which is single-token and doesn't compound).
        int fp16_all_count = 0;
        size_t fp16_all_bytes = 0;
        {
            auto cache_weight_fp16 = [&](const Tensor& w) {
                if (!w.data || wcache_->fp16.count(w.data))
                    return;
                QType qtype = w.qtype;
                if (qtype != QType::NVFP4)
                    return;

                int rows = static_cast<int>(w.shape[0]);
                int cols = static_cast<int>(w.shape[1]);
                int64_t logical_K = cols * 2;
                size_t fp16_bytes = static_cast<size_t>(rows) * logical_K * sizeof(half);

                if (fp16_all_bytes + fp16_bytes > fp8_budget)
                    return;

                void* fp16_buf = vram_alloc(vram_alloc_, fp16_bytes, "fp16_nvfp4_cache");
                if (!fp16_buf)
                    return;

                if (w.scales) {
                    NvFP4QuantResult nv;
                    nv.packed_data = w.data;
                    nv.micro_scales = w.scales;
                    nv.tensor_scale = w.tensor_scale;
                    nv.N = rows;
                    nv.K = cols * 2;
                    dequantize_nvfp4_to_fp16(nv, fp16_buf, stream);
                } else {
                    return;
                }

                int64_t fp16_shape[4] = {static_cast<int64_t>(rows), logical_K, 0, 0};
                Tensor fp16_tensor(fp16_buf, QType::F16, 2, fp16_shape, true);
                wcache_->fp16[w.data] = fp16_tensor;
                fp16_all_count++;
                fp16_all_bytes += fp16_bytes;
            };

            for_each_dense_weight(*model_, cfg, [&](const Tensor& w, QType) {
                cache_weight_fp16(w);
            });
            if (fp16_all_count > 0) {
                IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream));
                IMP_LOG_INFO("NVFP4→FP16 weight cache: %d tensors, %.2f MiB "
                             "(FP16 prefill, NVFP4 GEMV decode)",
                             fp16_all_count, fp16_all_bytes / (1024.0 * 1024.0));
            }
        }
        phase2_fp16_bytes = fp16_all_bytes;
        // After FP16 caching, remaining budget for FP8 (non-NVFP4 weights only)
        fp8_budget = (fp8_budget > fp16_all_bytes) ? (fp8_budget - fp16_all_bytes) : 0;

        // --- FP8 cache for remaining weights (non-NVFP4 GGUF quants) ---
        size_t fp8_total = 0;
        int fp8_count = 0;
        bool fp8_exhausted = false;

        struct FP8OverflowEntry {
            const void* orig_ptr;
            Tensor weight;
            QType qtype;
            size_t n_elems;
        };
        std::vector<FP8OverflowEntry> fp8_entries;

        auto collect_weight_fp8 = [&](const Tensor& w, QType qtype) {
            if (!w.data)
                return;
            if (!dequant_gpu_supported(qtype) && qtype != QType::NVFP4)
                return;
            if (wcache_->fp16.count(w.data))
                return;
            if (wcache_->fp8.count(w.data))
                return;
            if (fp8_exhausted)
                return;

            int64_t logical_K = (qtype == QType::NVFP4) ? w.shape[1] * 2 : w.shape[1];
            size_t n_elems = static_cast<size_t>(w.shape[0]) * logical_K;
            size_t fp8_bytes = n_elems;

            if (fp8_total + fp8_bytes + sizeof(float) > fp8_budget) {
                fp8_exhausted = true;
                IMP_LOG_INFO(
                    "FP8 cache: budget reached after %d tensors (%.1f / %.1f MiB, "
                    "saving %.1f MiB for NVFP4 decode)",
                    fp8_count, fp8_total / (1024.0 * 1024.0), fp8_budget / (1024.0 * 1024.0),
                    (remaining_budget - fp8_budget) / (1024.0 * 1024.0));
                return;
            }

            fp8_entries.push_back({w.data, w, qtype, n_elems});
            fp8_total += fp8_bytes + sizeof(float);
            fp8_count++;
        };

        // Same priority order — attention first, then SSM/FFN
        for_each_dense_weight(*model_, cfg, collect_weight_fp8);

        if (!fp8_entries.empty() && qscratch_->dequant) {
            // Pre-allocate reusable calibration temp buffers
            int max_grid = 0;
            size_t total_fp8_bytes = 0;
            for (auto& e : fp8_entries) {
                int threads_needed = (static_cast<int>(e.n_elems) + 3) / 4;
                int grid = (threads_needed + 255) / 256;
                if (grid > max_grid)
                    max_grid = grid;
                total_fp8_bytes += e.n_elems;
            }

            float* d_block_maxes = nullptr;
            float* d_absmax = nullptr;
            float* d_scales_all = nullptr;
            IMP_CUDA_CHECK_LOG(cudaMalloc(&d_block_maxes, (size_t)max_grid * sizeof(float)));
            IMP_CUDA_CHECK_LOG(cudaMalloc(&d_absmax, sizeof(float)));
            IMP_CUDA_CHECK_LOG(cudaMalloc(&d_scales_all, fp8_entries.size() * sizeof(float)));

            // Bulk-allocate all FP8 data
            uint8_t* d_fp8_bulk = static_cast<uint8_t*>(
                vram_alloc(vram_alloc_, total_fp8_bytes, "fp8_weight_cache"));
            if (!d_fp8_bulk) {
                cudaError_t e = cudaGetLastError();
                IMP_LOG_WARN("FP8 weight cache bulk alloc failed (%.1f MiB): %s",
                             total_fp8_bytes / (1024.0 * 1024.0), cudaGetErrorString(e));
            }

            int actual_count = 0;
            size_t fp8_offset = 0;
            for (size_t i = 0; i < fp8_entries.size() && d_fp8_bulk; i++) {
                auto& e = fp8_entries[i];
                int rows = static_cast<int>(e.weight.shape[0]);
                int cols = static_cast<int>(e.weight.shape[1]);

                if (e.qtype == QType::NVFP4 && e.weight.scales) {
                    NvFP4QuantResult nv;
                    nv.packed_data = e.weight.data;
                    nv.micro_scales = e.weight.scales;
                    nv.tensor_scale = e.weight.tensor_scale;
                    nv.N = rows;
                    nv.K = cols * 2;
                    dequantize_nvfp4_to_fp16(nv, qscratch_->dequant, stream);
                } else {
                    dequant_gpu(e.weight.data, qscratch_->dequant, e.qtype, rows, cols, stream);
                }

                void* fp8_buf = d_fp8_bulk + fp8_offset;
                fp8_offset += e.n_elems;

                // Async calibrate + quantize (no host sync)
                calibrate_and_quantize_fp8_async(qscratch_->dequant, fp8_buf,
                                                 static_cast<int64_t>(e.n_elems), d_block_maxes, max_grid,
                                                 d_absmax, d_scales_all + static_cast<ptrdiff_t>(i), stream);

                int64_t fp8_shape[4] = {e.weight.shape[0],
                    (e.qtype == QType::NVFP4) ? e.weight.shape[1] * 2 : e.weight.shape[1],
                    e.weight.shape[2], e.weight.shape[3]};
                Tensor fp8_t(fp8_buf, QType::FP8_E4M3, e.weight.ndim, fp8_shape, true);
                wcache_->fp8[e.orig_ptr] = {fp8_t, 0.0f, d_scales_all + static_cast<ptrdiff_t>(i)};
                actual_count++;
            }

            if (actual_count > 0) {
                IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream));
                // Read back scales
                std::vector<float> h_scales(actual_count);
                IMP_CUDA_CHECK_LOG(cudaMemcpy(h_scales.data(), d_scales_all, actual_count * sizeof(float),
                                              cudaMemcpyDeviceToHost));
                for (int i = 0; i < actual_count; i++) {
                    auto it = wcache_->fp8.find(fp8_entries[i].orig_ptr);
                    if (it != wcache_->fp8.end()) {
                        it->second.host_scale = h_scales[i];
                    }
                }
            }

            IMP_CUDA_CHECK_LOG(cudaFree(d_block_maxes));
            IMP_CUDA_CHECK_LOG(cudaFree(d_absmax));
            // Track bulk buffers for cleanup
            wcache_->fp8_overflow_scales = d_scales_all;
            wcache_->fp8_overflow_count = actual_count;
            wcache_->fp8_overflow_data = d_fp8_bulk;
            wcache_->fp8_overflow_data_size = total_fp8_bytes;
            fp8_count = actual_count;
        }

        if (fp8_count > 0) {
            wcache_->fp8_bytes = fp8_total;
            size_t fp16_equivalent = 0;
            for (auto& [ptr, entry] : wcache_->fp8) {
                fp16_equivalent += entry.weight.numel() * sizeof(half);
            }
            IMP_LOG_INFO("FP8 weight cache: %d tensors, %.2f MiB (%.2f MiB saved vs FP16)", fp8_count,
                         fp8_total / (1024.0 * 1024.0), (fp16_equivalent - fp8_total) / (1024.0 * 1024.0));
        } else {
            IMP_LOG_INFO("FP8 prefill: no weights cached (budget=0 or no eligible weights)");
        }
    }

    deduct_budget(remaining_budget, wcache_->fp8_bytes + phase2_fp16_bytes);
}

// Phase 2b: FP8 E4M3 decode sidecar for the GDN/Mamba in/out projections
// (gemm.fp8_ssm_proj). On native-NVFP4 hybrids the producer recipe leaves
// ssm_in/ssm_out BF16 → they decode as FP16 GEMVs, the single largest
// decode slice (34.6% on Qwen3.6-35B, 2026-07-10 nsys). NVFP4 on these
// wide GDN shapes REGRESSES (see config.h note); FP8 halves the bytes with
// byte-aligned loads instead. The sidecar only serves the M=1 decode GEMV —
// prefill and verify chunks keep the full-precision source, and the
// recurrent-scan state stays FP16, so the quality exposure is one 8-bit
// weight read per token, not error accumulation in the state.
//
// GGUF hybrids (e.g. Qwen3.6-35B-A3B UD-Q4_K_M): the recurrent projections
// are excluded from the NVFP4 decode cache (quality lock, phase 3) and are
// in no other cache, so their handles were Undefined-tier → decode paid a
// full dequant→cuBLAS round-trip per token. UD quants keep exactly these
// tensors at Q8_0, so an FP8 copy costs the same bytes but runs the tuned
// rowscale GEMV instead. Quantized sources are dequanted into the shared
// scratch first; only ≥8-bit sources (Q8_0) qualify — for 4/5/6-bit
// sources FP8 would *increase* the decode bytes, and stacking FP8 rounding
// on a coarser lattice is exactly the recurrent-scan quality risk the
// phase-3 exclusion exists for.
void QuantPipeline::pre_dequant_phase2b_fp8_ssm_sidecar_(const ModelConfig& cfg,
                                                         cudaStream_t stream) {
    const bool ssm_on = runtime_config().gemm.fp8_ssm_proj;
    // gemm.fp8_attn_proj (#984): same decode-only per-row-scale sidecar for
    // the FULL-PRECISION attention projections. "auto" = gpt-oss only — its
    // BF16 dense q/k/v/o get no NVFP4 decode cache (nvfp4_beneficial is
    // GGUF-only) and decode as 2 B/elem FP16 GEMVs, 33.5% of the decode
    // window (docs/audit/roofline_gptoss_2026_07_13.md).
    const std::string& ap = runtime_config().gemm.fp8_attn_proj;
    const bool attn_on = ap == "on" || (ap == "auto" && model_->profile().is_gpt_oss);
    if (!ssm_on && !attn_on)
        return;

    struct Entry {
        const void* ptr;
        size_t n_elems;
        int rows;
        int cols;
        QType src_qtype;  // F16 = quantize in place; else dequant to scratch first
    };
    std::vector<Entry> entries;
    size_t total_bytes = 0;
    for (int i = 0; i < cfg.n_layers; i++) {
        const auto& L = model_->layer(i);
        // At decode the fused GDN input pack ([ssm_in | gate | alpha | beta]
        // row-concat) replaces the ssm_in dispatch entirely (run_gdn n==1
        // fused_input path), so the sidecar must target the pack where it
        // exists; ssm_in then only serves prefill and stays full-precision.
        // Without a pack (quantized GGUF sources never build one) decode runs
        // the 4-call path — ssm_in AND gdn_gate GEMVs every token, so both
        // are sidecar targets there.
        const Tensor* in_side = (ssm_on && L.gdn_input_packed.data) ? &L.gdn_input_packed
                                : ssm_on                            ? &L.ssm_in
                                                                    : nullptr;
        const Tensor* gate_side =
            (ssm_on && !L.gdn_input_packed.data) ? &L.gdn_gate : nullptr;
        const Tensor* ssm_out_side = ssm_on ? &L.ssm_out : nullptr;
        const Tensor* q_side = attn_on ? &L.wq : nullptr;
        const Tensor* k_side = attn_on ? &L.wk : nullptr;
        const Tensor* v_side = attn_on ? &L.wv : nullptr;
        const Tensor* o_side = attn_on ? &L.wo : nullptr;
        for (const Tensor* w :
             {in_side, gate_side, ssm_out_side, q_side, k_side, v_side, o_side}) {
            if (!w || !w->data || !w->on_device)
                continue;
            // F16 (native residents; BF16 checkpoints are converted at
            // upload) quantizes straight from the resident weight — the
            // calibration kernel reads __half. Q8_0 (GGUF) dequants into the
            // shared scratch first; see the byte/quality rationale above.
            const bool f16_src = w->qtype == QType::F16;
            const bool q8_src = w->qtype == QType::Q8_0 && qscratch_->dequant != nullptr;
            if (!f16_src && !q8_src)
                continue;
            if (wcache_->fp8.count(w->data) || wcache_->nvfp4.count(w->data) ||
                wcache_->cutlass_nvfp4.count(w->data))
                continue;
            size_t n = static_cast<size_t>(w->shape[0]) * w->shape[1];
            if (n == 0 || (w->shape[1] % 16) != 0)
                continue;
            if (q8_src && n * sizeof(half) > qscratch_->dequant_size)
                continue;
            entries.push_back({w->data, n, static_cast<int>(w->shape[0]),
                               static_cast<int>(w->shape[1]), w->qtype});
            total_bytes += n;
        }
    }
    if (entries.empty())
        return;

    uint8_t* d_bulk =
        static_cast<uint8_t*>(vram_alloc(vram_alloc_, total_bytes, "fp8_ssm_sidecar"));
    if (!d_bulk) {
        cudaError_t e = cudaGetLastError();
        IMP_LOG_WARN("fp8_ssm_proj: sidecar alloc failed (%.1f MiB): %s — decode keeps FP16",
                     total_bytes / (1024.0 * 1024.0), cudaGetErrorString(e));
        return;
    }

    // Per-ROW scales: one scale per output channel. A single per-tensor scale
    // measurably hurts PPL here (+4% on Qwen3.6-35B) because the fused GDN
    // input pack concatenates row blocks with very different magnitudes
    // (conv | gate | alpha | beta) — one amax wastes e4m3 range on most rows.
    size_t total_rows = 0;
    for (const auto& e : entries)
        total_rows += static_cast<size_t>(e.rows);
    float* d_row_scales = nullptr;
    IMP_CUDA_CHECK_LOG(cudaMalloc(&d_row_scales, total_rows * sizeof(float)));
    if (!d_row_scales) {
        vram_free(vram_alloc_, d_bulk);
        IMP_LOG_WARN("fp8_ssm_proj: row-scale alloc failed — decode keeps FP16");
        return;
    }

    size_t offset = 0, row_offset = 0;
    for (const auto& e : entries) {
        const void* f16_src = e.ptr;
        if (e.src_qtype != QType::F16) {
            // Serialized on `stream`, so the shared scratch is safe to reuse
            // across entries: each dequant completes before the next overwrites it.
            dequant_gpu(e.ptr, qscratch_->dequant, e.src_qtype, e.rows, e.cols, stream);
            f16_src = qscratch_->dequant;
        }
        quantize_fp8_rows_async(f16_src, d_bulk + offset, e.rows, e.cols,
                                d_row_scales + row_offset, stream);
        int64_t fp8_shape[4] = {e.rows, e.cols, 0, 0};
        Tensor fp8_t(d_bulk + offset, QType::FP8_E4M3, 2, fp8_shape, true);
        FP8CacheEntry entry{};
        entry.weight = fp8_t;
        entry.d_row_scales = d_row_scales + row_offset;
        wcache_->fp8[e.ptr] = entry;
        offset += e.n_elems;
        row_offset += static_cast<size_t>(e.rows);
    }
    IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream));

    wcache_->fp8_bytes += total_bytes;
    wcache_->fp8_ssm_sidecar_data = d_bulk;
    wcache_->fp8_ssm_sidecar_data_size = total_bytes;
    wcache_->fp8_ssm_sidecar_row_scales = d_row_scales;

    IMP_LOG_INFO("fp8 decode sidecar: %zu projections%s%s (%.1f MiB, %zu per-row scales; "
                 "full-precision source retained for prefill)",
                 entries.size(), ssm_on ? " [ssm]" : "", attn_on ? " [attn]" : "",
                 total_bytes / (1024.0 * 1024.0), total_rows);
}

}  // namespace imp
