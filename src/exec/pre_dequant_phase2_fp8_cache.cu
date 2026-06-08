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
                calibrate_and_quantize_fp8_async(qscratch_->dequant, fp8_buf, static_cast<int>(e.n_elems),
                                                 d_block_maxes, max_grid, d_absmax,
                                                 d_scales_all + static_cast<ptrdiff_t>(i), stream);

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

}  // namespace imp
