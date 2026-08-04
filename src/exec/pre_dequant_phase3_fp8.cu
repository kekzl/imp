// Pre-dequant Phase 3 (FP8): mode-2 FP16-cache release with FP8 migration.
// Split out of pre_dequant_phase3_nvfp4_decode.cu to keep each .cu under the
// kernel file-size threshold. See pre_dequant_internal.h / quant_pipeline.h
// for shared declarations.

#include "core/dispatch_policy.h"
#include "exec/executor.h"
#include "exec/quant_pipeline.h"
#include "exec/pre_dequant_internal.h"
#include "quant/fp8_quant.h"
#include "core/logging.h"
#include "memory/vram_allocator.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <algorithm>
#include <cstdlib>
#include <vector>

namespace imp {

// Mode 2 ("only") FP16-cache release with FP8 migration. Migrate every
// FP16 entry not already FP8-cached into a contiguous FP8 buffer
// (calibrate + per-tensor scale), then free the FP16 cache except entries
// that have no NVFP4/FP8 alternative (GDN ssm_in/ssm_out on hybrids).
// Also frees the fused KV / gate-up prefill caches.
void QuantPipeline::nvfp4_decode_free_fp16_and_migrate_fp8_(size_t& remaining_budget,
                                                            cudaStream_t stream,
                                                            Nvfp4DecodeContext& dctx) {
    (void)dctx;
    int migrated = 0;
    size_t migrated_bytes = 0;
    if (wcache_->use_fp8) {
        struct MigrateEntry {
            const void* orig_ptr;
            Tensor fp16_tensor;
            size_t n_elems;
        };
        std::vector<MigrateEntry> to_migrate;
        for (auto& [orig_ptr, fp16_tensor] : wcache_->fp16) {
            if (wcache_->fp8.count(orig_ptr))
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
                                                 static_cast<int64_t>(e.n_elems), d_block_maxes, max_grid,
                                                 d_absmax, d_scales_all + i, stream);

                Tensor fp8_t(fp8_buf, QType::FP8_E4M3, e.fp16_tensor.ndim, e.fp16_tensor.shape, true);
                wcache_->fp8[e.orig_ptr] = {fp8_t, 0.0f, d_scales_all + static_cast<ptrdiff_t>(i)};
                migrated++;
                migrated_bytes += e.n_elems + sizeof(float);
            }

            wcache_->fp8_migrated_data = d_fp8_bulk;
            wcache_->fp8_migrated_data_size = total_fp8_bytes;

            if (migrated > 0) {
                IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream));
                std::vector<float> h_scales(migrated);
                IMP_CUDA_CHECK_LOG(cudaMemcpy(h_scales.data(), d_scales_all, migrated * sizeof(float),
                                              cudaMemcpyDeviceToHost));
                int idx = 0;
                for (size_t i = 0; i < to_migrate.size() && idx < migrated; i++, idx++) {
                    auto it = wcache_->fp8.find(to_migrate[i].orig_ptr);
                    if (it != wcache_->fp8.end()) {
                        it->second.host_scale = h_scales[idx];
                    }
                }
            }

            IMP_CUDA_CHECK_LOG(cudaFree(d_block_maxes));
            IMP_CUDA_CHECK_LOG(cudaFree(d_absmax));
            wcache_->fp8_migrated_scales = d_scales_all;
            wcache_->fp8_migrated_count = migrated;
        }
    }

    // Free remaining FP16 cache — but KEEP entries that have no NVFP4
    // or FP8 alternative (e.g. GDN `ssm_in`/`ssm_out` on hybrid models
    // like Qwen 3.5/3.6). Without this, run_gdn falls back to on-the-fly
    // dequant which produces ~5% per-element drift at L0 and cascades
    // to sign-flips at the shared MLP → garbage output.
    //
    // Also KEEP entries for native NVFP4 weights (CUTLASS_NVFP4 source):
    // the FP16 cache is the only correct prefill path — FP8 quantization
    // error compounds across 36 layers and shifts argmax at 152K vocab.
    size_t freed = 0;
    size_t kept_bytes = 0;
    int kept_count = 0;
    std::vector<const void*> to_erase;
    for (auto& [ptr, tensor] : wcache_->fp16) {
        const bool has_cutlass_nvfp4 = (wcache_->cutlass_nvfp4.find(ptr) != wcache_->cutlass_nvfp4.end());
        if (has_cutlass_nvfp4) {
            kept_bytes += static_cast<size_t>(tensor.shape[0]) * tensor.shape[1] * sizeof(half);
            kept_count++;
            continue;
        }
        const bool has_nvfp4 = (wcache_->nvfp4.find(ptr) != wcache_->nvfp4.end());
        const bool has_fp8 = (wcache_->fp8.find(ptr) != wcache_->fp8.end());
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
        wcache_->fp16.erase(p);
    wcache_->fp16_bytes = kept_bytes;
    if (kept_count > 0) {
        IMP_LOG_INFO(
            "NVFP4 only mode: preserved %d FP16 entries (%.2f MiB) "
            "with no NVFP4/FP8 alternative (GDN/hybrid weights)",
            kept_count, kept_bytes / (1024.0 * 1024.0));
    }

    // Free fused caches (prefill uses individual FP8 weights)
    for (auto& [idx, tensor] : wcache_->fused_kv) {
        if (tensor.data)
            vram_free(vram_alloc_, tensor.data);
    }
    wcache_->fused_kv.clear();
    for (auto& [idx, tensor] : wcache_->fused_gate_up) {
        if (tensor.data)
            vram_free(vram_alloc_, tensor.data);
    }
    wcache_->fused_gate_up.clear();

    remaining_budget += freed;
    wcache_->fp8_bytes += migrated_bytes;
    IMP_LOG_INFO(
        "NVFP4 only mode: freed FP16 cache (%.2f MiB), migrated %d weights to FP8 (%.2f MiB)",
        freed / (1024.0 * 1024.0), migrated, migrated_bytes / (1024.0 * 1024.0));
}

}  // namespace imp
