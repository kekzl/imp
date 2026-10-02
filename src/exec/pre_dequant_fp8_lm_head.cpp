// FP8 E4M3 per-row LM head (gemm.nvfp4_lm_head=fp8, #2156): quantizes the head once at load
// with quantize_fp8_rows_async; every LM-head dispatch then runs gemv_fp8_rowscale_fp32.

#include "exec/executor.h"
#include "exec/quant_pipeline.h"
#include "exec/pre_dequant_internal.h"
#include "core/config/lm_head_mode.h"
#include "core/logging.h"
#include "core/qtype.h"
#include "quant/dequant_gpu.h"
#include "quant/fp8_quant.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cstdint>

namespace imp {

void QuantPipeline::fp8_lm_head_cache_(cudaStream_t stream) {
    const LmHeadMode mode = lm_head_mode(dispatch_policy().gemm.nvfp4_lm_head);
    const Tensor& lm = model_->output_proj();
    const QType q = lm.qtype;
    if (pre_dequant_internal::lm_head_checkpoint_fp8_rows(*model_)) {
        adopt_checkpoint_fp8_lm_head_(mode);
        return;
    }
    if (!lm_head_mode_fp8(mode, q)) {
        if (lm_head_auto_keeps_source(mode, q))
            IMP_LOG_INFO("FP8 LM head: auto keeps the %s head at checkpoint precision (#2224)",
                         qtype_name(q));
        return;
    }
    // F16 quantizes in place; GGUF sources dequant in row slabs through the shared scratch.
    const bool f16_src = q == QType::F16;
    if (!pre_dequant_internal::fp8_lm_head_eligible(*model_) || (!f16_src && qscratch_->dequant == nullptr)) {
        // auto falls back to the #982 NVFP4 rule (Phase 3); an explicit fp8 keeps the source path.
        const bool f16_compute = !model_->output_norm().data || model_->output_norm().qtype == QType::F16;
        if (mode == LmHeadMode::Fp8)
            IMP_LOG_WARN(
                "FP8 LM head: skipped (source %s, on_device=%d, F16 compute=%d, d_model %lld); "
                "the head keeps its source path",
                qtype_name(q), lm.on_device ? 1 : 0, f16_compute ? 1 : 0,
                static_cast<long long>(lm.ndim == 2 ? lm.shape[1] : 0));
        else
            IMP_LOG_INFO(
                "FP8 LM head: auto, head not eligible (source %s, F16 compute=%d); NVFP4 rule applies",
                qtype_name(q), f16_compute ? 1 : 0);
        return;
    }
    const int rows = static_cast<int>(lm.shape[0]);
    const int cols = static_cast<int>(lm.shape[1]);
    const size_t code_bytes = (static_cast<size_t>(rows) * cols + 255) & ~static_cast<size_t>(255);
    const size_t total = code_bytes + static_cast<size_t>(rows) * sizeof(float);
    // Past the headroom like the NVFP4 head (raw cudaMalloc): Phase 4b then frees the source head.
    auto* bulk = static_cast<uint8_t*>(vram_alloc_force(vram_alloc_, total, "fp8_lm_head"));
    if (!bulk) {
        IMP_LOG_WARN("FP8 LM head: alloc of %.1f MiB failed; the head keeps its source path",
                     total / (1024.0 * 1024.0));
        return;
    }
    auto* scales = reinterpret_cast<float*>(bulk + code_bytes);

    if (f16_src) {
        quantize_fp8_rows_async(lm.data, bulk, rows, cols, scales, stream);
    } else {
        const size_t row_src = qtype_row_bytes(q, cols);
        const int slab = static_cast<int>(
            std::min<size_t>(rows, qscratch_->dequant_size / (static_cast<size_t>(cols) * sizeof(half))));
        if (slab <= 0) {
            vram_free(vram_alloc_, bulk);
            IMP_LOG_WARN("FP8 LM head: dequant scratch below one row; the head keeps its source path");
            return;
        }
        // Serialized on `stream`: each slab's quantize finishes before the next dequant reuses the scratch.
        for (int r0 = 0; r0 < rows; r0 += slab) {
            const int n = std::min(slab, rows - r0);
            dequant_gpu(static_cast<const uint8_t*>(lm.data) + static_cast<size_t>(r0) * row_src,
                        qscratch_->dequant, q, n, cols, stream);
            quantize_fp8_rows_async(qscratch_->dequant, bulk + static_cast<size_t>(r0) * cols, n, cols,
                                    scales + r0, stream);
        }
    }
    IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream));

    const int64_t shape[2] = {rows, cols};
    FP8CacheEntry e{};
    e.weight = Tensor(bulk, QType::FP8_E4M3, 2, shape, true);
    e.d_row_scales = scales;
    wcache_->lm_head_fp8 = e;
    wcache_->lm_head_fp8_bulk = bulk;
    wcache_->lm_head_fp8_bytes = total;
    IMP_LOG_INFO(
        "FP8 LM head: [%d x %d] %s -> E4M3 per-row scales (%.1f MiB), all LM-head rows; "
        "source released after load if no path reads it",
        rows, cols, qtype_name(q), total / (1024.0 * 1024.0));
}

// A per-row FP8 checkpoint head (#2479) IS the served head: no allocation, no conversion, every
// gemm.nvfp4_lm_head mode. Phase 4b keeps it (the cache borrows the source bytes).
void QuantPipeline::adopt_checkpoint_fp8_lm_head_(LmHeadMode mode) {
    const Tensor& lm = model_->output_proj();
    const Tensor& sc = model_->output_proj_row_scales();
    FP8CacheEntry e{};
    e.weight = Tensor(lm.data, QType::FP8_E4M3, 2, lm.shape, true);
    e.d_row_scales = static_cast<float*>(sc.data);
    wcache_->lm_head_fp8 = e;
    wcache_->lm_head_fp8_bulk = nullptr;
    wcache_->lm_head_fp8_bytes = 0;
    const double mib = ((static_cast<double>(lm.shape[0]) * lm.shape[1]) + sc.nbytes()) / (1024.0 * 1024.0);
    IMP_LOG_INFO(
        "FP8 LM head: [%lld x %lld] checkpoint E4M3 per-row scales served directly, no load-time "
        "conversion (%.1f MiB)",
        static_cast<long long>(lm.shape[0]), static_cast<long long>(lm.shape[1]), mib);
    if (mode == LmHeadMode::Nvfp4)
        IMP_LOG_INFO("FP8 LM head: gemm.nvfp4_lm_head=%s needs a 16-bit head; this checkpoint ships FP8",
                     dispatch_policy().gemm.nvfp4_lm_head.c_str());
}

}  // namespace imp
