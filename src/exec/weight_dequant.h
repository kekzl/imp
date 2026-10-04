#pragma once

// Whole-weight dequant to FP16 for the uncached GEMM fallbacks: GGUF block quants via
// dequant_gpu, native FP8 E4M3 (Modelopt mixed precision) with its per-tensor scale. An FP8
// weight the FP16 cache budget left out otherwise reached cuBLAS raw (status 15, #2563).

#include "core/tensor.h"
#include "model/model_config.h"
#include "quant/dequant_gpu.h"
#include "quant/fp8_quant.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <initializer_list>

namespace imp {

[[nodiscard]] inline bool weight_dequant_supported(QType qtype) {
    return qtype == QType::FP8_E4M3 || dequant_gpu_supported(qtype);
}

// Elements of the largest weight in `ws` the fallback may dequantize: the scratch size.
[[nodiscard]] inline size_t max_dequant_elems(std::initializer_list<const Tensor*> ws) {
    size_t n = 0;
    for (const Tensor* w : ws)
        if (w->data && weight_dequant_supported(w->qtype))
            n = std::max(n, static_cast<size_t>(w->numel()));
    return n;
}

// Scratch size in elements over the projections an uncached GEMM may dequantize.
[[nodiscard]] inline size_t max_dequant_elems(const TransformerLayer& L) {
    return max_dequant_elems({&L.wq, &L.wk, &L.wv, &L.wo, &L.w_gate, &L.w_up, &L.w_down, &L.w_gate_shared,
                              &L.w_up_shared, &L.w_down_shared, &L.ssm_in, &L.ssm_out, &L.gdn_gate});
}

// dst: FP16 [rows, cols], rows * cols elements.
inline void dequant_weight_fp16(const Tensor& w, void* dst, cudaStream_t stream) {
    const int rows = static_cast<int>(w.shape[0]);
    const int cols = static_cast<int>(w.shape[1]);
    if (w.qtype == QType::FP8_E4M3)
        dequantize_fp8_e4m3_to_fp16(w.data, dst, rows * cols, w.tensor_scale, stream);
    else
        dequant_gpu(w.data, dst, w.qtype, rows, cols, stream);
}

}  // namespace imp
