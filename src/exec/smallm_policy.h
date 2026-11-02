#pragma once

#include "quant/nvfp4_gemm.h"

namespace imp {

// Which small-M NVFP4 kernel serves a weight (or a sibling set of combined N_total, all of one K) on A4
// decode rows, by shape only: a row's bits never depend on M. impl 3: the weight-streaming kernel (ws) where
// v2 splits K 4+ ways (+ reduce launch), cannot fuse the siblings in one stripe, or refuses K % 256; v2
// elsewhere. Conc 16: Gemma-4 o / gate|up / down on ws +5 %; Qwen3-8B o / down (3 stripes) on ws -6 %.
enum class SmallMKernel { kNone, kV1, kV2, kWs };

inline constexpr int kSmallMWsMinStripes = 4;

inline SmallMKernel smallm_kernel_for(int impl, int N_total, int K, bool all_n64, bool all_ws,
                                      bool siblings) {
    const bool v2_ok = (K % 256) == 0 && all_n64;
    if (impl == 3) {
        const int stripes = v2_ok ? gemm_nvfp4_smallm_v2_stripes(N_total, K) : 0;
        if (v2_ok && (stripes == 1 || (!siblings && stripes < kSmallMWsMinStripes)))
            return SmallMKernel::kV2;
        if (all_ws)
            return SmallMKernel::kWs;
        if (v2_ok)
            return SmallMKernel::kV2;
    } else if (impl == 2 && v2_ok) {
        return SmallMKernel::kV2;
    }
    return (K % 128) == 0 ? SmallMKernel::kV1 : SmallMKernel::kNone;
}

// Split-K workspace bytes the chosen kernel reads (ws and refusals: none).
inline size_t smallm_workspace_bytes(SmallMKernel k, int N, int K) {
    switch (k) {
        case SmallMKernel::kV2:
            return gemm_nvfp4_smallm_v2_workspace_bytes(N, K);
        case SmallMKernel::kV1:
            return gemm_nvfp4_smallm_workspace_bytes(N);
        default:
            return 0;
    }
}

// One small-M launch on already-quantized activations; beta 0 or 1. False: the kernel refused.
[[nodiscard]] inline bool smallm_launch_a4(SmallMKernel k, const NvFP4QuantResult& w,
                                           const NvFP4QuantResult& xq, half* y, int M, int N, int K, void* ws,
                                           float beta, cudaStream_t stream) {
    switch (k) {
        case SmallMKernel::kWs:
            return gemm_nvfp4_smallm_ws(w, xq, y, M, N, K, beta, stream);
        case SmallMKernel::kV2:
            return gemm_nvfp4_smallm_v2_a4(w, xq, y, M, N, K, ws, stream, /*accumulate=*/beta == 1.0f);
        case SmallMKernel::kV1:
            return gemm_nvfp4_smallm_a4(w, xq, y, M, N, K, ws, stream, /*accumulate=*/beta == 1.0f);
        default:
            return false;
    }
}

inline SmallMKernel smallm_kernel_for(int impl, int N, int K) {
    return smallm_kernel_for(impl, N, K, (N % 64) == 0, gemm_nvfp4_smallm_ws_supports(N, K),
                             /*siblings=*/false);
}

}  // namespace imp
