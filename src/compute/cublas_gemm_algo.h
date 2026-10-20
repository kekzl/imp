#pragma once

#include "core/process_diag.h"

#include <cublas_api.h>

namespace imp {

// Algorithm for the cublasGemm*Ex calls. CUBLAS_GEMM_AUTOTUNE times the candidates on first use
// and caches the winner in the handle, so two processes can pick different algorithms and answer
// differently (#2168). Only gemm.algo_timed autotunes; otherwise the fixed default.
inline cublasGemmAlgo_t cublas_gemm_algo() {
    const bool timed = process_diag_gemm_algo_timed() && !process_diag_deterministic_gemm();
    return timed ? CUBLAS_GEMM_AUTOTUNE : CUBLAS_GEMM_DEFAULT;
}

}  // namespace imp
