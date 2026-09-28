#pragma once

#include "core/process_diag.h"

#include <cublas_api.h>

namespace imp {

// Algorithm for the cublasGemm*Ex calls. CUBLAS_GEMM_AUTOTUNE times the candidates on first use
// and caches the winner in the handle, so two processes can pick different algorithms and answer
// differently (#2168). Deterministic mode takes the fixed default.
inline cublasGemmAlgo_t cublas_gemm_algo() {
    return process_diag_deterministic_gemm() ? CUBLAS_GEMM_DEFAULT : CUBLAS_GEMM_AUTOTUNE;
}

}  // namespace imp
