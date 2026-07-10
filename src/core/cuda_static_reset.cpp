#include "core/cuda_static_reset.h"

#include "compute/gemm_cutlass_grouped_3x.h"  // gemm_grouped_3x_nvfp4_cleanup()

#include <cuda_runtime.h>

namespace imp {

void reset_static_cuda_state() {
    gemm_reset_static_cuda_state();
    gemm_grouped_reset_static_cuda_state();
    gemm_grouped_nvfp4_smallM_reset_static_cuda_state();
    attention_cublas_reset_static_cuda_state();
    attention_mxfp4_prefill_reset_static_cuda_state();
    vision_encoder_reset_static_cuda_state();
    gemm_cutlass_sm120_reset_static_cuda_state();
    gemm_cutlass_mxfp4_reset_static_cuda_state();
    fmha_sm120_reset_static_cuda_state();
    fmha_mxfp4_reset_static_cuda_state();
    moe_batch_reset_static_cuda_state();
    nvfp4_gemm_reset_static_cuda_state();

    // Persistent CUTLASS 3x grouped-GEMM staging/workspace + gemm instance.
    gemm_grouped_3x_nvfp4_cleanup();

    // Best-effort teardown: clear any sticky error left by the frees above.
    (void)cudaGetLastError();
}

}  // namespace imp
