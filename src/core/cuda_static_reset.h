#pragma once

// Reset of lazily-created module-static CUDA resources.
//
// Called ONLY from imp_gpu_release() immediately BEFORE cudaDeviceReset(),
// while the CUDA context is still valid. Several translation units hold
// file/function-scope statics (cuBLAS/cuBLASLt handles, device scratch
// buffers) behind lazy `if (!ptr)` / capacity guards. After a
// cudaDeviceReset() those pointers dangle but the guards stay armed, so the
// next use after an in-process reload would touch freed device memory.
// reset_static_cuda_state() frees + nulls every such resource so its guard
// re-arms on the next use.
//
// This is NOT part of normal engine teardown (~Engine's gemm_cleanup() etc.
// stay untouched). All hooks are idempotent and safe to call when the
// corresponding module was never used (its statics are still null).

namespace imp {

// Aggregator: calls every per-module hook below, then clears any sticky
// CUDA error.
void reset_static_cuda_state();

// Per-module hooks, each defined in the translation unit that owns the
// statics it resets.
void gemm_reset_static_cuda_state();                        // compute/gemm.cu
void gemm_grouped_reset_static_cuda_state();                // compute/gemm_grouped.cu
void gemm_grouped_nvfp4_smallM_reset_static_cuda_state();   // compute/gemm_grouped_nvfp4_smallM.cu
void attention_cublas_reset_static_cuda_state();            // compute/attention_cublas.cu
void attention_mxfp4_prefill_reset_static_cuda_state();     // compute/attention_mxfp4_prefill.cu
void vision_encoder_reset_static_cuda_state();              // vision/vision_encoder.cu
void fmha_sm120_reset_static_cuda_state();                  // compute/attention_fmha_sm120.cu
void fmha_mxfp4_reset_static_cuda_state();                  // compute/attention_fmha_mxfp4_sm120.cu
void moe_batch_reset_static_cuda_state();                   // exec/executor_forward_moe_batch.cu
void nvfp4_gemm_reset_static_cuda_state();                  // quant/nvfp4_gemm.cu

}  // namespace imp
