#pragma once

#include "core/tensor.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <string>

namespace imp {

// NVFP4 (FP4 E2M1) quantization with two-level scaling:
//   Level 1: FP8 E4M3 micro-scale per 16 values (micro-block)
//   Level 2: FP32 tensor-scale (global)
//
// Packed format: 2 FP4 values per byte (low nibble = even index, high = odd)
// Micro-scales: [N, K/16] as uint8_t (FP8 E4M3)
// Tensor-scale: single FP32

struct NvFP4QuantResult {
    void* packed_data = nullptr;   // [N, K/2] packed nibbles on device
    void* micro_scales = nullptr;  // [N, K/16] FP8 E4M3 micro-scales on device
    float tensor_scale = 1.0f;     // global tensor scale
    int64_t N = 0;
    int64_t K = 0;
    // True when packed_data/micro_scales were allocated by this result (the
    // quantize_* paths) and must be cudaFree'd on teardown. False when they
    // BORROW resident model weight storage (the data-borrow decode cache) —
    // cudaFree on a borrowed pointer fails with "invalid argument".
    bool owned = true;
};

// Quantize FP16 tensor to NVFP4 with two-level scaling.
// input: [N, K] FP16 on device. K must be multiple of 16.
void quantize_fp16_to_nvfp4(const Tensor& input, NvFP4QuantResult& result, cudaStream_t stream = nullptr);

// Calibrate optimal tensor scale and micro-scales.
// Returns tensor_scale = global_absmax / 6.0 (FP4 E2M1 max = 6.0)
// d_reusable_max: optional pre-allocated device buffer (1 float) to avoid per-call malloc
float calibrate_nvfp4_scales(const Tensor& input, cudaStream_t stream = nullptr,
                             float* d_reusable_max = nullptr);

// Fully async variant: no host sync. Caller provides reusable device buffers.
// tensor_scale is written to d_tensor_scale_buf and must be read back after sync.
void quantize_fp16_to_nvfp4_async(const Tensor& input, NvFP4QuantResult& result, float* d_absmax_buf,
                                  float* d_tensor_scale_buf, cudaStream_t stream = nullptr);

// Dequantize NVFP4 back to FP16 (for verification).
void dequantize_nvfp4_to_fp16(const NvFP4QuantResult& quant, void* output_fp16,
                              cudaStream_t stream = nullptr);

// Free NVFP4 result device memory.
void free_nvfp4_result(NvFP4QuantResult& result);

// ---------------------------------------------------------------------------
// NVFP4 MoE: per-expert quantization with independent tensor scales.
// Packed expert weights [n_experts, eff, K] are quantized expert-by-expert
// into contiguous NVFP4 buffers with one tensor_scale per expert.
// ---------------------------------------------------------------------------

struct NvFP4MoEQuantResult {
    void* packed_data = nullptr;     // [n_experts, eff, K/2] contiguous FP4
    void* micro_scales = nullptr;    // [n_experts, eff, K/16] FP8 E4M3
    float* tensor_scales = nullptr;  // [n_experts] on device, one FP32 per expert
    int n_experts = 0;
    int64_t N = 0;                    // eff (rows per expert)
    int64_t K = 0;                    // inner dim
    size_t expert_stride_packed = 0;  // N * K/2 bytes per expert
    size_t expert_stride_ms = 0;      // N * K/16 bytes per expert
    // When true, packed_data/micro_scales/tensor_scales are BORROWED (model
    // weights or VRAMAllocator sub-allocations), not cudaMalloc'd — so
    // free_nvfp4_moe_result must NOT cudaFree them (the owner reclaims them).
    bool borrowed = false;
};

// Quantize packed 3D expert weights (raw GGML format) to NVFP4.
// dequant_scratch: [eff, K] FP16 scratch buffer on device (reused per expert).
void quantize_packed_experts_to_nvfp4(const void* packed_ggml_data, QType qtype, int n_experts, int eff,
                                      int K, void* dequant_scratch, NvFP4MoEQuantResult& result,
                                      cudaStream_t stream);

// Dequantize all experts from NVFP4 MoE back to FP16.
// Output layout: [n_experts, N, K] contiguous FP16 (same as dequant_gpu output).
void dequantize_nvfp4_moe_to_fp16(const NvFP4MoEQuantResult& result, void* output_fp16,
                                  cudaStream_t stream = nullptr);

// Free NvFP4MoEQuantResult device memory.
void free_nvfp4_moe_result(NvFP4MoEQuantResult& result);

// ---------------------------------------------------------------------------
// Defensive promotion of weight_scale_2 into the GEMM-internal tensor_scale.
//
// Modelopt:        val = fp4 * micro_scale * weight_scale_2  (multiply)
// llm-compressor:  val = fp4 * micro_scale / weight_scale_2  (divide → store 1/x)
//
// A weight_scale_2 of 0, NaN, or ±Inf would produce non-finite GEMM output and
// contaminate the entire layer's hidden state. This helper folds in defensive
// zeroing for both formats:
//   - non-finite h_scale         → 0.0f
//   - llm-compressor h_scale=0   → 0.0f (1/0 would be Inf)
//   - llm-compressor 1/h_scale non-finite → 0.0f
//   - Modelopt h_scale=0         → 0.0f (intentionally null layer)
//
// `*was_zeroed` is set true when defensive zeroing fired (used for diagnostic
// counters at end of load).
//
// Pure host function — testable without CUDA. Defined in nvfp4_quant.cu.
float nvfp4_promote_weight_scale_2(float h_scale, bool is_llm_compressor, bool* was_zeroed);

// Valid wire dtype for NVFP4 weight_scale per compressed-tensors spec.
// Spec mandates float8_e4m3fn. NVFP4/MXFP4 cross-misroutes and corrupt
// checkpoints would arrive here with U8/I8 (UE8M0) or other dtypes — those
// must be rejected at promote time so the slow dequant→cuBLAS fallback runs
// instead of silently producing wrong output through the FP8 decoder.
bool nvfp4_validate_weight_scale_dtype(QType qt, std::string* err);

// Verify that weight_scale's inner dimension matches weight_packed's inner
// dimension at the spec's group_size=16. weight_packed is stored as [N, K/2]
// (packed halves) and weight_scale must be [N, K/16] = [N, weight_packed_K/8].
//
// Mismatch on outer dim or inner-dim ratio rejects promotion.
//   packed_inner_dim is `weight.shape[1]` from the SafeTensors loader (= K/2).
//   scale_inner_dim is `weight_scale.shape[1]`.
//   packed_outer_dim/scale_outer_dim are shape[0] of each.
bool nvfp4_validate_packed_scale_shapes(int64_t packed_outer_dim, int64_t packed_inner_dim,
                                        int64_t scale_outer_dim, int64_t scale_inner_dim,
                                        std::string* err);

}  // namespace imp
