#pragma once

#include "core/tensor.h"
#include "model/model_config.h" // GGMLQuantType (Q6_K, Q4_0, etc.)
#include "compute/gemm.h"       // block_q8_1

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <unordered_map>

namespace imp {

// Forward declarations for legacy gemm_dispatch parameters
struct FP8CacheEntry;
struct NvFP4QuantResult;
struct CutlassNvFP4Weight;
struct CutlassMxFP4Weight;

// ---------------------------------------------------------------------------
// CUDA kernels used by the executor
// ---------------------------------------------------------------------------

__global__ __launch_bounds__(256) void broadcast_add_bias_fp16_kernel(half* __restrict__ out, const half* __restrict__ bias,
                                                int rows, int cols);

__global__ __launch_bounds__(256) void scale_fp16_kernel(half* __restrict__ data, half scale, int64_t n);

__global__ __launch_bounds__(256) void elementwise_add_fp16_kernel(half* __restrict__ a, const half* __restrict__ b, int64_t n);

__global__ __launch_bounds__(256) void elementwise_add_store_fp16_kernel(const half* __restrict__ a, const half* __restrict__ b,
                                                   half* __restrict__ out, int64_t n);

__global__ __launch_bounds__(256) void fp32_accum_add_fp16_kernel(float* __restrict__ accum, const half* __restrict__ branch, int64_t n);

__global__ __launch_bounds__(256) void fp32_to_fp16_rowscale_kernel(const float* __restrict__ in,
                                             half* __restrict__ out,
                                             int rows, int cols);

__global__ __launch_bounds__(512) void rmsnorm_fp32_accum_to_fp16_kernel(
        const half* __restrict__ input,
        const half* __restrict__ norm_w,
        float* __restrict__ fp32_accum,
        half* __restrict__ output,
        int d_model,
        float eps,
        float weight_offset);

__global__ __launch_bounds__(256) void fp16_to_fp32_kernel(const half* __restrict__ in, float* __restrict__ out, int64_t n);

__global__ __launch_bounds__(256) void elementwise_add_fp32_kernel(float* __restrict__ a, const float* __restrict__ b, int64_t n);

__global__ __launch_bounds__(256) void write_kv_cache_kernel(
    const half* __restrict__ data_in,
    const int* __restrict__ positions,
    const int* __restrict__ block_tables,
    half* __restrict__ cache_base,
    int block_stride,
    int row_elems,
    int block_size,
    int n_tokens,
    int max_blocks_per_seq,
    int n_sequences);

__global__ __launch_bounds__(256) void write_kv_cache_fused_kernel(
    const half* __restrict__ k_in,
    const half* __restrict__ v_in,
    const int* __restrict__ positions,
    const int* __restrict__ block_tables,
    half* __restrict__ k_cache_base,
    half* __restrict__ v_cache_base,
    int block_stride,
    int row_elems,
    int block_size,
    int n_tokens,
    int max_blocks_per_seq,
    int n_sequences);

__global__ __launch_bounds__(256) void write_kv_cache_fp8_kernel(
    const half* __restrict__ data_in,
    const int* __restrict__ positions,
    const int* __restrict__ block_tables,
    __nv_fp8_e4m3* __restrict__ cache_base,
    float inv_scale,
    int block_stride,
    int row_elems,
    int block_size,
    int n_tokens,
    int max_blocks_per_seq,
    int n_sequences);

__global__ __launch_bounds__(256) void write_kv_cache_fp8_fused_kernel(
    const half* __restrict__ k_in,
    const half* __restrict__ v_in,
    const int* __restrict__ positions,
    const int* __restrict__ block_tables,
    __nv_fp8_e4m3* __restrict__ k_cache_base,
    __nv_fp8_e4m3* __restrict__ v_cache_base,
    float inv_scale,
    int block_stride,
    int row_elems,
    int block_size,
    int n_tokens,
    int max_blocks_per_seq,
    int n_sequences);

__global__ __launch_bounds__(256) void write_kv_cache_int8_kernel(
    const half* __restrict__ k_in,
    const half* __restrict__ v_in,
    const int* __restrict__ positions,
    const int* __restrict__ block_tables,
    int8_t* __restrict__ k_cache_base,
    int8_t* __restrict__ v_cache_base,
    half* __restrict__ k_scale_base,
    half* __restrict__ v_scale_base,
    int block_stride,
    int scale_block_stride,
    int n_kv_heads,
    int head_dim,
    int block_size,
    int n_tokens,
    int max_blocks_per_seq,
    int n_sequences);

__global__ __launch_bounds__(256) void write_kv_cache_int4_kernel(
    const half* __restrict__ k_in,
    const half* __restrict__ v_in,
    const int* __restrict__ positions,
    const int* __restrict__ block_tables,
    uint8_t* __restrict__ k_cache_base,
    uint8_t* __restrict__ v_cache_base,
    half* __restrict__ k_scale_base,
    half* __restrict__ v_scale_base,
    int block_stride,
    int scale_block_stride,
    int n_kv_heads,
    int head_dim,
    int block_size,
    int n_tokens,
    int max_blocks_per_seq,
    int n_sequences);

__global__ __launch_bounds__(256) void write_kv_cache_turboquant_kernel(
    const half* __restrict__ k_in,
    const half* __restrict__ v_in,
    const int* __restrict__ positions,
    const int* __restrict__ block_tables,
    uint8_t* __restrict__ k_dir_cache_base,    // INT4 packed normalized directions
    uint8_t* __restrict__ v_cache_base,         // INT4 packed values
    half* __restrict__ k_norm_base,             // FP16 PolarQuant norms (in scale_pool K region)
    half* __restrict__ v_scale_base,            // FP16 per-head V scales (in scale_pool V region)
    uint8_t* __restrict__ k_sketch_base,        // QJL 1-bit sketches
    const uint8_t* __restrict__ qjl_matrix,     // [sketch_dim, head_dim/8] packed Rademacher signs
    int block_stride,                           // kKVBlockSize * n_kv_heads * head_dim / 2 (bytes)
    int scale_block_stride,                     // kKVBlockSize * n_kv_heads (half elems)
    int sketch_block_stride,                    // kKVBlockSize * n_kv_heads * sketch_dim / 8 (bytes)
    int n_kv_heads,
    int head_dim,
    int sketch_dim,
    int block_size,
    int n_tokens,
    int max_blocks_per_seq,
    int n_sequences);

// TurboQuant MXFP4 variant: PolarQuant FP4 E2M1 K directions + UE8M0 micro-scales + QJL.
// K path: normalize → per-32-element UE8M0 scale → FP4 E2M1 quantize → QJL sketch
// V path: standard INT4 per-head quantization (same as non-MXFP4 variant)
__global__ __launch_bounds__(256) void write_kv_cache_turboquant_mxfp4_kernel(
    const half* __restrict__ k_in,
    const half* __restrict__ v_in,
    const int* __restrict__ positions,
    const int* __restrict__ block_tables,
    uint8_t* __restrict__ k_dir_cache_base,    // FP4 E2M1 packed directions (same layout as INT4)
    uint8_t* __restrict__ v_cache_base,         // INT4 packed values
    half* __restrict__ k_norm_base,             // FP16 PolarQuant norms
    half* __restrict__ v_scale_base,            // FP16 per-head V scales
    uint8_t* __restrict__ k_sketch_base,        // QJL 1-bit sketches
    uint8_t* __restrict__ k_mscale_base,        // UE8M0 micro-scales [block, slot, head, head_dim/32]
    const uint8_t* __restrict__ qjl_matrix,     // [sketch_dim, head_dim/8] packed Rademacher signs
    int block_stride,                           // kKVBlockSize * n_kv_heads * head_dim / 2 (bytes)
    int scale_block_stride,                     // kKVBlockSize * n_kv_heads (half elems)
    int sketch_block_stride,                    // kKVBlockSize * n_kv_heads * sketch_dim / 8 (bytes)
    int mscale_block_stride,                    // kKVBlockSize * n_kv_heads * (head_dim / 32) (bytes)
    int n_kv_heads,
    int head_dim,
    int sketch_dim,
    int block_size,
    int n_tokens,
    int max_blocks_per_seq,
    int n_sequences);

// TurboQuant Lite: QJL sketch-only K + INT4 V write kernel.
// K path (blockIdx.y == 0): Compute L2 norm + QJL sketch (no INT4 direction quantization).
// V path (blockIdx.y == 1): Standard INT4 per-head quantization.
__global__ __launch_bounds__(256) void write_kv_cache_turboquant_lite_kernel(
    const half* __restrict__ k_in,
    const half* __restrict__ v_in,
    const int* __restrict__ positions,
    const int* __restrict__ block_tables,
    uint8_t* __restrict__ v_cache_base,         // INT4 packed values (V only, no K in pool)
    half* __restrict__ k_norm_base,             // FP16 norms (in scale_pool K region)
    half* __restrict__ v_scale_base,            // FP16 per-head V scales (in scale_pool V region)
    uint8_t* __restrict__ k_sketch_base,        // QJL 1-bit sketches (primary K representation)
    const uint8_t* __restrict__ qjl_matrix,     // [sketch_dim, head_dim/8] packed Rademacher signs
    int v_block_stride,                         // kKVBlockSize * n_kv_heads * head_dim / 2 (bytes)
    int scale_block_stride,                     // kKVBlockSize * n_kv_heads (half elems)
    int sketch_block_stride,                    // kKVBlockSize * n_kv_heads * sketch_dim / 8 (bytes)
    int n_kv_heads,
    int head_dim,
    int sketch_dim,
    int block_size,
    int n_tokens,
    int max_blocks_per_seq,
    int n_sequences);

__global__ __launch_bounds__(256) void write_kv_cache_rope_fused_kernel(
    const half* __restrict__ k_in,
    const half* __restrict__ v_in,
    const int* __restrict__ positions,
    const int* __restrict__ block_tables,
    half* __restrict__ k_cache_base,
    half* __restrict__ v_cache_base,
    int block_stride,
    int row_elems,
    int block_size,
    int n_tokens,
    int max_blocks_per_seq,
    int n_sequences,
    int n_kv_heads,
    int head_dim,
    float theta,
    float inv_scaling,
    int rope_pairs,
    bool neox,
    const float* __restrict__ longrope_inv_freqs);

__global__ __launch_bounds__(256) void rope_q_only_fp16_kernel(
    half* __restrict__ Q,
    const int* __restrict__ positions,
    int n_heads,
    int head_dim,
    float theta,
    float inv_scaling,
    int rope_pairs,
    bool neox,
    const float* __restrict__ longrope_inv_freqs);

__global__ __launch_bounds__(256) void add_fp16_bias_to_fp32_kernel(float* __restrict__ data,
                                              const half* __restrict__ bias,
                                              int n_tokens, int n_cols);

__global__ __launch_bounds__(256) void scale_fp32_kernel(float* __restrict__ data, float scale, int64_t n);

__global__ __launch_bounds__(256) void logit_softcap_fp32_kernel(float* __restrict__ data,
                                          float softcap, float inv_softcap,
                                          int64_t n);

__global__ __launch_bounds__(256) void fp32_to_fp16_kernel(const float* __restrict__ in,
                                    half* __restrict__ out,
                                    int64_t n);

// ---------------------------------------------------------------------------
// dp4a GEMV helpers (shared by executor_forward.cu and executor_kernels.cu)
// ---------------------------------------------------------------------------

// Returns true if the quant type supports dp4a (Q8_1-input) GEMV kernels.
inline bool is_dp4a_qtype(GGMLQuantType qt) {
    return qt == GGMLQuantType::Q6_K || qt == GGMLQuantType::Q8_0 ||
           qt == GGMLQuantType::Q4_0 || qt == GGMLQuantType::Q4_K ||
           qt == GGMLQuantType::Q5_K || qt == GGMLQuantType::Q2_K ||
           qt == GGMLQuantType::Q3_K;
}

// Dispatch dp4a GEMV by quant type: y = W @ q8_1 (FP16 output).
void dispatch_dp4a_gemv(GGMLQuantType qtype,
                        const void* W, const block_q8_1* q8_1, const float* d8,
                        half* y, int M, int K, cudaStream_t stream);

// ---------------------------------------------------------------------------
// Host-side helper functions
// ---------------------------------------------------------------------------

void elementwise_add(Tensor& a, const Tensor& b, cudaStream_t stream);

void elementwise_add_store(const Tensor& a, const Tensor& b, Tensor& out,
                           cudaStream_t stream);

void add_bias(Tensor& out, const Tensor& bias, cudaStream_t stream);

// Fused 3-way bias add: out_a += bias_a, out_b += bias_b, out_c += bias_c in one launch.
// Skips any output where bias.data == nullptr.
void add_bias_3way(Tensor& out_a, const Tensor& bias_a,
                   Tensor& out_b, const Tensor& bias_b,
                   Tensor& out_c, const Tensor& bias_c,
                   cudaStream_t stream);

// Fused residual add + RMSNorm: hidden += residual; output = rmsnorm(hidden, weight).
// Saves 1 kernel launch + 1 DRAM round-trip vs separate add + norm.
void residual_add_rmsnorm(Tensor& hidden, const Tensor& residual,
                          const Tensor& weight, Tensor& output,
                          float eps, cudaStream_t stream,
                          float weight_offset = 0.0f);

// Fused add-store + RMSNorm: hidden = a + b; hidden = rmsnorm(hidden, weight).
// Replaces: elementwise_add_store(a, b, h) + rmsnorm(h, w, no) + memcpy(h, no).
// 3 ops → 1 kernel. Used by sandwich-norm post-attention and post-FFN paths.
void add_rmsnorm_inplace(const Tensor& a, const Tensor& b,
                         Tensor& hidden, const Tensor& weight,
                         float eps, cudaStream_t stream,
                         float weight_offset = 0.0f);

// Fused RMSNorm + residual add: output = rmsnorm(input, weight) + residual.
// Replaces: rmsnorm(in, w, out) + elementwise_add(out, r).
// 2 ops → 1 kernel. Used by sandwich-norm post-FFN path.
void rmsnorm_add_residual(const Tensor& input, const Tensor& weight,
                          const Tensor& residual, Tensor& output,
                          float eps, cudaStream_t stream,
                          float weight_offset = 0.0f);

Tensor slice_rows(const Tensor& buf, int n_tokens);

// New: simplified dispatch via GemmContext (preferred for new code)
struct GemmContext;  // forward decl — defined in gemm_context.h
void gemm_dispatch(const Tensor& input, const Tensor& weight,
                   GGMLQuantType qtype, Tensor& output,
                   const GemmContext& ctx);

// Legacy: 23-parameter dispatch (to be removed after migration)
void gemm_dispatch(const Tensor& input, const Tensor& weight,
                   const Tensor& scales, GGMLQuantType qtype,
                   Tensor& output, void* dequant_scratch,
                   cudaStream_t stream,
                   block_q8_1* q8_1_buf = nullptr,
                   float* d8_buf = nullptr,
                   const std::unordered_map<const void*, Tensor>* fp16_cache = nullptr,
                   const std::unordered_map<const void*, FP8CacheEntry>* fp8_cache = nullptr,
                   void* fp8_act_buf = nullptr,
                   float* d_act_scale = nullptr,
                   float* d_fp8_block_maxes = nullptr,
                   float* d_fp8_absmax = nullptr,
                   int fp8_max_grid = 0,
                   const std::unordered_map<const void*, NvFP4QuantResult>* nvfp4_cache = nullptr,
                   const std::unordered_map<const void*, CutlassNvFP4Weight>* cutlass_nvfp4_cache = nullptr,
                   void* cutlass_act_data = nullptr,
                   void* cutlass_act_sf = nullptr,
                   void* cutlass_workspace = nullptr,
                   size_t cutlass_workspace_size = 0,
                   const std::unordered_map<const void*, CutlassMxFP4Weight>* mxfp4_cache = nullptr,
                   void* mxfp4_act_sf = nullptr,
                   void* mxfp4_workspace = nullptr,
                   size_t mxfp4_workspace_size = 0);

} // namespace imp
