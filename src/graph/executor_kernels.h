#pragma once

#include "graph/executor.h"
#include "compute/gemm.h"
#include "quant/quant_gemm.h"
#include "quant/nvfp4_quant.h"
#include "compute/gemm_cutlass_sm120.h"
#include "compute/gemm_cutlass_mxfp4_sm120.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <unordered_map>

namespace imp {

// ---------------------------------------------------------------------------
// CUDA kernels used by the executor
// ---------------------------------------------------------------------------

__global__ void broadcast_add_bias_fp16_kernel(half* out, const half* bias,
                                                int rows, int cols);

__global__ void scale_fp16_kernel(half* data, half scale, int64_t n);

__global__ void elementwise_add_fp16_kernel(half* a, const half* b, int64_t n);

__global__ void elementwise_add_store_fp16_kernel(const half* a, const half* b,
                                                   half* out, int64_t n);

__global__ void fp32_accum_add_fp16_kernel(float* accum, const half* branch, int64_t n);

__global__ void fp32_to_fp16_rowscale_kernel(const float* __restrict__ in,
                                             half* __restrict__ out,
                                             int rows, int cols);

__global__ void rmsnorm_fp32_accum_to_fp16_kernel(
        const half* __restrict__ input,
        const half* __restrict__ norm_w,
        float* __restrict__ fp32_accum,
        half* __restrict__ output,
        int d_model,
        float eps,
        float weight_offset);

__global__ void fp16_to_fp32_kernel(const half* in, float* out, int64_t n);

__global__ void elementwise_add_fp32_kernel(float* a, const float* b, int64_t n);

__global__ void write_kv_cache_kernel(
    const half* data_in,
    const int* positions,
    const int* block_tables,
    half* cache_base,
    int block_stride,
    int row_elems,
    int block_size,
    int n_tokens,
    int max_blocks_per_seq,
    int n_sequences);

__global__ void write_kv_cache_fused_kernel(
    const half* k_in,
    const half* v_in,
    const int* positions,
    const int* block_tables,
    half* k_cache_base,
    half* v_cache_base,
    int block_stride,
    int row_elems,
    int block_size,
    int n_tokens,
    int max_blocks_per_seq,
    int n_sequences);

#ifdef __CUDA_FP8_TYPES_EXIST__
__global__ void write_kv_cache_fp8_kernel(
    const half* data_in,
    const int* positions,
    const int* block_tables,
    __nv_fp8_e4m3* cache_base,
    float inv_scale,
    int block_stride,
    int row_elems,
    int block_size,
    int n_tokens,
    int max_blocks_per_seq,
    int n_sequences);
#else
__global__ void write_kv_cache_fp8_kernel(
    const half* data_in,
    const int* positions,
    const int* block_tables,
    uint8_t* cache_base,
    float inv_scale,
    int block_stride,
    int row_elems,
    int block_size,
    int n_tokens,
    int max_blocks_per_seq,
    int n_sequences);
#endif

__global__ void write_kv_cache_fp8_fused_kernel(
    const half* __restrict__ k_in,
    const half* __restrict__ v_in,
    const int* __restrict__ positions,
    const int* __restrict__ block_tables,
    __nv_fp8_e4m3* k_cache_base,
    __nv_fp8_e4m3* v_cache_base,
    float inv_scale,
    int block_stride,
    int row_elems,
    int block_size,
    int n_tokens,
    int max_blocks_per_seq,
    int n_sequences);

__global__ void write_kv_cache_int8_kernel(
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

__global__ void write_kv_cache_rope_fused_kernel(
    const half* __restrict__ k_in,
    const half* __restrict__ v_in,
    const int* __restrict__ positions,
    const int* __restrict__ block_tables,
    half* k_cache_base,
    half* v_cache_base,
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

__global__ void rope_q_only_fp16_kernel(
    half* __restrict__ Q,
    const int* __restrict__ positions,
    int n_heads,
    int head_dim,
    float theta,
    float inv_scaling,
    int rope_pairs,
    bool neox,
    const float* __restrict__ longrope_inv_freqs);

__global__ void add_fp16_bias_to_fp32_kernel(float* __restrict__ data,
                                              const half* __restrict__ bias,
                                              int n_tokens, int n_cols);

__global__ void scale_fp32_kernel(float* __restrict__ data, float scale, int64_t n);

__global__ void logit_softcap_fp32_kernel(float* __restrict__ data,
                                          float softcap, float inv_softcap,
                                          int64_t n);

__global__ void fp32_to_fp16_kernel(const float* __restrict__ in,
                                    half* __restrict__ out,
                                    int64_t n);

// ---------------------------------------------------------------------------
// Host-side helper functions
// ---------------------------------------------------------------------------

void elementwise_add(Tensor& a, const Tensor& b, cudaStream_t stream);

void elementwise_add_store(const Tensor& a, const Tensor& b, Tensor& out,
                           cudaStream_t stream);

void add_bias(Tensor& out, const Tensor& bias, cudaStream_t stream);

Tensor slice_rows(const Tensor& buf, int n_tokens);

void gemm_dispatch(const Tensor& input, const Tensor& weight,
                   const Tensor& scales, GGMLQuantType qtype,
                   Tensor& output, void* dequant_scratch,
                   cudaStream_t stream,
                   block_q8_1* q8_1_buf = nullptr,
                   float* d8_buf = nullptr,
                   const std::unordered_map<const void*, Tensor>* fp16_cache = nullptr,
                   const std::unordered_map<const void*, GraphExecutor::FP8CacheEntry>* fp8_cache = nullptr,
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
