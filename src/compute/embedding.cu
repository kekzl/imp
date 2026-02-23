#include "compute/embedding.h"
#include "core/tensor.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace imp {

// --------------------------------------------------------------------------
// Vectorized embedding kernel (float4 path for FP32)
// Grid: (n_tokens), Block: min(256, ceil(d_model/4))
// Each thread copies 4 consecutive FP32 elements.
// --------------------------------------------------------------------------
__global__ void embedding_lookup_fp32_vec4(
    const float* __restrict__ table,
    const int32_t* __restrict__ token_ids,
    float* __restrict__ out,
    int d_model)
{
    const int token = blockIdx.x;
    const int tid   = threadIdx.x;
    const int vec_d = d_model / 4;  // number of float4 elements per row

    const int row = token_ids[token];
    const float4* src = reinterpret_cast<const float4*>(table + static_cast<int64_t>(row) * d_model);
    float4*       dst = reinterpret_cast<float4*>(out + static_cast<int64_t>(token) * d_model);

    for (int i = tid; i < vec_d; i += blockDim.x) {
        dst[i] = src[i];
    }

    // Handle tail elements (d_model not divisible by 4)
    const int tail_start = vec_d * 4;
    for (int i = tail_start + tid; i < d_model; i += blockDim.x) {
        out[static_cast<int64_t>(token) * d_model + i] =
            table[static_cast<int64_t>(row) * d_model + i];
    }
}

// --------------------------------------------------------------------------
// Scalar embedding kernel for FP32 fallback
// --------------------------------------------------------------------------
__global__ void embedding_lookup_fp32(
    const float* __restrict__ table,
    const int32_t* __restrict__ token_ids,
    float* __restrict__ out,
    int d_model)
{
    const int token = blockIdx.x;
    const int tid   = threadIdx.x;
    const int row   = token_ids[token];

    for (int i = tid; i < d_model; i += blockDim.x) {
        out[static_cast<int64_t>(token) * d_model + i] =
            table[static_cast<int64_t>(row) * d_model + i];
    }
}

// --------------------------------------------------------------------------
// Vectorized embedding kernel for FP16 (copies 4 half values = 2x uint32)
// --------------------------------------------------------------------------
__global__ void embedding_lookup_fp16_vec(
    const __half* __restrict__ table,
    const int32_t* __restrict__ token_ids,
    __half* __restrict__ out,
    int d_model)
{
    const int token = blockIdx.x;
    const int tid   = threadIdx.x;
    const int row   = token_ids[token];

    // Use float2 to load 4 half values at once (8 bytes)
    const int vec_d = d_model / 4;
    const float2* src = reinterpret_cast<const float2*>(table + static_cast<int64_t>(row) * d_model);
    float2*       dst = reinterpret_cast<float2*>(out + static_cast<int64_t>(token) * d_model);

    for (int i = tid; i < vec_d; i += blockDim.x) {
        dst[i] = src[i];
    }

    // Handle tail
    const int tail_start = vec_d * 4;
    for (int i = tail_start + tid; i < d_model; i += blockDim.x) {
        out[static_cast<int64_t>(token) * d_model + i] =
            table[static_cast<int64_t>(row) * d_model + i];
    }
}

// --------------------------------------------------------------------------
// Scalar FP16 fallback
// --------------------------------------------------------------------------
__global__ void embedding_lookup_fp16(
    const __half* __restrict__ table,
    const int32_t* __restrict__ token_ids,
    __half* __restrict__ out,
    int d_model)
{
    const int token = blockIdx.x;
    const int tid   = threadIdx.x;
    const int row   = token_ids[token];

    for (int i = tid; i < d_model; i += blockDim.x) {
        out[static_cast<int64_t>(token) * d_model + i] =
            table[static_cast<int64_t>(row) * d_model + i];
    }
}

// --------------------------------------------------------------------------
// BF16 embedding kernel (__nv_bfloat16 stored as uint16)
// --------------------------------------------------------------------------
__global__ void embedding_lookup_bf16_vec(
    const uint16_t* __restrict__ table,
    const int32_t* __restrict__ token_ids,
    uint16_t* __restrict__ out,
    int d_model)
{
    const int token = blockIdx.x;
    const int tid   = threadIdx.x;
    const int row   = token_ids[token];

    // Use uint2 to copy 4 bf16 values (8 bytes) at a time
    const int vec_d = d_model / 4;
    const uint2* src = reinterpret_cast<const uint2*>(table + static_cast<int64_t>(row) * d_model);
    uint2*       dst = reinterpret_cast<uint2*>(out + static_cast<int64_t>(token) * d_model);

    for (int i = tid; i < vec_d; i += blockDim.x) {
        dst[i] = src[i];
    }

    const int tail_start = vec_d * 4;
    for (int i = tail_start + tid; i < d_model; i += blockDim.x) {
        out[static_cast<int64_t>(token) * d_model + i] =
            table[static_cast<int64_t>(row) * d_model + i];
    }
}

// --------------------------------------------------------------------------
// Host dispatch
// --------------------------------------------------------------------------
void embedding_lookup(const Tensor& table, const int32_t* token_ids,
                      int n_tokens, Tensor& out,
                      cudaStream_t stream)
{
    const int d_model = static_cast<int>(table.shape[1]);
    const int block = 256;

    if (n_tokens == 0) return;

    switch (table.dtype) {
        case DType::FP32: {
            if (d_model % 4 == 0) {
                embedding_lookup_fp32_vec4<<<n_tokens, block, 0, stream>>>(
                    static_cast<const float*>(table.data),
                    token_ids,
                    static_cast<float*>(out.data),
                    d_model);
            } else {
                embedding_lookup_fp32<<<n_tokens, block, 0, stream>>>(
                    static_cast<const float*>(table.data),
                    token_ids,
                    static_cast<float*>(out.data),
                    d_model);
            }
            break;
        }
        case DType::FP16: {
            if (d_model % 4 == 0) {
                embedding_lookup_fp16_vec<<<n_tokens, block, 0, stream>>>(
                    static_cast<const __half*>(table.data),
                    token_ids,
                    static_cast<__half*>(out.data),
                    d_model);
            } else {
                embedding_lookup_fp16<<<n_tokens, block, 0, stream>>>(
                    static_cast<const __half*>(table.data),
                    token_ids,
                    static_cast<__half*>(out.data),
                    d_model);
            }
            break;
        }
        case DType::BF16: {
            embedding_lookup_bf16_vec<<<n_tokens, block, 0, stream>>>(
                static_cast<const uint16_t*>(table.data),
                token_ids,
                static_cast<uint16_t*>(out.data),
                d_model);
            break;
        }
        default:
            break;
    }
}

} // namespace imp
