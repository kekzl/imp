#include "compute/embedding.h"
#include "model/model.h"  // GGMLQuantType
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
// Q8_0 embedding lookup: dequantize only the needed rows on the fly.
// Q8_0 block format: 34 bytes per 32 elements (2 fp16 scale + 32 int8).
// Grid: (n_tokens), Block: 256
// --------------------------------------------------------------------------
__global__ void embedding_lookup_q8_0_kernel(
    const uint8_t* __restrict__ table_raw,
    const int32_t* __restrict__ token_ids,
    half* __restrict__ out,
    int d_model)
{
    const int token = blockIdx.x;
    const int row = token_ids[token];

    const int blocks_per_row = d_model / 32;
    const int64_t row_bytes = static_cast<int64_t>(blocks_per_row) * 34;
    const uint8_t* row_ptr = table_raw + static_cast<int64_t>(row) * row_bytes;
    half* out_row = out + static_cast<int64_t>(token) * d_model;

    for (int i = threadIdx.x; i < d_model; i += blockDim.x) {
        int blk = i / 32;
        int q_idx = i % 32;
        const uint8_t* block_ptr = row_ptr + blk * 34;
        half d_val = *reinterpret_cast<const half*>(block_ptr);
        int8_t q = reinterpret_cast<const int8_t*>(block_ptr + 2)[q_idx];
        out_row[i] = __float2half(__half2float(d_val) * static_cast<float>(q));
    }
}

// --------------------------------------------------------------------------
// Q6_K embedding lookup: dequantize only the needed rows on the fly.
// Q6_K block format: 210 bytes per 256 elements.
// Uses same GGML interleaved layout as dequant_gpu.cu.
// --------------------------------------------------------------------------
__global__ void embedding_lookup_q6k_kernel(
    const uint8_t* __restrict__ table_raw,
    const int32_t* __restrict__ token_ids,
    half* __restrict__ out,
    int d_model)
{
    const int token = blockIdx.x;
    const int row = token_ids[token];

    const int blocks_per_row = d_model / 256;
    const int64_t row_bytes = static_cast<int64_t>(blocks_per_row) * 210;
    const uint8_t* row_ptr = table_raw + static_cast<int64_t>(row) * row_bytes;
    half* out_row = out + static_cast<int64_t>(token) * d_model;

    for (int idx = threadIdx.x; idx < d_model; idx += blockDim.x) {
        int blk = idx / 256;
        int i   = idx % 256;
        const uint8_t* block_ptr = row_ptr + blk * 210;

        const uint8_t* ql    = block_ptr;
        const uint8_t* qh    = block_ptr + 128;
        const int8_t*  scales = reinterpret_cast<const int8_t*>(block_ptr + 192);
        half d_val = *reinterpret_cast<const half*>(block_ptr + 208);

        int group  = i >> 7;
        int within = i & 127;
        int quad   = within >> 5;
        int l      = within & 31;

        int ql_idx = (group << 6) + ((quad & 1) << 5) + l;
        int qh_idx = (group << 5) + l;

        uint8_t ql_byte = ql[ql_idx];
        uint8_t low4 = (quad >= 2) ? ((ql_byte >> 4) & 0xFu) : (ql_byte & 0xFu);
        uint8_t high2 = (qh[qh_idx] >> (quad * 2)) & 0x3u;
        int q6 = static_cast<int>((high2 << 4) | low4) - 32;

        float val = __half2float(d_val) * static_cast<float>(scales[i >> 4])
                    * static_cast<float>(q6);
        out_row[idx] = __float2half(val);
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

// --------------------------------------------------------------------------
// Overload with quantization type for raw quantized embedding tables.
// Falls through to standard dtype-based dispatch for non-quantized types.
// --------------------------------------------------------------------------
void embedding_lookup(const Tensor& table, const int32_t* token_ids,
                      int n_tokens, Tensor& out,
                      GGMLQuantType qtype,
                      cudaStream_t stream)
{
    if (n_tokens == 0) return;

    const int d_model = static_cast<int>(table.shape[1]);
    const int block = 256;

    if (qtype == GGMLQuantType::Q8_0) {
        embedding_lookup_q8_0_kernel<<<n_tokens, block, 0, stream>>>(
            static_cast<const uint8_t*>(table.data),
            token_ids,
            static_cast<half*>(out.data),
            d_model);
        return;
    }

    if (qtype == GGMLQuantType::Q6_K) {
        embedding_lookup_q6k_kernel<<<n_tokens, block, 0, stream>>>(
            static_cast<const uint8_t*>(table.data),
            token_ids,
            static_cast<half*>(out.data),
            d_model);
        return;
    }

    // Non-quantized: delegate to standard dtype-based dispatch
    embedding_lookup(table, token_ids, n_tokens, out, stream);
}

// --------------------------------------------------------------------------
// Device-side embedding lookup: reads token ID from device memory.
// For async decode where the sampled token stays on GPU.
// Reads d_token_id[0] in the kernel instead of a host-provided array.
// Only supports n_tokens=1 (single decode step).
// --------------------------------------------------------------------------

// FP16 device-side embedding (vectorized)
__global__ void embedding_lookup_fp16_device_kernel(
    const __half* __restrict__ table,
    const int32_t* __restrict__ d_token_id,
    __half* __restrict__ out,
    int d_model)
{
    const int tid = threadIdx.x;
    const int row = d_token_id[0];  // read from device memory

    const int vec_d = d_model / 4;
    const float2* src = reinterpret_cast<const float2*>(table + static_cast<int64_t>(row) * d_model);
    float2*       dst = reinterpret_cast<float2*>(out);

    for (int i = tid; i < vec_d; i += blockDim.x) {
        dst[i] = src[i];
    }

    const int tail_start = vec_d * 4;
    for (int i = tail_start + tid; i < d_model; i += blockDim.x) {
        out[i] = table[static_cast<int64_t>(row) * d_model + i];
    }
}

// Q8_0 device-side embedding
__global__ void embedding_lookup_q8_0_device_kernel(
    const uint8_t* __restrict__ table_raw,
    const int32_t* __restrict__ d_token_id,
    half* __restrict__ out,
    int d_model)
{
    const int row = d_token_id[0];

    const int blocks_per_row = d_model / 32;
    const int64_t row_bytes = static_cast<int64_t>(blocks_per_row) * 34;
    const uint8_t* row_ptr = table_raw + static_cast<int64_t>(row) * row_bytes;

    for (int i = threadIdx.x; i < d_model; i += blockDim.x) {
        int blk = i / 32;
        int q_idx = i % 32;
        const uint8_t* block_ptr = row_ptr + blk * 34;
        half d_val = *reinterpret_cast<const half*>(block_ptr);
        int8_t q = reinterpret_cast<const int8_t*>(block_ptr + 2)[q_idx];
        out[i] = __float2half(__half2float(d_val) * static_cast<float>(q));
    }
}

// Q6_K device-side embedding
__global__ void embedding_lookup_q6k_device_kernel(
    const uint8_t* __restrict__ table_raw,
    const int32_t* __restrict__ d_token_id,
    half* __restrict__ out,
    int d_model)
{
    const int row = d_token_id[0];

    const int blocks_per_row = d_model / 256;
    const int64_t row_bytes = static_cast<int64_t>(blocks_per_row) * 210;
    const uint8_t* row_ptr = table_raw + static_cast<int64_t>(row) * row_bytes;

    for (int idx = threadIdx.x; idx < d_model; idx += blockDim.x) {
        int blk = idx / 256;
        int i   = idx % 256;
        const uint8_t* block_ptr = row_ptr + blk * 210;

        const uint8_t* ql    = block_ptr;
        const uint8_t* qh    = block_ptr + 128;
        const int8_t*  scales = reinterpret_cast<const int8_t*>(block_ptr + 192);
        half d_val = *reinterpret_cast<const half*>(block_ptr + 208);

        int group  = i >> 7;
        int within = i & 127;
        int quad   = within >> 5;
        int l      = within & 31;

        int ql_idx = (group << 6) + ((quad & 1) << 5) + l;
        int qh_idx = (group << 5) + l;

        uint8_t ql_byte = ql[ql_idx];
        uint8_t low4 = (quad >= 2) ? ((ql_byte >> 4) & 0xFu) : (ql_byte & 0xFu);
        uint8_t high2 = (qh[qh_idx] >> (quad * 2)) & 0x3u;
        int q6 = static_cast<int>((high2 << 4) | low4) - 32;

        float val = __half2float(d_val) * static_cast<float>(scales[i >> 4])
                    * static_cast<float>(q6);
        out[idx] = __float2half(val);
    }
}

void embedding_lookup_from_device(const Tensor& table, const int32_t* d_token_id,
                                   Tensor& out, cudaStream_t stream) {
    const int d_model = static_cast<int>(table.shape[1]);
    const int block = 256;

    if (table.dtype == DType::FP16) {
        embedding_lookup_fp16_device_kernel<<<1, block, 0, stream>>>(
            static_cast<const __half*>(table.data),
            d_token_id,
            static_cast<__half*>(out.data),
            d_model);
    } else if (table.dtype == DType::FP32) {
        // For FP32 tables, fall back to regular path with a device-to-host copy
        // (FP32 embedding tables are uncommon in quantized models)
        int32_t h_token;
        cudaMemcpyAsync(&h_token, d_token_id, sizeof(int32_t),
                         cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        embedding_lookup(table, &h_token, 1, out, stream);
    }
}

void embedding_lookup_from_device(const Tensor& table, const int32_t* d_token_id,
                                   Tensor& out, GGMLQuantType qtype,
                                   cudaStream_t stream) {
    const int d_model = static_cast<int>(table.shape[1]);
    const int block = 256;

    if (qtype == GGMLQuantType::Q8_0) {
        embedding_lookup_q8_0_device_kernel<<<1, block, 0, stream>>>(
            static_cast<const uint8_t*>(table.data),
            d_token_id,
            static_cast<half*>(out.data),
            d_model);
        return;
    }

    if (qtype == GGMLQuantType::Q6_K) {
        embedding_lookup_q6k_device_kernel<<<1, block, 0, stream>>>(
            static_cast<const uint8_t*>(table.data),
            d_token_id,
            static_cast<half*>(out.data),
            d_model);
        return;
    }

    // Non-quantized: use the dtype-based device path
    embedding_lookup_from_device(table, d_token_id, out, stream);
}

} // namespace imp
