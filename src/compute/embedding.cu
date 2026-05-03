#include "compute/embedding.h"
#include "model/model_config.h"  // QType
#include "core/tensor.h"
#include "core/logging.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace imp {

// --------------------------------------------------------------------------
// Embedding vector type traits: maps element type to 4-element vector type
// FP32: float  -> float4 (16 bytes), FP16: __half -> float2 (8 bytes),
// BF16: uint16 -> uint2  (8 bytes).  All pack 4 elements per vector load.
// --------------------------------------------------------------------------
template <typename T>
struct EmbedVecTraits;
template <>
struct EmbedVecTraits<float> {
    using Vec = float4;
};
template <>
struct EmbedVecTraits<__half> {
    using Vec = float2;
};
template <>
struct EmbedVecTraits<uint16_t> {
    using Vec = uint2;
};

// --------------------------------------------------------------------------
// Scalar embedding kernel (fallback when d_model % 4 != 0)
// --------------------------------------------------------------------------
template <typename T>
__global__ void embedding_lookup_scalar_kernel(const T* __restrict__ table,
                                               const int32_t* __restrict__ token_ids, T* __restrict__ out,
                                               int d_model) {
    const int token = blockIdx.x;
    const int tid = threadIdx.x;
    const int row = token_ids[token];

    for (int i = tid; i < d_model; i += blockDim.x) {
        out[static_cast<int64_t>(token) * d_model + i] = table[static_cast<int64_t>(row) * d_model + i];
    }
}

// Explicit template instantiations
template __global__ void embedding_lookup_scalar_kernel<float>(const float*, const int32_t*, float*, int);
template __global__ void embedding_lookup_scalar_kernel<__half>(const __half*, const int32_t*, __half*, int);

// --------------------------------------------------------------------------
// Vectorized embedding kernel: copies 4 elements per vector load/store
// Grid: (n_tokens), Block: 256
// --------------------------------------------------------------------------
template <typename T>
__global__ void embedding_lookup_vec_kernel(const T* __restrict__ table,
                                            const int32_t* __restrict__ token_ids, T* __restrict__ out,
                                            int d_model) {
    using Vec = typename EmbedVecTraits<T>::Vec;

    const int token = blockIdx.x;
    const int tid = threadIdx.x;
    const int row = token_ids[token];
    const int vec_d = d_model / 4;  // number of vector elements per row

    const Vec* src = reinterpret_cast<const Vec*>(table + static_cast<int64_t>(row) * d_model);
    Vec* dst = reinterpret_cast<Vec*>(out + static_cast<int64_t>(token) * d_model);

    for (int i = tid; i < vec_d; i += blockDim.x) {
        dst[i] = src[i];
    }

    // Handle tail elements (d_model not divisible by 4)
    const int tail_start = vec_d * 4;
    for (int i = tail_start + tid; i < d_model; i += blockDim.x) {
        out[static_cast<int64_t>(token) * d_model + i] = table[static_cast<int64_t>(row) * d_model + i];
    }
}

// Explicit template instantiations
template __global__ void embedding_lookup_vec_kernel<float>(const float*, const int32_t*, float*, int);
template __global__ void embedding_lookup_vec_kernel<__half>(const __half*, const int32_t*, __half*, int);
template __global__ void embedding_lookup_vec_kernel<uint16_t>(const uint16_t*, const int32_t*, uint16_t*,
                                                               int);

// --------------------------------------------------------------------------
// Quantized dequantization helpers (shared by batch and device-side kernels)
// --------------------------------------------------------------------------

// Q8_0 block format: 34 bytes per 32 elements (2 fp16 scale + 32 int8)
static __device__ __forceinline__ half dequant_q8_0_element(const uint8_t* __restrict__ row_ptr, int i) {
    int blk = i / 32;
    int q_idx = i % 32;
    const uint8_t* block_ptr = row_ptr + blk * 34;
    half d_val = *reinterpret_cast<const half*>(block_ptr);
    int8_t q = reinterpret_cast<const int8_t*>(block_ptr + 2)[q_idx];
    return __float2half(__half2float(d_val) * static_cast<float>(q));
}

// Q6_K block format: 210 bytes per 256 elements (GGML interleaved layout)
static __device__ __forceinline__ half dequant_q6k_element(const uint8_t* __restrict__ row_ptr, int idx) {
    int blk = idx / 256;
    int i = idx % 256;
    const uint8_t* block_ptr = row_ptr + blk * 210;

    const uint8_t* ql = block_ptr;
    const uint8_t* qh = block_ptr + 128;
    const int8_t* scales = reinterpret_cast<const int8_t*>(block_ptr + 192);
    half d_val = *reinterpret_cast<const half*>(block_ptr + 208);

    int group = i >> 7;
    int within = i & 127;
    int quad = within >> 5;
    int l = within & 31;

    int ql_idx = (group << 6) + ((quad & 1) << 5) + l;
    int qh_idx = (group << 5) + l;

    uint8_t ql_byte = ql[ql_idx];
    uint8_t low4 = (quad >= 2) ? ((ql_byte >> 4) & 0xFu) : (ql_byte & 0xFu);
    uint8_t high2 = (qh[qh_idx] >> (quad * 2)) & 0x3u;
    int q6 = static_cast<int>((high2 << 4) | low4) - 32;

    float val = __half2float(d_val) * static_cast<float>(scales[i >> 4]) * static_cast<float>(q6);
    return __float2half(val);
}

// --------------------------------------------------------------------------
// Q8_0 embedding lookup: dequantize only the needed rows on the fly.
// Grid: (n_tokens), Block: 256
// --------------------------------------------------------------------------
__global__ void embedding_lookup_q8_0_kernel(const uint8_t* __restrict__ table_raw,
                                             const int32_t* __restrict__ token_ids, half* __restrict__ out,
                                             int d_model) {
    const int token = blockIdx.x;
    const int row = token_ids[token];

    const int blocks_per_row = d_model / 32;
    const int64_t row_bytes = static_cast<int64_t>(blocks_per_row) * 34;
    const uint8_t* row_ptr = table_raw + static_cast<int64_t>(row) * row_bytes;
    half* out_row = out + static_cast<int64_t>(token) * d_model;

    for (int i = threadIdx.x; i < d_model; i += blockDim.x) {
        out_row[i] = dequant_q8_0_element(row_ptr, i);
    }
}

// --------------------------------------------------------------------------
// Q6_K embedding lookup: dequantize only the needed rows on the fly.
// Grid: (n_tokens), Block: 256
// --------------------------------------------------------------------------
__global__ void embedding_lookup_q6k_kernel(const uint8_t* __restrict__ table_raw,
                                            const int32_t* __restrict__ token_ids, half* __restrict__ out,
                                            int d_model) {
    const int token = blockIdx.x;
    const int row = token_ids[token];

    const int blocks_per_row = d_model / 256;
    const int64_t row_bytes = static_cast<int64_t>(blocks_per_row) * 210;
    const uint8_t* row_ptr = table_raw + static_cast<int64_t>(row) * row_bytes;
    half* out_row = out + static_cast<int64_t>(token) * d_model;

    for (int idx = threadIdx.x; idx < d_model; idx += blockDim.x) {
        out_row[idx] = dequant_q6k_element(row_ptr, idx);
    }
}

// --------------------------------------------------------------------------
// Host dispatch
// --------------------------------------------------------------------------
void embedding_lookup(const Tensor& table, const int32_t* token_ids, int n_tokens, Tensor& out,
                      cudaStream_t stream) {
    const int d_model = static_cast<int>(table.shape[1]);
    const int block = 256;

    if (n_tokens == 0)
        return;

    switch (table.qtype) {
        case QType::F32: {
            if (d_model % 4 == 0) {
                embedding_lookup_vec_kernel<float>
                    <<<n_tokens, block, 0, stream>>>(static_cast<const float*>(table.data), token_ids,
                                                     static_cast<float*>(out.data), d_model);
            } else {
                embedding_lookup_scalar_kernel<float>
                    <<<n_tokens, block, 0, stream>>>(static_cast<const float*>(table.data), token_ids,
                                                     static_cast<float*>(out.data), d_model);
            }
            break;
        }
        case QType::F16: {
            if (d_model % 4 == 0) {
                embedding_lookup_vec_kernel<__half>
                    <<<n_tokens, block, 0, stream>>>(static_cast<const __half*>(table.data), token_ids,
                                                     static_cast<__half*>(out.data), d_model);
            } else {
                embedding_lookup_scalar_kernel<__half>
                    <<<n_tokens, block, 0, stream>>>(static_cast<const __half*>(table.data), token_ids,
                                                     static_cast<__half*>(out.data), d_model);
            }
            break;
        }
        case QType::BF16: {
            embedding_lookup_vec_kernel<uint16_t>
                <<<n_tokens, block, 0, stream>>>(static_cast<const uint16_t*>(table.data), token_ids,
                                                 static_cast<uint16_t*>(out.data), d_model);
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
void embedding_lookup(const Tensor& table, const int32_t* token_ids, int n_tokens, Tensor& out, QType qtype,
                      cudaStream_t stream) {
    if (n_tokens == 0)
        return;

    const int d_model = static_cast<int>(table.shape[1]);
    const int block = 256;

    if (qtype == QType::Q8_0) {
        embedding_lookup_q8_0_kernel<<<n_tokens, block, 0, stream>>>(static_cast<const uint8_t*>(table.data),
                                                                     token_ids, static_cast<half*>(out.data),
                                                                     d_model);
        return;
    }

    if (qtype == QType::Q6_K) {
        embedding_lookup_q6k_kernel<<<n_tokens, block, 0, stream>>>(static_cast<const uint8_t*>(table.data),
                                                                    token_ids, static_cast<half*>(out.data),
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

// FP16 device-side embedding (vectorized, reads token ID from device memory)
__global__ void embedding_lookup_fp16_device_kernel(const __half* __restrict__ table,
                                                    const int32_t* __restrict__ d_token_id,
                                                    __half* __restrict__ out, int d_model) {
    const int tid = threadIdx.x;
    const int row = d_token_id[0];  // read from device memory

    const int vec_d = d_model / 4;
    const float2* src = reinterpret_cast<const float2*>(table + static_cast<int64_t>(row) * d_model);
    float2* dst = reinterpret_cast<float2*>(out);

    for (int i = tid; i < vec_d; i += blockDim.x) {
        dst[i] = src[i];
    }

    const int tail_start = vec_d * 4;
    for (int i = tail_start + tid; i < d_model; i += blockDim.x) {
        out[i] = table[static_cast<int64_t>(row) * d_model + i];
    }
}

// Q8_0 device-side embedding (uses shared dequant helper)
__global__ void embedding_lookup_q8_0_device_kernel(const uint8_t* __restrict__ table_raw,
                                                    const int32_t* __restrict__ d_token_id,
                                                    half* __restrict__ out, int d_model) {
    const int row = d_token_id[0];

    const int blocks_per_row = d_model / 32;
    const int64_t row_bytes = static_cast<int64_t>(blocks_per_row) * 34;
    const uint8_t* row_ptr = table_raw + static_cast<int64_t>(row) * row_bytes;

    for (int i = threadIdx.x; i < d_model; i += blockDim.x) {
        out[i] = dequant_q8_0_element(row_ptr, i);
    }
}

// Q6_K device-side embedding (uses shared dequant helper)
__global__ void embedding_lookup_q6k_device_kernel(const uint8_t* __restrict__ table_raw,
                                                   const int32_t* __restrict__ d_token_id,
                                                   half* __restrict__ out, int d_model) {
    const int row = d_token_id[0];

    const int blocks_per_row = d_model / 256;
    const int64_t row_bytes = static_cast<int64_t>(blocks_per_row) * 210;
    const uint8_t* row_ptr = table_raw + static_cast<int64_t>(row) * row_bytes;

    for (int idx = threadIdx.x; idx < d_model; idx += blockDim.x) {
        out[idx] = dequant_q6k_element(row_ptr, idx);
    }
}

void embedding_lookup_from_device(const Tensor& table, const int32_t* d_token_id, Tensor& out,
                                  cudaStream_t stream) {
    const int d_model = static_cast<int>(table.shape[1]);
    const int block = 256;

    if (table.qtype == QType::F16) {
        embedding_lookup_fp16_device_kernel<<<1, block, 0, stream>>>(static_cast<const __half*>(table.data),
                                                                     d_token_id,
                                                                     static_cast<__half*>(out.data), d_model);
    } else if (table.qtype == QType::F32) {
        // For FP32 tables, fall back to regular path with a device-to-host copy
        // (FP32 embedding tables are uncommon in quantized models)
        int32_t h_token;
        IMP_CUDA_CHECK_LOG(
            cudaMemcpyAsync(&h_token, d_token_id, sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
        IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream));
        embedding_lookup(table, &h_token, 1, out, stream);
    }
}

void embedding_lookup_from_device(const Tensor& table, const int32_t* d_token_id, Tensor& out, QType qtype,
                                  cudaStream_t stream) {
    const int d_model = static_cast<int>(table.shape[1]);
    const int block = 256;

    if (qtype == QType::Q8_0) {
        embedding_lookup_q8_0_device_kernel<<<1, block, 0, stream>>>(static_cast<const uint8_t*>(table.data),
                                                                     d_token_id, static_cast<half*>(out.data),
                                                                     d_model);
        return;
    }

    if (qtype == QType::Q6_K) {
        embedding_lookup_q6k_device_kernel<<<1, block, 0, stream>>>(static_cast<const uint8_t*>(table.data),
                                                                    d_token_id, static_cast<half*>(out.data),
                                                                    d_model);
        return;
    }

    // Non-quantized: use the dtype-based device path
    embedding_lookup_from_device(table, d_token_id, out, stream);
}

}  // namespace imp
