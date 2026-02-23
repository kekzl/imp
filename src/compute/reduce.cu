#include "compute/reduce.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cfloat>
#include <cstdio>

namespace imp {

static constexpr int BLOCK_SIZE = 256;
static constexpr int WARP_SIZE = 32;

// ============================================================================
// Warp-level reduction primitives
// ============================================================================

__device__ __forceinline__
float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__device__ __forceinline__
float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float other = __shfl_xor_sync(0xFFFFFFFF, val, offset);
        val = fmaxf(val, other);
    }
    return val;
}

// ============================================================================
// Reduce LAST dimension  --  input [outer, inner] -> output [outer]
// One block per outer row.
// ============================================================================

__global__ void reduce_sum_last_dim_kernel(const float* __restrict__ input,
                                           float* __restrict__ output,
                                           int inner_size) {
    const int row = blockIdx.x;
    const float* row_ptr = input + static_cast<int64_t>(row) * inner_size;

    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < inner_size; i += blockDim.x) {
        local_sum += row_ptr[i];
    }

    // Warp reduction
    local_sum = warp_reduce_sum(local_sum);

    // Cross-warp reduction via shared memory
    constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
    __shared__ float s_warp[NUM_WARPS];

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    if (lane_id == 0) {
        s_warp[warp_id] = local_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < NUM_WARPS) ? s_warp[lane_id] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane_id == 0) {
            output[row] = val;
        }
    }
}

__global__ void reduce_sum_last_dim_fp16_kernel(const half* __restrict__ input,
                                                float* __restrict__ output,
                                                int inner_size) {
    const int row = blockIdx.x;
    const half* row_ptr = input + static_cast<int64_t>(row) * inner_size;

    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < inner_size; i += blockDim.x) {
        local_sum += __half2float(row_ptr[i]);
    }

    local_sum = warp_reduce_sum(local_sum);

    constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
    __shared__ float s_warp[NUM_WARPS];

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    if (lane_id == 0) {
        s_warp[warp_id] = local_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < NUM_WARPS) ? s_warp[lane_id] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane_id == 0) {
            output[row] = val;
        }
    }
}

__global__ void reduce_max_last_dim_kernel(const float* __restrict__ input,
                                           float* __restrict__ output,
                                           int inner_size) {
    const int row = blockIdx.x;
    const float* row_ptr = input + static_cast<int64_t>(row) * inner_size;

    float local_max = -FLT_MAX;
    for (int i = threadIdx.x; i < inner_size; i += blockDim.x) {
        local_max = fmaxf(local_max, row_ptr[i]);
    }

    local_max = warp_reduce_max(local_max);

    constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
    __shared__ float s_warp[NUM_WARPS];

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    if (lane_id == 0) {
        s_warp[warp_id] = local_max;
    }
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < NUM_WARPS) ? s_warp[lane_id] : -FLT_MAX;
        val = warp_reduce_max(val);
        if (lane_id == 0) {
            output[row] = val;
        }
    }
}

__global__ void reduce_max_last_dim_fp16_kernel(const half* __restrict__ input,
                                                float* __restrict__ output,
                                                int inner_size) {
    const int row = blockIdx.x;
    const half* row_ptr = input + static_cast<int64_t>(row) * inner_size;

    float local_max = -FLT_MAX;
    for (int i = threadIdx.x; i < inner_size; i += blockDim.x) {
        local_max = fmaxf(local_max, __half2float(row_ptr[i]));
    }

    local_max = warp_reduce_max(local_max);

    constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
    __shared__ float s_warp[NUM_WARPS];

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    if (lane_id == 0) {
        s_warp[warp_id] = local_max;
    }
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < NUM_WARPS) ? s_warp[lane_id] : -FLT_MAX;
        val = warp_reduce_max(val);
        if (lane_id == 0) {
            output[row] = val;
        }
    }
}

// ============================================================================
// Reduce along an ARBITRARY dimension (strided access)
//
// For input shape [d0, d1, ..., d_{n-1}] reducing along dimension `dim`:
//   outer_size = product of dims before `dim`
//   reduce_size = shape[dim]
//   inner_size = product of dims after `dim`
//
// Output has shape with dimension `dim` removed; total elements = outer * inner.
//
// Each output element output[outer_idx * inner_size + inner_idx] =
//     reduce over r: input[outer_idx * (reduce_size * inner_size)
//                          + r * inner_size + inner_idx]
//
// We launch one block per output element for reduce_size >= BLOCK_SIZE, or
// a grid of threads covering (outer * inner) with per-thread serial reduction.
// ============================================================================

__global__ void reduce_sum_general_kernel(const float* __restrict__ input,
                                          float* __restrict__ output,
                                          int outer_size,
                                          int reduce_size,
                                          int inner_size) {
    // Each block handles one output element.
    int out_idx = blockIdx.x;
    if (out_idx >= outer_size * inner_size) return;

    int outer_idx = out_idx / inner_size;
    int inner_idx = out_idx % inner_size;

    const float* base = input + static_cast<int64_t>(outer_idx) * reduce_size * inner_size
                              + inner_idx;

    float local_sum = 0.0f;
    for (int r = threadIdx.x; r < reduce_size; r += blockDim.x) {
        local_sum += base[static_cast<int64_t>(r) * inner_size];
    }

    local_sum = warp_reduce_sum(local_sum);

    constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
    __shared__ float s_warp[NUM_WARPS];

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    if (lane_id == 0) {
        s_warp[warp_id] = local_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < NUM_WARPS) ? s_warp[lane_id] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane_id == 0) {
            output[out_idx] = val;
        }
    }
}

__global__ void reduce_max_general_kernel(const float* __restrict__ input,
                                          float* __restrict__ output,
                                          int outer_size,
                                          int reduce_size,
                                          int inner_size) {
    int out_idx = blockIdx.x;
    if (out_idx >= outer_size * inner_size) return;

    int outer_idx = out_idx / inner_size;
    int inner_idx = out_idx % inner_size;

    const float* base = input + static_cast<int64_t>(outer_idx) * reduce_size * inner_size
                              + inner_idx;

    float local_max = -FLT_MAX;
    for (int r = threadIdx.x; r < reduce_size; r += blockDim.x) {
        local_max = fmaxf(local_max, base[static_cast<int64_t>(r) * inner_size]);
    }

    local_max = warp_reduce_max(local_max);

    constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
    __shared__ float s_warp[NUM_WARPS];

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    if (lane_id == 0) {
        s_warp[warp_id] = local_max;
    }
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < NUM_WARPS) ? s_warp[lane_id] : -FLT_MAX;
        val = warp_reduce_max(val);
        if (lane_id == 0) {
            output[out_idx] = val;
        }
    }
}

// ============================================================================
// General FP16 variants for arbitrary-dimension reduction
// ============================================================================

__global__ void reduce_sum_general_fp16_kernel(const half* __restrict__ input,
                                               float* __restrict__ output,
                                               int outer_size,
                                               int reduce_size,
                                               int inner_size) {
    int out_idx = blockIdx.x;
    if (out_idx >= outer_size * inner_size) return;

    int outer_idx = out_idx / inner_size;
    int inner_idx = out_idx % inner_size;

    const half* base = input + static_cast<int64_t>(outer_idx) * reduce_size * inner_size
                             + inner_idx;

    float local_sum = 0.0f;
    for (int r = threadIdx.x; r < reduce_size; r += blockDim.x) {
        local_sum += __half2float(base[static_cast<int64_t>(r) * inner_size]);
    }

    local_sum = warp_reduce_sum(local_sum);

    constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
    __shared__ float s_warp[NUM_WARPS];

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    if (lane_id == 0) {
        s_warp[warp_id] = local_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < NUM_WARPS) ? s_warp[lane_id] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane_id == 0) {
            output[out_idx] = val;
        }
    }
}

__global__ void reduce_max_general_fp16_kernel(const half* __restrict__ input,
                                               float* __restrict__ output,
                                               int outer_size,
                                               int reduce_size,
                                               int inner_size) {
    int out_idx = blockIdx.x;
    if (out_idx >= outer_size * inner_size) return;

    int outer_idx = out_idx / inner_size;
    int inner_idx = out_idx % inner_size;

    const half* base = input + static_cast<int64_t>(outer_idx) * reduce_size * inner_size
                             + inner_idx;

    float local_max = -FLT_MAX;
    for (int r = threadIdx.x; r < reduce_size; r += blockDim.x) {
        local_max = fmaxf(local_max, __half2float(base[static_cast<int64_t>(r) * inner_size]));
    }

    local_max = warp_reduce_max(local_max);

    constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
    __shared__ float s_warp[NUM_WARPS];

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    if (lane_id == 0) {
        s_warp[warp_id] = local_max;
    }
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < NUM_WARPS) ? s_warp[lane_id] : -FLT_MAX;
        val = warp_reduce_max(val);
        if (lane_id == 0) {
            output[out_idx] = val;
        }
    }
}

// ============================================================================
// Helper: compute outer_size, reduce_size, inner_size from shape and dim
// ============================================================================

static void compute_reduce_dims(const Tensor& input, int dim,
                                int& outer_size, int& reduce_size, int& inner_size) {
    outer_size = 1;
    for (int i = 0; i < dim; ++i) {
        outer_size *= static_cast<int>(input.shape[i]);
    }
    reduce_size = static_cast<int>(input.shape[dim]);
    inner_size = 1;
    for (int i = dim + 1; i < input.ndim; ++i) {
        inner_size *= static_cast<int>(input.shape[i]);
    }
}

// ============================================================================
// Public API
// ============================================================================

void reduce_sum(const Tensor& input, Tensor& output, int dim,
                cudaStream_t stream) {
    // Normalize negative dim
    if (dim < 0) dim += input.ndim;

    int outer_size, reduce_size, inner_size;
    compute_reduce_dims(input, dim, outer_size, reduce_size, inner_size);

    // Output is FP32 (reductions accumulate in float)
    float* d_output = static_cast<float*>(output.data);

    if (dim == input.ndim - 1) {
        // Reduce along last dimension (contiguous access)
        int num_rows = outer_size;  // inner_size == 1 for last-dim reduction
        if (input.dtype == DType::FP16) {
            const half* d_input = static_cast<const half*>(input.data);
            reduce_sum_last_dim_fp16_kernel<<<num_rows, BLOCK_SIZE, 0, stream>>>(
                d_input, d_output, reduce_size);
        } else {
            const float* d_input = static_cast<const float*>(input.data);
            reduce_sum_last_dim_kernel<<<num_rows, BLOCK_SIZE, 0, stream>>>(
                d_input, d_output, reduce_size);
        }
    } else {
        // General case: reduce along arbitrary dimension
        int num_output = outer_size * inner_size;
        if (input.dtype == DType::FP16) {
            const half* d_input = static_cast<const half*>(input.data);
            reduce_sum_general_fp16_kernel<<<num_output, BLOCK_SIZE, 0, stream>>>(
                d_input, d_output, outer_size, reduce_size, inner_size);
        } else {
            const float* d_input = static_cast<const float*>(input.data);
            reduce_sum_general_kernel<<<num_output, BLOCK_SIZE, 0, stream>>>(
                d_input, d_output, outer_size, reduce_size, inner_size);
        }
    }
}

void reduce_max(const Tensor& input, Tensor& output, int dim,
                cudaStream_t stream) {
    if (dim < 0) dim += input.ndim;

    int outer_size, reduce_size, inner_size;
    compute_reduce_dims(input, dim, outer_size, reduce_size, inner_size);

    float* d_output = static_cast<float*>(output.data);

    if (dim == input.ndim - 1) {
        int num_rows = outer_size;
        if (input.dtype == DType::FP16) {
            const half* d_input = static_cast<const half*>(input.data);
            reduce_max_last_dim_fp16_kernel<<<num_rows, BLOCK_SIZE, 0, stream>>>(
                d_input, d_output, reduce_size);
        } else {
            const float* d_input = static_cast<const float*>(input.data);
            reduce_max_last_dim_kernel<<<num_rows, BLOCK_SIZE, 0, stream>>>(
                d_input, d_output, reduce_size);
        }
    } else {
        int num_output = outer_size * inner_size;
        if (input.dtype == DType::FP16) {
            const half* d_input = static_cast<const half*>(input.data);
            reduce_max_general_fp16_kernel<<<num_output, BLOCK_SIZE, 0, stream>>>(
                d_input, d_output, outer_size, reduce_size, inner_size);
        } else {
            const float* d_input = static_cast<const float*>(input.data);
            reduce_max_general_kernel<<<num_output, BLOCK_SIZE, 0, stream>>>(
                d_input, d_output, outer_size, reduce_size, inner_size);
        }
    }
}

} // namespace imp
