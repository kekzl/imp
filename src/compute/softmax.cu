#include "compute/softmax.h"
#include "core/tensor.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cfloat>

namespace imp {

// --------------------------------------------------------------------------
// Warp-level reductions
// --------------------------------------------------------------------------
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// --------------------------------------------------------------------------
// Block-level max reduction (up to 8 warps = 256 threads)
// --------------------------------------------------------------------------
__device__ float block_reduce_max(float val) {
    __shared__ float shared[8];
    const int lane    = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;

    val = warp_reduce_max(val);

    if (lane == 0) {
        shared[warp_id] = val;
    }
    __syncthreads();

    const int num_warps = (blockDim.x + 31) / 32;
    val = (threadIdx.x < num_warps) ? shared[threadIdx.x] : -FLT_MAX;
    if (warp_id == 0) {
        val = warp_reduce_max(val);
    }
    return val;
}

// --------------------------------------------------------------------------
// Block-level sum reduction
// --------------------------------------------------------------------------
__device__ float block_reduce_sum_softmax(float val) {
    __shared__ float shared[8];
    const int lane    = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;

    val = warp_reduce_sum(val);

    if (lane == 0) {
        shared[warp_id] = val;
    }
    __syncthreads();

    const int num_warps = (blockDim.x + 31) / 32;
    val = (threadIdx.x < num_warps) ? shared[threadIdx.x] : 0.0f;
    if (warp_id == 0) {
        val = warp_reduce_sum(val);
    }
    return val;
}

// --------------------------------------------------------------------------
// Online softmax FP32 kernel: 2 passes
// Pass 1: compute max AND sum(exp(x - running_max)) in a single traversal
//         using the online softmax trick (Milakov & Gimelshein 2018)
// Pass 2: normalize output[i] = exp(x[i] - max) / sum
//
// One block per row. Block: 256 threads.
// --------------------------------------------------------------------------
__global__ void softmax_fp32_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int cols)
{
    const int row = blockIdx.x;
    const float* in_row  = input  + static_cast<int64_t>(row) * cols;
    float*       out_row = output + static_cast<int64_t>(row) * cols;

    // ---- Online max + sum computation ----
    // Each thread maintains a local (max, sum) pair
    float local_max = -FLT_MAX;
    float local_sum = 0.0f;

    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float val = in_row[i];
        if (val > local_max) {
            // Rescale previous sum to account for new max
            local_sum = local_sum * __expf(local_max - val);
            local_max = val;
        }
        local_sum += __expf(val - local_max);
    }

    // Reduce max across block
    // We need a two-step reduction: first find global max, then fix sums
    __shared__ float s_max;
    __shared__ float s_sum;

    // Step 1: reduce max
    float global_max = block_reduce_max(local_max);
    if (threadIdx.x == 0) {
        s_max = global_max;
    }
    __syncthreads();
    global_max = s_max;

    // Step 2: rescale each thread's local sum to the global max, then reduce
    local_sum = local_sum * __expf(local_max - global_max);
    float global_sum = block_reduce_sum_softmax(local_sum);
    if (threadIdx.x == 0) {
        s_sum = global_sum;
    }
    __syncthreads();
    global_sum = s_sum;

    // ---- Pass 2: normalize ----
    float inv_sum = 1.0f / global_sum;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        out_row[i] = __expf(in_row[i] - global_max) * inv_sum;
    }
}

// --------------------------------------------------------------------------
// Online softmax FP16 kernel (read half, compute in float, write half)
// --------------------------------------------------------------------------
__global__ void softmax_fp16_kernel(
    const __half* __restrict__ input,
    __half* __restrict__ output,
    int cols)
{
    const int row = blockIdx.x;
    const __half* in_row  = input  + static_cast<int64_t>(row) * cols;
    __half*       out_row = output + static_cast<int64_t>(row) * cols;

    float local_max = -FLT_MAX;
    float local_sum = 0.0f;

    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float val = __half2float(in_row[i]);
        if (val > local_max) {
            local_sum = local_sum * __expf(local_max - val);
            local_max = val;
        }
        local_sum += __expf(val - local_max);
    }

    __shared__ float s_max;
    __shared__ float s_sum;

    float global_max = block_reduce_max(local_max);
    if (threadIdx.x == 0) {
        s_max = global_max;
    }
    __syncthreads();
    global_max = s_max;

    local_sum = local_sum * __expf(local_max - global_max);
    float global_sum = block_reduce_sum_softmax(local_sum);
    if (threadIdx.x == 0) {
        s_sum = global_sum;
    }
    __syncthreads();
    global_sum = s_sum;

    float inv_sum = 1.0f / global_sum;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float val = __half2float(in_row[i]);
        out_row[i] = __float2half(__expf(val - global_max) * inv_sum);
    }
}

// --------------------------------------------------------------------------
// Host dispatch
// --------------------------------------------------------------------------
void softmax(const Tensor& input, Tensor& output,
             cudaStream_t stream)
{
    // input/output: [rows, cols] -- softmax over last dimension
    // For higher-dim tensors, treat all leading dims as "rows"
    int64_t cols = input.shape[input.ndim - 1];
    int64_t rows = 1;
    for (int i = 0; i < input.ndim - 1; ++i) {
        rows *= input.shape[i];
    }

    if (rows == 0 || cols == 0) return;

    const int block = 256;
    const int grid  = static_cast<int>(rows);

    switch (input.dtype) {
        case DType::FP32:
            softmax_fp32_kernel<<<grid, block, 0, stream>>>(
                static_cast<const float*>(input.data),
                static_cast<float*>(output.data),
                static_cast<int>(cols));
            break;
        case DType::FP16:
            softmax_fp16_kernel<<<grid, block, 0, stream>>>(
                static_cast<const __half*>(input.data),
                static_cast<__half*>(output.data),
                static_cast<int>(cols));
            break;
        default:
            break;
    }
}

} // namespace imp
