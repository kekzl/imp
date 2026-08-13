#include "compute/sampling.h"
#include "compute/sampling_internal.cuh"
#include "compute/warp_reduce.cuh"
#include "core/logging.h"
#include "memory/engine_arena.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cfloat>

namespace imp {

// ============================================================================
// Greedy sampling (argmax)
// ============================================================================

// Single-block argmax kernel (fallback for paths without pre-allocated scratch).
__global__ void argmax_kernel(const float* __restrict__ logits, int vocab_size,
                              int32_t* __restrict__ d_result) {
    float local_max = -FLT_MAX;
    int local_idx = 0;

    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        float v = logits[i];
        if (v > local_max || (v == local_max && i < local_idx)) {
            local_max = v;
            local_idx = i;
        }
    }

    warp_argmax(local_max, local_idx);

    constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
    __shared__ float s_val[NUM_WARPS];
    __shared__ int s_idx[NUM_WARPS];

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    if (lane_id == 0) {
        s_val[warp_id] = local_max;
        s_idx[warp_id] = local_idx;
    }
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < NUM_WARPS) ? s_val[lane_id] : -FLT_MAX;
        int idx = (lane_id < NUM_WARPS) ? s_idx[lane_id] : 0;

#pragma unroll
        for (int offset = NUM_WARPS / 2; offset > 0; offset >>= 1) {
            float other_val = __shfl_xor_sync(0xFFFFFFFF, val, offset);
            int other_idx = __shfl_xor_sync(0xFFFFFFFF, idx, offset);
            if (other_val > val || (other_val == val && other_idx < idx)) {
                val = other_val;
                idx = other_idx;
            }
        }

        if (lane_id == 0) {
            d_result[0] = static_cast<int32_t>(idx);
        }
    }
}

// Multi-block argmax: distributes work across ARGMAX_NBLOCKS blocks so all SMs
// participate.  The single-block kernel above uses 1 SM and takes ~190 us for
// vocab=152K; this version takes ~10 us.
//
// Scratch layout (passed as d_scratch, ARGMAX_SCRATCH_BYTES total):
//   float    partial_vals [ARGMAX_NBLOCKS]
//   int32_t  partial_idxs [ARGMAX_NBLOCKS]

// Phase 1: each block scans its stripe and writes its local max to partials.
__global__ void argmax_partial_kernel(const float* __restrict__ logits, int vocab_size,
                                      float* __restrict__ partial_vals, int32_t* __restrict__ partial_idxs) {
    float local_max = -FLT_MAX;
    int local_idx = 0;

    int start = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = start; i < vocab_size; i += stride) {
        float v = logits[i];
        if (v > local_max || (v == local_max && i < local_idx)) {
            local_max = v;
            local_idx = i;
        }
    }

    warp_argmax(local_max, local_idx);

    constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
    __shared__ float s_val[NUM_WARPS];
    __shared__ int s_idx[NUM_WARPS];

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    if (lane_id == 0) {
        s_val[warp_id] = local_max;
        s_idx[warp_id] = local_idx;
    }
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < NUM_WARPS) ? s_val[lane_id] : -FLT_MAX;
        int idx = (lane_id < NUM_WARPS) ? s_idx[lane_id] : 0;

#pragma unroll
        for (int offset = NUM_WARPS / 2; offset > 0; offset >>= 1) {
            float other_val = __shfl_xor_sync(0xFFFFFFFF, val, offset);
            int other_idx = __shfl_xor_sync(0xFFFFFFFF, idx, offset);
            if (other_val > val || (other_val == val && other_idx < idx)) {
                val = other_val;
                idx = other_idx;
            }
        }

        if (lane_id == 0) {
            partial_vals[blockIdx.x] = val;
            partial_idxs[blockIdx.x] = idx;
        }
    }
}

// Phase 2: single block reduces ARGMAX_NBLOCKS partial results.
__global__ void argmax_reduce_kernel(const float* __restrict__ partial_vals,
                                     const int32_t* __restrict__ partial_idxs, int n_blocks,
                                     int32_t* __restrict__ d_result) {
    float local_max = -FLT_MAX;
    int local_idx = 0;

    for (int i = threadIdx.x; i < n_blocks; i += blockDim.x) {
        float v = partial_vals[i];
        int idx = partial_idxs[i];
        if (v > local_max || (v == local_max && idx < local_idx)) {
            local_max = v;
            local_idx = idx;
        }
    }

    warp_argmax(local_max, local_idx);

    if (threadIdx.x == 0) {
        d_result[0] = static_cast<int32_t>(local_idx);
    }
}

int32_t sample_greedy(const Tensor& logits, cudaStream_t stream) {
    const int vocab_size = static_cast<int>(logits.shape[0]);
    const float* d_logits = static_cast<const float*>(logits.data);

    // Four bytes, allocated ONCE. This used to cudaMalloc and cudaFree per call
    // — an I2 violation on a sampling path, and the kind that hides because the
    // allocation is trivially small (docs/internals/MEMORY.md A3.2). The
    // buffer is write-then-read within this call and reused by every later one,
    // so engine-persistent is the correct tier.
    //
    // This overload is itself the fallback the executor takes when its own
    // d_sample_result_ is unavailable, and single-engine-per-process is the
    // supported deployment (memory/vram_query.h), so a file-static is safe here
    // in exactly the way it would not be for a per-request buffer.
    static int32_t* d_result = nullptr;
    if (!d_result) {
        if (auto slab = engine_arena().take_bytes(sizeof(int32_t)); !slab.empty()) {
            d_result = reinterpret_cast<int32_t*>(slab.data());
        } else if (cudaMalloc(&d_result, sizeof(int32_t)) != cudaSuccess) {
            // Kept because the arena is closed in a bare unit test, and returning
            // token 0 there would be a silently wrong sample rather than a loud
            // failure. It runs at most once per process.
            IMP_LOG_ERROR("sample_greedy: could not obtain the result scratch");
            d_result = nullptr;
            return 0;
        }
    }

    argmax_kernel<<<1, BLOCK_SIZE, 0, stream>>>(d_logits, vocab_size, d_result);
    IMP_CUDA_CHECK_LAUNCH();

    int32_t h_result = 0;
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(&h_result, d_result, sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
    return h_result;
}

int32_t sample_greedy(const Tensor& logits, int32_t* d_result, cudaStream_t stream) {
    const int vocab_size = static_cast<int>(logits.shape[0]);
    const float* d_logits = static_cast<const float*>(logits.data);

    // Use multi-block argmax: scratch lives right after d_result.
    // Layout: [result(4B)] [partial_vals(ARGMAX_NBLOCKS*4B)] [partial_idxs(ARGMAX_NBLOCKS*4B)]
    auto* base = reinterpret_cast<char*>(d_result);
    auto* partial_vals = reinterpret_cast<float*>(base + sizeof(int32_t));
    auto* partial_idxs = reinterpret_cast<int32_t*>(base + sizeof(int32_t) + ARGMAX_NBLOCKS * sizeof(float));

    argmax_partial_kernel<<<ARGMAX_NBLOCKS, BLOCK_SIZE, 0, stream>>>(d_logits, vocab_size, partial_vals,
                                                                     partial_idxs);
    IMP_CUDA_CHECK_LAUNCH();
    argmax_reduce_kernel<<<1, WARP_SIZE, 0, stream>>>(partial_vals, partial_idxs, ARGMAX_NBLOCKS, d_result);
    IMP_CUDA_CHECK_LAUNCH();

    int32_t h_result = 0;
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(&h_result, d_result, sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    return h_result;
}

void sample_greedy_async(const Tensor& logits, int32_t* d_result, cudaStream_t stream) {
    const int vocab_size = static_cast<int>(logits.shape[0]);
    const float* d_logits = static_cast<const float*>(logits.data);

    // Same multi-block argmax as sample_greedy(d_result), minus the readback:
    // the batched decode path gathers all sequences' tokens with one pinned
    // D2H + one sync (see sampling.h).
    auto* base = reinterpret_cast<char*>(d_result);
    auto* partial_vals = reinterpret_cast<float*>(base + sizeof(int32_t));
    auto* partial_idxs = reinterpret_cast<int32_t*>(base + sizeof(int32_t) + ARGMAX_NBLOCKS * sizeof(float));

    argmax_partial_kernel<<<ARGMAX_NBLOCKS, BLOCK_SIZE, 0, stream>>>(d_logits, vocab_size, partial_vals,
                                                                     partial_idxs);
    IMP_CUDA_CHECK_LAUNCH();
    argmax_reduce_kernel<<<1, WARP_SIZE, 0, stream>>>(partial_vals, partial_idxs, ARGMAX_NBLOCKS, d_result);
    IMP_CUDA_CHECK_LAUNCH();
}

// ===========================================================================
// Async (device-side) sampling — no host sync
// ===========================================================================

void sample_greedy_device(const Tensor& logits, int32_t* d_result, int32_t* h_mapped, cudaStream_t stream) {
    const int vocab_size = static_cast<int>(logits.shape[0]);
    const float* d_logits = static_cast<const float*>(logits.data);

    // Multi-block argmax: scratch lives right after d_result.
    auto* base = reinterpret_cast<char*>(d_result);
    auto* partial_vals = reinterpret_cast<float*>(base + sizeof(int32_t));
    auto* partial_idxs = reinterpret_cast<int32_t*>(base + sizeof(int32_t) + ARGMAX_NBLOCKS * sizeof(float));

    argmax_partial_kernel<<<ARGMAX_NBLOCKS, BLOCK_SIZE, 0, stream>>>(d_logits, vocab_size, partial_vals,
                                                                     partial_idxs);
    IMP_CUDA_CHECK_LAUNCH();
    argmax_reduce_kernel<<<1, WARP_SIZE, 0, stream>>>(partial_vals, partial_idxs, ARGMAX_NBLOCKS, d_result);
    IMP_CUDA_CHECK_LAUNCH();

    // Async copy to mapped pinned memory — no sync needed.
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(h_mapped, d_result, sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
}

}  // namespace imp
