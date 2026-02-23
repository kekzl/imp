#include "compute/sampling.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdio>
#include <cfloat>
#include <algorithm>

namespace imp {

static constexpr int BLOCK_SIZE = 256;
static constexpr int WARP_SIZE = 32;

// ============================================================================
// Greedy sampling (argmax)
// ============================================================================

// Warp-level argmax reduction: returns the (value, index) of the maximum
// across all lanes in the warp.
__device__ __forceinline__
void warp_argmax(float& val, int& idx) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float other_val = __shfl_xor_sync(0xFFFFFFFF, val, offset);
        int   other_idx = __shfl_xor_sync(0xFFFFFFFF, idx, offset);
        if (other_val > val || (other_val == val && other_idx < idx)) {
            val = other_val;
            idx = other_idx;
        }
    }
}

// Single-block argmax kernel.  Block of 256 threads scans the full vocab.
// Result (one int32) is written to d_result[0].
__global__ void argmax_kernel(const float* __restrict__ logits,
                              int vocab_size,
                              int32_t* __restrict__ d_result) {
    // Each thread finds its local max across its stripe of the vocab.
    float  local_max = -FLT_MAX;
    int    local_idx = 0;

    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        float v = logits[i];
        if (v > local_max || (v == local_max && i < local_idx)) {
            local_max = v;
            local_idx = i;
        }
    }

    // Warp reduction
    warp_argmax(local_max, local_idx);

    // Cross-warp reduction through shared memory
    constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;  // 8
    __shared__ float  s_val[NUM_WARPS];
    __shared__ int    s_idx[NUM_WARPS];

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    if (lane_id == 0) {
        s_val[warp_id] = local_max;
        s_idx[warp_id] = local_idx;
    }
    __syncthreads();

    // First warp reduces across all warps
    if (warp_id == 0) {
        float val = (lane_id < NUM_WARPS) ? s_val[lane_id] : -FLT_MAX;
        int   idx = (lane_id < NUM_WARPS) ? s_idx[lane_id] : 0;

        // Warp reduction among the first NUM_WARPS lanes
        #pragma unroll
        for (int offset = NUM_WARPS / 2; offset > 0; offset >>= 1) {
            float other_val = __shfl_xor_sync(0xFFFFFFFF, val, offset);
            int   other_idx = __shfl_xor_sync(0xFFFFFFFF, idx, offset);
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

int32_t sample_greedy(const Tensor& logits, cudaStream_t stream) {
    const int vocab_size = static_cast<int>(logits.shape[0]);
    const float* d_logits = static_cast<const float*>(logits.data);

    int32_t* d_result = nullptr;
    cudaMalloc(&d_result, sizeof(int32_t));

    argmax_kernel<<<1, BLOCK_SIZE, 0, stream>>>(d_logits, vocab_size, d_result);

    int32_t h_result = 0;
    cudaMemcpyAsync(&h_result, d_result, sizeof(int32_t),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    cudaFree(d_result);
    return h_result;
}

int32_t sample_greedy(const Tensor& logits, int32_t* d_result,
                      cudaStream_t stream) {
    const int vocab_size = static_cast<int>(logits.shape[0]);
    const float* d_logits = static_cast<const float*>(logits.data);

    argmax_kernel<<<1, BLOCK_SIZE, 0, stream>>>(d_logits, vocab_size, d_result);

    int32_t h_result = 0;
    cudaMemcpyAsync(&h_result, d_result, sizeof(int32_t),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    return h_result;
}

// ============================================================================
// Top-k / Top-p (nucleus) sampling with temperature
// ============================================================================

// Simple LCG random number generator for device code.
__device__ __forceinline__
unsigned int lcg_rand(unsigned int& state) {
    state = state * 1664525u + 1013904223u;
    return state;
}

// Convert LCG output to a float in [0, 1).
__device__ __forceinline__
float lcg_rand_float(unsigned int& state) {
    return static_cast<float>(lcg_rand(state)) / 4294967296.0f;
}

// Kernel 1: Apply temperature in-place.  logits[i] /= temperature.
__global__ void apply_temperature_kernel(float* __restrict__ logits,
                                         int vocab_size,
                                         float inv_temperature) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < vocab_size) {
        logits[idx] *= inv_temperature;
    }
}

// Kernel 2: Full top-k + top-p sampling in a single block.
//
// Strategy (single block, BLOCK_SIZE threads):
//   1. Find global max for numerical stability of softmax.
//   2. Compute exp(logits[i] - max) and the sum across all elements.
//   3. Each thread maintains a local top-k heap of (prob, index) pairs.
//   4. Merge the per-thread heaps into a shared top-k list.
//   5. Sort the shared top-k list by descending probability.
//   6. Cumulative sum; find top-p cutoff.
//   7. Renormalize remaining probabilities and sample using LCG RNG.
//
// For the per-thread local top-k, we use a simple insertion-sorted array
// (k is typically small, e.g. 2-50).
//
// Because top_k can vary at runtime, we cap it at a compile-time maximum
// and branch on the actual value.
static constexpr int MAX_TOP_K = 128;

__global__ void topk_topp_sample_kernel(
        const float* __restrict__ logits,
        int vocab_size,
        int top_k,
        float top_p,
        unsigned int seed,
        int32_t* __restrict__ d_result) {

    // --- Shared memory layout ---
    // We use dynamic shared memory to hold the merged top-k candidates.
    extern __shared__ char smem_raw[];

    // Partition shared memory:
    //   float  s_topk_val[top_k]    -- merged top-k probabilities
    //   int    s_topk_idx[top_k]    -- merged top-k vocab indices
    //   float  s_reduce[BLOCK_SIZE] -- scratch for reductions
    float* s_topk_val = reinterpret_cast<float*>(smem_raw);
    int*   s_topk_idx = reinterpret_cast<int*>(s_topk_val + top_k);
    float* s_reduce   = reinterpret_cast<float*>(s_topk_idx + top_k);
    // Additional single-value shared vars
    float* s_global_max = s_reduce + BLOCK_SIZE;  // 1 float
    float* s_global_sum = s_global_max + 1;       // 1 float

    const int tid = threadIdx.x;

    // ---- Step 1: Find global max (for softmax numerical stability) ----
    float local_max = -FLT_MAX;
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        float v = logits[i];
        if (v > local_max) local_max = v;
    }
    // Warp reduction
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float other = __shfl_xor_sync(0xFFFFFFFF, local_max, offset);
        if (other > local_max) local_max = other;
    }
    // Store per-warp result
    s_reduce[tid] = -FLT_MAX;
    __syncthreads();
    if (tid % WARP_SIZE == 0) s_reduce[tid / WARP_SIZE] = local_max;
    __syncthreads();
    // Thread 0 reduces across warps
    if (tid == 0) {
        float mx = -FLT_MAX;
        for (int w = 0; w < BLOCK_SIZE / WARP_SIZE; ++w) {
            if (s_reduce[w] > mx) mx = s_reduce[w];
        }
        s_global_max[0] = mx;
    }
    __syncthreads();
    float gmax = s_global_max[0];

    // ---- Step 2: Compute exp(logits[i] - gmax) and partial sum ----
    float local_sum = 0.0f;
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        float e = expf(logits[i] - gmax);
        local_sum += e;
    }
    // Warp reduction for sum
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        local_sum += __shfl_xor_sync(0xFFFFFFFF, local_sum, offset);
    }
    s_reduce[tid] = 0.0f;
    __syncthreads();
    if (tid % WARP_SIZE == 0) s_reduce[tid / WARP_SIZE] = local_sum;
    __syncthreads();
    if (tid == 0) {
        float sm = 0.0f;
        for (int w = 0; w < BLOCK_SIZE / WARP_SIZE; ++w) sm += s_reduce[w];
        s_global_sum[0] = sm;
    }
    __syncthreads();
    float inv_sum = 1.0f / s_global_sum[0];

    // ---- Step 3: Each thread maintains a local top-k min-heap ----
    // We store local_k pairs in registers (capped at MAX_TOP_K per thread,
    // but we only need top_k total, so each thread keeps top_k candidates
    // and we merge later).
    // For efficiency, limit per-thread candidates to top_k.
    int local_k = min(top_k, MAX_TOP_K);

    float local_vals[MAX_TOP_K];
    int   local_idxs[MAX_TOP_K];
    int   local_count = 0;
    float local_min_val = -FLT_MAX;  // min of current heap
    int   local_min_pos = 0;

    for (int i = tid; i < vocab_size; i += blockDim.x) {
        float prob = expf(logits[i] - gmax) * inv_sum;
        if (local_count < local_k) {
            local_vals[local_count] = prob;
            local_idxs[local_count] = i;
            local_count++;
            // Recompute min
            if (local_count == local_k) {
                local_min_val = local_vals[0];
                local_min_pos = 0;
                for (int j = 1; j < local_k; ++j) {
                    if (local_vals[j] < local_min_val) {
                        local_min_val = local_vals[j];
                        local_min_pos = j;
                    }
                }
            }
        } else if (prob > local_min_val) {
            // Replace the minimum
            local_vals[local_min_pos] = prob;
            local_idxs[local_min_pos] = i;
            // Recompute min
            local_min_val = local_vals[0];
            local_min_pos = 0;
            for (int j = 1; j < local_k; ++j) {
                if (local_vals[j] < local_min_val) {
                    local_min_val = local_vals[j];
                    local_min_pos = j;
                }
            }
        }
    }

    // ---- Step 4: Merge per-thread top-k into shared top-k ----
    // Initialize shared top-k with very small values
    if (tid < top_k) {
        s_topk_val[tid] = -1.0f;
        s_topk_idx[tid] = -1;
    }
    __syncthreads();

    // Each thread tries to insert its local candidates into the shared top-k.
    // We use atomicAdd-free approach: since we only need approximate top-k and
    // k is small, each thread serially tries to insert.
    // For correctness, we do a simple serial merge: threads go one at a time
    // through a lock, which is slow but k*BLOCK_SIZE is typically <32K ops.
    // Better approach: gather all local candidates to shared memory, then
    // do a parallel selection.

    // Gather: each thread writes its local_count candidates to a staging area.
    // staging area: BLOCK_SIZE * local_k entries in shared memory (too large).
    // Instead, we iterate: each thread contributes candidates in waves.

    // Practical approach: thread 0 gathers from all threads sequentially.
    // We store each thread's candidates to shared memory in batches.

    // Actually the simplest correct approach: have all threads write their
    // candidates to global memory scratch, then single-thread select top-k.
    // But that's wasteful. Instead, use a tournament approach:

    // Simple approach for small top_k: serialize through shared memory.
    // Each thread, in turn, merges its candidates into the shared top-k.
    // We process threads in parallel within warps using warp-level primitives.

    // Pragmatic: For correctness and simplicity, we use a shared-memory
    // min-value approach. Each thread iterates over its local candidates.
    // For each candidate, if it's larger than the current shared minimum,
    // the thread atomically tries to replace the minimum.
    // We use a spinlock per shared-memory slot -- too complex.

    // Final practical approach: Use a two-phase parallel merge.
    // Phase 1: Each of the 8 warps produces a warp-level top-k by
    //          reducing lane-local candidates across the warp.
    // Phase 2: Thread 0 merges the 8 warp-level top-k lists.

    // Phase 1: Warp-level merge
    // Each warp has 32 threads, each with up to local_k candidates.
    // Total per warp: up to 32 * local_k candidates; we want the top_k.
    // We iterate top_k times: in each iteration, every lane broadcasts its
    // best remaining candidate; the warp selects the global best; the lane
    // owning that candidate removes it.

    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;

    // Shared memory for warp-level top-k results: NUM_WARPS * top_k
    // We'll reuse s_reduce area which has BLOCK_SIZE floats + 2 extra.
    // That's 258 floats = 1032 bytes.  We need NUM_WARPS * top_k * (float + int).
    // For top_k = 50, NUM_WARPS = 8: 400 floats + 400 ints = 3200 bytes.
    // We already have s_topk_val/idx for final result.
    // Let's allocate per-warp results after s_global_sum.
    float* s_warp_vals = s_global_sum + 1;            // NUM_WARPS * top_k floats
    int*   s_warp_idxs = reinterpret_cast<int*>(s_warp_vals + NUM_WARPS * top_k);

    // Pointer to this warp's output area
    float* my_warp_vals = s_warp_vals + warp_id * top_k;
    int*   my_warp_idxs = s_warp_idxs + warp_id * top_k;

    // Each lane has local candidates in registers.
    // We'll iterate top_k times to extract the top_k from the warp.
    int my_ptr = 0;  // next candidate to offer from local sorted order

    // Pre-sort local candidates in descending order (insertion sort, local_k is small)
    for (int i = 0; i < local_count - 1; ++i) {
        for (int j = i + 1; j < local_count; ++j) {
            if (local_vals[j] > local_vals[i]) {
                float tv = local_vals[i]; local_vals[i] = local_vals[j]; local_vals[j] = tv;
                int ti = local_idxs[i]; local_idxs[i] = local_idxs[j]; local_idxs[j] = ti;
            }
        }
    }

    for (int k_iter = 0; k_iter < top_k; ++k_iter) {
        // Each lane offers its best remaining candidate
        float my_best_val = (my_ptr < local_count) ? local_vals[my_ptr] : -1.0f;
        int   my_best_idx = (my_ptr < local_count) ? local_idxs[my_ptr] : -1;

        // Warp reduction to find the max
        float best_val = my_best_val;
        int   best_idx = my_best_idx;
        int   best_lane = lane_id;

        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            float other_val  = __shfl_xor_sync(0xFFFFFFFF, best_val, offset);
            int   other_idx  = __shfl_xor_sync(0xFFFFFFFF, best_idx, offset);
            int   other_lane = __shfl_xor_sync(0xFFFFFFFF, best_lane, offset);
            if (other_val > best_val || (other_val == best_val && other_lane < best_lane)) {
                best_val  = other_val;
                best_idx  = other_idx;
                best_lane = other_lane;
            }
        }

        // The winning lane advances its pointer
        if (lane_id == best_lane && best_val >= 0.0f) {
            my_ptr++;
        }

        // Lane 0 writes the result for this warp
        if (lane_id == 0) {
            my_warp_vals[k_iter] = best_val;
            my_warp_idxs[k_iter] = best_idx;
        }
    }
    __syncthreads();

    // Phase 2: Thread 0 merges NUM_WARPS * top_k candidates into final top_k
    if (tid == 0) {
        // We have NUM_WARPS sorted lists, each of length top_k.
        // Merge using pointers (like k-way merge).
        int ptrs[NUM_WARPS];
        for (int w = 0; w < NUM_WARPS; ++w) ptrs[w] = 0;

        for (int k_iter = 0; k_iter < top_k; ++k_iter) {
            float best_val = -1.0f;
            int   best_idx = -1;
            int   best_warp = 0;
            for (int w = 0; w < NUM_WARPS; ++w) {
                if (ptrs[w] < top_k) {
                    float v = s_warp_vals[w * top_k + ptrs[w]];
                    if (v > best_val) {
                        best_val = v;
                        best_idx = s_warp_idxs[w * top_k + ptrs[w]];
                        best_warp = w;
                    }
                }
            }
            s_topk_val[k_iter] = best_val;
            s_topk_idx[k_iter] = best_idx;
            ptrs[best_warp]++;
        }
    }
    __syncthreads();

    // ---- Step 5: Top-p filtering and sampling (thread 0) ----
    if (tid == 0) {
        // s_topk_val is already sorted descending from the merge.
        // Compute cumulative sum and find top-p cutoff.
        float cumsum = 0.0f;
        int   cutoff = top_k;
        for (int i = 0; i < top_k; ++i) {
            cumsum += s_topk_val[i];
            if (cumsum >= top_p) {
                cutoff = i + 1;
                break;
            }
        }

        // Renormalize the remaining candidates
        float norm = 0.0f;
        for (int i = 0; i < cutoff; ++i) {
            norm += s_topk_val[i];
        }
        float inv_norm = (norm > 0.0f) ? (1.0f / norm) : 1.0f;

        // Sample using LCG RNG
        unsigned int rng_state = seed;
        float r = lcg_rand_float(rng_state);

        float acc = 0.0f;
        int chosen = s_topk_idx[0];  // fallback
        for (int i = 0; i < cutoff; ++i) {
            acc += s_topk_val[i] * inv_norm;
            if (r < acc) {
                chosen = s_topk_idx[i];
                break;
            }
        }

        d_result[0] = static_cast<int32_t>(chosen);
    }
}

int32_t sample_topk_topp(const Tensor& logits, int top_k, float top_p,
                         float temperature, unsigned int seed,
                         cudaStream_t stream) {
    const int vocab_size = static_cast<int>(logits.shape[0]);
    float* d_logits = static_cast<float*>(logits.data);

    if (top_k <= 0 || top_k > vocab_size) top_k = vocab_size;
    if (top_k > MAX_TOP_K) top_k = MAX_TOP_K;
    if (temperature <= 0.0f) temperature = 1.0f;
    float inv_temperature = 1.0f / temperature;

    int grid = (vocab_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    apply_temperature_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
        d_logits, vocab_size, inv_temperature);

    int32_t* d_result = nullptr;
    cudaMalloc(&d_result, sizeof(int32_t));

    constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
    size_t smem_bytes = static_cast<size_t>(top_k) * sizeof(float)
                      + static_cast<size_t>(top_k) * sizeof(int)
                      + BLOCK_SIZE * sizeof(float)
                      + 1 * sizeof(float)
                      + 1 * sizeof(float)
                      + NUM_WARPS * top_k * sizeof(float)
                      + NUM_WARPS * top_k * sizeof(int);

    topk_topp_sample_kernel<<<1, BLOCK_SIZE, smem_bytes, stream>>>(
        d_logits, vocab_size, top_k, top_p, seed, d_result);

    float temperature_restore = temperature;
    apply_temperature_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
        d_logits, vocab_size, temperature_restore);

    int32_t h_result = 0;
    cudaMemcpyAsync(&h_result, d_result, sizeof(int32_t),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    cudaFree(d_result);
    return h_result;
}

int32_t sample_topk_topp(const Tensor& logits, int top_k, float top_p,
                         float temperature, unsigned int seed,
                         int32_t* d_result,
                         cudaStream_t stream) {
    const int vocab_size = static_cast<int>(logits.shape[0]);
    float* d_logits = static_cast<float*>(logits.data);

    if (top_k <= 0 || top_k > vocab_size) top_k = vocab_size;
    if (top_k > MAX_TOP_K) top_k = MAX_TOP_K;
    if (temperature <= 0.0f) temperature = 1.0f;
    float inv_temperature = 1.0f / temperature;

    int grid = (vocab_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    apply_temperature_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
        d_logits, vocab_size, inv_temperature);

    constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
    size_t smem_bytes = static_cast<size_t>(top_k) * sizeof(float)
                      + static_cast<size_t>(top_k) * sizeof(int)
                      + BLOCK_SIZE * sizeof(float)
                      + 1 * sizeof(float)
                      + 1 * sizeof(float)
                      + NUM_WARPS * top_k * sizeof(float)
                      + NUM_WARPS * top_k * sizeof(int);

    topk_topp_sample_kernel<<<1, BLOCK_SIZE, smem_bytes, stream>>>(
        d_logits, vocab_size, top_k, top_p, seed, d_result);

    float temperature_restore = temperature;
    apply_temperature_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
        d_logits, vocab_size, temperature_restore);

    int32_t h_result = 0;
    cudaMemcpyAsync(&h_result, d_result, sizeof(int32_t),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    return h_result;
}

} // namespace imp
