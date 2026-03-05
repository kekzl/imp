#include "compute/sampling.h"
#include "core/logging.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdio>
#include <cfloat>
#include <algorithm>
#include <unordered_map>
#include <vector>

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

// Single-block argmax kernel (fallback for paths without pre-allocated scratch).
__global__ void argmax_kernel(const float* __restrict__ logits,
                              int vocab_size,
                              int32_t* __restrict__ d_result) {
    float  local_max = -FLT_MAX;
    int    local_idx = 0;

    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        float v = logits[i];
        if (v > local_max || (v == local_max && i < local_idx)) {
            local_max = v;
            local_idx = i;
        }
    }

    warp_argmax(local_max, local_idx);

    constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
    __shared__ float  s_val[NUM_WARPS];
    __shared__ int    s_idx[NUM_WARPS];

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    if (lane_id == 0) {
        s_val[warp_id] = local_max;
        s_idx[warp_id] = local_idx;
    }
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < NUM_WARPS) ? s_val[lane_id] : -FLT_MAX;
        int   idx = (lane_id < NUM_WARPS) ? s_idx[lane_id] : 0;

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

// Multi-block argmax: distributes work across ARGMAX_NBLOCKS blocks so all SMs
// participate.  The single-block kernel above uses 1 SM and takes ~190 us for
// vocab=152K; this version takes ~10 us.
//
// Scratch layout (passed as d_scratch, ARGMAX_SCRATCH_BYTES total):
//   float    partial_vals [ARGMAX_NBLOCKS]
//   int32_t  partial_idxs [ARGMAX_NBLOCKS]

// Phase 1: each block scans its stripe and writes its local max to partials.
__global__ void argmax_partial_kernel(
        const float* __restrict__ logits, int vocab_size,
        float* __restrict__ partial_vals, int32_t* __restrict__ partial_idxs) {
    float  local_max = -FLT_MAX;
    int    local_idx = 0;

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
    __shared__ float  s_val[NUM_WARPS];
    __shared__ int    s_idx[NUM_WARPS];

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    if (lane_id == 0) {
        s_val[warp_id] = local_max;
        s_idx[warp_id] = local_idx;
    }
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < NUM_WARPS) ? s_val[lane_id] : -FLT_MAX;
        int   idx = (lane_id < NUM_WARPS) ? s_idx[lane_id] : 0;

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
            partial_vals[blockIdx.x] = val;
            partial_idxs[blockIdx.x] = idx;
        }
    }
}

// Phase 2: single block reduces ARGMAX_NBLOCKS partial results.
__global__ void argmax_reduce_kernel(
        const float* __restrict__ partial_vals,
        const int32_t* __restrict__ partial_idxs,
        int n_blocks, int32_t* __restrict__ d_result) {
    float  local_max = -FLT_MAX;
    int    local_idx = 0;

    for (int i = threadIdx.x; i < n_blocks; i += blockDim.x) {
        float v = partial_vals[i];
        int   idx = partial_idxs[i];
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

    int32_t* d_result = nullptr;
    if (cudaMalloc(&d_result, sizeof(int32_t)) != cudaSuccess) {
        IMP_LOG_ERROR("sample_greedy: cudaMalloc failed");
        return 0;
    }

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

    // Use multi-block argmax: scratch lives right after d_result.
    // Layout: [result(4B)] [partial_vals(ARGMAX_NBLOCKS*4B)] [partial_idxs(ARGMAX_NBLOCKS*4B)]
    auto* base = reinterpret_cast<char*>(d_result);
    auto* partial_vals = reinterpret_cast<float*>(base + sizeof(int32_t));
    auto* partial_idxs = reinterpret_cast<int32_t*>(base + sizeof(int32_t) +
                                                     ARGMAX_NBLOCKS * sizeof(float));

    argmax_partial_kernel<<<ARGMAX_NBLOCKS, BLOCK_SIZE, 0, stream>>>(
        d_logits, vocab_size, partial_vals, partial_idxs);
    argmax_reduce_kernel<<<1, WARP_SIZE, 0, stream>>>(
        partial_vals, partial_idxs, ARGMAX_NBLOCKS, d_result);

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
    if (top_k > MAX_TOP_K) {
        IMP_LOG_WARN("top_k=%d exceeds MAX_TOP_K=%d, clamping", top_k, MAX_TOP_K);
        top_k = MAX_TOP_K;
    }
    if (temperature <= 0.0f) temperature = 1.0f;
    float inv_temperature = 1.0f / temperature;

    int grid = (vocab_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    apply_temperature_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
        d_logits, vocab_size, inv_temperature);

    int32_t* d_result = nullptr;
    if (cudaMalloc(&d_result, sizeof(int32_t)) != cudaSuccess) {
        IMP_LOG_ERROR("sample_topk_topp: cudaMalloc failed");
        return 0;
    }

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
    if (top_k > MAX_TOP_K) {
        IMP_LOG_WARN("top_k=%d exceeds MAX_TOP_K=%d, clamping", top_k, MAX_TOP_K);
        top_k = MAX_TOP_K;
    }
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

// ===========================================================================
// Async (device-side) sampling — no host sync
// ===========================================================================

void sample_greedy_device(const Tensor& logits, int32_t* d_result,
                          int32_t* h_mapped, cudaStream_t stream) {
    const int vocab_size = static_cast<int>(logits.shape[0]);
    const float* d_logits = static_cast<const float*>(logits.data);

    // Multi-block argmax: scratch lives right after d_result.
    auto* base = reinterpret_cast<char*>(d_result);
    auto* partial_vals = reinterpret_cast<float*>(base + sizeof(int32_t));
    auto* partial_idxs = reinterpret_cast<int32_t*>(base + sizeof(int32_t) +
                                                     ARGMAX_NBLOCKS * sizeof(float));

    argmax_partial_kernel<<<ARGMAX_NBLOCKS, BLOCK_SIZE, 0, stream>>>(
        d_logits, vocab_size, partial_vals, partial_idxs);
    argmax_reduce_kernel<<<1, WARP_SIZE, 0, stream>>>(
        partial_vals, partial_idxs, ARGMAX_NBLOCKS, d_result);

    // Async copy to mapped pinned memory — no sync needed.
    cudaMemcpyAsync(h_mapped, d_result, sizeof(int32_t),
                    cudaMemcpyDeviceToHost, stream);
}

void sample_topk_topp_device(const Tensor& logits, int top_k, float top_p,
                              float temperature, unsigned int seed,
                              int32_t* d_result, int32_t* h_mapped,
                              cudaStream_t stream) {
    const int vocab_size = static_cast<int>(logits.shape[0]);
    float* d_logits = static_cast<float*>(logits.data);

    if (top_k <= 0 || top_k > vocab_size) top_k = vocab_size;
    if (top_k > MAX_TOP_K) {
        IMP_LOG_WARN("top_k=%d exceeds MAX_TOP_K=%d, clamping", top_k, MAX_TOP_K);
        top_k = MAX_TOP_K;
    }
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

    // Async copy to mapped pinned memory — no sync needed.
    cudaMemcpyAsync(h_mapped, d_result, sizeof(int32_t),
                    cudaMemcpyDeviceToHost, stream);
}

// ===========================================================================
// Repetition / frequency / presence penalties
// ===========================================================================

// Kernel: for each token in history, adjust its logit.
// Uses atomics to handle tokens appearing multiple times.
// Strategy: first count occurrences, then apply penalties.
// For simplicity with small history, we iterate the history per thread.
__global__ void apply_penalties_kernel(
        float* __restrict__ logits,
        const int32_t* __restrict__ token_ids,
        int n_tokens,
        int vocab_size,
        float repetition_penalty,
        float frequency_penalty,
        float presence_penalty) {
    // Each thread handles one vocab entry
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= vocab_size) return;

    // Count occurrences of this token in history
    int count = 0;
    for (int i = 0; i < n_tokens; i++) {
        if (token_ids[i] == idx) count++;
    }
    if (count == 0) return;

    float logit = logits[idx];

    // Repetition penalty (multiplicative): divide positive, multiply negative
    if (repetition_penalty != 1.0f) {
        if (logit > 0.0f)
            logit /= repetition_penalty;
        else
            logit *= repetition_penalty;
    }

    // Frequency penalty (subtractive per-occurrence)
    logit -= frequency_penalty * static_cast<float>(count);

    // Presence penalty (subtractive binary)
    logit -= presence_penalty;

    logits[idx] = logit;
}

void apply_penalties(float* logits, int vocab_size,
                     const int32_t* token_ids, int n_tokens,
                     float repetition_penalty,
                     float frequency_penalty,
                     float presence_penalty,
                     cudaStream_t stream) {
    if (n_tokens == 0) return;
    if (repetition_penalty == 1.0f && frequency_penalty == 0.0f && presence_penalty == 0.0f)
        return;

    int blocks = (vocab_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    apply_penalties_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
        logits, token_ids, n_tokens, vocab_size,
        repetition_penalty, frequency_penalty, presence_penalty);
}

// ===========================================================================
// min_p filtering
// ===========================================================================

// Two-pass approach:
// Pass 1: find max logit (for softmax stability)
// Pass 2: set logits where exp(logit - max) < min_p to -inf
// This works because min_p threshold on probabilities = min_p * max_prob,
// and max_prob = exp(max_logit - max_logit) / sum = 1/sum, so the threshold
// in logit space is: logit < max_logit + log(min_p).

__global__ void find_max_logit_kernel(
        const float* __restrict__ logits,
        int vocab_size,
        float* __restrict__ d_max) {
    __shared__ float s_max[BLOCK_SIZE / WARP_SIZE];
    float local_max = -FLT_MAX;
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        float v = logits[i];
        if (v > local_max) local_max = v;
    }
    // Warp reduction
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float other = __shfl_xor_sync(0xFFFFFFFF, local_max, offset);
        if (other > local_max) local_max = other;
    }
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    if (lane_id == 0) s_max[warp_id] = local_max;
    __syncthreads();
    if (threadIdx.x == 0) {
        float mx = -FLT_MAX;
        for (int w = 0; w < BLOCK_SIZE / WARP_SIZE; w++) {
            if (s_max[w] > mx) mx = s_max[w];
        }
        d_max[0] = mx;
    }
}

__global__ void apply_min_p_kernel(
        float* __restrict__ logits,
        int vocab_size,
        float threshold_logit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= vocab_size) return;
    if (logits[idx] < threshold_logit)
        logits[idx] = -FLT_MAX;
}

void apply_min_p(float* logits, int vocab_size, float min_p,
                 cudaStream_t stream) {
    if (min_p <= 0.0f) return;

    // Allocate temp buffer for max value
    float* d_max = nullptr;
    if (cudaMalloc(&d_max, sizeof(float)) != cudaSuccess) {
        IMP_LOG_ERROR("apply_min_p: cudaMalloc failed");
        return;
    }

    find_max_logit_kernel<<<1, BLOCK_SIZE, 0, stream>>>(
        logits, vocab_size, d_max);

    // Read max back to compute threshold
    float h_max = 0.0f;
    cudaMemcpyAsync(&h_max, d_max, sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaFree(d_max);

    // threshold = max_logit + log(min_p)
    // Tokens with logit < threshold get -inf
    float threshold = h_max + logf(min_p);

    int blocks = (vocab_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    apply_min_p_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
        logits, vocab_size, threshold);
}

// ===========================================================================
// DRY (Don't Repeat Yourself) repetition penalty
// ===========================================================================

// Sparse penalty application kernel: subtracts penalty from each listed token.
__global__ void apply_dry_sparse_kernel(
        float* __restrict__ logits,
        const int32_t* __restrict__ penalty_tokens,
        const float* __restrict__ penalty_values,
        int n_penalties) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_penalties) {
        logits[penalty_tokens[idx]] -= penalty_values[idx];
    }
}

void apply_dry_penalty(float* d_logits, int vocab_size,
                       const int32_t* host_token_ids, int n_tokens,
                       float multiplier, float base,
                       int allowed_length, int penalty_last_n,
                       cudaStream_t stream) {
    if (multiplier <= 0.0f || n_tokens < 2) return;

    int search_start = (penalty_last_n > 0)
        ? std::max(0, n_tokens - penalty_last_n) : 0;

    // CPU: scan history for suffix matches, compute max match length per token
    std::unordered_map<int32_t, int> max_match;

    for (int pos = search_start; pos < n_tokens; pos++) {
        // Match suffix ending at (pos-1) with suffix ending at (n_tokens-1)
        int match_len = 0;
        int a = pos - 1;
        int b = n_tokens - 1;
        while (a >= search_start && b >= 0 &&
               host_token_ids[a] == host_token_ids[b]) {
            match_len++;
            a--;
            b--;
        }

        if (match_len > allowed_length) {
            int32_t token = host_token_ids[pos];
            if (token >= 0 && token < vocab_size) {
                auto it = max_match.find(token);
                if (it == max_match.end() || match_len > it->second)
                    max_match[token] = match_len;
            }
        }
    }

    if (max_match.empty()) return;

    // Build sparse penalty arrays
    int n = static_cast<int>(max_match.size());
    std::vector<int32_t> h_tokens(n);
    std::vector<float> h_values(n);
    int i = 0;
    for (auto& [tok, ml] : max_match) {
        h_tokens[i] = tok;
        h_values[i] = multiplier * std::pow(base,
            static_cast<float>(ml - allowed_length));
        i++;
    }

    // Upload to GPU and apply
    int32_t* d_tokens = nullptr;
    float* d_values = nullptr;
    if (cudaMalloc(&d_tokens, n * sizeof(int32_t)) != cudaSuccess ||
        cudaMalloc(&d_values, n * sizeof(float)) != cudaSuccess) {
        IMP_LOG_ERROR("apply_dry_penalty: cudaMalloc failed");
        if (d_tokens) cudaFree(d_tokens);
        if (d_values) cudaFree(d_values);
        return;
    }
    cudaMemcpyAsync(d_tokens, h_tokens.data(), n * sizeof(int32_t),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_values, h_values.data(), n * sizeof(float),
                    cudaMemcpyHostToDevice, stream);

    int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    apply_dry_sparse_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
        d_logits, d_tokens, d_values, n);

    cudaFreeAsync(d_tokens, stream);
    cudaFreeAsync(d_values, stream);
}

// ===========================================================================
// Typical-P (locally typical) filtering
// ===========================================================================

// Single-block kernel: computes entropy, deviation histogram, finds threshold,
// and filters tokens with deviation > threshold.
static constexpr int TYPICAL_NBUCKETS = 256;

__global__ void apply_typical_p_kernel(
        float* __restrict__ logits,
        int vocab_size,
        float typical_p) {

    constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
    __shared__ float s_warp[NUM_WARPS];
    __shared__ float s_max, s_sum, s_entropy, s_max_dev, s_threshold;
    __shared__ float s_buckets[TYPICAL_NBUCKETS];

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    // --- Pass 1: max logit ---
    float local_max = -FLT_MAX;
    for (int i = tid; i < vocab_size; i += blockDim.x)
        local_max = fmaxf(local_max, logits[i]);
    #pragma unroll
    for (int o = WARP_SIZE / 2; o > 0; o >>= 1) {
        float other = __shfl_xor_sync(0xFFFFFFFF, local_max, o);
        local_max = fmaxf(local_max, other);
    }
    if (lane_id == 0) s_warp[warp_id] = local_max;
    __syncthreads();
    if (tid == 0) {
        float mx = -FLT_MAX;
        for (int w = 0; w < NUM_WARPS; w++) mx = fmaxf(mx, s_warp[w]);
        s_max = mx;
    }
    __syncthreads();
    float gmax = s_max;

    // --- Pass 2: sum_exp ---
    float local_sum = 0.0f;
    for (int i = tid; i < vocab_size; i += blockDim.x)
        local_sum += expf(logits[i] - gmax);
    #pragma unroll
    for (int o = WARP_SIZE / 2; o > 0; o >>= 1)
        local_sum += __shfl_xor_sync(0xFFFFFFFF, local_sum, o);
    if (lane_id == 0) s_warp[warp_id] = local_sum;
    __syncthreads();
    if (tid == 0) {
        float sm = 0.0f;
        for (int w = 0; w < NUM_WARPS; w++) sm += s_warp[w];
        s_sum = sm;
    }
    __syncthreads();

    float sum_exp = s_sum;
    float log_sum_exp = gmax + logf(sum_exp);
    float inv_log2 = 1.4426950408889634f; // 1/ln(2)

    // --- Pass 3: entropy H = -sum(p_i * log2(p_i)) ---
    float local_ent = 0.0f;
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        float p = expf(logits[i] - gmax) / sum_exp;
        if (p > 1e-30f) local_ent -= p * log2f(p);
    }
    #pragma unroll
    for (int o = WARP_SIZE / 2; o > 0; o >>= 1)
        local_ent += __shfl_xor_sync(0xFFFFFFFF, local_ent, o);
    if (lane_id == 0) s_warp[warp_id] = local_ent;
    __syncthreads();
    if (tid == 0) {
        float e = 0.0f;
        for (int w = 0; w < NUM_WARPS; w++) e += s_warp[w];
        s_entropy = e;
    }
    __syncthreads();
    float H = s_entropy;

    // --- Pass 4: max deviation ---
    float local_md = 0.0f;
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        float surprise = -(logits[i] - log_sum_exp) * inv_log2;
        local_md = fmaxf(local_md, fabsf(surprise - H));
    }
    #pragma unroll
    for (int o = WARP_SIZE / 2; o > 0; o >>= 1) {
        float other = __shfl_xor_sync(0xFFFFFFFF, local_md, o);
        local_md = fmaxf(local_md, other);
    }
    if (lane_id == 0) s_warp[warp_id] = local_md;
    __syncthreads();
    if (tid == 0) {
        float md = 0.0f;
        for (int w = 0; w < NUM_WARPS; w++) md = fmaxf(md, s_warp[w]);
        s_max_dev = md;
    }
    __syncthreads();

    // --- Pass 5: build deviation histogram ---
    // Initialize buckets
    for (int b = tid; b < TYPICAL_NBUCKETS; b += blockDim.x)
        s_buckets[b] = 0.0f;
    __syncthreads();

    float bucket_scale = (s_max_dev > 1e-8f)
        ? (static_cast<float>(TYPICAL_NBUCKETS) / s_max_dev) : 1.0f;

    for (int i = tid; i < vocab_size; i += blockDim.x) {
        float surprise = -(logits[i] - log_sum_exp) * inv_log2;
        float dev = fabsf(surprise - H);
        int bucket = min(static_cast<int>(dev * bucket_scale), TYPICAL_NBUCKETS - 1);
        float p = expf(logits[i] - gmax) / sum_exp;
        atomicAdd(&s_buckets[bucket], p);
    }
    __syncthreads();

    // --- Pass 6: scan histogram to find threshold (thread 0) ---
    if (tid == 0) {
        float cum = 0.0f;
        s_threshold = s_max_dev + 1.0f; // default: keep all
        for (int b = 0; b < TYPICAL_NBUCKETS; b++) {
            cum += s_buckets[b];
            if (cum >= typical_p) {
                // Threshold = upper bound of this bucket
                s_threshold = static_cast<float>(b + 1) / bucket_scale;
                break;
            }
        }
    }
    __syncthreads();

    // --- Pass 7: filter tokens with deviation > threshold ---
    float thr = s_threshold;
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        float surprise = -(logits[i] - log_sum_exp) * inv_log2;
        float dev = fabsf(surprise - H);
        if (dev > thr) logits[i] = -FLT_MAX;
    }
}

void apply_typical_p(float* logits, int vocab_size, float typical_p,
                     cudaStream_t stream) {
    if (typical_p <= 0.0f || typical_p >= 1.0f) return;

    apply_typical_p_kernel<<<1, BLOCK_SIZE, 0, stream>>>(
        logits, vocab_size, typical_p);
}

// ===========================================================================
// Mirostat v2 sampling
// ===========================================================================

// Single-block kernel: computes log-sum-exp, filters by surprise threshold,
// samples from filtered set, and outputs token + surprise.
__global__ void mirostat_v2_sample_kernel(
        const float* __restrict__ logits,
        int vocab_size,
        float mu,
        unsigned int seed,
        int32_t* __restrict__ d_result,
        float* __restrict__ d_surprise) {

    constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
    __shared__ float s_warp[NUM_WARPS];
    __shared__ float s_max;
    __shared__ float s_sum;
    __shared__ float s_fsum;

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    // --- Step 1: Find max logit ---
    float local_max = -FLT_MAX;
    for (int i = tid; i < vocab_size; i += blockDim.x)
        local_max = fmaxf(local_max, logits[i]);

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float other = __shfl_xor_sync(0xFFFFFFFF, local_max, offset);
        local_max = fmaxf(local_max, other);
    }
    if (lane_id == 0) s_warp[warp_id] = local_max;
    __syncthreads();

    if (tid == 0) {
        float mx = -FLT_MAX;
        for (int w = 0; w < NUM_WARPS; w++) mx = fmaxf(mx, s_warp[w]);
        s_max = mx;
    }
    __syncthreads();
    float gmax = s_max;

    // --- Step 2: Compute sum of exp(logit - max) ---
    float local_sum = 0.0f;
    for (int i = tid; i < vocab_size; i += blockDim.x)
        local_sum += expf(logits[i] - gmax);

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        local_sum += __shfl_xor_sync(0xFFFFFFFF, local_sum, offset);
    if (lane_id == 0) s_warp[warp_id] = local_sum;
    __syncthreads();

    if (tid == 0) {
        float sm = 0.0f;
        for (int w = 0; w < NUM_WARPS; w++) sm += s_warp[w];
        s_sum = sm;
    }
    __syncthreads();

    // Mirostat threshold: keep tokens with surprise ≤ mu
    // surprise_i = -log2(p_i), p_i = exp(l_i - max) / sum_exp
    // surprise_i ≤ mu  ⟺  l_i ≥ max + log(sum_exp) - mu * ln(2)
    float log_sum_exp = gmax + logf(s_sum);
    float threshold = log_sum_exp - mu * 0.6931471805599453f;

    // --- Step 3: Compute filtered probability sum ---
    float local_fsum = 0.0f;
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        if (logits[i] >= threshold)
            local_fsum += expf(logits[i] - gmax);
    }

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        local_fsum += __shfl_xor_sync(0xFFFFFFFF, local_fsum, offset);
    if (lane_id == 0) s_warp[warp_id] = local_fsum;
    __syncthreads();

    if (tid == 0) {
        float fs = 0.0f;
        for (int w = 0; w < NUM_WARPS; w++) fs += s_warp[w];
        // Fallback: if no tokens pass threshold, use entire distribution
        s_fsum = (fs > 0.0f) ? fs : s_sum;
    }
    __syncthreads();

    float fsum = s_fsum;
    bool use_threshold = (fsum < s_sum * 0.9999f);

    // --- Step 4: Sample from filtered distribution ---
    // Thread 0 scans through vocab, accumulating filtered probabilities.
    if (tid == 0) {
        float inv_fsum = 1.0f / fsum;
        unsigned int rng = seed;
        float r = lcg_rand_float(rng);

        float acc = 0.0f;
        int chosen = 0;
        bool found = false;

        for (int i = 0; i < vocab_size; i++) {
            if (!use_threshold || logits[i] >= threshold) {
                float p = expf(logits[i] - gmax) * inv_fsum;
                acc += p;
                if (r < acc) {
                    chosen = i;
                    found = true;
                    break;
                }
            }
        }

        // Fallback: pick highest-logit token
        if (!found) {
            float best = -FLT_MAX;
            for (int i = 0; i < vocab_size; i++) {
                if (logits[i] > best) { best = logits[i]; chosen = i; }
            }
        }

        // Compute surprise using original (unfiltered) probability
        float chosen_prob = expf(logits[chosen] - gmax) / s_sum;
        float surprise = -log2f(fmaxf(chosen_prob, 1e-30f));

        d_result[0] = chosen;
        d_surprise[0] = surprise;
    }
}

static int32_t sample_mirostat_v2_impl(
        const Tensor& logits, float temperature,
        float tau, float eta, float* mu,
        unsigned int seed, int32_t* d_result, bool owns_result,
        cudaStream_t stream) {

    const int vocab_size = static_cast<int>(logits.shape[0]);
    float* d_logits = static_cast<float*>(logits.data);

    // Apply temperature
    if (temperature > 0.0f && temperature != 1.0f) {
        float inv_temp = 1.0f / temperature;
        int grid = (vocab_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        apply_temperature_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
            d_logits, vocab_size, inv_temp);
    }

    // Surprise value stored right after the token result
    float* d_surprise = reinterpret_cast<float*>(d_result + 1);

    mirostat_v2_sample_kernel<<<1, BLOCK_SIZE, 0, stream>>>(
        d_logits, vocab_size, *mu, seed, d_result, d_surprise);

    // Restore temperature
    if (temperature > 0.0f && temperature != 1.0f) {
        int grid = (vocab_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        apply_temperature_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
            d_logits, vocab_size, temperature);
    }

    // Read results
    int32_t h_result = 0;
    float h_surprise = 0.0f;
    cudaMemcpyAsync(&h_result, d_result, sizeof(int32_t),
                    cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&h_surprise, d_surprise, sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    if (owns_result) cudaFree(d_result);

    // Update mu: mu = mu - eta * (surprise - tau)
    *mu = *mu - eta * (h_surprise - tau);

    return h_result;
}

int32_t sample_mirostat_v2(const Tensor& logits, float temperature,
                           float tau, float eta, float* mu,
                           unsigned int seed, cudaStream_t stream) {
    // Allocate temp buffer: 4 bytes for token + 4 bytes for surprise
    int32_t* d_result = nullptr;
    if (cudaMalloc(&d_result, 2 * sizeof(int32_t)) != cudaSuccess) {
        IMP_LOG_ERROR("sample_mirostat_v2: cudaMalloc failed");
        return 0;
    }
    return sample_mirostat_v2_impl(logits, temperature, tau, eta, mu,
                                    seed, d_result, true, stream);
}

int32_t sample_mirostat_v2(const Tensor& logits, float temperature,
                           float tau, float eta, float* mu,
                           unsigned int seed, int32_t* d_result,
                           cudaStream_t stream) {
    return sample_mirostat_v2_impl(logits, temperature, tau, eta, mu,
                                    seed, d_result, false, stream);
}

// ============================================================================
// CPU-side logprob computation
// ============================================================================

void compute_logprobs_cpu(const float* logits, int vocab_size,
                          int32_t sampled_token, int top_n,
                          LogprobResult* out) {
    // 1. Find max for numerical stability
    float max_val = -FLT_MAX;
    for (int i = 0; i < vocab_size; i++) {
        if (logits[i] > max_val) max_val = logits[i];
    }

    // 2. Compute log-sum-exp
    double sum_exp = 0.0;
    for (int i = 0; i < vocab_size; i++) {
        sum_exp += std::exp(static_cast<double>(logits[i]) - static_cast<double>(max_val));
    }
    float log_sum_exp = static_cast<float>(std::log(sum_exp)) + max_val;

    // 3. Extract sampled token's logprob
    out->sampled_logprob = logits[sampled_token] - log_sum_exp;

    // 4. Top-N via partial sort with min-heap
    out->top.clear();
    if (top_n <= 0) return;

    // Use a simple approach: collect all (logprob, token) and partial sort
    // For vocab ~150K and top_n <= 20, this is fast enough (~0.3ms)
    struct Entry {
        float logprob;
        int32_t token;
        bool operator<(const Entry& o) const { return logprob > o.logprob; }  // max-heap order
    };

    // Min-heap of size top_n to track the top-N largest
    std::vector<Entry> heap;
    heap.reserve(top_n + 1);

    for (int i = 0; i < vocab_size; i++) {
        float lp = logits[i] - log_sum_exp;
        if (static_cast<int>(heap.size()) < top_n) {
            heap.push_back({lp, i});
            std::push_heap(heap.begin(), heap.end(), [](const Entry& a, const Entry& b) {
                return a.logprob > b.logprob;  // min-heap: smallest logprob at top
            });
        } else if (lp > heap[0].logprob) {
            std::pop_heap(heap.begin(), heap.end(), [](const Entry& a, const Entry& b) {
                return a.logprob > b.logprob;
            });
            heap.back() = {lp, i};
            std::push_heap(heap.begin(), heap.end(), [](const Entry& a, const Entry& b) {
                return a.logprob > b.logprob;
            });
        }
    }

    // Sort descending by logprob
    std::sort(heap.begin(), heap.end(), [](const Entry& a, const Entry& b) {
        return a.logprob > b.logprob;
    });

    out->top.reserve(heap.size());
    for (const auto& e : heap) {
        out->top.push_back({e.token, e.logprob});
    }
}

} // namespace imp
