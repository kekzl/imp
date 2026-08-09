#include "compute/sampling.h"
#include "compute/sampling_internal.cuh"
#include "compute/warp_reduce.cuh"
#include "core/logging.h"
#include "memory/engine_arena.h"
#include "runtime/process_diag.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cub/cub.cuh>
#include <cub/device/device_topk.cuh>
#include <cmath>
#include <cstdio>
#include <cfloat>
#include <algorithm>
#include <vector>

namespace imp {

// ============================================================================
// Top-k / Top-p (nucleus) sampling with temperature
// ============================================================================

// ============================================================================
// Multi-block top-k + top-p sampling (two phases).
//
// The original single-block kernel ran <<<1, BLOCK_SIZE>>>, using 1 of 170 SMs,
// and scanned the full vocab (~150k) three times — ~737 us/call, the #1 GPU
// consumer in batched server decode (profiled 2026-06-23). This splits the
// full-vocab work across SAMPLE_NBLOCKS blocks.
//
// Phase 1 (multi-block): each block scans a strided vocab subset and emits
//   block_max, block_sum (= sum exp((logit-block_max)*invT)), and the block's
//   top_k *logits* (sorted desc) into candidate scratch. Candidates store
//   logits, not probabilities, so the global softmax can be applied in phase 2.
// Phase 2 (single block): merges block partials into the global max/sum via the
//   online-softmax rescale, k-way merges the SAMPLE_NBLOCKS candidate lists into
//   the global top_k, converts to probabilities, applies top-p and samples with
//   the same LCG as before. Distribution-identical to the old kernel (not
//   bit-identical: reduction order differs).
// ============================================================================

// Phase 1: per-block max, sum, and top_k logit candidates over a strided subset.
// Body shared between the single-row kernel and the row-parallel batched
// wrapper (grid.y = row) — blockIdx.x / gridDim.x usage is identical, so
// per-row results are bit-identical across the two launch shapes.
__device__ __forceinline__ void topk_partial_body(const float* __restrict__ logits, int vocab_size,
                                                  int top_k, float inv_temperature,
                                                  float* __restrict__ block_max_out,
                                                  float* __restrict__ block_sum_out,
                                                  float* __restrict__ cand_val_out,
                                                  int* __restrict__ cand_idx_out) {
    extern __shared__ char smem_raw[];
    constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
    float* s_reduce = reinterpret_cast<float*>(smem_raw);  // BLOCK_SIZE
    float* s_gmax = s_reduce + BLOCK_SIZE;                 // 1
    float* s_gsum = s_gmax + 1;                            // 1
    float* s_warp_vals = s_gsum + 1;                       // NUM_WARPS * top_k
    int* s_warp_idxs = reinterpret_cast<int*>(s_warp_vals + NUM_WARPS * top_k);

    const int tid = threadIdx.x;
    const int gstride = blockDim.x * gridDim.x;
    const int gstart = blockIdx.x * blockDim.x + tid;

    // ---- block max over strided subset ----
    float local_max = -FLT_MAX;
    for (int i = gstart; i < vocab_size; i += gstride) {
        float v = logits[i];
        if (v > local_max)
            local_max = v;
    }
    local_max = warp_reduce_max(local_max);
    s_reduce[tid] = -FLT_MAX;
    __syncthreads();
    if (tid % WARP_SIZE == 0)
        s_reduce[tid / WARP_SIZE] = local_max;
    __syncthreads();
    if (tid == 0) {
        float mx = -FLT_MAX;
        for (int w = 0; w < NUM_WARPS; ++w)
            if (s_reduce[w] > mx)
                mx = s_reduce[w];
        s_gmax[0] = mx;
    }
    __syncthreads();
    float bmax = s_gmax[0];

    // ---- block sum exp((logit-bmax)*invT) ----
    float local_sum = 0.0f;
    if (bmax > -FLT_MAX) {
        for (int i = gstart; i < vocab_size; i += gstride)
            local_sum += expf((logits[i] - bmax) * inv_temperature);
    }
    local_sum = warp_reduce_sum(local_sum);
    s_reduce[tid] = 0.0f;
    __syncthreads();
    if (tid % WARP_SIZE == 0)
        s_reduce[tid / WARP_SIZE] = local_sum;
    __syncthreads();
    if (tid == 0) {
        float sm = 0.0f;
        for (int w = 0; w < NUM_WARPS; ++w)
            sm += s_reduce[w];
        block_max_out[blockIdx.x] = bmax;
        block_sum_out[blockIdx.x] = sm;
    }

    // ---- per-thread top_k by logit over the subset ----
    int local_k = min(top_k, SAMPLE_MAX_TOP_K);
    float local_vals[SAMPLE_MAX_TOP_K];
    int local_idxs[SAMPLE_MAX_TOP_K];
    int local_count = 0;
    float local_min_val = -FLT_MAX;
    int local_min_pos = 0;
    for (int i = gstart; i < vocab_size; i += gstride) {
        float v = logits[i];
        if (local_count < local_k) {
            local_vals[local_count] = v;
            local_idxs[local_count] = i;
            local_count++;
            if (local_count == local_k) {
                local_min_val = local_vals[0];
                local_min_pos = 0;
                for (int j = 1; j < local_k; ++j)
                    if (local_vals[j] < local_min_val) {
                        local_min_val = local_vals[j];
                        local_min_pos = j;
                    }
            }
        } else if (v > local_min_val) {
            local_vals[local_min_pos] = v;
            local_idxs[local_min_pos] = i;
            local_min_val = local_vals[0];
            local_min_pos = 0;
            for (int j = 1; j < local_k; ++j)
                if (local_vals[j] < local_min_val) {
                    local_min_val = local_vals[j];
                    local_min_pos = j;
                }
        }
    }
    // reduce per-thread candidates into the block's top_k (sorted desc) → scratch
    block_reduce_topk(local_vals, local_idxs, local_count, top_k, s_warp_vals, s_warp_idxs,
                      cand_val_out + static_cast<size_t>(blockIdx.x) * top_k,
                      cand_idx_out + static_cast<size_t>(blockIdx.x) * top_k);
}

__global__ void topk_partial_kernel(const float* __restrict__ logits, int vocab_size, int top_k,
                                    float inv_temperature, float* __restrict__ block_max_out,
                                    float* __restrict__ block_sum_out, float* __restrict__ cand_val_out,
                                    int* __restrict__ cand_idx_out) {
    topk_partial_body(logits, vocab_size, top_k, inv_temperature, block_max_out, block_sum_out,
                      cand_val_out, cand_idx_out);
}

// Row-parallel wrapper: blockIdx.y selects the row; per-row scratch is carved
// from the row's SAMPLE_SCRATCH_BYTES slot exactly like
// launch_topk_topp_multiblock does.
__global__ void topk_partial_rows_kernel(const TopkRowArgs* __restrict__ rows, int vocab_size) {
    const TopkRowArgs a = rows[blockIdx.y];
    char* base = reinterpret_cast<char*>(a.d_result);
    float* block_max = reinterpret_cast<float*>(base + sizeof(int32_t));
    float* block_sum = block_max + SAMPLE_NBLOCKS;
    float* cand_val = block_sum + SAMPLE_NBLOCKS;
    int* cand_idx = reinterpret_cast<int*>(cand_val + static_cast<size_t>(SAMPLE_NBLOCKS) * a.top_k);
    topk_partial_body(a.logits, vocab_size, a.top_k, a.inv_temperature, block_max, block_sum, cand_val,
                      cand_idx);
}

// Phase 2: merge the per-block candidate lists into the global top_k, apply
// top-p and sample. All threads cooperate to select the global top_k from the
// candidate pool (block_reduce_topk over the SAMPLE_NBLOCKS*top_k candidates read
// straight from global — coalesced, no big smem staging); only the final
// top-p/sample over top_k entries is serial. Runs inside graph capture.
__device__ __forceinline__ void topk_finalize_body(int top_k, float top_p, float inv_temperature,
                                                   unsigned int seed, int n_blocks,
                                                   const float* __restrict__ block_max_in,
                                                   const float* __restrict__ block_sum_in,
                                                   const float* __restrict__ cand_val_in,
                                                   const int* __restrict__ cand_idx_in,
                                                   int32_t* __restrict__ d_result) {
    extern __shared__ char smem_raw[];
    constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
    float* s_warp_vals = reinterpret_cast<float*>(smem_raw);  // NUM_WARPS * top_k
    int* s_warp_idxs = reinterpret_cast<int*>(s_warp_vals + NUM_WARPS * top_k);
    float* s_val = reinterpret_cast<float*>(s_warp_idxs + NUM_WARPS * top_k);  // top_k
    int* s_idx = reinterpret_cast<int*>(s_val + top_k);                        // top_k

    const int tid = threadIdx.x;
    const int n_cand = n_blocks * top_k;

    // each thread builds a local top_k over its slice of the candidate pool
    int local_k = min(top_k, SAMPLE_MAX_TOP_K);
    float local_vals[SAMPLE_MAX_TOP_K];
    int local_idxs[SAMPLE_MAX_TOP_K];
    int local_count = 0;
    float local_min_val = -FLT_MAX;
    int local_min_pos = 0;
    for (int i = tid; i < n_cand; i += blockDim.x) {
        int idx = cand_idx_in[i];
        if (idx < 0)
            continue;
        float v = cand_val_in[i];
        if (local_count < local_k) {
            local_vals[local_count] = v;
            local_idxs[local_count] = idx;
            local_count++;
            if (local_count == local_k) {
                local_min_val = local_vals[0];
                local_min_pos = 0;
                for (int j = 1; j < local_k; ++j)
                    if (local_vals[j] < local_min_val) {
                        local_min_val = local_vals[j];
                        local_min_pos = j;
                    }
            }
        } else if (v > local_min_val) {
            local_vals[local_min_pos] = v;
            local_idxs[local_min_pos] = idx;
            local_min_val = local_vals[0];
            local_min_pos = 0;
            for (int j = 1; j < local_k; ++j)
                if (local_vals[j] < local_min_val) {
                    local_min_val = local_vals[j];
                    local_min_pos = j;
                }
        }
    }
    block_reduce_topk(local_vals, local_idxs, local_count, top_k, s_warp_vals, s_warp_idxs, s_val, s_idx);

    if (tid != 0)
        return;

    // global max + online-softmax-merged global sum (block partials are tiny)
    float gmax = -FLT_MAX;
    for (int b = 0; b < n_blocks; ++b)
        if (block_max_in[b] > gmax)
            gmax = block_max_in[b];
    float gsum = 0.0f;
    for (int b = 0; b < n_blocks; ++b) {
        float bm = block_max_in[b];
        if (bm > -FLT_MAX)
            gsum += block_sum_in[b] * expf((bm - gmax) * inv_temperature);
    }
    float inv_sum = (gsum > 0.0f) ? (1.0f / gsum) : 1.0f;

    // logits -> probabilities (global softmax), top-p cutoff, renormalize, sample
    for (int i = 0; i < top_k; ++i) {
        if (s_idx[i] >= 0)
            s_val[i] = expf((s_val[i] - gmax) * inv_temperature) * inv_sum;
        else
            s_val[i] = 0.0f;
    }
    float cumsum = 0.0f;
    int cutoff = top_k;
    for (int i = 0; i < top_k; ++i) {
        cumsum += s_val[i];
        if (cumsum >= top_p) {
            cutoff = i + 1;
            break;
        }
    }
    float norm = 0.0f;
    for (int i = 0; i < cutoff; ++i)
        norm += s_val[i];
    float inv_norm = (norm > 0.0f) ? (1.0f / norm) : 1.0f;

    unsigned int rng_state = sampler_seed_scramble(seed);
    float r = lcg_rand_float(rng_state);
    float acc = 0.0f;
    int chosen = s_idx[0];
    for (int i = 0; i < cutoff; ++i) {
        acc += s_val[i] * inv_norm;
        if (r < acc) {
            chosen = s_idx[i];
            break;
        }
    }
    d_result[0] = static_cast<int32_t>(chosen);
}

__global__ void topk_finalize_kernel(int top_k, float top_p, float inv_temperature, unsigned int seed,
                                     int n_blocks, const float* __restrict__ block_max_in,
                                     const float* __restrict__ block_sum_in,
                                     const float* __restrict__ cand_val_in,
                                     const int* __restrict__ cand_idx_in, int32_t* __restrict__ d_result) {
    topk_finalize_body(top_k, top_p, inv_temperature, seed, n_blocks, block_max_in, block_sum_in,
                       cand_val_in, cand_idx_in, d_result);
}

// Row-parallel wrapper: blockIdx.x selects the row (one block per row — the
// n_rows serialized <<<1>>> finalize launches become one launch).
__global__ void topk_finalize_rows_kernel(const TopkRowArgs* __restrict__ rows, int n_blocks) {
    const TopkRowArgs a = rows[blockIdx.x];
    char* base = reinterpret_cast<char*>(a.d_result);
    float* block_max = reinterpret_cast<float*>(base + sizeof(int32_t));
    float* block_sum = block_max + SAMPLE_NBLOCKS;
    float* cand_val = block_sum + SAMPLE_NBLOCKS;
    int* cand_idx = reinterpret_cast<int*>(cand_val + static_cast<size_t>(SAMPLE_NBLOCKS) * a.top_k);
    topk_finalize_body(a.top_k, a.top_p, a.inv_temperature, a.seed, n_blocks, block_max, block_sum,
                       cand_val, cand_idx, a.d_result);
}

// Launch the two-phase multi-block sampler. Scratch lives right after d_result
// (caller guarantees >= SAMPLE_SCRATCH_BYTES). Both kernels are graph-capturable
// (no allocation, fixed topology for a given vocab_size/top_k).
static void launch_topk_topp_multiblock(const float* d_logits, int vocab_size, int top_k, float top_p,
                                        float inv_temperature, unsigned int seed, int32_t* d_result,
                                        cudaStream_t stream) {
    char* base = reinterpret_cast<char*>(d_result);
    float* block_max = reinterpret_cast<float*>(base + sizeof(int32_t));
    float* block_sum = block_max + SAMPLE_NBLOCKS;
    float* cand_val = block_sum + SAMPLE_NBLOCKS;
    int* cand_idx = reinterpret_cast<int*>(cand_val + static_cast<size_t>(SAMPLE_NBLOCKS) * top_k);

    constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
    // both phases use NUM_WARPS*top_k warp-merge scratch; phase 1 also needs the
    // block reduction scratch, phase 2 the merged top_k output.
    size_t smem1 = static_cast<size_t>(BLOCK_SIZE) * sizeof(float) + 2 * sizeof(float) +
                   static_cast<size_t>(NUM_WARPS) * top_k * (sizeof(float) + sizeof(int));
    size_t smem2 = static_cast<size_t>(NUM_WARPS) * top_k * (sizeof(float) + sizeof(int)) +
                   static_cast<size_t>(top_k) * (sizeof(float) + sizeof(int));

    topk_partial_kernel<<<SAMPLE_NBLOCKS, BLOCK_SIZE, smem1, stream>>>(
        d_logits, vocab_size, top_k, inv_temperature, block_max, block_sum, cand_val, cand_idx);
    IMP_CUDA_CHECK_LAUNCH();
    topk_finalize_kernel<<<1, BLOCK_SIZE, smem2, stream>>>(top_k, top_p, inv_temperature, seed,
                                                           SAMPLE_NBLOCKS, block_max, block_sum, cand_val,
                                                           cand_idx, d_result);
    IMP_CUDA_CHECK_LAUNCH();
}

void launch_topk_topp_rows(const TopkRowArgs* d_rows, int n_rows, int max_top_k, int vocab_size,
                           cudaStream_t stream) {
    constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
    // Shared-memory sizes carved with the batch's max top_k — every block
    // carves with its OWN row's top_k, which is <= max_top_k.
    size_t smem1 = static_cast<size_t>(BLOCK_SIZE) * sizeof(float) + 2 * sizeof(float) +
                   static_cast<size_t>(NUM_WARPS) * max_top_k * (sizeof(float) + sizeof(int));
    size_t smem2 = static_cast<size_t>(NUM_WARPS) * max_top_k * (sizeof(float) + sizeof(int)) +
                   static_cast<size_t>(max_top_k) * (sizeof(float) + sizeof(int));
    dim3 grid1(SAMPLE_NBLOCKS, n_rows);
    topk_partial_rows_kernel<<<grid1, BLOCK_SIZE, smem1, stream>>>(d_rows, vocab_size);
    IMP_CUDA_CHECK_LAUNCH();
    topk_finalize_rows_kernel<<<n_rows, BLOCK_SIZE, smem2, stream>>>(d_rows, SAMPLE_NBLOCKS);
    IMP_CUDA_CHECK_LAUNCH();
}

// ============================================================================
// CUB-based top-k sampling for k > MAX_TOP_K (128).
//
// Strategy:
//   1. Compute softmax probabilities with temperature scaling.
//   2. Sort (probability, vocab_index) pairs descending via CUB RadixSort.
//   3. Take first k elements, apply top-p cutoff, sample.
//
// This path is NOT used inside CUDA graph capture (CUB launches internal
// kernels). The single-block kernel above handles the graph-captured path.
// ============================================================================

// Kernel: compute softmax probabilities reading max/sum from device memory.
// d_max_sum[0] = global_max, d_max_sum[1] = sum. Avoids 2 D2H syncs.
__global__ void softmax_to_pairs_device_kernel(const float* __restrict__ logits, int vocab_size,
                                               float inv_temperature, const float* __restrict__ d_max_sum,
                                               float* __restrict__ d_keys, int32_t* __restrict__ d_values) {
    float global_max = d_max_sum[0];
    float inv_sum = 1.0f / d_max_sum[1];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= vocab_size)
        return;
    float prob = expf((logits[idx] - global_max) * inv_temperature) * inv_sum;
    d_keys[idx] = prob;
    d_values[idx] = idx;
}

// Kernel: find global max of logits (Phase 1)
__global__ void softmax_max_kernel(const float* __restrict__ logits, int vocab_size,
                                   float* __restrict__ d_max) {
    __shared__ float s_max[BLOCK_SIZE / WARP_SIZE];

    float local_max = -FLT_MAX;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < vocab_size; i += blockDim.x * gridDim.x) {
        float v = logits[i];
        if (v > local_max)
            local_max = v;
    }
    local_max = warp_reduce_max(local_max);
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    if (lane_id == 0)
        s_max[warp_id] = local_max;
    __syncthreads();
    if (threadIdx.x == 0) {
        float mx = -FLT_MAX;
        for (int w = 0; w < BLOCK_SIZE / WARP_SIZE; w++)
            if (s_max[w] > mx)
                mx = s_max[w];
        atomic_max_float(d_max, mx);
    }
}

// Kernel: compute sum of exp(logits - max) reading max from device memory.
// Avoids D2H sync between max and sum phases.
__global__ void softmax_sum_device_max_kernel(const float* __restrict__ logits, int vocab_size,
                                              float inv_temperature, const float* __restrict__ d_max,
                                              float* __restrict__ d_sum) {
    __shared__ float s_sum[BLOCK_SIZE / WARP_SIZE];

    float global_max = *d_max;
    float local_sum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < vocab_size; i += blockDim.x * gridDim.x) {
        local_sum += expf((logits[i] - global_max) * inv_temperature);
    }
    local_sum = warp_reduce_sum(local_sum);
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    if (lane_id == 0)
        s_sum[warp_id] = local_sum;
    __syncthreads();
    if (threadIdx.x == 0) {
        float sm = 0.0f;
        for (int w = 0; w < BLOCK_SIZE / WARP_SIZE; w++)
            sm += s_sum[w];
        atomicAdd(d_sum, sm);
    }
}

// Deterministic single-block variant of softmax_sum_device_max_kernel.
// A single block strides the whole vocab and reduces via a fixed-order
// shared-memory tree, writing the sum directly. This removes the cross-block
// FP atomicAdd of the multi-block path whose accumulation order varies
// run-to-run. Opt-in (deterministic mode) only.
__global__ void softmax_sum_device_max_single_block_kernel(const float* __restrict__ logits, int vocab_size,
                                                          float inv_temperature,
                                                          const float* __restrict__ d_max,
                                                          float* __restrict__ d_sum) {
    __shared__ float s_sum[BLOCK_SIZE / WARP_SIZE];

    float global_max = *d_max;
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        local_sum += expf((logits[i] - global_max) * inv_temperature);
    }
    local_sum = warp_reduce_sum(local_sum);
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    if (lane_id == 0)
        s_sum[warp_id] = local_sum;
    __syncthreads();
    if (threadIdx.x == 0) {
        float sm = 0.0f;
        for (int w = 0; w < BLOCK_SIZE / WARP_SIZE; w++)
            sm += s_sum[w];
        d_sum[0] = sm;
    }
}

// Kernel: top-p filter + sample from the first k sorted candidates
__global__ void topp_sample_from_sorted_kernel(const float* __restrict__ sorted_probs,
                                               const int32_t* __restrict__ sorted_indices, int top_k,
                                               float top_p, unsigned int seed,
                                               int32_t* __restrict__ d_result) {
    if (threadIdx.x != 0)
        return;

    float cumsum = 0.0f;
    int cutoff = top_k;
    for (int i = 0; i < top_k; i++) {
        cumsum += sorted_probs[i];
        if (cumsum >= top_p) {
            cutoff = i + 1;
            break;
        }
    }

    float norm = 0.0f;
    for (int i = 0; i < cutoff; i++)
        norm += sorted_probs[i];
    float inv_norm = (norm > 0.0f) ? (1.0f / norm) : 1.0f;

    unsigned int rng_state = sampler_seed_scramble(seed);
    float r = lcg_rand_float(rng_state);

    float acc = 0.0f;
    int32_t chosen = sorted_indices[0];
    for (int i = 0; i < cutoff; i++) {
        acc += sorted_probs[i] * inv_norm;
        if (r < acc) {
            chosen = sorted_indices[i];
            break;
        }
    }
    d_result[0] = chosen;
}

// Persistent scratch for CUB sort (lazily allocated, grows only)
struct CubSortScratch {
    float* d_keys_in = nullptr;
    float* d_keys_out = nullptr;
    int32_t* d_vals_in = nullptr;
    int32_t* d_vals_out = nullptr;
    float* d_max_sum = nullptr;
    void* d_temp = nullptr;
    size_t temp_bytes = 0;
    int capacity = 0;  // max vocab_size allocated for
    uint64_t gen = 0;  // arena generation these came from (model swap invalidates)

    bool ensure(int vocab_size, cudaStream_t stream) {
        const uint64_t g = engine_arena().generation();
        if (vocab_size <= capacity && gen == g)
            return true;
        // T2 (A7 step 8). One take per (model, arena) — `capacity` is the
        // vocabulary, which does not change while a model is loaded, so this
        // runs once and never climbs a staircase. The pre-arena code freed and
        // re-cudaMalloc'd six buffers from the SAMPLER, i.e. potentially while
        // serving; the arena take is pointer arithmetic and cannot.
        *this = CubSortScratch{};
        size_t elem_bytes = static_cast<size_t>(vocab_size) * sizeof(float);
        size_t idx_bytes = static_cast<size_t>(vocab_size) * sizeof(int32_t);
        auto ki = engine_arena().take_bytes(elem_bytes);
        auto ko = engine_arena().take_bytes(elem_bytes);
        auto vi = engine_arena().take_bytes(idx_bytes);
        auto vo = engine_arena().take_bytes(idx_bytes);
        auto ms = engine_arena().take_bytes(2 * sizeof(float));
        if (ki.empty() || ko.empty() || vi.empty() || vo.empty() || ms.empty()) {
            *this = CubSortScratch{};
            return false;
        }
        d_keys_in = reinterpret_cast<float*>(ki.data());
        d_keys_out = reinterpret_cast<float*>(ko.data());
        d_vals_in = reinterpret_cast<int32_t*>(vi.data());
        d_vals_out = reinterpret_cast<int32_t*>(vo.data());
        d_max_sum = reinterpret_cast<float*>(ms.data());
        // Query CUB temp storage: one full-vocabulary descending radix sort.
        // (The DeviceTopK + small-sort plan this used to size for is gone —
        // see the sampler for why, issue #1142.)
        temp_bytes = 0;
        cub::DeviceRadixSort::SortPairsDescending(nullptr, temp_bytes, d_keys_in, d_keys_out, d_vals_in,
                                                  d_vals_out, vocab_size, 0, 32, stream);
        auto tmp = engine_arena().take_bytes(temp_bytes);
        if (tmp.empty()) {
            *this = CubSortScratch{};
            return false;
        }
        d_temp = tmp.data();
        capacity = vocab_size;
        gen = g;
        IMP_LOG_DEBUG("CUB sort scratch: %.2f MiB from the T2 arena for vocab=%d (CUB temp %.2f MiB)",
                      (2 * elem_bytes + 2 * idx_bytes + 8 + temp_bytes) / (1024.0 * 1024.0), vocab_size,
                      temp_bytes / (1024.0 * 1024.0));
        return true;
    }

    // Arena-owned since A7 step 8: the region belongs to the T2 arena and
    // ~Engine closes it. `gen` is what notices a close, so this only re-arms.
    void free() { *this = CubSortScratch{}; }
};

static CubSortScratch s_cub_scratch;

// CUB-based top-k sampling for k > MAX_TOP_K.
static int32_t sample_topk_topp_cub(const float* d_logits, int vocab_size, int top_k, float top_p,
                                    float inv_temperature, unsigned int seed, int32_t* d_result,
                                    cudaStream_t stream) {
    if (!s_cub_scratch.ensure(vocab_size, stream)) {
        IMP_LOG_ERROR("CUB sort scratch allocation failed for vocab_size=%d", vocab_size);
        return 0;
    }

    auto& sc = s_cub_scratch;

    // Step 1: Compute softmax stats (max, then sum) entirely on device.
    // All intermediate values stay in d_max_sum — no D2H syncs needed.
    // d_max_sum[0] = global max, d_max_sum[1] = sum of exp.
    float neg_inf_val = -FLT_MAX;
    int neg_inf_bits;
    std::memcpy(&neg_inf_bits, &neg_inf_val, sizeof(int));
    IMP_CUDA_CHECK_LOG(
        cudaMemcpyAsync(sc.d_max_sum, &neg_inf_bits, sizeof(int), cudaMemcpyHostToDevice, stream));
    float zero = 0.0f;
    IMP_CUDA_CHECK_LOG(
        cudaMemcpyAsync(sc.d_max_sum + 1, &zero, sizeof(float), cudaMemcpyHostToDevice, stream));

    int stats_blocks = std::min((vocab_size + BLOCK_SIZE - 1) / BLOCK_SIZE, 128);

    const bool deterministic = process_diag_deterministic_gemm();

    // Phase 1: global max (result in d_max_sum[0]). atomicMax on the int-bitcast
    // of the max is order-independent and exact, so it is already deterministic.
    softmax_max_kernel<<<stats_blocks, BLOCK_SIZE, 0, stream>>>(d_logits, vocab_size, sc.d_max_sum);
    IMP_CUDA_CHECK_LAUNCH();

    // Phase 2: sum of exp — reads max from device memory (no D2H sync).
    // The default multi-block kernel sums via cross-block FP atomicAdd, whose
    // accumulation order varies run-to-run. In deterministic mode use a single
    // block with a fixed-order tree reduction instead.
    if (deterministic) {
        softmax_sum_device_max_single_block_kernel<<<1, BLOCK_SIZE, 0, stream>>>(
            d_logits, vocab_size, inv_temperature, sc.d_max_sum, sc.d_max_sum + 1);
        IMP_CUDA_CHECK_LAUNCH();
    } else {
        softmax_sum_device_max_kernel<<<stats_blocks, BLOCK_SIZE, 0, stream>>>(d_logits, vocab_size,
                                                                               inv_temperature, sc.d_max_sum,
                                                                               sc.d_max_sum + 1);
        IMP_CUDA_CHECK_LAUNCH();
    }

    // Step 2: Compute probabilities reading max/sum from device memory (no D2H sync)
    int pair_blocks = (vocab_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    softmax_to_pairs_device_kernel<<<pair_blocks, BLOCK_SIZE, 0, stream>>>(d_logits, vocab_size,
                                                                           inv_temperature, sc.d_max_sum,
                                                                           sc.d_keys_in, sc.d_vals_in);
    IMP_CUDA_CHECK_LAUNCH();

    // Step 3: sort the whole vocabulary by probability, descending. The top_k
    // entries the sampler needs are then simply the head of the result.
    //
    // This used to run cub::DeviceTopK::MaxPairs first and radix-sort only the
    // k survivors, which is asymptotically the better plan when k << vocab.
    // It is gone because it does not work here (issue #1142): instrumented on
    // Qwen3-8B-Q8_0 at top_k=129, MaxPairs writes all 129 slots on the first
    // call, writes NOTHING on the second while still returning cudaSuccess,
    // and from the fourth call on fails permanently with `invalid device
    // ordinal` — same thread, same stream, device 0, no pending error before
    // the call. The stale candidates left behind are what the model then
    // samples from, which is the `Okay,,,,,,,,` loop the issue reports.
    //
    // Nothing checked the return code, so the failure was silent. The full
    // sort is one well-trodden CUB entry point instead of two, the scratch is
    // already sized for it (`rs_full_bytes` in ensure()), and it only runs at
    // all when a request asks for top_k > MAX_TOP_K, which is rare and which
    // the caller has already opted into paying for.
    {
        size_t rs_bytes = sc.temp_bytes;
        cudaError_t rc = cub::DeviceRadixSort::SortPairsDescending(sc.d_temp, rs_bytes, sc.d_keys_in,
                                                                   sc.d_keys_out, sc.d_vals_in, sc.d_vals_out,
                                                                   vocab_size, 0, 32, stream);
        if (rc != cudaSuccess) {
            IMP_LOG_ERROR(
                "sample_topk_topp_cub: CUB sort failed (%s) for vocab=%d top_k=%d — no token "
                "sampled",
                cudaGetErrorString(rc), vocab_size, top_k);
            return 0;
        }
    }

    // Step 4: Top-p filter + sample from sorted top-k
    topp_sample_from_sorted_kernel<<<1, 1, 0, stream>>>(sc.d_keys_out, sc.d_vals_out, top_k, top_p, seed,
                                                        d_result);
    IMP_CUDA_CHECK_LAUNCH();

    int32_t h_result = 0;
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(&h_result, d_result, sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    return h_result;
}

// Shared implementation for both sample_topk_topp overloads.
// When owns_result is true, d_result was allocated internally and will be freed.
static int32_t sample_topk_topp_impl(const float* d_logits, int vocab_size, int top_k, float top_p,
                                     float inv_temperature, unsigned int seed, int32_t* d_result,
                                     bool owns_result, cudaStream_t stream) {
    // For large top_k, use CUB radix sort path (no MAX_TOP_K limit)
    if (top_k > MAX_TOP_K) {
        int32_t result = sample_topk_topp_cub(d_logits, vocab_size, top_k, top_p, inv_temperature, seed,
                                              d_result, stream);
        if (owns_result)
            IMP_CUDA_CHECK_LOG(cudaFree(d_result));
        return result;
    }

    launch_topk_topp_multiblock(d_logits, vocab_size, top_k, top_p, inv_temperature, seed, d_result, stream);

    int32_t h_result = 0;
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(&h_result, d_result, sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    if (owns_result)
        IMP_CUDA_CHECK_LOG(cudaFree(d_result));
    return h_result;
}

int32_t sample_topk_topp(const Tensor& logits, int top_k, float top_p, float temperature, unsigned int seed,
                         cudaStream_t stream) {
    const int vocab_size = static_cast<int>(logits.shape[0]);
    const float* d_logits = static_cast<const float*>(logits.data);

    if (top_k <= 0 || top_k > vocab_size)
        top_k = vocab_size;
    if (temperature <= 0.0f)
        temperature = 1.0f;
    float inv_temperature = 1.0f / temperature;

    int32_t* d_result = nullptr;
    if (cudaMalloc(&d_result, SAMPLE_SCRATCH_BYTES) != cudaSuccess) {
        IMP_LOG_ERROR("sample_topk_topp: cudaMalloc failed");
        return 0;
    }

    return sample_topk_topp_impl(d_logits, vocab_size, top_k, top_p, inv_temperature, seed, d_result, true,
                                 stream);
}

int32_t sample_topk_topp(const Tensor& logits, int top_k, float top_p, float temperature, unsigned int seed,
                         int32_t* d_result, cudaStream_t stream) {
    const int vocab_size = static_cast<int>(logits.shape[0]);
    const float* d_logits = static_cast<const float*>(logits.data);

    if (top_k <= 0 || top_k > vocab_size)
        top_k = vocab_size;
    if (temperature <= 0.0f)
        temperature = 1.0f;
    float inv_temperature = 1.0f / temperature;

    return sample_topk_topp_impl(d_logits, vocab_size, top_k, top_p, inv_temperature, seed, d_result, false,
                                 stream);
}

bool sample_topk_topp_async(const Tensor& logits, int top_k, float top_p, float temperature,
                            unsigned int seed, int32_t* d_result, cudaStream_t stream) {
    const int vocab_size = static_cast<int>(logits.shape[0]);
    const float* d_logits = static_cast<const float*>(logits.data);

    // Same parameter normalization as the synchronous variant (bit-identical
    // tokens for the same logits/seed).
    if (top_k <= 0 || top_k > vocab_size)
        top_k = vocab_size;
    if (top_k > MAX_TOP_K)
        return false;  // CUB regime syncs internally — caller uses the sync variant
    if (temperature <= 0.0f)
        temperature = 1.0f;
    float inv_temperature = 1.0f / temperature;

    launch_topk_topp_multiblock(d_logits, vocab_size, top_k, top_p, inv_temperature, seed, d_result, stream);
    return true;
}

void sample_topk_topp_device(const Tensor& logits, int top_k, float top_p, float temperature,
                             unsigned int seed, int32_t* d_result, int32_t* h_mapped, cudaStream_t stream) {
    const int vocab_size = static_cast<int>(logits.shape[0]);
    const float* d_logits = static_cast<const float*>(logits.data);

    if (top_k <= 0 || top_k > vocab_size)
        top_k = vocab_size;
    if (top_k > MAX_TOP_K) {
        IMP_LOG_WARN("top_k=%d exceeds MAX_TOP_K=%d, clamping", top_k, MAX_TOP_K);
        top_k = MAX_TOP_K;
    }
    if (temperature <= 0.0f)
        temperature = 1.0f;
    float inv_temperature = 1.0f / temperature;

    launch_topk_topp_multiblock(d_logits, vocab_size, top_k, top_p, inv_temperature, seed, d_result, stream);

    // Async copy to mapped pinned memory — no sync needed.
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(h_mapped, d_result, sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
}

// Free persistent CUB sort scratch. Called by sampling_cleanup().
void sampling_cleanup_cub() {
    s_cub_scratch.free();
}

}  // namespace imp
