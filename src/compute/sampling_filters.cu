#include "compute/sampling.h"
#include "compute/sampling_internal.cuh"
#include "compute/warp_reduce.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <cfloat>

namespace imp {

// ===========================================================================
// min_p filtering
// ===========================================================================

// Single-kernel min_p: finds max logit via cooperative reduction, then
// filters tokens in logit space.  threshold = max_logit + log(min_p).
// No host sync or temp allocation needed.
__global__ void apply_min_p_kernel(float* __restrict__ logits, int vocab_size, float log_min_p) {
    constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
    __shared__ float s_max[NUM_WARPS];
    __shared__ float s_threshold;

    const int tid = threadIdx.x;

    // Pass 1: find max logit (cooperative reduction)
    float local_max = -FLT_MAX;
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        float v = logits[i];
        if (v > local_max)
            local_max = v;
    }
    local_max = warp_reduce_max(local_max);
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    if (lane_id == 0)
        s_max[warp_id] = local_max;
    __syncthreads();
    if (tid == 0) {
        float mx = -FLT_MAX;
        for (int w = 0; w < NUM_WARPS; w++)
            if (s_max[w] > mx)
                mx = s_max[w];
        s_threshold = mx + log_min_p;
    }
    __syncthreads();

    // Pass 2: filter tokens below threshold
    float threshold = s_threshold;
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        if (logits[i] < threshold)
            logits[i] = -FLT_MAX;
    }
}

void apply_min_p(float* logits, int vocab_size, float min_p, cudaStream_t stream) {
    if (min_p <= 0.0f)
        return;

    float log_min_p = logf(min_p);
    apply_min_p_kernel<<<1, BLOCK_SIZE, 0, stream>>>(logits, vocab_size, log_min_p);
}

// ===========================================================================
// Typical-P (locally typical) filtering
// ===========================================================================

// Single-block kernel: computes entropy, deviation histogram, finds threshold,
// and filters tokens with deviation > threshold.
static constexpr int TYPICAL_NBUCKETS = 256;

__global__ void apply_typical_p_kernel(float* __restrict__ logits, int vocab_size, float typical_p) {
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
    local_max = warp_reduce_max(local_max);
    if (lane_id == 0)
        s_warp[warp_id] = local_max;
    __syncthreads();
    if (tid == 0) {
        float mx = -FLT_MAX;
        for (int w = 0; w < NUM_WARPS; w++)
            mx = fmaxf(mx, s_warp[w]);
        s_max = mx;
    }
    __syncthreads();
    float gmax = s_max;

    // --- Pass 2: sum_exp ---
    float local_sum = 0.0f;
    for (int i = tid; i < vocab_size; i += blockDim.x)
        local_sum += expf(logits[i] - gmax);
    local_sum = warp_reduce_sum(local_sum);
    if (lane_id == 0)
        s_warp[warp_id] = local_sum;
    __syncthreads();
    if (tid == 0) {
        float sm = 0.0f;
        for (int w = 0; w < NUM_WARPS; w++)
            sm += s_warp[w];
        s_sum = sm;
    }
    __syncthreads();

    float sum_exp = s_sum;
    float log_sum_exp = gmax + logf(sum_exp);
    float inv_log2 = 1.4426950408889634f;  // 1/ln(2)

    // --- Pass 3: entropy H = -sum(p_i * log2(p_i)) ---
    float local_ent = 0.0f;
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        float p = expf(logits[i] - gmax) / sum_exp;
        if (p > 1e-30f)
            local_ent -= p * log2f(p);
    }
    local_ent = warp_reduce_sum(local_ent);
    if (lane_id == 0)
        s_warp[warp_id] = local_ent;
    __syncthreads();
    if (tid == 0) {
        float e = 0.0f;
        for (int w = 0; w < NUM_WARPS; w++)
            e += s_warp[w];
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
    local_md = warp_reduce_max(local_md);
    if (lane_id == 0)
        s_warp[warp_id] = local_md;
    __syncthreads();
    if (tid == 0) {
        float md = 0.0f;
        for (int w = 0; w < NUM_WARPS; w++)
            md = fmaxf(md, s_warp[w]);
        s_max_dev = md;
    }
    __syncthreads();

    // --- Pass 5: build deviation histogram ---
    // Initialize buckets
    for (int b = tid; b < TYPICAL_NBUCKETS; b += blockDim.x)
        s_buckets[b] = 0.0f;
    __syncthreads();

    float bucket_scale = (s_max_dev > 1e-8f) ? (static_cast<float>(TYPICAL_NBUCKETS) / s_max_dev) : 1.0f;

    for (int i = tid; i < vocab_size; i += blockDim.x) {
        float surprise = -(logits[i] - log_sum_exp) * inv_log2;
        float dev = fabsf(surprise - H);
        int bucket = min(static_cast<int>(dev * bucket_scale), TYPICAL_NBUCKETS - 1);
        float p = expf(logits[i] - gmax) / sum_exp;
        // TODO(determinism): this shared-memory FP atomicAdd accumulates bucket
        // mass in scheduling-dependent order, so the cumulative cutoff bucket
        // can flip when typical_p lands near a bucket boundary. typical_p is a
        // sampling FILTER (not the greedy / top-k core covered by the
        // deterministic flag); make this an ordered per-bucket reduction if
        // typical_p ever needs bit-exact reproducibility.
        atomicAdd(&s_buckets[bucket], p);
    }
    __syncthreads();

    // --- Pass 6: scan histogram to find threshold (thread 0) ---
    if (tid == 0) {
        float cum = 0.0f;
        s_threshold = s_max_dev + 1.0f;  // default: keep all
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
        if (dev > thr)
            logits[i] = -FLT_MAX;
    }
}

void apply_typical_p(float* logits, int vocab_size, float typical_p, cudaStream_t stream) {
    if (typical_p <= 0.0f || typical_p >= 1.0f)
        return;

    apply_typical_p_kernel<<<1, BLOCK_SIZE, 0, stream>>>(logits, vocab_size, typical_p);
}

}  // namespace imp
