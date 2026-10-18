// Radix-select top-k/top-p sampler: same tokens, candidates and scratch bytes as the legacy
// multi-block kernels in sampling_topk_topp.cu, which stay the fallback past
// kRadixMaxPerThread * SAMPLE_NBLOCKS * BLOCK_SIZE vocab entries (tests/test_sampling_topk_radix.cu).
//
// Legacy order, reproduced exactly: each thread keeps its own top-k (evict the first minimum on a
// strictly larger value), selection-swap sorts it, then a warp and a block merge pick the head
// with the larger value and, on equal values, the smaller vocab index. That merge equals a sort
// by (value desc, run_max asc, position asc), run_max = the running max of the vocab index over
// the equal-value run of the thread's sorted list. -inf never becomes a candidate.

#include "compute/sampling.h"
#include "compute/sampling_internal.cuh"
#include "compute/warp_reduce.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <cfloat>
#include <cstdint>
#include <type_traits>

namespace imp {

namespace {

constexpr int kRadixBins = 256;
constexpr int kRadixWarps = BLOCK_SIZE / WARP_SIZE;
constexpr int kRadixMaxPerThread = 16;  // partial kernel: vocab <= 16 * 64 * 256 = 262144
static_assert(BLOCK_SIZE == kRadixBins, "one histogram bin per thread");

struct TopkSelectSmem {
    unsigned int hist[kRadixBins];
    unsigned int warp_sum[kRadixWarps];
    int digit;
    int need;
    int state;  // 0 next digit, 1 threshold found, 2 fewer keys than top_k
    unsigned int count;
    unsigned long long key[SAMPLE_MAX_TOP_K];
    float val[SAMPLE_MAX_TOP_K];
    int idx[SAMPLE_MAX_TOP_K];
};

// Larger key = earlier in the legacy order; 0 = not a candidate. Equal under float == (ftz
// denormals, -0) folds to one bit pattern, as the legacy compares see it.
__device__ __forceinline__ unsigned long long topk_order_key(float v, int run_max, int pos) {
    if (v == -INFINITY)
        return 0ull;
    const uint32_t u = (v == 0.0f) ? 0u : __float_as_uint(v);
    const uint32_t hi = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    const uint32_t lo = ~((static_cast<uint32_t>(run_max) << 5) | static_cast<uint32_t>(pos));
    return (static_cast<unsigned long long>(hi) << 32) | lo;
}

// Legacy per-thread sort (block_reduce_topk) on the valid slots, then the order keys.
template <int N>
__device__ __forceinline__ void topk_thread_keys(float (&v)[N], int (&ix)[N], const bool (&ok)[N],
                                                 unsigned long long (&key)[N]) {
#pragma unroll
    for (int i = 0; i < N - 1; ++i) {
#pragma unroll
        for (int j = i + 1; j < N; ++j) {
            if (ok[i] && ok[j] && v[j] > v[i]) {
                const float tv = v[i];
                v[i] = v[j];
                v[j] = tv;
                const int ti = ix[i];
                ix[i] = ix[j];
                ix[j] = ti;
            }
        }
    }
    float prev_v = 0.0f;
    int prev_r = 0;
    bool have = false;
#pragma unroll
    for (int p = 0; p < N; ++p) {
        key[p] = 0ull;
        if (ok[p]) {
            const int r = (have && v[p] == prev_v) ? max(prev_r, ix[p]) : ix[p];
            key[p] = topk_order_key(v[p], r, p);
            prev_v = v[p];
            prev_r = r;
            have = true;
        }
    }
}

// Histogram of the 8-bit digit at `shift` over the keys whose higher digits equal `prefix`.
template <int N>
__device__ __forceinline__ void topk_radix_histogram(const unsigned long long (&key)[N],
                                                     unsigned long long prefix, int shift,
                                                     TopkSelectSmem& sm) {
    const int lane = threadIdx.x % WARP_SIZE;
    sm.hist[threadIdx.x] = 0u;
    __syncthreads();
    const unsigned long long hi_mask = (shift == 56) ? 0ull : (~0ull << (shift + 8));
#pragma unroll
    for (int p = 0; p < N; ++p) {
        const bool in = key[p] != 0ull && (key[p] & hi_mask) == prefix;
        const unsigned int d = in ? static_cast<unsigned int>(key[p] >> shift) & 0xFFu : 0x100u;
        const unsigned int peers = __match_any_sync(0xFFFFFFFFu, d);
        if (in && lane == __ffs(peers) - 1)
            atomicAdd(&sm.hist[d], static_cast<unsigned int>(__popc(peers)));
    }
    __syncthreads();
}

// Finds the bin holding the need-th largest key; writes digit/need/state to sm.
__device__ __forceinline__ void topk_radix_pick(int need, int shift, TopkSelectSmem& sm) {
    const int tid = threadIdx.x;
    const int lane = tid % WARP_SIZE;
    // thread t owns bin 255 - t: inclusive scan = keys in this bin or a larger one
    const unsigned int c = sm.hist[kRadixBins - 1 - tid];
    unsigned int x = c;
#pragma unroll
    for (int off = 1; off < WARP_SIZE; off <<= 1) {
        const unsigned int y = __shfl_up_sync(0xFFFFFFFFu, x, off);
        if (lane >= off)
            x += y;
    }
    if (lane == WARP_SIZE - 1)
        sm.warp_sum[tid / WARP_SIZE] = x;
    __syncthreads();
    unsigned int base = 0u;
    for (int w = 0; w < tid / WARP_SIZE; ++w)
        base += sm.warp_sum[w];
    const unsigned int incl = base + x;
    const unsigned int excl = incl - c;
    const auto un = static_cast<unsigned int>(need);
    if (excl < un && un <= incl) {
        sm.digit = kRadixBins - 1 - tid;
        sm.need = static_cast<int>(un - excl);
        sm.state = (c == un - excl || shift == 0) ? 1 : 0;
    }
    if (tid == BLOCK_SIZE - 1 && incl < un)
        sm.state = 2;
    __syncthreads();
}

// Rank sort of the selected keys (unique) into out_val/out_idx, padding (-FLT_MAX, -1) like the legacy.
__device__ __forceinline__ void topk_write_sorted(int top_k, TopkSelectSmem& sm, float* __restrict__ out_val,
                                                  int* __restrict__ out_idx) {
    const int tid = threadIdx.x;
    const int m = min(static_cast<int>(sm.count), top_k);
    if (tid < m) {
        const unsigned long long k = sm.key[tid];
        int rank = 0;
        for (int j = 0; j < m; ++j)
            rank += (sm.key[j] > k) ? 1 : 0;
        out_val[rank] = sm.val[tid];
        out_idx[rank] = sm.idx[tid];
    } else if (tid < top_k) {
        out_val[tid] = -FLT_MAX;
        out_idx[tid] = -1;
    }
}

// Block top-k of the unique non-zero keys: 8-bit MSB radix select, then rank sort of the survivors.
template <int N>
__device__ __forceinline__ void topk_block_select(const unsigned long long (&key)[N], const float (&v)[N],
                                                  const int (&ix)[N], int top_k, TopkSelectSmem& sm,
                                                  float* __restrict__ out_val, int* __restrict__ out_idx) {
    unsigned long long prefix = 0ull;
    int need = top_k;
    int shift = 64;
    int state = 0;
    if (threadIdx.x == 0)
        sm.count = 0u;
    while (state == 0) {
        shift -= 8;
        topk_radix_histogram(key, prefix, shift, sm);
        topk_radix_pick(need, shift, sm);
        state = sm.state;
        if (state != 2) {
            prefix |= static_cast<unsigned long long>(sm.digit) << shift;
            need = sm.need;
        }
    }
    const unsigned long long thr = prefix >> shift;
#pragma unroll
    for (int p = 0; p < N; ++p) {
        if (key[p] != 0ull && (state == 2 || (key[p] >> shift) >= thr)) {
            const unsigned int slot = atomicAdd(&sm.count, 1u);
            if (slot < static_cast<unsigned int>(top_k)) {
                sm.key[slot] = key[p];
                sm.val[slot] = v[p];
                sm.idx[slot] = ix[p];
            }
        }
    }
    __syncthreads();
    topk_write_sorted(top_k, sm, out_val, out_idx);
}

// Legacy block max and sum of exp((logit - max) * inv_temperature), same reduction order.
template <int N>
__device__ __forceinline__ void topk_block_max_sum(const float (&v)[N], int n, float inv_temperature,
                                                   float* __restrict__ block_max_out,
                                                   float* __restrict__ block_sum_out) {
    __shared__ float s_reduce[BLOCK_SIZE];
    __shared__ float s_gmax;
    const int tid = threadIdx.x;
    float local_max = -FLT_MAX;
#pragma unroll
    for (int p = 0; p < N; ++p)
        if (p < n)
            local_max = std::max(local_max, v[p]);
    local_max = warp_reduce_max(local_max);
    s_reduce[tid] = -FLT_MAX;
    __syncthreads();
    if (tid % WARP_SIZE == 0)
        s_reduce[tid / WARP_SIZE] = local_max;
    __syncthreads();
    if (tid == 0) {
        float mx = -FLT_MAX;
        for (int w = 0; w < kRadixWarps; ++w)
            mx = std::max(mx, s_reduce[w]);
        s_gmax = mx;
    }
    __syncthreads();
    const float bmax = s_gmax;

    float local_sum = 0.0f;
    if (bmax > -FLT_MAX) {
#pragma unroll
        for (int p = 0; p < N; ++p)
            if (p < n)
                local_sum += expf((v[p] - bmax) * inv_temperature);
    }
    local_sum = warp_reduce_sum(local_sum);
    s_reduce[tid] = 0.0f;
    __syncthreads();
    if (tid % WARP_SIZE == 0)
        s_reduce[tid / WARP_SIZE] = local_sum;
    __syncthreads();
    if (tid == 0) {
        float sum = 0.0f;
        for (int w = 0; w < kRadixWarps; ++w)
            sum += s_reduce[w];
        block_max_out[blockIdx.x] = bmax;
        block_sum_out[blockIdx.x] = sum;
    }
}

// Legacy per-thread top-local_k of n > local_k elements: slots [0, local_k) fill in order,
// each later element strictly above the first minimum replaces it.
template <int N>
__device__ __forceinline__ void topk_legacy_evict(float (&v)[N], int (&ix)[N], int n, int local_k) {
    float min_v = v[0];
    int min_p = 0;
#pragma unroll
    for (int s = 1; s < N; ++s)
        if (s < local_k && v[s] < min_v) {
            min_v = v[s];
            min_p = s;
        }
#pragma unroll
    for (int j = 1; j < N; ++j) {
        if (j < local_k || j >= n || !(v[j] > min_v))
            continue;
#pragma unroll
        for (int s = 0; s < j; ++s)
            if (s == min_p) {
                v[s] = v[j];
                ix[s] = ix[j];
            }
        min_v = v[0];
        min_p = 0;
#pragma unroll
        for (int s = 1; s < j; ++s)
            if (s < local_k && v[s] < min_v) {
                min_v = v[s];
                min_p = s;
            }
    }
}

// Phase 1: one load of the block's strided subset into registers.
template <int N>
__device__ __forceinline__ void topk_radix_partial_body(const float* __restrict__ logits, int vocab_size,
                                                        int top_k, float inv_temperature,
                                                        float* __restrict__ block_max_out,
                                                        float* __restrict__ block_sum_out,
                                                        float* __restrict__ cand_val_out,
                                                        int* __restrict__ cand_idx_out) {
    __shared__ TopkSelectSmem sm;
    const int gstride = blockDim.x * gridDim.x;
    const int gstart = (blockIdx.x * blockDim.x) + threadIdx.x;
    float v[N];
    int ix[N];
    int n = 0;
#pragma unroll
    for (int p = 0; p < N; ++p) {
        const int i = gstart + (p * gstride);
        v[p] = 0.0f;
        ix[p] = -1;
        if (i < vocab_size) {
            v[p] = logits[i];
            ix[p] = i;
            n = p + 1;
        }
    }
    topk_block_max_sum(v, n, inv_temperature, block_max_out, block_sum_out);

    const int local_k = min(top_k, SAMPLE_MAX_TOP_K);
    if (n > local_k)
        topk_legacy_evict(v, ix, n, local_k);
    const int count = min(n, local_k);
    bool ok[N];
#pragma unroll
    for (int p = 0; p < N; ++p)
        ok[p] = p < count;
    unsigned long long key[N];
    topk_thread_keys(v, ix, ok, key);
    topk_block_select(key, v, ix, top_k, sm, cand_val_out + (static_cast<size_t>(blockIdx.x) * top_k),
                      cand_idx_out + (static_cast<size_t>(blockIdx.x) * top_k));
}

template <int N>
__global__ void __launch_bounds__(BLOCK_SIZE) topk_radix_partial_kernel(
    const float* __restrict__ logits, int vocab_size, int top_k, float inv_temperature,
    float* __restrict__ block_max_out, float* __restrict__ block_sum_out, float* __restrict__ cand_val_out,
    int* __restrict__ cand_idx_out) {
    topk_radix_partial_body<N>(logits, vocab_size, top_k, inv_temperature, block_max_out, block_sum_out,
                               cand_val_out, cand_idx_out);
}

template <int N>
__global__ void __launch_bounds__(BLOCK_SIZE) topk_radix_partial_rows_kernel(
    const TopkRowArgs* __restrict__ rows, int vocab_size) {
    const TopkRowArgs a = rows[blockIdx.y];
    char* base = reinterpret_cast<char*>(a.d_result);
    float* block_max = reinterpret_cast<float*>(base + sizeof(int32_t));
    float* block_sum = block_max + SAMPLE_NBLOCKS;
    float* cand_val = block_sum + SAMPLE_NBLOCKS;
    int* cand_idx = reinterpret_cast<int*>(cand_val + (static_cast<size_t>(SAMPLE_NBLOCKS) * a.top_k));
    topk_radix_partial_body<N>(a.logits, vocab_size, a.top_k, a.inv_temperature, block_max, block_sum,
                               cand_val, cand_idx);
}

// Legacy thread-0 tail of topk_finalize_body: global softmax over the block partials, top-p, sample.
__device__ __forceinline__ int topk_softmax_topp_sample(int top_k, float top_p, float inv_temperature,
                                                        unsigned int seed, int n_blocks, const float* s_bmax,
                                                        const float* s_bsum, float* s_val, const int* s_idx) {
    float gmax = -FLT_MAX;
    for (int b = 0; b < n_blocks; ++b)
        gmax = std::max(gmax, s_bmax[b]);
    float gsum = 0.0f;
    for (int b = 0; b < n_blocks; ++b) {
        float bm = s_bmax[b];
        if (bm > -FLT_MAX)
            gsum += s_bsum[b] * expf((bm - gmax) * inv_temperature);
    }
    float inv_sum = (gsum > 0.0f) ? (1.0f / gsum) : 1.0f;

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
    return chosen;
}

// Phase 2: radix select over the n_blocks * top_k candidate pool, then the legacy tail.
template <int N>
__device__ __forceinline__ void topk_radix_finalize_body(int top_k, float top_p, float inv_temperature,
                                                         unsigned int seed, int n_blocks,
                                                         const float* __restrict__ block_max_in,
                                                         const float* __restrict__ block_sum_in,
                                                         const float* __restrict__ cand_val_in,
                                                         const int* __restrict__ cand_idx_in,
                                                         int32_t* __restrict__ d_result) {
    __shared__ TopkSelectSmem sm;
    __shared__ float s_val[SAMPLE_MAX_TOP_K];
    __shared__ int s_idx[SAMPLE_MAX_TOP_K];
    __shared__ float s_bmax[SAMPLE_NBLOCKS];
    __shared__ float s_bsum[SAMPLE_NBLOCKS];
    const int tid = threadIdx.x;
    const int n_cand = n_blocks * top_k;
    for (int b = tid; b < n_blocks; b += BLOCK_SIZE) {
        s_bmax[b] = block_max_in[b];
        s_bsum[b] = block_sum_in[b];
    }
    float v[N];
    int ix[N];
    bool ok[N];
#pragma unroll
    for (int p = 0; p < N; ++p) {
        const int i = tid + (p * BLOCK_SIZE);
        const int idx = (i < n_cand) ? cand_idx_in[i] : -1;
        ok[p] = idx >= 0;
        ix[p] = idx;
        v[p] = ok[p] ? cand_val_in[i] : 0.0f;
    }
    unsigned long long key[N];
    topk_thread_keys(v, ix, ok, key);
    topk_block_select(key, v, ix, top_k, sm, s_val, s_idx);
    __syncthreads();
    if (tid == 0)
        d_result[0] = static_cast<int32_t>(topk_softmax_topp_sample(top_k, top_p, inv_temperature, seed,
                                                                    n_blocks, s_bmax, s_bsum, s_val, s_idx));
}

template <int N>
__global__ void __launch_bounds__(BLOCK_SIZE) topk_radix_finalize_kernel(
    int top_k, float top_p, float inv_temperature, unsigned int seed, const int* __restrict__ d_seed_salt,
    int n_blocks, const float* __restrict__ block_max_in, const float* __restrict__ block_sum_in,
    const float* __restrict__ cand_val_in, const int* __restrict__ cand_idx_in,
    int32_t* __restrict__ d_result) {
    const unsigned int salted = seed + (d_seed_salt ? static_cast<unsigned int>(d_seed_salt[0]) : 0u);
    topk_radix_finalize_body<N>(top_k, top_p, inv_temperature, salted, n_blocks, block_max_in, block_sum_in,
                                cand_val_in, cand_idx_in, d_result);
}

template <int N>
__global__ void __launch_bounds__(BLOCK_SIZE) topk_radix_finalize_rows_kernel(
    const TopkRowArgs* __restrict__ rows, int n_blocks) {
    const TopkRowArgs a = rows[blockIdx.x];
    char* base = reinterpret_cast<char*>(a.d_result);
    float* block_max = reinterpret_cast<float*>(base + sizeof(int32_t));
    float* block_sum = block_max + SAMPLE_NBLOCKS;
    float* cand_val = block_sum + SAMPLE_NBLOCKS;
    int* cand_idx = reinterpret_cast<int*>(cand_val + (static_cast<size_t>(SAMPLE_NBLOCKS) * a.top_k));
    topk_radix_finalize_body<N>(a.top_k, a.top_p, a.inv_temperature, a.seed, n_blocks, block_max, block_sum,
                                cand_val, cand_idx, a.d_result);
}

// Registers per thread: partial ceil(vocab / 16384) <= 16, finalize ceil(64 * top_k / 256) <= 32.
bool partial_fits_8(int vocab_size) { return vocab_size <= 8 * SAMPLE_NBLOCKS * BLOCK_SIZE; }

template <typename F>
void with_finalize_slots(int top_k, F&& f) {
    const int per = ((SAMPLE_NBLOCKS * top_k) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (per <= 8)
        f(std::integral_constant<int, 8>{});
    else if (per <= 16)
        f(std::integral_constant<int, 16>{});
    else
        f(std::integral_constant<int, 32>{});
}

}  // namespace

bool topk_radix_supported(int vocab_size, int max_top_k) {
    return vocab_size > 0 && vocab_size <= kRadixMaxPerThread * SAMPLE_NBLOCKS * BLOCK_SIZE &&
           max_top_k >= 1 && max_top_k <= SAMPLE_MAX_TOP_K;
}

cudaError_t launch_topk_topp_radix(const float* d_logits, int vocab_size, int top_k, float top_p,
                                   float inv_temperature, unsigned int seed, int32_t* d_result,
                                   cudaStream_t stream, const int* d_seed_salt) {
    char* base = reinterpret_cast<char*>(d_result);
    float* block_max = reinterpret_cast<float*>(base + sizeof(int32_t));
    float* block_sum = block_max + SAMPLE_NBLOCKS;
    float* cand_val = block_sum + SAMPLE_NBLOCKS;
    int* cand_idx = reinterpret_cast<int*>(cand_val + (static_cast<size_t>(SAMPLE_NBLOCKS) * top_k));

    sampler_launch_begin("topk_radix");
    cudaError_t e = cudaSuccess;
    if (partial_fits_8(vocab_size)) {
        topk_radix_partial_kernel<8>
            <<<SAMPLE_NBLOCKS, BLOCK_SIZE, 0, stream>>>(d_logits, vocab_size, top_k, inv_temperature,
                                                        block_max, block_sum, cand_val, cand_idx);
        e = cudaGetLastError();
    } else {
        topk_radix_partial_kernel<16>
            <<<SAMPLE_NBLOCKS, BLOCK_SIZE, 0, stream>>>(d_logits, vocab_size, top_k, inv_temperature,
                                                        block_max, block_sum, cand_val, cand_idx);
        e = cudaGetLastError();
    }
    if (sampler_launch_status(e, "topk_radix") != cudaSuccess)
        return e;
    with_finalize_slots(top_k, [&](auto n) {
        topk_radix_finalize_kernel<decltype(n)::value>
            <<<1, BLOCK_SIZE, 0, stream>>>(top_k, top_p, inv_temperature, seed, d_seed_salt, SAMPLE_NBLOCKS,
                                           block_max, block_sum, cand_val, cand_idx, d_result);
        e = cudaGetLastError();
    });
    return sampler_launch_status(e, "topk_radix");
}

cudaError_t launch_topk_topp_rows_radix(const TopkRowArgs* d_rows, int n_rows, int max_top_k, int vocab_size,
                                        cudaStream_t stream) {
    const dim3 grid1(SAMPLE_NBLOCKS, n_rows);
    sampler_launch_begin("topk_rows_radix");
    cudaError_t e = cudaSuccess;
    if (partial_fits_8(vocab_size)) {
        topk_radix_partial_rows_kernel<8><<<grid1, BLOCK_SIZE, 0, stream>>>(d_rows, vocab_size);
        e = cudaGetLastError();
    } else {
        topk_radix_partial_rows_kernel<16><<<grid1, BLOCK_SIZE, 0, stream>>>(d_rows, vocab_size);
        e = cudaGetLastError();
    }
    if (sampler_launch_status(e, "topk_rows_radix") != cudaSuccess)
        return e;
    with_finalize_slots(max_top_k, [&](auto n) {
        topk_radix_finalize_rows_kernel<decltype(n)::value>
            <<<n_rows, BLOCK_SIZE, 0, stream>>>(d_rows, SAMPLE_NBLOCKS);
        e = cudaGetLastError();
    });
    return sampler_launch_status(e, "topk_rows_radix");
}

cudaError_t launch_topk_topp_multiblock(const float* d_logits, int vocab_size, int top_k, float top_p,
                                        float inv_temperature, unsigned int seed, int32_t* d_result,
                                        cudaStream_t stream, const int* d_seed_salt) {
    if (topk_radix_supported(vocab_size, top_k))
        return launch_topk_topp_radix(d_logits, vocab_size, top_k, top_p, inv_temperature, seed, d_result,
                                      stream, d_seed_salt);
    return launch_topk_topp_multiblock_legacy(d_logits, vocab_size, top_k, top_p, inv_temperature, seed,
                                              d_result, stream, d_seed_salt);
}

cudaError_t launch_topk_topp_rows(const TopkRowArgs* d_rows, int n_rows, int max_top_k, int vocab_size,
                                  cudaStream_t stream) {
    if (n_rows > 0 && topk_radix_supported(vocab_size, max_top_k))
        return launch_topk_topp_rows_radix(d_rows, n_rows, max_top_k, vocab_size, stream);
    return launch_topk_topp_rows_legacy(d_rows, n_rows, max_top_k, vocab_size, stream);
}

}  // namespace imp
