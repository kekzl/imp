// MLA absorbed-decode kernels (Phase 3, opt-in). See mla_absorb.h.

#include "compute/mla_absorb.h"

namespace imp {

// ---------------------------------------------------------------------------
// Latent cache write
// ---------------------------------------------------------------------------
__global__ static void mla_latent_cache_write_kernel(
        const __half* __restrict__ latent,       // [n, kv_lora]
        const __half* __restrict__ k_assembled,  // [n, n_heads, head_dim]
        __half* __restrict__ cache,              // [max_seq, kv_lora + rope_dim]
        const int* __restrict__ positions,       // [n]
        int n, int n_heads, int head_dim, int rope_dim, int kv_lora, int max_seq) {
    const int t = blockIdx.x;
    if (t >= n) return;
    const int pos = positions[t];
    if (pos < 0 || pos >= max_seq) return;

    const int width = kv_lora + rope_dim;
    __half* dst = cache + static_cast<size_t>(pos) * width;
    const __half* lat = latent + static_cast<size_t>(t) * kv_lora;
    // k_rope: head 0 of the assembled K (replicated across heads, post-RoPE),
    // the first rope_dim dims of each head.
    const __half* krope = k_assembled + static_cast<size_t>(t) * n_heads * head_dim;

    for (int i = threadIdx.x; i < kv_lora; i += blockDim.x)
        dst[i] = lat[i];
    for (int i = threadIdx.x; i < rope_dim; i += blockDim.x)
        dst[kv_lora + i] = krope[i];
}

void mla_latent_cache_write(const half* latent, const half* k_assembled, half* cache,
                            const int* positions, int n_tokens, int n_heads, int head_dim,
                            int rope_dim, int kv_lora_rank, int max_seq, cudaStream_t stream) {
    if (n_tokens == 0) return;
    mla_latent_cache_write_kernel<<<n_tokens, 256, 0, stream>>>(
        reinterpret_cast<const __half*>(latent), reinterpret_cast<const __half*>(k_assembled),
        reinterpret_cast<__half*>(cache), positions, n_tokens, n_heads, head_dim, rope_dim,
        kv_lora_rank, max_seq);
}

// ---------------------------------------------------------------------------
// Absorbed decode attention (one block per head)
// ---------------------------------------------------------------------------
// Shared-memory block reduction helpers (256 threads).
__device__ __forceinline__ float block_reduce_max(float v, float* sh) {
    const int tid = threadIdx.x;
    sh[tid] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sh[tid] = fmaxf(sh[tid], sh[tid + s]);
        __syncthreads();
    }
    float r = sh[0];
    __syncthreads();
    return r;
}
__device__ __forceinline__ float block_reduce_sum(float v, float* sh) {
    const int tid = threadIdx.x;
    sh[tid] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sh[tid] += sh[tid + s];
        __syncthreads();
    }
    float r = sh[0];
    __syncthreads();
    return r;
}

__global__ static void mla_absorbed_decode_kernel(
        const __half* __restrict__ q,      // [n_heads, head_dim] = [pe|nope]
        const __half* __restrict__ kv_b,   // [n_heads*(nope+v), kv_lora]
        const __half* __restrict__ cache,  // [max_seq, kv_lora + rope_dim]
        __half* __restrict__ out,          // [n_heads, v_head_dim]
        float* __restrict__ scores,        // [n_heads, max_seq]
        const int* __restrict__ context_lens,
        int n_heads, int head_dim, int rope_dim, int nope_dim, int kv_lora, int v_head_dim,
        int max_seq, float scale) {
    const int h = blockIdx.x;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const int ctx = context_lens[0];
    if (ctx <= 0) return;

    const int width = kv_lora + rope_dim;     // cache row width
    const int head_out = nope_dim + v_head_dim;  // kv_b rows per head

    extern __shared__ float sh[];
    float* sh_qabs = sh;                       // [kv_lora]
    float* sh_qpe = sh_qabs + kv_lora;         // [rope_dim]
    float* sh_ctx = sh_qpe + rope_dim;         // [kv_lora]
    float* sh_red = sh_ctx + kv_lora;          // [nthreads]

    const __half* q_h = q + static_cast<size_t>(h) * head_dim;  // [pe|nope]
    const __half* q_pe = q_h;                                   // [rope_dim]
    const __half* q_nope = q_h + rope_dim;                      // [nope_dim]
    const __half* WUK = kv_b + static_cast<size_t>(h) * head_out * kv_lora;          // [nope, kv_lora]
    const __half* WUV = kv_b + (static_cast<size_t>(h) * head_out + nope_dim) * kv_lora;  // [v, kv_lora]
    float* my_scores = scores + static_cast<size_t>(h) * max_seq;

    // Load q_pe into shared (small).
    for (int i = tid; i < rope_dim; i += nthreads)
        sh_qpe[i] = __half2float(q_pe[i]);
    __syncthreads();

    // 1. q_absorbed[c] = sum_r q_nope[r] * WUK[r][c]
    for (int c = tid; c < kv_lora; c += nthreads) {
        float acc = 0.0f;
        for (int r = 0; r < nope_dim; r++)
            acc += __half2float(q_nope[r]) * __half2float(WUK[static_cast<size_t>(r) * kv_lora + c]);
        sh_qabs[c] = acc;
    }
    __syncthreads();

    // 2. scores[t] = scale * (q_absorbed . latent[t] + q_pe . k_rope[t])
    float local_max = -1e30f;
    for (int t = tid; t < ctx; t += nthreads) {
        const __half* lat = cache + static_cast<size_t>(t) * width;
        const __half* kr = lat + kv_lora;
        float s = 0.0f;
        for (int c = 0; c < kv_lora; c++)
            s += sh_qabs[c] * __half2float(lat[c]);
        for (int i = 0; i < rope_dim; i++)
            s += sh_qpe[i] * __half2float(kr[i]);
        s *= scale;
        my_scores[t] = s;
        local_max = fmaxf(local_max, s);
    }
    float gmax = block_reduce_max(local_max, sh_red);

    // 3. softmax: exp(score - max), accumulate denom.
    float local_sum = 0.0f;
    for (int t = tid; t < ctx; t += nthreads) {
        float e = __expf(my_scores[t] - gmax);
        my_scores[t] = e;
        local_sum += e;
    }
    float gsum = block_reduce_sum(local_sum, sh_red);
    float inv_sum = (gsum > 0.0f) ? (1.0f / gsum) : 0.0f;

    // 4. ctx[c] = (1/sum) * sum_t p[t] * latent[t][c]
    for (int c = tid; c < kv_lora; c += nthreads) {
        float acc = 0.0f;
        for (int t = 0; t < ctx; t++) {
            const __half* lat = cache + static_cast<size_t>(t) * width;
            acc += my_scores[t] * __half2float(lat[c]);
        }
        sh_ctx[c] = acc * inv_sum;
    }
    __syncthreads();

    // 5. out[d] = sum_c ctx[c] * WUV[d][c]
    __half* out_h = out + static_cast<size_t>(h) * v_head_dim;
    for (int d = tid; d < v_head_dim; d += nthreads) {
        float acc = 0.0f;
        for (int c = 0; c < kv_lora; c++)
            acc += sh_ctx[c] * __half2float(WUV[static_cast<size_t>(d) * kv_lora + c]);
        out_h[d] = __float2half(acc);
    }
}

void mla_absorbed_decode(const half* q, const half* kv_b, const half* cache, half* out,
                         float* scores, const int* context_lens, int n_heads, int head_dim,
                         int rope_dim, int nope_dim, int kv_lora_rank, int v_head_dim, int max_seq,
                         float scale, cudaStream_t stream) {
    const int nthreads = 256;
    // sh: q_absorbed[kv_lora] + q_pe[rope] + ctx[kv_lora] + reduce[nthreads]
    size_t shmem = (static_cast<size_t>(2 * kv_lora_rank) + rope_dim + nthreads) * sizeof(float);
    mla_absorbed_decode_kernel<<<n_heads, nthreads, shmem, stream>>>(
        reinterpret_cast<const __half*>(q), reinterpret_cast<const __half*>(kv_b),
        reinterpret_cast<const __half*>(cache), reinterpret_cast<__half*>(out), scores, context_lens,
        n_heads, head_dim, rope_dim, nope_dim, kv_lora_rank, v_head_dim, max_seq, scale);
}

}  // namespace imp
