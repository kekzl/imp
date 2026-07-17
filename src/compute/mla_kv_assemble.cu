// MLA KV-buffer assembly kernels (Task 2.3).
//
// RoPE layout choice (b): pe FIRST in each K (and Q) head so that the
// existing rope kernel (which rotates the first rope_dim dims) applies
// unchanged to both Q and K.
//
// kv_b layout from kv_b_proj: per token, per head h:
//   kv_b[h] = [k_nope(nope_dim) | v(v_head_dim)]
// k_rope: [n_tokens, rope_dim]  — MQA-style, shared across all heads.
// K output: [n_tokens, n_heads, rope_dim+nope_dim] — [pe | nope]
// V output: [n_tokens, n_heads, v_head_dim]

#include "compute/mla_kv_assemble.h"
#include "core/logging.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace imp {

// ---------------------------------------------------------------------------
// Kernel: assemble K[pe|nope] and V from kv_b + k_rope
//   Grid:  (n_heads, n_tokens)
//   Block: 64 threads (sufficient for all practical head dims ≤ 192)
// ---------------------------------------------------------------------------
__global__ static void mla_assemble_kv_kernel(
        const __half* __restrict__ kv_b,    // [n, n_heads*(nope+v)]
        const __half* __restrict__ k_rope,  // [n, rope_dim]
        __half* __restrict__ K_out,         // [n, n_heads, rope+nope]
        __half* __restrict__ V_out,         // [n, n_heads, v_dst_hd]
        int n_heads, int nope_dim, int v_head_dim, int rope_dim, int v_dst_hd)
{
    const int head_dim  = rope_dim + nope_dim;
    const int kv_stride = n_heads * (nope_dim + v_head_dim);

    const int h = blockIdx.x;   // head index
    const int t = blockIdx.y;   // token index

    const __half* kv_b_h  = kv_b   + t * kv_stride + h * (nope_dim + v_head_dim);
    const __half* rope_t  = k_rope + t * rope_dim;
    __half* k_dst = K_out + t * n_heads * head_dim   + h * head_dim;
    __half* v_dst = V_out + t * n_heads * v_dst_hd   + h * v_dst_hd;

    // pe (rope) first — approach (b)
    for (int j = threadIdx.x; j < rope_dim; j += blockDim.x)
        k_dst[j] = rope_t[j];

    // nope after pe
    for (int j = threadIdx.x; j < nope_dim; j += blockDim.x)
        k_dst[rope_dim + j] = kv_b_h[j];

    // V: real values first, then zero-pad the tail when v_dst_hd > v_head_dim
    // (over-allocation so V shares K's head_dim layout for the attention kernels).
    for (int j = threadIdx.x; j < v_dst_hd; j += blockDim.x)
        v_dst[j] = (j < v_head_dim) ? kv_b_h[nope_dim + j] : __float2half(0.0f);
}

// ---------------------------------------------------------------------------
// Kernel: reorder Q per-head from [nope | pe] to [pe | nope] in-place
//   Grid:  (n_heads, n_tokens)
//   Block: 64 threads
//   Smem:  head_dim halfs
// ---------------------------------------------------------------------------
__global__ static void mla_reorder_q_kernel(
        __half* __restrict__ q_data,
        int n_heads, int nope_dim, int rope_dim)
{
    const int head_dim = nope_dim + rope_dim;
    const int h = blockIdx.x;
    const int t = blockIdx.y;

    __half* q_head = q_data + t * n_heads * head_dim + h * head_dim;

    extern __shared__ __half smem[];
    for (int j = threadIdx.x; j < head_dim; j += blockDim.x)
        smem[j] = q_head[j];
    __syncthreads();

    // Write back: [pe | nope]
    for (int j = threadIdx.x; j < rope_dim; j += blockDim.x)
        q_head[j]          = smem[nope_dim + j];
    for (int j = threadIdx.x; j < nope_dim; j += blockDim.x)
        q_head[rope_dim + j] = smem[j];
}

// ---------------------------------------------------------------------------
// Kernel: compact attention output [n, n_heads, head_dim] -> [n, n_heads, v_hd]
//   Grid:  (n_heads, n_tokens)
//   Block: up to 256 threads
// Reads the first v_hd dims of each head's head_dim-strided slot; writes compact.
// ---------------------------------------------------------------------------
__global__ static void mla_compact_attn_output_kernel(
        const __half* __restrict__ src,  // [n, n_heads, head_dim]
        __half* __restrict__ dst,        // [n, n_heads, v_hd]
        int n_heads, int head_dim, int v_hd)
{
    const int h = blockIdx.x;
    const int t = blockIdx.y;
    const __half* s = src + (static_cast<int64_t>(t) * n_heads + h) * head_dim;
    __half* d = dst + (static_cast<int64_t>(t) * n_heads + h) * v_hd;
    for (int j = threadIdx.x; j < v_hd; j += blockDim.x)
        d[j] = s[j];
}

// ---------------------------------------------------------------------------
// Host wrappers
// ---------------------------------------------------------------------------
void mla_assemble_kv(const half* kv_b, const half* k_rope,
                     half* K_out, half* V_out,
                     int n_tokens, int n_heads,
                     int nope_dim, int v_head_dim, int rope_dim,
                     cudaStream_t stream, int v_dst_head_dim) {
    if (n_tokens == 0 || n_heads == 0) return;
    // 0 = compact (per-head stride == v_head_dim); >v_head_dim pads the tail.
    int v_dst_hd = (v_dst_head_dim > 0) ? v_dst_head_dim : v_head_dim;
    dim3 grid(n_heads, n_tokens);
    mla_assemble_kv_kernel<<<grid, 64, 0, stream>>>(
        reinterpret_cast<const __half*>(kv_b),
        reinterpret_cast<const __half*>(k_rope),
        reinterpret_cast<__half*>(K_out),
        reinterpret_cast<__half*>(V_out),
        n_heads, nope_dim, v_head_dim, rope_dim, v_dst_hd);
    IMP_CUDA_CHECK_LAUNCH();
}

void mla_reorder_q(half* q_data, int n_tokens, int n_heads,
                   int nope_dim, int rope_dim, cudaStream_t stream) {
    if (n_tokens == 0 || n_heads == 0) return;
    int head_dim = nope_dim + rope_dim;
    size_t smem = static_cast<size_t>(head_dim) * sizeof(__half);
    dim3 grid(n_heads, n_tokens);
    mla_reorder_q_kernel<<<grid, 64, smem, stream>>>(
        reinterpret_cast<__half*>(q_data),
        n_heads, nope_dim, rope_dim);
    IMP_CUDA_CHECK_LAUNCH();
}

void mla_compact_attn_output(const half* src, half* dst,
                             int n_tokens, int n_heads,
                             int head_dim, int v_head_dim,
                             cudaStream_t stream) {
    if (n_tokens == 0 || n_heads == 0) return;
    int threads = (v_head_dim < 256) ? v_head_dim : 256;
    dim3 grid(n_heads, n_tokens);
    mla_compact_attn_output_kernel<<<grid, threads, 0, stream>>>(
        reinterpret_cast<const __half*>(src),
        reinterpret_cast<__half*>(dst),
        n_heads, head_dim, v_head_dim);
    IMP_CUDA_CHECK_LAUNCH();
}

}  // namespace imp
