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
        __half* __restrict__ V_out,         // [n, n_heads, v_head_dim]
        int n_heads, int nope_dim, int v_head_dim, int rope_dim)
{
    const int head_dim  = rope_dim + nope_dim;
    const int kv_stride = n_heads * (nope_dim + v_head_dim);

    const int h = blockIdx.x;   // head index
    const int t = blockIdx.y;   // token index

    const __half* kv_b_h  = kv_b   + t * kv_stride + h * (nope_dim + v_head_dim);
    const __half* rope_t  = k_rope + t * rope_dim;
    __half* k_dst = K_out + t * n_heads * head_dim  + h * head_dim;
    __half* v_dst = V_out + t * n_heads * v_head_dim + h * v_head_dim;

    // pe (rope) first — approach (b)
    for (int j = threadIdx.x; j < rope_dim; j += blockDim.x)
        k_dst[j] = rope_t[j];

    // nope after pe
    for (int j = threadIdx.x; j < nope_dim; j += blockDim.x)
        k_dst[rope_dim + j] = kv_b_h[j];

    // V
    for (int j = threadIdx.x; j < v_head_dim; j += blockDim.x)
        v_dst[j] = kv_b_h[nope_dim + j];
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
// Host wrappers
// ---------------------------------------------------------------------------
void mla_assemble_kv(const half* kv_b, const half* k_rope,
                     half* K_out, half* V_out,
                     int n_tokens, int n_heads,
                     int nope_dim, int v_head_dim, int rope_dim,
                     cudaStream_t stream) {
    if (n_tokens == 0 || n_heads == 0) return;
    dim3 grid(n_heads, n_tokens);
    mla_assemble_kv_kernel<<<grid, 64, 0, stream>>>(
        reinterpret_cast<const __half*>(kv_b),
        reinterpret_cast<const __half*>(k_rope),
        reinterpret_cast<__half*>(K_out),
        reinterpret_cast<__half*>(V_out),
        n_heads, nope_dim, v_head_dim, rope_dim);
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
}

}  // namespace imp
