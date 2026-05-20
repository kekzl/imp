#include "exec/executor_kernels.h"
#include "exec/gemm_context.h"
#include "exec/gemm_kernel_registry.h"
#include "exec/executor.h"
#include "core/logging.h"
#include "runtime/config.h"
#include "compute/gemm.h"
#include "compute/gemm_q6k.h"
#include "compute/gemm_cutlass_sm120.h"
#include "compute/gemm_cutlass_mxfp4_sm120.h"
#include "compute/hadamard.h"
#include "quant/quant_gemm.h"
#include "quant/dequant_gpu.h"
#include "quant/fp8_quant.h"
#include "quant/nvfp4_gemm.h"
#include "quant/mxfp4_gemm.h"
#include "compute/ggml_mmvq.h"
#include "compute/hadamard.h"
#include "runtime/pdl.h"
#include "compute/ptx92_utils.cuh"
#include "compute/warp_reduce.cuh"  // kWarpSize

namespace imp {

// ---------------------------------------------------------------------------
// Device helpers for paged KV cache block table lookup
// ---------------------------------------------------------------------------

// Resolve the physical block ID from the block table for a given token.
// For batched decode (n_sequences > 1), each token maps to its own sequence row.
// For single-sequence or legacy mode, uses a flat block table.
__device__ __forceinline__ int kv_get_block_id(const int* block_tables, int block_idx, int token_idx,
                                               int max_blocks_per_seq, int n_sequences) {
    if (max_blocks_per_seq > 0 && n_sequences > 1)
        return block_tables[token_idx * max_blocks_per_seq + block_idx];
    return block_tables[block_idx];
}

// Compute block index and slot within block from a token's position.
// Returns the physical block ID via kv_get_block_id.
__device__ __forceinline__ int kv_resolve_slot(const int* block_tables, int pos, int block_size,
                                               int token_idx, int max_blocks_per_seq, int n_sequences,
                                               int& slot_in_block) {
    int block_idx = pos / block_size;
    slot_in_block = pos % block_size;
    return kv_get_block_id(block_tables, block_idx, token_idx, max_blocks_per_seq, n_sequences);
}

// ---------------------------------------------------------------------------
// dp4a GEMV dispatch helper (file-local)
// ---------------------------------------------------------------------------

// Dispatch dp4a GEMV by quant type: y = W @ q8_1 (FP16 output).
// Defined here, declared in executor_kernels.h for use by executor_forward.cu.
void dispatch_dp4a_gemv(QType qtype, const void* W, const block_q8_1* q8_1, const float* d8, half* y, int M,
                        int K, cudaStream_t stream) {
    switch (qtype) {
        case QType::Q6_K:
            gemv_q6k_q8_1(W, q8_1, d8, y, M, K, stream);
            break;
        case QType::Q4_0:
            gemv_q4_0_q8_1(W, q8_1, d8, y, M, K, stream);
            break;
        case QType::Q4_K:
            gemv_q4_k_q8_1(W, q8_1, d8, y, M, K, stream);
            break;
        case QType::Q5_K:
            gemv_q5_k_q8_1(W, q8_1, d8, y, M, K, stream);
            break;
        case QType::Q2_K:
            gemv_q2_k_q8_1(W, q8_1, d8, y, M, K, stream);
            break;
        case QType::Q3_K:
            gemv_q3_k_q8_1(W, q8_1, d8, y, M, K, stream);
            break;
        default:
            gemv_q8_0_q8_1(W, q8_1, d8, y, M, K, stream);
            break;
    }
}

// ---------------------------------------------------------------------------
// Small CUDA kernels used by the executor
// ---------------------------------------------------------------------------

// Broadcast bias addition: out[row, col] += bias[col] for rows x cols elements
__global__ __launch_bounds__(256) void broadcast_add_bias_fp16_kernel(half* __restrict__ out,
                                                                      const half* __restrict__ bias, int rows,
                                                                      int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    for (int i = idx; i < total; i += blockDim.x * gridDim.x) {
        int col = i % cols;
        out[i] = __hadd(out[i], bias[col]);
    }
}

// Element-wise scale: out[i] *= scale, for FP16 data (Gemma embedding scaling)
__global__ __launch_bounds__(256) void scale_fp16_kernel(half* __restrict__ data, half scale, int64_t n) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t n2 = n / 2;
    half2 s2 = __half2half2(scale);
    if (idx < n2) {
        half2* d2 = reinterpret_cast<half2*>(data);
        d2[idx] = __hmul2(d2[idx], s2);
    }
    // Handle odd element
    if (idx == n2 && (n & 1)) {
        data[n - 1] = __hmul(data[n - 1], scale);
    }
}

// Element-wise addition: a[i] += b[i], for FP16 data
__global__ __launch_bounds__(256) void elementwise_add_fp16_kernel(half* __restrict__ a,
                                                                   const half* __restrict__ b, int64_t n) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t n2 = n / 2;
    if (idx < n2) {
        half2* a2 = reinterpret_cast<half2*>(a);
        const half2* b2 = reinterpret_cast<const half2*>(b);
        a2[idx] = __hadd2(a2[idx], b2[idx]);
    }
    if (idx == 0 && (n & 1)) {
        a[n - 1] = __hadd(a[n - 1], b[n - 1]);
    }
}

// Element-wise add-store: out[i] = a[i] + b[i], for FP16 data
__global__ __launch_bounds__(256) void elementwise_add_store_fp16_kernel(const half* __restrict__ a,
                                                                         const half* __restrict__ b,
                                                                         half* __restrict__ out, int64_t n) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t n2 = n / 2;
    if (idx < n2) {
        const half2* a2 = reinterpret_cast<const half2*>(a);
        const half2* b2 = reinterpret_cast<const half2*>(b);
        half2* o2 = reinterpret_cast<half2*>(out);
        o2[idx] = __hadd2(a2[idx], b2[idx]);
    }
    if (idx == 0 && (n & 1)) {
        out[n - 1] = __hadd(a[n - 1], b[n - 1]);
    }
}

// FP32 accumulator += FP16 branch: accum[i] += __half2float(branch[i])
__global__ __launch_bounds__(256) void fp32_accum_add_fp16_kernel(float* __restrict__ accum,
                                                                  const half* __restrict__ branch,
                                                                  int64_t n) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        accum[idx] += __half2float(branch[idx]);
    }
}

// Convert FP32 → FP16 with per-row dynamic scaling.
// Each row is independently scaled so max_abs maps to ≤65000, preserving
// the ratio between elements.  Since subsequent operations (RMSNorm) are
// scale-invariant per row, this produces correct normalized output even
// when the FP32 residual stream far exceeds FP16 range.
// Launch: <<<n_rows, 256, 256 * sizeof(float)>>>
__global__ __launch_bounds__(256) void fp32_to_fp16_rowscale_kernel(const float* __restrict__ in,
                                                                    half* __restrict__ out, int rows,
                                                                    int cols) {
    extern __shared__ float smem[];
    int row = blockIdx.x;
    if (row >= rows)
        return;

    const float* row_in = in + static_cast<int64_t>(row) * cols;
    half* row_out = out + static_cast<int64_t>(row) * cols;

    // Phase 1: parallel reduction to find max |value| in this row
    float local_max = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x)
        local_max = fmaxf(local_max, fabsf(row_in[c]));

    smem[threadIdx.x] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + s]);
        __syncthreads();
    }
    float row_max = smem[0];

    // Only scale if values actually exceed safe FP16 range
    float inv_scale = (row_max > 65000.0f) ? (65000.0f / row_max) : 1.0f;

    // Phase 2: scale and convert to FP16
    for (int c = threadIdx.x; c < cols; c += blockDim.x)
        row_out[c] = __float2half(row_in[c] * inv_scale);
}

// Fused RMSNorm + FP32 accumulator add + FP32→FP16 row-scale conversion.
// Replaces 3 separate kernels in the post-norm FP32 accumulator path:
//   rmsnorm(input, weight, tmp) → fp32_accum_add(accum, tmp) → fp32_to_fp16_rowscale(accum, out)
// Saves 2 kernel launches + 2 DRAM round-trips per invocation.
// Uses same register-cached, warp-level reduction pattern as rmsnorm_quantize_q8_1.
// Launch: <<<n_rows, 256>>>
__global__ __launch_bounds__(512) void rmsnorm_fp32_accum_to_fp16_kernel(
    const half* __restrict__ input,   // [n, d_model] pre-norm data (e.g. GEMV output)
    const half* __restrict__ norm_w,  // [d_model] RMSNorm weights
    float* __restrict__ fp32_accum,   // [n, d_model] FP32 accumulator (read-modify-write)
    half* __restrict__ output,        // [n, d_model] FP16 output for next layer
    int d_model, float eps, float weight_offset) {
    __shared__ float warp_reduce[kWarpSize];  // support up to 1024 threads (32 warps)
    __shared__ float s_inv_rms;
    __shared__ float s_row_max;

    const int tid = threadIdx.x;
    const int lane = tid % kWarpSize;
    const int warp_id = tid / kWarpSize;
    const int n_warps = blockDim.x / kWarpSize;
    const int row = blockIdx.x;

    // Vectorized: process 8 halfs (1 float4 = 2 half2) per iteration.
    const int d_model_v = d_model / 8;  // number of float4-sized chunks

    const float4* x_row4 = reinterpret_cast<const float4*>(input + static_cast<int64_t>(row) * d_model);
    const float4* nw_row4 = reinterpret_cast<const float4*>(norm_w);
    float4* accum_row4 = reinterpret_cast<float4*>(fp32_accum + static_cast<int64_t>(row) * d_model);
    float4* out_row4 = reinterpret_cast<float4*>(output + static_cast<int64_t>(row) * d_model);

    // Phase 1: Load input (half→float via float4 loads), compute sum of squares.
    // Each thread handles d_model_v / blockDim.x chunks, each chunk = 8 halfs.
    float sum_sq = 0.0f;
    for (int i = tid; i < d_model_v; i += blockDim.x) {
        float4 h4 = x_row4[i];  // 8 halfs packed as float4
        const half2* h2 = reinterpret_cast<const half2*>(&h4);
        float2 f0 = __half22float2(h2[0]);
        float2 f1 = __half22float2(h2[1]);
        float2 f2 = __half22float2(h2[2]);
        float2 f3 = __half22float2(h2[3]);
        sum_sq += f0.x * f0.x + f0.y * f0.y + f1.x * f1.x + f1.y * f1.y + f2.x * f2.x + f2.y * f2.y +
                  f3.x * f3.x + f3.y * f3.y;
    }

// Block reduce sum_sq
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        sum_sq += __shfl_xor_sync(0xFFFFFFFF, sum_sq, off);
    if (lane == 0)
        warp_reduce[warp_id] = sum_sq;
    __syncthreads();

    if (warp_id == 0) {
        float total = (lane < n_warps) ? warp_reduce[lane] : 0.0f;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            total += __shfl_xor_sync(0xFFFFFFFF, total, off);
        if (lane == 0)
            s_inv_rms = rsqrtf(total / static_cast<float>(d_model) + eps);
    }
    __syncthreads();
    float inv_rms = s_inv_rms;

    // Phase 2: Normalize, add to FP32 accumulator, find max_abs.
    // Vectorized: read float4 from accum (4 floats), half2×4 from input/norm_w.
    float local_max = 0.0f;
    for (int i = tid; i < d_model_v; i += blockDim.x) {
        // Re-read input (small enough to stay in L1/L2)
        float4 h4 = x_row4[i];
        const half2* h2 = reinterpret_cast<const half2*>(&h4);
        float4 nw4 = nw_row4[i];
        const half2* nw2 = reinterpret_cast<const half2*>(&nw4);

        // Read FP32 accumulator (2 float4s = 8 floats)
        float4 acc_lo = accum_row4[i * 2];
        float4 acc_hi = accum_row4[i * 2 + 1];
        float* acc_f = reinterpret_cast<float*>(&acc_lo);
        float* acc_f_hi = reinterpret_cast<float*>(&acc_hi);

        float2 f0 = __half22float2(h2[0]);
        float2 f1 = __half22float2(h2[1]);
        float2 f2 = __half22float2(h2[2]);
        float2 f3 = __half22float2(h2[3]);
        float2 w0 = __half22float2(nw2[0]);
        float2 w1 = __half22float2(nw2[1]);
        float2 w2 = __half22float2(nw2[2]);
        float2 w3 = __half22float2(nw2[3]);

        acc_f[0] += f0.x * inv_rms * (w0.x + weight_offset);
        acc_f[1] += f0.y * inv_rms * (w0.y + weight_offset);
        acc_f[2] += f1.x * inv_rms * (w1.x + weight_offset);
        acc_f[3] += f1.y * inv_rms * (w1.y + weight_offset);
        acc_f_hi[0] += f2.x * inv_rms * (w2.x + weight_offset);
        acc_f_hi[1] += f2.y * inv_rms * (w2.y + weight_offset);
        acc_f_hi[2] += f3.x * inv_rms * (w3.x + weight_offset);
        acc_f_hi[3] += f3.y * inv_rms * (w3.y + weight_offset);

        accum_row4[i * 2] = acc_lo;
        accum_row4[i * 2 + 1] = acc_hi;

        local_max = fmaxf(local_max, fmaxf(fmaxf(fabsf(acc_f[0]), fabsf(acc_f[1])),
                                           fmaxf(fabsf(acc_f[2]), fabsf(acc_f[3]))));
        local_max = fmaxf(local_max, fmaxf(fmaxf(fabsf(acc_f_hi[0]), fabsf(acc_f_hi[1])),
                                           fmaxf(fabsf(acc_f_hi[2]), fabsf(acc_f_hi[3]))));
    }

// Block reduce max_abs
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        local_max = fmaxf(local_max, __shfl_xor_sync(0xFFFFFFFF, local_max, off));
    if (lane == 0)
        warp_reduce[warp_id] = local_max;
    __syncthreads();

    if (warp_id == 0) {
        float m = (lane < n_warps) ? warp_reduce[lane] : 0.0f;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFF, m, off));
        if (lane == 0)
            s_row_max = m;
    }
    __syncthreads();
    float inv_scale = (s_row_max > 65000.0f) ? (65000.0f / s_row_max) : 1.0f;

    // Phase 3: Scale FP32 accum → FP16 output (vectorized float4 reads, half2×4 writes).
    for (int i = tid; i < d_model_v; i += blockDim.x) {
        float4 acc_lo = accum_row4[i * 2];
        float4 acc_hi = accum_row4[i * 2 + 1];
        float* af = reinterpret_cast<float*>(&acc_lo);
        float* af_hi = reinterpret_cast<float*>(&acc_hi);

        float4 out4;
        half2* oh2 = reinterpret_cast<half2*>(&out4);
        oh2[0] = __floats2half2_rn(af[0] * inv_scale, af[1] * inv_scale);
        oh2[1] = __floats2half2_rn(af[2] * inv_scale, af[3] * inv_scale);
        oh2[2] = __floats2half2_rn(af_hi[0] * inv_scale, af_hi[1] * inv_scale);
        oh2[3] = __floats2half2_rn(af_hi[2] * inv_scale, af_hi[3] * inv_scale);
        out_row4[i] = out4;
    }
}

// FP32-input variant of rmsnorm_fp32_accum_to_fp16_kernel.
// Input is FP32 (e.g. attention output projection kept in FP32 to preserve
// cuBLAS internal accumulator precision). Same accum + overflow protection as
// the FP16-input variant. Used by IMP_GEMMA4_FP32_GEMM_OUT for attention.
__global__ __launch_bounds__(512) void rmsnorm_fp32in_fp32_accum_to_fp16_kernel(
    const float* __restrict__ input,  // [n, d_model] FP32 pre-norm data
    const half* __restrict__ norm_w,  // [d_model] RMSNorm weights
    float* __restrict__ fp32_accum,   // [n, d_model] FP32 accumulator (RMW)
    half* __restrict__ output,        // [n, d_model] FP16 output for next layer
    int d_model, float eps, float weight_offset) {
    __shared__ float warp_reduce[32];
    __shared__ float s_inv_rms;
    __shared__ float s_row_max;

    const int tid = threadIdx.x;
    const int lane = tid % 32;
    const int warp_id = tid / 32;
    const int n_warps = blockDim.x / 32;
    const int row = blockIdx.x;

    const float* x_row = input + static_cast<int64_t>(row) * d_model;
    const half* nw = norm_w;
    float* accum_row = fp32_accum + static_cast<int64_t>(row) * d_model;
    half* out_row = output + static_cast<int64_t>(row) * d_model;

    // Phase 1: sum of squares (input already FP32)
    float sum_sq = 0.0f;
    for (int i = tid; i < d_model; i += blockDim.x) {
        float v = x_row[i];
        sum_sq += v * v;
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        sum_sq += __shfl_xor_sync(0xFFFFFFFF, sum_sq, off);
    if (lane == 0)
        warp_reduce[warp_id] = sum_sq;
    __syncthreads();
    if (warp_id == 0) {
        float total = (lane < n_warps) ? warp_reduce[lane] : 0.0f;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            total += __shfl_xor_sync(0xFFFFFFFF, total, off);
        if (lane == 0)
            s_inv_rms = rsqrtf(total / static_cast<float>(d_model) + eps);
    }
    __syncthreads();
    float inv_rms = s_inv_rms;

    // Phase 2: accum += norm(x) * weight; track max
    float local_max = 0.0f;
    for (int i = tid; i < d_model; i += blockDim.x) {
        float v = x_row[i];
        float w = __half2float(nw[i]) + weight_offset;
        float val = v * inv_rms * w;
        float new_acc = accum_row[i] + val;
        accum_row[i] = new_acc;
        local_max = fmaxf(local_max, fabsf(new_acc));
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        local_max = fmaxf(local_max, __shfl_xor_sync(0xFFFFFFFF, local_max, off));
    if (lane == 0)
        warp_reduce[warp_id] = local_max;
    __syncthreads();
    if (warp_id == 0) {
        float m = (lane < n_warps) ? warp_reduce[lane] : 0.0f;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFF, m, off));
        if (lane == 0)
            s_row_max = m;
    }
    __syncthreads();
    float inv_scale = (s_row_max > 65000.0f) ? (65000.0f / s_row_max) : 1.0f;

    // Phase 3: write FP16 output (with overflow scaling)
    for (int i = tid; i < d_model; i += blockDim.x) {
        out_row[i] = __float2half(accum_row[i] * inv_scale);
    }
}

// Convert FP16 → FP32: out[i] = __half2float(in[i])
__global__ __launch_bounds__(256) void fp16_to_fp32_kernel(const half* __restrict__ in,
                                                           float* __restrict__ out, int64_t n) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __half2float(in[idx]);
    }
}

// Element-wise addition: a[i] += b[i], for FP32 data
__global__ __launch_bounds__(256) void elementwise_add_fp32_kernel(float* __restrict__ a,
                                                                   const float* __restrict__ b, int64_t n) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    for (int64_t i = idx; i < n; i += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        a[i] += b[i];
    }
}

// Copy K/V for a set of tokens into paged KV cache blocks.
// Each token's K (or V) slice is copied to the correct slot in the right block.
//
// data_in:          [n_tokens, n_kv_heads * head_dim] contiguous
// positions:        [n_tokens] position of each token in the sequence
// block_tables:     [n_sequences, max_blocks_per_seq] or [max_blocks] block IDs
// cache_base:       base pointer of the KV pool for this layer (block 0)
// block_stride:     elements per block = kKVBlockSize * n_kv_heads * head_dim
// row_elems:        n_kv_heads * head_dim (elements per token)
// max_blocks_per_seq: stride for 2D block table (0 = legacy flat)
// n_sequences:      number of sequences in the batch
__global__ __launch_bounds__(256) void write_kv_cache_kernel(const half* __restrict__ data_in,
                                                             const int* __restrict__ positions,
                                                             const int* __restrict__ block_tables,
                                                             half* __restrict__ cache_base, int block_stride,
                                                             int row_elems, int block_size, int n_tokens,
                                                             int max_blocks_per_seq, int n_sequences) {
    int token_idx = blockIdx.x;
    if (token_idx >= n_tokens)
        return;

    int pos = positions[token_idx];
    int slot_in_block;
    int block_id = kv_resolve_slot(block_tables, pos, block_size, token_idx, max_blocks_per_seq, n_sequences,
                                   slot_in_block);

    half* dst = cache_base + static_cast<int64_t>(block_id) * block_stride +
                static_cast<int64_t>(slot_in_block) * row_elems;
    const half* src = data_in + static_cast<int64_t>(token_idx) * row_elems;

    // Vectorized 128-bit copy (8 FP16 per store) — row_elems is always a
    // multiple of 8 (n_kv_heads * head_dim, where head_dim is power of 2).
    const int vec_elems = row_elems / 8;
    const float4* src4 = reinterpret_cast<const float4*>(src);
    float4* dst4 = reinterpret_cast<float4*>(dst);
    for (int i = threadIdx.x; i < vec_elems; i += blockDim.x) {
        dst4[i] = src4[i];
    }
}

// Fused K+V write to paged KV cache in a single launch.
// blockIdx.x = token index, blockIdx.y = 0 (K) or 1 (V).
// Saves one kernel launch per attention layer.
__global__ __launch_bounds__(256) void write_kv_cache_fused_kernel(
    const half* __restrict__ k_in,  // [n_tokens, n_kv_heads * head_dim]
    const half* __restrict__ v_in,  // [n_tokens, n_kv_heads * head_dim]
    const int* __restrict__ positions, const int* __restrict__ block_tables, half* __restrict__ k_cache_base,
    half* __restrict__ v_cache_base, int block_stride, int row_elems, int block_size, int n_tokens,
    int max_blocks_per_seq, int n_sequences) {
    int token_idx = blockIdx.x;
    if (token_idx >= n_tokens)
        return;

    int pos = positions[token_idx];
    int slot_in_block;
    int block_id = kv_resolve_slot(block_tables, pos, block_size, token_idx, max_blocks_per_seq, n_sequences,
                                   slot_in_block);

    // blockIdx.y selects K (0) or V (1)
    const half* src;
    half* dst_base;
    if (blockIdx.y == 0) {
        src = k_in + static_cast<int64_t>(token_idx) * row_elems;
        dst_base = k_cache_base;
    } else {
        src = v_in + static_cast<int64_t>(token_idx) * row_elems;
        dst_base = v_cache_base;
    }

    half* dst = dst_base + static_cast<int64_t>(block_id) * block_stride +
                static_cast<int64_t>(slot_in_block) * row_elems;

    // Vectorized 128-bit copy (8 FP16 per store)
    const int vec_elems = row_elems / 8;
    const float4* src4 = reinterpret_cast<const float4*>(src);
    float4* dst4 = reinterpret_cast<float4*>(dst);
    for (int i = threadIdx.x; i < vec_elems; i += blockDim.x) {
        dst4[i] = src4[i];
    }
}

// FP16 -> FP8 E4M3 quantization + write to paged KV cache
__global__ __launch_bounds__(256) void write_kv_cache_fp8_kernel(
    const half* __restrict__ data_in, const int* __restrict__ positions, const int* __restrict__ block_tables,
    __nv_fp8_e4m3* __restrict__ cache_base,  // FP8 cache
    float inv_scale,                         // 1.0 / kv_scale
    int block_stride, int row_elems, int block_size, int n_tokens, int max_blocks_per_seq, int n_sequences) {
    int token_idx = blockIdx.x;
    if (token_idx >= n_tokens)
        return;

    int pos = positions[token_idx];
    int slot_in_block;
    int block_id = kv_resolve_slot(block_tables, pos, block_size, token_idx, max_blocks_per_seq, n_sequences,
                                   slot_in_block);

    __nv_fp8_e4m3* dst = cache_base + static_cast<int64_t>(block_id) * block_stride +
                         static_cast<int64_t>(slot_in_block) * row_elems;
    const half* src = data_in + static_cast<int64_t>(token_idx) * row_elems;

    // Packed PTX cvt: 2 paired conversions per 4 elements (half→e4m3x2).
    // Scale applied in FP16 before conversion — sufficient precision for E4M3.
    const half inv_scale_h = __float2half(inv_scale);
    const half2 inv_scale_h2 = make_half2(inv_scale_h, inv_scale_h);
    const int vec_elems = row_elems / 4;
    const half2* src2 = reinterpret_cast<const half2*>(src);
    uint32_t* dst4 = reinterpret_cast<uint32_t*>(dst);
    for (int i = threadIdx.x; i < vec_elems; i += blockDim.x) {
        half2 lo = __hmul2(src2[2 * i], inv_scale_h2);
        half2 hi = __hmul2(src2[2 * i + 1], inv_scale_h2);
        uint16_t e4m3_lo = cvt_f16x2_to_e4m3x2(*reinterpret_cast<uint32_t*>(&lo));
        uint16_t e4m3_hi = cvt_f16x2_to_e4m3x2(*reinterpret_cast<uint32_t*>(&hi));
        dst4[i] = static_cast<uint32_t>(e4m3_lo) | (static_cast<uint32_t>(e4m3_hi) << 16);
    }
    // Scalar tail for non-aligned remainder
    for (int i = vec_elems * 4 + threadIdx.x; i < row_elems; i += blockDim.x) {
        dst[i] = __nv_fp8_e4m3(__half2float(src[i]) * inv_scale);
    }
}

// ---------------------------------------------------------------------------
// FP16 -> INT8 quantization + write to paged KV cache with per-head scales.
// Each warp processes one KV head independently: compute absmax via warp shuffle,
// then quantize and write int8 data + half scale.
//
// blockIdx.x = token_idx, blockIdx.y = 0 (K) or 1 (V).
// blockDim.x = 256 (8 warps). Each warp loops over heads.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256) void write_kv_cache_int8_kernel(
    const half* __restrict__ k_in,  // [n_tokens, n_kv_heads * head_dim]
    const half* __restrict__ v_in, const int* __restrict__ positions, const int* __restrict__ block_tables,
    int8_t* __restrict__ k_cache_base, int8_t* __restrict__ v_cache_base,
    half* __restrict__ k_scale_base,  // [total_blocks, kKVBlockSize, n_kv_heads]
    half* __restrict__ v_scale_base,
    int block_stride,        // kKVBlockSize * n_kv_heads * head_dim (int8 elems)
    int scale_block_stride,  // kKVBlockSize * n_kv_heads (half elems)
    int n_kv_heads, int head_dim, int block_size, int n_tokens, int max_blocks_per_seq, int n_sequences) {
    const int token_idx = blockIdx.x;
    if (token_idx >= n_tokens)
        return;

    const int pos = positions[token_idx];
    int slot_in_block;
    int block_id = kv_resolve_slot(block_tables, pos, block_size, token_idx, max_blocks_per_seq, n_sequences,
                                   slot_in_block);

    // Select K or V based on blockIdx.y
    const half* src_base = (blockIdx.y == 0) ? k_in : v_in;
    int8_t* cache_base = (blockIdx.y == 0) ? k_cache_base : v_cache_base;
    half* scale_base = (blockIdx.y == 0) ? k_scale_base : v_scale_base;

    const int row_elems = n_kv_heads * head_dim;
    const half* src = src_base + static_cast<int64_t>(token_idx) * row_elems;
    int8_t* dst = cache_base + static_cast<int64_t>(block_id) * block_stride +
                  static_cast<int64_t>(slot_in_block) * row_elems;
    half* scale_dst = scale_base + static_cast<int64_t>(block_id) * scale_block_stride +
                      static_cast<int64_t>(slot_in_block) * n_kv_heads;

    const int warp_id = threadIdx.x / kWarpSize;
    const int lane_id = threadIdx.x % kWarpSize;
    const int num_warps = blockDim.x / kWarpSize;

    // Each warp processes one head at a time, looping over heads
    for (int h = warp_id; h < n_kv_heads; h += num_warps) {
        const int head_offset = h * head_dim;

        // Step 1: Load FP16 values and compute per-head absmax
        // Vectorized: load 2 FP16 per iteration via half2 for better coalescing.
        // head_dim is always even (64, 128, 256).
        float amax = 0.0f;
        const half2* src2 = reinterpret_cast<const half2*>(src + head_offset);
        for (int d = lane_id; d < head_dim / 2; d += kWarpSize) {
            half2 h2 = src2[d];
            amax = fmaxf(amax, fabsf(__half2float(h2.x)));
            amax = fmaxf(amax, fabsf(__half2float(h2.y)));
        }
// Warp-level absmax reduction
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset));

        // Step 2: Compute scale
        float sc = amax / 127.0f;
        float inv_sc = (amax > 1e-8f) ? (127.0f / amax) : 0.0f;

        // Step 3: Quantize and write int8 data (vectorized: load 4 FP16 via float2,
        // store 4 INT8 via uint32_t). head_dim is always a multiple of 4 (64, 128, 256).
        // Each lane processes 4 consecutive elements per iteration, stride = 32*4 = 128.
        const float2* src_head4 = reinterpret_cast<const float2*>(src + head_offset);
        for (int d4 = lane_id; d4 < head_dim / 4; d4 += 32) {
            float2 h2 = src_head4[d4];
            const half* hp = reinterpret_cast<const half*>(&h2);
            uint32_t packed;
            int8_t* p = reinterpret_cast<int8_t*>(&packed);
#pragma unroll
            for (int j = 0; j < 4; j++) {
                p[j] = static_cast<int8_t>(__float2int_rn(__half2float(hp[j]) * inv_sc));
            }
            reinterpret_cast<uint32_t*>(dst + head_offset)[d4] = packed;
        }

        // Step 4: Write scale (one half per head per token)
        if (lane_id == 0) {
            scale_dst[h] = __float2half(sc);
        }
    }
}

// ---------------------------------------------------------------------------
// INT4 KV cache write: FP16 → 4-bit symmetric quantization with per-head scales.
// Two INT4 values packed into one byte (low nibble = even index, high nibble = odd).
// Range: [-8, 7] symmetric. Scale = absmax / 7.0.
// blockIdx.x = token, blockIdx.y = 0 (K) or 1 (V).
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256) void write_kv_cache_int4_kernel(
    const half* __restrict__ k_in, const half* __restrict__ v_in, const int* __restrict__ positions,
    const int* __restrict__ block_tables,
    uint8_t* __restrict__ k_cache_base,  // packed INT4 pairs
    uint8_t* __restrict__ v_cache_base, half* __restrict__ k_scale_base, half* __restrict__ v_scale_base,
    int block_stride,        // kKVBlockSize * n_kv_heads * head_dim / 2 (bytes)
    int scale_block_stride,  // kKVBlockSize * n_kv_heads (half elems)
    int n_kv_heads, int head_dim, int block_size, int n_tokens, int max_blocks_per_seq, int n_sequences) {
    const int token_idx = blockIdx.x;
    if (token_idx >= n_tokens)
        return;

    const int pos = positions[token_idx];
    int slot_in_block;
    int block_id = kv_resolve_slot(block_tables, pos, block_size, token_idx, max_blocks_per_seq, n_sequences,
                                   slot_in_block);

    const half* src_base = (blockIdx.y == 0) ? k_in : v_in;
    uint8_t* cache_base = (blockIdx.y == 0) ? k_cache_base : v_cache_base;
    half* scale_base = (blockIdx.y == 0) ? k_scale_base : v_scale_base;

    const int row_elems = n_kv_heads * head_dim;
    const int row_bytes = row_elems / 2;  // 2 INT4 values per byte
    const half* src = src_base + static_cast<int64_t>(token_idx) * row_elems;
    uint8_t* dst = cache_base + static_cast<int64_t>(block_id) * block_stride +
                   static_cast<int64_t>(slot_in_block) * row_bytes;
    half* scale_dst = scale_base + static_cast<int64_t>(block_id) * scale_block_stride +
                      static_cast<int64_t>(slot_in_block) * n_kv_heads;

    const int warp_id = threadIdx.x / kWarpSize;
    const int lane_id = threadIdx.x % kWarpSize;
    const int num_warps = blockDim.x / kWarpSize;

    for (int h = warp_id; h < n_kv_heads; h += num_warps) {
        const int head_offset = h * head_dim;

        // Step 1: Per-head absmax
        float amax = 0.0f;
        for (int d = lane_id; d < head_dim; d += kWarpSize) {
            float val = __half2float(src[head_offset + d]);
            amax = fmaxf(amax, fabsf(val));
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset));

        // Step 2: Scale (symmetric INT4: [-8, 7], use 7 for range)
        float sc = amax / 7.0f;
        float inv_sc = (amax > 1e-8f) ? (7.0f / amax) : 0.0f;

        // Step 3: Quantize and pack pairs into bytes
        // Each lane handles 2 elements at a time (d, d+1) → 1 byte
        const int head_byte_offset = h * head_dim / 2;
        for (int d = lane_id * 2; d < head_dim; d += 2 * kWarpSize) {
            float v0 = __half2float(src[head_offset + d]);
            float v1 = (d + 1 < head_dim) ? __half2float(src[head_offset + d + 1]) : 0.0f;

            int q0 = __float2int_rn(v0 * inv_sc);
            int q1 = __float2int_rn(v1 * inv_sc);
            q0 = max(-8, min(7, q0));
            q1 = max(-8, min(7, q1));

            // Pack: low nibble = q0, high nibble = q1
            uint8_t packed = (static_cast<uint8_t>(q0 & 0xF)) | (static_cast<uint8_t>(q1 & 0xF) << 4);
            dst[head_byte_offset + d / 2] = packed;
        }

        // Step 4: Write scale
        if (lane_id == 0) {
            scale_dst[h] = __float2half(sc);
        }
    }
}

// ---------------------------------------------------------------------------
// NVFP4 KV cache write kernel
// Per (token, head, group of 16 elems along head_dim):
//   1. absmax over 16 elems
//   2. scale = absmax / 6.0  (FP4 E2M1 max = 6.0); store as UE4M3 byte
//   3. quant each elem to E2M1 nibble (nearest-magnitude + sign), pack 2/byte
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint8_t e2m1_quantize(float v, float inv_scale) {
    float n = v * inv_scale;
    uint8_t sign = (n < 0.0f) ? 0x8u : 0u;
    float m = fabsf(n);
    // Nearest in {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}.
    // Tested midpoint boundaries: 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0
    uint8_t mag;
    if (m < 0.25f)
        mag = 0;
    else if (m < 0.75f)
        mag = 1;
    else if (m < 1.25f)
        mag = 2;
    else if (m < 1.75f)
        mag = 3;
    else if (m < 2.5f)
        mag = 4;
    else if (m < 3.5f)
        mag = 5;
    else if (m < 5.0f)
        mag = 6;
    else
        mag = 7;
    return sign | mag;
}

__global__ __launch_bounds__(256) void write_kv_cache_nvfp4_kernel(
    const half* __restrict__ k_in, const half* __restrict__ v_in, const int* __restrict__ positions,
    const int* __restrict__ block_tables, uint8_t* __restrict__ k_cache_base,
    uint8_t* __restrict__ v_cache_base, uint8_t* __restrict__ k_scale_base,
    uint8_t* __restrict__ v_scale_base, int block_stride, int scale_block_stride, int n_kv_heads,
    int head_dim, int block_size, int n_tokens, int max_blocks_per_seq, int n_sequences) {
    constexpr int kGroup = 16;

    const int token_idx = blockIdx.x;
    if (token_idx >= n_tokens)
        return;

    const int pos = positions[token_idx];
    int slot_in_block;
    int block_id = kv_resolve_slot(block_tables, pos, block_size, token_idx, max_blocks_per_seq, n_sequences,
                                   slot_in_block);

    const half* src_base = (blockIdx.y == 0) ? k_in : v_in;
    uint8_t* cache_base = (blockIdx.y == 0) ? k_cache_base : v_cache_base;
    uint8_t* scale_base = (blockIdx.y == 0) ? k_scale_base : v_scale_base;

    const int row_elems = n_kv_heads * head_dim;
    const int row_bytes = row_elems / 2;
    const int row_scale_bytes = n_kv_heads * (head_dim / kGroup);
    const half* src = src_base + static_cast<int64_t>(token_idx) * row_elems;
    uint8_t* dst = cache_base + static_cast<int64_t>(block_id) * block_stride +
                   static_cast<int64_t>(slot_in_block) * row_bytes;
    uint8_t* scale_dst = scale_base + static_cast<int64_t>(block_id) * scale_block_stride +
                         static_cast<int64_t>(slot_in_block) * row_scale_bytes;

    const int n_groups_per_head = head_dim / kGroup;
    const int total_groups = n_kv_heads * n_groups_per_head;

    // One thread per group (each group = 16 elems = 8 bytes packed FP4 + 1 UE4M3 scale byte).
    for (int g = threadIdx.x; g < total_groups; g += blockDim.x) {
        int h = g / n_groups_per_head;
        int gh = g % n_groups_per_head;             // group within head
        int base_elem = h * head_dim + gh * kGroup;  // first elem in this group

        // absmax
        float amax = 0.0f;
#pragma unroll
        for (int i = 0; i < kGroup; i++) {
            float v = __half2float(src[base_elem + i]);
            amax = fmaxf(amax, fabsf(v));
        }
        float sc = amax / 6.0f;
        float inv_sc = (sc > 1e-30f) ? (1.0f / sc) : 0.0f;

        // pack 16 nibbles → 8 bytes
        int dst_byte_off = h * (head_dim / 2) + gh * (kGroup / 2);
#pragma unroll
        for (int p = 0; p < kGroup / 2; p++) {
            float v0 = __half2float(src[base_elem + 2 * p]);
            float v1 = __half2float(src[base_elem + 2 * p + 1]);
            uint8_t q0 = e2m1_quantize(v0, inv_sc);
            uint8_t q1 = e2m1_quantize(v1, inv_sc);
            dst[dst_byte_off + p] = static_cast<uint8_t>(q0 | (q1 << 4));
        }

        // store UE4M3 scale (saturates oversize values, 0 if amax==0)
        __nv_fp8_e4m3 ue4m3(sc);
        scale_dst[h * n_groups_per_head + gh] = *reinterpret_cast<uint8_t*>(&ue4m3);
    }
}

// ---------------------------------------------------------------------------
// BitDecoding Phase 3c: FP16 residual ring write.
//
// blockIdx.x = token_idx (0..n_tokens-1); blockIdx.y selects K (0) or V (1).
// blockDim.x threads stripe across slot_elems = n_kv_heads * head_dim. The
// per-token destination pointer (already resolved on the host to the right
// (seq_slot, layer, K|V, ring_slot) location) is read from the per-token
// pointer array.
//
// Replaces a pair of `cudaMemcpyAsync(dst, src, slot_elems*sizeof(half),
// cudaMemcpyDeviceToDevice, stream)` calls per layer, which were observed
// to serialize on the copy engine and dominate decode tg/s when residual
// was enabled (-3× regression on Qwen3-4B Q8 NVFP4-KV bench at 4K ctx).
// ---------------------------------------------------------------------------
__global__ void residual_kv_write_single_kernel(
    const half* __restrict__ k_in,
    const half* __restrict__ v_in,
    half* __restrict__ residual_k_dst,
    half* __restrict__ residual_v_dst,
    int slot_elems) {
    const bool is_v = (blockIdx.x == 1);
    half* dst = is_v ? residual_v_dst : residual_k_dst;
    if (dst == nullptr) return;
    const half* src = is_v ? v_in : k_in;
    const int i = threadIdx.x + blockIdx.y * blockDim.x;
    if (i < slot_elems) {
        dst[i] = src[i];
    }
}

__global__ void residual_kv_write_multi_kernel(
    const half* __restrict__ k_in,
    const half* __restrict__ v_in,
    half* const* __restrict__ residual_k_dst_ptrs,
    half* const* __restrict__ residual_v_dst_ptrs,
    int slot_elems) {
    const int token_idx = blockIdx.x;
    const bool is_v = (blockIdx.y == 1);

    half* dst = is_v ? residual_v_dst_ptrs[token_idx] : residual_k_dst_ptrs[token_idx];
    if (dst == nullptr) return;
    const half* src = (is_v ? v_in : k_in) + static_cast<int64_t>(token_idx) * slot_elems;

    for (int i = threadIdx.x; i < slot_elems; i += blockDim.x) {
        dst[i] = src[i];
    }
}

// Graph-safe single-seq variant: reads write_idx from a device pointer at
// kernel execution time, so the captured kernel sees the current ring state
// across graph replays. blockIdx.x ∈ {0, 1} selects K or V; threads stripe
// across slot_elems.
__global__ void residual_kv_write_indirect_kernel(
    const half* __restrict__ k_in,
    const half* __restrict__ v_in,
    half* __restrict__ residual_k_layer_seq_base,
    half* __restrict__ residual_v_layer_seq_base,
    const int* __restrict__ d_residual_widx_ptr,
    int seq_slot,
    int slot_elems) {
    const bool is_v = (blockIdx.x == 1);
    half* base = is_v ? residual_v_layer_seq_base : residual_k_layer_seq_base;
    if (base == nullptr) return;
    const half* src = is_v ? v_in : k_in;
    const int widx = d_residual_widx_ptr[seq_slot];
    half* dst = base + static_cast<int64_t>(widx) * slot_elems;

    const int i = threadIdx.x + blockIdx.y * blockDim.x;
    if (i < slot_elems) {
        dst[i] = src[i];
    }
}

__global__ void advance_residual_state_kernel(
    int* __restrict__ d_widx,
    int* __restrict__ d_fc,
    int slot,
    int residual_n_tokens) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int w = d_widx[slot];
        int f = d_fc[slot];
        d_widx[slot] = (w + 1) % residual_n_tokens;
        d_fc[slot] = (f < residual_n_tokens) ? (f + 1) : f;
    }
}

// FP4/UE8M0 helpers — shared by write_kv_cache_mxfp4_kv_kernel and
// attention_paged_nvfp4.cu (decode_kv_scale<UE8M0> specialization).
// Close namespace before including (header defines in imp::).
}  // namespace imp
#include "quant/turboquant_fp4.cuh"
namespace imp {

// Aliases for shorter names in write_kv_cache_mxfp4_kv_kernel below.
#define tq_float_to_ue8m0 tq_fp4_float_to_ue8m0
#define tq_ue8m0_to_float tq_fp4_ue8m0_to_float


// ---------------------------------------------------------------------------
// MXFP4-KV write kernel: same layout as NVFP4 but stores UE8M0 scale bytes.
//
// The only difference from write_kv_cache_nvfp4_kernel is the scale encoding:
//   NVFP4:    `__nv_fp8_e4m3 ue4m3(sc); scale_dst[...] = reinterpret_cast<uint8_t>(&ue4m3);`
//   MXFP4_KV: `scale_dst[...] = tq_float_to_ue8m0(sc);`   (pure-exponent 8-bit)
//
// All other fields (block_stride, scale_block_stride, FP4 packing, group size)
// are identical. The tq_float_to_ue8m0 alias is defined at line 1084.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256) void write_kv_cache_mxfp4_kv_kernel(
    const half* __restrict__ k_in, const half* __restrict__ v_in, const int* __restrict__ positions,
    const int* __restrict__ block_tables, uint8_t* __restrict__ k_cache_base,
    uint8_t* __restrict__ v_cache_base, uint8_t* __restrict__ k_scale_base,
    uint8_t* __restrict__ v_scale_base, int block_stride, int scale_block_stride, int n_kv_heads,
    int head_dim, int block_size, int n_tokens, int max_blocks_per_seq, int n_sequences) {
    constexpr int kGroup = 16;

    const int token_idx = blockIdx.x;
    if (token_idx >= n_tokens)
        return;

    const int pos = positions[token_idx];
    int slot_in_block;
    int block_id = kv_resolve_slot(block_tables, pos, block_size, token_idx, max_blocks_per_seq, n_sequences,
                                   slot_in_block);

    const half* src_base = (blockIdx.y == 0) ? k_in : v_in;
    uint8_t* cache_base = (blockIdx.y == 0) ? k_cache_base : v_cache_base;
    uint8_t* scale_base = (blockIdx.y == 0) ? k_scale_base : v_scale_base;

    const int row_elems = n_kv_heads * head_dim;
    const int row_bytes = row_elems / 2;
    const int row_scale_bytes = n_kv_heads * (head_dim / kGroup);
    const half* src = src_base + static_cast<int64_t>(token_idx) * row_elems;
    uint8_t* dst = cache_base + static_cast<int64_t>(block_id) * block_stride +
                   static_cast<int64_t>(slot_in_block) * row_bytes;
    uint8_t* scale_dst = scale_base + static_cast<int64_t>(block_id) * scale_block_stride +
                         static_cast<int64_t>(slot_in_block) * row_scale_bytes;

    const int n_groups_per_head = head_dim / kGroup;
    const int total_groups = n_kv_heads * n_groups_per_head;

    for (int g = threadIdx.x; g < total_groups; g += blockDim.x) {
        int h = g / n_groups_per_head;
        int gh = g % n_groups_per_head;
        int base_elem = h * head_dim + gh * kGroup;

        // absmax
        float amax = 0.0f;
#pragma unroll
        for (int i = 0; i < kGroup; i++) {
            float v = __half2float(src[base_elem + i]);
            amax = fmaxf(amax, fabsf(v));
        }
        float sc_exact = amax / 6.0f;
        // Round-trip-consistent scale: quantize to UE8M0 first, then use the
        // ACTUAL decoded scale for nibble quantization. The NVFP4 write kernel
        // gets away with using sc_exact directly because E4M3's mantissa keeps
        // the rounding error ~1.5%, but UE8M0 is power-of-2 only (up to 2x
        // rounding error per group) — a mismatch between encoder/decoder
        // scales compounds catastrophically over 32 layers (degenerate output
        // observed in Phase 2 NIAH re-run, 0% retrieval even at 4K context).
        uint8_t sc_byte = tq_float_to_ue8m0(sc_exact);
        float sc_actual = tq_ue8m0_to_float(sc_byte);
        float inv_sc = (sc_actual > 1e-30f) ? (1.0f / sc_actual) : 0.0f;

        // pack 16 nibbles → 8 bytes
        int dst_byte_off = h * (head_dim / 2) + gh * (kGroup / 2);
#pragma unroll
        for (int p = 0; p < kGroup / 2; p++) {
            float v0 = __half2float(src[base_elem + 2 * p]);
            float v1 = __half2float(src[base_elem + 2 * p + 1]);
            uint8_t q0 = e2m1_quantize(v0, inv_sc);
            uint8_t q1 = e2m1_quantize(v1, inv_sc);
            dst[dst_byte_off + p] = static_cast<uint8_t>(q0 | (q1 << 4));
        }

        scale_dst[h * n_groups_per_head + gh] = sc_byte;
    }
}

// Fused KV cache write with RoPE on K: applies RoPE to K during write, copies V directly.
// blockIdx.x = token index, blockIdx.y = 0 (K+RoPE) or 1 (V copy).
// Eliminates the separate RoPE kernel launch for K in the decode path.
__global__ __launch_bounds__(256) void write_kv_cache_rope_fused_kernel(
    const half* __restrict__ k_in,  // [n_tokens, n_kv_heads * head_dim] raw K (no RoPE)
    const half* __restrict__ v_in,  // [n_tokens, n_kv_heads * head_dim]
    const int* __restrict__ positions, const int* __restrict__ block_tables, half* __restrict__ k_cache_base,
    half* __restrict__ v_cache_base, int block_stride, int row_elems, int block_size, int n_tokens,
    int max_blocks_per_seq, int n_sequences, int n_kv_heads, int head_dim, float theta, float inv_scaling,
    int rope_pairs,  // effective_rope_dim / 2
    bool neox, const float* __restrict__ longrope_inv_freqs) {
    int token_idx = blockIdx.x;
    if (token_idx >= n_tokens)
        return;

    int pos = positions[token_idx];
    int slot_in_block;
    int block_id = kv_resolve_slot(block_tables, pos, block_size, token_idx, max_blocks_per_seq, n_sequences,
                                   slot_in_block);

    if (blockIdx.y == 0) {
        // K path: apply RoPE during write
        const half* k_src = k_in + static_cast<int64_t>(token_idx) * row_elems;
        half* k_dst = k_cache_base + static_cast<int64_t>(block_id) * block_stride +
                      static_cast<int64_t>(slot_in_block) * row_elems;

        // Process RoPE pairs
        int total_pairs = n_kv_heads * rope_pairs;
        for (int p = threadIdx.x; p < total_pairs; p += blockDim.x) {
            int head = p / rope_pairs;
            int pair_idx = p % rope_pairs;
            int head_offset = head * head_dim;

            int idx0, idx1;
            if (neox) {
                idx0 = head_offset + pair_idx;
                idx1 = head_offset + pair_idx + rope_pairs;
            } else {
                idx0 = head_offset + 2 * pair_idx;
                idx1 = head_offset + 2 * pair_idx + 1;
            }

            float freq;
            if (longrope_inv_freqs) {
                // Pre-computed effective frequencies (see gguf_loader.cpp rope_freqs conversion)
                freq = longrope_inv_freqs[pair_idx];
            } else {
                freq = 1.0f / (powf(theta, (2.0f * pair_idx) / static_cast<float>(2 * rope_pairs)));
                freq *= inv_scaling;
            }
            float angle = static_cast<float>(pos) * freq;
            float cos_val = __cosf(angle);
            float sin_val = __sinf(angle);

            float k0 = __half2float(k_src[idx0]);
            float k1 = __half2float(k_src[idx1]);
            k_dst[idx0] = __float2half(k0 * cos_val - k1 * sin_val);
            k_dst[idx1] = __float2half(k0 * sin_val + k1 * cos_val);
        }

        // Copy non-rotated dimensions (partial RoPE: rope_dim < head_dim)
        int effective_rope_dim = rope_pairs * 2;
        if (effective_rope_dim < head_dim) {
            for (int h = 0; h < n_kv_heads; h++) {
                int base = h * head_dim;
                for (int d = effective_rope_dim + threadIdx.x; d < head_dim; d += blockDim.x) {
                    k_dst[base + d] = k_src[base + d];
                }
            }
        }
    } else {
        // V path: vectorized 128-bit copy (no RoPE)
        const half* v_src = v_in + static_cast<int64_t>(token_idx) * row_elems;
        half* v_dst = v_cache_base + static_cast<int64_t>(block_id) * block_stride +
                      static_cast<int64_t>(slot_in_block) * row_elems;
        const int vec_elems = row_elems / 8;
        const float4* vs4 = reinterpret_cast<const float4*>(v_src);
        float4* vd4 = reinterpret_cast<float4*>(v_dst);
        for (int i = threadIdx.x; i < vec_elems; i += blockDim.x) {
            vd4[i] = vs4[i];
        }
    }
}

// Fused K+V FP8 write: combines K and V quantize+write into one kernel launch.
// blockIdx.x = token index, blockIdx.y = 0 (K) or 1 (V).
__global__ __launch_bounds__(256) void write_kv_cache_fp8_fused_kernel(
    const half* __restrict__ k_in, const half* __restrict__ v_in, const int* __restrict__ positions,
    const int* __restrict__ block_tables, __nv_fp8_e4m3* __restrict__ k_cache_base,
    __nv_fp8_e4m3* __restrict__ v_cache_base, float inv_scale, int block_stride, int row_elems,
    int block_size, int n_tokens, int max_blocks_per_seq, int n_sequences) {
    int token_idx = blockIdx.x;
    if (token_idx >= n_tokens)
        return;

    int pos = positions[token_idx];
    int slot_in_block;
    int block_id = kv_resolve_slot(block_tables, pos, block_size, token_idx, max_blocks_per_seq, n_sequences,
                                   slot_in_block);

    const half* src;
    __nv_fp8_e4m3* dst;
    if (blockIdx.y == 0) {
        src = k_in + static_cast<int64_t>(token_idx) * row_elems;
        dst = k_cache_base + static_cast<int64_t>(block_id) * block_stride +
              static_cast<int64_t>(slot_in_block) * row_elems;
    } else {
        src = v_in + static_cast<int64_t>(token_idx) * row_elems;
        dst = v_cache_base + static_cast<int64_t>(block_id) * block_stride +
              static_cast<int64_t>(slot_in_block) * row_elems;
    }

    // Packed PTX cvt: 2 paired conversions per 4 elements (half→e4m3x2).
    const half inv_scale_h = __float2half(inv_scale);
    const half2 inv_scale_h2 = make_half2(inv_scale_h, inv_scale_h);
    const int vec_elems = row_elems / 4;
    const half2* src2 = reinterpret_cast<const half2*>(src);
    uint32_t* dst4 = reinterpret_cast<uint32_t*>(dst);
    for (int i = threadIdx.x; i < vec_elems; i += blockDim.x) {
        half2 lo = __hmul2(src2[2 * i], inv_scale_h2);
        half2 hi = __hmul2(src2[2 * i + 1], inv_scale_h2);
        uint16_t e4m3_lo = cvt_f16x2_to_e4m3x2(*reinterpret_cast<uint32_t*>(&lo));
        uint16_t e4m3_hi = cvt_f16x2_to_e4m3x2(*reinterpret_cast<uint32_t*>(&hi));
        dst4[i] = static_cast<uint32_t>(e4m3_lo) | (static_cast<uint32_t>(e4m3_hi) << 16);
    }
    // Scalar tail for non-aligned remainder
    for (int i = vec_elems * 4 + threadIdx.x; i < row_elems; i += blockDim.x) {
        dst[i] = __nv_fp8_e4m3(__half2float(src[i]) * inv_scale);
    }
}

// Q-only RoPE for decode (n=1): applies RoPE to Q in-place.
// Grid: (1, n_heads), Block: rope_pairs.
__global__ __launch_bounds__(256) void rope_q_only_fp16_kernel(half* __restrict__ Q,  // [n_heads * head_dim]
                                                               const int* __restrict__ positions, int n_heads,
                                                               int head_dim, float theta, float inv_scaling,
                                                               int rope_pairs, bool neox,
                                                               const float* __restrict__ longrope_inv_freqs) {
    int head_idx = blockIdx.y;
    int pair_idx = threadIdx.x;
    if (head_idx >= n_heads || pair_idx >= rope_pairs)
        return;

    int pos = positions[0];  // decode: single token

    float freq;
    if (longrope_inv_freqs) {
        freq = longrope_inv_freqs[pair_idx];
    } else {
        freq = 1.0f / (powf(theta, (2.0f * pair_idx) / static_cast<float>(2 * rope_pairs)));
        freq *= inv_scaling;
    }
    float angle = static_cast<float>(pos) * freq;
    float cos_val = __cosf(angle);
    float sin_val = __sinf(angle);

    int64_t base = static_cast<int64_t>(head_idx) * head_dim;
    int idx0 = neox ? pair_idx : (2 * pair_idx);
    int idx1 = neox ? (pair_idx + rope_pairs) : (2 * pair_idx + 1);

    float q0 = __half2float(Q[base + idx0]);
    float q1 = __half2float(Q[base + idx1]);
    Q[base + idx0] = __float2half(q0 * cos_val - q1 * sin_val);
    Q[base + idx1] = __float2half(q0 * sin_val + q1 * cos_val);
}

// Add FP16 bias to each row of FP32 matrix: out[i,j] += bias[j]
// Grid: n_tokens, Block: 256, each thread handles multiple expert indices.
__global__ __launch_bounds__(256) void add_fp16_bias_to_fp32_kernel(float* __restrict__ data,
                                                                    const half* __restrict__ bias,
                                                                    int n_tokens, int n_cols) {
    int token = blockIdx.x;
    if (token >= n_tokens)
        return;
    float* row = data + static_cast<int64_t>(token) * n_cols;
    for (int j = threadIdx.x; j < n_cols; j += blockDim.x) {
        row[j] += __half2float(bias[j]);
    }
}

// Scale FP32 expert weights in-place: weights[i] *= scale
__global__ __launch_bounds__(256) void scale_fp32_kernel(float* __restrict__ data, float scale, int64_t n) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= scale;
    }
}

// Logit soft-capping: logit = softcap * tanh(logit / softcap)  (Gemma-2/3)
__global__ __launch_bounds__(256) void logit_softcap_fp32_kernel(float* __restrict__ data, float softcap,
                                                                 float inv_softcap, int64_t n) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = softcap * tanhf(data[idx] * inv_softcap);
    }
}

// FP32 -> FP16 conversion kernel (for scatter output back to compute_dtype)
__global__ __launch_bounds__(256) void fp32_to_fp16_kernel(const float* __restrict__ in,
                                                           half* __restrict__ out, int64_t n) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2half(in[idx]);
    }
}

// ---------------------------------------------------------------------------
// GDN attention output-gate split: replaces nh × 2 cudaMemcpy2DAsync loop
// with one launch. Source row layout per token is interleaved
// [Q_h0 | Gate_h0 | Q_h1 | Gate_h1 | ...] each chunk of size hd; both
// destinations are contiguous [n, nh*hd]. Grid: (n × nh) blocks of hd
// threads — each block copies one (token, head) pair's Q + gate vectors.
// ---------------------------------------------------------------------------
template <typename T>
__global__ __launch_bounds__(256) void attn_gate_split_interleaved_kernel(
    const T* __restrict__ src, T* __restrict__ q_dst, T* __restrict__ gate_dst, int n_tokens, int nh,
    int hd, int q_out_dim) {
    int t = blockIdx.x;
    int h = blockIdx.y;
    int tid = threadIdx.x;
    if (t >= n_tokens || h >= nh || tid >= hd)
        return;
    const T* src_row = src + static_cast<int64_t>(t) * q_out_dim;
    int64_t dst_off = static_cast<int64_t>(t) * (nh * hd) + static_cast<int64_t>(h) * hd + tid;
    int q_src = h * 2 * hd + tid;
    int g_src = h * 2 * hd + hd + tid;
    q_dst[dst_off] = src_row[q_src];
    gate_dst[dst_off] = src_row[g_src];
}

// Explicit template instantiations (FP16 + BF16 paths used by attention).
template __global__ void attn_gate_split_interleaved_kernel<half>(
    const half*, half*, half*, int, int, int, int);
template __global__ void attn_gate_split_interleaved_kernel<__nv_bfloat16>(
    const __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, int, int, int, int);

void attn_gate_split_interleaved(const void* src, void* q_dst, void* gate_dst, int n_tokens, int nh, int hd,
                                 int q_out_dim, int element_bytes, cudaStream_t stream) {
    if (n_tokens <= 0 || nh <= 0 || hd <= 0)
        return;
    int threads = (hd <= 256) ? hd : 256;
    dim3 grid(n_tokens, nh);
    if (element_bytes == 2) {
        // FP16 path (also used for BF16 since same byte width — caller
        // controls reinterpret).
        attn_gate_split_interleaved_kernel<half><<<grid, threads, 0, stream>>>(
            static_cast<const half*>(src), static_cast<half*>(q_dst), static_cast<half*>(gate_dst), n_tokens,
            nh, hd, q_out_dim);
    } else {
        // FP32 fallback — uses uint32_t reinterpret since templated half→FP32
        // dispatch requires another instantiation. For now log + fall through
        // to caller's loop on unsupported dtype.
        // (No FP32 path expected in attention compute; keep guard for safety.)
    }
}

// ---------------------------------------------------------------------------
// Host-side helpers
// ---------------------------------------------------------------------------

void elementwise_add(Tensor& a, const Tensor& b, cudaStream_t stream) {
    int64_t n = a.numel();
    if (a.qtype == QType::F16) {
        int64_t n2 = (n + 1) / 2;
        int threads = 256;
        int blocks = static_cast<int>((n2 + threads - 1) / threads);
        pdl::launch(elementwise_add_fp16_kernel, dim3(blocks), dim3(threads), 0, stream,
                    static_cast<half*>(a.data), static_cast<const half*>(b.data), n);
    } else {
        int threads = 256;
        int blocks = static_cast<int>((n + threads - 1) / threads);
        pdl::launch(elementwise_add_fp32_kernel, dim3(blocks), dim3(threads), 0, stream,
                    static_cast<float*>(a.data), static_cast<const float*>(b.data), n);
    }
}

// Element-wise add-store: out[i] = a[i] + b[i] — avoids in-place + copy pattern
void elementwise_add_store(const Tensor& a, const Tensor& b, Tensor& out, cudaStream_t stream) {
    int64_t n = a.numel();
    int64_t n2 = (n + 1) / 2;
    int threads = 256;
    int blocks = static_cast<int>((n2 + threads - 1) / threads);
    pdl::launch(elementwise_add_store_fp16_kernel, dim3(blocks), dim3(threads), 0, stream,
                static_cast<const half*>(a.data), static_cast<const half*>(b.data),
                static_cast<half*>(out.data), n);
}

// Add 1D bias to each row of a 2D output: out[row, col] += bias[col]
void add_bias(Tensor& out, const Tensor& bias, cudaStream_t stream) {
    if (bias.data == nullptr)
        return;
    int rows = static_cast<int>(out.shape[0]);
    int cols = static_cast<int>(bias.shape[0]);
    if (rows == 0 || cols == 0)
        return;
    int total = rows * cols;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    broadcast_add_bias_fp16_kernel<<<blocks, threads, 0, stream>>>(static_cast<half*>(out.data),
                                                                   static_cast<const half*>(bias.data), rows,
                                                                   cols);
}

// Fused 3-way bias add: applies up to 3 biases in a single kernel launch.
// blockIdx.y selects which output/bias pair (0, 1, or 2).
__global__ __launch_bounds__(256) void add_bias_3way_kernel(
    half* __restrict__ out0, const half* __restrict__ bias0, int cols0, half* __restrict__ out1,
    const half* __restrict__ bias1, int cols1, half* __restrict__ out2, const half* __restrict__ bias2,
    int cols2, int rows) {
    int which = blockIdx.y;
    half* out;
    const half* bias;
    int cols;
    if (which == 0) {
        out = out0;
        bias = bias0;
        cols = cols0;
    } else if (which == 1) {
        out = out1;
        bias = bias1;
        cols = cols1;
    } else {
        out = out2;
        bias = bias2;
        cols = cols2;
    }
    if (!out || !bias)
        return;

    int total = rows * cols;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += blockDim.x * gridDim.x) {
        int col = i % cols;
        out[i] = __hadd(out[i], bias[col]);
    }
}

void add_bias_3way(Tensor& out_a, const Tensor& bias_a, Tensor& out_b, const Tensor& bias_b, Tensor& out_c,
                   const Tensor& bias_c, cudaStream_t stream) {
    // Count how many actually have biases
    int n_active = (bias_a.data ? 1 : 0) + (bias_b.data ? 1 : 0) + (bias_c.data ? 1 : 0);
    if (n_active == 0)
        return;

    // Fall back to individual calls if only 1-2 biases
    if (n_active <= 2) {
        add_bias(out_a, bias_a, stream);
        add_bias(out_b, bias_b, stream);
        add_bias(out_c, bias_c, stream);
        return;
    }

    int rows = static_cast<int>(out_a.shape[0]);
    int cols_a = bias_a.data ? static_cast<int>(bias_a.shape[0]) : 0;
    int cols_b = bias_b.data ? static_cast<int>(bias_b.shape[0]) : 0;
    int cols_c = bias_c.data ? static_cast<int>(bias_c.shape[0]) : 0;

    int max_cols = std::max({cols_a, cols_b, cols_c});
    int total = rows * max_cols;
    int threads = 256;
    int blocks_x = (total + threads - 1) / threads;
    dim3 grid(blocks_x, 3);

    add_bias_3way_kernel<<<grid, threads, 0, stream>>>(
        bias_a.data ? static_cast<half*>(out_a.data) : nullptr,
        bias_a.data ? static_cast<const half*>(bias_a.data) : nullptr, cols_a,
        bias_b.data ? static_cast<half*>(out_b.data) : nullptr,
        bias_b.data ? static_cast<const half*>(bias_b.data) : nullptr, cols_b,
        bias_c.data ? static_cast<half*>(out_c.data) : nullptr,
        bias_c.data ? static_cast<const half*>(bias_c.data) : nullptr, cols_c, rows);
}

// Fused residual add + RMSNorm: hidden += residual; output = rmsnorm(hidden, weight).
// Saves 1 DRAM round-trip: reads hidden+residual+weight, writes hidden+output.
// Launch: <<<n_rows, 256>>>
__global__ __launch_bounds__(256) void residual_add_rmsnorm_kernel(
    half* __restrict__ hidden,          // [n, d_model] — modified in-place (residual added)
    const half* __restrict__ residual,  // [n, d_model]
    const half* __restrict__ weight,    // [d_model] RMSNorm weight
    half* __restrict__ output,          // [n, d_model] normalized output
    int d_model, float eps, float weight_offset) {
    __shared__ float warp_reduce[kWarpSize];
    const int tid = threadIdx.x;
    const int row = blockIdx.x;

    half* h_row = hidden + static_cast<int64_t>(row) * d_model;
    const half* r_row = residual + static_cast<int64_t>(row) * d_model;
    half* o_row = output + static_cast<int64_t>(row) * d_model;

    // Phase 1: Add residual to hidden + compute sum of squares
    float sum_sq = 0.0f;
    for (int d = tid; d < d_model; d += blockDim.x) {
        float h = __half2float(h_row[d]) + __half2float(r_row[d]);
        h_row[d] = __float2half(h);
        sum_sq += h * h;
    }

// Warp-level reduction
#pragma unroll
    for (int off = kWarpSize / 2; off > 0; off >>= 1)
        sum_sq += __shfl_xor_sync(0xFFFFFFFF, sum_sq, off);

    int warp_id = tid / kWarpSize;
    int lane = tid % kWarpSize;
    if (lane == 0)
        warp_reduce[warp_id] = sum_sq;
    __syncthreads();

    if (warp_id == 0) {
        int n_warps = blockDim.x / kWarpSize;
        float total = (lane < n_warps) ? warp_reduce[lane] : 0.0f;
#pragma unroll
        for (int off = kWarpSize / 2; off > 0; off >>= 1)
            total += __shfl_xor_sync(0xFFFFFFFF, total, off);
        if (lane == 0)
            warp_reduce[0] = rsqrtf(total / static_cast<float>(d_model) + eps);
    }
    __syncthreads();
    float inv_rms = warp_reduce[0];

    // Phase 2: Apply normalization
    for (int d = tid; d < d_model; d += blockDim.x) {
        float h = __half2float(h_row[d]);
        float w = __half2float(weight[d]) + weight_offset;
        o_row[d] = __float2half(h * inv_rms * w);
    }
}

void residual_add_rmsnorm(Tensor& hidden, const Tensor& residual, const Tensor& weight, Tensor& output,
                          float eps, cudaStream_t stream, float weight_offset) {
    int n = static_cast<int>(hidden.shape[0]);
    int d_model = static_cast<int>(hidden.shape[hidden.ndim - 1]);
    residual_add_rmsnorm_kernel<<<n, 256, 0, stream>>>(static_cast<half*>(hidden.data),
                                                       static_cast<const half*>(residual.data),
                                                       static_cast<const half*>(weight.data),
                                                       static_cast<half*>(output.data), d_model, eps,
                                                       weight_offset);
}

// Fused add-store + RMSNorm in-place: hidden = rmsnorm(a + b, weight).
// Saves 2 kernel launches + 1 memcpy vs separate add_store + rmsnorm + copy.
// Launch: <<<n_rows, 256>>>
__global__ __launch_bounds__(256) void add_rmsnorm_inplace_kernel(
    const half* __restrict__ a,       // [n, d_model]
    const half* __restrict__ b,       // [n, d_model]
    half* __restrict__ hidden,        // [n, d_model] — output (a + b, then normalized)
    const half* __restrict__ weight,  // [d_model] RMSNorm weight
    int d_model, float eps, float weight_offset) {
    __shared__ float warp_reduce[kWarpSize];
    const int tid = threadIdx.x;
    const int row = blockIdx.x;

    const half* a_row = a + static_cast<int64_t>(row) * d_model;
    const half* b_row = b + static_cast<int64_t>(row) * d_model;
    half* h_row = hidden + static_cast<int64_t>(row) * d_model;

    // Phase 1: Compute a+b and sum of squares in one pass
    float sum_sq = 0.0f;
    for (int d = tid; d < d_model; d += blockDim.x) {
        float h = __half2float(a_row[d]) + __half2float(b_row[d]);
        h_row[d] = __float2half(h);  // store sum for phase 2
        sum_sq += h * h;
    }

// Warp-level reduction
#pragma unroll
    for (int off = kWarpSize / 2; off > 0; off >>= 1)
        sum_sq += __shfl_xor_sync(0xFFFFFFFF, sum_sq, off);

    int warp_id = tid / kWarpSize;
    int lane = tid % kWarpSize;
    if (lane == 0)
        warp_reduce[warp_id] = sum_sq;
    __syncthreads();

    if (warp_id == 0) {
        int n_warps = blockDim.x / kWarpSize;
        float total = (lane < n_warps) ? warp_reduce[lane] : 0.0f;
#pragma unroll
        for (int off = kWarpSize / 2; off > 0; off >>= 1)
            total += __shfl_xor_sync(0xFFFFFFFF, total, off);
        if (lane == 0)
            warp_reduce[0] = rsqrtf(total / static_cast<float>(d_model) + eps);
    }
    __syncthreads();
    float inv_rms = warp_reduce[0];

    // Phase 2: Normalize in-place
    for (int d = tid; d < d_model; d += blockDim.x) {
        float h = __half2float(h_row[d]);
        float w = __half2float(weight[d]) + weight_offset;
        h_row[d] = __float2half(h * inv_rms * w);
    }
}

void add_rmsnorm_inplace(const Tensor& a, const Tensor& b, Tensor& hidden, const Tensor& weight, float eps,
                         cudaStream_t stream, float weight_offset) {
    int n = static_cast<int>(a.shape[0]);
    int d_model = static_cast<int>(a.shape[a.ndim - 1]);
    add_rmsnorm_inplace_kernel<<<n, 256, 0, stream>>>(static_cast<const half*>(a.data),
                                                      static_cast<const half*>(b.data),
                                                      static_cast<half*>(hidden.data),
                                                      static_cast<const half*>(weight.data), d_model, eps,
                                                      weight_offset);
}

// Fused RMSNorm + residual add: output = rmsnorm(input, weight) + residual.
// Launch: <<<n_rows, 256>>>
__global__ __launch_bounds__(256) void rmsnorm_add_residual_kernel(
    const half* __restrict__ input,     // [n, d_model]
    const half* __restrict__ weight,    // [d_model]
    const half* __restrict__ residual,  // [n, d_model]
    half* __restrict__ output,          // [n, d_model]
    int d_model, float eps, float weight_offset) {
    __shared__ float warp_reduce[kWarpSize];
    const int tid = threadIdx.x;
    const int row = blockIdx.x;

    const half* in_row = input + static_cast<int64_t>(row) * d_model;
    const half* r_row = residual + static_cast<int64_t>(row) * d_model;
    half* o_row = output + static_cast<int64_t>(row) * d_model;

    // Phase 1: Compute sum of squares of input
    float sum_sq = 0.0f;
    for (int d = tid; d < d_model; d += blockDim.x) {
        float v = __half2float(in_row[d]);
        sum_sq += v * v;
    }

#pragma unroll
    for (int off = kWarpSize / 2; off > 0; off >>= 1)
        sum_sq += __shfl_xor_sync(0xFFFFFFFF, sum_sq, off);

    int warp_id = tid / kWarpSize;
    int lane = tid % kWarpSize;
    if (lane == 0)
        warp_reduce[warp_id] = sum_sq;
    __syncthreads();

    if (warp_id == 0) {
        int n_warps = blockDim.x / kWarpSize;
        float total = (lane < n_warps) ? warp_reduce[lane] : 0.0f;
#pragma unroll
        for (int off = kWarpSize / 2; off > 0; off >>= 1)
            total += __shfl_xor_sync(0xFFFFFFFF, total, off);
        if (lane == 0)
            warp_reduce[0] = rsqrtf(total / static_cast<float>(d_model) + eps);
    }
    __syncthreads();
    float inv_rms = warp_reduce[0];

    // Phase 2: Normalize + add residual
    for (int d = tid; d < d_model; d += blockDim.x) {
        float v = __half2float(in_row[d]);
        float w = __half2float(weight[d]) + weight_offset;
        float r = __half2float(r_row[d]);
        o_row[d] = __float2half(v * inv_rms * w + r);
    }
}

void rmsnorm_add_residual(const Tensor& input, const Tensor& weight, const Tensor& residual, Tensor& output,
                          float eps, cudaStream_t stream, float weight_offset) {
    int n = static_cast<int>(input.shape[0]);
    int d_model = static_cast<int>(input.shape[input.ndim - 1]);
    rmsnorm_add_residual_kernel<<<n, 256, 0, stream>>>(static_cast<const half*>(input.data),
                                                       static_cast<const half*>(weight.data),
                                                       static_cast<const half*>(residual.data),
                                                       static_cast<half*>(output.data), d_model, eps,
                                                       weight_offset);
}

// Create a view of the first n_tokens rows from a [max_tokens, cols] buffer.
// Never modifies the source tensor.
Tensor slice_rows(const Tensor& buf, int n_tokens) {
    if (n_tokens == static_cast<int>(buf.shape[0]))
        return buf;
    // buf.slice(0, n) returns a view with shape[0] = n, same data pointer.
    return buf.slice(0, n_tokens);
}

// ---------------------------------------------------------------------------
// gemm_dispatch — single entry point that walks the GemmKernel registry in
// tier-priority order. R5 Slice 8.6 closes the cross-axis refactor: the
// legacy 21-parameter `gemm_dispatch_impl` switch (~250 LOC) is retired and
// the registry is now the unconditional path. Every tier registers its
// adapter from its own .cu file at static-init time (see
// gemm_kernel_*.cu); adding a new qtype/quantization tier is a one-file
// change.
//
// Tier order (MXFP4 GGUF → FP8 → NVFP4 GEMV → CUTLASS_NVFP4 → NVFP4 GEMM →
// FP16 cache/raw → small-M GGUF → generic-dequant catch-all) mirrors the
// historical legacy precedence and is the same as Slice 8 documented; it is
// observed-equivalent to the production behaviour from when Slice 8 flipped
// the registry default ON.
// ---------------------------------------------------------------------------
void gemm_dispatch(const Tensor& input, const Tensor& weight, Tensor& output, const GemmContext& ctx) {
    const auto* wc = ctx.wcache;
    const auto* qs = ctx.qscratch;
    if (!wc || !qs)
        return;

    const QType qtype = weight.qtype;

    // Residual-add fuse (beta != 0): only FP16 weight cache or dequantable-to-FP16
    // is supported. GEMV and block-scaled quant paths don't honor beta — callers
    // for those cases must continue to use their explicit fast paths.
    if (ctx.beta != 0.0f) {
        auto it = wc->fp16.find(weight.data);
        if (it != wc->fp16.end()) {
            gemm(input, it->second, output, 1.0f, ctx.beta, ctx.stream);
            return;
        }
        if (qs->dequant != nullptr && dequant_gpu_supported(qtype)) {
            int rows = static_cast<int>(weight.shape[0]);
            int cols = static_cast<int>(weight.shape[1]);
            dequant_gpu(weight.data, qs->dequant, qtype, rows, cols, ctx.stream);
            Tensor w_fp16(qs->dequant, QType::F16, weight.ndim, weight.shape, true);
            gemm(input, w_fp16, output, 1.0f, ctx.beta, ctx.stream);
            return;
        }
        if (weight.qtype == QType::F16 || weight.qtype == QType::BF16) {
            gemm(input, weight, output, 1.0f, ctx.beta, ctx.stream);
            return;
        }
        IMP_LOG_ERROR(
            "gemm_dispatch: beta=%.3f requested but no FP16 path available "
            "for qtype=%d",
            ctx.beta, (int)qtype);
        return;
    }

    const auto* fp16 = &wc->fp16;
    const auto* fp8 = (wc->use_fp8 && !ctx.force_fp16) ? &wc->fp8 : nullptr;
    const auto* nv4 = (wc->nvfp4.empty() || ctx.force_fp16) ? nullptr : &wc->nvfp4;
    const auto* ct4 = (wc->cutlass_nvfp4.empty() || ctx.force_fp16) ? nullptr : &wc->cutlass_nvfp4;
    const auto* mx4 = (wc->cutlass_mxfp4.empty() || ctx.force_fp16) ? nullptr : &wc->cutlass_mxfp4;

    const int M = static_cast<int>(input.shape[0]);

    // MXFP4 native GGUF (top priority). The kernel adapter gates internally
    // on `linear_scales != nullptr`; cache miss + dequant_scratch nullptr at
    // M>1 returns PreconditionFail so we fall through to the rest of the
    // table — same as the legacy `else if (dequant_scratch != nullptr)` skip.
    if (mx4 != nullptr && input.qtype == QType::F16) {
        auto mx_it = mx4->find(weight.data);
        if (mx_it != mx4->end() && mx_it->second.linear_scales != nullptr) {
            GemmKernelArgs args{};
            args.input = &input;
            args.output = &output;
            args.stream = ctx.stream;
            args.beta = ctx.beta;
            args.weight_payload = &mx_it->second;
            args.dequant_scratch = qs->dequant;  // only consumed by M>1 path
            GemmStrategy strat{StorageTier::MXFP4, QType::F16, /*m_is_one=*/(M == 1)};
            if (GemmKernelRegistry::instance().dispatch(strat, args) == GemmDispatchResult::Ok)
                return;
        }
    }
    // Q4_K_M direct-GEMM via INT8 IMMA (Phase 2C). Opt-in via
    // `gemm.q4k_imma_enabled` (default off); production-recommended for dense
    // Q4_K_M weights at M ≥ 1024 where the kernel's ~40 TOPS plateau beats
    // the dequant→cuBLAS fallback. Falls through to FP8 / generic dequant on
    // any precondition failure inside the handler. See
    // docs/superpowers/plans/2026-05-18-q4k-imma-phase2b-ceiling.md.
    if (qtype == QType::Q4_K && input.qtype == QType::F16 && M >= 1024 &&
        ctx.beta == 0.0f && RuntimeConfig::current().gemm.q4k_imma_enabled) {
        GemmKernelArgs args{};
        args.input = &input;
        args.output = &output;
        args.stream = ctx.stream;
        args.weight_payload = &weight;
        GemmStrategy strat{StorageTier::FP16, QType::Q4_K, /*m_is_one=*/false};
        if (GemmKernelRegistry::instance().dispatch(strat, args) == GemmDispatchResult::Ok)
            return;
    }
    // FP8 prefill (M>1): cache hit → FP8xFP8 cuBLASLt; cache miss with
    // dequant-supported qtype → dequant→FP16 cuBLAS (Slice 8.1). Raw FP16
    // weights drop to the FP16 strategy below.
    if (fp8 != nullptr && M > 1 && qs->fp8_act != nullptr && qs->d_act_scale != nullptr) {
        auto it = fp8->find(weight.data);
        if (it != fp8->end()) {
            GemmKernelArgs args{};
            args.input = &input;
            args.output = &output;
            args.stream = ctx.stream;
            args.beta = ctx.beta;
            args.weight_payload = &it->second;
            args.fp8_act_buf = qs->fp8_act;
            args.d_act_scale = qs->d_act_scale;
            args.d_fp8_block_maxes = qs->d_fp8_block_maxes;
            args.d_fp8_absmax = qs->d_fp8_absmax;
            args.fp8_max_grid = qs->fp8_max_grid;
            GemmStrategy strat{StorageTier::FP8, QType::F16, /*m_is_one=*/false};
            if (GemmKernelRegistry::instance().dispatch(strat, args) == GemmDispatchResult::Ok)
                return;
        } else {
            GemmKernelArgs args{};
            args.input = &input;
            args.output = &output;
            args.stream = ctx.stream;
            args.beta = ctx.beta;
            args.weight_payload = &weight;
            args.dequant_scratch = qs->dequant;
            GemmStrategy strat{StorageTier::FP8, QType::NONE, /*m_is_one=*/false};
            if (GemmKernelRegistry::instance().dispatch(strat, args) == GemmDispatchResult::Ok)
                return;
        }
    }
    // NVFP4 decode GEMV (M==1). Prequant Tensor sidecars (qtype=NVFP4 on the
    // weight) take precedence over the nv4 cache; the temp NvFP4QuantResult
    // is constructed here when sidecars are present.
    if (M == 1 && input.qtype == QType::F16) {
        NvFP4QuantResult nvfp4_tmp;
        const NvFP4QuantResult* nvfp4_view = nullptr;
        if (weight.qtype == QType::NVFP4 && weight.scales != nullptr) {
            nvfp4_tmp.packed_data = weight.data;
            nvfp4_tmp.micro_scales = weight.scales;
            nvfp4_tmp.tensor_scale = weight.tensor_scale;
            nvfp4_tmp.N = weight.shape[0];
            nvfp4_tmp.K = weight.shape[1] * 2;  // shape[1] is packed K/2
            nvfp4_view = &nvfp4_tmp;
        } else if (nv4 != nullptr) {
            auto it = nv4->find(weight.data);
            if (it != nv4->end())
                nvfp4_view = &it->second;
        }
        if (nvfp4_view != nullptr) {
            GemmKernelArgs args{};
            args.input = &input;
            args.output = &output;
            args.stream = ctx.stream;
            args.beta = ctx.beta;
            args.weight_payload = nvfp4_view;
            GemmStrategy strat{StorageTier::NVFP4, QType::F16, /*m_is_one=*/true};
            if (GemmKernelRegistry::instance().dispatch(strat, args) == GemmDispatchResult::Ok)
                return;
        }
    }
    // CUTLASS_NVFP4 prefill GEMM (M>1, native sm_120 block-scaled FP4 —
    // preferred path). Slice 8.6 — QW7 dual-cache MXFP4 hand-off. When
    // `--mxfp4-prefill` is on, executor_pre_dequant.cu populates
    // `cutlass_mxfp4` by iterating every NVFP4 entry: cache membership is a
    // superset of cutlass_nvfp4. We forward the MXFP4 payload + scratch
    // through `args.mxfp4_payload` so the handler can try the MXFP4 CUTLASS
    // GEMM first and fall back to NVFP4 CUTLASS on failure (mirrors the
    // retired legacy QW7 probe).
    if (M > 1 && input.qtype == QType::F16 && ct4 != nullptr &&
        qs->cutlass_act_data != nullptr) {
        auto ct_it = ct4->find(weight.data);
        if (ct_it != ct4->end()) {
            GemmKernelArgs args{};
            args.input = &input;
            args.output = &output;
            args.stream = ctx.stream;
            args.beta = ctx.beta;
            args.weight_payload = &ct_it->second;
            args.cutlass_act_data = qs->cutlass_act_data;
            args.cutlass_act_sf = qs->cutlass_act_sf;
            args.cutlass_workspace = qs->cutlass_workspace;
            args.cutlass_workspace_size = qs->cutlass_workspace_size;
            if (mx4 != nullptr && qs->mxfp4_act_sf != nullptr) {
                auto mx_it = mx4->find(weight.data);
                if (mx_it != mx4->end()) {
                    args.mxfp4_payload = &mx_it->second;
                    args.mxfp4_act_sf = qs->mxfp4_act_sf;
                    args.mxfp4_workspace = qs->mxfp4_workspace;
                    args.mxfp4_workspace_size = qs->mxfp4_workspace_size;
                }
            }
            GemmStrategy strat{StorageTier::CUTLASS_NVFP4, QType::F16, /*m_is_one=*/false};
            if (GemmKernelRegistry::instance().dispatch(strat, args) == GemmDispatchResult::Ok)
                return;
        }
    }
    // NVFP4 prefill GEMM (M>1, dequant→cuBLAS fallback) — fires when the
    // CUTLASS path is unavailable or returned PreconditionFail.
    if (M > 1 && input.qtype == QType::F16) {
        NvFP4QuantResult nvfp4_tmp;
        const NvFP4QuantResult* nvfp4_view = nullptr;
        if (weight.qtype == QType::NVFP4 && weight.scales != nullptr) {
            nvfp4_tmp.packed_data = weight.data;
            nvfp4_tmp.micro_scales = weight.scales;
            nvfp4_tmp.tensor_scale = weight.tensor_scale;
            nvfp4_tmp.N = weight.shape[0];
            nvfp4_tmp.K = weight.shape[1] * 2;  // shape[1] is packed K/2
            nvfp4_view = &nvfp4_tmp;
        } else if (nv4 != nullptr) {
            auto it = nv4->find(weight.data);
            if (it != nv4->end())
                nvfp4_view = &it->second;
        }
        if (nvfp4_view != nullptr) {
            GemmKernelArgs args{};
            args.input = &input;
            args.output = &output;
            args.stream = ctx.stream;
            args.beta = ctx.beta;
            args.weight_payload = nvfp4_view;
            GemmStrategy strat{StorageTier::NVFP4, QType::F16, /*m_is_one=*/false};
            if (GemmKernelRegistry::instance().dispatch(strat, args) == GemmDispatchResult::Ok)
                return;
        }
    }
    // FP16 cache hit OR raw FP16/BF16-source weight (BF16 source is always
    // converted to F16 at upload; weight.qtype is F16 here).
    if (auto it = wc->fp16.find(weight.data); it != wc->fp16.end()) {
        GemmKernelArgs args{};
        args.input = &input;
        args.output = &output;
        args.stream = ctx.stream;
        args.weight_payload = &it->second;
        GemmStrategy strat{StorageTier::FP16, QType::F16, M == 1};
        if (GemmKernelRegistry::instance().dispatch(strat, args) == GemmDispatchResult::Ok)
            return;
    }
    if ((qtype == QType::F16 || qtype == QType::BF16) && weight.qtype == QType::F16) {
        GemmKernelArgs args{};
        args.input = &input;
        args.output = &output;
        args.stream = ctx.stream;
        args.weight_payload = &weight;
        GemmStrategy strat{StorageTier::FP16, QType::F16, M == 1};
        if (GemmKernelRegistry::instance().dispatch(strat, args) == GemmDispatchResult::Ok)
            return;
    }
    // GGUF small-M (mmvq / dp4a / fused-gemv): M==1, FP16 input, non-FP32
    // output. `prefer_fp16_cache` keeps the cache-vs-raw decision at the
    // dispatch site because it depends on cache membership + stride — not
    // expressible via GemmKernelArgs alone.
    const bool fp32_output = (output.qtype == QType::F32);
    const bool prefer_fp16_cache =
        (fp16 != nullptr && fp16->count(weight.data) > 0 && input.stride[0] != weight.shape[1]);
    if (M == 1 && input.qtype == QType::F16 && !fp32_output && !prefer_fp16_cache) {
        GemmKernelArgs args{};
        args.input = &input;
        args.output = &output;
        args.stream = ctx.stream;
        args.beta = ctx.beta;
        args.weight_payload = &weight;
        args.q8_1_buf = qs->q8_1_buf;
        args.d8_buf = qs->d8_buf;
        args.dequant_scratch = qs->dequant;  // fused-gemv readiness sentinel
        GemmStrategy strat{StorageTier::FP16, qtype, /*m_is_one=*/true};
        if (GemmKernelRegistry::instance().dispatch(strat, args) == GemmDispatchResult::Ok)
            return;
    }
    // Qtype-agnostic generic-dequant catch-all (M>1 prefill). Handler reads
    // weight.qtype off the Tensor and switches via `dequant_gpu_supported`.
    if (M > 1 && input.qtype == QType::F16) {
        GemmKernelArgs args{};
        args.input = &input;
        args.output = &output;
        args.stream = ctx.stream;
        args.beta = ctx.beta;
        args.weight_payload = &weight;
        args.dequant_scratch = qs->dequant;
        GemmStrategy strat{StorageTier::FP16, QType::NONE, /*m_is_one=*/false};
        if (GemmKernelRegistry::instance().dispatch(strat, args) == GemmDispatchResult::Ok)
            return;
    }
    // Final fallback: raw FP16/BF16 cuBLAS. Reached only when none of the
    // tiers above match — typically a raw FP16/BF16 weight at a shape no
    // cache covered. Matches the legacy `else { gemm(...); }` arm.
    gemm(input, weight, output, 1.0f, 0.0f, ctx.stream);
}

}  // namespace imp
