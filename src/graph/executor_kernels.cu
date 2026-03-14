#include "graph/executor_kernels.h"
#include "graph/executor.h"
#include "compute/gemm.h"
#include "compute/gemm_q6k.h"
#ifdef IMP_USE_CUTLASS
#include "compute/gemm_cutlass.h"
#include "compute/gemm_cutlass_sm120.h"
#include "compute/gemm_cutlass_mxfp4_sm120.h"
#endif
#include "compute/hadamard.h"
#include "quant/quant_gemm.h"
#include "quant/dequant_gpu.h"
#include "quant/fp8_quant.h"
#include "quant/nvfp4_gemm.h"
#include "compute/gemm_cublaslt_nvfp4.h"
#include "runtime/pdl.h"

namespace imp {

// ---------------------------------------------------------------------------
// Small CUDA kernels used by the executor
// ---------------------------------------------------------------------------

// Broadcast bias addition: out[row, col] += bias[col] for rows x cols elements
__global__ void broadcast_add_bias_fp16_kernel(half* out, const half* bias,
                                                int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    for (int i = idx; i < total; i += blockDim.x * gridDim.x) {
        int col = i % cols;
        out[i] = __hadd(out[i], bias[col]);
    }
}


// Element-wise scale: out[i] *= scale, for FP16 data (Gemma embedding scaling)
__global__ void scale_fp16_kernel(half* data, half scale, int64_t n) {
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
__global__ void elementwise_add_fp16_kernel(half* a, const half* b, int64_t n) {
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
__global__ void elementwise_add_store_fp16_kernel(const half* a, const half* b,
                                                   half* out, int64_t n) {
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
__global__ void fp32_accum_add_fp16_kernel(float* accum, const half* branch, int64_t n) {
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
__global__ void fp32_to_fp16_rowscale_kernel(const float* __restrict__ in,
                                             half* __restrict__ out,
                                             int rows, int cols) {
    extern __shared__ float smem[];
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* row_in = in + static_cast<int64_t>(row) * cols;
    half* row_out       = out + static_cast<int64_t>(row) * cols;

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
__global__ void rmsnorm_fp32_accum_to_fp16_kernel(
        const half* __restrict__ input,     // [n, d_model] pre-norm data (e.g. GEMV output)
        const half* __restrict__ norm_w,    // [d_model] RMSNorm weights
        float* __restrict__ fp32_accum,     // [n, d_model] FP32 accumulator (read-modify-write)
        half* __restrict__ output,          // [n, d_model] FP16 output for next layer
        int d_model,
        float eps,
        float weight_offset) {
    __shared__ float warp_reduce[32];  // support up to 1024 threads (32 warps)
    __shared__ float s_inv_rms;
    __shared__ float s_row_max;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp_id = tid >> 5;
    const int n_warps = blockDim.x >> 5;
    const int row = blockIdx.x;

    // Vectorized: process 8 halfs (1 float4 = 2 half2) per iteration.
    const int d_model_v = d_model / 8;  // number of float4-sized chunks

    const float4* x_row4 = reinterpret_cast<const float4*>(
        input + static_cast<int64_t>(row) * d_model);
    const float4* nw_row4 = reinterpret_cast<const float4*>(norm_w);
    float4* accum_row4 = reinterpret_cast<float4*>(
        fp32_accum + static_cast<int64_t>(row) * d_model);
    float4* out_row4 = reinterpret_cast<float4*>(
        output + static_cast<int64_t>(row) * d_model);

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
        sum_sq += f0.x*f0.x + f0.y*f0.y + f1.x*f1.x + f1.y*f1.y
                + f2.x*f2.x + f2.y*f2.y + f3.x*f3.x + f3.y*f3.y;
    }

    // Block reduce sum_sq
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        sum_sq += __shfl_xor_sync(0xFFFFFFFF, sum_sq, off);
    if (lane == 0) warp_reduce[warp_id] = sum_sq;
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

        local_max = fmaxf(local_max, fmaxf(
            fmaxf(fabsf(acc_f[0]), fabsf(acc_f[1])),
            fmaxf(fabsf(acc_f[2]), fabsf(acc_f[3]))));
        local_max = fmaxf(local_max, fmaxf(
            fmaxf(fabsf(acc_f_hi[0]), fabsf(acc_f_hi[1])),
            fmaxf(fabsf(acc_f_hi[2]), fabsf(acc_f_hi[3]))));
    }

    // Block reduce max_abs
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        local_max = fmaxf(local_max, __shfl_xor_sync(0xFFFFFFFF, local_max, off));
    if (lane == 0) warp_reduce[warp_id] = local_max;
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

// Convert FP16 → FP32: out[i] = __half2float(in[i])
__global__ void fp16_to_fp32_kernel(const half* in, float* out, int64_t n) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __half2float(in[idx]);
    }
}

// Element-wise addition: a[i] += b[i], for FP32 data
__global__ void elementwise_add_fp32_kernel(float* a, const float* b, int64_t n) {
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
__global__ void write_kv_cache_kernel(
    const half* data_in,
    const int* positions,
    const int* block_tables,
    half* cache_base,
    int block_stride,
    int row_elems,
    int block_size,
    int n_tokens,
    int max_blocks_per_seq,
    int n_sequences
) {
    int token_idx = blockIdx.x;
    if (token_idx >= n_tokens) return;

    int pos = positions[token_idx];
    int block_idx = pos / block_size;
    int slot_in_block = pos % block_size;

    int block_id;
    if (max_blocks_per_seq > 0 && n_sequences > 1) {
        // Batched: for decode, token i = sequence i
        int seq_idx = token_idx;  // 1 token per sequence in decode
        block_id = block_tables[seq_idx * max_blocks_per_seq + block_idx];
    } else {
        // Single-sequence or legacy path
        block_id = block_tables[block_idx];
    }

    half* dst = cache_base + static_cast<int64_t>(block_id) * block_stride
                           + static_cast<int64_t>(slot_in_block) * row_elems;
    const half* src = data_in + static_cast<int64_t>(token_idx) * row_elems;

    for (int i = threadIdx.x; i < row_elems; i += blockDim.x) {
        dst[i] = src[i];
    }
}

// Fused K+V write to paged KV cache in a single launch.
// blockIdx.x = token index, blockIdx.y = 0 (K) or 1 (V).
// Saves one kernel launch per attention layer.
__global__ void write_kv_cache_fused_kernel(
    const half* k_in,        // [n_tokens, n_kv_heads * head_dim]
    const half* v_in,        // [n_tokens, n_kv_heads * head_dim]
    const int* positions,
    const int* block_tables,
    half* k_cache_base,
    half* v_cache_base,
    int block_stride,
    int row_elems,
    int block_size,
    int n_tokens,
    int max_blocks_per_seq,
    int n_sequences
) {
    int token_idx = blockIdx.x;
    if (token_idx >= n_tokens) return;

    int pos = positions[token_idx];
    int block_idx = pos / block_size;
    int slot_in_block = pos % block_size;

    int block_id;
    if (max_blocks_per_seq > 0 && n_sequences > 1) {
        int seq_idx = token_idx;
        block_id = block_tables[seq_idx * max_blocks_per_seq + block_idx];
    } else {
        block_id = block_tables[block_idx];
    }

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

    half* dst = dst_base + static_cast<int64_t>(block_id) * block_stride
                         + static_cast<int64_t>(slot_in_block) * row_elems;

    for (int i = threadIdx.x; i < row_elems; i += blockDim.x) {
        dst[i] = src[i];
    }
}

// FP16 -> FP8 E4M3 quantization + write to paged KV cache
#ifdef __CUDA_FP8_TYPES_EXIST__
__global__ void write_kv_cache_fp8_kernel(
    const half* data_in,
    const int* positions,
    const int* block_tables,
    __nv_fp8_e4m3* cache_base,  // FP8 cache
    float inv_scale,            // 1.0 / kv_scale
    int block_stride,
    int row_elems,
    int block_size,
    int n_tokens,
    int max_blocks_per_seq,
    int n_sequences
) {
    int token_idx = blockIdx.x;
    if (token_idx >= n_tokens) return;

    int pos = positions[token_idx];
    int block_idx = pos / block_size;
    int slot_in_block = pos % block_size;

    int block_id;
    if (max_blocks_per_seq > 0 && n_sequences > 1) {
        int seq_idx = token_idx;
        block_id = block_tables[seq_idx * max_blocks_per_seq + block_idx];
    } else {
        block_id = block_tables[block_idx];
    }

    __nv_fp8_e4m3* dst = cache_base + static_cast<int64_t>(block_id) * block_stride
                                     + static_cast<int64_t>(slot_in_block) * row_elems;
    const half* src = data_in + static_cast<int64_t>(token_idx) * row_elems;

    for (int i = threadIdx.x; i < row_elems; i += blockDim.x) {
        float val = __half2float(src[i]) * inv_scale;
        dst[i] = __nv_fp8_e4m3(val);
    }
}
#else
// Software fallback: clamp FP16 to FP8 E4M3 range and pack to uint8_t
__global__ void write_kv_cache_fp8_kernel(
    const half* data_in,
    const int* positions,
    const int* block_tables,
    uint8_t* cache_base,        // FP8 cache (as raw bytes)
    float inv_scale,            // 1.0 / kv_scale
    int block_stride,
    int row_elems,
    int block_size,
    int n_tokens,
    int max_blocks_per_seq,
    int n_sequences
) {
    int token_idx = blockIdx.x;
    if (token_idx >= n_tokens) return;

    int pos = positions[token_idx];
    int block_idx = pos / block_size;
    int slot_in_block = pos % block_size;

    int block_id;
    if (max_blocks_per_seq > 0 && n_sequences > 1) {
        int seq_idx = token_idx;
        block_id = block_tables[seq_idx * max_blocks_per_seq + block_idx];
    } else {
        block_id = block_tables[block_idx];
    }

    uint8_t* dst = cache_base + static_cast<int64_t>(block_id) * block_stride
                               + static_cast<int64_t>(slot_in_block) * row_elems;
    const half* src = data_in + static_cast<int64_t>(token_idx) * row_elems;

    // FP8 E4M3 range: [-448, 448]
    const float fp8_max = 448.0f;
    for (int i = threadIdx.x; i < row_elems; i += blockDim.x) {
        float val = __half2float(src[i]) * inv_scale;
        val = fminf(fmaxf(val, -fp8_max), fp8_max);
        // Simple rounding: convert float to FP8 E4M3 bit pattern
        // Sign(1) | Exponent(4) | Mantissa(3)
        uint32_t bits = __float_as_uint(val);
        uint8_t sign = (bits >> 24) & 0x80;
        int exponent = ((bits >> 23) & 0xFF) - 127 + 7; // rebias to E4M3
        uint8_t mantissa = (bits >> 20) & 0x07;
        if (exponent <= 0) {
            exponent = 0;
            mantissa = 0;
        } else if (exponent >= 15) {
            exponent = 15;
            mantissa = 0x06; // max finite for E4M3 (no inf/nan encoding)
        }
        dst[i] = sign | (static_cast<uint8_t>(exponent) << 3) | mantissa;
    }
}
#endif

// ---------------------------------------------------------------------------
// FP16 -> INT8 quantization + write to paged KV cache with per-head scales.
// Each warp processes one KV head independently: compute absmax via warp shuffle,
// then quantize and write int8 data + half scale.
//
// blockIdx.x = token_idx, blockIdx.y = 0 (K) or 1 (V).
// blockDim.x = 256 (8 warps). Each warp loops over heads.
// ---------------------------------------------------------------------------
__global__ void write_kv_cache_int8_kernel(
    const half* __restrict__ k_in,        // [n_tokens, n_kv_heads * head_dim]
    const half* __restrict__ v_in,
    const int* __restrict__ positions,
    const int* __restrict__ block_tables,
    int8_t* __restrict__ k_cache_base,
    int8_t* __restrict__ v_cache_base,
    half* __restrict__ k_scale_base,      // [total_blocks, kKVBlockSize, n_kv_heads]
    half* __restrict__ v_scale_base,
    int block_stride,                     // kKVBlockSize * n_kv_heads * head_dim (int8 elems)
    int scale_block_stride,               // kKVBlockSize * n_kv_heads (half elems)
    int n_kv_heads,
    int head_dim,
    int block_size,
    int n_tokens,
    int max_blocks_per_seq,
    int n_sequences)
{
    const int token_idx = blockIdx.x;
    if (token_idx >= n_tokens) return;

    const int pos = positions[token_idx];
    const int block_idx = pos / block_size;
    const int slot_in_block = pos % block_size;

    int block_id;
    if (max_blocks_per_seq > 0 && n_sequences > 1) {
        int seq_idx = token_idx;
        block_id = block_tables[seq_idx * max_blocks_per_seq + block_idx];
    } else {
        block_id = block_tables[block_idx];
    }

    // Select K or V based on blockIdx.y
    const half* src_base = (blockIdx.y == 0) ? k_in : v_in;
    int8_t* cache_base = (blockIdx.y == 0) ? k_cache_base : v_cache_base;
    half* scale_base = (blockIdx.y == 0) ? k_scale_base : v_scale_base;

    const int row_elems = n_kv_heads * head_dim;
    const half* src = src_base + static_cast<int64_t>(token_idx) * row_elems;
    int8_t* dst = cache_base + static_cast<int64_t>(block_id) * block_stride
                             + static_cast<int64_t>(slot_in_block) * row_elems;
    half* scale_dst = scale_base + static_cast<int64_t>(block_id) * scale_block_stride
                                 + static_cast<int64_t>(slot_in_block) * n_kv_heads;

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int num_warps = blockDim.x / 32;

    // Each warp processes one head at a time, looping over heads
    for (int h = warp_id; h < n_kv_heads; h += num_warps) {
        const int head_offset = h * head_dim;

        // Step 1: Load FP16 values and compute per-head absmax
        float amax = 0.0f;
        // Each lane handles head_dim/32 elements (head_dim is typically 64/96/128)
        for (int d = lane_id; d < head_dim; d += 32) {
            float val = __half2float(src[head_offset + d]);
            amax = fmaxf(amax, fabsf(val));
        }
        // Warp-level absmax reduction
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset));

        // Step 2: Compute scale
        float sc = amax / 127.0f;
        float inv_sc = (amax > 1e-8f) ? (127.0f / amax) : 0.0f;

        // Step 3: Quantize and write int8 data
        for (int d = lane_id; d < head_dim; d += 32) {
            float val = __half2float(src[head_offset + d]);
            int8_t q = static_cast<int8_t>(__float2int_rn(val * inv_sc));
            dst[head_offset + d] = q;
        }

        // Step 4: Write scale (one half per head per token)
        if (lane_id == 0) {
            scale_dst[h] = __float2half(sc);
        }
    }
}

// Fused KV cache write with RoPE on K: applies RoPE to K during write, copies V directly.
// blockIdx.x = token index, blockIdx.y = 0 (K+RoPE) or 1 (V copy).
// Eliminates the separate RoPE kernel launch for K in the decode path.
__global__ void write_kv_cache_rope_fused_kernel(
    const half* __restrict__ k_in,       // [n_tokens, n_kv_heads * head_dim] raw K (no RoPE)
    const half* __restrict__ v_in,       // [n_tokens, n_kv_heads * head_dim]
    const int* __restrict__ positions,
    const int* __restrict__ block_tables,
    half* k_cache_base,
    half* v_cache_base,
    int block_stride,
    int row_elems,
    int block_size,
    int n_tokens,
    int max_blocks_per_seq,
    int n_sequences,
    int n_kv_heads,
    int head_dim,
    float theta,
    float inv_scaling,
    int rope_pairs,      // effective_rope_dim / 2
    bool neox,
    const float* __restrict__ longrope_inv_freqs
) {
    int token_idx = blockIdx.x;
    if (token_idx >= n_tokens) return;

    int pos = positions[token_idx];
    int block_idx_kv = pos / block_size;
    int slot_in_block = pos % block_size;

    int block_id;
    if (max_blocks_per_seq > 0 && n_sequences > 1) {
        int seq_idx = token_idx;
        block_id = block_tables[seq_idx * max_blocks_per_seq + block_idx_kv];
    } else {
        block_id = block_tables[block_idx_kv];
    }

    if (blockIdx.y == 0) {
        // K path: apply RoPE during write
        const half* k_src = k_in + static_cast<int64_t>(token_idx) * row_elems;
        half* k_dst = k_cache_base + static_cast<int64_t>(block_id) * block_stride
                                   + static_cast<int64_t>(slot_in_block) * row_elems;

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
                freq = longrope_inv_freqs[pair_idx];
            } else {
                freq = 1.0f / (powf(theta, (2.0f * pair_idx) / static_cast<float>(head_dim)));
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
        // V path: direct copy (no RoPE)
        const half* v_src = v_in + static_cast<int64_t>(token_idx) * row_elems;
        half* v_dst = v_cache_base + static_cast<int64_t>(block_id) * block_stride
                                   + static_cast<int64_t>(slot_in_block) * row_elems;
        for (int i = threadIdx.x; i < row_elems; i += blockDim.x) {
            v_dst[i] = v_src[i];
        }
    }
}

// Fused K+V FP8 write: combines K and V quantize+write into one kernel launch.
// blockIdx.x = token index, blockIdx.y = 0 (K) or 1 (V).
__global__ void write_kv_cache_fp8_fused_kernel(
    const half* __restrict__ k_in,
    const half* __restrict__ v_in,
    const int* __restrict__ positions,
    const int* __restrict__ block_tables,
    __nv_fp8_e4m3* k_cache_base,
    __nv_fp8_e4m3* v_cache_base,
    float inv_scale,
    int block_stride,
    int row_elems,
    int block_size,
    int n_tokens,
    int max_blocks_per_seq,
    int n_sequences
) {
    int token_idx = blockIdx.x;
    if (token_idx >= n_tokens) return;

    int pos = positions[token_idx];
    int block_idx = pos / block_size;
    int slot_in_block = pos % block_size;

    int block_id;
    if (max_blocks_per_seq > 0 && n_sequences > 1) {
        block_id = block_tables[token_idx * max_blocks_per_seq + block_idx];
    } else {
        block_id = block_tables[block_idx];
    }

    const half* src;
    __nv_fp8_e4m3* dst;
    if (blockIdx.y == 0) {
        src = k_in + static_cast<int64_t>(token_idx) * row_elems;
        dst = k_cache_base + static_cast<int64_t>(block_id) * block_stride
                           + static_cast<int64_t>(slot_in_block) * row_elems;
    } else {
        src = v_in + static_cast<int64_t>(token_idx) * row_elems;
        dst = v_cache_base + static_cast<int64_t>(block_id) * block_stride
                           + static_cast<int64_t>(slot_in_block) * row_elems;
    }

    for (int i = threadIdx.x; i < row_elems; i += blockDim.x) {
        dst[i] = __nv_fp8_e4m3(__half2float(src[i]) * inv_scale);
    }
}

// Q-only RoPE for decode (n=1): applies RoPE to Q in-place.
// Grid: (1, n_heads), Block: rope_pairs.
__global__ void rope_q_only_fp16_kernel(
    half* __restrict__ Q,       // [n_heads * head_dim]
    const int* __restrict__ positions,
    int n_heads,
    int head_dim,
    float theta,
    float inv_scaling,
    int rope_pairs,
    bool neox,
    const float* __restrict__ longrope_inv_freqs
) {
    int head_idx  = blockIdx.y;
    int pair_idx  = threadIdx.x;
    if (head_idx >= n_heads || pair_idx >= rope_pairs) return;

    int pos = positions[0];  // decode: single token

    float freq;
    if (longrope_inv_freqs) {
        freq = longrope_inv_freqs[pair_idx];
    } else {
        freq = 1.0f / (powf(theta, (2.0f * pair_idx) / static_cast<float>(head_dim)));
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
__global__ void add_fp16_bias_to_fp32_kernel(float* __restrict__ data,
                                              const half* __restrict__ bias,
                                              int n_tokens, int n_cols) {
    int token = blockIdx.x;
    if (token >= n_tokens) return;
    float* row = data + static_cast<int64_t>(token) * n_cols;
    for (int j = threadIdx.x; j < n_cols; j += blockDim.x) {
        row[j] += __half2float(bias[j]);
    }
}

// Scale FP32 expert weights in-place: weights[i] *= scale
__global__ void scale_fp32_kernel(float* __restrict__ data, float scale, int64_t n) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= scale;
    }
}

// Logit soft-capping: logit = softcap * tanh(logit / softcap)  (Gemma-2/3)
__global__ void logit_softcap_fp32_kernel(float* __restrict__ data,
                                          float softcap, float inv_softcap,
                                          int64_t n) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = softcap * tanhf(data[idx] * inv_softcap);
    }
}

// FP32 -> FP16 conversion kernel (for scatter output back to compute_dtype)
__global__ void fp32_to_fp16_kernel(const float* __restrict__ in,
                                    half* __restrict__ out,
                                    int64_t n) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2half(in[idx]);
    }
}

// ---------------------------------------------------------------------------
// Host-side helpers
// ---------------------------------------------------------------------------

void elementwise_add(Tensor& a, const Tensor& b, cudaStream_t stream) {
    int64_t n = a.numel();
    if (a.dtype == DType::FP16) {
        int64_t n2 = (n + 1) / 2;
        int threads = 256;
        int blocks = static_cast<int>((n2 + threads - 1) / threads);
        pdl::launch(elementwise_add_fp16_kernel,
                    dim3(blocks), dim3(threads), 0, stream,
                    static_cast<half*>(a.data),
                    static_cast<const half*>(b.data),
                    n);
    } else {
        int threads = 256;
        int blocks = static_cast<int>((n + threads - 1) / threads);
        pdl::launch(elementwise_add_fp32_kernel,
                    dim3(blocks), dim3(threads), 0, stream,
                    static_cast<float*>(a.data),
                    static_cast<const float*>(b.data),
                    n);
    }
}

// Element-wise add-store: out[i] = a[i] + b[i] — avoids in-place + copy pattern
void elementwise_add_store(const Tensor& a, const Tensor& b, Tensor& out,
                                   cudaStream_t stream) {
    int64_t n = a.numel();
    int64_t n2 = (n + 1) / 2;
    int threads = 256;
    int blocks = static_cast<int>((n2 + threads - 1) / threads);
    pdl::launch(elementwise_add_store_fp16_kernel,
                dim3(blocks), dim3(threads), 0, stream,
                static_cast<const half*>(a.data),
                static_cast<const half*>(b.data),
                static_cast<half*>(out.data),
                n);
}

// Add 1D bias to each row of a 2D output: out[row, col] += bias[col]
void add_bias(Tensor& out, const Tensor& bias, cudaStream_t stream) {
    if (bias.data == nullptr) return;
    int rows = static_cast<int>(out.shape[0]);
    int cols = static_cast<int>(bias.shape[0]);
    if (rows == 0 || cols == 0) return;
    int total = rows * cols;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    broadcast_add_bias_fp16_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<half*>(out.data),
        static_cast<const half*>(bias.data),
        rows, cols);
}


// Create a view of the first n_tokens rows from a [max_tokens, cols] buffer.
// Never modifies the source tensor.
Tensor slice_rows(const Tensor& buf, int n_tokens) {
    if (n_tokens == static_cast<int>(buf.shape[0])) return buf;
    // buf.slice(0, n) returns a view with shape[0] = n, same data pointer.
    return buf.slice(0, n_tokens);
}

// Dispatch GEMM based on weight quantization type.
// For Q4_0/Q4_1: uses fused quant_gemm_int4 with packed nibbles + scales.
// For Q8_0/Q6_K (with dequant_scratch): dequant into scratch, then cuBLAS gemm.
// For NONE/F16/BF16: uses standard cuBLAS gemm.
//
// When q8_1_buf/d8_buf are non-null and input is a single vector (M=1), the
// dp4a MMVQ path is used: input is pre-quantized to Q8_1 and dot products use
// native INT8 SIMD (dp4a). This is ~2x faster than FP16 dequant for Q6_K/Q8_0.
void gemm_dispatch(const Tensor& input, const Tensor& weight,
                           const Tensor& scales, GGMLQuantType qtype,
                           Tensor& output, void* dequant_scratch,
                           cudaStream_t stream,
                           block_q8_1* q8_1_buf,
                           float* d8_buf,
                           const std::unordered_map<const void*, Tensor>* fp16_cache,
                           const std::unordered_map<const void*, GraphExecutor::FP8CacheEntry>* fp8_cache,
                           void* fp8_act_buf,
                           float* d_act_scale,
                           float* d_fp8_block_maxes,
                           float* d_fp8_absmax,
                           int fp8_max_grid,
                           const std::unordered_map<const void*, NvFP4QuantResult>* nvfp4_cache,
                           const std::unordered_map<const void*, CutlassNvFP4Weight>* cutlass_nvfp4_cache,
                           void* cutlass_act_data,
                           void* cutlass_act_sf,
                           void* cutlass_workspace,
                           size_t cutlass_workspace_size,
                           const std::unordered_map<const void*, CutlassMxFP4Weight>* mxfp4_cache,
                           void* mxfp4_act_sf,
                           void* mxfp4_workspace,
                           size_t mxfp4_workspace_size) {
    // NVFP4 cache path: GEMV for M=1 (decode), CUTLASS/dequant for M>1 (prefill).
    if (nvfp4_cache != nullptr && input.dtype == DType::FP16) {
        auto it = nvfp4_cache->find(weight.data);
        if (it != nvfp4_cache->end()) {
            if (input.shape[0] == 1) {
                gemv_nvfp4_kpar(it->second,
                                reinterpret_cast<const half*>(input.data),
                                reinterpret_cast<half*>(output.data),
                                static_cast<int>(it->second.N),
                                static_cast<int>(it->second.K), stream);
            } else if (cutlass_nvfp4_cache != nullptr && cutlass_act_data != nullptr) {
                // Native FP4 GEMM: try cuBLASLt first, then CUTLASS sm_120
                auto ct_it = cutlass_nvfp4_cache->find(weight.data);
                if (ct_it != cutlass_nvfp4_cache->end()) {
                    int M = static_cast<int>(input.shape[0]);
                    int K = static_cast<int>(input.shape[1]);
                    int N = static_cast<int>(it->second.N);
                    quantize_fp16_to_nvfp4_cutlass(input.data, cutlass_act_data,
                                                    cutlass_act_sf, M, K, stream);

                    // MXFP4 CUTLASS: UE8M0 scales per 32 elements (alternative to NVFP4)
                    if (mxfp4_cache != nullptr && mxfp4_act_sf != nullptr && K % 32 == 0) {
                        auto mx_it = mxfp4_cache->find(weight.data);
                        if (mx_it != mxfp4_cache->end()) {
                            quantize_fp16_to_mxfp4_cutlass(input.data, cutlass_act_data,
                                                            mxfp4_act_sf, M, K, stream);
                            bool ok = gemm_mxfp4_cutlass_sm120(
                                cutlass_act_data, mxfp4_act_sf,
                                mx_it->second,
                                output.data, M, N, K,
                                mxfp4_workspace, mxfp4_workspace_size,
                                stream);
                            if (ok) return;
                        }
                    }

                    // cuBLASLt NVFP4: same data/scale format, auto-tuned kernels
                    if (cublaslt_nvfp4_available()) {
                        bool ok = gemm_nvfp4_cublaslt(
                            cutlass_act_data, cutlass_act_sf,
                            ct_it->second,
                            output.data, M, N, K, stream);
                        if (ok) return;
                    }

                    // Fallback: CUTLASS sm_120 block-scaled kernel
                    bool ok = gemm_nvfp4_cutlass_sm120(
                        cutlass_act_data, cutlass_act_sf,
                        ct_it->second,
                        output.data, M, N, K,
                        cutlass_workspace, cutlass_workspace_size,
                        stream);
                    if (ok) return;
                }
            }
            if (input.shape[0] > 1) {
                gemm_nvfp4(it->second, input, output, stream);
            }
            return;
        }
    }
    if (input.shape[0] == 1 && input.dtype == DType::FP16 &&
               q8_1_buf != nullptr && d8_buf != nullptr && qtype == GGMLQuantType::Q4_0) {
        // dp4a MMVQ Q4_0: quantize input to Q8_1, then dp4a dot product
        int K = static_cast<int>(weight.shape[1]);
        quantize_fp16_to_q8_1(static_cast<const half*>(input.data),
                               q8_1_buf, d8_buf, K, stream);
        gemv_q4_0_q8_1(weight.data, q8_1_buf, d8_buf,
                        static_cast<half*>(output.data),
                        static_cast<int>(weight.shape[0]), K, stream);
    } else if (qtype == GGMLQuantType::Q4_1 && fp16_cache != nullptr) {
        // Prefer pre-dequantized FP16 cache (P3 optimization)
        auto it = fp16_cache->find(weight.data);
        if (it != fp16_cache->end()) {
            gemm(input, it->second, output, 1.0f, 0.0f, stream);
        } else {
            quant_gemm_int4(input, weight, scales, output, stream);
        }
    } else if (qtype == GGMLQuantType::Q4_1) {
        // weight is [N, K/2] packed nibbles, scales is [N, num_groups]
        quant_gemm_int4(input, weight, scales, output, stream);
    } else if (input.shape[0] == 1 && input.dtype == DType::FP16 &&
               q8_1_buf != nullptr && d8_buf != nullptr && qtype == GGMLQuantType::Q6_K) {
        // dp4a MMVQ Q6_K: quantize input to Q8_1, then dp4a dot product
        int K = static_cast<int>(weight.shape[1]);
        quantize_fp16_to_q8_1(static_cast<const half*>(input.data),
                               q8_1_buf, d8_buf, K, stream);
        gemv_q6k_q8_1(weight.data, q8_1_buf, d8_buf,
                       static_cast<half*>(output.data),
                       static_cast<int>(weight.shape[0]), K, stream);
    } else if (input.shape[0] == 1 && input.dtype == DType::FP16 &&
               q8_1_buf != nullptr && d8_buf != nullptr && qtype == GGMLQuantType::Q8_0) {
        // dp4a MMVQ Q8_0: quantize input to Q8_1, then dp4a dot product
        int K = static_cast<int>(weight.shape[1]);
        quantize_fp16_to_q8_1(static_cast<const half*>(input.data),
                               q8_1_buf, d8_buf, K, stream);
        gemv_q8_0_q8_1(weight.data, q8_1_buf, d8_buf,
                        static_cast<half*>(output.data),
                        static_cast<int>(weight.shape[0]), K, stream);
    } else if (input.shape[0] == 1 && input.dtype == DType::FP16 &&
               q8_1_buf != nullptr && d8_buf != nullptr && qtype == GGMLQuantType::Q4_K) {
        // dp4a MMVQ Q4_K
        int K = static_cast<int>(weight.shape[1]);
        quantize_fp16_to_q8_1(static_cast<const half*>(input.data),
                               q8_1_buf, d8_buf, K, stream);
        gemv_q4_k_q8_1(weight.data, q8_1_buf, d8_buf,
                         static_cast<half*>(output.data),
                         static_cast<int>(weight.shape[0]), K, stream);
    } else if (input.shape[0] == 1 && input.dtype == DType::FP16 &&
               q8_1_buf != nullptr && d8_buf != nullptr && qtype == GGMLQuantType::Q5_K) {
        // dp4a MMVQ Q5_K
        int K = static_cast<int>(weight.shape[1]);
        quantize_fp16_to_q8_1(static_cast<const half*>(input.data),
                               q8_1_buf, d8_buf, K, stream);
        gemv_q5_k_q8_1(weight.data, q8_1_buf, d8_buf,
                         static_cast<half*>(output.data),
                         static_cast<int>(weight.shape[0]), K, stream);
    } else if (input.shape[0] == 1 && input.dtype == DType::FP16 &&
               q8_1_buf != nullptr && d8_buf != nullptr && qtype == GGMLQuantType::Q2_K) {
        // dp4a MMVQ Q2_K
        int K = static_cast<int>(weight.shape[1]);
        quantize_fp16_to_q8_1(static_cast<const half*>(input.data),
                               q8_1_buf, d8_buf, K, stream);
        gemv_q2_k_q8_1(weight.data, q8_1_buf, d8_buf,
                         static_cast<half*>(output.data),
                         static_cast<int>(weight.shape[0]), K, stream);
    } else if (input.shape[0] == 1 && input.dtype == DType::FP16 &&
               q8_1_buf != nullptr && d8_buf != nullptr && qtype == GGMLQuantType::Q3_K) {
        // dp4a MMVQ Q3_K
        int K = static_cast<int>(weight.shape[1]);
        quantize_fp16_to_q8_1(static_cast<const half*>(input.data),
                               q8_1_buf, d8_buf, K, stream);
        gemv_q3_k_q8_1(weight.data, q8_1_buf, d8_buf,
                         static_cast<half*>(output.data),
                         static_cast<int>(weight.shape[0]), K, stream);
    } else if (input.shape[0] == 1 && input.dtype == DType::FP16 &&
               dequant_scratch != nullptr && qtype == GGMLQuantType::Q6_K) {
        // Fallback: Fused Q6_K GEMV (FP16 dequant path)
        gemv_q6k(weight.data, static_cast<const half*>(input.data),
                 static_cast<half*>(output.data),
                 static_cast<int>(weight.shape[0]), static_cast<int>(weight.shape[1]), stream);
    } else if (input.shape[0] == 1 && input.dtype == DType::FP16 &&
               dequant_scratch != nullptr && qtype == GGMLQuantType::Q8_0) {
        // Fallback: Fused Q8_0 GEMV (FP16 dequant path)
        gemv_q8_0(weight.data, static_cast<const half*>(input.data),
                  static_cast<half*>(output.data),
                  static_cast<int>(weight.shape[0]), static_cast<int>(weight.shape[1]), stream);
    } else if (fp8_cache != nullptr && input.shape[0] > 1 && fp8_act_buf != nullptr && d_act_scale != nullptr) {
        // FP8 cache: quantize activation → FP8, then FP8×FP8 cuBLASLt GEMM (2x throughput on sm_120)
        auto it = fp8_cache->find(weight.data);
        if (it != fp8_cache->end()) {
            Tensor fp8_act(fp8_act_buf, DType::FP8_E4M3, input.ndim, input.shape, true);
            quantize_fp16_to_fp8_e4m3(input, fp8_act, d_act_scale, stream,
                                       d_fp8_block_maxes, d_fp8_absmax, fp8_max_grid);
            gemm_cublaslt(fp8_act, it->second.weight, output, 1.0f, 0.0f,
                          d_act_scale, it->second.d_scale, stream);
        } else if (dequant_scratch != nullptr && dequant_gpu_supported(qtype)) {
            int rows = static_cast<int>(weight.shape[0]);
            int cols = static_cast<int>(weight.shape[1]);
            dequant_gpu(weight.data, dequant_scratch, qtype, rows, cols, stream);
            Tensor w_fp16(dequant_scratch, DType::FP16, weight.ndim, weight.shape, true);
            gemm(input, w_fp16, output, 1.0f, 0.0f, stream);
        } else {
            gemm(input, weight, output, 1.0f, 0.0f, stream);
        }
    } else if (fp16_cache != nullptr && dequant_gpu_supported(qtype)) {
        // Pre-dequantized FP16 cache: zero per-GEMM dequant overhead
        auto it = fp16_cache->find(weight.data);
        if (it != fp16_cache->end()) {
            gemm(input, it->second, output, 1.0f, 0.0f, stream);
        } else if (dequant_scratch != nullptr) {
            // Cache miss (shouldn't happen) — fall back to on-the-fly dequant
            int rows = static_cast<int>(weight.shape[0]);
            int cols = static_cast<int>(weight.shape[1]);
            dequant_gpu(weight.data, dequant_scratch, qtype, rows, cols, stream);
            Tensor w_fp16(dequant_scratch, DType::FP16, weight.ndim, weight.shape, true);
            gemm(input, w_fp16, output, 1.0f, 0.0f, stream);
        } else {
            gemm(input, weight, output, 1.0f, 0.0f, stream);
        }
    } else if (dequant_scratch != nullptr && dequant_gpu_supported(qtype)) {
        // Raw quantized bytes on GPU — dequant into scratch, then GEMM
        int rows = static_cast<int>(weight.shape[0]);
        int cols = static_cast<int>(weight.shape[1]);
        dequant_gpu(weight.data, dequant_scratch, qtype, rows, cols, stream);
        Tensor w_fp16(dequant_scratch, DType::FP16, weight.ndim, weight.shape, true);
        gemm(input, w_fp16, output, 1.0f, 0.0f, stream);
    } else {
        // Standard FP16/BF16 GEMM
        gemm(input, weight, output, 1.0f, 0.0f, stream);
    }
}


} // namespace imp
