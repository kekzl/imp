#include "graph/executor.h"
#include "compute/embedding.h"
#include "compute/layernorm.h"
#include "compute/rope.h"
#include "compute/gemm.h"
#include "compute/gemm_grouped.h"
#include "compute/gemm_moe_fused.h"
#include "compute/gemm_q6k.h"
#ifdef IMP_USE_CUTLASS
#include "compute/gemm_cutlass.h"
#endif
#include "compute/activation.h"
#include "compute/attention.h"
#include "compute/attention_cublas.h"
#include "compute/attention_paged.h"
#include "compute/moe_routing.h"
#include "compute/sampling.h"
#include "compute/ssm.h"
#include "quant/quant_gemm.h"
#include "quant/dequant_gpu.h"
#include "quant/fp8_quant.h"
#include "core/logging.h"
#include "memory/kv_cache.h"
#include "runtime/pdl.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#ifdef __CUDA_FP8_TYPES_EXIST__
#include <cuda_fp8.h>
#endif
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <algorithm>

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
    bool neox
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

            float freq = 1.0f / (powf(theta, (2.0f * pair_idx) / static_cast<float>(head_dim)));
            freq *= inv_scaling;
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
    bool neox
) {
    int head_idx  = blockIdx.y;
    int pair_idx  = threadIdx.x;
    if (head_idx >= n_heads || pair_idx >= rope_pairs) return;

    int pos = positions[0];  // decode: single token

    float freq = 1.0f / (powf(theta, (2.0f * pair_idx) / static_cast<float>(head_dim)));
    freq *= inv_scaling;
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

static void elementwise_add(Tensor& a, const Tensor& b, cudaStream_t stream) {
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
static void elementwise_add_store(const Tensor& a, const Tensor& b, Tensor& out,
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
static void add_bias(Tensor& out, const Tensor& bias, cudaStream_t stream) {
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
static Tensor slice_rows(const Tensor& buf, int n_tokens) {
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
static void gemm_dispatch(const Tensor& input, const Tensor& weight,
                           const Tensor& scales, GGMLQuantType qtype,
                           Tensor& output, void* dequant_scratch,
                           cudaStream_t stream,
                           block_q8_1* q8_1_buf = nullptr,
                           float* d8_buf = nullptr,
                           const std::unordered_map<const void*, Tensor>* fp16_cache = nullptr) {
    if (input.shape[0] == 1 && input.dtype == DType::FP16 &&
               q8_1_buf != nullptr && d8_buf != nullptr && qtype == GGMLQuantType::Q4_0) {
        // dp4a MMVQ Q4_0: quantize input to Q8_1, then dp4a dot product
        int K = static_cast<int>(weight.shape[1]);
        quantize_fp16_to_q8_1(static_cast<const half*>(input.data),
                               q8_1_buf, d8_buf, K, stream);
        gemv_q4_0_q8_1(weight.data, q8_1_buf, d8_buf,
                        static_cast<half*>(output.data),
                        static_cast<int>(weight.shape[0]), K, stream);
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

// ---------------------------------------------------------------------------
// GraphExecutor lifetime
// ---------------------------------------------------------------------------

GraphExecutor::~GraphExecutor() {
    free_buffers();
}

bool GraphExecutor::init(const Model& model, DType compute_dtype, bool use_pdl,
                         int max_batch_size, int max_seq_len) {
    if (initialized_) {
        free_buffers();
    }

    model_ = &model;
    compute_dtype_ = compute_dtype;
    norm_w_off_ = model.config().norm_weight_offset;
    use_pdl_ = use_pdl;

    const auto& cfg = model.config();

    // Detect model features for workspace sizing
    has_moe_ = (cfg.n_experts > 0 && cfg.n_experts_active > 0);
    has_ssm_ = (cfg.ssm_inner_size > 0);
    has_dense_ffn_ = (cfg.d_ff > 0);

    // Compute max expert FFN hidden dim from actual packed tensor shapes.
    // cfg.expert_d_ff may not match the actual tensor dimensions (e.g. Nemotron-H).
    max_expert_eff_ = cfg.expert_d_ff;
    if (has_moe_) {
        for (int li = 0; li < cfg.n_layers; li++) {
            const auto& L = model.layer(li);
            // gate/up packed: shape [n_experts, expert_d_ff, d_model]
            for (const auto* p : {&L.expert_gate_packed, &L.expert_up_packed}) {
                if (p->data && p->ndim >= 3)
                    max_expert_eff_ = std::max(max_expert_eff_, static_cast<int>(p->shape[1]));
            }
            // down packed: shape [n_experts, d_model, expert_d_ff]
            if (L.expert_down_packed.data && L.expert_down_packed.ndim >= 3)
                max_expert_eff_ = std::max(max_expert_eff_, static_cast<int>(L.expert_down_packed.shape[2]));
        }
        if (max_expert_eff_ != cfg.expert_d_ff) {
            IMP_LOG_WARN("expert_d_ff mismatch: config=%d, actual packed tensors=%d — using %d",
                         cfg.expert_d_ff, max_expert_eff_, max_expert_eff_);
        }
    }

    // Use engine-provided max_seq_len if given, otherwise fall back to model config.
    int effective_seq_len = (max_seq_len > 0) ? max_seq_len : cfg.max_seq_len;
    max_tokens_ = std::min(effective_seq_len, 4096);
    if (max_tokens_ <= 0) {
        max_tokens_ = 4096;
    }

    // Logits buffer only needs to hold tokens that require LM head projection:
    // - Prefill: 1 (last token only)
    // - Decode:  n_sequences (one per batch slot)
    max_logit_tokens_ = std::max(max_batch_size, 1);

    // Compute shared workspace sizes (no allocation — deferred to allocate_workspaces()).
    // Deferring GPU allocation maximizes VRAM available for expert weight upload.
    compute_shared_sizes(max_tokens_);

    // Build SSM layer index mapping
    if (has_ssm_) {
        ssm_layer_map_.resize(cfg.n_layers, -1);
        int ssm_idx = 0;
        for (int i = 0; i < cfg.n_layers; i++) {
            if (model_->layer(i).ssm_in.data != nullptr) {
                ssm_layer_map_[i] = ssm_idx++;
            }
        }
        IMP_LOG_INFO("SSM layers: %d out of %d total", ssm_idx, cfg.n_layers);
    }

    // Enable Programmatic Dependent Launch on custom kernels if requested.
    if (use_pdl_ && pdl::is_available()) {
        pdl::enable(reinterpret_cast<const void*>(&elementwise_add_fp16_kernel));
        pdl::enable(reinterpret_cast<const void*>(&elementwise_add_fp32_kernel));
        pdl::enable(reinterpret_cast<const void*>(&write_kv_cache_kernel));
        pdl::enable(reinterpret_cast<const void*>(&write_kv_cache_fused_kernel));
        pdl::enable(reinterpret_cast<const void*>(&write_kv_cache_rope_fused_kernel));
        pdl::enable(reinterpret_cast<const void*>(&fp16_to_fp32_kernel));
        pdl::enable(reinterpret_cast<const void*>(&fp32_to_fp16_kernel));
        // Register compute kernels for PDL overlap (run between GEMMs in hot path)
        layernorm_pdl_register();
        rope_pdl_register();
        activation_pdl_register();
        IMP_LOG_INFO("PDL enabled on executor + compute kernels");
    } else if (use_pdl_) {
        IMP_LOG_WARN("PDL requested but not available on this device/CUDA version");
        use_pdl_ = false;
    }

    initialized_ = true;

    IMP_LOG_INFO("GraphExecutor initialized: max_tokens=%d, d_model=%d, "
                 "n_layers=%d, dtype=%s, pdl=%s",
                 max_tokens_, cfg.d_model, cfg.n_layers,
                 dtype_name(compute_dtype_),
                 use_pdl_ ? "on" : "off");
    return true;
}

// ---------------------------------------------------------------------------
// Phase 2: allocate all GPU workspace buffers (called after weight upload)
// ---------------------------------------------------------------------------

bool GraphExecutor::allocate_workspaces(bool experts_on_host) {
    if (!initialized_ || !model_) return false;

    allocate_persistent_workspace(max_tokens_);
    allocate_shared_workspace(max_tokens_);
    allocate_auxiliary_buffers(/*skip_batch_dequant=*/experts_on_host);

    return true;
}

size_t GraphExecutor::workspace_estimate() const {
    if (!model_) return 0;
    const auto& cfg = model_->config();
    int d = cfg.d_model;
    size_t es = dtype_size(compute_dtype_);
    auto align256 = [](size_t x) -> size_t { return (x + 255) & ~size_t(255); };

    // Persistent: hidden + residual + norm_out + logits
    size_t persistent = 3 * align256(static_cast<size_t>(max_tokens_) * d * es)
                      + align256(static_cast<size_t>(max_logit_tokens_) * cfg.vocab_size * sizeof(float));

    // Shared: max of phases (already computed in compute_shared_sizes)
    size_t shared = std::max({attn_shared_size_, ffn_shared_size_,
                              moe_shared_size_, ssm_shared_size_});

    // S-matrix is NOT included here — it's optional (flash attention fallback works).
    // This maximizes VRAM available for expert layers during weight upload.
    // S-matrix is allocated opportunistically from remaining VRAM.

    // FP32 accumulator for post-norm models (Gemma-3): 1 × max_tokens × d_model × 4
    bool has_post_norms = (cfg.norm_placement == NormPlacement::POST_NORM);
    size_t fp32_accum = has_post_norms ? align256(static_cast<size_t>(max_tokens_) * d * sizeof(float)) : 0;

    // Auxiliary: dequant scratch, MoE dequant/staging, sampling, split-K, etc.
    size_t auxiliary = 64ULL << 20;  // conservative 64 MiB for misc buffers

    return persistent + shared + fp32_accum + auxiliary;
}

// ---------------------------------------------------------------------------
// Unified workspace allocation
// ---------------------------------------------------------------------------

void GraphExecutor::compute_shared_sizes(int max_tokens) {
    const auto& cfg = model_->config();
    int d   = cfg.d_model;
    int ff  = cfg.d_ff;
    int nh  = cfg.n_heads;
    int nkv = cfg.n_kv_heads;
    int hd  = cfg.head_dim > 0 ? cfg.head_dim : (d / nh);
    size_t es = dtype_size(compute_dtype_);

    auto align256 = [](size_t x) -> size_t { return (x + 255) & ~size_t(255); };

    // Attention phase: q, k+v (contiguous for batched GEMM), attn_out, proj_out
    size_t kv_raw = static_cast<size_t>(max_tokens) * nkv * hd * es;
    attn_shared_size_ = align256(static_cast<size_t>(max_tokens) * nh * hd * es)    // q
                       + align256(2 * kv_raw)                                       // k+v contiguous
                       + align256(static_cast<size_t>(max_tokens) * nh * hd * es)   // attn_out
                       + align256(static_cast<size_t>(max_tokens) * d * es);        // proj_out

    // Dense FFN phase: gate, up, swiglu, ffn_out
    if (has_dense_ffn_ && ff > 0) {
        ffn_shared_size_ = align256(static_cast<size_t>(max_tokens) * ff * es)   // gate_out
                          + align256(static_cast<size_t>(max_tokens) * ff * es)  // up_out
                          + align256(static_cast<size_t>(max_tokens) * ff * es)  // swiglu_out
                          + align256(static_cast<size_t>(max_tokens) * d * es);  // ffn_out
    }

    // MoE phase
    if (has_moe_) {
        int ne    = cfg.n_experts;
        int top_k = cfg.n_experts_active;
        int eff   = max_expert_eff_;
        int expanded = max_tokens * top_k;

        moe_shared_size_ = align256(static_cast<size_t>(max_tokens) * ne * sizeof(float))  // gate_logits
                          + align256(static_cast<size_t>(expanded) * d * es)                // gathered
                          + align256(static_cast<size_t>(expanded) * eff * es)              // expert_gate
                          + align256(static_cast<size_t>(expanded) * eff * es)              // expert_up
                          + align256(static_cast<size_t>(expanded) * eff * es)              // expert_swiglu
                          + align256(static_cast<size_t>(expanded) * d * es)                // expert_down
                          + align256(static_cast<size_t>(max_tokens) * d * sizeof(float));  // scatter_out
    }

    // SSM phase
    if (has_ssm_) {
        int inner = cfg.ssm_inner_size;
        int n_groups = cfg.ssm_group_count;
        int state_size = cfg.ssm_state_size;
        int n_heads = cfg.ssm_dt_rank;
        int conv_channels = inner + 2 * n_groups * state_size;
        int ssm_in_dim = inner + conv_channels + n_heads;

        ssm_shared_size_ = align256(static_cast<size_t>(max_tokens) * ssm_in_dim * es)       // proj
                          + align256(static_cast<size_t>(max_tokens) * conv_channels * es)   // xBC
                          + align256(static_cast<size_t>(max_tokens) * inner * es)           // y
                          + align256(static_cast<size_t>(max_tokens) * inner * es)           // z
                          + align256(static_cast<size_t>(max_tokens) * d * es)               // out
                          + align256(static_cast<size_t>(max_tokens) * n_heads * es);        // dt
    }
}

void GraphExecutor::allocate_persistent_workspace(int max_tokens) {
    const auto& cfg = model_->config();
    int d = cfg.d_model;
    int v = cfg.vocab_size;
    size_t es = dtype_size(compute_dtype_);

    auto align256 = [](size_t x) -> size_t { return (x + 255) & ~size_t(255); };

    size_t hidden_sz   = align256(static_cast<size_t>(max_tokens) * d * es);
    size_t residual_sz = align256(static_cast<size_t>(max_tokens) * d * es);
    size_t norm_out_sz = align256(static_cast<size_t>(max_tokens) * d * es);
    size_t logits_sz   = align256(static_cast<size_t>(max_logit_tokens_) * v * sizeof(float));

    size_t total = hidden_sz + residual_sz + norm_out_sz + logits_sz;

    cudaError_t err = cudaMalloc(&persistent_workspace_, total);
    if (err != cudaSuccess) {
        IMP_LOG_ERROR("Failed to allocate persistent workspace (%zu bytes): %s",
                      total, cudaGetErrorString(err));
        return;
    }
    persistent_workspace_size_ = total;

    char* ptr = static_cast<char*>(persistent_workspace_);

    auto make = [&](int64_t cols, size_t aligned_sz) -> Tensor {
        int64_t shape[2] = {static_cast<int64_t>(max_tokens), cols};
        Tensor t(ptr, compute_dtype_, 2, shape, true);
        ptr += aligned_sz;
        return t;
    };

    hidden_   = make(d, hidden_sz);
    residual_ = make(d, residual_sz);
    norm_out_ = make(d, norm_out_sz);

    {
        int64_t shape[2] = {static_cast<int64_t>(max_logit_tokens_), static_cast<int64_t>(v)};
        logits_ = Tensor(ptr, DType::FP32, 2, shape, true);
        ptr += logits_sz;
    }

    IMP_LOG_INFO("Persistent workspace: %.2f MiB (hidden+residual+norm+logits)",
                 total / (1024.0 * 1024.0));

    // FP32 residual accumulator for post-norm architectures (Gemma-3).
    if (cfg.norm_placement == NormPlacement::POST_NORM) {
        size_t fp32_sz = align256(static_cast<size_t>(max_tokens) * d * sizeof(float));
        cudaError_t e2 = cudaMalloc(&fp32_accum_buf_, fp32_sz);
        if (e2 == cudaSuccess) {
            int64_t shape[2] = {static_cast<int64_t>(max_tokens), static_cast<int64_t>(d)};
            fp32_hidden_ = Tensor(fp32_accum_buf_, DType::FP32, 2, shape, true);
            IMP_LOG_INFO("FP32 residual accumulator: %.2f MiB (post-norm architecture)",
                         fp32_sz / (1024.0 * 1024.0));
        } else {
            IMP_LOG_WARN("Failed to allocate FP32 accumulator (%zu bytes): %s — falling back to FP16",
                         fp32_sz, cudaGetErrorString(e2));
        }
    }
}

void GraphExecutor::allocate_shared_workspace(int max_tokens) {
    size_t max_shared = std::max({attn_shared_size_, ffn_shared_size_,
                                  moe_shared_size_, ssm_shared_size_});
    if (max_shared == 0) return;

    cudaError_t err = cudaMalloc(&shared_workspace_, max_shared);
    if (err != cudaSuccess) {
        IMP_LOG_ERROR("Failed to allocate shared workspace (%zu bytes): %s",
                      max_shared, cudaGetErrorString(err));
        return;
    }
    shared_workspace_size_ = max_shared;
    shared_workspace_max_tokens_ = max_tokens;

    IMP_LOG_INFO("Shared workspace: %.2f MiB = max(attn=%.1f, ffn=%.1f, moe=%.1f, ssm=%.1f MiB) "
                 "— saved %.2f MiB vs separate allocation",
                 max_shared / (1024.0 * 1024.0),
                 attn_shared_size_ / (1024.0 * 1024.0),
                 ffn_shared_size_ / (1024.0 * 1024.0),
                 moe_shared_size_ / (1024.0 * 1024.0),
                 ssm_shared_size_ / (1024.0 * 1024.0),
                 (attn_shared_size_ + ffn_shared_size_ + moe_shared_size_ + ssm_shared_size_
                  - max_shared) / (1024.0 * 1024.0));

    // Pre-allocate MoE routing buffers (separate from shared workspace)
    if (has_moe_) {
        const auto& cfg = model_->config();
        moe_routing_buffers_.allocate(max_tokens, cfg.n_experts, cfg.n_experts_active);
    }
}

void GraphExecutor::allocate_auxiliary_buffers(bool skip_batch_dequant) {
    const auto& cfg = model_->config();

    // Dequant scratch buffer for on-the-fly weight dequantization.
    {
        size_t max_weight_elems = 0;
        for (int i = 0; i < cfg.n_layers; i++) {
            const auto& L = model_->layer(i);
            for (const auto* w : {&L.wq, &L.wk, &L.wv, &L.wo,
                                   &L.w_gate, &L.w_up, &L.w_down,
                                   &L.w_gate_shared, &L.w_up_shared, &L.w_down_shared,
                                   &L.ssm_in, &L.ssm_out}) {
                if (w->data) max_weight_elems = std::max(max_weight_elems,
                                                          static_cast<size_t>(w->numel()));
            }
        }
        if (max_weight_elems > 0) {
            dequant_scratch_size_ = max_weight_elems * sizeof(uint16_t);
            cudaError_t err = cudaMalloc(&dequant_scratch_, dequant_scratch_size_);
            if (err != cudaSuccess) {
                IMP_LOG_ERROR("Failed to allocate dequant scratch (%zu bytes): %s",
                              dequant_scratch_size_, cudaGetErrorString(err));
                dequant_scratch_ = nullptr;
                dequant_scratch_size_ = 0;
            } else {
                IMP_LOG_INFO("Dequant scratch buffer: %.2f MiB",
                             dequant_scratch_size_ / (1024.0 * 1024.0));
            }
        }
    }

    // Sampling result buffer (avoids cudaMalloc/cudaFree per token)
    {
        cudaError_t err = cudaMalloc(&d_sample_result_, sizeof(int32_t));
        if (err != cudaSuccess) {
            IMP_LOG_ERROR("Failed to allocate sampling result buffer: %s",
                          cudaGetErrorString(err));
            d_sample_result_ = nullptr;
        }
    }

    // MMVQ (dp4a) scratch buffers for quantized input vectors.
    // Find the max Q8_1 block count needed across all uses:
    //   1. Dense GEMV: max_k / 32 blocks (one input vector)
    //   2. MoE down projection: top_k * expert_d_ff / 32 blocks (per-expert quantized activations)
    {
        int max_k = 0;
        int max_moe_down_blocks = 0;
        for (int i = 0; i < cfg.n_layers; i++) {
            const auto& L = model_->layer(i);
            for (const auto* w : {&L.wq, &L.wk, &L.wv, &L.wo,
                                   &L.w_gate, &L.w_up, &L.w_down,
                                   &L.w_gate_shared, &L.w_up_shared, &L.w_down_shared,
                                   &L.ssm_in, &L.ssm_out}) {
                if (w->data && w->ndim >= 2) {
                    max_k = std::max(max_k, static_cast<int>(w->shape[1]));
                }
            }
            // MoE expert weight inner dims
            if (L.expert_up_packed.data && L.expert_up_packed.ndim >= 3) {
                max_k = std::max(max_k, static_cast<int>(L.expert_up_packed.shape[2]));
            }
            if (L.expert_down_packed.data && L.expert_down_packed.ndim >= 3) {
                int down_k = static_cast<int>(L.expert_down_packed.shape[2]);
                max_k = std::max(max_k, down_k);
                // MoE down projection quantizes top_k expert activations contiguously
                max_moe_down_blocks = std::max(max_moe_down_blocks,
                    cfg.n_experts_active * (down_k / 32));
            }
            if (L.expert_gate_packed.data && L.expert_gate_packed.ndim >= 3) {
                max_k = std::max(max_k, static_cast<int>(L.expert_gate_packed.shape[2]));
            }
        }
        int max_blocks = std::max(max_k / 32, max_moe_down_blocks);
        if (max_blocks > 0) {
            q8_1_max_blocks_ = max_blocks;
            size_t q8_1_sz = static_cast<size_t>(q8_1_max_blocks_) * sizeof(block_q8_1);
            size_t d8_sz = static_cast<size_t>(q8_1_max_blocks_) * sizeof(float);
            cudaError_t err1 = cudaMalloc(&q8_1_buf_, q8_1_sz);
            cudaError_t err2 = cudaMalloc(reinterpret_cast<void**>(&d8_buf_), d8_sz);
            if (err1 != cudaSuccess || err2 != cudaSuccess) {
                IMP_LOG_WARN("Failed to allocate MMVQ scratch buffers, dp4a path disabled");
                if (q8_1_buf_) { cudaFree(q8_1_buf_); q8_1_buf_ = nullptr; }
                if (d8_buf_) { cudaFree(d8_buf_); d8_buf_ = nullptr; }
                q8_1_max_blocks_ = 0;
            } else {
                IMP_LOG_INFO("MMVQ scratch buffers: %.2f KiB (q8_1) + %.2f KiB (d8), max_blocks=%d (max_k=%d, moe_down=%d)",
                             q8_1_sz / 1024.0, d8_sz / 1024.0, max_blocks, max_k, max_moe_down_blocks);
            }
        }
    }

    // Split-K paged attention scratch buffer.
    // Sized for max_batch_size * n_heads * max_splits * (2 + head_dim) floats.
    {
        int nh = cfg.n_heads;
        int hd = cfg.head_dim > 0 ? cfg.head_dim : (cfg.d_model / nh);
        int max_splits = 32;
        int partial_stride = 2 + hd;
        int max_batch = max_logit_tokens_;  // = max_batch_size
        size_t sz = static_cast<size_t>(max_batch) * nh * max_splits * partial_stride * sizeof(float);
        cudaError_t err = cudaMalloc(&splitk_scratch_, sz);
        if (err != cudaSuccess) {
            IMP_LOG_WARN("Failed to allocate split-K scratch (%zu bytes), split-K disabled", sz);
            splitk_scratch_ = nullptr;
            splitk_scratch_size_ = 0;
        } else {
            splitk_scratch_size_ = sz;
            IMP_LOG_INFO("Split-K paged attention scratch: %.2f KiB", sz / 1024.0);
        }
    }

    // cuBLAS attention S-matrix workspace: [n_heads, attn_seq, attn_seq] FP16
    // Only useful for prefill; budget-capped to avoid eating VRAM needed for KV cache.
    // For prefills longer than attn_seq, falls back to WMMA attention.
    // When VRAM is tight (experts on host), skip entirely to preserve VRAM for FP16 cache.
    if (!skip_batch_dequant) {
        int nh = cfg.n_heads;
        constexpr size_t kMaxAttnScoresMiB = 256;  // cap at 256 MiB
        size_t max_s_sz = kMaxAttnScoresMiB << 20;
        // max seq = sqrt(budget / (n_heads * sizeof(half)))
        int attn_seq = max_tokens_;
        size_t s_sz = static_cast<size_t>(nh) * attn_seq * attn_seq * sizeof(half);
        if (s_sz > max_s_sz) {
            attn_seq = static_cast<int>(std::sqrt(
                static_cast<double>(max_s_sz) / (nh * sizeof(half))));
            attn_seq = (attn_seq / 16) * 16;  // round down to multiple of 16
            if (attn_seq < 32) attn_seq = 0;  // too small to be useful
            s_sz = static_cast<size_t>(nh) * attn_seq * attn_seq * sizeof(half);
        }
        if (attn_seq > 0) {
            cudaError_t err = cudaMalloc(&attn_scores_buf_, s_sz);
            if (err != cudaSuccess) {
                cudaGetLastError();  // clear sticky error from failed cudaMalloc
                IMP_LOG_WARN("Failed to allocate cuBLAS attention S-matrix (%zu bytes, %.1f MiB), "
                             "will fall back to WMMA attention for prefill",
                             s_sz, s_sz / (1024.0 * 1024.0));
                attn_scores_buf_ = nullptr;
                attn_scores_buf_size_ = 0;
            } else {
                attn_scores_buf_size_ = s_sz;
                int64_t s_shape[3] = {static_cast<int64_t>(nh),
                                      static_cast<int64_t>(attn_seq),
                                      static_cast<int64_t>(attn_seq)};
                attn_scores_ = Tensor(attn_scores_buf_, DType::FP16, 3, s_shape, true);
                IMP_LOG_INFO("cuBLAS attention S-matrix: %.2f MiB (%d heads x %d x %d)",
                             s_sz / (1024.0 * 1024.0), nh, attn_seq, attn_seq);
            }
        }
    } else {
        IMP_LOG_INFO("cuBLAS attention S-matrix: skipped (VRAM-constrained, using WMMA/TCGEN05 fallback)");
    }

    // MoE dequant and staging buffers
    if (has_moe_) {
        int d   = cfg.d_model;
        int eff = max_expert_eff_;

        // Dequant buffer: 1 expert slot
        {
            size_t expert_fp16_elems = static_cast<size_t>(eff) * d;
            size_t dequant_sz = expert_fp16_elems * sizeof(uint16_t);
            cudaError_t err = cudaMalloc(&moe_dequant_buf_, dequant_sz);
            if (err != cudaSuccess) {
                IMP_LOG_ERROR("Failed to allocate MoE dequant buffer (%zu bytes): %s",
                              dequant_sz, cudaGetErrorString(err));
                moe_dequant_buf_ = nullptr;
                moe_dequant_buf_size_ = 0;
            } else {
                moe_dequant_buf_size_ = dequant_sz;
                IMP_LOG_INFO("MoE dequant buffer: %.2f MiB (1 expert slot)",
                             dequant_sz / (1024.0 * 1024.0));
            }
        }

        // Staging buffer for host→device expert weight transfer
        {
            size_t max_expert_raw = 0;
            for (int li = 0; li < model_->n_layers(); li++) {
                const auto& L = model_->layer(li);
                auto check = [&](const Tensor& p, GGMLQuantType qt) {
                    if (!p.data || p.ndim < 3) return;
                    size_t rb = ggml_quant_row_bytes(qt, p.shape[2]);
                    size_t expert_raw = static_cast<size_t>(p.shape[1]) * rb;
                    max_expert_raw = std::max(max_expert_raw, expert_raw);
                };
                check(L.expert_up_packed, L.expert_up_qtype);
                check(L.expert_down_packed, L.expert_down_qtype);
                check(L.expert_gate_packed, L.expert_gate_qtype);
            }
            if (max_expert_raw > 0) {
                cudaError_t err = cudaMalloc(&moe_raw_staging_buf_, max_expert_raw);
                if (err != cudaSuccess) {
                    IMP_LOG_ERROR("Failed to allocate MoE staging buffer (%zu bytes): %s",
                                  max_expert_raw, cudaGetErrorString(err));
                    moe_raw_staging_buf_ = nullptr;
                    moe_raw_staging_size_ = 0;
                } else {
                    moe_raw_staging_size_ = max_expert_raw;
                    IMP_LOG_INFO("MoE staging buffer: %.2f MiB (1 expert raw)",
                                 max_expert_raw / (1024.0 * 1024.0));
                }
            }
        }

        // Batch dequant buffer: sized for a chunk of experts (L2-resident strategy).
        // We dequant a chunk of experts to FP16, then immediately GEMM while the
        // FP16 data is still warm in L2 cache (~96 MB on RTX 5090). This avoids
        // writing the FP16 intermediate to DRAM entirely, saving ~5x DRAM traffic.
        // Skip allocation if experts are on host (batch dequant only useful for on-device experts).
        if (!skip_batch_dequant) {
            int targets[] = {cfg.n_experts, cfg.n_experts / 2, 32, 16};
            bool allocated = false;
            for (int ne_try : targets) {
                if (ne_try <= 0) continue;
                ne_try = std::min(ne_try, cfg.n_experts);
                size_t sz = static_cast<size_t>(ne_try) * eff * d * sizeof(half);
                cudaError_t err = cudaMalloc(&moe_batch_dequant_buf_, sz);
                if (err != cudaSuccess) {
                    IMP_LOG_DEBUG("MoE dequant buf alloc failed for %d experts: %s",
                                 ne_try, cudaGetErrorString(cudaGetLastError()));
                    continue;
                }
                moe_batch_dequant_buf_size_ = sz;
                allocated = true;
                IMP_LOG_INFO("MoE batch dequant buffer: %.2f MiB (%d experts)",
                             sz / (1024.0 * 1024.0), ne_try);
                break;
            }
            if (!allocated) {
                IMP_LOG_INFO("MoE batch dequant buffer: skipped (VRAM insufficient)");
                moe_batch_dequant_buf_ = nullptr;
                moe_batch_dequant_buf_size_ = 0;
            }
        } else {
            IMP_LOG_INFO("MoE batch dequant buffer: skipped (experts on host)");
            moe_batch_dequant_buf_ = nullptr;
            moe_batch_dequant_buf_size_ = 0;
        }

        // Pre-allocated device pointer arrays for batched MoE GEMM.
        // 3 arrays × n_experts void pointers = trivial memory (< 4 KB).
        // Eliminates cudaMallocAsync/FreeAsync from the hot path.
        if (cfg.n_experts > 0) {
            size_t ptr_bytes = 3 * static_cast<size_t>(cfg.n_experts) * sizeof(void*);
            cudaError_t err = cudaMalloc(&d_moe_work_ptrs_, ptr_bytes);
            if (err == cudaSuccess) {
                d_moe_work_ptrs_count_ = cfg.n_experts;
            } else {
                IMP_LOG_DEBUG("Cleared optional MoE work ptrs alloc error: %s", cudaGetErrorString(cudaGetLastError()));
                d_moe_work_ptrs_ = nullptr;
                d_moe_work_ptrs_count_ = 0;
            }

            // Per-expert FP8 scale buffer (trivial: 128 experts × 4 bytes = 512 bytes).
            size_t scale_bytes = static_cast<size_t>(cfg.n_experts) * sizeof(float);
            err = cudaMalloc(&d_moe_fp8_scales_, scale_bytes);
            if (err != cudaSuccess) {
                IMP_LOG_DEBUG("Cleared optional MoE FP8 scales alloc error: %s", cudaGetErrorString(cudaGetLastError()));
                d_moe_fp8_scales_ = nullptr;
            }

            // Device-side weight pointer array for device-grouped GEMM.
            size_t wptr_bytes = static_cast<size_t>(cfg.n_experts) * sizeof(void*);
            err = cudaMalloc(&d_moe_weight_ptrs_, wptr_bytes);
            if (err == cudaSuccess) {
                d_moe_weight_ptrs_count_ = cfg.n_experts;
            } else {
                IMP_LOG_DEBUG("Cleared optional MoE weight ptrs alloc error: %s", cudaGetErrorString(cudaGetLastError()));
                d_moe_weight_ptrs_ = nullptr;
                d_moe_weight_ptrs_count_ = 0;
            }
        }
    }
}

void GraphExecutor::release_moe_batch_buf() {
    if (moe_batch_dequant_buf_) {
        size_t freed = moe_batch_dequant_buf_size_;
        cudaFree(moe_batch_dequant_buf_);
        moe_batch_dequant_buf_ = nullptr;
        moe_batch_dequant_buf_size_ = 0;
        IMP_LOG_INFO("Released MoE batch dequant buffer: %.2f MiB (experts on host)",
                     freed / (1024.0 * 1024.0));
    }
}

void GraphExecutor::free_buffers() {
    // Free fused KV weight cache
    for (auto& [idx, tensor] : fused_kv_cache_) {
        if (tensor.data) cudaFree(tensor.data);
    }
    fused_kv_cache_.clear();

    // Free fused gate+up weight cache
    for (auto& [idx, tensor] : fused_gate_up_cache_) {
        if (tensor.data) cudaFree(tensor.data);
    }
    fused_gate_up_cache_.clear();

    // Free FP16 weight cache
    for (auto& [ptr, tensor] : fp16_cache_) {
        cudaFree(tensor.data);
    }
    fp16_cache_.clear();
    fp16_cache_bytes_ = 0;

    moe_routing_buffers_.free();
    if (moe_dequant_buf_) {
        cudaFree(moe_dequant_buf_);
        moe_dequant_buf_ = nullptr;
        moe_dequant_buf_size_ = 0;
    }
    if (moe_raw_staging_buf_) {
        cudaFree(moe_raw_staging_buf_);
        moe_raw_staging_buf_ = nullptr;
        moe_raw_staging_size_ = 0;
    }
    if (moe_batch_dequant_buf_) {
        cudaFree(moe_batch_dequant_buf_);
        moe_batch_dequant_buf_ = nullptr;
    }
    moe_batch_dequant_buf_size_ = 0;
    if (d_moe_work_ptrs_) {
        cudaFree(d_moe_work_ptrs_);
        d_moe_work_ptrs_ = nullptr;
        d_moe_work_ptrs_count_ = 0;
    }
    if (d_moe_fp8_scales_) {
        cudaFree(d_moe_fp8_scales_);
        d_moe_fp8_scales_ = nullptr;
    }
    if (d_moe_weight_ptrs_) {
        cudaFree(d_moe_weight_ptrs_);
        d_moe_weight_ptrs_ = nullptr;
        d_moe_weight_ptrs_count_ = 0;
    }
    if (dequant_scratch_) {
        cudaFree(dequant_scratch_);
        dequant_scratch_ = nullptr;
        dequant_scratch_size_ = 0;
    }
    if (d_sample_result_) {
        cudaFree(d_sample_result_);
        d_sample_result_ = nullptr;
    }
    if (q8_1_buf_) {
        cudaFree(q8_1_buf_);
        q8_1_buf_ = nullptr;
    }
    if (d8_buf_) {
        cudaFree(d8_buf_);
        d8_buf_ = nullptr;
    }
    q8_1_max_blocks_ = 0;
    if (splitk_scratch_) {
        cudaFree(splitk_scratch_);
        splitk_scratch_ = nullptr;
        splitk_scratch_size_ = 0;
    }
    if (attn_scores_buf_) {
        cudaFree(attn_scores_buf_);
        attn_scores_buf_ = nullptr;
        attn_scores_buf_size_ = 0;
    }
    if (shared_workspace_) {
        cudaFree(shared_workspace_);
        shared_workspace_ = nullptr;
        shared_workspace_size_ = 0;
    }
    if (persistent_workspace_) {
        cudaFree(persistent_workspace_);
        persistent_workspace_ = nullptr;
        persistent_workspace_size_ = 0;
    }
    if (fp32_accum_buf_) {
        cudaFree(fp32_accum_buf_);
        fp32_accum_buf_ = nullptr;
    }
    ssm_layer_map_.clear();
    initialized_ = false;
}

// ---------------------------------------------------------------------------
// Pre-dequantize quantized weights to FP16 on GPU
// ---------------------------------------------------------------------------

void GraphExecutor::pre_dequant_weights(cudaStream_t stream) {
    if (!initialized_ || !model_) return;

    const auto& cfg = model_->config();
    size_t total_fp16_bytes = 0;
    int cached_count = 0;
    bool budget_exhausted = false;

    // Query free VRAM and set a budget. Reserve enough for KV cache + runtime.
    // On WSL2/WDDM, cudaMalloc succeeds beyond physical VRAM via overcommit,
    // which exhausts virtual address space and causes OOM for later allocations.
    size_t free_vram = 0, total_vram = 0;
    cudaMemGetInfo(&free_vram, &total_vram);
    constexpr size_t kVramReserveMiB = 1024;  // Reserve headroom for CUDA graph instantiation + runtime + driver
    size_t vram_budget = (free_vram > kVramReserveMiB * 1024ULL * 1024)
                         ? (free_vram - kVramReserveMiB * 1024ULL * 1024) : 0;

    auto cache_weight = [&](const Tensor& w, GGMLQuantType qtype) {
        if (!w.data || !dequant_gpu_supported(qtype)) return;
        if (fp16_cache_.count(w.data)) return;  // already cached
        if (budget_exhausted) return;

        int rows = static_cast<int>(w.shape[0]);
        int cols = static_cast<int>(w.shape[1]);
        size_t fp16_bytes = static_cast<size_t>(rows) * cols * sizeof(half);

        if (total_fp16_bytes + fp16_bytes > vram_budget) {
            budget_exhausted = true;
            IMP_LOG_INFO("FP16 cache: VRAM budget reached after %d tensors (%.1f / %.1f MiB), "
                         "remaining weights will use on-the-fly dequant",
                         cached_count, total_fp16_bytes / (1024.0 * 1024.0),
                         vram_budget / (1024.0 * 1024.0));
            return;
        }

        void* fp16_buf = nullptr;
        cudaError_t err = cudaMalloc(&fp16_buf, fp16_bytes);
        if (err != cudaSuccess) {
            IMP_LOG_DEBUG("Cleared FP16 cache alloc error: %s", cudaGetErrorString(cudaGetLastError()));
            budget_exhausted = true;
            IMP_LOG_WARN("FP16 cache: cudaMalloc failed after %d tensors (%.1f MiB)",
                         cached_count, total_fp16_bytes / (1024.0 * 1024.0));
            return;
        }

        dequant_gpu(w.data, fp16_buf, qtype, rows, cols, stream);

        Tensor fp16_tensor(fp16_buf, DType::FP16, w.ndim, w.shape, true);
        fp16_cache_[w.data] = fp16_tensor;
        total_fp16_bytes += fp16_bytes;
        cached_count++;
    };

    // Priority order: attention weights first (critical for cuBLAS prefill),
    // then SSM, shared experts, and dense FFN.  This ensures hybrid models
    // like Nemotron (23 SSM + 6 attention layers) cache all attention weights
    // before SSM weights exhaust the VRAM budget.
    for (int i = 0; i < cfg.n_layers; i++) {
        const auto& L = model_->layer(i);
        cache_weight(L.wq, L.wq_qtype);
        cache_weight(L.wk, L.wk_qtype);
        cache_weight(L.wv, L.wv_qtype);
        cache_weight(L.wo, L.wo_qtype);
    }
    for (int i = 0; i < cfg.n_layers; i++) {
        const auto& L = model_->layer(i);
        cache_weight(L.ssm_in, L.ssm_in_qtype);
        cache_weight(L.ssm_out, L.ssm_out_qtype);
        cache_weight(L.w_gate_shared, L.w_gate_shared_qtype);
        cache_weight(L.w_up_shared, L.w_up_shared_qtype);
        cache_weight(L.w_down_shared, L.w_down_shared_qtype);
        cache_weight(L.w_gate, L.w_gate_qtype);
        cache_weight(L.w_up, L.w_up_qtype);
        cache_weight(L.w_down, L.w_down_qtype);
    }

    // Create fused KV weights for strided batched prefill GEMM.
    // Each entry concatenates [wk; wv] as [2*nkv*hd, d_model] FP16 for one layer.
    int fused_kv_count = 0;
    for (int i = 0; i < cfg.n_layers; i++) {
        const auto& L = model_->layer(i);
        if (!L.wk.data || !L.wv.data) continue;
        auto wk_it = fp16_cache_.find(L.wk.data);
        auto wv_it = fp16_cache_.find(L.wv.data);
        if (wk_it == fp16_cache_.end() || wv_it == fp16_cache_.end()) continue;

        int k_rows = static_cast<int>(L.wk.shape[0]);  // nkv * hd
        int K = static_cast<int>(L.wk.shape[1]);        // d_model
        size_t one_sz = static_cast<size_t>(k_rows) * K * sizeof(half);

        // Respect VRAM budget — on WSL2/WDDM, cudaMalloc silently spills to
        // shared (system) memory beyond physical VRAM, causing massive slowdowns.
        if (total_fp16_bytes + 2 * one_sz > vram_budget) break;

        void* fused_buf = nullptr;
        cudaError_t err = cudaMalloc(&fused_buf, 2 * one_sz);
        if (err != cudaSuccess) {
            IMP_LOG_DEBUG("Cleared fused KV alloc error: %s", cudaGetErrorString(cudaGetLastError()));
            break;
        }

        cudaMemcpyAsync(fused_buf, wk_it->second.data, one_sz,
                         cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(static_cast<char*>(fused_buf) + one_sz,
                         wv_it->second.data, one_sz,
                         cudaMemcpyDeviceToDevice, stream);

        int64_t shape[2] = {2 * k_rows, static_cast<int64_t>(K)};
        fused_kv_cache_[i] = Tensor(fused_buf, DType::FP16, 2, shape, true);
        total_fp16_bytes += 2 * one_sz;
        fused_kv_count++;
    }

    // Create fused gate+up weights for strided batched prefill GEMM.
    // Each entry concatenates [w_gate; w_up] as [2*d_ff, d_model] FP16 for one layer.
    int fused_gu_count = 0;
    for (int i = 0; i < cfg.n_layers; i++) {
        const auto& L = model_->layer(i);
        if (!L.w_gate.data || !L.w_up.data) continue;
        // Both must be the same shape (d_ff x d_model)
        if (L.w_gate.shape[0] != L.w_up.shape[0] ||
            L.w_gate.shape[1] != L.w_up.shape[1]) continue;
        auto wg_it = fp16_cache_.find(L.w_gate.data);
        auto wu_it = fp16_cache_.find(L.w_up.data);
        if (wg_it == fp16_cache_.end() || wu_it == fp16_cache_.end()) continue;

        int g_rows = static_cast<int>(L.w_gate.shape[0]);  // d_ff
        int K = static_cast<int>(L.w_gate.shape[1]);        // d_model
        size_t one_sz = static_cast<size_t>(g_rows) * K * sizeof(half);

        // Respect VRAM budget (see fused KV comment above)
        if (total_fp16_bytes + 2 * one_sz > vram_budget) break;

        void* fused_buf = nullptr;
        cudaError_t err = cudaMalloc(&fused_buf, 2 * one_sz);
        if (err != cudaSuccess) {
            IMP_LOG_DEBUG("Cleared fused gate+up alloc error: %s", cudaGetErrorString(cudaGetLastError()));
            break;
        }

        cudaMemcpyAsync(fused_buf, wg_it->second.data, one_sz,
                         cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(static_cast<char*>(fused_buf) + one_sz,
                         wu_it->second.data, one_sz,
                         cudaMemcpyDeviceToDevice, stream);

        int64_t shape[2] = {2 * g_rows, static_cast<int64_t>(K)};
        fused_gate_up_cache_[i] = Tensor(fused_buf, DType::FP16, 2, shape, true);
        total_fp16_bytes += 2 * one_sz;
        fused_gu_count++;
    }

    if (cached_count > 0) {
        cudaStreamSynchronize(stream);
        fp16_cache_bytes_ = total_fp16_bytes;
        IMP_LOG_INFO("FP16 weight cache: %d tensors, %.2f MiB (incl. %d fused KV, %d fused gate+up)",
                     cached_count, total_fp16_bytes / (1024.0 * 1024.0),
                     fused_kv_count, fused_gu_count);
    }
}

// ---------------------------------------------------------------------------
// Shared workspace configuration (pure pointer arithmetic, no allocation)
// ---------------------------------------------------------------------------

void GraphExecutor::configure_attn_workspace(int max_tokens) {
    const auto& cfg = model_->config();
    int d   = cfg.d_model;
    int nh  = cfg.n_heads;
    int nkv = cfg.n_kv_heads;
    int hd  = cfg.head_dim > 0 ? cfg.head_dim : (d / nh);
    size_t es = dtype_size(compute_dtype_);

    auto align256 = [](size_t x) -> size_t { return (x + 255) & ~size_t(255); };
    char* ptr = static_cast<char*>(shared_workspace_);

    auto make = [&](int64_t cols, size_t aligned_sz) -> Tensor {
        int64_t shape[2] = {static_cast<int64_t>(max_tokens), cols};
        Tensor t(ptr, compute_dtype_, 2, shape, true);
        ptr += aligned_sz;
        return t;
    };

    q_        = make(nh * hd,  align256(static_cast<size_t>(max_tokens) * nh * hd * es));
    // K and V are contiguous (no alignment gap) to enable strided batched GEMM.
    // v_.data == k_.data + kv_raw exactly, so output_stride = kv_raw / es.
    {
        size_t kv_raw = static_cast<size_t>(max_tokens) * nkv * hd * es;
        int64_t kv_shape[2] = {static_cast<int64_t>(max_tokens), static_cast<int64_t>(nkv * hd)};
        k_ = Tensor(ptr, compute_dtype_, 2, kv_shape, true);
        v_ = Tensor(ptr + kv_raw, compute_dtype_, 2, kv_shape, true);
        ptr += align256(2 * kv_raw);
    }
    attn_out_ = make(nh * hd,  align256(static_cast<size_t>(max_tokens) * nh * hd * es));
    proj_out_ = make(d,        align256(static_cast<size_t>(max_tokens) * d * es));
}

void GraphExecutor::configure_ffn_workspace(int max_tokens) {
    const auto& cfg = model_->config();
    int d  = cfg.d_model;
    int ff = cfg.d_ff;
    size_t es = dtype_size(compute_dtype_);

    auto align256 = [](size_t x) -> size_t { return (x + 255) & ~size_t(255); };
    char* ptr = static_cast<char*>(shared_workspace_);

    auto make = [&](int64_t cols, size_t aligned_sz) -> Tensor {
        int64_t shape[2] = {static_cast<int64_t>(max_tokens), cols};
        Tensor t(ptr, compute_dtype_, 2, shape, true);
        ptr += aligned_sz;
        return t;
    };

    gate_out_   = make(ff, align256(static_cast<size_t>(max_tokens) * ff * es));
    up_out_     = make(ff, align256(static_cast<size_t>(max_tokens) * ff * es));
    swiglu_out_ = make(ff, align256(static_cast<size_t>(max_tokens) * ff * es));
    ffn_out_    = make(d,  align256(static_cast<size_t>(max_tokens) * d * es));
}

void GraphExecutor::configure_moe_workspace(int max_tokens) {
    const auto& cfg = model_->config();
    int d     = cfg.d_model;
    int ne    = cfg.n_experts;
    int top_k = cfg.n_experts_active;
    int eff   = max_expert_eff_;
    size_t es = dtype_size(compute_dtype_);
    int expanded = max_tokens * top_k;

    auto align256 = [](size_t x) -> size_t { return (x + 255) & ~size_t(255); };
    char* ptr = static_cast<char*>(shared_workspace_);

    // gate_logits: FP32
    {
        int64_t shape[2] = {static_cast<int64_t>(max_tokens), static_cast<int64_t>(ne)};
        moe_gate_logits_ = Tensor(ptr, DType::FP32, 2, shape, true);
        ptr += align256(static_cast<size_t>(max_tokens) * ne * sizeof(float));
    }

    auto make_moe = [&](Tensor& t, int64_t rows, int64_t cols, size_t aligned_sz, DType dt) {
        int64_t shape[2] = {rows, cols};
        t = Tensor(ptr, dt, 2, shape, true);
        ptr += aligned_sz;
    };

    make_moe(moe_gathered_,      expanded, d,   align256(static_cast<size_t>(expanded) * d * es),   compute_dtype_);
    make_moe(moe_expert_gate_,   expanded, eff, align256(static_cast<size_t>(expanded) * eff * es), compute_dtype_);
    make_moe(moe_expert_up_,     expanded, eff, align256(static_cast<size_t>(expanded) * eff * es), compute_dtype_);
    make_moe(moe_expert_swiglu_, expanded, eff, align256(static_cast<size_t>(expanded) * eff * es), compute_dtype_);
    make_moe(moe_expert_down_,   expanded, d,   align256(static_cast<size_t>(expanded) * d * es),   compute_dtype_);
    make_moe(moe_scatter_out_,   max_tokens, d, align256(static_cast<size_t>(max_tokens) * d * sizeof(float)), DType::FP32);
}

void GraphExecutor::configure_ssm_workspace(int max_tokens) {
    const auto& cfg = model_->config();
    int d = cfg.d_model;
    int inner = cfg.ssm_inner_size;
    int n_groups = cfg.ssm_group_count;
    int state_size = cfg.ssm_state_size;
    int n_heads = cfg.ssm_dt_rank;
    int conv_channels = inner + 2 * n_groups * state_size;
    int ssm_in_dim = inner + conv_channels + n_heads;
    size_t es = dtype_size(compute_dtype_);

    auto align256 = [](size_t x) -> size_t { return (x + 255) & ~size_t(255); };
    char* ptr = static_cast<char*>(shared_workspace_);

    auto make = [&](int64_t cols, size_t aligned_sz) -> Tensor {
        int64_t shape[2] = {static_cast<int64_t>(max_tokens), cols};
        Tensor t(ptr, compute_dtype_, 2, shape, true);
        ptr += aligned_sz;
        return t;
    };

    ssm_proj_buf_ = make(ssm_in_dim,    align256(static_cast<size_t>(max_tokens) * ssm_in_dim * es));
    ssm_xBC_buf_  = make(conv_channels,  align256(static_cast<size_t>(max_tokens) * conv_channels * es));
    ssm_y_buf_    = make(inner,          align256(static_cast<size_t>(max_tokens) * inner * es));
    ssm_z_buf_    = make(inner,          align256(static_cast<size_t>(max_tokens) * inner * es));
    ssm_out_buf_  = make(d,              align256(static_cast<size_t>(max_tokens) * d * es));
    ssm_dt_buf_   = make(n_heads,        align256(static_cast<size_t>(max_tokens) * n_heads * es));
}

bool GraphExecutor::resize_workspace(int new_max_tokens, cudaStream_t stream) {
    if (new_max_tokens == shared_workspace_max_tokens_ || new_max_tokens <= 0) return true;
    if (new_max_tokens > max_tokens_) new_max_tokens = max_tokens_;  // never exceed init-time max

    // Recompute shared sizes for the new token count
    int saved_max = max_tokens_;
    max_tokens_ = new_max_tokens;
    compute_shared_sizes(new_max_tokens);
    max_tokens_ = saved_max;

    size_t new_shared = std::max({attn_shared_size_, ffn_shared_size_,
                                  moe_shared_size_, ssm_shared_size_});
    if (new_shared == 0) return true;

    if (new_shared != shared_workspace_size_) {
        if (shared_workspace_) {
            cudaFreeAsync(shared_workspace_, stream);
        }
        cudaError_t err = cudaMallocAsync(&shared_workspace_, new_shared, stream);
        if (err != cudaSuccess) {
            IMP_LOG_ERROR("Failed to resize shared workspace to %zu bytes: %s",
                          new_shared, cudaGetErrorString(err));
            shared_workspace_ = nullptr;
            shared_workspace_size_ = 0;
            return false;
        }
        shared_workspace_size_ = new_shared;
    }
    shared_workspace_max_tokens_ = new_max_tokens;
    return true;
}

bool GraphExecutor::layer_has_attention(int layer) const {
    return model_->layer(layer).wq.data != nullptr;
}

bool GraphExecutor::layer_has_ssm(int layer) const {
    return model_->layer(layer).ssm_in.data != nullptr;
}

bool GraphExecutor::layer_has_moe(int layer) const {
    const auto& ly = model_->layer(layer);
    return ly.moe_gate.data != nullptr;
}

bool GraphExecutor::layer_has_dense_ffn(int layer) const {
    const auto& ly = model_->layer(layer);
    return ly.w_up.data != nullptr && ly.moe_gate.data == nullptr;
}

Tensor GraphExecutor::view_tokens(const Tensor& buf, int n_tokens) const {
    // buf is always [max_tokens_, cols] from allocate_buffers.
    // Return a [n_tokens, cols] view.
    return slice_rows(buf, n_tokens);
}

// ---------------------------------------------------------------------------
// KV cache write
// ---------------------------------------------------------------------------

void GraphExecutor::write_kv_cache(int layer, const InferenceState& state,
                                   cudaStream_t stream) {
    if (!state.kv_cache || !state.block_tables) return;

    // Map global layer index to KV cache layer index
    int kv_layer = layer;
    if (!kv_layer_map_.empty()) {
        kv_layer = kv_layer_map_[layer];
        if (kv_layer < 0) return;  // not an attention layer
    }

    KVCache* cache = state.kv_cache;
    int n        = state.n_tokens;
    int nkv      = cache->n_kv_heads();
    int hd       = cache->head_dim();
    int row_elems    = nkv * hd;
    int block_stride = kKVBlockSize * row_elems;

    int threads = std::min(row_elems, 256);
    int nblocks = n;   // one CUDA block per token

    bool use_fp8 = (cache->dtype() == DType::FP8_E4M3);

    if (use_fp8) {
        // FP8 E4M3 quantized KV cache write path
        float inv_scale = 1.0f;  // default; in production, scale comes from QuantConfig

        // K view: [n_tokens, nkv * hd]
        Tensor kv = view_tokens(k_, n);
#ifdef __CUDA_FP8_TYPES_EXIST__
        write_kv_cache_fp8_kernel<<<nblocks, threads, 0, stream>>>(
            static_cast<const half*>(kv.data),
            state.positions,
            state.block_tables,
            static_cast<__nv_fp8_e4m3*>(cache->k_ptr(kv_layer, 0)),
            inv_scale,
            block_stride, row_elems, kKVBlockSize, n,
            state.max_blocks_per_seq, state.n_sequences);
#else
        write_kv_cache_fp8_kernel<<<nblocks, threads, 0, stream>>>(
            static_cast<const half*>(kv.data),
            state.positions,
            state.block_tables,
            static_cast<uint8_t*>(cache->k_ptr(kv_layer, 0)),
            inv_scale,
            block_stride, row_elems, kKVBlockSize, n,
            state.max_blocks_per_seq, state.n_sequences);
#endif

        // V view
        Tensor vv = view_tokens(v_, n);
#ifdef __CUDA_FP8_TYPES_EXIST__
        write_kv_cache_fp8_kernel<<<nblocks, threads, 0, stream>>>(
            static_cast<const half*>(vv.data),
            state.positions,
            state.block_tables,
            static_cast<__nv_fp8_e4m3*>(cache->v_ptr(kv_layer, 0)),
            inv_scale,
            block_stride, row_elems, kKVBlockSize, n,
            state.max_blocks_per_seq, state.n_sequences);
#else
        write_kv_cache_fp8_kernel<<<nblocks, threads, 0, stream>>>(
            static_cast<const half*>(vv.data),
            state.positions,
            state.block_tables,
            static_cast<uint8_t*>(cache->v_ptr(kv_layer, 0)),
            inv_scale,
            block_stride, row_elems, kKVBlockSize, n,
            state.max_blocks_per_seq, state.n_sequences);
#endif
    } else {
        // Standard FP16 KV cache write path — fused K+V in single launch
        Tensor kv = view_tokens(k_, n);
        Tensor vv = view_tokens(v_, n);
        dim3 fused_grid(n, 2);  // blockIdx.y: 0=K, 1=V
        write_kv_cache_fused_kernel<<<fused_grid, threads, 0, stream>>>(
            static_cast<const half*>(kv.data),
            static_cast<const half*>(vv.data),
            state.positions,
            state.block_tables,
            static_cast<half*>(cache->k_ptr(kv_layer, 0)),
            static_cast<half*>(cache->v_ptr(kv_layer, 0)),
            block_stride, row_elems, kKVBlockSize, n,
            state.max_blocks_per_seq, state.n_sequences);
    }
}

// ---------------------------------------------------------------------------
// Forward pass diagnostics (IMP_DEBUG_FORWARD=1)
// ---------------------------------------------------------------------------

static bool debug_forward_enabled() {
    static const bool enabled = (std::getenv("IMP_DEBUG_FORWARD") != nullptr);
    return enabled;
}

// Print min/max/mean/L2norm of a GPU tensor (first row only for multi-row tensors).
// Syncs the stream — only call when IMP_DEBUG_FORWARD is active.
static void debug_tensor_stats(const char* name, const Tensor& t, cudaStream_t stream,
                                int row = 0, int max_rows = 1) {
    if (!debug_forward_enabled()) return;
    int cols = static_cast<int>(t.shape[t.ndim - 1]);
    int nrows = std::min(max_rows, static_cast<int>(t.shape[0]) - row);
    int n = cols * nrows;
    std::vector<float> host(n);

    if (t.dtype == DType::FP16) {
        std::vector<half> tmp(n);
        cudaMemcpyAsync(tmp.data(), static_cast<const half*>(t.data) + (int64_t)row * cols,
                         n * sizeof(half), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        for (int i = 0; i < n; i++) host[i] = __half2float(tmp[i]);
    } else if (t.dtype == DType::FP32) {
        cudaMemcpyAsync(host.data(), static_cast<const float*>(t.data) + (int64_t)row * cols,
                         n * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
    } else {
        fprintf(stderr, "[DEBUG_FWD] %s: unsupported dtype %d\n", name, (int)t.dtype);
        return;
    }

    float vmin = host[0], vmax = host[0], vsum = 0, vl2 = 0;
    int nan_count = 0, inf_count = 0;
    for (int i = 0; i < n; i++) {
        float v = host[i];
        if (std::isnan(v)) { nan_count++; continue; }
        if (std::isinf(v)) { inf_count++; continue; }
        if (v < vmin) vmin = v;
        if (v > vmax) vmax = v;
        vsum += v;
        vl2 += v * v;
    }
    float mean = vsum / std::max(n - nan_count - inf_count, 1);
    float l2 = std::sqrt(vl2);
    fprintf(stderr, "[DEBUG_FWD] %-30s  min=%+.6e  max=%+.6e  mean=%+.6e  L2=%.6e",
            name, vmin, vmax, mean, l2);
    if (nan_count > 0) fprintf(stderr, "  NaN=%d", nan_count);
    if (inf_count > 0) fprintf(stderr, "  Inf=%d", inf_count);
    fprintf(stderr, "\n");
}

// Print top-k logits with token IDs
static void debug_top_logits(const Tensor& logits, cudaStream_t stream, int topk = 10) {
    if (!debug_forward_enabled()) return;
    int vocab = static_cast<int>(logits.shape[logits.ndim - 1]);
    std::vector<float> host(vocab);

    if (logits.dtype == DType::FP32) {
        cudaMemcpyAsync(host.data(), logits.data, vocab * sizeof(float),
                         cudaMemcpyDeviceToHost, stream);
    } else if (logits.dtype == DType::FP16) {
        std::vector<half> tmp(vocab);
        cudaMemcpyAsync(tmp.data(), logits.data, vocab * sizeof(half),
                         cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        for (int i = 0; i < vocab; i++) host[i] = __half2float(tmp[i]);
    }
    cudaStreamSynchronize(stream);

    // Find top-k by partial sort
    std::vector<std::pair<float, int>> scored(vocab);
    for (int i = 0; i < vocab; i++) scored[i] = {host[i], i};
    std::partial_sort(scored.begin(), scored.begin() + std::min(topk, vocab),
                      scored.end(), [](auto& a, auto& b) { return a.first > b.first; });
    fprintf(stderr, "[DEBUG_FWD] Top-%d logits:\n", topk);
    for (int i = 0; i < std::min(topk, vocab); i++) {
        fprintf(stderr, "  [%2d] token_id=%6d  logit=%+.6f\n",
                i, scored[i].second, scored[i].first);
    }
}

// ---------------------------------------------------------------------------
// Attention sub-pass for one layer
// ---------------------------------------------------------------------------

void GraphExecutor::run_attention(int layer, const InferenceState& state,
                                  cudaStream_t stream) {
    // Configure shared workspace for attention phase
    configure_attn_workspace(shared_workspace_max_tokens_);

    const auto& cfg = model_->config();
    const auto& ly  = model_->layer(layer);
    int n   = state.n_tokens;
    int nh  = cfg.n_heads;
    int nkv = cfg.n_kv_heads;
    int hd  = cfg.head_dim > 0 ? cfg.head_dim : (cfg.d_model / nh);
    float eps = cfg.rms_norm_eps;


    // Sized views for this call (never mutates member tensors).
    Tensor h  = view_tokens(hidden_,   n);
    Tensor r  = view_tokens(residual_, n);
    Tensor no = view_tokens(norm_out_, n);
    Tensor qv = view_tokens(q_,        n);
    Tensor kk = view_tokens(k_,        n);
    Tensor vv = view_tokens(v_,        n);
    Tensor ao = view_tokens(attn_out_, n);
    Tensor po = view_tokens(proj_out_, n);

    // 1. Save residual for later add-back.
    //    Optimization: for decode (n=1) with dp4a, fuse residual into GEMV.
    //    For prefill (n>1) with FP16 cache, use cuBLAS beta=1 to fuse residual
    //    into the wo projection GEMM — no separate residual save/add/copy needed.
    //    For FP32 accumulator path: residual is kept in fp32_hidden_, skip FP16 copy.
    const bool has_post_attn_norm = (ly.post_attn_norm.data != nullptr);
    const bool using_fp32_accum = (fp32_accum_buf_ != nullptr && has_post_attn_norm);
    bool will_fuse_o_residual = (!has_post_attn_norm &&
                                  n == 1 && q8_1_buf_ != nullptr && d8_buf_ != nullptr &&
                                  h.dtype == DType::FP16 &&
                                  (ly.wo_qtype == GGMLQuantType::Q6_K || ly.wo_qtype == GGMLQuantType::Q8_0));
    bool will_fuse_o_beta1 = (!has_post_attn_norm && !will_fuse_o_residual && n > 1 &&
                               fp16_cache_.count(ly.wo.data));
    if (!will_fuse_o_residual && !will_fuse_o_beta1 && !using_fp32_accum) {
        cudaMemcpyAsync(r.data, h.data, h.nbytes(),
                        cudaMemcpyDeviceToDevice, stream);
    }

    // 3. QKV projections:  [n, d] @ W^T -> [n, proj_dim]
    //    For decode (n=1) with matching quant types: fused RMSNorm→Q8_1→QKV GEMV.
    //    This skips the intermediate norm_out FP16 buffer entirely.
    //    Otherwise falls back to separate RMSNorm + 3 dp4a/cuBLAS dispatches.
    {
        auto* q8 = static_cast<block_q8_1*>(q8_1_buf_);
        bool fused_qkv = (n == 1 && q8 != nullptr && d8_buf_ != nullptr &&
                          no.dtype == DType::FP16 &&
                          ly.wq_qtype == ly.wk_qtype && ly.wk_qtype == ly.wv_qtype &&
                          (ly.wq_qtype == GGMLQuantType::Q6_K ||
                           ly.wq_qtype == GGMLQuantType::Q8_0 ||
                           ly.wq_qtype == GGMLQuantType::Q4_0));
        if (fused_qkv) {
            // Fused: RMSNorm + Q8_1 quantization in one kernel (no norm_out write)
            int K = static_cast<int>(ly.wq.shape[1]);
            rmsnorm_quantize_q8_1(static_cast<const half*>(h.data),
                                    static_cast<const half*>(ly.attn_norm.data),
                                    q8, d8_buf_, nullptr /*skip norm_out*/,
                                    K, eps, stream, norm_w_off_);
            int q_rows = static_cast<int>(ly.wq.shape[0]);
            int k_rows = static_cast<int>(ly.wk.shape[0]);
            int v_rows = static_cast<int>(ly.wv.shape[0]);
            if (ly.wq_qtype == GGMLQuantType::Q6_K) {
                gemv_qkv_fused_q6k_q8_1(ly.wq.data, ly.wk.data, ly.wv.data,
                                          q8, d8_buf_,
                                          static_cast<half*>(qv.data),
                                          static_cast<half*>(kk.data),
                                          static_cast<half*>(vv.data),
                                          q_rows, k_rows, v_rows, K, stream);
            } else if (ly.wq_qtype == GGMLQuantType::Q4_0) {
                gemv_qkv_fused_q4_0_q8_1(ly.wq.data, ly.wk.data, ly.wv.data,
                                           q8, d8_buf_,
                                           static_cast<half*>(qv.data),
                                           static_cast<half*>(kk.data),
                                           static_cast<half*>(vv.data),
                                           q_rows, k_rows, v_rows, K, stream);
            } else {
                gemv_qkv_fused_q8_0_q8_1(ly.wq.data, ly.wk.data, ly.wv.data,
                                           q8, d8_buf_,
                                           static_cast<half*>(qv.data),
                                           static_cast<half*>(kk.data),
                                           static_cast<half*>(vv.data),
                                           q_rows, k_rows, v_rows, K, stream);
            }
        } else {
            // Separate RMSNorm + dispatch
            rmsnorm(h, ly.attn_norm, no, eps, stream, norm_w_off_);

            // Try fused K+V path: single strided batched GEMM for both projections
            auto fused_kv_it = fused_kv_cache_.find(layer);
            if (n > 1 && fused_kv_it != fused_kv_cache_.end()) {
                // Q: still separate (different output dim with GQA)
                gemm_dispatch(no, ly.wq, ly.wq_scales, ly.wq_qtype, qv,
                              dequant_scratch_, stream, q8, d8_buf_, &fp16_cache_);
                // K+V: one batched cuBLAS call
                gemm_kv_batched(no, fused_kv_it->second, kk, vv, stream);
            } else {
                gemm_dispatch(no, ly.wq, ly.wq_scales, ly.wq_qtype, qv, dequant_scratch_, stream, q8, d8_buf_, &fp16_cache_);
                gemm_dispatch(no, ly.wk, ly.wk_scales, ly.wk_qtype, kk, dequant_scratch_, stream, q8, d8_buf_, &fp16_cache_);
                gemm_dispatch(no, ly.wv, ly.wv_scales, ly.wv_qtype, vv, dequant_scratch_, stream, q8, d8_buf_, &fp16_cache_);
            }
        }

        // Apply Q/K/V biases if present (Qwen2)
        add_bias(qv, ly.q_bias, stream);
        add_bias(kk, ly.k_bias, stream);
        add_bias(vv, ly.v_bias, stream);
    }

    // Per-layer RoPE theta and sliding window (Gemma-3: alternating local/global layers)
    float layer_rope_theta = cfg.rope_theta;
    float layer_rope_freq_scale = cfg.rope_freq_scale;
    int layer_sliding_window = cfg.sliding_window;
    if (cfg.sliding_window_pattern > 0) {
        bool is_global = (layer % cfg.sliding_window_pattern) == (cfg.sliding_window_pattern - 1);
        if (is_global) {
            // Global layer: full attention, model-level rope_theta, with freq scaling
            layer_sliding_window = 0;
        } else {
            // Local layer: sliding window, local rope_theta, no freq scaling
            if (cfg.rope_local_theta > 0.0f)
                layer_rope_theta = cfg.rope_local_theta;
            layer_rope_freq_scale = 1.0f;  // no scaling for local layers
        }
    }

    // 4+5+6. QK-norm + RoPE: fused into single kernel for decode (n=1)
    //    For prefill or models without QK-norm, use separate kernels.
    //    For decode with FP16 cache: fuse K-RoPE into KV write (saves 1 launch).
    bool rope_k_deferred = false;  // true when K-RoPE will be fused into KV write
    {
        bool has_qk_norm = (ly.attn_q_norm.data != nullptr && ly.attn_k_norm.data != nullptr);
        // Determine if we can fuse K-RoPE into KV cache write
        bool can_fuse_rope_kv = (!state.is_prefill && n == 1 &&
                                  qv.dtype == DType::FP16 &&
                                  state.kv_cache &&
                                  state.kv_cache->dtype() != DType::FP8_E4M3);
        if (has_qk_norm && n == 1 && qv.dtype == DType::FP16) {
            // Fused: QK-norm + RoPE in one kernel launch (saves 2 launches)
            qknorm_rope_fused(static_cast<half*>(qv.data),
                               static_cast<half*>(kk.data),
                               static_cast<const half*>(ly.attn_q_norm.data),
                               static_cast<const half*>(ly.attn_k_norm.data),
                               nh, nkv, hd, eps,
                               state.positions,  // device pointer
                               layer_rope_theta, layer_rope_freq_scale,
                               cfg.rope_dim, cfg.rope_neox, stream, norm_w_off_);
        } else if (can_fuse_rope_kv && !has_qk_norm) {
            // Fused path: Q-only RoPE here, K-RoPE deferred to KV write
            const int effective_rope_dim = (cfg.rope_dim > 0) ? cfg.rope_dim : hd;
            const int pairs = effective_rope_dim / 2;
            const float inv_scaling = 1.0f / layer_rope_freq_scale;
            rope_q_only_fp16_kernel<<<dim3(1, nh), pairs, 0, stream>>>(
                static_cast<half*>(qv.data), state.positions,
                nh, hd, layer_rope_theta, inv_scaling, pairs, cfg.rope_neox);
            rope_k_deferred = true;
        } else {
            // Separate path: QK-norm (if present) + RoPE on both Q and K
            if (ly.attn_q_norm.data != nullptr) {
                int64_t q_flat[2] = {static_cast<int64_t>(n) * nh, static_cast<int64_t>(hd)};
                Tensor q_flat_view = qv.reshape(2, q_flat);
                rmsnorm(q_flat_view, ly.attn_q_norm, q_flat_view, eps, stream, norm_w_off_);
            }
            if (ly.attn_k_norm.data != nullptr) {
                int64_t k_flat[2] = {static_cast<int64_t>(n) * nkv, static_cast<int64_t>(hd)};
                Tensor k_flat_view = kk.reshape(2, k_flat);
                rmsnorm(k_flat_view, ly.attn_k_norm, k_flat_view, eps, stream, norm_w_off_);
            }
            int64_t q4r[4] = {1, n, nh,  hd};
            int64_t k4r[4] = {1, n, nkv, hd};
            Tensor q4r_t = qv.reshape(4, q4r);
            Tensor k4r_t = kk.reshape(4, k4r);
            rope_forward(q4r_t, k4r_t, state.positions, hd, layer_rope_theta, layer_rope_freq_scale,
                         cfg.rope_dim, cfg.rope_neox, stream);
        }
    }


    // 7. Attention
    float scale = 1.0f / std::sqrt(static_cast<float>(hd));

    if (state.is_prefill) {
        // cuBLAS batched-GEMM attention for prefill (when S-matrix workspace is available).
        // Use cuBLAS when sliding window doesn't actually restrict attention (n <= window).
        bool sliding_active = (layer_sliding_window > 0 && n > layer_sliding_window);
        if (attn_scores_buf_ && n <= static_cast<int>(attn_scores_.shape[1]) &&
            !sliding_active) {
            // S workspace view: [n_heads, n, n]
            int64_t s_shape[3] = {static_cast<int64_t>(nh),
                                  static_cast<int64_t>(n),
                                  static_cast<int64_t>(n)};
            Tensor s_view(attn_scores_buf_, DType::FP16, 3, s_shape, true);

            attention_cublas_prefill(qv, kk, vv, ao, s_view,
                                     nh, nkv, hd, scale, /*causal=*/true, stream);
        } else {
            // Fallback: WMMA / scalar flash attention
            int64_t q4s[4]  = {1, n, nh,  hd};
            int64_t kv4s[4] = {1, n, nkv, hd};
            int64_t o4s[4]  = {1, n, nh,  hd};

            Tensor q4  = qv.reshape(4, q4s);
            Tensor k4  = kk.reshape(4, kv4s);
            Tensor v4  = vv.reshape(4, kv4s);
            Tensor o4  = ao.reshape(4, o4s);

            attention_prefill_dispatch(q4, k4, v4, o4, scale, /*causal=*/true,
                                       layer_sliding_window, stream);
        }

        // Persist K, V into cache for later decode steps
        write_kv_cache(layer, state, stream);
    } else {
        // Decode: write new token's K/V to cache first
        if (rope_k_deferred) {
            // Fused: apply RoPE to K during KV cache write (saves 1 kernel launch)
            int kv_layer = layer;
            if (!kv_layer_map_.empty()) kv_layer = kv_layer_map_[layer];
            KVCache* cache = state.kv_cache;
            int row_elems    = nkv * hd;
            int block_stride = kKVBlockSize * row_elems;
            int threads = std::min(row_elems, 256);
            const int effective_rope_dim = (cfg.rope_dim > 0) ? cfg.rope_dim : hd;
            const int pairs = effective_rope_dim / 2;
            const float inv_scaling = 1.0f / layer_rope_freq_scale;
            Tensor kv_view = view_tokens(k_, n);
            Tensor vv_view = view_tokens(v_, n);
            dim3 fused_grid(n, 2);
            write_kv_cache_rope_fused_kernel<<<fused_grid, threads, 0, stream>>>(
                static_cast<const half*>(kv_view.data),
                static_cast<const half*>(vv_view.data),
                state.positions, state.block_tables,
                static_cast<half*>(cache->k_ptr(kv_layer, 0)),
                static_cast<half*>(cache->v_ptr(kv_layer, 0)),
                block_stride, row_elems, kKVBlockSize, n,
                state.max_blocks_per_seq, state.n_sequences,
                nkv, hd, layer_rope_theta, inv_scaling, pairs, cfg.rope_neox);
        } else {
            write_kv_cache(layer, state, stream);
        }

        // Paged attention: Q shape depends on batch size
        int n_seq = state.n_sequences;
        // For decode, n_tokens == n_sequences (one token per seq)
        int64_t qd[4] = {n_seq, 1, nh, hd};
        int64_t od[4] = {n_seq, 1, nh, hd};
        Tensor q4 = qv.reshape(4, qd);
        Tensor o4 = ao.reshape(4, od);

        KVCache* cache = state.kv_cache;
        int total_blk  = cache->total_blocks();
        DType cache_dtype = cache->dtype();
        int64_t cs[4]  = {static_cast<int64_t>(total_blk),
                          static_cast<int64_t>(kKVBlockSize),
                          static_cast<int64_t>(nkv),
                          static_cast<int64_t>(hd)};
        // Use mapped KV layer index for hybrid models (attention layers only)
        int kv_layer = layer;
        if (!kv_layer_map_.empty()) {
            kv_layer = kv_layer_map_[layer];
        }
        Tensor k_c(cache->k_ptr(kv_layer, 0), cache_dtype, 4, cs, true);
        Tensor v_c(cache->v_ptr(kv_layer, 0), cache_dtype, 4, cs, true);

        if (cache_dtype == DType::FP8_E4M3) {
            // FP8 paged attention with on-the-fly dequant
            float kv_scale = 1.0f;  // default; in production, from QuantConfig
            paged_attention_decode_fp8(q4, k_c, v_c, o4,
                                        state.block_tables, state.context_lens,
                                        kKVBlockSize, scale, kv_scale,
                                        state.max_context_len, layer_sliding_window,
                                        stream);
        } else {
            paged_attention_set_splitk_scratch(splitk_scratch_, splitk_scratch_size_);
            paged_attention_decode(q4, k_c, v_c, o4,
                                    state.block_tables, state.context_lens,
                                    kKVBlockSize, scale, state.max_context_len,
                                    layer_sliding_window, stream);
        }
    }


    // 8+9. O projection + residual connection.
    //    For decode (n=1) with dp4a: fuse residual add into GEMV, write directly
    //    to hidden buffer. When will_fuse_o_residual is set, we skipped the
    //    initial h→r memcpy and use h.data itself as the residual source.
    //    This is safe because h.data is only READ (never written) between the
    //    start of run_attention and this point.
    if (will_fuse_o_residual) {
        auto* q8 = static_cast<block_q8_1*>(q8_1_buf_);
        int K_o = static_cast<int>(ly.wo.shape[1]);
        int M_o = static_cast<int>(ly.wo.shape[0]);
        quantize_fp16_to_q8_1(static_cast<const half*>(ao.data), q8, d8_buf_, K_o, stream);
        // Use h.data as both residual source and output destination.
        // Safe: each warp reads residual[row] before writing y[row].
        const half* residual_ptr = static_cast<const half*>(h.data);
        if (ly.wo_qtype == GGMLQuantType::Q6_K) {
            gemv_q6k_q8_1_residual(ly.wo.data, q8, d8_buf_,
                                    static_cast<half*>(h.data), residual_ptr,
                                    M_o, K_o, stream);
        } else {
            gemv_q8_0_q8_1_residual(ly.wo.data, q8, d8_buf_,
                                      static_cast<half*>(h.data), residual_ptr,
                                      M_o, K_o, stream);
        }
    } else if (will_fuse_o_beta1) {
        // Fused: hidden = attn_out @ wo^T + hidden (cuBLAS beta=1).
        // Safe: hidden is only READ (never written) between attn_norm and here.
        const Tensor& wo_fp16 = fp16_cache_.at(ly.wo.data);
        gemm(ao, wo_fp16, h, 1.0f, 1.0f, stream);
    } else {
        // Fallback: separate O-projection + optional post-norm + residual add
        gemm_dispatch(ao, ly.wo, ly.wo_scales, ly.wo_qtype, po, dequant_scratch_, stream,
                      static_cast<block_q8_1*>(q8_1_buf_), d8_buf_, &fp16_cache_);
        if (has_post_attn_norm && using_fp32_accum) {
            // Post-attn norm → FP32 accumulation (no D2D copy needed)
            Tensor tmp = view_tokens(norm_out_, n);
            rmsnorm(po, ly.post_attn_norm, tmp, eps, stream, norm_w_off_);
            int64_t total = static_cast<int64_t>(n) * cfg.d_model;
            int threads = 256;
            int blocks = static_cast<int>((total + threads - 1) / threads);
            Tensor fp32_h = view_tokens(fp32_hidden_, n);
            fp32_accum_add_fp16_kernel<<<blocks, threads, 0, stream>>>(
                static_cast<float*>(fp32_h.data),
                static_cast<const half*>(tmp.data), total);
            fp32_to_fp16_rowscale_kernel<<<n, 256, 256 * sizeof(float), stream>>>(
                static_cast<const float*>(fp32_h.data),
                static_cast<half*>(h.data), n, cfg.d_model);
        } else if (has_post_attn_norm) {
            // Post-attn norm → residual add (norm directly to h, no copies)
            rmsnorm(po, ly.post_attn_norm, h, eps, stream, norm_w_off_);
            elementwise_add(h, r, stream);
        } else {
            // No post-norm: h = po + residual (fused add-store, no copy)
            elementwise_add_store(po, r, h, stream);
        }
    }

}

// ---------------------------------------------------------------------------
// FFN sub-pass for one layer
// ---------------------------------------------------------------------------

void GraphExecutor::run_ffn(int layer, cudaStream_t stream) {
    // Configure shared workspace for dense FFN phase
    configure_ffn_workspace(shared_workspace_max_tokens_);

    const auto& cfg = model_->config();
    const auto& ly  = model_->layer(layer);

    // cur_n_tokens_ is set by forward_logits before the layer loop.
    int n   = cur_n_tokens_;
    float eps = cfg.rms_norm_eps;

    Tensor h  = view_tokens(hidden_,     n);
    Tensor r  = view_tokens(residual_,   n);
    Tensor no = view_tokens(norm_out_,   n);
    Tensor go = view_tokens(gate_out_,   n);
    Tensor uo = view_tokens(up_out_,     n);
    Tensor so = view_tokens(swiglu_out_, n);
    Tensor fo = view_tokens(ffn_out_,    n);

    // 1. Save residual (skip if fused down-proj+residual will handle it).
    //    For FP32 accumulator path: residual is kept in fp32_hidden_, skip FP16 copy.
    const Tensor& ffn_norm_w = (ly.ffn_norm.data != nullptr) ? ly.ffn_norm : ly.attn_norm;
    const bool has_post_ffn_norm = (ly.post_ffn_norm.data != nullptr);
    const bool using_fp32_accum = (fp32_accum_buf_ != nullptr && has_post_ffn_norm);
    bool will_fuse_down_residual = (!has_post_ffn_norm &&
                                     n == 1 && q8_1_buf_ != nullptr && d8_buf_ != nullptr &&
                                     h.dtype == DType::FP16 &&
                                     (ly.w_down_qtype == GGMLQuantType::Q6_K ||
                                      ly.w_down_qtype == GGMLQuantType::Q8_0 ||
                                      ly.w_down_qtype == GGMLQuantType::Q4_0));
    bool will_fuse_down_beta1 = (!has_post_ffn_norm && !will_fuse_down_residual && n > 1 &&
                                  fp16_cache_.count(ly.w_down.data));
    bool will_fuse_down_dequant_beta1 = (!has_post_ffn_norm && !will_fuse_down_residual &&
                                          !will_fuse_down_beta1 && n > 1 &&
                                          dequant_scratch_ != nullptr &&
                                          dequant_gpu_supported(ly.w_down_qtype));
    if (!will_fuse_down_residual && !will_fuse_down_beta1 &&
        !will_fuse_down_dequant_beta1 && !using_fp32_accum) {
        cudaMemcpyAsync(r.data, h.data, h.nbytes(),
                        cudaMemcpyDeviceToDevice, stream);
    }

    // 3. Gate and Up projections
    //    For decode (n=1): fuse RMSNorm→Q8_1→GEMV to avoid redundant quantization.
    {
        auto* q8 = static_cast<block_q8_1*>(q8_1_buf_);
        int d = static_cast<int>(h.shape[1]);
        bool fused_ffn_norm = (n == 1 && q8 != nullptr && d8_buf_ != nullptr &&
                               h.dtype == DType::FP16 &&
                               (ly.w_gate_qtype == GGMLQuantType::Q6_K ||
                                ly.w_gate_qtype == GGMLQuantType::Q8_0 ||
                                ly.w_gate_qtype == GGMLQuantType::Q4_0));
        if (fused_ffn_norm) {
            // Fused RMSNorm + Q8_1: quantize once, use for both gate and up
            rmsnorm_quantize_q8_1(static_cast<const half*>(h.data),
                                    static_cast<const half*>(ffn_norm_w.data),
                                    q8, d8_buf_, static_cast<half*>(no.data),
                                    d, eps, stream, norm_w_off_);
            // Fused gate+up GEMV: single kernel launch for both projections
            int ffn_rows = static_cast<int>(ly.w_gate.shape[0]);
            gemv_gate_up_fused(ly.w_gate.data, ly.w_up.data, q8, d8_buf_,
                                static_cast<half*>(go.data),
                                static_cast<half*>(uo.data),
                                ffn_rows, d, ly.w_gate_qtype, stream);
        } else {
            rmsnorm(h, ffn_norm_w, no, eps, stream, norm_w_off_);
            auto fused_gu_it = fused_gate_up_cache_.find(layer);
            if (n > 1 && fused_gu_it != fused_gate_up_cache_.end()) {
                // Batched gate+up: single cuBLAS call for both projections
                gemm_pair_batched(no, fused_gu_it->second, go, uo, stream);
            } else {
                gemm_dispatch(no, ly.w_gate, ly.w_gate_scales, ly.w_gate_qtype, go, dequant_scratch_, stream, q8, d8_buf_, &fp16_cache_);
                gemm_dispatch(no, ly.w_up,   ly.w_up_scales,   ly.w_up_qtype,   uo, dequant_scratch_, stream, q8, d8_buf_, &fp16_cache_);
            }
        }
    }

    // 4+5+6. Gated activation + Down projection + residual add.
    //    For decode (n=1) with dp4a: fuse activation→Q8_1→GEMV+residual.
    //    SwiGLU case: swiglu_quantize_q8_1 fuses activation + Q8_1 in one kernel,
    //    eliminating the intermediate FP16 buffer write and one kernel launch.
    {
        auto* q8 = static_cast<block_q8_1*>(q8_1_buf_);
        bool fused_down_residual = (!has_post_ffn_norm &&
                                     n == 1 && q8 != nullptr && d8_buf_ != nullptr &&
                                     so.dtype == DType::FP16 &&
                                     (ly.w_down_qtype == GGMLQuantType::Q6_K ||
                                      ly.w_down_qtype == GGMLQuantType::Q8_0 ||
                                      ly.w_down_qtype == GGMLQuantType::Q4_0));
        if (fused_down_residual) {
            int K_d = static_cast<int>(ly.w_down.shape[1]);
            int M_d = static_cast<int>(ly.w_down.shape[0]);
            // Fuse activation + Q8_1 quantization into a single kernel when possible.
            // This saves 1 kernel launch per layer (activation + quantize → single kernel).
            if (cfg.ffn_activation != FFNActivation::GEGLU) {
                swiglu_quantize_q8_1(static_cast<const half*>(go.data),
                                     static_cast<const half*>(uo.data),
                                     q8, d8_buf_, K_d, stream);
            } else {
                geglu(go, uo, so, stream);
                quantize_fp16_to_q8_1(static_cast<const half*>(so.data), q8, d8_buf_, K_d, stream);
            }
            // Use h.data as residual source (memcpy was skipped)
            const half* residual_ptr = static_cast<const half*>(h.data);
            if (ly.w_down_qtype == GGMLQuantType::Q6_K) {
                gemv_q6k_q8_1_residual(ly.w_down.data, q8, d8_buf_,
                                        static_cast<half*>(h.data), residual_ptr,
                                        M_d, K_d, stream);
            } else if (ly.w_down_qtype == GGMLQuantType::Q4_0) {
                gemv_q4_0_q8_1_residual(ly.w_down.data, q8, d8_buf_,
                                          static_cast<half*>(h.data), residual_ptr,
                                          M_d, K_d, stream);
            } else {
                gemv_q8_0_q8_1_residual(ly.w_down.data, q8, d8_buf_,
                                          static_cast<half*>(h.data), residual_ptr,
                                          M_d, K_d, stream);
            }
        } else {
            // Non-dp4a paths: activation must produce FP16 intermediate in so.
            switch (cfg.ffn_activation) {
                case FFNActivation::GEGLU:  geglu(go, uo, so, stream);  break;
                default:                    swiglu(go, uo, so, stream);  break;
            }
            if (will_fuse_down_beta1) {
                // Fused: hidden = swiglu_out @ w_down^T + hidden (cuBLAS beta=1).
                const Tensor& wd_fp16 = fp16_cache_.at(ly.w_down.data);
                gemm(so, wd_fp16, h, 1.0f, 1.0f, stream);
            } else if (will_fuse_down_dequant_beta1) {
                // Dequant into scratch, then beta=1.0 GEMM directly into hidden (which holds residual)
                int rows = static_cast<int>(ly.w_down.shape[0]);
                int cols = static_cast<int>(ly.w_down.shape[1]);
                dequant_gpu(ly.w_down.data, dequant_scratch_, ly.w_down_qtype, rows, cols, stream);
                Tensor w_fp16(dequant_scratch_, DType::FP16, ly.w_down.ndim, ly.w_down.shape, true);
                gemm(so, w_fp16, h, 1.0f, 1.0f, stream);
            } else {
                gemm_dispatch(so, ly.w_down, ly.w_down_scales, ly.w_down_qtype, fo, dequant_scratch_, stream,
                              static_cast<block_q8_1*>(q8_1_buf_), d8_buf_, &fp16_cache_);
                if (has_post_ffn_norm && using_fp32_accum) {
                    // Post-FFN norm → FP32 accumulation (no D2D copy needed)
                    Tensor tmp = view_tokens(norm_out_, n);
                    rmsnorm(fo, ly.post_ffn_norm, tmp, eps, stream, norm_w_off_);
                    int64_t total = static_cast<int64_t>(n) * cfg.d_model;
                    int threads = 256;
                    int blocks = static_cast<int>((total + threads - 1) / threads);
                    Tensor fp32_h = view_tokens(fp32_hidden_, n);
                    fp32_accum_add_fp16_kernel<<<blocks, threads, 0, stream>>>(
                        static_cast<float*>(fp32_h.data),
                        static_cast<const half*>(tmp.data), total);
                    fp32_to_fp16_rowscale_kernel<<<n, 256, 256 * sizeof(float), stream>>>(
                        static_cast<const float*>(fp32_h.data),
                        static_cast<half*>(h.data), n, cfg.d_model);
                } else if (has_post_ffn_norm) {
                    // Post-FFN norm → residual add (norm directly to h, no copies)
                    rmsnorm(fo, ly.post_ffn_norm, h, eps, stream, norm_w_off_);
                    elementwise_add(h, r, stream);
                } else {
                    // No post-norm: h = fo + residual (fused add-store, no copy)
                    elementwise_add_store(fo, r, h, stream);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// MoE FFN sub-pass for one layer
// ---------------------------------------------------------------------------

void GraphExecutor::run_moe_ffn(int layer, cudaStream_t stream) {
    // Configure shared workspace for MoE phase
    configure_moe_workspace(shared_workspace_max_tokens_);

    const auto& cfg = model_->config();
    const auto& ly  = model_->layer(layer);

    int n       = cur_n_tokens_;
    int d       = cfg.d_model;
    int ne      = cfg.n_experts;
    int top_k   = cfg.n_experts_active;
    int eff     = max_expert_eff_;
    float eps   = cfg.rms_norm_eps;
    size_t es   = dtype_size(compute_dtype_);
    int expanded = n * top_k;

    Tensor h  = view_tokens(hidden_,   n);
    Tensor r  = view_tokens(residual_, n);
    Tensor no = view_tokens(norm_out_, n);
    bool residual_fused = false;  // set true if decode fast path fuses residual add

    // 1. Save residual (skip if decode fast path will handle it —
    //    h.data is never written before the final weighted_sum_residual).
    const Tensor& norm_w = (ly.ffn_norm.data != nullptr) ? ly.ffn_norm : ly.attn_norm;

    // Pre-check decode fast path (same logic as will_decode_fast below)
    GGMLQuantType up_qtype_pre = ly.expert_up_qtype;
    bool will_skip_residual_copy = (n == 1 &&
        ly.expert_up_packed.data != nullptr && moe_dequant_buf_ != nullptr &&
        compute_dtype_ == DType::FP16 &&
        ly.expert_up_packed.on_device &&
        (up_qtype_pre == GGMLQuantType::Q6_K || up_qtype_pre == GGMLQuantType::Q8_0) &&
        ly.w_up_shared.data == nullptr);  // must not have shared expert for full residual fusion

    if (!will_skip_residual_copy) {
        cudaMemcpyAsync(r.data, h.data, h.nbytes(),
                        cudaMemcpyDeviceToDevice, stream);
    }
    bool moe_fused_norm_q8 = (n == 1 && q8_1_buf_ != nullptr && d8_buf_ != nullptr &&
                               h.dtype == DType::FP16);
    if (moe_fused_norm_q8) {
        // Fused: RMSNorm + Q8_1 (also writes FP16 norm_out for gate logits)
        rmsnorm_quantize_q8_1(static_cast<const half*>(h.data),
                                static_cast<const half*>(norm_w.data),
                                static_cast<block_q8_1*>(q8_1_buf_), d8_buf_,
                                static_cast<half*>(no.data),
                                d, eps, stream, norm_w_off_);
    } else {
        rmsnorm(h, norm_w, no, eps, stream, norm_w_off_);
    }

    // 3. Gate logits + top-k routing
    //    For n=1 decode with FP16 weights and pre-allocated buffers: use fused
    //    kernel that computes gate GEMV + softmax/sigmoid + top-k in one launch.
    //    Otherwise: separate gate GEMV + topk gating kernels.
    const void* router_bias_ptr = ly.moe_router_bias.data;
    bool use_sigmoid = cfg.moe_sigmoid_gating;
    bool norm_weights = cfg.expert_weights_norm;

    GGMLQuantType up_qtype = ly.expert_up_qtype;
    bool will_decode_fast = (n == 1 &&
                             ly.expert_up_packed.data != nullptr && moe_dequant_buf_ != nullptr &&
                             compute_dtype_ == DType::FP16 &&
                             ly.expert_up_packed.on_device &&
                             (up_qtype == GGMLQuantType::Q6_K || up_qtype == GGMLQuantType::Q8_0));

    MoeRoutingResult routing;

    // Fused gate GEMV + topk is only beneficial when n_experts fits in the
    // number of warps (8). For high expert counts (e.g., 128 in Qwen3-Coder),
    // the separate gemv_gate_fp32 (128 parallel blocks) is much faster than
    // serializing 128/8=16 experts per warp in a single block.
    constexpr int kMaxFusedExperts = 8;
    if (ne <= kMaxFusedExperts &&
        n == 1 && compute_dtype_ == DType::FP16 && ly.moe_gate.dtype == DType::FP16 &&
        moe_routing_buffers_.pool && will_decode_fast) {
        // Fused: gate GEMV + softmax/sigmoid + top-k in one kernel (1 launch)
        moe_gate_topk_fused(static_cast<const half*>(ly.moe_gate.data),
                            static_cast<const half*>(no.data),
                            ne, d, top_k,
                            moe_routing_buffers_, routing, stream,
                            use_sigmoid, norm_weights, router_bias_ptr);
    } else {
        // Separate: gate GEMV → intermediate logits → topk gating
        Tensor gate_logits_f32 = slice_rows(moe_gate_logits_, n);

        if (n == 1 && compute_dtype_ == DType::FP16 && ly.moe_gate.dtype == DType::FP16) {
            gemv_gate_fp32(static_cast<const half*>(ly.moe_gate.data),
                           static_cast<const half*>(no.data),
                           static_cast<float*>(gate_logits_f32.data),
                           ne, d, stream);
        } else {
            int64_t gl_shape[2] = {static_cast<int64_t>(n), static_cast<int64_t>(ne)};
            Tensor gate_logits_tmp(moe_gathered_.data, compute_dtype_, 2, gl_shape, true);
            gemm(no, ly.moe_gate, gate_logits_tmp, 1.0f, 0.0f, stream);

            int64_t numel = static_cast<int64_t>(n) * ne;
            int threads = 256;
            int blocks = static_cast<int>((numel + threads - 1) / threads);
            if (compute_dtype_ == DType::FP16) {
                fp16_to_fp32_kernel<<<blocks, threads, 0, stream>>>(
                    static_cast<const half*>(gate_logits_tmp.data),
                    static_cast<float*>(gate_logits_f32.data),
                    numel);
            } else {
                cudaMemcpyAsync(gate_logits_f32.data, gate_logits_tmp.data,
                                static_cast<size_t>(numel) * sizeof(float),
                                cudaMemcpyDeviceToDevice, stream);
            }
        }

        if (moe_routing_buffers_.pool) {
            moe_topk_gating(gate_logits_f32, top_k, moe_routing_buffers_, routing, stream, use_sigmoid, norm_weights, router_bias_ptr, /*skip_sorting=*/will_decode_fast);
        } else {
            moe_topk_gating(gate_logits_f32, top_k, routing, stream, use_sigmoid, norm_weights, router_bias_ptr);
        }
    }

    // 4b. Expert weight scaling (Nemotron: scale = 2.5)
    if (cfg.expert_weights_scale != 1.0f) {
        int64_t n_weights = static_cast<int64_t>(n) * top_k;
        int threads_s = 256;
        int blocks_s = static_cast<int>((n_weights + threads_s - 1) / threads_s);
        scale_fp32_kernel<<<blocks_s, threads_s, 0, stream>>>(
            static_cast<float*>(routing.expert_weights.data),
            cfg.expert_weights_scale, n_weights);
    }

    // Build per-expert tensor views for grouped GEMM.
    // Two paths:
    // - Pre-dequanted: expert_w_gate[e] etc. are FP16 on GPU (legacy / unquantized packed)
    // - On-the-fly dequant: expert_*_packed is raw Q6_K/Q8_0/Q4_0 on GPU, dequant per GEMM
    bool use_packed_dequant = (ly.expert_up_packed.data != nullptr &&
                               moe_dequant_buf_ != nullptr);

    // Non-gated expert FFN detection: no gate weights (Nemotron uses SiLU(up(x)) instead of SwiGLU)
    // Note: can't use expert_w_gate.empty() because loader pre-allocates the vector for all layers.
    // Instead check if gate data is actually present (packed or first unpacked entry).
    bool non_gated_experts = (ly.expert_gate_packed.data == nullptr &&
                              (ly.expert_w_gate.empty() || ly.expert_w_gate[0].data == nullptr));

    // Validate expert_d_ff matches packed tensor shapes (critical for buffer offsets)
    if (use_packed_dequant) {
        int64_t ref_eff = non_gated_experts
            ? ly.expert_up_packed.shape[1]
            : ly.expert_gate_packed.shape[1];
        int64_t down_eff = ly.expert_down_packed.shape[2];
        if (ref_eff != eff || down_eff != eff) {
            IMP_LOG_ERROR("CRITICAL: expert_d_ff mismatch! config=%d, packed.shape=%ld, "
                         "down_packed.shape[2]=%ld. Using packed tensor shapes instead.",
                         eff, (long)ref_eff, (long)down_eff);
            eff = static_cast<int>(ref_eff);
        }
    }

    // =========================================================================
    // DECODE FAST PATH: n=1, device-resident packed experts, Q6_K or Q8_0.
    // Skips gather/scatter and D2H sync. All top_k experts dispatched in a
    // single kernel launch per projection. CUDA-graph capturable.
    // =========================================================================
    // decode_fast_path == will_decode_fast (computed earlier before routing).
    // will_decode_fast already checks packed data + dequant buf + FP16 + on_device + Q6K/Q8_0.
    bool decode_fast_path = will_decode_fast;

    if (decode_fast_path) {
        // Device pointers from routing result (no D2H copy needed)
        const int32_t* expert_indices = static_cast<const int32_t*>(routing.expert_indices.data);
        const float* expert_weights   = static_cast<const float*>(routing.expert_weights.data);

        // Compute expert stride (bytes between experts in packed tensor)
        auto expert_stride = [](const Tensor& packed, GGMLQuantType qtype) -> size_t {
            int64_t rows = packed.shape[1];
            int64_t cols = packed.shape[2];
            return static_cast<size_t>(rows) * ggml_quant_row_bytes(qtype, cols);
        };

        half* norm_ptr = static_cast<half*>(no.data);
        half* gate_buf = static_cast<half*>(moe_expert_gate_.data);   // [top_k, eff]
        half* up_buf   = static_cast<half*>(moe_expert_up_.data);     // [top_k, eff]
        half* act_buf  = static_cast<half*>(moe_expert_swiglu_.data); // [top_k, eff]
        half* down_buf = static_cast<half*>(moe_expert_down_.data);   // [top_k, d]

        // Use dp4a MMVQ path when Q8_1 buffers are available
        bool use_dp4a = (q8_1_buf_ != nullptr && d8_buf_ != nullptr);

        if (use_dp4a) {
            // Q8_1 may already be computed by the fused norm+quant above.
            // If not (e.g., prefill or non-FP16), quantize norm_out now.
            auto* q8 = static_cast<block_q8_1*>(q8_1_buf_);
            if (!moe_fused_norm_q8) {
                quantize_fp16_to_q8_1(norm_ptr, q8, d8_buf_, d, stream);
            }

            size_t up_stride_bytes = expert_stride(ly.expert_up_packed, up_qtype);

            // 5'+6'. Fused gate+up projection (single kernel launch)
            if (!non_gated_experts) {
                size_t gate_stride = expert_stride(ly.expert_gate_packed, ly.expert_gate_qtype);
                if (up_qtype == GGMLQuantType::Q6_K) {
                    gemv_q6k_q8_1_moe_gate_up_fused(
                        ly.expert_gate_packed.data, ly.expert_up_packed.data,
                        expert_indices, q8, d8_buf_, gate_buf, up_buf,
                        eff, d, gate_stride, up_stride_bytes,
                        /*q8_1_stride=*/0, /*d8_stride=*/0, top_k, stream);
                } else {
                    gemv_q8_0_q8_1_moe_gate_up_fused(
                        ly.expert_gate_packed.data, ly.expert_up_packed.data,
                        expert_indices, q8, d8_buf_, gate_buf, up_buf,
                        eff, d, gate_stride, up_stride_bytes,
                        /*q8_1_stride=*/0, /*d8_stride=*/0, top_k, stream);
                }
            } else {
                // Non-gated: up projection only
                auto moe_gemv_dp4a = (up_qtype == GGMLQuantType::Q6_K)
                    ? gemv_q6k_q8_1_moe_decode : gemv_q8_0_q8_1_moe_decode;
                moe_gemv_dp4a(ly.expert_up_packed.data, expert_indices,
                              q8, d8_buf_, up_buf,
                              eff, d, up_stride_bytes,
                              /*q8_1_stride=*/0, /*d8_stride=*/0, top_k, stream);
            }
        } else {
            // Fallback: FP16 dequant path
            size_t up_stride_bytes = expert_stride(ly.expert_up_packed, up_qtype);

            if (!non_gated_experts) {
                size_t gate_stride = expert_stride(ly.expert_gate_packed, ly.expert_gate_qtype);
                if (up_qtype == GGMLQuantType::Q6_K) {
                    gemv_q6k_moe_gate_up_fused(
                        ly.expert_gate_packed.data, ly.expert_up_packed.data,
                        expert_indices, norm_ptr, gate_buf, up_buf,
                        eff, d, gate_stride, up_stride_bytes,
                        /*x_stride=*/0, top_k, stream);
                } else {
                    gemv_q8_0_moe_gate_up_fused(
                        ly.expert_gate_packed.data, ly.expert_up_packed.data,
                        expert_indices, norm_ptr, gate_buf, up_buf,
                        eff, d, gate_stride, up_stride_bytes,
                        /*x_stride=*/0, top_k, stream);
                }
            } else {
                auto moe_gemv = (up_qtype == GGMLQuantType::Q6_K)
                    ? gemv_q6k_moe_decode : gemv_q8_0_moe_decode;
                moe_gemv(ly.expert_up_packed.data, expert_indices,
                         norm_ptr, up_buf,
                         eff, d, up_stride_bytes, /*x_stride=*/0, top_k, stream);
            }
        }

        // 7'+8'. Activation + down projection
        //
        // When dp4a is active and experts are gated (SwiGLU), fuse the activation
        // and Q8_1 quantization into a single kernel, eliminating the intermediate
        // FP16 act_buf write+read.
        if (use_dp4a) {
            auto* q8 = static_cast<block_q8_1*>(q8_1_buf_);
            int eff_q8_blocks = eff / 32;

            if (!non_gated_experts) {
                // Fused SwiGLU → Q8_1 (1 kernel instead of 2)
                swiglu_quantize_q8_1(gate_buf, up_buf, q8, d8_buf_,
                                      top_k * eff, stream);
            } else {
                // Non-gated (relu²): fused relu² + Q8_1 quantization (1 kernel)
                relu_sqr_quantize_q8_1(up_buf, q8, d8_buf_, top_k * eff, stream);
            }

            // Down projection with dp4a GEMV
            auto moe_gemv_dp4a = (up_qtype == GGMLQuantType::Q6_K)
                ? gemv_q6k_q8_1_moe_decode : gemv_q8_0_q8_1_moe_decode;
            size_t down_stride = expert_stride(ly.expert_down_packed, ly.expert_down_qtype);
            moe_gemv_dp4a(ly.expert_down_packed.data, expert_indices,
                          q8, d8_buf_, down_buf,
                          d, eff, down_stride,
                          /*q8_1_stride=*/eff_q8_blocks, /*d8_stride=*/eff_q8_blocks,
                          top_k, stream);
        } else {
            // Non-dp4a: separate activation + FP16 down GEMV
            int64_t act_shape[2] = {static_cast<int64_t>(top_k),
                                     static_cast<int64_t>(eff)};
            if (non_gated_experts) {
                // relu² in-place on up_buf, then use up_buf directly for down projection
                Tensor up_t(up_buf, compute_dtype_, 2, act_shape, true);
                relu_sqr_inplace(up_t, stream);
            } else {
                Tensor gate_t(gate_buf, compute_dtype_, 2, act_shape, true);
                Tensor up_t(up_buf, compute_dtype_, 2, act_shape, true);
                Tensor act_t(act_buf, compute_dtype_, 2, act_shape, true);
                swiglu(gate_t, up_t, act_t, stream);
            }
            auto moe_gemv = (up_qtype == GGMLQuantType::Q6_K)
                ? gemv_q6k_moe_decode : gemv_q8_0_moe_decode;
            size_t down_stride = expert_stride(ly.expert_down_packed, ly.expert_down_qtype);
            half* down_input = non_gated_experts ? up_buf : act_buf;
            moe_gemv(ly.expert_down_packed.data, expert_indices,
                     down_input, down_buf,
                     d, eff, down_stride, /*x_stride=*/eff, top_k, stream);
        }

        // 9'. Fused weighted sum + FP16 output (+ residual if no shared expert)
        {
            bool has_shared_expert = (ly.w_up_shared.data != nullptr);
            // Use h.data as residual source when memcpy was skipped
            const void* res_ptr = has_shared_expert ? nullptr :
                (will_skip_residual_copy ? h.data : r.data);
            moe_weighted_sum_residual(down_buf, expert_weights, res_ptr,
                                      h.data, d, top_k, stream);
            if (!has_shared_expert) residual_fused = true;
        }

        goto moe_after_experts;
    }

    // =========================================================================
    // GENERAL PATH: prefill or host-offloaded or non-Q6K/Q8_0 experts
    // =========================================================================

    // 5. Gather: reorder tokens by expert assignment
    //    norm_out [n, d_model] -> gathered [expanded, d_model]
    {
        int64_t gath_shape[2] = {static_cast<int64_t>(expanded),
                                  static_cast<int64_t>(d)};
        Tensor gathered(moe_gathered_.data, compute_dtype_, 2, gath_shape, true);
        moe_gather(no, routing, gathered, stream);
    }

    // =========================================================================
    // FP8 TENSOR CORE PREFILL PATH: Q6_K → FP8 dequant + cuBLAS FP8 GEMM.
    // 1.7x less DRAM traffic than FP16 path + 2x tensor core throughput.
    // Dequants Q6_K weights to FP8 E4M3 (half the FP16 size), quantizes
    // FP16 activations to FP8, then runs FP8×FP8→FP16 grouped GEMM.
    // =========================================================================
    {
    bool can_fp8_batch = (moe_batch_dequant_buf_ != nullptr &&
                          ly.expert_up_packed.data && ly.expert_up_packed.on_device &&
                          ly.expert_down_packed.data && ly.expert_down_packed.on_device &&
                          up_qtype == GGMLQuantType::Q6_K &&
                          ly.expert_down_qtype == GGMLQuantType::Q6_K &&
                          compute_dtype_ == DType::FP16 &&
                          !fp16_cache_.count(ly.expert_up_packed.data));
    if (can_fp8_batch && !non_gated_experts)
        can_fp8_batch = (ly.expert_gate_packed.data &&
                         ly.expert_gate_packed.on_device &&
                         ly.expert_gate_qtype == GGMLQuantType::Q6_K);

    if (can_fp8_batch) {
        // Expert offsets: device-grouped path uses d_offsets directly on GPU.
        // Host offsets + sync are deferred to the legacy fallback path only.
        char* buf = static_cast<char*>(moe_batch_dequant_buf_);

        // FP8 batched GEMM lambda: dequant Q6_K→FP8, quantize FP16 acts→FP8, cuBLAS FP8 GEMM→FP16
        auto chunked_fp8_gemm = [&](const Tensor& packed, GGMLQuantType qtype,
                                     const char* a_base_fp16, char* c_base_fp16,
                                     int K_dim, int N_dim) {
            int64_t rows = packed.shape[1];
            int64_t cols = packed.shape[2];
            size_t weight_fp8_bytes = static_cast<size_t>(ne) * rows * cols;  // 1 byte per FP8 element

            // Buffer layout in moe_batch_dequant_buf_:
            //   [0 .. weight_fp8_bytes)                     = FP8 weights for all experts
            //   [weight_fp8_bytes .. weight_fp8_bytes + act) = FP8 activations
            uint8_t* fp8_weights = reinterpret_cast<uint8_t*>(buf);
            uint8_t* fp8_acts = fp8_weights + weight_fp8_bytes;

            // 1. Dequant all experts Q6_K → FP8 E4M3
            dequant_gpu_fp8(packed.data, fp8_weights, qtype,
                            ne * static_cast<int>(rows), static_cast<int>(cols), stream);

            // 2. Per-expert FP8 scaling: calibrate absmax per expert, quantize with
            //    per-expert scale. Falls back to scale=1.0 if scale buffer unavailable.
            const int32_t* d_offsets = static_cast<const int32_t*>(routing.expert_offsets.data);
            if (d_moe_fp8_scales_) {
                // Calibrate per-expert: writes scales to d_moe_fp8_scales_
                calibrate_fp8_scales_per_expert(a_base_fp16, K_dim, d_offsets, ne,
                                                 d_moe_fp8_scales_, stream);
                // Quantize with per-expert scale
                quantize_fp16_to_fp8_e4m3_per_expert(a_base_fp16, fp8_acts,
                                                      K_dim, d_offsets, ne,
                                                      d_moe_fp8_scales_, stream);
                // Note: no D2H sync here — device-grouped path uses device-side
                // scales directly. Host scales are only needed by the fallback path.
            } else {
                // Fallback: uniform scale=1.0
                quantize_fp16_to_fp8_e4m3_scaled(a_base_fp16, fp8_acts,
                                                  expanded * K_dim, 1.0f, stream);
            }

            // 3. Build per-expert FP8 weight pointers and dispatch GEMM
            size_t expert_fp8_sz = static_cast<size_t>(rows) * cols;

#if IMP_CUDA_13_1
            // Device-grouped path: upload weight pointers to device, avoid host sync for GEMM.
            // Uses device-side FP8 scales — no D2H copy needed.
            if (d_moe_weight_ptrs_ && d_moe_weight_ptrs_count_ >= ne) {
                // Build host weight pointers, upload to device
                std::vector<const void*> weight_ptrs(ne);
                for (int e = 0; e < ne; ++e)
                    weight_ptrs[e] = fp8_weights + static_cast<size_t>(e) * expert_fp8_sz;
                cudaMemcpyAsync(d_moe_weight_ptrs_,
                                weight_ptrs.data(),
                                static_cast<size_t>(ne) * sizeof(void*),
                                cudaMemcpyHostToDevice, stream);

                // Upper bound on tokens per expert: at most n tokens can be
                // assigned to any single expert. Avoids D2H sync of offsets.
                int max_tpe = std::max(n, 1);

                gemm_moe_device_grouped(fp8_acts, c_base_fp16,
                                         d_offsets,
                                         reinterpret_cast<const void* const*>(d_moe_weight_ptrs_),
                                         K_dim, N_dim, DType::FP8_E4M3,
                                         ne, max_tpe, stream,
                                         d_moe_fp8_scales_ ? d_moe_fp8_scales_ : nullptr);
                return;
            }
#endif
            // 4. Fallback: host-pointer batched GEMM path — needs host offsets + scales
            std::vector<int32_t> h_offsets(ne + 1);
            cudaMemcpyAsync(h_offsets.data(), routing.expert_offsets.data,
                            static_cast<size_t>(ne + 1) * sizeof(int32_t),
                            cudaMemcpyDeviceToHost, stream);
            std::vector<float> h_act_scales(ne, 1.0f);
            if (d_moe_fp8_scales_) {
                cudaMemcpyAsync(h_act_scales.data(), d_moe_fp8_scales_,
                                static_cast<size_t>(ne) * sizeof(float),
                                cudaMemcpyDeviceToHost, stream);
            }
            cudaStreamSynchronize(stream);
            std::vector<const void*> weight_ptrs(ne);
            for (int e = 0; e < ne; ++e)
                weight_ptrs[e] = fp8_weights + static_cast<size_t>(e) * expert_fp8_sz;

            gemm_moe_batched(fp8_acts, c_base_fp16,
                             h_offsets.data(), weight_ptrs.data(),
                             K_dim, N_dim, DType::FP8_E4M3, ne, stream,
                             d_moe_work_ptrs_, /*output_dtype=*/DType::FP16,
                             d_moe_fp8_scales_ ? h_act_scales.data() : nullptr);
        };

        char* gathered_base     = static_cast<char*>(moe_gathered_.data);
        char* expert_gate_base  = static_cast<char*>(moe_expert_gate_.data);
        char* expert_up_base    = static_cast<char*>(moe_expert_up_.data);
        char* expert_swiglu_base= static_cast<char*>(moe_expert_swiglu_.data);
        char* expert_down_base  = static_cast<char*>(moe_expert_down_.data);

        // Gate projection (gated models only)
        if (!non_gated_experts)
            chunked_fp8_gemm(ly.expert_gate_packed, ly.expert_gate_qtype,
                             gathered_base, expert_gate_base, d, eff);

        // Up projection
        chunked_fp8_gemm(ly.expert_up_packed, up_qtype,
                         gathered_base, expert_up_base, d, eff);

        // Activation (FP16 — reuse existing kernels)
        {
            int64_t act_shape[2] = {static_cast<int64_t>(expanded), static_cast<int64_t>(eff)};
            if (non_gated_experts) {
                // relu² in-place on up buffer (no memcpy needed)
                Tensor up_t(moe_expert_up_.data, compute_dtype_, 2, act_shape, true);
                relu_sqr_inplace(up_t, stream);
            } else {
                Tensor g(moe_expert_gate_.data, compute_dtype_, 2, act_shape, true);
                Tensor u(moe_expert_up_.data, compute_dtype_, 2, act_shape, true);
                Tensor a(moe_expert_swiglu_.data, compute_dtype_, 2, act_shape, true);
                swiglu(g, u, a, stream);
            }
        }

        // Down projection: up buffer for non-gated (relu² in-place), swiglu for gated
        char* fp8_down_act = non_gated_experts ? expert_up_base : expert_swiglu_base;
        chunked_fp8_gemm(ly.expert_down_packed, ly.expert_down_qtype,
                         fp8_down_act, expert_down_base, eff, d);

        // Falls through to existing scatter (step 7)

    } else {
    // 6b. Per-expert FFN via grouped GEMM (non-fused paths)
    //
    //    Read expert_offsets from device to host to determine per-expert token
    //    counts. This is a small transfer (n_experts+1 ints).
    {
    std::vector<int32_t> h_offsets(ne + 1);
    cudaMemcpyAsync(h_offsets.data(), routing.expert_offsets.data,
                    static_cast<size_t>(ne + 1) * sizeof(int32_t),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // Helper: dequant one expert's weight from packed tensor into dequant scratch slot 0.
    // Returns a Tensor view into the scratch buffer with shape [rows, cols], FP16.
    // Uses slot 0 always -- safe because all ops are on the same stream, so the previous
    // GEMM reading from slot 0 completes before the next dequant writes to it.
    auto dequant_expert = [&](const Tensor& packed, GGMLQuantType qtype,
                              int expert_idx) -> Tensor {
        int64_t rows = packed.shape[1];
        int64_t cols = packed.shape[2];
        size_t row_bytes = ggml_quant_row_bytes(qtype, cols);
        size_t expert_raw = static_cast<size_t>(rows) * row_bytes;
        size_t total_raw = static_cast<size_t>(packed.shape[0]) * expert_raw;
        size_t offset = static_cast<size_t>(expert_idx) * expert_raw;

        // Bounds check: verify offset + expert_raw <= total allocated
        if (offset + expert_raw > total_raw) {
            IMP_LOG_ERROR("dequant_expert: OOB! expert %d offset=%zu + raw=%zu > total=%zu "
                    "(packed shape [%ld,%ld,%ld] qtype=%u)",
                    expert_idx, offset, expert_raw, total_raw,
                    (long)packed.shape[0], (long)packed.shape[1], (long)packed.shape[2],
                    (unsigned)qtype);
            return Tensor();
        }

        // Check dequant buffer is large enough
        size_t dequant_needed = static_cast<size_t>(rows) * cols * sizeof(uint16_t);
        if (dequant_needed > moe_dequant_buf_size_) {
            IMP_LOG_ERROR("dequant_expert: dequant buffer too small! "
                    "need=%zu have=%zu (rows=%ld cols=%ld)",
                    dequant_needed, moe_dequant_buf_size_, (long)rows, (long)cols);
            return Tensor();
        }

        const char* src;
        if (!packed.on_device) {
            // Expert weights offloaded to host — direct H2D to staging buffer.
            // If host data is pinned (cudaHostAlloc'd), this is true async DMA.
            // If unpinned (mmap'd fallback), CUDA runtime handles staging internally.
            const char* host_ptr = static_cast<const char*>(packed.data) + offset;
            cudaMemcpyAsync(moe_raw_staging_buf_, host_ptr, expert_raw,
                            cudaMemcpyHostToDevice, stream);
            src = static_cast<const char*>(moe_raw_staging_buf_);
        } else {
            src = static_cast<const char*>(packed.data) + offset;
        }

        char* dst = static_cast<char*>(moe_dequant_buf_);  // always slot 0

        dequant_gpu(src, dst, qtype, static_cast<int>(rows), static_cast<int>(cols), stream);

        int64_t shape[2] = {rows, cols};
        return Tensor(dst, DType::FP16, 2, shape, true);
    };

    // Helper: try fused quantized GEMV for count=1 decode (dequant+dot in one kernel),
    // else fall back to dequant_expert + cuBLAS gemm.
    // For host-resident experts: H2D to staging buffer, then fused GEMV on staging —
    // eliminates separate dequant_gpu + cuBLAS gemm overhead.
    auto expert_gemm = [&](const Tensor& a, Tensor& c,
                            const Tensor& packed, GGMLQuantType qtype,
                            const std::vector<Tensor>& fallback, int eidx) {
        if (a.shape[0] == 1 && use_packed_dequant &&
            compute_dtype_ == DType::FP16 &&
            (qtype == GGMLQuantType::Q6_K || qtype == GGMLQuantType::Q8_0)) {
            int64_t rows = packed.shape[1];
            int64_t cols = packed.shape[2];
            size_t rb = ggml_quant_row_bytes(qtype, cols);
            const void* w = nullptr;

            if (packed.on_device) {
                // On-device: point directly into packed tensor
                w = static_cast<const char*>(packed.data) +
                    (size_t)eidx * (size_t)rows * rb;
            } else if (moe_raw_staging_buf_) {
                // Host-resident: H2D to staging buffer, fused GEMV reads from there.
                // If host data is pinned (cudaHostAlloc'd), this is true async DMA.
                // If unpinned (mmap'd fallback), CUDA runtime handles staging internally.
                size_t expert_raw = (size_t)rows * rb;
                if (expert_raw <= moe_raw_staging_size_) {
                    size_t offset = (size_t)eidx * expert_raw;
                    const char* host_ptr = static_cast<const char*>(packed.data) + offset;
                    cudaMemcpyAsync(moe_raw_staging_buf_, host_ptr, expert_raw,
                                    cudaMemcpyHostToDevice, stream);
                    w = moe_raw_staging_buf_;
                }
            }

            if (w) {
                auto fn = (qtype == GGMLQuantType::Q6_K) ? gemv_q6k : gemv_q8_0;
                fn(w, static_cast<const half*>(a.data), static_cast<half*>(c.data),
                   static_cast<int>(rows), static_cast<int>(cols), stream);
                return;
            }
        }
        // Fallback: separate dequant + cuBLAS GEMM
        {
            Tensor b = use_packed_dequant
                ? dequant_expert(packed, qtype, eidx)
                : fallback[eidx];
            if (!b.data) return;  // dequant_expert failed (OOB or buffer too small)
            gemm(a, b, c, 1.0f, 0.0f, stream);
        }
    };

        char* gathered_base     = static_cast<char*>(moe_gathered_.data);
        char* expert_gate_base  = static_cast<char*>(moe_expert_gate_.data);
        char* expert_up_base    = static_cast<char*>(moe_expert_up_.data);
        char* expert_swiglu_base= static_cast<char*>(moe_expert_swiglu_.data);
        char* expert_down_base  = static_cast<char*>(moe_expert_down_.data);

        // Helper: get FP16 expert weight pointer from pre-dequant cache or unpacked weights.
        auto get_fp16_expert_ptr = [&](const Tensor& packed, GGMLQuantType qtype,
                                        const std::vector<Tensor>& fallback,
                                        int eidx) -> const void* {
            if (packed.data && fp16_cache_.count(packed.data)) {
                const Tensor& cached = fp16_cache_.at(packed.data);
                int64_t rows = packed.shape[1];
                int64_t cols = packed.shape[2];
                size_t expert_offset = static_cast<size_t>(eidx) * rows * cols * sizeof(half);
                return static_cast<const char*>(cached.data) + expert_offset;
            }
            if (!fallback.empty() && static_cast<size_t>(eidx) < fallback.size() &&
                fallback[eidx].data && fallback[eidx].dtype == DType::FP16 &&
                fallback[eidx].on_device) {
                return fallback[eidx].data;
            }
            return nullptr;
        };

        // Helper: batch dequant all experts + single grouped GEMM.
        // Dequants all experts to FP16, then runs a single batched GEMM.
        // CUTLASS 2.x GemmGrouped provides lower launch overhead than cuBLAS.
        auto chunked_dequant_gemm = [&](const Tensor& packed, GGMLQuantType qtype,
                                        const std::vector<Tensor>& fallback,
                                        const char* a_base, char* c_base,
                                        int K_dim, int N_dim) {
            int64_t rows = packed.shape[1];
            int64_t cols = packed.shape[2];
            size_t expert_fp16_sz = static_cast<size_t>(rows) * cols * sizeof(half);
            size_t expert_raw_sz = static_cast<size_t>(rows)
                                   * ggml_quant_row_bytes(qtype, cols);

            if (!moe_batch_dequant_buf_ || expert_fp16_sz == 0) {
                // No buffer — serial fallback
                for (int e = 0; e < ne; ++e) {
                    int start = h_offsets[e];
                    int count = h_offsets[e + 1] - start;
                    if (count == 0) continue;
                    int64_t count64 = static_cast<int64_t>(count);
                    int64_t a_shape[2] = {count64, static_cast<int64_t>(K_dim)};
                    Tensor a_view(const_cast<void*>(static_cast<const void*>(
                                  a_base + static_cast<size_t>(start) * K_dim * es)),
                                  compute_dtype_, 2, a_shape, true);
                    int64_t c_shape[2] = {count64, static_cast<int64_t>(N_dim)};
                    Tensor c_view(c_base + static_cast<size_t>(start) * N_dim * es,
                                  compute_dtype_, 2, c_shape, true);
                    expert_gemm(a_view, c_view, packed, qtype, fallback, e);
                }
                return;
            }

            const uint8_t* raw_base = static_cast<const uint8_t*>(packed.data);
            char* buf = static_cast<char*>(moe_batch_dequant_buf_);

            // Dequant all experts in one batch, then single GEMM.
            // With pp=512 and top_k=8, nearly all 128 experts are active, so
            // dequanting all at once is optimal (one big bandwidth-saturating kernel).
            dequant_gpu(raw_base, buf, qtype,
                        ne * static_cast<int>(rows), static_cast<int>(cols), stream);

            std::vector<const void*> b_ptrs(ne);
            for (int e = 0; e < ne; ++e)
                b_ptrs[e] = buf + static_cast<size_t>(e) * expert_fp16_sz;

#if IMP_CUDA_13_1
            // Device-grouped path: upload FP16 weight pointers, use device-side offsets
            if (d_moe_weight_ptrs_ && d_moe_weight_ptrs_count_ >= ne) {
                const int32_t* d_offsets_nd = static_cast<const int32_t*>(routing.expert_offsets.data);
                cudaMemcpyAsync(d_moe_weight_ptrs_,
                                b_ptrs.data(),
                                static_cast<size_t>(ne) * sizeof(void*),
                                cudaMemcpyHostToDevice, stream);
                int max_tpe = 0;
                for (int e = 0; e < ne; e++) {
                    int cnt = h_offsets[e + 1] - h_offsets[e];
                    if (cnt > max_tpe) max_tpe = cnt;
                }
                if (max_tpe == 0) max_tpe = 1;
                gemm_moe_device_grouped(a_base, c_base,
                                         d_offsets_nd,
                                         reinterpret_cast<const void* const*>(d_moe_weight_ptrs_),
                                         K_dim, N_dim, DType::FP16,
                                         ne, max_tpe, stream);
            } else
#endif
            {
#ifdef IMP_USE_CUTLASS
                gemm_moe_cutlass(a_base, c_base,
                                 h_offsets.data(), b_ptrs.data(),
                                 K_dim, N_dim, DType::FP16, ne, stream,
                                 d_moe_work_ptrs_);
#else
                gemm_moe_batched(a_base, c_base,
                                 h_offsets.data(), b_ptrs.data(),
                                 K_dim, N_dim, DType::FP16, ne, stream,
                                 d_moe_work_ptrs_);
#endif
            }
        };

        // Determine which path to use:
        // 1. Pre-cached FP16 path: all experts in fp16_cache_ (fastest, no dequant)
        // 2. Dequant-then-batch path: packed experts on device + batch buffer available
        // 3. Serial path: fallback (one expert at a time)
        // Note: fused Q6K dp4a path is handled above (before the D2H sync).

        bool has_precached_up = (ly.expert_up_packed.data && fp16_cache_.count(ly.expert_up_packed.data));
        bool can_dequant_batch = (moe_batch_dequant_buf_ != nullptr &&
                                   ly.expert_up_packed.data != nullptr &&
                                   ly.expert_up_packed.on_device &&
                                   dequant_gpu_supported(ly.expert_up_qtype));

        if (has_precached_up) {
            // Pre-cached FP16 path — all expert packs in fp16_cache_
            // ===== PRE-CACHED FP16 BATCHED GEMM PATH =====
            std::vector<const void*> gate_w_ptrs(ne, nullptr);
            std::vector<const void*> up_w_ptrs(ne, nullptr);
            std::vector<const void*> down_w_ptrs(ne, nullptr);

            for (int e = 0; e < ne; e++) {
                up_w_ptrs[e] = get_fp16_expert_ptr(ly.expert_up_packed, ly.expert_up_qtype,
                                                     ly.expert_w_up, e);
                if (!non_gated_experts)
                    gate_w_ptrs[e] = get_fp16_expert_ptr(ly.expert_gate_packed, ly.expert_gate_qtype,
                                                           ly.expert_w_gate, e);
                down_w_ptrs[e] = get_fp16_expert_ptr(ly.expert_down_packed, ly.expert_down_qtype,
                                                       ly.expert_w_down, e);
            }

            if (!non_gated_experts)
                gemm_moe_batched(gathered_base, expert_gate_base,
                                  h_offsets.data(), gate_w_ptrs.data(),
                                  d, eff, compute_dtype_, ne, stream, d_moe_work_ptrs_);
            gemm_moe_batched(gathered_base, expert_up_base,
                              h_offsets.data(), up_w_ptrs.data(),
                              d, eff, compute_dtype_, ne, stream, d_moe_work_ptrs_);

            {
                int64_t act_shape[2] = {static_cast<int64_t>(expanded), static_cast<int64_t>(eff)};
                if (non_gated_experts) {
                    // relu² in-place on up buffer (no memcpy needed)
                    Tensor up_t(moe_expert_up_.data, compute_dtype_, 2, act_shape, true);
                    relu_sqr_inplace(up_t, stream);
                } else {
                    Tensor g(moe_expert_gate_.data, compute_dtype_, 2, act_shape, true);
                    Tensor u(moe_expert_up_.data, compute_dtype_, 2, act_shape, true);
                    Tensor a(moe_expert_swiglu_.data, compute_dtype_, 2, act_shape, true);
                    swiglu(g, u, a, stream);
                }
            }

            {
                char* batch_down_act = non_gated_experts ? expert_up_base : expert_swiglu_base;
                gemm_moe_batched(batch_down_act, expert_down_base,
                                  h_offsets.data(), down_w_ptrs.data(),
                                  eff, d, compute_dtype_, ne, stream, d_moe_work_ptrs_);
            }

        } else if (can_dequant_batch) {
            // ===== BATCH DEQUANT + GROUPED GEMM =====
            // Dequant all experts to FP16, then single grouped GEMM via CUTLASS.

            if (!non_gated_experts)
                chunked_dequant_gemm(ly.expert_gate_packed, ly.expert_gate_qtype,
                                     ly.expert_w_gate, gathered_base, expert_gate_base, d, eff);
            chunked_dequant_gemm(ly.expert_up_packed, ly.expert_up_qtype,
                                 ly.expert_w_up, gathered_base, expert_up_base, d, eff);

            {
                int64_t act_shape[2] = {static_cast<int64_t>(expanded), static_cast<int64_t>(eff)};
                if (non_gated_experts) {
                    // relu² in-place on up buffer (no memcpy needed)
                    Tensor up_t(moe_expert_up_.data, compute_dtype_, 2, act_shape, true);
                    relu_sqr_inplace(up_t, stream);
                } else {
                    Tensor g(moe_expert_gate_.data, compute_dtype_, 2, act_shape, true);
                    Tensor u(moe_expert_up_.data, compute_dtype_, 2, act_shape, true);
                    Tensor a(moe_expert_swiglu_.data, compute_dtype_, 2, act_shape, true);
                    swiglu(g, u, a, stream);
                }
            }

            {
                char* dequant_down_act = non_gated_experts ? expert_up_base : expert_swiglu_base;
                chunked_dequant_gemm(ly.expert_down_packed, ly.expert_down_qtype,
                                     ly.expert_w_down, dequant_down_act, expert_down_base, eff, d);
            }

        } else {
            // ===== SERIAL PATH (fallback) =====
            for (int e = 0; e < ne; ++e) {
                int start = h_offsets[e];
                int count = h_offsets[e + 1] - start;
                if (count == 0) continue;

                int64_t count64 = static_cast<int64_t>(count);

                int64_t a_shape[2] = {count64, static_cast<int64_t>(d)};
                Tensor a_view(gathered_base + static_cast<size_t>(start) * d * es,
                              compute_dtype_, 2, a_shape, true);

                if (!non_gated_experts) {
                    int64_t c_shape[2] = {count64, static_cast<int64_t>(eff)};
                    Tensor c_view(expert_gate_base + static_cast<size_t>(start) * eff * es,
                                  compute_dtype_, 2, c_shape, true);
                    expert_gemm(a_view, c_view, ly.expert_gate_packed,
                                ly.expert_gate_qtype, ly.expert_w_gate, e);
                }

                {
                    int64_t c_shape[2] = {count64, static_cast<int64_t>(eff)};
                    Tensor c_view(expert_up_base + static_cast<size_t>(start) * eff * es,
                                  compute_dtype_, 2, c_shape, true);
                    expert_gemm(a_view, c_view, ly.expert_up_packed,
                                ly.expert_up_qtype, ly.expert_w_up, e);
                }
            }

            {
                int64_t act_shape[2] = {static_cast<int64_t>(expanded), static_cast<int64_t>(eff)};
                if (non_gated_experts) {
                    // relu² in-place on up buffer (no memcpy needed)
                    Tensor up_t(moe_expert_up_.data, compute_dtype_, 2, act_shape, true);
                    relu_sqr_inplace(up_t, stream);
                } else {
                    Tensor g(moe_expert_gate_.data, compute_dtype_, 2, act_shape, true);
                    Tensor u(moe_expert_up_.data, compute_dtype_, 2, act_shape, true);
                    Tensor a(moe_expert_swiglu_.data, compute_dtype_, 2, act_shape, true);
                    swiglu(g, u, a, stream);
                }
            }

            // Down projection activation source: up buffer for non-gated (relu² in-place),
            // swiglu buffer for gated.
            char* down_act_base = non_gated_experts ? expert_up_base : expert_swiglu_base;
            for (int e = 0; e < ne; ++e) {
                int start = h_offsets[e];
                int count = h_offsets[e + 1] - start;
                if (count == 0) continue;

                int64_t count64 = static_cast<int64_t>(count);

                int64_t a_shape[2] = {count64, static_cast<int64_t>(eff)};
                Tensor a_view(down_act_base + static_cast<size_t>(start) * eff * es,
                              compute_dtype_, 2, a_shape, true);
                int64_t c_shape[2] = {count64, static_cast<int64_t>(d)};
                Tensor c_view(expert_down_base + static_cast<size_t>(start) * d * es,
                              compute_dtype_, 2, c_shape, true);
                expert_gemm(a_view, c_view, ly.expert_down_packed,
                            ly.expert_down_qtype, ly.expert_w_down, e);
            }
        }
    }
    } // non-fused path scope (else branch of can_fp8_batch)
    } // FP8 prefill scope

    // 7. Scatter: weighted scatter-add expert outputs back to token positions.
    //    Output is FP32 (atomicAdd on floats). Must zero-init before atomic adds.
    {
        int64_t expert_out_shape[2] = {static_cast<int64_t>(expanded),
                                        static_cast<int64_t>(d)};
        Tensor expert_down_view(moe_expert_down_.data, compute_dtype_,
                                2, expert_out_shape, true);
        Tensor scatter_out = slice_rows(moe_scatter_out_, n);
        cudaMemsetAsync(scatter_out.data, 0,
                        static_cast<size_t>(n) * d * sizeof(float), stream);
        moe_scatter(expert_down_view, routing, scatter_out, stream);
    }

    // 8. Convert scatter output FP32 -> compute_dtype into hidden
    {
        int64_t numel = static_cast<int64_t>(n) * d;
        int threads = 256;
        int blocks = static_cast<int>((numel + threads - 1) / threads);
        if (compute_dtype_ == DType::FP16) {
            fp32_to_fp16_kernel<<<blocks, threads, 0, stream>>>(
                static_cast<const float*>(moe_scatter_out_.data),
                static_cast<half*>(h.data),
                numel);
        } else {
            // FP32 compute_dtype: just copy
            cudaMemcpyAsync(h.data, moe_scatter_out_.data,
                            static_cast<size_t>(numel) * sizeof(float),
                            cudaMemcpyDeviceToDevice, stream);
        }
    }

moe_after_experts:
    // 8b. Shared expert FFN: all tokens pass through an additional
    //     dense FFN whose output is added to the routed expert output.
    //     Reuses MoE workspace buffers (routed computation is complete).
    //     Supports both gated (Qwen3: gate+up+SwiGLU) and non-gated (Nemotron: up+SiLU).
    if (ly.w_up_shared.data != nullptr) {
        int eff_shared = static_cast<int>(ly.w_up_shared.shape[0]);
        bool shared_gated = (ly.w_gate_shared.data != nullptr);

        // Reuse moe_expert_gate_, moe_expert_up_, moe_expert_swiglu_ as scratch.
        int64_t sh_shape[2] = {static_cast<int64_t>(n), static_cast<int64_t>(eff_shared)};
        Tensor sh_up(moe_expert_up_.data, compute_dtype_, 2, sh_shape, true);
        Tensor sh_swiglu(moe_expert_swiglu_.data, compute_dtype_, 2, sh_shape, true);

        // Down projection output: [n, d_model]. Reuse moe_expert_down_.
        int64_t sh_down_shape[2] = {static_cast<int64_t>(n), static_cast<int64_t>(d)};
        Tensor sh_down(moe_expert_down_.data, compute_dtype_, 2, sh_down_shape, true);

        // Up projection (dp4a MMVQ for decode)
        {
            auto* q8 = static_cast<block_q8_1*>(q8_1_buf_);
            gemm_dispatch(no, ly.w_up_shared, Tensor(), ly.w_up_shared_qtype,
                          sh_up, dequant_scratch_, stream, q8, d8_buf_, &fp16_cache_);

            if (shared_gated) {
                // Gated: gate + SwiGLU
                Tensor sh_gate(moe_expert_gate_.data, compute_dtype_, 2, sh_shape, true);
                gemm_dispatch(no, ly.w_gate_shared, Tensor(), ly.w_gate_shared_qtype,
                              sh_gate, dequant_scratch_, stream, q8, d8_buf_, &fp16_cache_);
                swiglu(sh_gate, sh_up, sh_swiglu, stream);
            } else {
                // Non-gated: relu^2(up) in-place [Nemotron-H uses squared ReLU]
                relu_sqr_inplace(sh_up, stream);
            }

            // Down projection (reads from sh_up for non-gated since relu² was in-place)
            Tensor& sh_act = shared_gated ? sh_swiglu : sh_up;
            gemm_dispatch(sh_act, ly.w_down_shared, Tensor(), ly.w_down_shared_qtype,
                          sh_down, dequant_scratch_, stream, q8, d8_buf_, &fp16_cache_);
        }

        // Add shared expert output to hidden (which already has routed expert output)
        elementwise_add(h, sh_down, stream);
    }

    // 9. Residual connection: hidden += residual
    //    Skipped when decode fast path already fused residual into weighted_sum.
    if (!residual_fused) {
        elementwise_add(h, r, stream);
    }

    // 10. Free routing result tensors only if allocated by moe_topk_gating.
    //     When using pre-allocated buffers, memory belongs to moe_routing_buffers_.
    if (routing.owns_memory) {
        cudaFree(routing.expert_indices.data);
        cudaFree(routing.expert_weights.data);
        cudaFree(routing.sorted_token_ids.data);
        cudaFree(routing.expert_offsets.data);
    }
}

// ---------------------------------------------------------------------------
// SSM (Mamba2) sub-pass for one layer
// ---------------------------------------------------------------------------

void GraphExecutor::run_ssm(int layer, const InferenceState& state,
                            cudaStream_t stream) {
    // Configure shared workspace for SSM phase
    configure_ssm_workspace(shared_workspace_max_tokens_);

    const auto& cfg = model_->config();
    const auto& ly  = model_->layer(layer);
    int n = state.n_tokens;
    float eps = cfg.rms_norm_eps;
    int inner = cfg.ssm_inner_size;
    int n_groups = cfg.ssm_group_count;
    int ssize = cfg.ssm_state_size;
    int conv_kernel = cfg.ssm_conv_kernel;
    int conv_channels = inner + 2 * n_groups * ssize;
    int n_heads = cfg.ssm_dt_rank;
    int head_dim_ssm = inner / n_heads;

    Tensor h  = view_tokens(hidden_,   n);
    Tensor r  = view_tokens(residual_, n);
    Tensor no = view_tokens(norm_out_, n);

    // 1. Save residual + RMSNorm
    cudaMemcpyAsync(r.data, h.data, h.nbytes(),
                    cudaMemcpyDeviceToDevice, stream);
    rmsnorm(h, ly.attn_norm, no, eps, stream, norm_w_off_);

    // 2. ssm_in projection: [n, d_model] @ ssm_in^T -> [n, ssm_in_dim]
    //    ssm_in_dim = inner(z) + conv_channels(xBC) + n_heads(dt)
    Tensor proj = view_tokens(ssm_proj_buf_, n);
    gemm_dispatch(no, ly.ssm_in, Tensor(), ly.ssm_in_qtype, proj, dequant_scratch_, stream,
                  static_cast<block_q8_1*>(q8_1_buf_), d8_buf_, &fp16_cache_);

    // 3. Split projection output [n, total_dim] into z, xBC, dt by column slices.
    //    proj layout: each row has [z(inner) | xBC(conv_channels) | dt(n_heads)].
    size_t es = dtype_size(compute_dtype_);
    int total_dim = inner + conv_channels + n_heads;

    Tensor z_buf, xBC_in, dt_buf;
    bool views_into_proj = (n == 1);

    if (views_into_proj) {
        // Decode (n=1): create views directly into proj — no copies needed.
        // Conv1d output is redirected to ssm_xBC_buf_ to preserve z/dt views.
        int64_t z_shape[2] = {1, static_cast<int64_t>(inner)};
        z_buf = Tensor(proj.data, compute_dtype_, 2, z_shape, true);

        char* xbc_ptr = static_cast<char*>(proj.data) + static_cast<size_t>(inner) * es;
        int64_t xbc_shape[2] = {1, static_cast<int64_t>(conv_channels)};
        xBC_in = Tensor(xbc_ptr, compute_dtype_, 2, xbc_shape, true);

        char* dt_ptr = static_cast<char*>(proj.data) + static_cast<size_t>(inner + conv_channels) * es;
        int64_t dt_shape2[2] = {1, static_cast<int64_t>(n_heads)};
        dt_buf = Tensor(dt_ptr, compute_dtype_, 2, dt_shape2, true);
    } else {
        // Prefill (n>1): strided column extraction via cudaMemcpy2DAsync.
        size_t src_pitch = static_cast<size_t>(total_dim) * es;

        z_buf = view_tokens(ssm_z_buf_, n);
        cudaMemcpy2DAsync(z_buf.data, static_cast<size_t>(inner) * es,
                          proj.data, src_pitch,
                          static_cast<size_t>(inner) * es, n,
                          cudaMemcpyDeviceToDevice, stream);

        xBC_in = view_tokens(ssm_xBC_buf_, n);
        char* xBC_src = static_cast<char*>(proj.data) + static_cast<size_t>(inner) * es;
        cudaMemcpy2DAsync(xBC_in.data, static_cast<size_t>(conv_channels) * es,
                          xBC_src, src_pitch,
                          static_cast<size_t>(conv_channels) * es, n,
                          cudaMemcpyDeviceToDevice, stream);

        dt_buf = view_tokens(ssm_dt_buf_, n);
        char* dt_src = static_cast<char*>(proj.data) + static_cast<size_t>(inner + conv_channels) * es;
        cudaMemcpy2DAsync(dt_buf.data, static_cast<size_t>(n_heads) * es,
                          dt_src, src_pitch,
                          static_cast<size_t>(n_heads) * es, n,
                          cudaMemcpyDeviceToDevice, stream);
    }

    // 4. Conv1d on xBC
    //    For decode with views_into_proj: output to ssm_xBC_buf_ (preserves z/dt in proj).
    //    For prefill: output to ssm_proj_buf_ (proj data already copied out).
    int64_t conv_out_shape[2] = {static_cast<int64_t>(n), static_cast<int64_t>(conv_channels)};
    void* conv_out_ptr = views_into_proj ? ssm_xBC_buf_.data : ssm_proj_buf_.data;
    Tensor xBC_out(conv_out_ptr, compute_dtype_, 2, conv_out_shape, true);

    int ssm_idx = ssm_layer_map_[layer];
    void* conv_st = (state.ssm_state && ssm_idx >= 0)
                    ? state.ssm_state->conv_state(state.ssm_seq_id, ssm_idx)
                    : nullptr;

    if (conv_st) {
        if (state.is_prefill) {
            ssm_conv1d_prefill(conv_st, xBC_in, ly.ssm_conv1d_w, ly.ssm_conv1d_b,
                               xBC_out, conv_kernel, stream);
        } else {
            ssm_conv1d_decode(conv_st, xBC_in, ly.ssm_conv1d_w, ly.ssm_conv1d_b,
                              xBC_out, conv_kernel, stream);
        }
    }

    // 5. SiLU on full conv output (x, B, and C together).
    //    Mamba2 applies SiLU to the ENTIRE conv1d output, not just x.
    //    This matches causal_conv1d_fn(..., activation="silu").
    silu_inplace(xBC_out, stream);

    // 6-7. Split conv output into x/B/C per token, run SSM scan.
    int BC_size = n_groups * ssize;
    Tensor y_buf = view_tokens(ssm_y_buf_, n);

    void* h_st = (state.ssm_state && ssm_idx >= 0)
                 ? state.ssm_state->h_state(state.ssm_seq_id, ssm_idx)
                 : nullptr;

    if (h_st) {
        // xBC_out layout: [n, conv_channels] where each row = [x(inner) | B(BC_size) | C(BC_size)]
        // We need contiguous [n, inner], [n, BC_size], [n, BC_size] for the fused scan.
        // Extract x, B, C from interleaved xBC_out into contiguous y_buf (x), and
        // reuse ssm_xBC_buf_ for B and C. However, ssm_y_buf_ is the output so
        // we need separate buffers. Instead, use cudaMemcpy2DAsync to de-interleave.
        //
        // Actually, the stride within xBC_out is conv_channels per row, while
        // ssm_scan_kernel expects stride = inner_size per row for x.
        // For n=1 decode this is just pointer arithmetic (no copy needed).
        // For n>1 prefill, we must de-interleave.

        DType h_dtype = (state.ssm_state) ? state.ssm_state->h_dtype() : DType::FP32;

        if (n == 1) {
            // Decode: single token, just pass pointers directly into xBC_out row
            char* row = static_cast<char*>(xBC_out.data);
            int64_t x_shape[1] = {static_cast<int64_t>(inner)};
            Tensor x_t(row, compute_dtype_, 1, x_shape, true);

            int64_t bc_shape[1] = {static_cast<int64_t>(BC_size)};
            Tensor B_t(row + static_cast<size_t>(inner) * es, compute_dtype_, 1, bc_shape, true);
            Tensor C_t(row + static_cast<size_t>(inner + BC_size) * es, compute_dtype_, 1, bc_shape, true);

            int64_t dt_shape[1] = {static_cast<int64_t>(n_heads)};
            Tensor dt_t(dt_buf.data, compute_dtype_, 1, dt_shape, true);

            int64_t y_shape[1] = {static_cast<int64_t>(inner)};
            Tensor y_t(y_buf.data, compute_dtype_, 1, y_shape, true);

            ssm_scan_decode(x_t, B_t, C_t, dt_t,
                            ly.ssm_a, ly.ssm_d, ly.ssm_dt_b, h_st,
                            y_t, n_heads, head_dim_ssm, ssize, n_groups, h_dtype, stream);
        } else {
            // Prefill: de-interleave x, B, C from xBC_out [n, conv_channels]
            // into contiguous buffers, then single fused kernel launch.
            // Reuse ssm_y_buf_ tail for temporary B/C storage.
            // ssm_y_buf_ is [max_tokens, inner] — we need [n, BC_size] for B and C.
            // B/C total = n * BC_size * 2 * es. inner >= BC_size for typical configs,
            // so y_buf has enough space. Alternatively, use ssm_xBC_buf_ (already [n, conv_channels]).
            //
            // Strategy: extract x into y_buf (will be overwritten by scan output after),
            // extract B into xBC_in (reusable since conv1d is done),
            // extract C into second half of xBC_in.

            // x: extract [n, inner] from xBC_out with src_pitch=conv_channels*es
            char* x_contig = static_cast<char*>(y_buf.data);  // temp, overwritten by scan
            cudaMemcpy2DAsync(x_contig, static_cast<size_t>(inner) * es,
                              xBC_out.data, static_cast<size_t>(conv_channels) * es,
                              static_cast<size_t>(inner) * es, n,
                              cudaMemcpyDeviceToDevice, stream);

            // B: extract [n, BC_size] from offset inner in xBC_out
            char* B_contig = static_cast<char*>(xBC_in.data);  // conv1d done, safe to reuse
            char* B_src = static_cast<char*>(xBC_out.data) + static_cast<size_t>(inner) * es;
            cudaMemcpy2DAsync(B_contig, static_cast<size_t>(BC_size) * es,
                              B_src, static_cast<size_t>(conv_channels) * es,
                              static_cast<size_t>(BC_size) * es, n,
                              cudaMemcpyDeviceToDevice, stream);

            // C: extract [n, BC_size] from offset inner+BC_size in xBC_out
            char* C_contig = B_contig + static_cast<size_t>(n) * BC_size * es;
            char* C_src = static_cast<char*>(xBC_out.data) + static_cast<size_t>(inner + BC_size) * es;
            cudaMemcpy2DAsync(C_contig, static_cast<size_t>(BC_size) * es,
                              C_src, static_cast<size_t>(conv_channels) * es,
                              static_cast<size_t>(BC_size) * es, n,
                              cudaMemcpyDeviceToDevice, stream);

            // Build tensors for the fused scan
            int64_t x_shape[2] = {static_cast<int64_t>(n), static_cast<int64_t>(inner)};
            Tensor x_all(x_contig, compute_dtype_, 2, x_shape, true);

            int64_t bc_shape[2] = {static_cast<int64_t>(n), static_cast<int64_t>(BC_size)};
            Tensor B_all(B_contig, compute_dtype_, 2, bc_shape, true);
            Tensor C_all(C_contig, compute_dtype_, 2, bc_shape, true);

            int64_t dt_shape_all[2] = {static_cast<int64_t>(n), static_cast<int64_t>(n_heads)};
            Tensor dt_all(dt_buf.data, compute_dtype_, 2, dt_shape_all, true);

            // Output goes into y_buf (overwrites x_contig which was temporary)
            Tensor y_all(y_buf.data, compute_dtype_, 2, x_shape, true);

            ssm_scan_prefill(x_all, B_all, C_all, dt_all,
                             ly.ssm_a, ly.ssm_d, ly.ssm_dt_b, h_st,
                             y_all, n, n_heads, head_dim_ssm, ssize, n_groups, h_dtype, stream);
        }
    }

    // 8. Gating: y = y * SiLU(z)  [BEFORE GroupRMSNorm, per llama.cpp reference]
    silu_inplace(z_buf, stream);
    elementwise_mul(y_buf, z_buf, y_buf, stream);

    // 9. Group RMSNorm on y  [AFTER gating, per llama.cpp reference]
    group_rmsnorm(y_buf, ly.ssm_norm_w, y_buf, n_groups, eps, stream);

    // 10. ssm_out projection: [n, inner] @ ssm_out^T -> [n, d_model]
    Tensor out_buf = view_tokens(ssm_out_buf_, n);
    gemm_dispatch(y_buf, ly.ssm_out, Tensor(), ly.ssm_out_qtype, out_buf, dequant_scratch_, stream,
                  static_cast<block_q8_1*>(q8_1_buf_), d8_buf_, &fp16_cache_);

    // 11. Residual add: hidden = output + residual
    elementwise_add(out_buf, r, stream);
    cudaMemcpyAsync(h.data, out_buf.data, h.nbytes(),
                    cudaMemcpyDeviceToDevice, stream);

}

// ---------------------------------------------------------------------------
// Full forward pass
// ---------------------------------------------------------------------------

void GraphExecutor::forward_logits(const InferenceState& state,
                                   Tensor& logits_out,
                                   cudaStream_t stream) {
    if (!initialized_) {
        IMP_LOG_ERROR("GraphExecutor::forward_logits called before init()");
        return;
    }

    const auto& cfg = model_->config();
    int n = state.n_tokens;
    if (n <= 0) {
        IMP_LOG_ERROR("n_tokens must be positive, got %d", n);
        return;
    }
    if (n > max_tokens_) {
        IMP_LOG_ERROR("n_tokens (%d) exceeds max_tokens (%d)", n, max_tokens_);
        return;
    }

    // Store for use by run_ffn (which doesn't receive the InferenceState).
    cur_n_tokens_ = n;

    // Clear any stale CUDA error state before starting the forward pass.
    { cudaError_t e_ = cudaGetLastError();
      if (e_ != cudaSuccess) IMP_LOG_DEBUG("Cleared stale error before forward: %s", cudaGetErrorString(e_)); }

    // ---- Optional per-component profiling (IMP_PROFILE=1) ----
    // Profiling disables CUDA graph capture (they are incompatible).
    // Use IMP_PROFILE=1 for diagnostic runs only.
    static const bool do_profile = (std::getenv("IMP_PROFILE") != nullptr);
    static int profile_step_ = 0;
    static float acc_total = 0, acc_attn = 0, acc_ffn = 0, acc_lm = 0;
    bool profiling = do_profile;
    int profile_idx = profiling ? profile_step_++ : 0;
    // Skip first 2 decode steps (warmup / graph capture attempt)
    bool profile_active = profiling && (profile_idx >= 2);

    cudaEvent_t ev_start, ev_emb, ev_lm;
    std::vector<cudaEvent_t> ev_attn, ev_ffn;
    if (profile_active) {
        cudaEventCreate(&ev_start);
        cudaEventCreate(&ev_emb);
        cudaEventCreate(&ev_lm);
        ev_attn.resize(cfg.n_layers);
        ev_ffn.resize(cfg.n_layers);
        for (int i = 0; i < cfg.n_layers; i++) {
            cudaEventCreate(&ev_attn[i]);
            cudaEventCreate(&ev_ffn[i]);
        }
        cudaEventRecord(ev_start, stream);
    }

    // All member tensors are [max_tokens_, cols]. view_tokens creates [n, cols]
    // views on the fly without modifying the members.

    // ---- Step 1: Embedding lookup ----
    //    For Q8_0/Q6_K embeddings, dequantizes only the needed rows on the fly.
    if (debug_forward_enabled()) {
        std::vector<int32_t> h_ids(n);
        cudaMemcpy(h_ids.data(), state.token_ids, n * sizeof(int32_t), cudaMemcpyDeviceToHost);
        fprintf(stderr, "[DEBUG_FWD] input_tokens (%d):", n);
        for (int i = 0; i < n; i++) fprintf(stderr, " %d", h_ids[i]);
        fprintf(stderr, "\n");
    }
    Tensor h = view_tokens(hidden_, n);
    embedding_lookup(model_->token_embedding(), state.token_ids, n, h,
                     model_->tok_emb_qtype_, stream);

    // Gemma: scale embeddings by sqrt(d_model)
    if (cfg.embed_scale > 0.0f && h.dtype == DType::FP16) {
        int64_t total = static_cast<int64_t>(n) * cfg.d_model;
        int threads = 256;
        int blocks = static_cast<int>((total / 2 + threads - 1) / threads);
        scale_fp16_kernel<<<blocks, threads, 0, stream>>>(
            static_cast<half*>(h.data), __float2half(cfg.embed_scale), total);
    }

    debug_tensor_stats("after_embedding", h, stream);

    // Initialize FP32 residual accumulator from FP16 embedding (post-norm models only).
    if (fp32_accum_buf_) {
        int64_t total = static_cast<int64_t>(n) * cfg.d_model;
        int threads = 256;
        int blocks = static_cast<int>((total + threads - 1) / threads);
        fp16_to_fp32_kernel<<<blocks, threads, 0, stream>>>(
            static_cast<const half*>(h.data),
            static_cast<float*>(view_tokens(fp32_hidden_, n).data), total);
    }

    if (profile_active) cudaEventRecord(ev_emb, stream);

    // ---- Step 2: Transformer/Hybrid layers ----
    for (int i = 0; i < cfg.n_layers; ++i) {
        // Layer offloading: ensure weights are on GPU, prefetch next layer
        if (offload_mgr_) {
            offload_mgr_->ensure_layer(i, stream);
            if (i + 1 < cfg.n_layers) {
                offload_mgr_->prefetch_layer(i + 1);
            }
        }

        // Attention or SSM (mutually exclusive per layer)
        if (layer_has_attention(i)) {
            run_attention(i, state, stream);
        } else if (layer_has_ssm(i)) {
            run_ssm(i, state, stream);
        }
        if (i <= 1) {
            char buf[64];
            snprintf(buf, sizeof(buf), "after_layer%d_%s", i,
                     layer_has_attention(i) ? "attn" : "ssm");
            debug_tensor_stats(buf, h, stream);
        }
        if (profile_active) cudaEventRecord(ev_attn[i], stream);

        // FFN: MoE, dense, or none (attention-only layers may have no FFN)
        if (layer_has_moe(i)) {
            run_moe_ffn(i, stream);
        } else if (layer_has_dense_ffn(i)) {
            run_ffn(i, stream);
        }
        if (i <= 1) {
            char buf[64];
            snprintf(buf, sizeof(buf), "after_layer%d_%s", i,
                     layer_has_moe(i) ? "moe" : (layer_has_dense_ffn(i) ? "ffn" : "no_ffn"));
            debug_tensor_stats(buf, h, stream);
        }
        if (i == cfg.n_layers - 1) {
            debug_tensor_stats("after_last_layer", h, stream);
        }
        if (profile_active) cudaEventRecord(ev_ffn[i], stream);

        // Release offloaded layer (restore host pointers)
        if (offload_mgr_) {
            offload_mgr_->release_layer(i);
        }
    }

    // Final FP32→FP16 conversion for the tokens that need LM head projection.
    // run_attention/run_ffn already keep hidden_ in sync with fp32_hidden_,
    // but this ensures the final state is clean (no stale data from earlier layers).
    if (fp32_accum_buf_) {
        fp32_to_fp16_rowscale_kernel<<<n, 256, 256 * sizeof(float), stream>>>(
            static_cast<const float*>(view_tokens(fp32_hidden_, n).data),
            static_cast<half*>(h.data), n, cfg.d_model);
    }

    // ---- Step 3+4: Final RMSNorm + LM head projection ----
    // Only project the tokens that actually need sampling:
    //   Prefill: last token only (all others just populate KV cache)
    //   Decode:  all tokens (one per sequence)
    //
    // For raw Q6_K/Q8_0 output projection with single token (n=1 or prefill last):
    // use fused RMSNorm→Q8_1 + dp4a GEMV with FP32 output. Saves ~2.45x VRAM
    // bandwidth vs cuBLAS FP16 path (reads quantized weights directly).
    const auto out_qtype = model_->out_proj_qtype_;
    const bool use_dp4a_lm = q8_1_buf_ && compute_dtype_ == DType::FP16 &&
        (out_qtype == GGMLQuantType::Q6_K || out_qtype == GGMLQuantType::Q8_0 ||
         out_qtype == GGMLQuantType::Q4_0);

    if (state.is_prefill) {
        Tensor h_last = view_tokens(hidden_, n).slice(n - 1, n);
        Tensor lg = view_tokens(logits_, 1);

        if (use_dp4a_lm) {
            // For debug: compute norm_out separately so we can inspect it
            if (debug_forward_enabled()) {
                Tensor no_last = view_tokens(norm_out_, 1);
                rmsnorm(h_last, model_->output_norm(), no_last, cfg.rms_norm_eps, stream, norm_w_off_);
                debug_tensor_stats("after_final_rmsnorm", no_last, stream);
            }
            auto* q8 = static_cast<block_q8_1*>(q8_1_buf_);
            rmsnorm_quantize_q8_1(
                static_cast<const half*>(h_last.data),
                static_cast<const half*>(model_->output_norm().data),
                q8, d8_buf_, nullptr, cfg.d_model, cfg.rms_norm_eps, stream, norm_w_off_);
            if (out_qtype == GGMLQuantType::Q6_K)
                gemv_q6k_q8_1_fp32(model_->output_proj().data, q8, d8_buf_,
                                   static_cast<float*>(lg.data), cfg.vocab_size, cfg.d_model, stream);
            else if (out_qtype == GGMLQuantType::Q4_0)
                gemv_q4_0_q8_1_fp32(model_->output_proj().data, q8, d8_buf_,
                                    static_cast<float*>(lg.data), cfg.vocab_size, cfg.d_model, stream);
            else
                gemv_q8_0_q8_1_fp32(model_->output_proj().data, q8, d8_buf_,
                                    static_cast<float*>(lg.data), cfg.vocab_size, cfg.d_model, stream);
        } else {
            Tensor no_last = view_tokens(norm_out_, 1);
            rmsnorm(h_last, model_->output_norm(), no_last, cfg.rms_norm_eps, stream, norm_w_off_);
            debug_tensor_stats("after_final_rmsnorm", no_last, stream);
            gemm(no_last, model_->output_proj(), lg, 1.0f, 0.0f, stream);
        }
        logits_out = lg;
        debug_top_logits(lg, stream);
    } else {
        Tensor h_final = view_tokens(hidden_, n);
        Tensor lg = view_tokens(logits_, n);

        if (n == 1 && use_dp4a_lm) {
            if (debug_forward_enabled()) {
                Tensor no_final = view_tokens(norm_out_, 1);
                rmsnorm(h_final, model_->output_norm(), no_final, cfg.rms_norm_eps, stream, norm_w_off_);
                debug_tensor_stats("after_final_rmsnorm", no_final, stream);
            }
            auto* q8 = static_cast<block_q8_1*>(q8_1_buf_);
            rmsnorm_quantize_q8_1(
                static_cast<const half*>(h_final.data),
                static_cast<const half*>(model_->output_norm().data),
                q8, d8_buf_, nullptr, cfg.d_model, cfg.rms_norm_eps, stream, norm_w_off_);
            if (out_qtype == GGMLQuantType::Q6_K)
                gemv_q6k_q8_1_fp32(model_->output_proj().data, q8, d8_buf_,
                                   static_cast<float*>(lg.data), cfg.vocab_size, cfg.d_model, stream);
            else if (out_qtype == GGMLQuantType::Q4_0)
                gemv_q4_0_q8_1_fp32(model_->output_proj().data, q8, d8_buf_,
                                    static_cast<float*>(lg.data), cfg.vocab_size, cfg.d_model, stream);
            else
                gemv_q8_0_q8_1_fp32(model_->output_proj().data, q8, d8_buf_,
                                    static_cast<float*>(lg.data), cfg.vocab_size, cfg.d_model, stream);
        } else {
            Tensor no_final = view_tokens(norm_out_, n);
            rmsnorm(h_final, model_->output_norm(), no_final, cfg.rms_norm_eps, stream, norm_w_off_);
            debug_tensor_stats("after_final_rmsnorm", no_final, stream);
            gemm(no_final, model_->output_proj(), lg, 1.0f, 0.0f, stream);
        }
        logits_out = lg;
        debug_top_logits(lg, stream);
    }

    // ---- Profile summary ----
    if (profile_active) {
        cudaEventRecord(ev_lm, stream);
        cudaStreamSynchronize(stream);

        float t_emb = 0, t_lm = 0;
        float t_attn_total = 0, t_ffn_total = 0;
        cudaEventElapsedTime(&t_emb, ev_start, ev_emb);

        cudaEvent_t prev = ev_emb;
        for (int i = 0; i < cfg.n_layers; i++) {
            float t_attn = 0, t_ffn = 0;
            cudaEventElapsedTime(&t_attn, prev, ev_attn[i]);
            cudaEventElapsedTime(&t_ffn, ev_attn[i], ev_ffn[i]);
            t_attn_total += t_attn;
            t_ffn_total += t_ffn;
            prev = ev_ffn[i];
        }
        cudaEventElapsedTime(&t_lm, prev, ev_lm);

        float t_total = 0;
        cudaEventElapsedTime(&t_total, ev_start, ev_lm);
        acc_total += t_total;
        acc_attn += t_attn_total;
        acc_ffn += t_ffn_total;
        acc_lm += t_lm;

        int steps_profiled = profile_idx - 1;  // subtract warmup steps
        // Print every 32 steps
        if ((profile_idx & 31) == 0) {
            IMP_LOG_INFO("PROFILE avg over %d steps: total=%.2fms  attn=%.2fms (%.0f%%)  "
                         "ffn/moe=%.2fms (%.0f%%)  lm_head=%.2fms (%.0f%%)  "
                         "(per-layer: attn=%.3fms  ffn=%.3fms)",
                         steps_profiled,
                         acc_total / steps_profiled,
                         acc_attn / steps_profiled,
                         100.0f * acc_attn / acc_total,
                         acc_ffn / steps_profiled,
                         100.0f * acc_ffn / acc_total,
                         acc_lm / steps_profiled,
                         100.0f * acc_lm / acc_total,
                         acc_attn / steps_profiled / cfg.n_layers,
                         acc_ffn / steps_profiled / cfg.n_layers);
        }

        // Cleanup events
        cudaEventDestroy(ev_start);
        cudaEventDestroy(ev_emb);
        cudaEventDestroy(ev_lm);
        for (int i = 0; i < cfg.n_layers; i++) {
            cudaEventDestroy(ev_attn[i]);
            cudaEventDestroy(ev_ffn[i]);
        }
    }
}

int32_t GraphExecutor::forward(const InferenceState& state, cudaStream_t stream) {
    Tensor logits;
    forward_logits(state, logits, stream);

#ifdef IMP_DEBUG
    // Check for CUDA errors after the forward pass (debug only)
    {
        cudaError_t err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) {
            IMP_LOG_ERROR("CUDA error after forward: %s", cudaGetErrorString(err));
        }
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            IMP_LOG_ERROR("CUDA last error: %s", cudaGetErrorString(err));
        }
    }
#endif

    // Sample from the last token's logits.
    // forward_logits returns [1, V] for prefill, [n, V] for decode.
    // For single-token forward, always use row 0.
    Tensor last_logits = logits.slice(0, 1);
    int64_t vocab_shape[1] = {last_logits.shape[1]};
    last_logits = last_logits.reshape(1, vocab_shape);

    int32_t token;
    if (state.temperature <= 0.0f || state.top_k == 1) {
        token = d_sample_result_
            ? sample_greedy(last_logits, d_sample_result_, stream)
            : sample_greedy(last_logits, stream);
    } else {
        int top_k  = state.top_k > 0  ? state.top_k  : 50;
        float top_p = state.top_p > 0.0f ? state.top_p : 1.0f;
        unsigned int seed = state.seed >= 0
                                ? static_cast<unsigned int>(state.seed)
                                : 42u;
        token = d_sample_result_
            ? sample_topk_topp(last_logits, top_k, top_p,
                               state.temperature, seed, d_sample_result_, stream)
            : sample_topk_topp(last_logits, top_k, top_p,
                               state.temperature, seed, stream);
    }

    return token;
}

std::vector<int32_t> GraphExecutor::sample_from_logits(const Tensor& logits,
                                                        const InferenceState& state,
                                                        cudaStream_t stream) {
    int n_seq = state.n_sequences;
    std::vector<int32_t> tokens(n_seq);

    // Helper: flatten [1, V] to [V] for sampling
    auto flatten_logits = [](Tensor t) -> Tensor {
        int64_t vocab_shape[1] = {t.shape[t.ndim - 1]};
        return t.reshape(1, vocab_shape);
    };

    if (state.is_prefill || n_seq <= 1) {
        // Single sequence or prefill: logits is [1, V] (forward_logits already sliced)
        Tensor last_logits = flatten_logits(logits.slice(0, 1));

        tokens[0] = (state.temperature <= 0.0f || state.top_k == 1)
            ? (d_sample_result_ ? sample_greedy(last_logits, d_sample_result_, stream)
                                : sample_greedy(last_logits, stream))
            : (d_sample_result_
                ? sample_topk_topp(last_logits,
                                   state.top_k > 0 ? state.top_k : 50,
                                   state.top_p > 0.0f ? state.top_p : 1.0f,
                                   state.temperature,
                                   state.seed >= 0 ? static_cast<unsigned int>(state.seed) : 42u,
                                   d_sample_result_, stream)
                : sample_topk_topp(last_logits,
                                   state.top_k > 0 ? state.top_k : 50,
                                   state.top_p > 0.0f ? state.top_p : 1.0f,
                                   state.temperature,
                                   state.seed >= 0 ? static_cast<unsigned int>(state.seed) : 42u,
                                   stream));
    } else {
        // Batched decode: n_tokens == n_sequences, each row is one sequence's logits
        for (int i = 0; i < n_seq; i++) {
            Tensor seq_logits = flatten_logits(logits.slice(i, i + 1));
            tokens[i] = (state.temperature <= 0.0f || state.top_k == 1)
                ? (d_sample_result_ ? sample_greedy(seq_logits, d_sample_result_, stream)
                                    : sample_greedy(seq_logits, stream))
                : (d_sample_result_
                    ? sample_topk_topp(seq_logits,
                                       state.top_k > 0 ? state.top_k : 50,
                                       state.top_p > 0.0f ? state.top_p : 1.0f,
                                       state.temperature,
                                       state.seed >= 0 ? static_cast<unsigned int>(state.seed + i) : (42u + i),
                                       d_sample_result_, stream)
                    : sample_topk_topp(seq_logits,
                                       state.top_k > 0 ? state.top_k : 50,
                                       state.top_p > 0.0f ? state.top_p : 1.0f,
                                       state.temperature,
                                       state.seed >= 0 ? static_cast<unsigned int>(state.seed + i) : (42u + i),
                                       stream));
        }
    }

    return tokens;
}

std::vector<int32_t> GraphExecutor::forward_batch(const InferenceState& state,
                                                  cudaStream_t stream) {
    Tensor logits;
    forward_logits(state, logits, stream);
    return sample_from_logits(logits, state, stream);
}

// ---------------------------------------------------------------------------
// Async decode: embedding from device token → forward → sample to device
// ---------------------------------------------------------------------------

void GraphExecutor::forward_decode_async(const InferenceState& state,
                                          int32_t* d_token_id, int32_t* h_mapped,
                                          cudaStream_t stream) {
    if (!initialized_) {
        IMP_LOG_ERROR("GraphExecutor::forward_decode_async called before init()");
        return;
    }

    const auto& cfg = model_->config();
    int n = state.n_tokens;  // should be 1 for decode
    cur_n_tokens_ = n;
    // Clear any stale CUDA error state before starting the decode pass.
    { cudaError_t e_ = cudaGetLastError();
      if (e_ != cudaSuccess) IMP_LOG_DEBUG("Cleared stale error before decode: %s", cudaGetErrorString(e_)); }

    // ---- Step 1: Embedding lookup from device-side token ID ----
    Tensor h = view_tokens(hidden_, n);
    embedding_lookup_from_device(model_->token_embedding(), d_token_id, h,
                                  model_->tok_emb_qtype_, stream);

    // Gemma: scale embeddings by sqrt(d_model)
    if (cfg.embed_scale > 0.0f && h.dtype == DType::FP16) {
        int64_t total = static_cast<int64_t>(n) * cfg.d_model;
        int threads = 256;
        int blocks = static_cast<int>((total / 2 + threads - 1) / threads);
        scale_fp16_kernel<<<blocks, threads, 0, stream>>>(
            static_cast<half*>(h.data), __float2half(cfg.embed_scale), total);
    }

    // Initialize FP32 residual accumulator from FP16 embedding (post-norm models).
    // Without this, the graph loop would accumulate on stale FP32 state from
    // the previous iteration instead of starting fresh from the embedding.
    if (fp32_accum_buf_) {
        int64_t total = static_cast<int64_t>(n) * cfg.d_model;
        int threads = 256;
        int blocks = static_cast<int>((total + threads - 1) / threads);
        fp16_to_fp32_kernel<<<blocks, threads, 0, stream>>>(
            static_cast<const half*>(h.data),
            static_cast<float*>(view_tokens(fp32_hidden_, n).data), total);
    }

    // ---- Step 2: Transformer layers ----
    for (int i = 0; i < cfg.n_layers; ++i) {
        if (offload_mgr_) {
            offload_mgr_->ensure_layer(i, stream);
            if (i + 1 < cfg.n_layers) offload_mgr_->prefetch_layer(i + 1);
        }

        if (layer_has_attention(i)) run_attention(i, state, stream);
        else if (layer_has_ssm(i))  run_ssm(i, state, stream);

        if (layer_has_moe(i))            run_moe_ffn(i, stream);
        else if (layer_has_dense_ffn(i)) run_ffn(i, stream);

        if (offload_mgr_) offload_mgr_->release_layer(i);
    }

    // ---- Step 3: Final RMSNorm + LM head ----
    Tensor h_final = view_tokens(hidden_, n);
    Tensor lg = view_tokens(logits_, n);

    const auto out_qtype = model_->out_proj_qtype_;
    if (q8_1_buf_ && compute_dtype_ == DType::FP16 &&
        (out_qtype == GGMLQuantType::Q6_K || out_qtype == GGMLQuantType::Q8_0 ||
         out_qtype == GGMLQuantType::Q4_0)) {
        auto* q8 = static_cast<block_q8_1*>(q8_1_buf_);
        rmsnorm_quantize_q8_1(
            static_cast<const half*>(h_final.data),
            static_cast<const half*>(model_->output_norm().data),
            q8, d8_buf_, nullptr, cfg.d_model, cfg.rms_norm_eps, stream, norm_w_off_);
        if (out_qtype == GGMLQuantType::Q6_K)
            gemv_q6k_q8_1_fp32(model_->output_proj().data, q8, d8_buf_,
                               static_cast<float*>(lg.data), cfg.vocab_size, cfg.d_model, stream);
        else if (out_qtype == GGMLQuantType::Q8_0)
            gemv_q8_0_q8_1_fp32(model_->output_proj().data, q8, d8_buf_,
                                static_cast<float*>(lg.data), cfg.vocab_size, cfg.d_model, stream);
        else
            gemv_q4_0_q8_1_fp32(model_->output_proj().data, q8, d8_buf_,
                                static_cast<float*>(lg.data), cfg.vocab_size, cfg.d_model, stream);
    } else {
        Tensor no_final = view_tokens(norm_out_, n);
        rmsnorm(h_final, model_->output_norm(), no_final, cfg.rms_norm_eps, stream, norm_w_off_);
        gemm(no_final, model_->output_proj(), lg, 1.0f, 0.0f, stream);
    }

    // ---- Step 4: Async sampling → write to d_token_id + h_mapped ----
    Tensor last_logits = lg.slice(0, 1);
    int64_t vocab_shape[1] = {last_logits.shape[1]};
    last_logits = last_logits.reshape(1, vocab_shape);

    if (state.temperature <= 0.0f || state.top_k == 1) {
        sample_greedy_device(last_logits, d_token_id, h_mapped, stream);
    } else {
        int top_k  = state.top_k > 0  ? state.top_k  : 50;
        float top_p = state.top_p > 0.0f ? state.top_p : 1.0f;
        unsigned int seed = state.seed >= 0
                                ? static_cast<unsigned int>(state.seed)
                                : 42u;
        sample_topk_topp_device(last_logits, top_k, top_p,
                                 state.temperature, seed,
                                 d_token_id, h_mapped, stream);
    }
    // No cudaStreamSynchronize — host polls h_mapped asynchronously
}

} // namespace imp
