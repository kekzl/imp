#include "compute/gemm.h"
#include "compute/gemv_dp4a_traits.cuh"
#include "runtime/pdl.h"
#include "model/model_config.h"  // QType
#include "core/logging.h"

#include <cuda_fp16.h>
#include <cstdio>

namespace imp {

// ===========================================================================
// MMVQ: dp4a-accelerated quantized GEMV
// ===========================================================================
//
// Instead of dequantizing weights to FP16 and doing float*float per element,
// we quantize the input vector to Q8_1 (INT8 + scale) and use dp4a (native
// INT8x4 dot product) for accumulation. This processes 4 elements per
// instruction and halves input bandwidth (INT8 vs FP16).
//
// Q8_1 block: 32 values quantized to INT8 with per-block scale (d) and sum (s).
// ===========================================================================

// ---------------------------------------------------------------------------
// Activation functors for fused activation + Q8_1 quantization kernels.
//
// Each functor computes the activated value for one element. Gated variants
// (SwiGLU, GeGLU) read from both a gate and an up projection. Non-gated
// variants (Identity, ReLU²) use a single input.
// ---------------------------------------------------------------------------

struct IdentityAct {
    __device__ __forceinline__ static float operator()(const half* input, const half* /*unused*/, int idx,
                                                       int total) {
        // Early-out for OOB threads (preserves original quantize_fp16_to_q8_1 behavior
        // where partial-warp threads return instead of contributing val=0).
        (void)total;
        return __half2float(input[idx]);
    }
    static constexpr bool kEarlyReturn = true;  // OOB threads return before quantize
    static constexpr bool kClampQuantize = false;
};

struct SwiGLUAct {
    // SwiGLU: silu(gate) * up = gate * sigmoid(gate) * up
    __device__ __forceinline__ static float operator()(const half* gate, const half* up, int idx,
                                                       int /*total*/) {
        float g = __half2float(gate[idx]);
        float u = __half2float(up[idx]);
        return g / (1.0f + expf(-g)) * u;
    }
    static constexpr bool kEarlyReturn = false;
    static constexpr bool kClampQuantize = false;
};

struct GeGLUAct {
    // GEGLU: gelu_tanh(gate) * up
    __device__ __forceinline__ static float operator()(const half* gate, const half* up, int idx,
                                                       int /*total*/) {
        constexpr float SQRT_2_PI = 0.7978845608028654f;
        constexpr float COEFF = 0.044715f;
        float g = __half2float(gate[idx]);
        float u = __half2float(up[idx]);
        float gelu_g = g * 0.5f * (1.0f + tanhf(SQRT_2_PI * (g + COEFF * g * g * g)));
        return gelu_g * u;
    }
    static constexpr bool kEarlyReturn = false;
    static constexpr bool kClampQuantize = false;
};

struct ReLUSqrAct {
    // relu²(x) = max(0, x)²
    __device__ __forceinline__ static float operator()(const half* input, const half* /*unused*/, int idx,
                                                       int /*total*/) {
        float v = __half2float(input[idx]);
        return (v > 0.0f) ? v * v : 0.0f;
    }
    static constexpr bool kEarlyReturn = true;  // OOB threads return before quantize
    static constexpr bool kUsesGate = false;
    // ReLU² uses slightly different quantization: roundf + clamping (original behavior)
    static constexpr bool kClampQuantize = true;
};

// ---------------------------------------------------------------------------
// Fused activation + Q8_1 quantization kernel (templated).
//
// Each block handles 32 contiguous elements (one Q8_1 block), 32 threads/block.
// The activation functor determines which activation is applied before quantization.
// ---------------------------------------------------------------------------
template <typename Act>
__global__ void fused_act_quantize_q8_1_kernel(
    const half* __restrict__ input,         // primary input (or gate for gated acts)
    const half* __restrict__ gate_or_null,  // up projection for gated acts, null otherwise
    block_q8_1* __restrict__ q8_out, float* __restrict__ d8_out, int total_elements, Act act) {
    const int blk = blockIdx.x;
    const int tid = threadIdx.x;  // 0..31
    const int idx = blk * 32 + tid;

    // For kEarlyReturn functors: OOB threads return before shuffle reduction.
    // This preserves the original behavior of quantize_fp16_to_q8_1 and
    // relu_sqr_quantize_q8_1 where partial-warp threads exit early.
    if constexpr (Act::kEarlyReturn) {
        if (idx >= total_elements)
            return;
    }

    // Compute activated value
    float val = 0.0f;
    if constexpr (Act::kEarlyReturn) {
        val = act(input, gate_or_null, idx, total_elements);
    } else {
        if (idx < total_elements)
            val = act(input, gate_or_null, idx, total_elements);
    }

    // Warp-cooperative quantization: find max abs
    float amax = fabsf(val);
    for (int off = 16; off > 0; off >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, off));

    float d = amax / 127.0f;
    int8_t q;
    if constexpr (Act::kClampQuantize) {
        // ReLU² path: uses roundf + explicit clamping (original behavior)
        float id = (d > 0.0f) ? 127.0f / amax : 0.0f;
        q = (int8_t)fminf(fmaxf(roundf(val * id), -127.0f), 127.0f);
    } else {
        // Standard path: __float2int_rn
        float id = (d != 0.0f) ? (1.0f / d) : 0.0f;
        q = static_cast<int8_t>(__float2int_rn(val * id));
    }

    // Write Q8_1 block
    q8_out[blk].qs[tid] = q;
    if (tid == 0) {
        q8_out[blk].d = __float2half(d);
        d8_out[blk] = d;
    }
}

// Quantize K FP16 values to Q8_1 blocks.
// Grid: K/32 blocks, 32 threads each. One block per Q8_1 output block.
void quantize_fp16_to_q8_1(const half* x, block_q8_1* q8_1_out, float* d8_out, int K, cudaStream_t stream) {
    int n_blocks = K / 32;
    if (n_blocks <= 0)
        return;
    fused_act_quantize_q8_1_kernel<<<n_blocks, 32, 0, stream>>>(x, nullptr, q8_1_out, d8_out, K,
                                                                IdentityAct{});
    IMP_CUDA_CHECK_LAUNCH();

    // Pad to the next Q6_K block boundary (256 elements = 8 Q8_1 blocks).
    // When K is not a multiple of 256, the dp4a GEMV reads ceil(K/256)*8
    // Q8_1 blocks but only K/32 exist. Zero-fill the gap to prevent
    // out-of-bounds garbage causing NaN (Nemotron: K=2688=10.5*256).
    int padded_blocks = ((K + 255) / 256) * 8;  // ceil(K/256) * (256/32)
    if (padded_blocks > n_blocks) {
        int pad_count = padded_blocks - n_blocks;
        cudaMemsetAsync(q8_1_out + n_blocks, 0, pad_count * sizeof(block_q8_1), stream);
        cudaMemsetAsync(d8_out + n_blocks, 0, pad_count * sizeof(float), stream);
    }
}

// Fused SwiGLU + Q8_1 quantization.
// Computes silu(gate) * up and quantizes the result to Q8_1 in a single pass.
// Eliminates the intermediate FP16 activation buffer write+read.
void swiglu_quantize_q8_1(const half* gate, const half* up, block_q8_1* q8_out, float* d8_out,
                          int total_elements, cudaStream_t stream) {
    int n_blocks = total_elements / 32;
    if (n_blocks <= 0)
        return;
    fused_act_quantize_q8_1_kernel<<<n_blocks, 32, 0, stream>>>(gate, up, q8_out, d8_out, total_elements,
                                                                SwiGLUAct{});
    IMP_CUDA_CHECK_LAUNCH();
}

// Fused GEGLU + Q8_1 quantization.
// Computes gelu_tanh(gate) * up and quantizes the result to Q8_1 in one pass.
// Eliminates the intermediate FP16 activation buffer write+read for GEGLU models
// (Gemma-3).
void geglu_quantize_q8_1(const half* gate, const half* up, block_q8_1* q8_out, float* d8_out,
                         int total_elements, cudaStream_t stream) {
    int n_blocks = total_elements / 32;
    if (n_blocks <= 0)
        return;
    fused_act_quantize_q8_1_kernel<<<n_blocks, 32, 0, stream>>>(gate, up, q8_out, d8_out, total_elements,
                                                                GeGLUAct{});
    IMP_CUDA_CHECK_LAUNCH();
}

// Fused relu² + Q8_1 quantization.
// Reads FP16 input, applies relu²(x) = max(0, x)², quantizes to Q8_1.
// Replaces 3 separate operations (memcpy + relu_sqr_inplace + quantize).
// Used by non-gated MoE experts (Nemotron).
void relu_sqr_quantize_q8_1(const half* input, block_q8_1* q8_out, float* d8_out, int total_elements,
                            cudaStream_t stream) {
    int n_blocks = total_elements / 32;
    if (n_blocks <= 0)
        return;
    fused_act_quantize_q8_1_kernel<<<n_blocks, 32, 0, stream>>>(input, nullptr, q8_out, d8_out,
                                                                total_elements, ReLUSqrAct{});
    IMP_CUDA_CHECK_LAUNCH();
}

// ---------------------------------------------------------------------------
// Fused RMSNorm + Q8_1 quantization kernel.
//
// Combines RMSNorm (with weight) and FP16→Q8_1 quantization in one kernel,
// eliminating the intermediate norm_out FP16 buffer write+read.
//
// Single-row only (n=1 decode). One CUDA block, 256 threads.
// Phase 1: Load hidden, compute sum of squares, block-reduce for RMS.
// Phase 2: Normalize (multiply by inv_rms * weight), write to shared memory.
// Phase 3: Quantize from shared memory to Q8_1 blocks (32 elements per warp).
//
// Also writes the FP16 norm_out if norm_out_ptr is non-null (needed when
// the GEMV path doesn't consume Q8_1, e.g. non-quantized weights).
// ---------------------------------------------------------------------------
__global__ void rmsnorm_quantize_q8_1_kernel(
    const half* __restrict__ x,       // [d_model] input hidden state
    const half* __restrict__ weight,  // [d_model] RMSNorm weight
    block_q8_1* __restrict__ q8_out,  // [d_model/32] Q8_1 output blocks
    float* __restrict__ d8_out,       // [d_model/32] Q8_1 block scales
    half* __restrict__ norm_out_ptr,  // [d_model] optional FP16 output (can be null)
    int d_model, float eps, float weight_offset) {
    // Shared memory for cross-warp reduction + inv_rms.
    // Support up to 32 warps (1024 threads).
    __shared__ float warp_reduce[32];
    __shared__ float s_inv_rms;

    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    const int n_warps = blockDim.x >> 5;
    const int n_q8_blocks = d_model >> 5;  // d_model / 32

    // Phase 1: Load x values with warp-aligned Q8_1 block access (coalesced),
    // cache in registers, and compute sum of squares.
    // Each warp handles blocks in stride-n_warps order.
    // Max blocks per warp = d_model / (32 * n_warps) = d_model / 256.
    // For d_model=8192, that's 32 — fits in registers easily.
    float x_cache[32];
    float sum_sq = 0.0f;
    int n_cached = 0;

    for (int b = warp_id; b < n_q8_blocks; b += n_warps) {
        float v = __half2float(x[b * 32 + lane]);
        x_cache[n_cached++] = v;
        sum_sq += v * v;
    }

// Warp reduce sum_sq
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        sum_sq += __shfl_xor_sync(0xFFFFFFFF, sum_sq, off);
    if (lane == 0)
        warp_reduce[warp_id] = sum_sq;
    __syncthreads();

    // First warp reduces across all warps
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

    // Phase 2+3: Normalize (from register cache) + Quantize to Q8_1 (fused).
    // No intermediate shared memory buffer needed.
    int cache_idx = 0;
    for (int b = warp_id; b < n_q8_blocks; b += n_warps) {
        float val = x_cache[cache_idx++] * inv_rms * (__half2float(weight[b * 32 + lane]) + weight_offset);

        if (norm_out_ptr)
            norm_out_ptr[b * 32 + lane] = __float2half(val);

        // Warp-level amax for Q8_1 quantization
        float amax = fabsf(val);
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, off));

        float d = amax / 127.0f;
        float id = (d > 0.0f) ? (1.0f / d) : 0.0f;

        int8_t q = static_cast<int8_t>(__float2int_rn(val * id));

        q8_out[b].qs[lane] = q;
        if (lane == 0) {
            q8_out[b].d = __float2half(d);
            d8_out[b] = d;
        }
    }
}

void rmsnorm_quantize_q8_1(const half* x, const half* weight, block_q8_1* q8_out, float* d8_out,
                           half* norm_out, int d_model, float eps, cudaStream_t stream, float weight_offset) {
    const int threads = 1024;
    rmsnorm_quantize_q8_1_kernel<<<1, threads, 0, stream>>>(x, weight, q8_out, d8_out, norm_out, d_model, eps,
                                                            weight_offset);
    IMP_CUDA_CHECK_LAUNCH();
}
// ===========================================================================
// dp4a GEMV template instantiations (consolidated from 33 hand-written kernels)
// See gemv_dp4a_traits.cuh for DequantTraits<QType> and 6 template kernels.
// ===========================================================================

// ---------------------------------------------------------------------------
// Basic + Residual wrappers (14 functions = 7 quant types × 2 variants)
// ---------------------------------------------------------------------------

#define IMP_DP4A_QUANT_TYPES(X) \
    X(q6k, Q6_K_Traits)         \
    X(q8_0, Q8_0_Traits)        \
    X(q4_0, Q4_0_Traits)        \
    X(q4_k, Q4_K_Traits)        \
    X(q5_k, Q5_K_Traits)        \
    X(q2_k, Q2_K_Traits)        \
    X(q3_k, Q3_K_Traits)

#define IMP_DEFINE_GEMV_DP4A(name, traits)                                                                   \
    void gemv_##name##_q8_1(const void* W, const block_q8_1* q8_1, const float* d8, half* y, int M, int K,   \
                            cudaStream_t stream) {                                                           \
        launch_gemv_dp4a<traits>(static_cast<const uint8_t*>(W), q8_1, d8, y, nullptr, false, M, K, stream); \
    }                                                                                                        \
    void gemv_##name##_q8_1_residual(const void* W, const block_q8_1* q8_1, const float* d8, half* y,        \
                                     const half* residual, int M, int K, cudaStream_t stream) {              \
        launch_gemv_dp4a<traits>(static_cast<const uint8_t*>(W), q8_1, d8, y, residual, true, M, K, stream); \
    }
IMP_DP4A_QUANT_TYPES(IMP_DEFINE_GEMV_DP4A)
#undef IMP_DEFINE_GEMV_DP4A

// ---------------------------------------------------------------------------
// FP32 output wrappers (5 functions)
// ---------------------------------------------------------------------------

void gemv_q6k_q8_1_fp32(const void* W, const block_q8_1* q8_1, const float* d8, float* y, int M, int K,
                        cudaStream_t stream) {
    launch_gemv_dp4a_fp32<Q6_K_Traits>(static_cast<const uint8_t*>(W), q8_1, d8, y, M, K, stream);
}

void gemv_q8_0_q8_1_fp32(const void* W, const block_q8_1* q8_1, const float* d8, float* y, int M, int K,
                         cudaStream_t stream) {
    launch_gemv_dp4a_fp32<Q8_0_Traits>(static_cast<const uint8_t*>(W), q8_1, d8, y, M, K, stream);
}

void gemv_q4_0_q8_1_fp32(const void* W, const block_q8_1* q8_1, const float* d8, float* y, int M, int K,
                         cudaStream_t stream) {
    launch_gemv_dp4a_fp32<Q4_0_Traits>(static_cast<const uint8_t*>(W), q8_1, d8, y, M, K, stream);
}

void gemv_q4_k_q8_1_fp32(const void* W, const block_q8_1* q8_1, const float* d8, float* y, int M, int K,
                         cudaStream_t stream) {
    launch_gemv_dp4a_fp32<Q4_K_Traits>(static_cast<const uint8_t*>(W), q8_1, d8, y, M, K, stream);
}

void gemv_q5_k_q8_1_fp32(const void* W, const block_q8_1* q8_1, const float* d8, float* y, int M, int K,
                         cudaStream_t stream) {
    launch_gemv_dp4a_fp32<Q5_K_Traits>(static_cast<const uint8_t*>(W), q8_1, d8, y, M, K, stream);
}

void gemv_q2_k_q8_1_fp32(const void* W, const block_q8_1* q8_1, const float* d8, float* y, int M, int K,
                         cudaStream_t stream) {
    launch_gemv_dp4a_fp32<Q2_K_Traits>(static_cast<const uint8_t*>(W), q8_1, d8, y, M, K, stream);
}

void gemv_q3_k_q8_1_fp32(const void* W, const block_q8_1* q8_1, const float* d8, float* y, int M, int K,
                         cudaStream_t stream) {
    launch_gemv_dp4a_fp32<Q3_K_Traits>(static_cast<const uint8_t*>(W), q8_1, d8, y, M, K, stream);
}

void gemv_dp4a_fp32_batched(QType qtype, const void* W, const block_q8_1* q8_1, const float* d8,
                            float* y, int M, int K, int n_act, int act_stride_blocks,
                            cudaStream_t stream) {
    const uint8_t* w = static_cast<const uint8_t*>(W);
    switch (qtype) {
        case QType::Q6_K:
            launch_gemv_dp4a_fp32_batched<Q6_K_Traits>(w, q8_1, d8, y, M, K, n_act, act_stride_blocks,
                                                       stream);
            break;
        case QType::Q4_0:
            launch_gemv_dp4a_fp32_batched<Q4_0_Traits>(w, q8_1, d8, y, M, K, n_act, act_stride_blocks,
                                                       stream);
            break;
        case QType::Q4_K:
            launch_gemv_dp4a_fp32_batched<Q4_K_Traits>(w, q8_1, d8, y, M, K, n_act, act_stride_blocks,
                                                       stream);
            break;
        case QType::Q5_K:
            launch_gemv_dp4a_fp32_batched<Q5_K_Traits>(w, q8_1, d8, y, M, K, n_act, act_stride_blocks,
                                                       stream);
            break;
        case QType::Q2_K:
            launch_gemv_dp4a_fp32_batched<Q2_K_Traits>(w, q8_1, d8, y, M, K, n_act, act_stride_blocks,
                                                       stream);
            break;
        case QType::Q3_K:
            launch_gemv_dp4a_fp32_batched<Q3_K_Traits>(w, q8_1, d8, y, M, K, n_act, act_stride_blocks,
                                                       stream);
            break;
        default:
            launch_gemv_dp4a_fp32_batched<Q8_0_Traits>(w, q8_1, d8, y, M, K, n_act, act_stride_blocks,
                                                       stream);
            break;
    }
}

// ---------------------------------------------------------------------------
// QKV fused wrappers (7 functions, one per quant type)
// ---------------------------------------------------------------------------

#define IMP_DEFINE_GEMV_DP4A_QKV(name, traits)                                                           \
    void gemv_qkv_fused_##name##_q8_1(const void* W_q, const void* W_k, const void* W_v,                 \
                                      const block_q8_1* q8_1, const float* d8, half* y_q, half* y_k,     \
                                      half* y_v, int q_rows, int k_rows, int v_rows, int K,              \
                                      cudaStream_t stream) {                                             \
        launch_gemv_dp4a_qkv<traits>(static_cast<const uint8_t*>(W_q), static_cast<const uint8_t*>(W_k), \
                                     static_cast<const uint8_t*>(W_v), q8_1, d8, y_q, y_k, y_v, q_rows,  \
                                     k_rows, v_rows, K, stream);                                         \
    }
IMP_DP4A_QUANT_TYPES(IMP_DEFINE_GEMV_DP4A_QKV)
#undef IMP_DEFINE_GEMV_DP4A_QKV

// ---------------------------------------------------------------------------
// Gate+Up fused dispatcher (1 function, dispatches by qtype)
// ---------------------------------------------------------------------------

void gemv_gate_up_fused(const void* gate_weights, const void* up_weights, const block_q8_1* q8_1,
                        const float* d8, half* y_gate, half* y_up, int M, int K, QType qtype,
                        cudaStream_t stream) {
    const int threads_per_block = 256;
    const int warps_per_block = threads_per_block / 32;

    const auto* gw = static_cast<const uint8_t*>(gate_weights);
    const auto* uw = static_cast<const uint8_t*>(up_weights);

    // Unified dispatch: K-par check (quant-type-aware) + row-par NR selection.
    // K-par check compares against NR=1 (max occupancy baseline), with tie-breaking
    // determined by QT::kPreferKpar (complex dequant types prefer K-par on ties).

#define LAUNCH_KPAR_GU(QT)                                                                                 \
    pdl::launch(gemv_dp4a_kpar_gate_up_kernel<QT>, dim3(M, 2), dim3(128), size_t(0), stream, gw, uw, q8_1, \
                d8, y_gate, y_up, M, K)

#define LAUNCH_GATE_UP_NR(QT, NR)                                                                          \
    do {                                                                                                   \
        const int rows_per_block = warps_per_block * NR;                                                   \
        const int blocks = (M + rows_per_block - 1) / rows_per_block;                                      \
        dim3 grid(blocks, 2);                                                                              \
        const int total_q8 = (K / QT::kBlockElems) * QT::kQ8PerWeight;                                     \
        const size_t smem = (size_t)total_q8 * (40 + QT::kSmemExtra);                                      \
        pdl::launch(gemv_dp4a_gate_up_kernel<QT, NR>, grid, dim3(threads_per_block), smem, stream, gw, uw, \
                    q8_1, d8, y_gate, y_up, M, K);                                                         \
    } while (0)

#define DISPATCH_GATE_UP(QT)                                                        \
    do {                                                                            \
        int nr1 = (M + warps_per_block - 1) / warps_per_block;                      \
        if (kpar_is_better<QT::kPreferKpar>(M, nr1)) {                              \
            LAUNCH_KPAR_GU(QT);                                                     \
            break;                                                                  \
        }                                                                           \
        constexpr int MAX_NR = QT::kMaxNRows;                                       \
        if constexpr (MAX_NR >= 4) {                                                \
            int nr4_blocks = (M + warps_per_block * 4 - 1) / (warps_per_block * 4); \
            if (nr4_blocks >= 128) {                                                \
                LAUNCH_GATE_UP_NR(QT, 4);                                           \
                break;                                                              \
            }                                                                       \
        }                                                                           \
        if constexpr (MAX_NR >= 2) {                                                \
            int nr2_blocks = (M + warps_per_block * 2 - 1) / (warps_per_block * 2); \
            if (nr2_blocks >= 64) {                                                 \
                LAUNCH_GATE_UP_NR(QT, 2);                                           \
                break;                                                              \
            }                                                                       \
        }                                                                           \
        LAUNCH_GATE_UP_NR(QT, 1);                                                   \
    } while (0)

    if (qtype == QType::Q6_K) {
        DISPATCH_GATE_UP(Q6_K_Traits);
    } else if (qtype == QType::Q8_0) {
        DISPATCH_GATE_UP(Q8_0_Traits);
    } else if (qtype == QType::Q4_0) {
        DISPATCH_GATE_UP(Q4_0_Traits);
    } else if (qtype == QType::Q4_K) {
        DISPATCH_GATE_UP(Q4_K_Traits);
    } else if (qtype == QType::Q5_K) {
        DISPATCH_GATE_UP(Q5_K_Traits);
    } else if (qtype == QType::Q2_K) {
        DISPATCH_GATE_UP(Q2_K_Traits);
    } else if (qtype == QType::Q3_K) {
        DISPATCH_GATE_UP(Q3_K_Traits);
    }

#undef DISPATCH_GATE_UP
#undef LAUNCH_GATE_UP_NR
#undef LAUNCH_KPAR_GU
}
// ---------------------------------------------------------------------------
// dp4a MoE decode wrappers (4 functions)
// ---------------------------------------------------------------------------

void gemv_q6k_q8_1_moe_decode(const void* packed_weights, const int32_t* expert_indices,
                              const block_q8_1* q8_1, const float* d8, half* y, int rows, int K,
                              size_t expert_stride_bytes, int q8_1_stride, int d8_stride, int top_k,
                              cudaStream_t stream) {
    launch_gemv_dp4a_moe_decode<Q6_K_Traits>(static_cast<const uint8_t*>(packed_weights), expert_indices,
                                             q8_1, d8, y, rows, K, expert_stride_bytes, q8_1_stride,
                                             d8_stride, top_k, stream);
}

void gemv_q8_0_q8_1_moe_decode(const void* packed_weights, const int32_t* expert_indices,
                               const block_q8_1* q8_1, const float* d8, half* y, int rows, int K,
                               size_t expert_stride_bytes, int q8_1_stride, int d8_stride, int top_k,
                               cudaStream_t stream) {
    launch_gemv_dp4a_moe_decode<Q8_0_Traits>(static_cast<const uint8_t*>(packed_weights), expert_indices,
                                             q8_1, d8, y, rows, K, expert_stride_bytes, q8_1_stride,
                                             d8_stride, top_k, stream);
}

void gemv_q4_k_q8_1_moe_decode(const void* packed_weights, const int32_t* expert_indices,
                               const block_q8_1* q8_1, const float* d8, half* y, int rows, int K,
                               size_t expert_stride_bytes, int q8_1_stride, int d8_stride, int top_k,
                               cudaStream_t stream) {
    launch_gemv_dp4a_moe_decode<Q4_K_Traits>(static_cast<const uint8_t*>(packed_weights), expert_indices,
                                             q8_1, d8, y, rows, K, expert_stride_bytes, q8_1_stride,
                                             d8_stride, top_k, stream);
}

void gemv_q5_k_q8_1_moe_decode(const void* packed_weights, const int32_t* expert_indices,
                               const block_q8_1* q8_1, const float* d8, half* y, int rows, int K,
                               size_t expert_stride_bytes, int q8_1_stride, int d8_stride, int top_k,
                               cudaStream_t stream) {
    launch_gemv_dp4a_moe_decode<Q5_K_Traits>(static_cast<const uint8_t*>(packed_weights), expert_indices,
                                             q8_1, d8, y, rows, K, expert_stride_bytes, q8_1_stride,
                                             d8_stride, top_k, stream);
}

void gemv_q4_0_q8_1_moe_decode(const void* packed_weights, const int32_t* expert_indices,
                               const block_q8_1* q8_1, const float* d8, half* y, int rows, int K,
                               size_t expert_stride_bytes, int q8_1_stride, int d8_stride, int top_k,
                               cudaStream_t stream) {
    launch_gemv_dp4a_moe_decode<Q4_0_Traits>(static_cast<const uint8_t*>(packed_weights), expert_indices,
                                             q8_1, d8, y, rows, K, expert_stride_bytes, q8_1_stride,
                                             d8_stride, top_k, stream);
}

void gemv_q2_k_q8_1_moe_decode(const void* packed_weights, const int32_t* expert_indices,
                               const block_q8_1* q8_1, const float* d8, half* y, int rows, int K,
                               size_t expert_stride_bytes, int q8_1_stride, int d8_stride, int top_k,
                               cudaStream_t stream) {
    launch_gemv_dp4a_moe_decode<Q2_K_Traits>(static_cast<const uint8_t*>(packed_weights), expert_indices,
                                             q8_1, d8, y, rows, K, expert_stride_bytes, q8_1_stride,
                                             d8_stride, top_k, stream);
}

void gemv_q3_k_q8_1_moe_decode(const void* packed_weights, const int32_t* expert_indices,
                               const block_q8_1* q8_1, const float* d8, half* y, int rows, int K,
                               size_t expert_stride_bytes, int q8_1_stride, int d8_stride, int top_k,
                               cudaStream_t stream) {
    launch_gemv_dp4a_moe_decode<Q3_K_Traits>(static_cast<const uint8_t*>(packed_weights), expert_indices,
                                             q8_1, d8, y, rows, K, expert_stride_bytes, q8_1_stride,
                                             d8_stride, top_k, stream);
}

void gemv_q5_1_q8_1_moe_decode(const void* packed_weights, const int32_t* expert_indices,
                               const block_q8_1* q8_1, const float* d8, half* y, int rows, int K,
                               size_t expert_stride_bytes, int q8_1_stride, int d8_stride, int top_k,
                               cudaStream_t stream) {
    launch_gemv_dp4a_moe_decode<Q5_1_Traits>(static_cast<const uint8_t*>(packed_weights), expert_indices,
                                             q8_1, d8, y, rows, K, expert_stride_bytes, q8_1_stride,
                                             d8_stride, top_k, stream);
}

// ---------------------------------------------------------------------------
// dp4a MoE gate+up fused wrappers (4 functions)
// ---------------------------------------------------------------------------

void gemv_q6k_q8_1_moe_gate_up_fused(const void* gate_weights, const void* up_weights,
                                     const int32_t* expert_indices, const block_q8_1* q8_1, const float* d8,
                                     half* y_gate, half* y_up, int rows, int K, size_t gate_stride_bytes,
                                     size_t up_stride_bytes, int q8_1_stride, int d8_stride, int top_k,
                                     cudaStream_t stream) {
    launch_gemv_dp4a_moe_gate_up<Q6_K_Traits>(static_cast<const uint8_t*>(gate_weights),
                                              static_cast<const uint8_t*>(up_weights), expert_indices, q8_1,
                                              d8, y_gate, y_up, rows, K, gate_stride_bytes, up_stride_bytes,
                                              q8_1_stride, d8_stride, top_k, stream);
}

void gemv_q8_0_q8_1_moe_gate_up_fused(const void* gate_weights, const void* up_weights,
                                      const int32_t* expert_indices, const block_q8_1* q8_1, const float* d8,
                                      half* y_gate, half* y_up, int rows, int K, size_t gate_stride_bytes,
                                      size_t up_stride_bytes, int q8_1_stride, int d8_stride, int top_k,
                                      cudaStream_t stream) {
    launch_gemv_dp4a_moe_gate_up<Q8_0_Traits>(static_cast<const uint8_t*>(gate_weights),
                                              static_cast<const uint8_t*>(up_weights), expert_indices, q8_1,
                                              d8, y_gate, y_up, rows, K, gate_stride_bytes, up_stride_bytes,
                                              q8_1_stride, d8_stride, top_k, stream);
}

void gemv_q4_k_q8_1_moe_gate_up_fused(const void* gate_weights, const void* up_weights,
                                      const int32_t* expert_indices, const block_q8_1* q8_1, const float* d8,
                                      half* y_gate, half* y_up, int rows, int K, size_t gate_stride_bytes,
                                      size_t up_stride_bytes, int q8_1_stride, int d8_stride, int top_k,
                                      cudaStream_t stream) {
    launch_gemv_dp4a_moe_gate_up<Q4_K_Traits>(static_cast<const uint8_t*>(gate_weights),
                                              static_cast<const uint8_t*>(up_weights), expert_indices, q8_1,
                                              d8, y_gate, y_up, rows, K, gate_stride_bytes, up_stride_bytes,
                                              q8_1_stride, d8_stride, top_k, stream);
}

void gemv_q5_k_q8_1_moe_gate_up_fused(const void* gate_weights, const void* up_weights,
                                      const int32_t* expert_indices, const block_q8_1* q8_1, const float* d8,
                                      half* y_gate, half* y_up, int rows, int K, size_t gate_stride_bytes,
                                      size_t up_stride_bytes, int q8_1_stride, int d8_stride, int top_k,
                                      cudaStream_t stream) {
    launch_gemv_dp4a_moe_gate_up<Q5_K_Traits>(static_cast<const uint8_t*>(gate_weights),
                                              static_cast<const uint8_t*>(up_weights), expert_indices, q8_1,
                                              d8, y_gate, y_up, rows, K, gate_stride_bytes, up_stride_bytes,
                                              q8_1_stride, d8_stride, top_k, stream);
}

void gemv_q4_0_q8_1_moe_gate_up_fused(const void* gate_weights, const void* up_weights,
                                      const int32_t* expert_indices, const block_q8_1* q8_1, const float* d8,
                                      half* y_gate, half* y_up, int rows, int K, size_t gate_stride_bytes,
                                      size_t up_stride_bytes, int q8_1_stride, int d8_stride, int top_k,
                                      cudaStream_t stream) {
    launch_gemv_dp4a_moe_gate_up<Q4_0_Traits>(static_cast<const uint8_t*>(gate_weights),
                                              static_cast<const uint8_t*>(up_weights), expert_indices, q8_1,
                                              d8, y_gate, y_up, rows, K, gate_stride_bytes, up_stride_bytes,
                                              q8_1_stride, d8_stride, top_k, stream);
}

void gemv_q2_k_q8_1_moe_gate_up_fused(const void* gate_weights, const void* up_weights,
                                      const int32_t* expert_indices, const block_q8_1* q8_1, const float* d8,
                                      half* y_gate, half* y_up, int rows, int K, size_t gate_stride_bytes,
                                      size_t up_stride_bytes, int q8_1_stride, int d8_stride, int top_k,
                                      cudaStream_t stream) {
    launch_gemv_dp4a_moe_gate_up<Q2_K_Traits>(static_cast<const uint8_t*>(gate_weights),
                                              static_cast<const uint8_t*>(up_weights), expert_indices, q8_1,
                                              d8, y_gate, y_up, rows, K, gate_stride_bytes, up_stride_bytes,
                                              q8_1_stride, d8_stride, top_k, stream);
}

void gemv_q3_k_q8_1_moe_gate_up_fused(const void* gate_weights, const void* up_weights,
                                      const int32_t* expert_indices, const block_q8_1* q8_1, const float* d8,
                                      half* y_gate, half* y_up, int rows, int K, size_t gate_stride_bytes,
                                      size_t up_stride_bytes, int q8_1_stride, int d8_stride, int top_k,
                                      cudaStream_t stream) {
    launch_gemv_dp4a_moe_gate_up<Q3_K_Traits>(static_cast<const uint8_t*>(gate_weights),
                                              static_cast<const uint8_t*>(up_weights), expert_indices, q8_1,
                                              d8, y_gate, y_up, rows, K, gate_stride_bytes, up_stride_bytes,
                                              q8_1_stride, d8_stride, top_k, stream);
}
// ---------------------------------------------------------------------------
// PDL registration for all dp4a GEMV kernel template instantiations.
// Called from GraphExecutor::init() when PDL is enabled.
// ---------------------------------------------------------------------------
// GEMV kernels are bandwidth-bound with minimal SMEM: maximize L1 cache.
// Variadic to handle template commas: SET_MAXL1(kernel<A, B, C>)
#define SET_MAXL1(...)                                                                \
    cudaFuncSetAttribute(__VA_ARGS__, cudaFuncAttributePreferredSharedMemoryCarveout, \
                         cudaSharedmemCarveoutMaxL1)

void gemv_pdl_register() {
// Kernel #1: basic + residual
#define REG1(QT, NR)                                     \
    pdl::enable_kernel(gemv_dp4a_kernel<QT, NR, true>);  \
    pdl::enable_kernel(gemv_dp4a_kernel<QT, NR, false>); \
    SET_MAXL1(gemv_dp4a_kernel<QT, NR, true>);           \
    SET_MAXL1(gemv_dp4a_kernel<QT, NR, false>)
    REG1(Q6_K_Traits, 1);
    REG1(Q6_K_Traits, 2);
    REG1(Q8_0_Traits, 1);
    REG1(Q8_0_Traits, 2);
    REG1(Q8_0_Traits, 4);
    REG1(Q4_0_Traits, 1);
    REG1(Q4_0_Traits, 2);
    REG1(Q4_0_Traits, 4);
    REG1(Q4_K_Traits, 1);
    REG1(Q4_K_Traits, 2);
    REG1(Q4_K_Traits, 4);
    REG1(Q5_K_Traits, 1);
    REG1(Q5_K_Traits, 2);
    REG1(Q5_K_Traits, 4);
    REG1(Q2_K_Traits, 1);
    REG1(Q2_K_Traits, 2);
    REG1(Q2_K_Traits, 4);
    REG1(Q3_K_Traits, 1);
    REG1(Q3_K_Traits, 2);
#undef REG1

// Kernel #2: FP32 output
#define REG2(QT, NR)                                   \
    pdl::enable_kernel(gemv_dp4a_fp32_kernel<QT, NR>); \
    SET_MAXL1(gemv_dp4a_fp32_kernel<QT, NR>)
    REG2(Q6_K_Traits, 1);
    REG2(Q6_K_Traits, 2);
    REG2(Q8_0_Traits, 1);
    REG2(Q8_0_Traits, 2);
    REG2(Q8_0_Traits, 4);
    REG2(Q4_0_Traits, 1);
    REG2(Q4_0_Traits, 2);
    REG2(Q4_0_Traits, 4);
    REG2(Q4_K_Traits, 1);
    REG2(Q4_K_Traits, 2);
    REG2(Q4_K_Traits, 4);
    REG2(Q5_K_Traits, 1);
    REG2(Q5_K_Traits, 2);
    REG2(Q5_K_Traits, 4);
    REG2(Q2_K_Traits, 1);
    REG2(Q2_K_Traits, 2);
    REG2(Q2_K_Traits, 4);
    REG2(Q3_K_Traits, 1);
    REG2(Q3_K_Traits, 2);
#undef REG2

// Kernel #3: QKV fused
#define REG3(QT, NR)                                  \
    pdl::enable_kernel(gemv_dp4a_qkv_kernel<QT, NR>); \
    SET_MAXL1(gemv_dp4a_qkv_kernel<QT, NR>)
    REG3(Q6_K_Traits, 1);
    REG3(Q6_K_Traits, 2);
    REG3(Q8_0_Traits, 1);
    REG3(Q8_0_Traits, 2);
    REG3(Q8_0_Traits, 4);
    REG3(Q4_0_Traits, 1);
    REG3(Q4_0_Traits, 2);
    REG3(Q4_0_Traits, 4);
    REG3(Q4_K_Traits, 1);
    REG3(Q4_K_Traits, 2);
    REG3(Q4_K_Traits, 4);
    REG3(Q5_K_Traits, 1);
    REG3(Q5_K_Traits, 2);
    REG3(Q5_K_Traits, 4);
    REG3(Q2_K_Traits, 1);
    REG3(Q2_K_Traits, 2);
    REG3(Q2_K_Traits, 4);
    REG3(Q3_K_Traits, 1);
    REG3(Q3_K_Traits, 2);
#undef REG3

// Kernel #4: gate+up fused
#define REG4(QT, NR)                                      \
    pdl::enable_kernel(gemv_dp4a_gate_up_kernel<QT, NR>); \
    SET_MAXL1(gemv_dp4a_gate_up_kernel<QT, NR>)
    REG4(Q6_K_Traits, 1);
    REG4(Q6_K_Traits, 2);
    REG4(Q8_0_Traits, 1);
    REG4(Q8_0_Traits, 2);
    REG4(Q8_0_Traits, 4);
    REG4(Q4_0_Traits, 1);
    REG4(Q4_0_Traits, 2);
    REG4(Q4_0_Traits, 4);
    REG4(Q4_K_Traits, 1);
    REG4(Q4_K_Traits, 2);
    REG4(Q4_K_Traits, 4);
    REG4(Q5_K_Traits, 1);
    REG4(Q5_K_Traits, 2);
    REG4(Q5_K_Traits, 4);
    REG4(Q2_K_Traits, 1);
    REG4(Q2_K_Traits, 2);
    REG4(Q2_K_Traits, 4);
    REG4(Q3_K_Traits, 1);
    REG4(Q3_K_Traits, 2);
#undef REG4

// Kernels #5 and #6 (MoE decode/gate+up): NOT registered with PDL.
// MoE kernels are small (top_k=2, few blocks per expert) and launched
// frequently (23 MoE layers × 3 kernels = 69 per decode step on Nemotron).
// cudaLaunchKernelEx overhead outweighs PDL tail/head overlap benefit
// for these tiny kernels.

// K-parallel kernels (all types)
#define REG_KPAR(QT)                                       \
    pdl::enable_kernel(gemv_dp4a_kpar_kernel<QT, true>);   \
    pdl::enable_kernel(gemv_dp4a_kpar_kernel<QT, false>);  \
    pdl::enable_kernel(gemv_dp4a_kpar_fp32_kernel<QT>);    \
    pdl::enable_kernel(gemv_dp4a_kpar_qkv_kernel<QT>);     \
    pdl::enable_kernel(gemv_dp4a_kpar_gate_up_kernel<QT>); \
    SET_MAXL1(gemv_dp4a_kpar_kernel<QT, true>);            \
    SET_MAXL1(gemv_dp4a_kpar_kernel<QT, false>);           \
    SET_MAXL1(gemv_dp4a_kpar_fp32_kernel<QT>);             \
    SET_MAXL1(gemv_dp4a_kpar_qkv_kernel<QT>);              \
    SET_MAXL1(gemv_dp4a_kpar_gate_up_kernel<QT>)
    REG_KPAR(Q6_K_Traits);
    REG_KPAR(Q8_0_Traits);
    REG_KPAR(Q4_0_Traits);
    REG_KPAR(Q4_K_Traits);
    REG_KPAR(Q5_K_Traits);
    REG_KPAR(Q2_K_Traits);
    REG_KPAR(Q3_K_Traits);
#undef REG_KPAR
}

#undef SET_MAXL1

}  // namespace imp
