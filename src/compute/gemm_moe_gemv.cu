#include "compute/gemm.h"
#include "compute/gemm_internal.cuh"
#include "core/logging.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace imp {

// ---------------------------------------------------------------------------
// MoE decode GEMV: processes all top_k experts in a single kernel launch.
// expert_indices[slot] selects which expert's weights to read from packed_weights.
// Grid: top_k * blocks_per_expert blocks. Each block group handles one expert slot.
// x_stride: 0 = shared input for all experts (gate/up), >0 = per-expert input (down).
// ---------------------------------------------------------------------------

__global__ void gemv_q6k_moe_decode_kernel(const uint8_t* __restrict__ packed_weights,
                                           const int32_t* __restrict__ expert_indices,
                                           const half* __restrict__ x, half* __restrict__ y, int rows, int K,
                                           size_t expert_stride_bytes, int x_stride, int blocks_per_expert) {
    const int warps_per_block = blockDim.x / 32;
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;

    const int expert_slot = blockIdx.x / blocks_per_expert;
    const int local_block = blockIdx.x % blocks_per_expert;
    const int row = local_block * warps_per_block + warp_id;

    if (row >= rows)
        return;

    const int expert_id = expert_indices[expert_slot];
    const uint8_t* W = packed_weights + (size_t)expert_id * expert_stride_bytes;

    const int blocks_per_row = K / 256;
    const size_t row_bytes = (size_t)blocks_per_row * 210;
    const uint8_t* W_row = W + (size_t)row * row_bytes;

    const half* x_ptr = x + expert_slot * x_stride;
    float sum = 0.0f;

    for (int b = 0; b < blocks_per_row; ++b) {
        const uint8_t* bp = W_row + b * 210;
        const uint8_t* ql = bp;
        const uint8_t* qh = bp + 128;
        const int8_t* sc = (const int8_t*)(bp + 192);
        float d = __half2float(*(const half*)(bp + 208));
        const int base = b * 256;

        uint8_t ql_a = ql[lane];
        uint8_t ql_b = ql[lane + 32];
        uint8_t ql_c = ql[64 + lane];
        uint8_t ql_d = ql[64 + lane + 32];
        uint8_t qh0 = qh[lane];
        uint8_t qh1 = qh[32 + lane];

        int q0 = (int)(((qh0 & 0x03) << 4) | (ql_a & 0x0F)) - 32;
        int q1 = (int)((((qh0 >> 2) & 0x03) << 4) | (ql_b & 0x0F)) - 32;
        int q2 = (int)((((qh0 >> 4) & 0x03) << 4) | ((ql_a >> 4) & 0x0F)) - 32;
        int q3 = (int)((((qh0 >> 6) & 0x03) << 4) | ((ql_b >> 4) & 0x0F)) - 32;
        int q4 = (int)(((qh1 & 0x03) << 4) | (ql_c & 0x0F)) - 32;
        int q5 = (int)((((qh1 >> 2) & 0x03) << 4) | (ql_d & 0x0F)) - 32;
        int q6 = (int)((((qh1 >> 4) & 0x03) << 4) | ((ql_c >> 4) & 0x0F)) - 32;
        int q7 = (int)((((qh1 >> 6) & 0x03) << 4) | ((ql_d >> 4) & 0x0F)) - 32;

        int sc_idx = lane >> 4;
        sum += d * ((float)sc[sc_idx] * (float)q0 * __half2float(x_ptr[base + lane]) +
                    (float)sc[sc_idx + 2] * (float)q1 * __half2float(x_ptr[base + lane + 32]) +
                    (float)sc[sc_idx + 4] * (float)q2 * __half2float(x_ptr[base + lane + 64]) +
                    (float)sc[sc_idx + 6] * (float)q3 * __half2float(x_ptr[base + lane + 96]) +
                    (float)sc[sc_idx + 8] * (float)q4 * __half2float(x_ptr[base + lane + 128]) +
                    (float)sc[sc_idx + 10] * (float)q5 * __half2float(x_ptr[base + lane + 160]) +
                    (float)sc[sc_idx + 12] * (float)q6 * __half2float(x_ptr[base + lane + 192]) +
                    (float)sc[sc_idx + 14] * (float)q7 * __half2float(x_ptr[base + lane + 224]));
    }

    sum = warp_reduce_sum(sum);

    if (lane == 0)
        y[expert_slot * rows + row] = __float2half(sum);
}

void gemv_q6k_moe_decode(const void* packed_weights, const int32_t* expert_indices, const half* x, half* y,
                         int rows, int K, size_t expert_stride_bytes, int x_stride, int top_k,
                         cudaStream_t stream) {
    const int blocks_per_expert = gemv_blocks(rows);
    gemv_q6k_moe_decode_kernel<<<top_k * blocks_per_expert, kGemvThreads, 0, stream>>>(
        static_cast<const uint8_t*>(packed_weights), expert_indices, x, y, rows, K, expert_stride_bytes,
        x_stride, blocks_per_expert);
    IMP_CUDA_CHECK_LAUNCH();
}

__global__ void gemv_q8_0_moe_decode_kernel(const uint8_t* __restrict__ packed_weights,
                                            const int32_t* __restrict__ expert_indices,
                                            const half* __restrict__ x, half* __restrict__ y, int rows, int K,
                                            size_t expert_stride_bytes, int x_stride, int blocks_per_expert) {
    const int warps_per_block = blockDim.x / 32;
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;

    const int expert_slot = blockIdx.x / blocks_per_expert;
    const int local_block = blockIdx.x % blocks_per_expert;
    const int row = local_block * warps_per_block + warp_id;

    if (row >= rows)
        return;

    const int expert_id = expert_indices[expert_slot];
    const uint8_t* W = packed_weights + (size_t)expert_id * expert_stride_bytes;

    const int blocks_per_row = K / 32;
    const size_t row_bytes = (size_t)blocks_per_row * 34;
    const uint8_t* W_row = W + (size_t)row * row_bytes;

    const half* x_ptr = x + expert_slot * x_stride;
    float sum = 0.0f;

    for (int b = 0; b < blocks_per_row; ++b) {
        const uint8_t* bp = W_row + b * 34;
        float d = __half2float(*(const half*)bp);
        int8_t q = ((const int8_t*)(bp + 2))[lane];
        sum += d * (float)q * __half2float(x_ptr[b * 32 + lane]);
    }

    sum = warp_reduce_sum(sum);

    if (lane == 0)
        y[expert_slot * rows + row] = __float2half(sum);
}

void gemv_q8_0_moe_decode(const void* packed_weights, const int32_t* expert_indices, const half* x, half* y,
                          int rows, int K, size_t expert_stride_bytes, int x_stride, int top_k,
                          cudaStream_t stream) {
    const int blocks_per_expert = gemv_blocks(rows);
    gemv_q8_0_moe_decode_kernel<<<top_k * blocks_per_expert, kGemvThreads, 0, stream>>>(
        static_cast<const uint8_t*>(packed_weights), expert_indices, x, y, rows, K, expert_stride_bytes,
        x_stride, blocks_per_expert);
    IMP_CUDA_CHECK_LAUNCH();
}

// ---------------------------------------------------------------------------
// FP16 GEMV with FP32 output for MoE gate logits: y = W @ x
// W: [M, K] FP16 (row-major), x: [K] FP16, y: [M] FP32.
// Designed for M=n_experts (64-256), K=d_model (2048-8192), n=1 decode.
// Replaces cuBLAS gemm() + fp16_to_fp32 cast for tiny M=1 GEMMs.
// Each warp handles one output row. Uses half2 vectorized loads for 2x bandwidth.
// ---------------------------------------------------------------------------
__global__ void gemv_gate_fp32_kernel(const half* __restrict__ W, const half* __restrict__ x,
                                      float* __restrict__ y, int M, int K) {
    const int warps_per_block = blockDim.x / 32;
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int row = blockIdx.x * warps_per_block + warp_id;

    if (row >= M)
        return;

    const half* W_row = W + (size_t)row * K;
    float sum = 0.0f;

    // Process 2 elements per thread per iteration using half2
    const int K2 = K / 2;
    const half2* W2 = reinterpret_cast<const half2*>(W_row);
    const half2* x2 = reinterpret_cast<const half2*>(x);

    for (int i = lane; i < K2; i += 32) {
        half2 w = W2[i];
        half2 v = x2[i];
        sum += __half2float(w.x) * __half2float(v.x);
        sum += __half2float(w.y) * __half2float(v.y);
    }

    // Handle odd K (unlikely but safe)
    if ((K & 1) && lane == 0) {
        sum += __half2float(W_row[K - 1]) * __half2float(x[K - 1]);
    }

    // Warp shuffle reduction
    sum = warp_reduce_sum(sum);

    if (lane == 0)
        y[row] = sum;
}

void gemv_gate_fp32(const half* W, const half* x, float* y, int M, int K, cudaStream_t stream) {
    gemv_gate_fp32_kernel<<<gemv_blocks(M), kGemvThreads, 0, stream>>>(W, x, y, M, K);
    IMP_CUDA_CHECK_LAUNCH();
}

// FP32-input variant: avoids FP16 truncation of router input for MoE precision.
__global__ void gemv_gate_fp32_fp32input_kernel(const half* __restrict__ W, const float* __restrict__ x,
                                                float* __restrict__ y, int M, int K) {
    const int warps_per_block = blockDim.x / 32;
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int row = blockIdx.x * warps_per_block + warp_id;

    if (row >= M)
        return;

    const half* W_row = W + (size_t)row * K;
    float sum = 0.0f;

    // Process 2 weight elements per iteration (half2), read FP32 input directly
    const int K2 = K / 2;
    const half2* W2 = reinterpret_cast<const half2*>(W_row);

    for (int i = lane; i < K2; i += 32) {
        half2 w = W2[i];
        sum += __half2float(w.x) * x[i * 2];
        sum += __half2float(w.y) * x[i * 2 + 1];
    }

    if ((K & 1) && lane == 0) {
        sum += __half2float(W_row[K - 1]) * x[K - 1];
    }

    sum = warp_reduce_sum(sum);
    if (lane == 0)
        y[row] = sum;
}

void gemv_gate_fp32_fp32input(const half* W, const float* x, float* y, int M, int K, cudaStream_t stream) {
    gemv_gate_fp32_fp32input_kernel<<<gemv_blocks(M), kGemvThreads, 0, stream>>>(W, x, y, M, K);
    IMP_CUDA_CHECK_LAUNCH();
}

// ---------------------------------------------------------------------------
// Fused gate+up MoE GEMV (scalar FP16 variants — NOT dp4a, kept as-is)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Fused gate+up MoE GEMV: computes both gate and up projections in a single
// kernel launch. blockIdx.y selects projection: 0=gate, 1=up.
// Saves one kernel launch per MoE layer (48 launches for Qwen3-Coder).
// ---------------------------------------------------------------------------

__global__ void gemv_q6k_moe_gate_up_fused_kernel(const uint8_t* __restrict__ gate_weights,
                                                  const uint8_t* __restrict__ up_weights,
                                                  const int32_t* __restrict__ expert_indices,
                                                  const half* __restrict__ x, half* __restrict__ y_gate,
                                                  half* __restrict__ y_up, int rows, int K,
                                                  size_t gate_stride_bytes, size_t up_stride_bytes,
                                                  int x_stride, int blocks_per_expert) {
    const int warps_per_block = blockDim.x / 32;
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;

    const int expert_slot = blockIdx.x / blocks_per_expert;
    const int local_block = blockIdx.x % blocks_per_expert;
    const int row = local_block * warps_per_block + warp_id;

    if (row >= rows)
        return;

    // blockIdx.y: 0 = gate, 1 = up
    const bool is_up = (blockIdx.y == 1);
    const uint8_t* packed = is_up ? up_weights : gate_weights;
    size_t stride = is_up ? up_stride_bytes : gate_stride_bytes;
    half* y = is_up ? y_up : y_gate;

    const int expert_id = expert_indices[expert_slot];
    const uint8_t* W = packed + (size_t)expert_id * stride;

    const int blocks_per_row = K / 256;
    const size_t row_bytes = (size_t)blocks_per_row * 210;
    const uint8_t* W_row = W + (size_t)row * row_bytes;

    const half* x_ptr = x + expert_slot * x_stride;
    float sum = 0.0f;

    for (int b = 0; b < blocks_per_row; ++b) {
        const uint8_t* bp = W_row + b * 210;
        const uint8_t* ql = bp;
        const uint8_t* qh = bp + 128;
        const int8_t* sc = (const int8_t*)(bp + 192);
        float d = __half2float(*(const half*)(bp + 208));
        const int base = b * 256;

        uint8_t ql_a = ql[lane];
        uint8_t ql_b = ql[lane + 32];
        uint8_t ql_c = ql[64 + lane];
        uint8_t ql_d = ql[64 + lane + 32];
        uint8_t qh0 = qh[lane];
        uint8_t qh1 = qh[32 + lane];

        int q0 = (int)(((qh0 & 0x03) << 4) | (ql_a & 0x0F)) - 32;
        int q1 = (int)((((qh0 >> 2) & 0x03) << 4) | (ql_b & 0x0F)) - 32;
        int q2 = (int)((((qh0 >> 4) & 0x03) << 4) | ((ql_a >> 4) & 0x0F)) - 32;
        int q3 = (int)((((qh0 >> 6) & 0x03) << 4) | ((ql_b >> 4) & 0x0F)) - 32;
        int q4 = (int)(((qh1 & 0x03) << 4) | (ql_c & 0x0F)) - 32;
        int q5 = (int)((((qh1 >> 2) & 0x03) << 4) | (ql_d & 0x0F)) - 32;
        int q6 = (int)((((qh1 >> 4) & 0x03) << 4) | ((ql_c >> 4) & 0x0F)) - 32;
        int q7 = (int)((((qh1 >> 6) & 0x03) << 4) | ((ql_d >> 4) & 0x0F)) - 32;

        int sc_idx = lane >> 4;
        sum += d * ((float)sc[sc_idx] * (float)q0 * __half2float(x_ptr[base + lane]) +
                    (float)sc[sc_idx + 2] * (float)q1 * __half2float(x_ptr[base + lane + 32]) +
                    (float)sc[sc_idx + 4] * (float)q2 * __half2float(x_ptr[base + lane + 64]) +
                    (float)sc[sc_idx + 6] * (float)q3 * __half2float(x_ptr[base + lane + 96]) +
                    (float)sc[sc_idx + 8] * (float)q4 * __half2float(x_ptr[base + lane + 128]) +
                    (float)sc[sc_idx + 10] * (float)q5 * __half2float(x_ptr[base + lane + 160]) +
                    (float)sc[sc_idx + 12] * (float)q6 * __half2float(x_ptr[base + lane + 192]) +
                    (float)sc[sc_idx + 14] * (float)q7 * __half2float(x_ptr[base + lane + 224]));
    }

    sum = warp_reduce_sum(sum);

    if (lane == 0)
        y[expert_slot * rows + row] = __float2half(sum);
}

void gemv_q6k_moe_gate_up_fused(const void* gate_weights, const void* up_weights,
                                const int32_t* expert_indices, const half* x, half* y_gate, half* y_up,
                                int rows, int K, size_t gate_stride_bytes, size_t up_stride_bytes,
                                int x_stride, int top_k, cudaStream_t stream) {
    const int blocks_per_expert = gemv_blocks(rows);
    dim3 grid(top_k * blocks_per_expert, 2);
    gemv_q6k_moe_gate_up_fused_kernel<<<grid, kGemvThreads, 0, stream>>>(
        static_cast<const uint8_t*>(gate_weights), static_cast<const uint8_t*>(up_weights), expert_indices, x,
        y_gate, y_up, rows, K, gate_stride_bytes, up_stride_bytes, x_stride, blocks_per_expert);
    IMP_CUDA_CHECK_LAUNCH();
}

__global__ void gemv_q8_0_moe_gate_up_fused_kernel(const uint8_t* __restrict__ gate_weights,
                                                   const uint8_t* __restrict__ up_weights,
                                                   const int32_t* __restrict__ expert_indices,
                                                   const half* __restrict__ x, half* __restrict__ y_gate,
                                                   half* __restrict__ y_up, int rows, int K,
                                                   size_t gate_stride_bytes, size_t up_stride_bytes,
                                                   int x_stride, int blocks_per_expert) {
    const int warps_per_block = blockDim.x / 32;
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;

    const int expert_slot = blockIdx.x / blocks_per_expert;
    const int local_block = blockIdx.x % blocks_per_expert;
    const int row = local_block * warps_per_block + warp_id;

    if (row >= rows)
        return;

    const bool is_up = (blockIdx.y == 1);
    const uint8_t* packed = is_up ? up_weights : gate_weights;
    size_t stride = is_up ? up_stride_bytes : gate_stride_bytes;
    half* y = is_up ? y_up : y_gate;

    const int expert_id = expert_indices[expert_slot];
    const uint8_t* W = packed + (size_t)expert_id * stride;

    const int blocks_per_row = K / 32;
    const size_t row_bytes = (size_t)blocks_per_row * 34;
    const uint8_t* W_row = W + (size_t)row * row_bytes;

    const half* x_ptr = x + expert_slot * x_stride;
    float sum = 0.0f;

    for (int b = 0; b < blocks_per_row; ++b) {
        const uint8_t* bp = W_row + b * 34;
        float d = __half2float(*(const half*)bp);
        int8_t q = ((const int8_t*)(bp + 2))[lane];
        sum += d * (float)q * __half2float(x_ptr[b * 32 + lane]);
    }

    sum = warp_reduce_sum(sum);

    if (lane == 0)
        y[expert_slot * rows + row] = __float2half(sum);
}

void gemv_q8_0_moe_gate_up_fused(const void* gate_weights, const void* up_weights,
                                 const int32_t* expert_indices, const half* x, half* y_gate, half* y_up,
                                 int rows, int K, size_t gate_stride_bytes, size_t up_stride_bytes,
                                 int x_stride, int top_k, cudaStream_t stream) {
    const int blocks_per_expert = gemv_blocks(rows);
    dim3 grid(top_k * blocks_per_expert, 2);
    gemv_q8_0_moe_gate_up_fused_kernel<<<grid, kGemvThreads, 0, stream>>>(
        static_cast<const uint8_t*>(gate_weights), static_cast<const uint8_t*>(up_weights), expert_indices, x,
        y_gate, y_up, rows, K, gate_stride_bytes, up_stride_bytes, x_stride, blocks_per_expert);
    IMP_CUDA_CHECK_LAUNCH();
}

}  // namespace imp
