// Fused Q4_K/Q5_K × Q8_1 dp4a GEMM kernels for MoE expert prefill AND dense prefill.
// Weight-stationary, Q8_1 activations in shared memory, dp4a integer accumulation.
// Eliminates the FP16 dequant intermediate that caused 8.3× bandwidth overhead.
//
// Based on the proven gemm_q6k_moe_fused architecture: each CTA loads a tile of
// Q8_1 activations into shared memory, then each warp computes one output column
// across the M_TILE tokens using dp4a dot products with the quantized weights.
//
// Q4_K difference from Q6_K: unsigned nibbles (0-15) with per-sub-block min offset.
// The min correction uses dp4a(ones, q8, 0) to compute sum(q8) inline — no need
// for the Q8_1 sum field (block_q8_1::s).

#include "compute/gemm_q4k.h"
#include "compute/gemm.h"
#include "core/logging.h"

#include <cuda_fp16.h>
#include <cstdint>

namespace imp {

enum class QKType { Q4_K, Q5_K };

template <QKType BT>
struct BlockTraits;

template <>
struct BlockTraits<QKType::Q4_K> {
    static constexpr int BYTES = 144;
    static constexpr int SCALES_OFFSET = 4;
    static constexpr int QH_OFFSET = 0;
    static constexpr int QS_OFFSET = 16;
};

template <>
struct BlockTraits<QKType::Q5_K> {
    static constexpr int BYTES = 176;
    static constexpr int SCALES_OFFSET = 4;
    static constexpr int QH_OFFSET = 16;
    static constexpr int QS_OFFSET = 48;
};

static constexpr int BLOCK_ELEMS = 256;
static constexpr int WARPS_PER_CTA = 8;
static constexpr int CTA_THREADS = WARPS_PER_CTA * 32;
static constexpr int TILE_M = 32;

__device__ __forceinline__ void get_scale_min_q4k(const uint8_t* sc, int j,
                                                   uint8_t& sc_val, uint8_t& min_val) {
    if (j < 4) {
        sc_val = sc[j] & 63;
        min_val = sc[j + 4] & 63;
    } else {
        sc_val = (sc[j + 4] & 0xF) | ((sc[j - 4] >> 6) << 4);
        min_val = (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4);
    }
}

// Each Q4_K super-block (256 elements) = 8 Q8_1 blocks.
// Each Q8_1 block covers 32 elements = one sub-block of Q4_K.
// Sub-blocks 0,2,4,6 are low nibbles of qs chunks. 1,3,5,7 are high nibbles.
//
// For Q8_1 block index g (0..7) within a Q4_K super-block:
//   chunk = g / 2  (which 64-element group, 0..3)
//   is_high = g & 1 (low vs high nibbles)
//   qs starts at chunk * 32 bytes
//   If is_high: use (qs[i] >> 4) & 0xF. Else: use qs[i] & 0xF.
//   For Q5_K: additionally, qh bit at position (2*chunk + is_high) gives the 5th bit.

template <QKType BT>
__global__ void __launch_bounds__(CTA_THREADS, 1)
gemm_qk_dp4a_moe_fused_kernel(
    const uint8_t* __restrict__ packed_weight,
    const block_q8_1* __restrict__ q8_base,
    const float* __restrict__ d8_base,
    half* __restrict__ c_base,
    const int32_t* __restrict__ offsets,
    int K, int N, size_t weight_stride,
    int q8_per_row) {

    using Traits = BlockTraits<BT>;

    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int tid = threadIdx.x;
    const int expert = blockIdx.y;

    const int m_start = offsets[expert];
    const int M_e = offsets[expert + 1] - m_start;
    if (M_e <= 0)
        return;

    const int n_col = blockIdx.x * WARPS_PER_CTA + warp_id;
    if (n_col >= N)
        return;

    const int blocks_per_row = K / BLOCK_ELEMS;
    const size_t row_bytes = static_cast<size_t>(blocks_per_row) * Traits::BYTES;
    const uint8_t* w_row = packed_weight + static_cast<size_t>(expert) * weight_stride +
                           static_cast<size_t>(n_col) * row_bytes;

    extern __shared__ char smem_raw[];
    int8_t* smem_qs = reinterpret_cast<int8_t*>(smem_raw);
    float* smem_d8 = reinterpret_cast<float*>(smem_raw + TILE_M * q8_per_row * 32);

    for (int m_base = 0; m_base < M_e; m_base += TILE_M) {
        const int m_count = min(TILE_M, M_e - m_base);

        // Phase 1: Load Q8_1 data into shared memory (all threads cooperate)
        {
            const int total_items = m_count * q8_per_row;
            for (int i = tid; i < total_items; i += CTA_THREADS) {
                const int mi = i / q8_per_row;
                const int qi = i % q8_per_row;
                const int tok = m_start + m_base + mi;

                const block_q8_1& src = q8_base[tok * q8_per_row + qi];
                int4* dst_qs = reinterpret_cast<int4*>(smem_qs + (mi * q8_per_row + qi) * 32);
                int4 tmp0, tmp1;
                memcpy(&tmp0, src.qs, 16);
                memcpy(&tmp1, src.qs + 16, 16);
                dst_qs[0] = tmp0;
                dst_qs[1] = tmp1;

                smem_d8[mi * q8_per_row + qi] = d8_base[tok * q8_per_row + qi];
            }
        }
        __syncthreads();

        // Phase 2: dp4a accumulation
        float acc[TILE_M];
        for (int i = 0; i < TILE_M; i++)
            acc[i] = 0.0f;

        // K-loop: each lane handles one Q8_1 block per stride
        for (int q8_idx = lane; q8_idx < q8_per_row; q8_idx += 32) {
            const int super_blk = q8_idx / 8;
            const int g = q8_idx % 8;
            const int chunk = g / 2;
            const int is_high = g & 1;
            const int sub_block = g;

            const uint8_t* bp = w_row + static_cast<size_t>(super_blk) * Traits::BYTES;

            const float d_w = __half2float(*reinterpret_cast<const half*>(bp));
            const float dmin_w = __half2float(*reinterpret_cast<const half*>(bp + 2));

            uint8_t sc_val, min_val;
            get_scale_min_q4k(bp + Traits::SCALES_OFFSET, sub_block, sc_val, min_val);

            const float d_sc = d_w * static_cast<float>(sc_val);
            const float dmin_mn = dmin_w * static_cast<float>(min_val);

            // Load 32 bytes of Q4_K nibbles (= 32 elements for this sub-block)
            // Low nibbles: qs[chunk*32..chunk*32+31] & 0xF
            // High nibbles: qs[chunk*32..chunk*32+31] >> 4
            const uint8_t* qs_ptr = bp + Traits::QS_OFFSET + chunk * 32;
            uint32_t q4_packed[8];
            memcpy(q4_packed, qs_ptr, 32);

            // For Q5_K: load the qh byte for each element
            [[maybe_unused]] uint32_t qh_packed[8] = {};
            [[maybe_unused]] int qh_shift = 0;
            if constexpr (BT == QKType::Q5_K) {
                memcpy(qh_packed, bp + Traits::QH_OFFSET, 32);
                qh_shift = 2 * chunk + is_high;
            }

            // Extract 8 packed int32s of nibble values (4 nibbles per int32)
            uint32_t nib[8];
#pragma unroll
            for (int d4 = 0; d4 < 8; d4++) {
                if constexpr (BT == QKType::Q4_K) {
                    nib[d4] = is_high ? ((q4_packed[d4] >> 4) & 0x0F0F0F0Fu)
                                      : (q4_packed[d4] & 0x0F0F0F0Fu);
                } else {
                    uint32_t lo4 = is_high ? ((q4_packed[d4] >> 4) & 0x0F0F0F0Fu)
                                           : (q4_packed[d4] & 0x0F0F0F0Fu);
                    // Extract 5th bit from qh for each of the 4 bytes
                    uint32_t qh4 = qh_packed[d4];
                    uint32_t hi1_0 = ((qh4 >> (qh_shift + 0)) & 0x01u) << 4;
                    uint32_t hi1_1 = ((qh4 >> (qh_shift + 8)) & 0x01u) << 12;
                    uint32_t hi1_2 = ((qh4 >> (qh_shift + 16)) & 0x01u) << 20;
                    uint32_t hi1_3 = ((qh4 >> (qh_shift + 24)) & 0x01u) << 28;
                    nib[d4] = lo4 | hi1_0 | hi1_1 | hi1_2 | hi1_3;
                }
            }

            // M-loop
            for (int mi = 0; mi < m_count; mi++) {
                const int8_t* qs_act = smem_qs + (mi * q8_per_row + q8_idx) * 32;
                int4* qs_v = reinterpret_cast<int4*>(const_cast<int8_t*>(qs_act));
                int4 v0 = qs_v[0];
                int4 v1 = qs_v[1];
                int xqs[8];
                memcpy(&xqs[0], &v0, 16);
                memcpy(&xqs[4], &v1, 16);

                const float dq = smem_d8[mi * q8_per_row + q8_idx];

                int32_t sumi = 0;
                int32_t sum_ones = 0;
                constexpr int ones = 0x01010101;
#pragma unroll
                for (int d4 = 0; d4 < 8; d4++) {
                    int ni;
                    memcpy(&ni, &nib[d4], 4);
                    sumi = __dp4a(ni, xqs[d4], sumi);
                    sum_ones = __dp4a(ones, xqs[d4], sum_ones);
                }

                acc[mi] += dq * (d_sc * static_cast<float>(sumi) -
                                 dmin_mn * static_cast<float>(sum_ones));
            }
        }

        // Phase 3: Warp reduction + output
        for (int mi = 0; mi < m_count; mi++) {
#pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                acc[mi] += __shfl_down_sync(0xFFFFFFFF, acc[mi], off);
        }

        if (lane == 0) {
            for (int mi = 0; mi < m_count; mi++) {
                const int tok = m_start + m_base + mi;
                c_base[static_cast<size_t>(tok) * N + n_col] = __float2half(acc[mi]);
            }
        }

        __syncthreads();
    }
}

// Scalar FP16 fallback (for benchmarking / small M_e)
static constexpr int SCALAR_WARPS = 8;
static constexpr int SCALAR_BLOCK = SCALAR_WARPS * 32;
static constexpr int SCALAR_M_TILE = 8;

template <QKType BT>
__global__ void __launch_bounds__(SCALAR_BLOCK)
gemm_qk_scalar_moe_prefill_kernel(
    const uint8_t* __restrict__ packed_weights,
    const half* __restrict__ activations,
    half* __restrict__ output,
    const int32_t* __restrict__ offsets,
    int N, int K, size_t expert_stride_bytes, int n_experts) {

    using Traits = BlockTraits<BT>;

    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;

    const int row = blockIdx.x * SCALAR_WARPS + warp_id;
    const int expert_id = blockIdx.y;

    if (row >= N || expert_id >= n_experts)
        return;

    const int start = offsets[expert_id];
    const int M = offsets[expert_id + 1] - start;
    if (M == 0)
        return;

    const int blocks_per_row = K / BLOCK_ELEMS;
    const size_t row_bytes = static_cast<size_t>(blocks_per_row) * Traits::BYTES;
    const uint8_t* W_row = packed_weights + static_cast<size_t>(expert_id) * expert_stride_bytes +
                           static_cast<size_t>(row) * row_bytes;

    const int chunk = lane / 8;
    const int is_high = (lane % 8) >= 4;
    const int sub_block = chunk * 2 + is_high;
    const int qs_byte_base = chunk * 32 + (lane % 4) * 8;
    const int k_lane_offset = lane * 8;

    for (int m_base = 0; m_base < M; m_base += SCALAR_M_TILE) {
        const int M_cur = min(SCALAR_M_TILE, M - m_base);

        float acc[SCALAR_M_TILE];
#pragma unroll
        for (int i = 0; i < SCALAR_M_TILE; i++)
            acc[i] = 0.0f;

        for (int blk = 0; blk < blocks_per_row; blk++) {
            const uint8_t* bp = W_row + static_cast<size_t>(blk) * Traits::BYTES;

            const float d = __half2float(*reinterpret_cast<const half*>(bp));
            const float dmin = __half2float(*reinterpret_cast<const half*>(bp + 2));

            uint8_t sc_val, min_val;
            get_scale_min_q4k(bp + Traits::SCALES_OFFSET, sub_block, sc_val, min_val);

            const float w_d = d * static_cast<float>(sc_val);
            const float w_m = dmin * static_cast<float>(min_val);

            const uint8_t* qs = bp + Traits::QS_OFFSET + qs_byte_base;
            uint64_t qs8;
            memcpy(&qs8, qs, 8);

            [[maybe_unused]] uint64_t qh8 = 0;
            [[maybe_unused]] int qh_shift = 0;
            if constexpr (BT == QKType::Q5_K) {
                const uint8_t* qh_base = bp + Traits::QH_OFFSET + (lane % 4) * 8;
                memcpy(&qh8, qh_base, 8);
                qh_shift = 2 * chunk + is_high;
            }

            float w[8];
#pragma unroll
            for (int i = 0; i < 8; i++) {
                const uint32_t byte = static_cast<uint32_t>((qs8 >> (i * 8)) & 0xFFu);
                int nibble;
                if constexpr (BT == QKType::Q4_K) {
                    nibble = is_high ? static_cast<int>(byte >> 4) : static_cast<int>(byte & 0xFu);
                } else {
                    int lo4 = is_high ? static_cast<int>(byte >> 4) : static_cast<int>(byte & 0xFu);
                    int hi1 = static_cast<int>((static_cast<uint32_t>((qh8 >> (i * 8)) & 0xFFu) >> qh_shift) & 1u);
                    nibble = lo4 | (hi1 << 4);
                }
                w[i] = w_d * static_cast<float>(nibble) - w_m;
            }

            const int k_offset = blk * BLOCK_ELEMS + k_lane_offset;

#pragma unroll
            for (int m = 0; m < SCALAR_M_TILE; m++) {
                if (m >= M_cur)
                    break;
                const int64_t token = start + m_base + m;
                const half* a_ptr = activations + token * K + k_offset;
                uint4 a_vec = *reinterpret_cast<const uint4*>(a_ptr);
                const half* ah = reinterpret_cast<const half*>(&a_vec);

                float dot = 0.0f;
#pragma unroll
                for (int i = 0; i < 8; i++)
                    dot += w[i] * __half2float(ah[i]);

                acc[m] += dot;
            }
        }

#pragma unroll
        for (int m = 0; m < SCALAR_M_TILE; m++) {
            if (m >= M_cur)
                break;
#pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                acc[m] += __shfl_down_sync(0xFFFFFFFF, acc[m], off);
        }

        if (lane == 0) {
#pragma unroll
            for (int m = 0; m < SCALAR_M_TILE; m++) {
                if (m >= M_cur)
                    break;
                output[static_cast<int64_t>(start + m_base + m) * N + row] = __float2half(acc[m]);
            }
        }
    }
}


// ---------------------------------------------------------------------------
// Dense (non-MoE) dp4a kernel: no expert offsets, grid over (N, 1).
// sm_120 caps shared memory at 99 KiB (101,376 B), so TILE_M is smaller
// than the MoE kernel to fit the Q8_1 tile within that budget.
// ---------------------------------------------------------------------------

static constexpr int DENSE_TILE_M = 16;

template <QKType BT>
__global__ void __launch_bounds__(CTA_THREADS, 1)
gemm_qk_dp4a_dense_kernel(
    const uint8_t* __restrict__ packed_weight,
    const block_q8_1* __restrict__ q8_base,
    const float* __restrict__ d8_base,
    half* __restrict__ output,
    int M, int K, int N,
    int q8_per_row) {

    using Traits = BlockTraits<BT>;

    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int tid = threadIdx.x;

    const int n_col = blockIdx.x * WARPS_PER_CTA + warp_id;
    if (n_col >= N)
        return;

    const int blocks_per_row = K / BLOCK_ELEMS;
    const size_t row_bytes = static_cast<size_t>(blocks_per_row) * Traits::BYTES;
    const uint8_t* w_row = packed_weight + static_cast<size_t>(n_col) * row_bytes;

    extern __shared__ char smem_raw[];
    int8_t* smem_qs = reinterpret_cast<int8_t*>(smem_raw);
    float* smem_d8 = reinterpret_cast<float*>(smem_raw + DENSE_TILE_M * q8_per_row * 32);

    for (int m_base = 0; m_base < M; m_base += DENSE_TILE_M) {
        const int m_count = min(DENSE_TILE_M, M - m_base);

        {
            const int total_items = m_count * q8_per_row;
            for (int i = tid; i < total_items; i += CTA_THREADS) {
                const int mi = i / q8_per_row;
                const int qi = i % q8_per_row;
                const int tok = m_base + mi;

                const block_q8_1& src = q8_base[tok * q8_per_row + qi];
                int4* dst_qs = reinterpret_cast<int4*>(smem_qs + (mi * q8_per_row + qi) * 32);
                int4 tmp0, tmp1;
                memcpy(&tmp0, src.qs, 16);
                memcpy(&tmp1, src.qs + 16, 16);
                dst_qs[0] = tmp0;
                dst_qs[1] = tmp1;

                smem_d8[mi * q8_per_row + qi] = d8_base[tok * q8_per_row + qi];
            }
        }
        __syncthreads();

        float acc[DENSE_TILE_M];
        for (int i = 0; i < DENSE_TILE_M; i++)
            acc[i] = 0.0f;

        for (int q8_idx = lane; q8_idx < q8_per_row; q8_idx += 32) {
            const int super_blk = q8_idx / 8;
            const int g = q8_idx % 8;
            const int chunk = g / 2;
            const int is_high = g & 1;
            const int sub_block = g;

            const uint8_t* bp = w_row + static_cast<size_t>(super_blk) * Traits::BYTES;

            const float d_w = __half2float(*reinterpret_cast<const half*>(bp));
            const float dmin_w = __half2float(*reinterpret_cast<const half*>(bp + 2));

            uint8_t sc_val, min_val;
            get_scale_min_q4k(bp + Traits::SCALES_OFFSET, sub_block, sc_val, min_val);

            const float d_sc = d_w * static_cast<float>(sc_val);
            const float dmin_mn = dmin_w * static_cast<float>(min_val);

            const uint8_t* qs_ptr = bp + Traits::QS_OFFSET + chunk * 32;
            uint32_t q4_packed[8];
            memcpy(q4_packed, qs_ptr, 32);

            [[maybe_unused]] uint32_t qh_packed[8] = {};
            [[maybe_unused]] int qh_shift = 0;
            if constexpr (BT == QKType::Q5_K) {
                memcpy(qh_packed, bp + Traits::QH_OFFSET, 32);
                qh_shift = 2 * chunk + is_high;
            }

            uint32_t nib[8];
#pragma unroll
            for (int d4 = 0; d4 < 8; d4++) {
                if constexpr (BT == QKType::Q4_K) {
                    nib[d4] = is_high ? ((q4_packed[d4] >> 4) & 0x0F0F0F0Fu)
                                      : (q4_packed[d4] & 0x0F0F0F0Fu);
                } else {
                    uint32_t lo4 = is_high ? ((q4_packed[d4] >> 4) & 0x0F0F0F0Fu)
                                           : (q4_packed[d4] & 0x0F0F0F0Fu);
                    uint32_t qh4 = qh_packed[d4];
                    uint32_t hi1_0 = ((qh4 >> (qh_shift + 0)) & 0x01u) << 4;
                    uint32_t hi1_1 = ((qh4 >> (qh_shift + 8)) & 0x01u) << 12;
                    uint32_t hi1_2 = ((qh4 >> (qh_shift + 16)) & 0x01u) << 20;
                    uint32_t hi1_3 = ((qh4 >> (qh_shift + 24)) & 0x01u) << 28;
                    nib[d4] = lo4 | hi1_0 | hi1_1 | hi1_2 | hi1_3;
                }
            }

            for (int mi = 0; mi < m_count; mi++) {
                const int8_t* qs_act = smem_qs + (mi * q8_per_row + q8_idx) * 32;
                int4* qs_v = reinterpret_cast<int4*>(const_cast<int8_t*>(qs_act));
                int4 v0 = qs_v[0];
                int4 v1 = qs_v[1];
                int xqs[8];
                memcpy(&xqs[0], &v0, 16);
                memcpy(&xqs[4], &v1, 16);

                const float dq = smem_d8[mi * q8_per_row + q8_idx];

                int32_t sumi = 0;
                int32_t sum_ones = 0;
                constexpr int ones = 0x01010101;
#pragma unroll
                for (int d4 = 0; d4 < 8; d4++) {
                    int ni;
                    memcpy(&ni, &nib[d4], 4);
                    sumi = __dp4a(ni, xqs[d4], sumi);
                    sum_ones = __dp4a(ones, xqs[d4], sum_ones);
                }

                acc[mi] += dq * (d_sc * static_cast<float>(sumi) -
                                 dmin_mn * static_cast<float>(sum_ones));
            }
        }

        for (int mi = 0; mi < m_count; mi++) {
#pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                acc[mi] += __shfl_down_sync(0xFFFFFFFF, acc[mi], off);
        }

        if (lane == 0) {
            for (int mi = 0; mi < m_count; mi++) {
                output[static_cast<size_t>(m_base + mi) * N + n_col] = __float2half(acc[mi]);
            }
        }

        __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// Host launchers — dense dp4a
// ---------------------------------------------------------------------------

template <QKType BT>
static void launch_dense_dp4a(const void* packed_weight, const block_q8_1* q8_base,
                               const float* d8_base, half* output,
                               int M, int N, int K, cudaStream_t stream) {
    if (M <= 0 || K <= 0 || N <= 0)
        return;

    const int q8_per_row = K / 32;
    const int n_col_blocks = (N + WARPS_PER_CTA - 1) / WARPS_PER_CTA;
    const dim3 grid(n_col_blocks);
    const dim3 block(CTA_THREADS);

    const size_t smem_qs_bytes = static_cast<size_t>(DENSE_TILE_M) * q8_per_row * 32;
    const size_t smem_d8_bytes = static_cast<size_t>(DENSE_TILE_M) * q8_per_row * sizeof(float);
    const size_t smem_bytes = smem_qs_bytes + smem_d8_bytes;

    static size_t smem_max_configured = 0;
    if (smem_bytes > smem_max_configured) {
        if (smem_bytes > 48 * 1024) {
            cudaFuncSetAttribute(gemm_qk_dp4a_dense_kernel<BT>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 static_cast<int>(smem_bytes));
        }
        smem_max_configured = smem_bytes;
    }

    gemm_qk_dp4a_dense_kernel<BT><<<grid, block, smem_bytes, stream>>>(
        static_cast<const uint8_t*>(packed_weight), q8_base, d8_base,
        output, M, K, N, q8_per_row);
    IMP_CUDA_CHECK_LAUNCH();
}

// ---------------------------------------------------------------------------
// Host launchers — MoE dp4a (Q8_1 activations in shared memory)
// ---------------------------------------------------------------------------

template <QKType BT>
static void launch_dp4a(const void* packed_weight, const block_q8_1* q8_base, const float* d8_base,
                        void* c_base, const int32_t* offsets, int K, int N, int n_experts,
                        size_t weight_stride, cudaStream_t stream) {
    if (n_experts <= 0 || K <= 0 || N <= 0)
        return;

    const int q8_per_row = K / 32;
    const int n_col_blocks = (N + WARPS_PER_CTA - 1) / WARPS_PER_CTA;
    const dim3 grid(n_col_blocks, n_experts);
    const dim3 block(CTA_THREADS);

    const size_t smem_qs_bytes = static_cast<size_t>(TILE_M) * q8_per_row * 32;
    const size_t smem_d8_bytes = static_cast<size_t>(TILE_M) * q8_per_row * sizeof(float);
    const size_t smem_bytes = smem_qs_bytes + smem_d8_bytes;

    static bool smem_configured = false;
    if (!smem_configured && smem_bytes > 48 * 1024) {
        cudaFuncSetAttribute(gemm_qk_dp4a_moe_fused_kernel<BT>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(smem_bytes));
        smem_configured = true;
    }

    gemm_qk_dp4a_moe_fused_kernel<BT><<<grid, block, smem_bytes, stream>>>(
        static_cast<const uint8_t*>(packed_weight), q8_base, d8_base,
        static_cast<half*>(c_base), offsets, K, N, weight_stride, q8_per_row);
    IMP_CUDA_CHECK_LAUNCH();
}

// ---------------------------------------------------------------------------
// Host launchers — scalar FP16 (used for benchmarking, small shapes)
// ---------------------------------------------------------------------------

template <QKType BT>
static void launch_scalar(const void* packed_weights, const void* activations, void* output,
                          const int32_t* d_offsets, int N, int K, size_t expert_stride_bytes,
                          int n_experts, cudaStream_t stream) {
    if (n_experts == 0)
        return;

    const int blocks_x = (N + SCALAR_WARPS - 1) / SCALAR_WARPS;
    dim3 grid(blocks_x, n_experts);
    dim3 block(SCALAR_BLOCK);

    gemm_qk_scalar_moe_prefill_kernel<BT><<<grid, block, 0, stream>>>(
        static_cast<const uint8_t*>(packed_weights),
        static_cast<const half*>(activations),
        static_cast<half*>(output),
        d_offsets, N, K, expert_stride_bytes, n_experts);
    IMP_CUDA_CHECK_LAUNCH();
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

void gemm_q4k_fused_moe_prefill(const void* packed_weights, const void* activations, void* output,
                                const int32_t* d_offsets, int N, int K, size_t expert_stride_bytes,
                                int n_experts, cudaStream_t stream) {
    launch_scalar<QKType::Q4_K>(packed_weights, activations, output, d_offsets, N, K,
                                expert_stride_bytes, n_experts, stream);
}

void gemm_q4k_dp4a_moe_fused(const void* packed_weight, const block_q8_1* q8_base,
                              const float* d8_base, void* c_base, const int32_t* offsets,
                              int K, int N, int n_experts, size_t weight_stride,
                              cudaStream_t stream) {
    launch_dp4a<QKType::Q4_K>(packed_weight, q8_base, d8_base, c_base, offsets, K, N,
                              n_experts, weight_stride, stream);
}

void gemm_q5k_dp4a_moe_fused(const void* packed_weight, const block_q8_1* q8_base,
                              const float* d8_base, void* c_base, const int32_t* offsets,
                              int K, int N, int n_experts, size_t weight_stride,
                              cudaStream_t stream) {
    launch_dp4a<QKType::Q5_K>(packed_weight, q8_base, d8_base, c_base, offsets, K, N,
                              n_experts, weight_stride, stream);
}

void gemm_q4k_dp4a_dense(const void* packed_q4k, const half* activations, half* output,
                          void* q8_scratch, float* d8_scratch,
                          int M, int N, int K, cudaStream_t stream) {
    auto* q8 = reinterpret_cast<block_q8_1*>(q8_scratch);
    quantize_fp16_to_q8_1(activations, q8, d8_scratch, M * K, stream);
    launch_dense_dp4a<QKType::Q4_K>(packed_q4k, q8, d8_scratch, output, M, N, K, stream);
}

void gemm_q5k_dp4a_dense(const void* packed_q5k, const half* activations, half* output,
                          void* q8_scratch, float* d8_scratch,
                          int M, int N, int K, cudaStream_t stream) {
    auto* q8 = reinterpret_cast<block_q8_1*>(q8_scratch);
    quantize_fp16_to_q8_1(activations, q8, d8_scratch, M * K, stream);
    launch_dense_dp4a<QKType::Q5_K>(packed_q5k, q8, d8_scratch, output, M, N, K, stream);
}

}  // namespace imp
