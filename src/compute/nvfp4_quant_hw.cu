// =============================================================================
// nvfp4_quant_hw.cu -- NVFP4 quantization with HW MMA scale layout
// =============================================================================
//
// Adapted from thu-ml/SageAttention3 (Apache-2.0 License),
// sageattention3_blackwell/sageattn3/quantization/fp4_quantization_4d.cu.
// Copyright (c) 2025 SageAttention team.
//
// Modifications from the original:
//   - Stripped the `permute` path (imp's attention doesn't need it at the
//     quant step — permutation happens inside the FMHA tile loop).
//   - Adapted to imp's coding style, logging, and namespace.
//   - Simplified dispatch: only head_dim ∈ {64, 128} supported.
//   - Added a matching dequant kernel that inverts the HW layout for
//     round-trip validation.
//
// Key PTX instruction:
//   cvt.rn.satfinite.e2m1x2.f32 byte, f32_hi, f32_lo
//
// Scale layout (critical for MMA consumption) — from lines 245-256 of the
// upstream file. For CVT_FP4_ELTS_PER_THREAD=16 (head_dim=128):
//   offset_local = (col_id_local / 4) * 256
//                + (col_id_local % 4)
//                + (token_id_local / 16) * 4
//                + (token_id_local % 16) * 16
//   where col_id_local = 0..7  (scale groups along K dim)
//         token_id_local = 0..63 (row within the current 64-token block)
//
// For CVT_FP4_ELTS_PER_THREAD=8 (head_dim=64):
//   Only even threadIdx writes scale, after cross-lane max combine.
//
// =============================================================================

#include "compute/nvfp4_quant_hw.h"
#include "core/logging.h"
#include <cuda_fp8.h>
#include <cstdint>

namespace imp {

constexpr int CVT_FP4_ELTS_PER_THREAD = 16;

// ---------------------------------------------------------------------------
// FP32→E2M1 packed conversion (4 float2 → uint32 holding 8 E2M1 nibbles).
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint32_t fp32x8_to_e2m1x8_hw(const float2* v) {
    uint32_t out;
    asm volatile(
        "{\n"
        ".reg .b8 b0, b1, b2, b3;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b0, %2, %1;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b1, %4, %3;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b2, %6, %5;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b3, %8, %7;\n"
        "mov.b32 %0, {b0, b1, b2, b3};\n"
        "}"
        : "=r"(out)
        : "f"(v[0].x), "f"(v[0].y), "f"(v[1].x), "f"(v[1].y), "f"(v[2].x), "f"(v[2].y), "f"(v[3].x),
          "f"(v[3].y));
    return out;
}

// ---------------------------------------------------------------------------
// E2M1 (4-bit) → FP32 LUT (for dequant reference).
// E2M1 encoding (sign + 3-bit mag): ±{0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float e2m1_to_fp32_hw(uint8_t nib) {
    static const float mags[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    float m = mags[nib & 0x7];
    return (nib & 0x8) ? -m : m;
}

// ---------------------------------------------------------------------------
// HW scale offset for CVT_FP4_ELTS_PER_THREAD=16 (head_dim=128).
// Formula from upstream fp4_quantization_4d.cu:245-249.
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint32_t
hw_scale_offset_hd128(uint32_t col_id_local,    // 0..7 (scale group along K)
                      uint32_t token_id_local)  // 0..63 (row in current 64-token block)
{
    return (col_id_local / 4) * 256 + (col_id_local % 4) + (token_id_local / 16) * 4 +
           (token_id_local % 16) * 16;
}

// ---------------------------------------------------------------------------
// Vector type
// ---------------------------------------------------------------------------
template <typename T>
struct PackedVec16 {
    // For T=half, each slot holds 2 elements (half2). 16 elems = 8 slots.
    typename std::conditional<std::is_same<T, half>::value, half2, half2>::type elts[8];
};

// ---------------------------------------------------------------------------
// Quant kernel: each thread handles 16 elements (one scale group).
// Grid layout:
//   blockIdx.x = token_block (covers BLOCK_SIZE tokens; BLOCK_SIZE = 64)
//   blockIdx.y = batch
//   blockIdx.z = head
//   threadIdx.x = (token_within_block * NUM_THREADS_PER_TOKEN) + col_scale_group
// ---------------------------------------------------------------------------
template <uint32_t HEAD_DIM, uint32_t BLOCK_SIZE>
__global__ void nvfp4_quant_hw_kernel(const half* __restrict__ input, uint8_t* __restrict__ nvfp4_out,
                                      uint8_t* __restrict__ sf_out, int batch_size, int n_heads, int n_tokens,
                                      int stride_bz_input, int stride_h_input, int stride_seq_input,
                                      int stride_bz_output, int stride_h_output, int stride_seq_output,
                                      int stride_bz_output_sf, int stride_h_output_sf,
                                      int stride_seq_output_sf) {
    constexpr uint32_t NUM_THREADS_PER_TOKEN = HEAD_DIM / CVT_FP4_ELTS_PER_THREAD;
    // head_dim=128 → 8 threads/token, head_dim=64 → 4 threads/token
    static_assert(HEAD_DIM == 64 || HEAD_DIM == 128, "Only 64 and 128 supported");

    const int batch_id = blockIdx.y;
    const int head_id = blockIdx.z;
    const int token_block_id = blockIdx.x;

    const int token_id = token_block_id * BLOCK_SIZE + threadIdx.x / NUM_THREADS_PER_TOKEN;

    // Load 16 FP16 elements assigned to this thread.
    PackedVec16<half> in_vec;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        reinterpret_cast<uint32_t&>(in_vec.elts[i]) = 0u;
    }

    if (token_id < n_tokens) {
        const half* src = input + batch_id * stride_bz_input + head_id * stride_h_input +
                          token_id * stride_seq_input +
                          (threadIdx.x % NUM_THREADS_PER_TOKEN) * CVT_FP4_ELTS_PER_THREAD;
        in_vec = *reinterpret_cast<const PackedVec16<half>*>(src);
    }

    // Max-abs across 16 elements.
    auto local_max = __habs2(in_vec.elts[0]);
#pragma unroll
    for (int i = 1; i < 8; ++i) {
        local_max = __hmax2(local_max, __habs2(in_vec.elts[i]));
    }
    // For 8-elems-per-thread (HEAD_DIM=64), combine adjacent threads.
    if constexpr (CVT_FP4_ELTS_PER_THREAD == 8) {
        local_max = __hmax2(__shfl_xor_sync(0xffffffffu, local_max, 1, 32), local_max);
    }
    float vec_max = float(__hmax(local_max.x, local_max.y));

    // Scale = max / 6.0 (E2M1 max = 6.0), round-trip through FP8 UE4M3.
    float sc = vec_max / 6.0f;
    uint8_t sc_fp8;
    reinterpret_cast<__nv_fp8_e4m3&>(sc_fp8) = __nv_fp8_e4m3(sc);
    sc = float(reinterpret_cast<__nv_fp8_e4m3&>(sc_fp8));
    float inv_sc = (sc == 0.0f) ? 0.0f : 1.0f / sc;

    // Apply inverse scale → FP32 pairs.
    float2 fp2[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        fp2[i] = __half22float2(in_vec.elts[i]);
        fp2[i].x *= inv_sc;
        fp2[i].y *= inv_sc;
    }

    // Pack to 2× uint32 (16 E2M1 nibbles = 8 bytes).
    uint32_t e2m1_lo = fp32x8_to_e2m1x8_hw(&fp2[0]);
    uint32_t e2m1_hi = fp32x8_to_e2m1x8_hw(&fp2[4]);

    // Store NVFP4 bytes (contiguous in output row).
    uint8_t* dst = nvfp4_out + batch_id * stride_bz_output + head_id * stride_h_output +
                   token_id * stride_seq_output +
                   (threadIdx.x % NUM_THREADS_PER_TOKEN) * CVT_FP4_ELTS_PER_THREAD / 2;

    if (token_id < n_tokens) {
        reinterpret_cast<uint64_t*>(dst)[0] = (static_cast<uint64_t>(e2m1_hi) << 32) | e2m1_lo;
    }

    // Store scale at HW offset.
    uint8_t* sf_base = sf_out + batch_id * stride_bz_output_sf + head_id * stride_h_output_sf +
                       (token_id / 64) * 64 * stride_seq_output_sf;
    uint32_t token_id_local = token_id % 64;

    if constexpr (CVT_FP4_ELTS_PER_THREAD == 16) {
        uint32_t col_id_local = threadIdx.x % NUM_THREADS_PER_TOKEN;
        uint32_t offset_local = hw_scale_offset_hd128(col_id_local, token_id_local);
        if (token_id < n_tokens)
            sf_base[offset_local] = sc_fp8;
    } else {
        // head_dim=64 case: only even threads write (scale is shared with odd).
        if ((threadIdx.x % 2) == 0) {
            uint32_t col_id_local = (threadIdx.x % NUM_THREADS_PER_TOKEN) / 2;
            uint32_t offset_local = (col_id_local / 4) * 256 + (col_id_local % 4) +
                                    (token_id_local / 16) * 4 + (token_id_local % 16) * 16;
            if (token_id < n_tokens)
                sf_base[offset_local] = sc_fp8;
        }
    }
}

// ---------------------------------------------------------------------------
// Dequant kernel: inverse of the above for round-trip validation.
// Same grid: one thread per 16-element group.
// ---------------------------------------------------------------------------
template <uint32_t HEAD_DIM, uint32_t BLOCK_SIZE>
__global__ void nvfp4_dequant_hw_kernel(const uint8_t* __restrict__ nvfp4_in,
                                        const uint8_t* __restrict__ sf_in, half* __restrict__ output,
                                        int batch_size, int n_heads, int n_tokens, int stride_bz_input,
                                        int stride_h_input, int stride_seq_input, int stride_bz_output,
                                        int stride_h_output, int stride_seq_output, int stride_bz_input_sf,
                                        int stride_h_input_sf, int stride_seq_input_sf) {
    constexpr uint32_t NUM_THREADS_PER_TOKEN = HEAD_DIM / CVT_FP4_ELTS_PER_THREAD;
    static_assert(HEAD_DIM == 64 || HEAD_DIM == 128, "Only 64 and 128 supported");

    const int batch_id = blockIdx.y;
    const int head_id = blockIdx.z;
    const int token_block_id = blockIdx.x;
    const int token_id = token_block_id * BLOCK_SIZE + threadIdx.x / NUM_THREADS_PER_TOKEN;

    if (token_id >= n_tokens)
        return;

    // Load scale from HW offset.
    const uint8_t* sf_base = sf_in + batch_id * stride_bz_input_sf + head_id * stride_h_input_sf +
                             (token_id / 64) * 64 * stride_seq_input_sf;
    uint32_t token_id_local = token_id % 64;
    uint32_t col_id_local = threadIdx.x % NUM_THREADS_PER_TOKEN;
    uint32_t offset_local;
    if constexpr (CVT_FP4_ELTS_PER_THREAD == 16) {
        offset_local = hw_scale_offset_hd128(col_id_local, token_id_local);
    } else {
        uint32_t col_half = col_id_local / 2;
        offset_local = (col_half / 4) * 256 + (col_half % 4) + (token_id_local / 16) * 4 +
                       (token_id_local % 16) * 16;
    }
    uint8_t sc_fp8 = sf_base[offset_local];
    float sc = float(reinterpret_cast<const __nv_fp8_e4m3&>(sc_fp8));

    // Load 8 NVFP4 bytes (16 nibbles).
    const uint8_t* src = nvfp4_in + batch_id * stride_bz_input + head_id * stride_h_input +
                         token_id * stride_seq_input +
                         (threadIdx.x % NUM_THREADS_PER_TOKEN) * CVT_FP4_ELTS_PER_THREAD / 2;

    half* dst = output + batch_id * stride_bz_output + head_id * stride_h_output +
                token_id * stride_seq_output +
                (threadIdx.x % NUM_THREADS_PER_TOKEN) * CVT_FP4_ELTS_PER_THREAD;

#pragma unroll
    for (int i = 0; i < 16; ++i) {
        int byte_idx = i / 2;
        uint8_t byte = src[byte_idx];
        uint8_t nib = (i & 1) ? (byte >> 4) : (byte & 0xF);
        float v = e2m1_to_fp32_hw(nib) * sc;
        dst[i] = __float2half(v);
    }
}

// ---------------------------------------------------------------------------
// Host entry points
// ---------------------------------------------------------------------------
bool nvfp4_quant_hw_fp16(const half* d_input, uint8_t* d_nvfp4, uint8_t* d_sf, int batch_size, int n_heads,
                         int n_tokens, int head_dim, int stride_bz_input, int stride_h_input,
                         int stride_seq_input, int stride_bz_output, int stride_h_output,
                         int stride_seq_output, int stride_bz_output_sf, int stride_h_output_sf,
                         int stride_seq_output_sf, cudaStream_t stream) {
    constexpr int BLOCK_SIZE = 64;  // tokens per threadblock
    if (head_dim != 64 && head_dim != 128) {
        IMP_LOG_ERROR("nvfp4_quant_hw: head_dim %d unsupported (only 64 / 128)", head_dim);
        return false;
    }
    if (n_tokens <= 0 || batch_size <= 0 || n_heads <= 0) {
        IMP_LOG_ERROR("nvfp4_quant_hw: invalid dims batch=%d heads=%d tokens=%d", batch_size, n_heads,
                      n_tokens);
        return false;
    }

    dim3 grid((n_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE, batch_size, n_heads);
    int threads_per_token = head_dim / CVT_FP4_ELTS_PER_THREAD;
    dim3 block(BLOCK_SIZE * threads_per_token, 1, 1);

    if (head_dim == 128) {
        nvfp4_quant_hw_kernel<128, BLOCK_SIZE>
            <<<grid, block, 0, stream>>>(d_input, d_nvfp4, d_sf, batch_size, n_heads, n_tokens,
                                         stride_bz_input, stride_h_input, stride_seq_input, stride_bz_output,
                                         stride_h_output, stride_seq_output, stride_bz_output_sf,
                                         stride_h_output_sf, stride_seq_output_sf);
    } else {
        nvfp4_quant_hw_kernel<64, BLOCK_SIZE>
            <<<grid, block, 0, stream>>>(d_input, d_nvfp4, d_sf, batch_size, n_heads, n_tokens,
                                         stride_bz_input, stride_h_input, stride_seq_input, stride_bz_output,
                                         stride_h_output, stride_seq_output, stride_bz_output_sf,
                                         stride_h_output_sf, stride_seq_output_sf);
    }
    return cudaGetLastError() == cudaSuccess;
}

bool nvfp4_dequant_hw_fp16(const uint8_t* d_nvfp4, const uint8_t* d_sf, half* d_output, int batch_size,
                           int n_heads, int n_tokens, int head_dim, int stride_bz_input, int stride_h_input,
                           int stride_seq_input, int stride_bz_output, int stride_h_output,
                           int stride_seq_output, int stride_bz_input_sf, int stride_h_input_sf,
                           int stride_seq_input_sf, cudaStream_t stream) {
    constexpr int BLOCK_SIZE = 64;
    if (head_dim != 64 && head_dim != 128)
        return false;
    if (n_tokens <= 0 || batch_size <= 0 || n_heads <= 0)
        return false;

    dim3 grid((n_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE, batch_size, n_heads);
    int threads_per_token = head_dim / CVT_FP4_ELTS_PER_THREAD;
    dim3 block(BLOCK_SIZE * threads_per_token, 1, 1);

    if (head_dim == 128) {
        nvfp4_dequant_hw_kernel<128, BLOCK_SIZE>
            <<<grid, block, 0, stream>>>(d_nvfp4, d_sf, d_output, batch_size, n_heads, n_tokens,
                                         stride_bz_input, stride_h_input, stride_seq_input, stride_bz_output,
                                         stride_h_output, stride_seq_output, stride_bz_input_sf,
                                         stride_h_input_sf, stride_seq_input_sf);
    } else {
        nvfp4_dequant_hw_kernel<64, BLOCK_SIZE>
            <<<grid, block, 0, stream>>>(d_nvfp4, d_sf, d_output, batch_size, n_heads, n_tokens,
                                         stride_bz_input, stride_h_input, stride_seq_input, stride_bz_output,
                                         stride_h_output, stride_seq_output, stride_bz_input_sf,
                                         stride_h_input_sf, stride_seq_input_sf);
    }
    return cudaGetLastError() == cudaSuccess;
}

}  // namespace imp
