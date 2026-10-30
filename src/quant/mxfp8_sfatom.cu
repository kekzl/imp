// MXFP8 operand bytes in the SfAtom layout the own tile GEMM reads (gemm_mxfp8_tile): E4M3 [rows, K]
// plus UE8M0 per 32 K elements, 128 rows x 4 scales = 512 B per atom, atoms K-major. Quantizer, FP8
// row-scale view and the FP16 rebuild; the MXFP4 twin is quant/mxfp4_sfatom.cu.

#include "quant/mxfp8_sfatom.h"

#include "core/logging.h"

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>

namespace imp {

// SfAtom layout, SFVecSize 32: 128 rows x 4 K-groups per 512-byte atom, K tiles inner.
static constexpr int kMx8SFVecSize = 32;
static constexpr int kMx8AtomRows = 128;
static constexpr int kMx8AtomKGroups = 4;
static constexpr int kMx8AtomKElems = kMx8SFVecSize * kMx8AtomKGroups;  // 128
static constexpr int kMx8AtomSize = kMx8AtomRows * kMx8AtomKGroups;     // 512

__device__ __host__ __forceinline__ int mx8_sfatom_offset(int row, int k_group, int n_k_tiles) {
    const int tile_row = row / kMx8AtomRows;
    const int tile_k = k_group / kMx8AtomKGroups;
    const int row_local = row % kMx8AtomRows;
    const int k_local = k_group % kMx8AtomKGroups;
    const int n0 = row_local % 32;
    const int n1 = row_local / 32;
    return (((tile_row * n_k_tiles) + tile_k) * kMx8AtomSize) + (n0 * 16) + (n1 * 4) + k_local;
}

size_t cutlass_mxfp8_sf_size(int rows, int K) {
    const int n_row_tiles = (rows + kMx8AtomRows - 1) / kMx8AtomRows;
    const int n_k_tiles = (K + kMx8AtomKElems - 1) / kMx8AtomKElems;
    return static_cast<size_t>(n_row_tiles) * n_k_tiles * kMx8AtomSize;
}

// UE8M0 exponent e with 2^(e-127) >= absmax / 448, i.e. the block absmax lands in (224, 448]
// of the E4M3 range. 448 = 1.75 * 2^8, so the target is exact at absmax = 448 * 2^k.
__device__ __forceinline__ uint8_t mx8_scale_ue8m0(float absmax) {
    if (absmax <= 0.0f)
        return 127;  // scale 1.0: an all-zero block encodes as zeros either way
    const float target = absmax * (1.0f / 448.0f);
    uint32_t bits = __float_as_uint(target);
    int e = static_cast<int>((bits >> 23) & 0xFF);
    if (bits & 0x7FFFFF)
        e++;  // ceil to the next power of two
    e = std::max(e, 1);
    e = std::min(e, 254);
    return static_cast<uint8_t>(e);
}

__device__ __forceinline__ float mx8_ue8m0_to_float(uint8_t e) {
    return __uint_as_float(static_cast<uint32_t>(e) << 23);
}

// Two FP32 -> one E4M3 pair, v0 in the low byte. RNE, saturating (E4M3 has no inf).
__device__ __forceinline__ uint16_t mx8_pack_e4m3_pair(float v0, float v1) {
    uint16_t out;
    asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(out) : "f"(v0), "f"(v1));
    return out;
}

// One thread per 32-element block: 64 B in (4 x uint4), 32 B out (2 x uint4), one scale byte.
__global__ void quantize_fp16_mxfp8_cutlass_kernel(const half* __restrict__ input, uint8_t* __restrict__ out,
                                                   uint8_t* __restrict__ sf_out, int rows, int K,
                                                   int n_k_tiles) {
    const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int K_groups = K / kMx8SFVecSize;
    if (idx >= rows * K_groups)
        return;
    const int row = idx / K_groups;
    const int k_group = idx % K_groups;
    const size_t base = (static_cast<size_t>(row) * K) + (static_cast<size_t>(k_group) * kMx8SFVecSize);

    float vals[kMx8SFVecSize];
    float absmax = 0.0f;
    const uint4* src = reinterpret_cast<const uint4*>(input + base);
#pragma unroll
    for (int v = 0; v < 4; v++) {
        const uint4 q = src[v];
        const half2* h = reinterpret_cast<const half2*>(&q);
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const float2 f = __half22float2(h[j]);
            vals[(v * 8) + (j * 2)] = f.x;
            vals[(v * 8) + (j * 2) + 1] = f.y;
            absmax = fmaxf(absmax, fmaxf(fabsf(f.x), fabsf(f.y)));
        }
    }
    const uint8_t e = mx8_scale_ue8m0(absmax);
    sf_out[mx8_sfatom_offset(row, k_group, n_k_tiles)] = e;
    const float inv = 1.0f / mx8_ue8m0_to_float(e);

    uint32_t words[8];
    // #2218 bounded: w * 4 <= 28 (loop w < 8 literal)
#pragma unroll
    for (int w = 0; w < 8; w++) {
        const uint32_t lo = mx8_pack_e4m3_pair(vals[static_cast<ptrdiff_t>(w * 4)] * inv,
                                               vals[(w * 4) + 1] * inv);
        const uint32_t hi = mx8_pack_e4m3_pair(vals[(w * 4) + 2] * inv, vals[(w * 4) + 3] * inv);
        words[w] = lo | (hi << 16);
    }
    uint4* dst = reinterpret_cast<uint4*>(out + base);
    dst[0] = make_uint4(words[0], words[1], words[2], words[3]);
    dst[1] = make_uint4(words[4], words[5], words[6], words[7]);
}

void quantize_fp16_to_mxfp8_cutlass(const void* src_fp16, void* dst_data, void* dst_sf, int M, int K,
                                    cudaStream_t stream) {
    IMP_CHECK(K % kMx8SFVecSize == 0, "quantize_fp16_to_mxfp8_cutlass: K=%d must be a multiple of %d", K,
              kMx8SFVecSize);
    IMP_CUDA_CHECK_LOG(cudaMemsetAsync(dst_sf, 0, cutlass_mxfp8_sf_size(M, K), stream));
    const int K_groups = K / kMx8SFVecSize;
    const int total = M * K_groups;
    const int n_k_tiles = (K + kMx8AtomKElems - 1) / kMx8AtomKElems;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    quantize_fp16_mxfp8_cutlass_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const half*>(src_fp16), reinterpret_cast<uint8_t*>(dst_data),
        reinterpret_cast<uint8_t*>(dst_sf), M, K, n_k_tiles);
    IMP_CUDA_CHECK_LAUNCH();
}

// One thread per (row, k_group): the row's power of two in every K group, residual per row.
__global__ void mxfp8_sf_from_row_scales_kernel(const float* __restrict__ row_scales,
                                                uint8_t* __restrict__ sf, float* __restrict__ col_scale,
                                                int rows, int K_groups, int n_k_tiles) {
    const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= rows * K_groups)
        return;
    const int row = idx / K_groups;
    const int k_group = idx % K_groups;
    const float s = row_scales[row];
    int e = 127;
    float r = 0.0f;
    if (s > 0.0f) {
        int x = 0;
        frexpf(s, &x);  // s = m * 2^x, m in [0.5, 1)
        e = min(max(x - 1 + 127, 1), 254);
        r = s / mx8_ue8m0_to_float(static_cast<uint8_t>(e));
    }
    sf[mx8_sfatom_offset(row, k_group, n_k_tiles)] = static_cast<uint8_t>(e);
    if (k_group == 0)
        col_scale[row] = r;
}

void mxfp8_sf_from_row_scales(const float* row_scales, void* dst_sf, float* dst_col_scale, int rows, int K,
                              cudaStream_t stream) {
    IMP_CHECK(row_scales && dst_sf && dst_col_scale && rows > 0 && K % kMx8SFVecSize == 0,
              "mxfp8_sf_from_row_scales: empty input or K=%d not a multiple of %d", K, kMx8SFVecSize);
    IMP_CUDA_CHECK_LOG(cudaMemsetAsync(dst_sf, 0, cutlass_mxfp8_sf_size(rows, K), stream));
    const int K_groups = K / kMx8SFVecSize;
    const int n_k_tiles = (K + kMx8AtomKElems - 1) / kMx8AtomKElems;
    const int total = rows * K_groups;
    const int threads = 256;
    mxfp8_sf_from_row_scales_kernel<<<(total + threads - 1) / threads, threads, 0, stream>>>(
        row_scales, static_cast<uint8_t*>(dst_sf), dst_col_scale, rows, K_groups, n_k_tiles);
    IMP_CUDA_CHECK_LAUNCH();
}

// Inverse of quantize_fp16_mxfp8_cutlass_kernel: one thread per 32-element block.
__global__ void dequantize_mxfp8_cutlass_fp16_kernel(const uint8_t* __restrict__ data,
                                                     const uint8_t* __restrict__ sf,
                                                     const float* __restrict__ row_scale,
                                                     half* __restrict__ out, int rows, int K, int n_k_tiles) {
    const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int K_groups = K / kMx8SFVecSize;
    if (idx >= rows * K_groups)
        return;
    const int row = idx / K_groups;
    const int k_group = idx % K_groups;
    const size_t base = (static_cast<size_t>(row) * K) + (static_cast<size_t>(k_group) * kMx8SFVecSize);
    const float s = mx8_ue8m0_to_float(sf[mx8_sfatom_offset(row, k_group, n_k_tiles)]) *
                    (row_scale ? row_scale[row] : 1.0f);
    const uint4* src = reinterpret_cast<const uint4*>(data + base);
    uint4* dst = reinterpret_cast<uint4*>(out + base);
#pragma unroll
    for (int v = 0; v < 2; v++) {
        const uint4 q = src[v];
        const uint8_t* b = reinterpret_cast<const uint8_t*>(&q);
#pragma unroll
        for (int h = 0; h < 2; h++) {
            __align__(16) half o[8];
#pragma unroll
            for (int j = 0; j < 8; j++) {
                __nv_fp8_e4m3 e;
                *reinterpret_cast<uint8_t*>(&e) = b[(h * 8) + j];
                o[j] = __float2half(static_cast<float>(e) * s);
            }
            dst[(v * 2) + h] = *reinterpret_cast<const uint4*>(o);
        }
    }
}

void dequantize_mxfp8_cutlass_to_fp16(const CutlassMxFP8Weight& w, void* dst_fp16, cudaStream_t stream) {
    IMP_CHECK(w.data && w.scale_factors && dst_fp16 && w.K % kMx8SFVecSize == 0,
              "dequantize_mxfp8_cutlass_to_fp16: empty weight or K=%lld not a multiple of %d",
              static_cast<long long>(w.K), kMx8SFVecSize);
    const int rows = static_cast<int>(w.N), K = static_cast<int>(w.K);
    const int total = rows * (K / kMx8SFVecSize);
    const int n_k_tiles = (K + kMx8AtomKElems - 1) / kMx8AtomKElems;
    const int threads = 256;
    dequantize_mxfp8_cutlass_fp16_kernel<<<(total + threads - 1) / threads, threads, 0, stream>>>(
        static_cast<const uint8_t*>(w.data), static_cast<const uint8_t*>(w.scale_factors), w.col_scale,
        static_cast<half*>(dst_fp16), rows, K, n_k_tiles);
    IMP_CUDA_CHECK_LAUNCH();
}

}  // namespace imp
