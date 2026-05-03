// =============================================================================
// mxf4nvf4_qkt_validate.cu -- End-to-end Q·K^T correctness against FP32 reference
// =============================================================================
//
// Takes FP16 Q[M=16, K=64] and K[N=8, K=64], runs the
//   mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64
// with uniform scale = 1.0, and writes D[M, N] as FP32.
//
// Uses CUTLASS MMA_Traits from mma_traits_sm120.hpp:
//   ALayout = Layout<Shape <Shape <  _4,_8>,Shape < _8,_2,  _2>>,
//                    Stride<Stride<_128,_1>,Stride<_16,_8,_512>>>   // (T32,V32)→(M16,K64)
//   BLayout = Layout<Shape <Shape < _4,_8>,Shape <_8,  _2>>,
//                    Stride<Stride<_64,_1>,Stride<_8,_256>>>         // (T32,V16)→(N8,K64)
//   CLayout = SM80_16x8_Row                                          // (T32,V4)→(M16,N8)
//
// KEY INSIGHT: CuTe layouts `(T,V)→(M,K)` are COLUMN-MAJOR over the
// (M,K) coord space — offset = k*M + m, NOT m*K + k. This means per
// thread t=(T0=t%4, T1=t/4), v=(V0=v%8, V1=(v>>3)&1, V2=(v>>4)&1):
//   A: m = T1 + V1*8, k = T0*8 + V0 + V2*32
//   B: n = T1,        k = T0*8 + V0 + V1*32
//   C: m = T1 + V1*8, n = T0*2 + V0 (V0 = v%2, V1 = v>>1)
//
// Scale = 1.0 uniform (FP8 UE4M3 byte 0x38) so the MMA reduces to plain
// E2M1 × E2M1 dot product.
// =============================================================================

#include "compute/mxf4nvf4_qkt_validate.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>

namespace imp {

// E2M1 magnitudes (sign bit separate).
// Values: {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}
__device__ __forceinline__ uint8_t fp32_to_e2m1(float v) {
    uint8_t sign = (v < 0.0f) ? 0x8 : 0x0;
    float a = fabsf(v);
    // Midpoint thresholds between consecutive magnitudes
    uint8_t mag = (a >= 0.25f) + (a >= 0.75f) + (a >= 1.25f) + (a >= 1.75f) + (a >= 2.5f) + (a >= 3.5f) +
                  (a >= 5.0f);
    return sign | mag;
}

// Kernel: 1 warp (32 threads). Each thread loads 32 A values + 16 B values
// per the CUTLASS layout, packs to uint32 registers, issues one MMA, and
// writes its 4 D outputs back.
__global__ void qkt_mxf4nvf4_kernel(const half* __restrict__ Q,     // [M=16, KD=64] row-major
                                    const half* __restrict__ Kmat,  // [N=8,  KD=64] row-major
                                    float* __restrict__ D)          // [M=16, N=8]   row-major
{
    constexpr int M = 16;
    constexpr int KD = 64;  // K-dim (renamed from K to avoid clash with param)
    constexpr int N = 8;

    const int tid = threadIdx.x;
    if (tid >= 32)
        return;

    // CuTe: T = T0 + T1*4, T0=t%4 (inner, stride 128 for A), T1=t/4.
    const int T0 = tid % 4;
    const int T1 = tid / 4;

    // --- Load A operand: 32 FP4 values per thread into 4 uint32 ---
    // Per-thread coverage (column-major (M,K) linearization):
    //   m = T1 + V1*8       with V1 ∈ {0,1}
    //   k = T0*8 + V0 + V2*32  with V0 ∈ [0,8), V2 ∈ {0,1}
    // Register packing: nibble offset (v&7) = V0, register (v>>3) = V1+V2*2.
    uint32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    for (int v = 0; v < 32; ++v) {
        int V0 = v & 7;
        int V1 = (v >> 3) & 1;
        int V2 = (v >> 4) & 1;
        int m = T1 + V1 * 8;
        int k = T0 * 8 + V0 + V2 * 32;
        float val = (m < M && k < KD) ? __half2float(Q[m * KD + k]) : 0.0f;
        uint8_t nibble = fp32_to_e2m1(val);
        int reg_idx = v >> 3;
        int nib_in_reg = v & 7;
        uint32_t shifted = static_cast<uint32_t>(nibble) << (nib_in_reg * 4);
        if (reg_idx == 0)
            a0 |= shifted;
        else if (reg_idx == 1)
            a1 |= shifted;
        else if (reg_idx == 2)
            a2 |= shifted;
        else
            a3 |= shifted;
    }

    // --- Load B operand: 16 FP4 values per thread into 2 uint32 ---
    // Per-thread coverage (column-major (N,K) linearization):
    //   n = T1                          (one n-value per thread group of 4)
    //   k = T0*8 + V0 + V1*32     with V0 ∈ [0,8), V1 ∈ {0,1}
    // Register packing: nib (v&7) = V0, register (v>>3) = V1.
    uint32_t b0 = 0, b1 = 0;
    for (int v = 0; v < 16; ++v) {
        int V0 = v & 7;
        int V1 = (v >> 3) & 1;
        int n = T1;
        int k = T0 * 8 + V0 + V1 * 32;
        float val = (n < N && k < KD) ? __half2float(Kmat[n * KD + k]) : 0.0f;
        uint8_t nibble = fp32_to_e2m1(val);
        int reg_idx = v >> 3;
        int nib_in_reg = v & 7;
        uint32_t shifted = static_cast<uint32_t>(nibble) << (nib_in_reg * 4);
        if (reg_idx == 0)
            b0 |= shifted;
        else
            b1 |= shifted;
    }

    // Uniform scale = 1.0 in FP8 UE4M3 (byte 0x38). 4 bytes per scale operand.
    uint32_t sfa = 0x38383838u;
    uint32_t sfb = 0x38383838u;

    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;

    constexpr uint16_t tidA = 0;
    constexpr uint16_t bidA = 0;
    constexpr uint16_t bidB = 0;
    constexpr uint16_t tidB0 = 0;

#if __CUDA_ARCH__ >= 1200
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0, %1, %2, %3},"
        "{%4, %5, %6, %7},"
        "{%8, %9},"
        "{%10, %11, %12, %13},"
        "{%14},"
        "{%15, %16},"
        "{%17},"
        "{%18, %19};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(d0), "f"(d1), "f"(d2), "f"(d3), "r"(sfa),
          "h"(bidA), "h"(tidA), "r"(sfb), "h"(bidB), "h"(tidB0));
#endif

    // --- Write D outputs ---
    // CLayout (T32,V4)→(M16,N8), column-major: offset = n*M + m.
    // For thread (T0, T1), V=(V0=v%2, V1=v>>1):
    //   m = T1 + V1*8
    //   n = T0*2 + V0
    // d0 → (T1,   T0*2);     d1 → (T1,   T0*2+1)
    // d2 → (T1+8, T0*2);     d3 → (T1+8, T0*2+1)
    const int m0 = T1;
    const int m1 = T1 + 8;
    const int n0 = T0 * 2;
    const int n1 = n0 + 1;

    if (m0 < M && n0 < N)
        D[m0 * N + n0] = d0;
    if (m0 < M && n1 < N)
        D[m0 * N + n1] = d1;
    if (m1 < M && n0 < N)
        D[m1 * N + n0] = d2;
    if (m1 < M && n1 < N)
        D[m1 * N + n1] = d3;
}

bool qkt_mxf4nvf4_validate(const half* d_Q, const half* d_K, float* d_D, cudaStream_t stream) {
    qkt_mxf4nvf4_kernel<<<1, 32, 0, stream>>>(d_Q, d_K, d_D);
    return cudaGetLastError() == cudaSuccess;
}

}  // namespace imp
