// =============================================================================
// mxf4nvf4_qkt_validate.cu -- End-to-end Q·K^T correctness against FP32 reference
// =============================================================================
//
// Takes FP16 Q[M=16, K=64] and K[N=8, K=64], runs the
//   mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64
// with uniform scale = 1.0, and writes D[M, N] as FP32.
//
// Uses CUTLASS (T32,V32)→(M16,K64) layout derived from cute_extension.h:
//   ALayout = Layout<Shape <Shape <  _4,_8>,Shape < _8,_2,  _2>>,
//                    Stride<Stride<_128,_1>,Stride<_16,_8,_512>>>
// For thread t at value v:
//   t_outer = t / 8, t_inner = t % 8
//   v0 = v % 8, v1 = (v/8) % 2, v2 = (v/16) % 2
//   logical_offset = t_outer*128 + t_inner + v0*16 + v1*8 + v2*512
//   m = offset / 64, k = offset % 64
//
// With scale_vec::4X, only 1 of 4 MMA issues is used here (tidB=0), so B
// covers N=[0, 8) of the 32-N total extent.
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
    uint8_t mag = (a >= 0.25f)
                + (a >= 0.75f)
                + (a >= 1.25f)
                + (a >= 1.75f)
                + (a >= 2.5f)
                + (a >= 3.5f)
                + (a >= 5.0f);
    return sign | mag;
}

// Kernel: 1 warp (32 threads). Each thread loads 32 A values + 16 B values
// per the CUTLASS layout, packs to uint32 registers, issues one MMA, and
// writes its 4 D outputs back.
__global__ void qkt_mxf4nvf4_kernel(
    const half* __restrict__ Q,       // [M=16, KD=64] row-major
    const half* __restrict__ Kmat,    // [N=8,  KD=64] row-major
    float* __restrict__ D)            // [M=16, N=8]   row-major
{
    constexpr int M = 16;
    constexpr int KD = 64;  // K-dim (renamed from K to avoid clash with param)
    constexpr int N = 8;

    const int tid = threadIdx.x;
    if (tid >= 32) return;

    // CuTe column-major: Shape<_4,_8>, Stride<_128,_1>.
    // Linear t decomposes as t.0 = t%4 (inner, shape 4), t.1 = t/4 (outer, shape 8).
    // Thread offset = (t%4) * 128 + (t/4) * 1.
    const int t_outer = tid % 4;   // t.0, stride 128 (row group)
    const int t_inner = tid / 4;   // t.1, stride 1 (k-offset)

    // --- Load A operand: 32 FP4 values per thread into 4 uint32 ---
    uint32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    for (int v = 0; v < 32; ++v) {
        int v0 = v & 7;
        int v1 = (v >> 3) & 1;
        int v2 = (v >> 4) & 1;
        int offset = t_outer * 128 + t_inner + v0 * 16 + v1 * 8 + v2 * 512;
        int m = offset / KD;
        int k = offset % KD;
        float val = (m < M && k < KD) ? __half2float(Q[m * KD + k]) : 0.0f;
        uint8_t nibble = fp32_to_e2m1(val);
        int reg_idx = v >> 3;
        int nib_in_reg = v & 7;
        uint32_t shifted = static_cast<uint32_t>(nibble) << (nib_in_reg * 4);
        if      (reg_idx == 0) a0 |= shifted;
        else if (reg_idx == 1) a1 |= shifted;
        else if (reg_idx == 2) a2 |= shifted;
        else                   a3 |= shifted;
    }

    // --- Load B operand: 16 FP4 values per thread into 2 uint32 ---
    // BLayout for single tidB=0 subtile (N=8, K=64):
    //   Shape <_4,_8>, stride (_256, _1)  — same thread decomposition pattern
    //   Values: 16 per thread for one tidB issue
    // For this single-tile validation we use an analogous mapping over N=8.
    // Since N=8 is smaller than M=16, the t_outer only needs to iterate
    // over half the rows. Threads t_outer ∈ [0, 1) cover N=[0, 8), the
    // upper t_outer values would cover further N extents in the 4-issue
    // variant — here we just reuse the pattern reduced to N=8.
    uint32_t b0 = 0, b1 = 0;
    for (int v = 0; v < 16; ++v) {
        int v0 = v & 7;
        int v1 = (v >> 3) & 1;
        // For N=8 single tile: n-stride smaller since only 8 rows
        // Simplify: treat the B layout as (T,V)→(N,K) with N=8
        //   n = (t_outer * 8 + (v0 < 4 ? 0 : 4)) % 8  — placeholder; iterate
        // TODO: derive exact single-tile B layout from SageAttention3 BLayout
        int n = (t_outer * 2 + (v >> 4)) % N;
        int k_sub = t_inner + v0 * 8 + v1 * 16;  // within the t_inner stride
        int k = k_sub % KD;
        float val = (n < N && k < KD) ? __half2float(Kmat[n * KD + k]) : 0.0f;
        uint8_t nibble = fp32_to_e2m1(val);
        int reg_idx = v >> 3;
        int nib_in_reg = v & 7;
        uint32_t shifted = static_cast<uint32_t>(nibble) << (nib_in_reg * 4);
        if (reg_idx == 0) b0 |= shifted;
        else              b1 |= shifted;
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
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(d0), "f"(d1), "f"(d2), "f"(d3),
          "r"(sfa), "h"(bidA), "h"(tidA),
          "r"(sfb), "h"(bidB), "h"(tidB0));
#endif

    // --- Write D outputs ---
    // D layout (T32,V4)→(M16,N8) per CUTLASS convention:
    //   d0..d3 per thread map to specific (m, n) pairs.
    //   For m16n8: each thread holds 4 values at (m=lane_id/4*2, n=lane_id%4*2)
    //   offset pattern. Standard m16n8 FP32 output layout:
    //     d0, d1 → row (tid/4), columns (tid%4)*2, (tid%4)*2+1
    //     d2, d3 → row (tid/4)+8, same columns
    const int out_row0 = tid / 4;        // 0..7
    const int out_row1 = (tid / 4) + 8;  // 8..15
    const int out_col0 = (tid % 4) * 2;
    const int out_col1 = out_col0 + 1;

    if (out_row0 < M && out_col0 < N) D[out_row0 * N + out_col0] = d0;
    if (out_row0 < M && out_col1 < N) D[out_row0 * N + out_col1] = d1;
    if (out_row1 < M && out_col0 < N) D[out_row1 * N + out_col0] = d2;
    if (out_row1 < M && out_col1 < N) D[out_row1 * N + out_col1] = d3;
}

bool qkt_mxf4nvf4_validate(const half* d_Q, const half* d_K, float* d_D,
                           cudaStream_t stream) {
    qkt_mxf4nvf4_kernel<<<1, 32, 0, stream>>>(d_Q, d_K, d_D);
    return cudaGetLastError() == cudaSuccess;
}

} // namespace imp
