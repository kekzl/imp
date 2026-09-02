// Internal header of the chunk-parallel GDN prefill scan: the pieces both
// kernel TUs share (workspace layout, fp16 mma helpers, shared-tile swizzles)
// and the per-kernel launch wrappers. The narrative lives in
// gdn_scan_chunkpar.cu; kernel 2 is gdn_scan_chunkpar_pass.cu.
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace imp {
namespace chunkpar {


constexpr int kChunk = 64;           // tokens per chunk (WY tile)
constexpr int kMaxStripChunks = 16;  // workspace slots per head; the strip actually run is <= this
constexpr int kColSplit = 4;     // K2 CTAs per head (state columns split): 32 columns each

// Workspace float layout, per strip slot (slot = c * n_heads + h):
//   W    [slots][kChunk*SS]   phase A: solve RHS; phase B: solved W
//   KD   [slots][kChunk*SS]
//   UA   [slots][kChunk*HD]
//   QE   [slots][kChunk*SS]   phase A: D[0..t+1] q~; phase C: finished Qeff
//   YA   [slots][kChunk*HD]
//   D0L  [slots]
//   H32  [n_heads*SS*HD]      FP32 inter-strip state
struct ChunkparWs {
    float* W;
    float* KD;
    float* UA;
    float* QE;
    float* YA;
    float* D0L;
    float* H32;
};

template <int HD, int SS>
__host__ __device__ inline ChunkparWs chunkpar_ws_layout(float* base, int n_heads) {
    const size_t slots = static_cast<size_t>(kMaxStripChunks) * n_heads;
    const size_t arr = slots * kChunk * SS;  // SS == HD
    ChunkparWs w;
    w.W = base;
    w.KD = base + arr;
    w.UA = base + 2 * arr;
    w.QE = base + 3 * arr;
    w.YA = base + 4 * arr;
    w.D0L = base + 5 * arr;
    w.H32 = w.D0L + slots;
    return w;
}

// mma.sync m16n8k8 tf32 with FP32 accumulate (kernel 1's output-only Y_A GEMM).
__device__ __forceinline__ uint32_t f32_to_tf32(float f) {
    uint32_t r;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(f));
    return r;
}

__device__ __forceinline__ void mma_tf32_16x8x8(float* c, const uint32_t* a, const uint32_t* b) {
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

struct Tf32A {
    uint32_t v[4];
};
struct Tf32B {
    uint32_t v[2];
};
__device__ __forceinline__ Tf32A tf32_a(const float* a) {
    Tf32A r;
#pragma unroll
    for (int i = 0; i < 4; i++)
        r.v[i] = f32_to_tf32(a[i]);
    return r;
}
__device__ __forceinline__ Tf32B tf32_b(const float* b) {
    Tf32B r;
#pragma unroll
    for (int i = 0; i < 2; i++)
        r.v[i] = f32_to_tf32(b[i]);
    return r;
}
__device__ __forceinline__ void mma_tf32(float* c, const Tf32A& a, const Tf32B& b) {
    mma_tf32_16x8x8(c, a.v, b.v);
}

// mma.sync m16n8k16 f16 with FP32 accumulate: a[4] = packed half2 A fragment,
// b[2] = packed half2 B fragment (PTX ISA fragment layouts for .m16n8k16).
__device__ __forceinline__ void mma_f16_16x8x16(float* c, const uint32_t* a, const uint32_t* b) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ void split_f16x2(float x0, float x1, uint32_t& hi, uint32_t& lo) {
    const __half h0 = __float2half_rn(x0), h1 = __float2half_rn(x1);
    const __half l0 = __float2half_rn(x0 - __half2float(h0)), l1 = __float2half_rn(x1 - __half2float(h1));
    hi = static_cast<uint32_t>(__half_as_ushort(h0)) | (static_cast<uint32_t>(__half_as_ushort(h1)) << 16);
    lo = static_cast<uint32_t>(__half_as_ushort(l0)) | (static_cast<uint32_t>(__half_as_ushort(l1)) << 16);
}

// Operand forms split once per k-step and reused across the n-tiles that
// share the fragment.
struct F16A {
    uint32_t hi[4], lo[4];
};
struct F16B {
    uint32_t hi[2], lo[2];
};
__device__ __forceinline__ F16A f16_split_a(const float* a8) {
    F16A r;
#pragma unroll
    for (int i = 0; i < 4; i++)
        split_f16x2(a8[2 * i], a8[2 * i + 1], r.hi[i], r.lo[i]);
    return r;
}
__device__ __forceinline__ F16B f16_split_b(const float* b4) {
    F16B r;
#pragma unroll
    for (int i = 0; i < 2; i++)
        split_f16x2(b4[2 * i], b4[2 * i + 1], r.hi[i], r.lo[i]);
    return r;
}
__device__ __forceinline__ void mma_f16x3(float* c, const F16A& a, const F16B& b) {
    mma_f16_16x8x16(c, a.lo, b.hi);
    mma_f16_16x8x16(c, a.hi, b.lo);
    mma_f16_16x8x16(c, a.hi, b.hi);
}
// Plain fp16 (11-bit products, the tf32 class) for output-only terms.
__device__ __forceinline__ void mma_f16x1(float* c, const F16A& a, const F16B& b) {
    mma_f16_16x8x16(c, a.hi, b.hi);
}

// Element offset in a [64 x 128] FP32 shared tile with an XOR swizzle on the
// float4 column index: physical chunk = (col / 4) ^ ((row & 7) * 2). Both
// warp patterns of kernel 1 land on 8 distinct 16-B chunks (= 32 banks):
//   - mma A/B fragments, 8 rows (g) x 4 consecutive floats (tg): rows 0..7
//     hit chunk c ^ {0, 2, .., 14};
//   - the history B operand, 4 rows (tg) x 8 consecutive floats (g): rows
//     0..3 hit {c, c+1} ^ {0, 2, 4, 6}.
// Unswizzled, the 128-float stride put every row of a fragment on the same
// banks: ncu read 11.1M bank conflicts on 17.3M shared wavefronts in this
// kernel (64%). Padding the histories to stride 132 does not fit the 99 KB
// budget next to the padded T/P tiles; the swizzle costs no bytes.
__device__ __forceinline__ int swz128(int row, int col) {
    return row * 128 + ((((col >> 2) ^ ((row & 7) << 1))) << 2) + (col & 3);
}

// Kernel 1 (gdn_scan_chunkpar.cu): per-(chunk, head) factors of one strip,
// grid (n_chunks x n_heads).
void chunkpar_intra_128(const float* conv_f32, const half* alpha, const half* beta, const float* A_log,
                        const float* dt_bias, float* ws_base, int strip_t0, int strip_tokens, int n_chunks,
                        int n_heads, int n_groups, int conv_channels, int grouped_layout,
                        cudaStream_t stream);

// Kernel 2 (gdn_scan_chunkpar_pass.cu): the sequential state pass over the
// strip's chunks, grid (n_heads x kColSplit). StateT float or __nv_bfloat16.
template <typename StateT>
void chunkpar_pass_128(float* ws_base, StateT* h_state, half* y, int strip_t0, int strip_tokens,
                       int n_chunks, int n_heads, int load_statet, int store_statet, cudaStream_t stream);

}  // namespace chunkpar
}  // namespace imp
