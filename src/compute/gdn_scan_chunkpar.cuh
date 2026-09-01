// Internal header of the chunk-parallel GDN prefill scan: the pieces both
// kernel TUs share (workspace layout, tf32 mma helpers, shared-tile swizzle)
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

// mma.sync m16n8k8 tf32 helpers, shared by both kernels.
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

// Fragment MMA on FP32 operands. X3 = the 3xTF32 error-compensated form
// (a = a_hi + a_lo, three MMAs: a_lo*b_hi + a_hi*b_lo + a_hi*b_hi), ~FP32
// accuracy on the products; plain tf32 otherwise. Plain tf32 on all three
// chunk GEMMs read PPL +0.13% on Qwen3.6-35B (6.8216 -> 6.8304): the state
// path compounds the 10-bit operand rounding across chunks.
template <bool X3>
__device__ __forceinline__ void mma_frag(float* c, const float* a, const float* b) {
    uint32_t ah[4], bh[2];
#pragma unroll
    for (int i = 0; i < 4; i++)
        ah[i] = f32_to_tf32(a[i]);
#pragma unroll
    for (int i = 0; i < 2; i++)
        bh[i] = f32_to_tf32(b[i]);
    if constexpr (X3) {
        uint32_t al[4], bl[2];
#pragma unroll
        for (int i = 0; i < 4; i++)
            al[i] = f32_to_tf32(a[i] - __uint_as_float(ah[i]));
#pragma unroll
        for (int i = 0; i < 2; i++)
            bl[i] = f32_to_tf32(b[i] - __uint_as_float(bh[i]));
        mma_tf32_16x8x8(c, al, bh);
        mma_tf32_16x8x8(c, ah, bl);
    }
    mma_tf32_16x8x8(c, ah, bh);
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
