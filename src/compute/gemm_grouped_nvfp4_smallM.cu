// src/compute/gemm_grouped_nvfp4_smallM.cu
#include "compute/gemm_grouped_nvfp4_smallM.h"
#include "core/logging.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <vector>
#include <cstdint>

#include "cute/tensor.hpp"
#include "cute/atom/copy_atom.hpp"
#include "cute/atom/copy_traits_sm90_tma.hpp"  // SM90_TMA_LOAD / make_tma_copy

namespace imp {

namespace {

// cp.async helpers (file-local mirror of compute/attention_paged_common.cuh).
// Kept local to avoid pulling in attention-specific headers from a GEMM TU.
__device__ __forceinline__ void cp_async_cg_16_local(void* smem, const void* glob) {
    uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(glob));
}
// cp.async.cg only supports 16-byte transfers on SM120; the 8-byte path uses
// cp.async.ca (which is what the existing paged-attention code in
// attention_paged_common.cuh uses for sub-16B chunks).
__device__ __forceinline__ void cp_async_ca_8_local(void* smem, const void* glob) {
    uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" ::"r"(s), "l"(glob));
}
__device__ __forceinline__ void cp_async_commit_local() {
    asm volatile("cp.async.commit_group;\n");
}
template <int N>
__device__ __forceinline__ void cp_async_wait_group_local() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

// Inline-PTX wrapper for the block-scaled MMA on SM120.
// Issues 1 mma.sync that consumes:
//   A: 16x64 FP4 (4 b32 registers per warp)
//   B: 8x64 FP4 (2 b32 registers per warp)
//   SFA: 4 UE4M3 scales per group (packed in 1 b32)
//   SFB: 4 UE4M3 scales per group (packed in 1 b32)
//   D: accumulator FP32, 4 floats per thread (16x8 owned by warp)
//
// Validated 268 TOPS via tests/test_mxf4nvf4_mma_variants_bench.cu.
// bid/tid for scale addressing are zero (all threads in the same tile see the
// same scale register — matches the load pattern in the full kernel).
__device__ __forceinline__ void mma_sync_mxf4nvf4_m16n8k64(
    float* d,           // 4 floats in/out (FP32 accumulator, C→D)
    const uint32_t* a,  // 4 uint32 (A fragment for the warp)
    const uint32_t* b,  // 2 uint32 (B fragment for the warp)
    uint32_t sfa,       // 1 uint32 = 4 UE4M3 scales for A
    uint32_t sfb) {     // 1 uint32 = 4 UE4M3 scales for B
#if (__CUDA_ARCH__ >= 1200)
    constexpr uint16_t bid = 0, tid = 0;
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32."
        "ue4m3 "
        "{%0,%1,%2,%3},"
        "{%4,%5,%6,%7},"
        "{%8,%9},"
        "{%10,%11,%12,%13},"
        "{%14},{%15,%16},"
        "{%17},{%18,%19};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "f"(d[0]), "f"(d[1]), "f"(d[2]), "f"(d[3]),
          "r"(sfa),  "h"(bid),  "h"(tid),
          "r"(sfb),  "h"(bid),  "h"(tid));
#else
    (void)d; (void)a; (void)b; (void)sfa; (void)sfb;
#endif
}

#ifdef SMALLM_SOFTWARE_REF
// ---------------------------------------------------------------------------
// smallM kernel v1 SOFTWARE REFERENCE (debug-only; previous correctness path).
//
// Original implementation from commit 77a7807 (T1.7). Retained as a
// ground-truth reference for cross-checking the production HW MMA kernel.
// Compiled only when SMALLM_SOFTWARE_REF is defined (debug builds / unit
// tests that compare HW vs SW outputs).
//
// Each thread computes one output cell with software FP4-decode +
// FP32 accumulation. Slow but bit-stable; not on the production path.
//
// Layout consumed:
//   A_e   : [M_e, K/2]   row-major packed FP4 (low-nibble = even index)
//   SFA_e : [M_e, K/16]  row-major UE4M3 byte (1 byte / 16 elements)
//   B_e   : [N,   K/2]   row-major packed FP4
//   SFB_e : [N,   K/16]  row-major UE4M3 byte
//   D_e   : [M_e, N]     row-major FP16 output
// ---------------------------------------------------------------------------

// Lookup: e2m1 nibble → FP32 magnitude.  Sign comes from the high bit of nibble.
__device__ __forceinline__ float e2m1_nibble_to_fp32(uint8_t nib) {
    // E2M1: 1 sign | 2 exp | 1 mant.  Magnitudes: 0, .5, 1, 1.5, 2, 3, 4, 6.
    static constexpr float kMag[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    float mag = kMag[nib & 0x7];
    return (nib & 0x8) ? -mag : mag;
}

// Software FP8 E4M3 byte → FP32 (canonical fast bit-repack from fp8_utils.cuh).
//   Normal (exp>0): value = (1 + man/8) * 2^(exp - 7)   [bias 7]
//   Denorm (exp=0): value = man * 2^-9
__device__ __forceinline__ float ue4m3_to_fp32(uint8_t bits) {
    uint32_t sign = (bits >> 7) & 1;
    uint32_t exp  = (bits >> 3) & 0x0F;
    uint32_t man  = bits & 0x07;
    uint32_t fp32;
    if (exp == 0) {
        float v = (float)man * (1.0f / 512.0f);
        fp32 = (sign << 31) | __float_as_uint(v);
    } else {
        fp32 = (sign << 31) | ((exp + 120u) << 23) | (man << 20);
    }
    return __uint_as_float(fp32);
}

template <int TILE_M, int TILE_N, int TILE_K>
__global__ void smallM_kernel_v1_software_ref(
    const void* const* __restrict__ d_A,
    const void* const* __restrict__ d_SFA,
    const void* const* __restrict__ d_B,
    const void* const* __restrict__ d_SFB,
    void* const* __restrict__ d_D,
    const float* __restrict__ d_alpha,
    const int* __restrict__ d_M_per_expert,
    int N, int K) {
    const int e      = blockIdx.x;
    const int n_tile = blockIdx.y;
    const int M_e = d_M_per_expert[e];
    if (M_e <= 0) return;

    const uint8_t* A_e   = static_cast<const uint8_t*>(d_A[e]);
    const uint8_t* B_e   = static_cast<const uint8_t*>(d_B[e]);
    const uint8_t* SFA_e = static_cast<const uint8_t*>(d_SFA[e]);
    const uint8_t* SFB_e = static_cast<const uint8_t*>(d_SFB[e]);
    half*          D_e   = static_cast<half*>(d_D[e]);
    const float    alpha = d_alpha[e];

    const int K_half  = K / 2;
    const int K_block = K / 16;

    const int n_base = n_tile * TILE_N;
    const int M_eff = min(M_e, TILE_M);

    const int total_cells = TILE_M * TILE_N;
    for (int idx = (int)threadIdx.x; idx < total_cells; idx += (int)blockDim.x) {
        int m = idx / TILE_N;
        int n = idx % TILE_N;
        if (m >= M_eff) continue;
        if (n_base + n >= N) continue;

        float acc = 0.f;
        for (int kb = 0; kb < K_block; ++kb) {
            float sfa = ue4m3_to_fp32(SFA_e[(int64_t)m * K_block + kb]);
            float sfb = ue4m3_to_fp32(SFB_e[(int64_t)(n_base + n) * K_block + kb]);
            float scale = sfa * sfb;

            const uint8_t* a_ptr = A_e + (int64_t)m * K_half + kb * 8;
            const uint8_t* b_ptr = B_e + (int64_t)(n_base + n) * K_half + kb * 8;

            float partial = 0.f;
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                uint8_t ab = a_ptr[i];
                uint8_t bb = b_ptr[i];
                float a0v = e2m1_nibble_to_fp32(ab & 0xF);
                float a1v = e2m1_nibble_to_fp32((ab >> 4) & 0xF);
                float b0v = e2m1_nibble_to_fp32(bb & 0xF);
                float b1v = e2m1_nibble_to_fp32((bb >> 4) & 0xF);
                partial += a0v * b0v + a1v * b1v;
            }
            acc += partial * scale;
        }
        acc *= alpha;
        D_e[(int64_t)m * N + (n_base + n)] = __float2half(acc);
    }
}
#endif  // SMALLM_SOFTWARE_REF

// ---------------------------------------------------------------------------
// smallM kernel v1 — PRODUCTION HW MMA PATH.
//
// Grid:  (n_experts, N / TILE_N).  Each CTA owns one expert × one n-tile.
// Block: 256 threads (8 warps); single-warp (warp 0) issues the MMAs
//        for correctness baseline.  Multi-warp split deferred to T1.10.
//
// Pipeline:
//   1. cp.async A, B, SFA, SFB tiles from global → SMEM (all warps).
//   2. cp.async.wait_group + __syncthreads.
//   3. Warp 0 walks (mi, ni) sub-tiles (8 × 16 = 128 sub-tiles per CTA),
//      issuing 2 mma.sync.kind::mxf4nvf4 per sub-tile (one per K-stripe of 64).
//   4. Each lane writes its 4 FP32 → FP16 outputs directly to D_e in global
//      with the alpha factor folded in.
//
// Per-lane fragment assembly (mirrors mxf4nvf4_qkt_validate.cu and
// attention_fmha_mxfp4_sm120.cu UseBlockScaleMma=true path):
//   group_id     = lane / 4   (T1, range 0..7)
//   thread_in_g  = lane % 4   (T0, range 0..3)
//   byte_offset  = thread_in_g * 4    (4 bytes = 8 FP4 nibbles per b32)
//
//   For sub-tile (mi, ni) at K-stripe ki ∈ {0,1}:
//     a0 = SMEM_A[(mi*16 + group_id)     * (TILE_K/2) + ki*32 +  0 + byte_off]
//     a1 = SMEM_A[(mi*16 + group_id + 8) * (TILE_K/2) + ki*32 +  0 + byte_off]
//     a2 = SMEM_A[(mi*16 + group_id)     * (TILE_K/2) + ki*32 + 16 + byte_off]
//     a3 = SMEM_A[(mi*16 + group_id + 8) * (TILE_K/2) + ki*32 + 16 + byte_off]
//     b0 = SMEM_B[(ni*8  + group_id)     * (TILE_K/2) + ki*32 +  0 + byte_off]
//     b1 = SMEM_B[(ni*8  + group_id)     * (TILE_K/2) + ki*32 + 16 + byte_off]
//
//   Per-lane scale fragment (CUTLASS SFALayout/SFBLayout @ scale_vec::4X):
//     m_sfa = group_id + (thread_in_g & 1) * 8     // 16 unique m's; T1*1 broadcast
//     n_sfb = group_id                             // 8 unique n's
//     kg_base = ki * 4                             // 4 K-groups per stripe of 64
//     sfa = SMEM_SFA[(mi*16 + m_sfa) * (TILE_K/16) + kg_base]   (4 UE4M3 bytes)
//     sfb = SMEM_SFB[(ni*8  + n_sfb) * (TILE_K/16) + kg_base]
//
// Epilogue (CLayout = SM80_16x8_Row):
//     m0 = mi*16 + group_id        n0 = n_tile*128 + ni*8 + thread_in_g*2
//     m1 = mi*16 + group_id + 8    n1 = n0 + 1
//     d0 → (m0, n0),  d1 → (m0, n1),  d2 → (m1, n0),  d3 → (m1, n1)
// ---------------------------------------------------------------------------

template <int TILE_M, int TILE_N, int TILE_K>
__global__ void smallM_kernel_v1(
    const void* const* __restrict__ d_A,
    const void* const* __restrict__ d_SFA,
    const void* const* __restrict__ d_B,
    const void* const* __restrict__ d_SFB,
    void* const* __restrict__ d_D,
    const float* __restrict__ d_alpha,
    const int* __restrict__ d_M_per_expert,
    int N, int K) {
    static_assert(TILE_M == 128 && TILE_N == 128 && TILE_K == 128,
                  "smallM_kernel_v1 currently fixed at 128×128×128");

    const int e      = blockIdx.x;
    const int n_tile = blockIdx.y;
    const int M_e = d_M_per_expert[e];
    if (M_e <= 0) return;

    const uint8_t* A_e   = static_cast<const uint8_t*>(d_A[e]);
    const uint8_t* B_e   = static_cast<const uint8_t*>(d_B[e]);
    const uint8_t* SFA_e = static_cast<const uint8_t*>(d_SFA[e]);
    const uint8_t* SFB_e = static_cast<const uint8_t*>(d_SFB[e]);
    half*          D_e   = static_cast<half*>(d_D[e]);
    const float    alpha = d_alpha[e];

    constexpr int A_BYTES_ROW   = TILE_K / 2;     // 64 bytes per A-row in tile
    constexpr int B_BYTES_ROW   = TILE_K / 2;     // 64 bytes per B-row in tile
    constexpr int SFA_BYTES_ROW = TILE_K / 16;    // 8 bytes per A-row in tile
    constexpr int SFB_BYTES_ROW = TILE_K / 16;    // 8 bytes per B-row in tile
    constexpr int A_TILE_BYTES   = TILE_M * A_BYTES_ROW;    // 8 KiB
    constexpr int B_TILE_BYTES   = TILE_N * B_BYTES_ROW;    // 8 KiB
    constexpr int SFA_TILE_BYTES = TILE_M * SFA_BYTES_ROW;  // 1 KiB
    constexpr int SFB_TILE_BYTES = TILE_N * SFB_BYTES_ROW;  // 1 KiB

    const int K_half     = K / 2;
    const int K_groups   = K / 16;
    const int n_base     = n_tile * TILE_N;
    const int M_eff      = min(M_e, TILE_M);
    const int N_eff      = min(TILE_N, N - n_base);

    // SMEM layout: [A_TILE | B_TILE | SFA_TILE | SFB_TILE | D_TILE_FP16]
    // D_TILE not used by this kernel (direct global writes), but reserved per
    // task spec layout (32 KiB) so the SMEM footprint matches the documented
    // "8+8+1+1+32 KiB" budget. Keep zero-init off (no need).
    extern __shared__ uint8_t smem_raw[];
    uint8_t* smem_A   = smem_raw;
    uint8_t* smem_B   = smem_A   + A_TILE_BYTES;
    uint8_t* smem_SFA = smem_B   + B_TILE_BYTES;
    uint8_t* smem_SFB = smem_SFA + SFA_TILE_BYTES;
    // half* smem_D = reinterpret_cast<half*>(smem_SFB + SFB_TILE_BYTES);  // unused

    const int tid = threadIdx.x;
    const int n_threads = blockDim.x;
    // Warp-level identifiers. Single-warp version: only warp 0 issues MMAs.
    const int warp_id   = tid / 32;
    const int lane_id   = tid & 31;

    // ---- Stage 1: cp.async A tile. 8 KiB in 16-byte chunks = 512 ops / 256 = 2/thread.
    {
        constexpr int N_OPS = A_TILE_BYTES / 16;          // 512
        for (int op = tid; op < N_OPS; op += n_threads) {
            int row = op / (A_BYTES_ROW / 16);             // op / 4
            int chunk = op % (A_BYTES_ROW / 16);           // 0..3
            // Bounds: rows beyond M_eff just zero-load. cp.async can't zero;
            // for simplicity, we instead copy from a zero scratch via a
            // ternary ld + sync write. Out-of-bounds A rows are extremely
            // rare (M_eff < TILE_M only when expert M_e<128); just skip
            // copy and rely on SMEM init below.
            uint8_t* dst = smem_A + (size_t)row * A_BYTES_ROW + chunk * 16;
            if (row < M_eff) {
                const uint8_t* src = A_e + (size_t)row * K_half + chunk * 16;
                cp_async_cg_16_local(dst, src);
            } else {
                // Zero-fill: store 0s to SMEM via vector store. Safe because
                // dst is 16B-aligned (starts every 16B chunk).
                ulonglong2 z = {0ULL, 0ULL};
                *reinterpret_cast<ulonglong2*>(dst) = z;
            }
        }
    }

    // ---- Stage 2: cp.async B tile. 8 KiB / 16 / 256 = 2/thread.
    {
        constexpr int N_OPS = B_TILE_BYTES / 16;
        for (int op = tid; op < N_OPS; op += n_threads) {
            int row = op / (B_BYTES_ROW / 16);
            int chunk = op % (B_BYTES_ROW / 16);
            int n_global = n_base + row;
            uint8_t* dst = smem_B + (size_t)row * B_BYTES_ROW + chunk * 16;
            if (n_global < N) {
                const uint8_t* src = B_e + (size_t)n_global * K_half + chunk * 16;
                cp_async_cg_16_local(dst, src);
            } else {
                ulonglong2 z = {0ULL, 0ULL};
                *reinterpret_cast<ulonglong2*>(dst) = z;
            }
        }
    }

    // ---- Stage 3: cp.async SFA. 1 KiB row stride 8 → 1 op of 8 bytes per row;
    // 128 rows / 256 threads → many idle. Keep it simple with 8-byte cp.async.
    {
        constexpr int N_OPS = SFA_TILE_BYTES / 8;          // 128
        for (int op = tid; op < N_OPS; op += n_threads) {
            int row = op;
            uint8_t* dst = smem_SFA + (size_t)row * SFA_BYTES_ROW;
            if (row < M_eff) {
                const uint8_t* src = SFA_e + (size_t)row * K_groups;
                cp_async_ca_8_local(dst, src);
            } else {
                *reinterpret_cast<uint64_t*>(dst) = 0ULL;
            }
        }
    }

    // ---- Stage 4: cp.async SFB. Same shape.
    {
        constexpr int N_OPS = SFB_TILE_BYTES / 8;          // 128
        for (int op = tid; op < N_OPS; op += n_threads) {
            int row = op;
            int n_global = n_base + row;
            uint8_t* dst = smem_SFB + (size_t)row * SFB_BYTES_ROW;
            if (n_global < N) {
                const uint8_t* src = SFB_e + (size_t)n_global * K_groups;
                cp_async_ca_8_local(dst, src);
            } else {
                *reinterpret_cast<uint64_t*>(dst) = 0ULL;
            }
        }
    }

    cp_async_commit_local();
    cp_async_wait_group_local<0>();
    __syncthreads();

    // ---- MMA loop: warp 0 only, walks 8×16 sub-tiles, 2 K-stripes each.
    if (warp_id != 0) return;

    const int T0 = lane_id & 3;          // thread_in_group ∈ [0,4)
    const int T1 = lane_id >> 2;         // group_id ∈ [0,8)
    const int byte_offset = T0 * 4;
    const int m_sfa = T1 + (T0 & 1) * 8; // 16 unique m's (T0%2 broadcasts)
    const int n_sfb = T1;                // 8 unique n's

    constexpr int M_SUBTILES = TILE_M / 16;   // 8
    constexpr int N_SUBTILES = TILE_N / 8;    // 16
    constexpr int K_STRIPES  = TILE_K / 64;   // 2

    #pragma unroll 1
    for (int mi = 0; mi < M_SUBTILES; ++mi) {
        const int m_lo = mi * 16 + T1;        // m for a0, a2
        const int m_hi = m_lo + 8;            // m for a1, a3

        #pragma unroll 1
        for (int ni = 0; ni < N_SUBTILES; ++ni) {
            const int n_b = ni * 8 + T1;      // n for b0, b1

            float d[4] = {0.f, 0.f, 0.f, 0.f};

            #pragma unroll
            for (int ki = 0; ki < K_STRIPES; ++ki) {
                const int stripe_byte = ki * 32;
                const int kg_base = ki * 4;

                // A fragment (4 b32). Reads 4 bytes (8 nibbles) per register.
                uint32_t a0, a1, a2, a3;
                a0 = *reinterpret_cast<const uint32_t*>(
                    smem_A + (size_t)m_lo * A_BYTES_ROW + stripe_byte +  0 + byte_offset);
                a1 = *reinterpret_cast<const uint32_t*>(
                    smem_A + (size_t)m_hi * A_BYTES_ROW + stripe_byte +  0 + byte_offset);
                a2 = *reinterpret_cast<const uint32_t*>(
                    smem_A + (size_t)m_lo * A_BYTES_ROW + stripe_byte + 16 + byte_offset);
                a3 = *reinterpret_cast<const uint32_t*>(
                    smem_A + (size_t)m_hi * A_BYTES_ROW + stripe_byte + 16 + byte_offset);

                // B fragment (2 b32).
                uint32_t b0, b1;
                b0 = *reinterpret_cast<const uint32_t*>(
                    smem_B + (size_t)n_b * B_BYTES_ROW + stripe_byte +  0 + byte_offset);
                b1 = *reinterpret_cast<const uint32_t*>(
                    smem_B + (size_t)n_b * B_BYTES_ROW + stripe_byte + 16 + byte_offset);

                // Scale fragments (1 b32 each = 4 UE4M3 bytes).
                uint32_t sfa = *reinterpret_cast<const uint32_t*>(
                    smem_SFA + (size_t)(mi * 16 + m_sfa) * SFA_BYTES_ROW + kg_base);
                uint32_t sfb = *reinterpret_cast<const uint32_t*>(
                    smem_SFB + (size_t)(ni * 8  + n_sfb) * SFB_BYTES_ROW + kg_base);

                uint32_t a_arr[4] = {a0, a1, a2, a3};
                uint32_t b_arr[2] = {b0, b1};
                mma_sync_mxf4nvf4_m16n8k64(d, a_arr, b_arr, sfa, sfb);
            }

            // ---- Epilogue: SM80_16x8_Row (T32,V4)→(M16,N8). Apply alpha.
            const int m0 = mi * 16 + T1;
            const int m1 = m0 + 8;
            const int n0_local = ni * 8 + T0 * 2;
            const int n1_local = n0_local + 1;
            const int n0_g = n_base + n0_local;
            const int n1_g = n_base + n1_local;

            const float a0_out = d[0] * alpha;
            const float a1_out = d[1] * alpha;
            const float a2_out = d[2] * alpha;
            const float a3_out = d[3] * alpha;

            if (m0 < M_eff && n0_g < N)
                D_e[(size_t)m0 * N + n0_g] = __float2half(a0_out);
            if (m0 < M_eff && n1_g < N)
                D_e[(size_t)m0 * N + n1_g] = __float2half(a1_out);
            if (m1 < M_eff && n0_g < N)
                D_e[(size_t)m1 * N + n0_g] = __float2half(a2_out);
            if (m1 < M_eff && n1_g < N)
                D_e[(size_t)m1 * N + n1_g] = __float2half(a3_out);
        }
    }
}

}  // anonymous namespace

#ifdef SMALLM_SOFTWARE_REF
// Debug entry point: dispatch the SOFTWARE reference kernel via the same
// public API. Used by the SmallMKernel.HwMatchesSoftwareReference test.
extern "C" bool gemm_grouped_nvfp4_smallM_software_ref(
    int n_experts, const int* host_M, int N, int K,
    const void* const* host_ptr_A,   const void* const* host_ptr_SFA,
    const void* const* host_ptr_B,   const void* const* host_ptr_SFB,
    void* const* host_ptr_D,         const float* host_alpha,
    cudaStream_t stream) {
    if (!gemm_grouped_nvfp4_smallM_available()) return false;
    if (n_experts <= 0 || N <= 0 || K <= 0) return false;
    if ((K % 128) != 0 || (N % 128) != 0) return false;
    int max_M = 0;
    for (int e = 0; e < n_experts; ++e) max_M = std::max(max_M, host_M[e]);
    if (max_M > 128) return false;

    constexpr int TILE_M = 128, TILE_N = 128, TILE_K = 128;

    void** d_A = nullptr;   void** d_SFA = nullptr;
    void** d_B = nullptr;   void** d_SFB = nullptr;
    void** d_D = nullptr;   float* d_alpha = nullptr;
    int*   d_M = nullptr;
    cudaMallocAsync(&d_A,     sizeof(void*) * n_experts, stream);
    cudaMallocAsync(&d_SFA,   sizeof(void*) * n_experts, stream);
    cudaMallocAsync(&d_B,     sizeof(void*) * n_experts, stream);
    cudaMallocAsync(&d_SFB,   sizeof(void*) * n_experts, stream);
    cudaMallocAsync(&d_D,     sizeof(void*) * n_experts, stream);
    cudaMallocAsync(&d_alpha, sizeof(float) * n_experts, stream);
    cudaMallocAsync(&d_M,     sizeof(int)   * n_experts, stream);
    cudaMemcpyAsync(d_A,     host_ptr_A,   sizeof(void*) * n_experts, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_SFA,   host_ptr_SFA, sizeof(void*) * n_experts, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_B,     host_ptr_B,   sizeof(void*) * n_experts, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_SFB,   host_ptr_SFB, sizeof(void*) * n_experts, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_D,     host_ptr_D,   sizeof(void*) * n_experts, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_alpha, host_alpha,   sizeof(float) * n_experts, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_M,     host_M,       sizeof(int)   * n_experts, cudaMemcpyHostToDevice, stream);

    dim3 grid(n_experts, N / TILE_N);
    dim3 block(256);
    smallM_kernel_v1_software_ref<TILE_M, TILE_N, TILE_K><<<grid, block, 0, stream>>>(
        (const void* const*)d_A, (const void* const*)d_SFA,
        (const void* const*)d_B, (const void* const*)d_SFB,
        d_D, d_alpha, d_M, N, K);

    cudaFreeAsync(d_A, stream);   cudaFreeAsync(d_SFA, stream);
    cudaFreeAsync(d_B, stream);   cudaFreeAsync(d_SFB, stream);
    cudaFreeAsync(d_D, stream);   cudaFreeAsync(d_alpha, stream);
    cudaFreeAsync(d_M, stream);
    return true;
}
#endif  // SMALLM_SOFTWARE_REF

#ifdef SMALLM_TEST_HOOKS
namespace imp_test {

__global__ void smallM_smoke_single_mma_kernel(
    float* d_out, const uint32_t* a, const uint32_t* b,
    uint32_t sfa, uint32_t sfb) {
    if (threadIdx.x < 32) {  // single warp
        float acc[4] = {0.f, 0.f, 0.f, 0.f};
        mma_sync_mxf4nvf4_m16n8k64(acc, a, b, sfa, sfb);
        if (threadIdx.x == 0) {
            d_out[0] = acc[0]; d_out[1] = acc[1];
            d_out[2] = acc[2]; d_out[3] = acc[3];
        }
    }
}

}  // namespace imp_test

extern "C" void smallM_smoke_single_mma(
    float* d_out, const uint32_t* a, const uint32_t* b,
    uint32_t sfa, uint32_t sfb, cudaStream_t stream) {
    imp_test::smallM_smoke_single_mma_kernel<<<1, 32, 0, stream>>>(d_out, a, b, sfa, sfb);
}
#endif  // SMALLM_TEST_HOOKS

namespace detail {

using namespace cute;

// ---------------------------------------------------------------------------
// TMA descriptor builders
//
// Four separate descriptors per work-item — CUTLASS Sm120 NVFP4 pattern
// (validated by Phase 0 / commit a591dac): TMA_A + TMA_SFA + TMA_B + TMA_SFB.
//
// FP4 data is nibble-packed: a row of K elements occupies K/2 bytes.
// Block scales are 1 UE4M3 byte per group of 16 FP4 elements: K/16 bytes/row.
//
// All builders return `auto`; the return type is only instantiated when the
// kernel in Task 1.7 calls them with concrete template args.
// ---------------------------------------------------------------------------

// A matrix: M_e × K FP4, packed → shape (M_e, K/2) in uint8 byte space.
template <int TILE_M, int TILE_K>
auto build_tma_a(const void* d_ptr, int M_e, int K) {
    auto tensor = make_tensor(
        make_gmem_ptr(static_cast<const uint8_t*>(d_ptr)),
        make_layout(make_shape(M_e, K / 2), make_stride(K / 2, _1{})));
    auto smem_layout = make_layout(Shape<Int<TILE_M>, Int<TILE_K / 2>>{});
    return make_tma_copy(SM90_TMA_LOAD{}, tensor, smem_layout);
}

// B matrix: N × K FP4, packed → shape (N, K/2) in uint8 byte space.
template <int TILE_N, int TILE_K>
auto build_tma_b(const void* d_ptr, int N, int K) {
    auto tensor = make_tensor(
        make_gmem_ptr(static_cast<const uint8_t*>(d_ptr)),
        make_layout(make_shape(N, K / 2), make_stride(K / 2, _1{})));
    auto smem_layout = make_layout(Shape<Int<TILE_N>, Int<TILE_K / 2>>{});
    return make_tma_copy(SM90_TMA_LOAD{}, tensor, smem_layout);
}

// SFA (A block scales): M_e × (K/16) UE4M3 bytes.
template <int TILE_M, int TILE_K>
auto build_tma_sfa(const void* d_ptr, int M_e, int K) {
    auto tensor = make_tensor(
        make_gmem_ptr(static_cast<const uint8_t*>(d_ptr)),
        make_layout(make_shape(M_e, K / 16), make_stride(K / 16, _1{})));
    auto smem_layout = make_layout(Shape<Int<TILE_M>, Int<TILE_K / 16>>{});
    return make_tma_copy(SM90_TMA_LOAD{}, tensor, smem_layout);
}

// SFB (B block scales): N × (K/16) UE4M3 bytes.
template <int TILE_N, int TILE_K>
auto build_tma_sfb(const void* d_ptr, int N, int K) {
    auto tensor = make_tensor(
        make_gmem_ptr(static_cast<const uint8_t*>(d_ptr)),
        make_layout(make_shape(N, K / 16), make_stride(K / 16, _1{})));
    auto smem_layout = make_layout(Shape<Int<TILE_N>, Int<TILE_K / 16>>{});
    return make_tma_copy(SM90_TMA_LOAD{}, tensor, smem_layout);
}

int pick_m_tile(int M_e) {
    if (M_e <= 16) return 16;
    if (M_e <= 32) return 32;
    if (M_e <= 64) return 64;
    return 128;
}

std::vector<WorkItem> build_work_queue(int n_experts, const int* M_per, int N) {
    std::vector<WorkItem> q;
    q.reserve((size_t)n_experts * (size_t)((N + 127) / 128) + 8);
    for (int e = 0; e < n_experts; ++e) {
        if (M_per[e] <= 0) continue;
        int tm = pick_m_tile(M_per[e]);
        int nm = (M_per[e] + tm - 1) / tm;
        int nn = (N + 127) / 128;
        for (int mi = 0; mi < nm; ++mi)
            for (int ni = 0; ni < nn; ++ni)
                q.push_back({e, mi, ni, (uint8_t)tm});
    }
    std::stable_sort(q.begin(), q.end(),
        [](const WorkItem& a, const WorkItem& b) {
            return a.m_tile_size > b.m_tile_size;
        });
    return q;
}

}  // namespace detail

static int s_smallM_available = -1;

bool gemm_grouped_nvfp4_smallM_available() {
    if (s_smallM_available >= 0) return s_smallM_available;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    s_smallM_available = (prop.major * 10 + prop.minor >= 120) ? 1 : 0;
    return s_smallM_available;
}

void gemm_grouped_nvfp4_smallM_cleanup() {}

bool gemm_grouped_nvfp4_smallM(
    int n_experts, const int* host_M, int N, int K,
    const void* const* host_ptr_A,   const void* const* host_ptr_SFA,
    const void* const* host_ptr_B,   const void* const* host_ptr_SFB,
    void* const* host_ptr_D,         const float* host_alpha,
    cudaStream_t stream) {
    if (!gemm_grouped_nvfp4_smallM_available()) return false;
    if (n_experts <= 0 || N <= 0 || K <= 0) return false;
    if ((K % 128) != 0 || (N % 128) != 0) return false;

    // Phase A constraint: only support max_M ≤ 128 (single M-tile per expert).
    int max_M = 0;
    for (int e = 0; e < n_experts; ++e) max_M = std::max(max_M, host_M[e]);
    if (max_M > 128) return false;

    constexpr int TILE_M = 128, TILE_N = 128, TILE_K = 128;

    // Upload pointer arrays + M to device.
    void** d_A = nullptr;   void** d_SFA = nullptr;
    void** d_B = nullptr;   void** d_SFB = nullptr;
    void** d_D = nullptr;   float* d_alpha = nullptr;
    int*   d_M = nullptr;
    cudaMallocAsync(&d_A,     sizeof(void*) * n_experts, stream);
    cudaMallocAsync(&d_SFA,   sizeof(void*) * n_experts, stream);
    cudaMallocAsync(&d_B,     sizeof(void*) * n_experts, stream);
    cudaMallocAsync(&d_SFB,   sizeof(void*) * n_experts, stream);
    cudaMallocAsync(&d_D,     sizeof(void*) * n_experts, stream);
    cudaMallocAsync(&d_alpha, sizeof(float) * n_experts, stream);
    cudaMallocAsync(&d_M,     sizeof(int)   * n_experts, stream);

    cudaMemcpyAsync(d_A,     host_ptr_A,   sizeof(void*) * n_experts, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_SFA,   host_ptr_SFA, sizeof(void*) * n_experts, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_B,     host_ptr_B,   sizeof(void*) * n_experts, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_SFB,   host_ptr_SFB, sizeof(void*) * n_experts, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_D,     host_ptr_D,   sizeof(void*) * n_experts, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_alpha, host_alpha,   sizeof(float) * n_experts, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_M,     host_M,       sizeof(int)   * n_experts, cudaMemcpyHostToDevice, stream);

    dim3 grid(n_experts, N / TILE_N);
    dim3 block(256);

    // SMEM: A(8K) + B(8K) + SFA(1K) + SFB(1K) = 18 KiB. The 32 KiB D-epilogue
    // stub mentioned in T1.7b spec is omitted in this version because the
    // kernel writes directly to global; it'll be re-added in T1.10 when we
    // need to rotate epilogues across warps. 18 KiB stays under the static
    // SMEM limit so no cudaFuncSetAttribute opt-in is required.
    constexpr int SMEM_BYTES = 8192 + 8192 + 1024 + 1024;  // 18 KiB
    smallM_kernel_v1<TILE_M, TILE_N, TILE_K><<<grid, block, SMEM_BYTES, stream>>>(
        (const void* const*)d_A, (const void* const*)d_SFA,
        (const void* const*)d_B, (const void* const*)d_SFB,
        d_D, d_alpha, d_M, N, K);

    cudaFreeAsync(d_A, stream);   cudaFreeAsync(d_SFA, stream);
    cudaFreeAsync(d_B, stream);   cudaFreeAsync(d_SFB, stream);
    cudaFreeAsync(d_D, stream);   cudaFreeAsync(d_alpha, stream);
    cudaFreeAsync(d_M, stream);
    return true;
}

}  // namespace imp
