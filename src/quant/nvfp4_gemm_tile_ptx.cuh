#pragma once

// PTX primitives of the own NVFP4 prefill GEMM (nvfp4_gemm_tile.cu): cp.async, mbarrier, TMA,
// block-scaled MMA, ldmatrix. sm_120a.

#include <cuda.h>
#include <cstdint>

namespace imp::nvfp4_tile {

__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, int src_bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "r"(src_bytes));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;"); }
template <int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;" ::"n"(N));
}

__device__ __forceinline__ void mbar_init(uint32_t bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" ::"r"(bar), "r"(count));
}
__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
    asm volatile("mbarrier.arrive.shared.b64 _, [%0];" ::"r"(bar) : "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint32_t bar, uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" ::"r"(bar), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
    asm volatile(
        "{\n\t.reg .pred P;\n"
        "TILE_WAIT_%=:\n\t"
        "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n\t"
        "@!P bra TILE_WAIT_%=;\n\t}" ::"r"(bar),
        "r"(parity)
        : "memory");
}
__device__ __forceinline__ void tma_2d(uint32_t dst, const CUtensorMap* map, int x, int y, uint32_t bar) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], "
        "[%4];" ::"r"(dst),
        "l"(map), "r"(x), "r"(y), "r"(bar)
        : "memory");
}
__device__ __forceinline__ void bulk_1d(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(dst),
        "l"(src), "r"(bytes), "r"(bar)
        : "memory");
}

__device__ __forceinline__ void mma_mxf4nvf4(float acc[4], const uint32_t a[4], uint32_t b0, uint32_t b1,
                                             uint32_t sfa, uint32_t sfb) {
#if __CUDA_ARCH__ >= 1200
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, "
        "%15};\n"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sfa),
          "h"(static_cast<uint16_t>(0)), "h"(static_cast<uint16_t>(0)), "r"(sfb),
          "h"(static_cast<uint16_t>(0)), "h"(static_cast<uint16_t>(0)));
#endif
}

// MXFP4 m16n8k64: the NVFP4 fragments, two UE8M0 per row / column in bytes 0-1 (one per 32 K).
__device__ __forceinline__ void mma_mxf4(float acc[4], const uint32_t a[4], uint32_t b0, uint32_t b1,
                                         uint32_t sfa, uint32_t sfb) {
#if __CUDA_ARCH__ >= 1200
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, "
        "%15};\n"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sfa),
          "h"(static_cast<uint16_t>(0)), "h"(static_cast<uint16_t>(0)), "r"(sfb),
          "h"(static_cast<uint16_t>(0)), "h"(static_cast<uint16_t>(0)));
#endif
}

// MXFP8 m16n8k32: E4M3 x E4M3, one UE8M0 per row / column in byte 0 (CUTLASS SM120_16x8x32_TN_VS);
// A/B fragments are the m16n8k64 FP4 ones byte for byte.
__device__ __forceinline__ void mma_mxf8(float acc[4], const uint32_t a[4], uint32_t b0, uint32_t b1,
                                         uint32_t sfa, uint32_t sfb) {
#if __CUDA_ARCH__ >= 1200
    asm volatile(
        "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, "
        "%15};\n"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sfa),
          "h"(static_cast<uint16_t>(0)), "h"(static_cast<uint16_t>(0)), "r"(sfb),
          "h"(static_cast<uint16_t>(0)), "h"(static_cast<uint16_t>(0)));
#endif
}

// ldmatrix x4: lane l addresses row l%8 of 8x16 B matrix l/8; register j of every lane is the
// u32 at (row T1, byte T0*4) of matrix j, the m16n8k64 fragment order.
__device__ __forceinline__ void ldsm_x4(uint32_t r[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}

// Host: 2D uint8 TMA map over [rows, row_bytes], box [box_bytes, box_rows], 64B/128B swizzle by
// box_bytes (nvfp4_gemm_tile.cu).
[[nodiscard]] bool encode_2d(CUtensorMap* map, const void* base, int rows, int row_bytes, int box_bytes,
                             int box_rows);

}  // namespace imp::nvfp4_tile
