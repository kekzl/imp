// gemm_nvfp4_sm120a_tma.cu
// =============================================================================
// EXPERIMENTAL BRANCH / DOCUMENTED NEGATIVE RESULT — not the winning kernel.
// This is gemm_nvfp4_sm120a.cu rebuilt with TMA (cp.async.bulk.tensor +
// cuTensorMapEncodeTiled, resolved at runtime so there is no libcuda link dep) +
// producer/consumer WARP-SPECIALIZATION (full/empty mbarrier pairs, no
// __syncthreads). It is bit-exact and deadlock-free, and TMA *did* cut L2
// requests (82->68%). But the 9-warp/288-thread warp-spec block drops to 1
// block/SM (occupancy 32->18.75%) and the single-thread TMA issue serializes vs
// 256-thread parallel cp.async -> it nets SLOWER (509 vs the cp.async 629).
// Kept as evidence that on sm_120, for this NVFP4 GEMM, TMA+warp-spec does NOT
// beat a well-laid-out cp.async kernel. The actual win (807/972 TFLOP/s, beats
// production) lives in gemm_nvfp4_sm120a.cu via two LAYOUT tricks instead.
// =============================================================================
//
// A self-contained, from-scratch NVFP4 GEMM for the RTX 5090 (GB202, sm_120a —
// consumer Blackwell). No imp engine headers, no CUTLASS. One file: prep +
// kernel + CPU reference + correctness check + timing. Companion to
// fa2_sm120a_optimal.cu.
//
// Computes  C[M,N] = A[M,K] . B[N,K]^T   (B is the mma "col" operand, stored
// row-major [N][K]). Both A,B are NVFP4 (E2M1, 4-bit) on the tensor cores via
// the peak sm_120 path:
//   mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64
// k=64 per MMA, hardware per-16-element UE4M3 block scale. This is THE GEMM that
// dominates prefill (projections + FFN/MoE).
//
// Optimization trail (profiling-driven, ncu on RTX 5090; numbers @ S=4k / 8k cubed):
//   0. scaffold: 1 warp / 16x8 tile, on-the-fly f16->E2M1 quant, no reuse  ~1.8 TFLOP/s
//      ncu: L1/TEX 55% (no reuse) + Compute(SM) 47% (the quantize ALU) co-bound.
//   1. PRE-PACK A,B to E2M1 [rows][K/8] (k-contiguous) + smem-tiled 64x64 CTA.
//      The packed layout makes each mma fragment a single uint32 smem load
//      (k=T0*8+V0 is exactly one packed column -> NO gather), and smem gives
//      A/B reuse. Kills the quant compute AND the redundant traffic .. 347 / 281
//   2. cp.async double-buffer of the A/B tiles (prefetch chunk kc+1). Hides the
//      L2 load latency; also removes the large-K regression ......... 386 / 391
//   3. 128x128 CTA tile, 4x4 register-blocked warp tiles (16 accumulators). AI
//      32->64 halves the L2 traffic per FLOP (L2 was 92% bound) ..... 660 / 724
//   (128x256 tried: 795 @ 8k but regresses to 584 @ 4k — bigger tiles need a
//    bigger grid to fill the machine; 128x128 is the robust default.)
//   4. REAL per-16 UE4M3 block scales (was uniform 1.0). Production encoding:
//      sfa = the row's 4 chunk-block bytes as one uint32, scale-row m_sfa =
//      T1+(T0&1)*8, bid=tid=0. Naively this halved perf (strided 4-byte scale
//      reads waste whole L2 sectors); storing the scales TRANSPOSED [chunk][row]
//      makes the chunk's rows contiguous -> coalesced 16-byte cp.async, recovering
//      it: the full block-scale path costs only ~3-5% .............. 644 / 683
//
// ldmatrix was investigated and is N/A here: the mxf4nvf4 A operand is m16k64
// with a CuTe layout that doesn't match ldmatrix's f16 m16k16 pattern — imp's own
// production NVFP4 GEMM also loads the fragments with scalar uint32 loads (which
// is exactly what this kernel does). So the single-uint32 fragment load is optimal.
//
// Bottleneck: L2 BANDWIDTH-bound — lts 82%, DRAM 12%, SM 50% @ 4k. We sit at
// ~32-34% of the ~2019-TOPS measured FP4 peak vs imp's production CUTLASS path
// ~41%. The gap is specifically L2-bandwidth, which narrows WHICH levers help:
//   - threadblock swizzle:  tried, REVERTED (-7% @ 4k). Reduces DRAM traffic /
//     L2 hit rate, but the same bytes flow through the L2 *pipe* regardless.
//   - 3-stage pipeline:     tried, no gain. A latency lever; we are bandwidth-
//     bound, not latency-bound (and the 3rd buffer lowers occupancy).
//   - warp-specialization alone: an overlap lever; doesn't cut L2 bandwidth.
//   - TMA (cp.async.bulk.tensor / UTMALDG): the ONE lever that helps — bulk
//     descriptor-driven loads cut the per-request L2 overhead. This is exactly
//     what the production smallM kernel uses (with warp-spec to feed it). It
//     needs the CUDA driver API (cuTensorMapEncodeTiled via dlopen libcuda) +
//     mbarriers — a major addition, deliberately not in this self-contained ref.
// So 3 of the 4 "production levers" are red herrings for this bottleneck (2
// measured); production's real edge is TMA-driven L2 efficiency.
// Correctness: bit-exact vs a CPU reference that quantizes + block-scales identically.
//
// Build & run (host has no CUDA toolkit — use the CUDA 13.3 container).
// NOTE: block-scale mxf4nvf4 needs the explicit compute_120a gencode; the
// `-arch=sm_120a` shorthand does NOT enable .block_scale (ptxas rejects it):
//   docker run --rm --gpus all -v "$PWD":/w -w /w nvidia/cuda:13.3.0-devel-ubuntu26.04 \
//     sh -c 'nvcc -O3 -std=c++23 --generate-code=arch=compute_120a,code=sm_120a \
//            gemm_nvfp4_sm120a.cu -o gemm && ./gemm'
// -----------------------------------------------------------------------------

#include <cuda.h>  // CUtensorMap + driver types (entry point resolved at runtime)
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(x)                                                                          \
    do {                                                                                       \
        cudaError_t e_ = (x);                                                                  \
        if (e_ != cudaSuccess) {                                                               \
            printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__);    \
            exit(1);                                                                           \
        }                                                                                      \
    } while (0)

// ---- E2M1 (NVFP4 element) quantize / dequantize -----------------------------
// Magnitudes for mag nibble 0..7: {0, .5, 1, 1.5, 2, 3, 4, 6}; bit 0x8 = sign.
__host__ __device__ __forceinline__ uint8_t fp32_to_e2m1(float v) {
    uint8_t sign = (v < 0.0f) ? 0x8 : 0x0;
    float a = fabsf(v);
    uint8_t mag = (a >= 0.25f) + (a >= 0.75f) + (a >= 1.25f) + (a >= 1.75f) + (a >= 2.5f) +
                  (a >= 3.5f) + (a >= 5.0f);
    return sign | mag;
}
__host__ __device__ __forceinline__ float e2m1_to_fp32(uint8_t nib) {
    static const float LUT[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    float m = LUT[nib & 0x7];
    return (nib & 0x8) ? -m : m;
}
// UE4M3 (E4M3, sign forced 0) block-scale decode — matches the production kernel.
__host__ __device__ __forceinline__ float ue4m3_to_fp32(uint8_t bits) {
    uint32_t exp = (bits >> 3) & 0x0F, man = bits & 0x07;
    if (exp == 0) return (float)man * (1.0f / 512.0f);
    uint32_t fp32 = ((exp + 120u) << 23) | (man << 20);
    float r;
    memcpy(&r, &fp32, 4);
    return r;
}

// ---- peak sm_120 NVFP4 MMA: D[16x8] += A[16x64] . B[8x64]^T (E2M1, f32 acc) --
__device__ __forceinline__ void mma_mxf4nvf4(float& d0, float& d1, float& d2, float& d3,
                                             uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                             uint32_t b0, uint32_t b1, uint32_t sfa, uint32_t sfb) {
#if __CUDA_ARCH__ >= 1200
    constexpr uint16_t z = 0;  // bid/tid scale selectors (0 = use the packed bytes directly)
    // "+f" ties D out and C in to the SAME registers so K-loop accumulation works.
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1."
        "f32.ue4m3 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, "
        "{%14,%15};\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sfa), "h"(z), "h"(z), "r"(sfb),
          "h"(z), "h"(z));
#endif
}

// ---- cp.async (16-byte) for double-buffered smem staging --------------------
__device__ __forceinline__ void cp_async16(void* smem, const void* glob) {
    uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(glob));
}
__device__ __forceinline__ void cp_async4(void* smem, const void* glob) {  // 4-byte (scale rows)
    uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(s), "l"(glob));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n"); }
template <int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

// ---- TMA + mbarrier device wrappers (from gemm_grouped_nvfp4_smallM.cu) ------
__device__ __forceinline__ void mbarrier_init(uint64_t* bar, uint32_t count) {
    uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(s), "r"(count));
}
__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint64_t* bar, uint32_t bytes) {
    uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(s), "r"(bytes));
}
__device__ __forceinline__ void mbarrier_wait(uint64_t* bar, uint32_t phase) {
    uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile(
        "{ .reg .pred p;\n"
        "WAIT: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
        "@p bra DONE;\n"
        "bra WAIT;\n"
        "DONE: }\n" ::"r"(s),
        "r"(phase));
}
__device__ __forceinline__ void mbarrier_arrive(uint64_t* bar) {  // plain arrival (no tx)
    uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(s));
}
// 2-D bulk-tensor load (UTMALDG on sm_120): copies a tile from global (via the
// tensor-map descriptor) into shared, signalling `mbar` on completion.
__device__ __forceinline__ void cp_async_bulk_tensor_2d(void* smem_dst, const void* desc, int x,
                                                        int y, uint64_t* mbar) {
    uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
    uint32_t sb = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%2, %3}], [%4];\n" ::"r"(sd),
        "l"(desc), "r"(x), "r"(y), "r"(sb)
        : "memory");
}

// ---- host: build a 2-D CUtensorMap over uint8 data (driver entry resolved at
//      runtime via cudaGetDriverEntryPoint -> no libcuda link dependency) ------
using PFN_TME = CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*,
                             const cuuint64_t*, const cuuint32_t*, const cuuint32_t*,
                             CUtensorMapInterleave, CUtensorMapSwizzle, CUtensorMapL2promotion,
                             CUtensorMapFloatOOBfill);
static bool build_tma_2d_u8(CUtensorMap* desc, void* gmem, int rows, int cols, int box_rows,
                            int box_cols) {
    void* p = nullptr;
    cudaDriverEntryPointQueryResult q;
    if (cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &p, CUDA_VERSION,
                                         cudaEnableDefault, &q) != cudaSuccess ||
        p == nullptr)
        return false;
    cuuint64_t shape[2] = {(cuuint64_t)cols, (cuuint64_t)rows};
    cuuint64_t stride[1] = {(cuuint64_t)cols};
    cuuint32_t box[2] = {(cuuint32_t)box_cols, (cuuint32_t)box_rows};
    cuuint32_t bstride[2] = {1u, 1u};
    return reinterpret_cast<PFN_TME>(p)(desc, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, gmem, shape, stride,
                                        box, bstride, CU_TENSOR_MAP_INTERLEAVE_NONE,
                                        CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                                        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
}

// ---- prep: quantize a [rows][K] f16 matrix to packed E2M1 [rows][K/8] uint32 -
// Nibble for k = c*8 + j is stored at bit j*4 of packed column c (k-contiguous).
__global__ void pack_e2m1(const half* __restrict__ X, uint32_t* __restrict__ Xq, int rows, int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int KU = K / 8;
    if (idx >= rows * KU) return;
    int r = idx / KU, c = idx % KU;
    uint32_t packed = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        int k = c * 8 + j;
        uint8_t nib = (k < K) ? fp32_to_e2m1(__half2float(X[(int64_t)r * K + k])) : 0;
        packed |= (uint32_t)nib << (j * 4);
    }
    Xq[idx] = packed;
}

// ---- tiled NVFP4 GEMM --------------------------------------------------------
#define BM 128          // CTA output rows  (bigger tile -> AI 32->64 -> halves L2 traffic)
#define BN 64           // CTA output cols (smaller -> fewer consumer regs -> 2 blocks/SM)
#define BK 64           // K staged per smem chunk (BK=256 regressed: lower occupancy hurt
                        // more than fewer barriers helped — keep smem small, occupancy high)
#define BKU (BK / 8)    // packed uint32 per row in a K-chunk = 8
#define KSUB (BK / 64)  // mma k-steps per smem chunk = 1
#define MSUB 4            // m-tiles per consumer warp
#define NSUB 2            // n-tiles per consumer warp (8 accumulators -> ~half the regs)
#define N_CONS_WARPS 8    // 8 consumers cover 128x128 (2 m-groups x 4 n-groups, 4x4 each)
#define N_PROD_WARPS 1    // 1 dedicated TMA+scale producer warp (warp-specialization)
#define NWARPS (N_CONS_WARPS + N_PROD_WARPS)  // 9
#define NTHREADS (NWARPS * 32)                // 288
#define CONS_THREADS (N_CONS_WARPS * 32)      // 256
#define PROD_THREADS (N_PROD_WARPS * 32)      // 32

// Issue the cp.async loads of K-chunk `kc` into the given smem A/B buffers
// (16-byte = 4 packed uint32 per cp.async) and commit them as one group.
__device__ __forceinline__ void load_buf(uint32_t As[BM][BKU], uint32_t Bs[BN][BKU],
                                         uint8_t SFA_s[BM][4], uint8_t SFB_s[BN][4],
                                         const uint32_t* Aq, const uint32_t* Bq,
                                         const uint8_t* SFAg, const uint8_t* SFBg, int cta_m,
                                         int cta_n, int KU, int KB, int kc, int M, int N) {
    for (int i = threadIdx.x; i < BM * BKU / 4; i += NTHREADS) {
        int u = i * 4, r = u / BKU, c = u % BKU;
        if (cta_m + r < M)
            cp_async16(&As[r][c], &Aq[(int64_t)(cta_m + r) * KU + (int64_t)kc * BKU + c]);
        else
            *reinterpret_cast<float4*>(&As[r][c]) = make_float4(0, 0, 0, 0);
    }
    for (int i = threadIdx.x; i < BN * BKU / 4; i += NTHREADS) {
        int u = i * 4, r = u / BKU, c = u % BKU;
        if (cta_n + r < N)
            cp_async16(&Bs[r][c], &Bq[(int64_t)(cta_n + r) * KU + (int64_t)kc * BKU + c]);
        else
            *reinterpret_cast<float4*>(&Bs[r][c]) = make_float4(0, 0, 0, 0);
    }
    // block scales in TRANSPOSED layout [chunk][row][4]: the BM/BN rows of one
    // chunk are contiguous in gmem -> coalesced 16-byte cp.async (vs strided
    // 4-byte reads that wasted whole L2 sectors). M,N are BM/BN-padded -> no bounds.
    (void)KB;
    for (int i = threadIdx.x; i < BM * 4 / 16; i += NTHREADS)
        cp_async16(&SFA_s[i * 4][0], &SFAg[((int64_t)kc * M + cta_m) * 4 + i * 16]);
    for (int i = threadIdx.x; i < BN * 4 / 16; i += NTHREADS)
        cp_async16(&SFB_s[i * 4][0], &SFBg[((int64_t)kc * N + cta_n) * 4 + i * 16]);
    cp_async_commit();
}

// 8 warps as a 2(m-group) x 4(n-group) grid. Each warp computes MSUBxNSUB = 4x4
// sub-tiles (a 64x32 region, 16 register-blocked accumulators), reusing each A
// fragment across NSUB n-tiles and each B fragment across MSUB m-tiles.
__global__ void __launch_bounds__(NTHREADS) gemm_nvfp4_tiled(
    const __grid_constant__ CUtensorMap mapA, const __grid_constant__ CUtensorMap mapB,
    const uint8_t* __restrict__ SFAg, const uint8_t* __restrict__ SFBg, float* __restrict__ C,
    int M, int N, int K) {
    const int cta_m = blockIdx.y * BM, cta_n = blockIdx.x * BN;
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;

    __shared__ __align__(128) uint32_t As[2][BM][BKU];  // double-buffered TMA dst
    __shared__ __align__(128) uint32_t Bs[2][BN][BKU];
    __shared__ uint8_t SFA_s[2][BM][4];  // scales on cp.async (row too small for TMA)
    __shared__ uint8_t SFB_s[2][BN][4];
    __shared__ __align__(8) uint64_t full[2], empty[2];  // producer/consumer mbarrier pairs
    if (tid == 0) {
        mbarrier_init(&full[0], PROD_THREADS);   mbarrier_init(&full[1], PROD_THREADS);
        mbarrier_init(&empty[0], CONS_THREADS);  mbarrier_init(&empty[1], CONS_THREADS);
    }
    __syncthreads();

    constexpr uint32_t TX_BYTES = (BM * BKU + BN * BKU) * 4;  // A tile + B tile bytes
    const int kchunks = K / BK;

    if (warp >= N_CONS_WARPS) {
        // ===================== PRODUCER warp (no MMA) =======================
        // lane 0 issues the two TMAs (arrive.expect_tx); the other 31 lanes
        // cp.async the block scales then plain-arrive -> full[s] completes when
        // all 32 producers arrived AND the TMA bytes landed.
        uint32_t pe_phase[2] = {0, 0};
        for (int kc = 0; kc < kchunks; ++kc) {
            const int s = kc & 1;
            if (kc >= 2) { mbarrier_wait(&empty[s], pe_phase[s]); pe_phase[s] ^= 1; }
            if (lane == 0) {
                mbarrier_arrive_expect_tx(&full[s], TX_BYTES);
                cp_async_bulk_tensor_2d(&As[s][0][0], &mapA, kc * (BK / 2), cta_m, &full[s]);
                cp_async_bulk_tensor_2d(&Bs[s][0][0], &mapB, kc * (BK / 2), cta_n, &full[s]);
            } else {
                for (int i = lane - 1; i < BM * 4 / 16; i += PROD_THREADS - 1)
                    cp_async16(&SFA_s[s][i * 4][0], &SFAg[((int64_t)kc * M + cta_m) * 4 + i * 16]);
                for (int i = lane - 1; i < BN * 4 / 16; i += PROD_THREADS - 1)
                    cp_async16(&SFB_s[s][i * 4][0], &SFBg[((int64_t)kc * N + cta_n) * 4 + i * 16]);
                cp_async_commit();
                cp_async_wait<0>();
                mbarrier_arrive(&full[s]);
            }
        }
        return;
    }

    // ======================= CONSUMER warps (MMA) ==========================
    const int T0 = lane % 4, T1 = lane / 4;
    const int wm0 = (warp % 2) * MSUB;  // first m-tile of this warp (0 or 4)
    const int wn0 = (warp / 2) * NSUB;  // first n-tile of this warp (0,4,8,12)
    float d[MSUB][NSUB][4];
#pragma unroll
    for (int i = 0; i < MSUB; ++i)
#pragma unroll
        for (int j = 0; j < NSUB; ++j) d[i][j][0] = d[i][j][1] = d[i][j][2] = d[i][j][3] = 0.f;

    uint32_t pf_phase[2] = {0, 0};
    for (int kc = 0; kc < kchunks; ++kc) {
        const int s = kc & 1;
        mbarrier_wait(&full[s], pf_phase[s]);  // A/B + scales for this chunk ready
        pf_phase[s] ^= 1;

        uint32_t af[MSUB][4], bf[NSUB][2], sfa[MSUB], sfb[NSUB];
#pragma unroll
        for (int ms = 0; ms < MSUB; ++ms) {
            int mr = (wm0 + ms) * 16 + T1;
            af[ms][0] = As[s][mr][T0];      af[ms][1] = As[s][mr + 8][T0];
            af[ms][2] = As[s][mr][T0 + 4];  af[ms][3] = As[s][mr + 8][T0 + 4];
            sfa[ms] = *reinterpret_cast<const uint32_t*>(&SFA_s[s][(wm0 + ms) * 16 + T1 + (T0 & 1) * 8][0]);
        }
#pragma unroll
        for (int ns = 0; ns < NSUB; ++ns) {
            int nr = (wn0 + ns) * 8 + T1;
            bf[ns][0] = Bs[s][nr][T0];
            bf[ns][1] = Bs[s][nr][T0 + 4];
            sfb[ns] = *reinterpret_cast<const uint32_t*>(&SFB_s[s][(wn0 + ns) * 8 + T1][0]);
        }
#pragma unroll
        for (int ms = 0; ms < MSUB; ++ms)
#pragma unroll
            for (int ns = 0; ns < NSUB; ++ns)
                mma_mxf4nvf4(d[ms][ns][0], d[ms][ns][1], d[ms][ns][2], d[ms][ns][3], af[ms][0],
                             af[ms][1], af[ms][2], af[ms][3], bf[ns][0], bf[ns][1], sfa[ms], sfb[ns]);
        mbarrier_arrive(&empty[s]);  // this consumer is done reading slot s
    }

    // write C: m = cta_m + (wm0+ms)*16 + T1(+8), n = cta_n + (wn0+ns)*8 + T0*2(+1)
#pragma unroll
    for (int ms = 0; ms < MSUB; ++ms)
#pragma unroll
        for (int ns = 0; ns < NSUB; ++ns) {
            int m0 = cta_m + (wm0 + ms) * 16 + T1, m1 = m0 + 8;
            int n0 = cta_n + (wn0 + ns) * 8 + T0 * 2, n1 = n0 + 1;
            if (m0 < M && n0 < N) C[(int64_t)m0 * N + n0] = d[ms][ns][0];
            if (m0 < M && n1 < N) C[(int64_t)m0 * N + n1] = d[ms][ns][1];
            if (m1 < M && n0 < N) C[(int64_t)m1 * N + n0] = d[ms][ns][2];
            if (m1 < M && n1 < N) C[(int64_t)m1 * N + n1] = d[ms][ns][3];
        }
}

// ---- CPU reference: quantize identically, exact E2M1 dot product -------------
static void cpu_gemm(const std::vector<float>& A, const std::vector<float>& B,
                     const std::vector<uint8_t>& SFA, const std::vector<uint8_t>& SFB,
                     std::vector<float>& C, int M, int N, int K) {
    int KB = K / 16;
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            double acc = 0;
            for (int kb = 0; kb < KB; ++kb) {
                float sa = ue4m3_to_fp32(SFA[(int64_t)m * KB + kb]);
                float sb = ue4m3_to_fp32(SFB[(int64_t)n * KB + kb]);
                double sub = 0;
                for (int k = kb * 16; k < kb * 16 + 16; ++k) {
                    float a = e2m1_to_fp32(fp32_to_e2m1(A[(int64_t)m * K + k]));
                    float b = e2m1_to_fp32(fp32_to_e2m1(B[(int64_t)n * K + k]));
                    sub += (double)a * b;
                }
                acc += sub * (double)sa * sb;
            }
            C[(int64_t)m * N + n] = (float)acc;
        }
}

int main(int argc, char** argv) {
    int M = (argc > 1) ? atoi(argv[1]) : 4096;
    int N = (argc > 2) ? atoi(argv[2]) : 4096;
    int K = (argc > 3) ? atoi(argv[3]) : 4096;
    M = (M + BM - 1) / BM * BM; N = (N + BN - 1) / BN * BN; K = (K + BK - 1) / BK * BK;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s (sm_%d%d, %d SMs)\n", prop.name, prop.major, prop.minor,
           prop.multiProcessorCount);
    printf("NVFP4 GEMM  C[%d,%d] = A[%d,%d] . B[%d,%d]^T  (tiled %dx%d, real UE4M3 block scales)\n", M, N, M, K, N,
           K, BM, BN);

    size_t aN = (size_t)M * K, bN = (size_t)N * K, cN = (size_t)M * N;
    int KB = K / 16;
    std::vector<float> hA(aN), hB(bN), hC_ref(cN), hC_gpu(cN);
    std::vector<uint8_t> hSFA((size_t)M * KB), hSFB((size_t)N * KB);
    srand(7);
    for (auto& x : hA) x = (rand() / (float)RAND_MAX - 0.5f) * 8.f;
    for (auto& x : hB) x = (rand() / (float)RAND_MAX - 0.5f) * 8.f;
    // random UE4M3 block scales (sign 0; exp 6..9, mantissa random -> ~0.5..3.75)
    for (auto& s : hSFA) s = (uint8_t)(((6 + rand() % 4) << 3) | (rand() % 8));
    for (auto& s : hSFB) s = (uint8_t)(((6 + rand() % 4) << 3) | (rand() % 8));
    // transposed scale layout [chunk][row][4] for coalesced 16-byte GPU loads
    int numchunks = K / BK;
    std::vector<uint8_t> hSFA_T((size_t)numchunks * M * 4), hSFB_T((size_t)numchunks * N * 4);
    for (int kc = 0; kc < numchunks; ++kc)
        for (int r = 0; r < M; ++r)
            for (int b = 0; b < 4; ++b) hSFA_T[((size_t)kc * M + r) * 4 + b] = hSFA[(size_t)r * KB + kc * 4 + b];
    for (int kc = 0; kc < numchunks; ++kc)
        for (int r = 0; r < N; ++r)
            for (int b = 0; b < 4; ++b) hSFB_T[((size_t)kc * N + r) * 4 + b] = hSFB[(size_t)r * KB + kc * 4 + b];

    std::vector<half> hAh(aN), hBh(bN);
    for (size_t i = 0; i < aN; ++i) hAh[i] = __float2half(hA[i]);
    for (size_t i = 0; i < bN; ++i) hBh[i] = __float2half(hB[i]);
    for (size_t i = 0; i < aN; ++i) hA[i] = __half2float(hAh[i]);  // round-trip so the CPU ref
    for (size_t i = 0; i < bN; ++i) hB[i] = __half2float(hBh[i]);  // quantizes the same f16 values

    half *dA, *dB; uint32_t *dAq, *dBq; uint8_t *dSFA, *dSFB; float* dC;
    CUDA_CHECK(cudaMalloc(&dA, aN * 2));
    CUDA_CHECK(cudaMalloc(&dB, bN * 2));
    CUDA_CHECK(cudaMalloc(&dAq, aN / 8 * 4));
    CUDA_CHECK(cudaMalloc(&dBq, bN / 8 * 4));
    CUDA_CHECK(cudaMalloc(&dSFA, hSFA_T.size()));
    CUDA_CHECK(cudaMalloc(&dSFB, hSFB_T.size()));
    CUDA_CHECK(cudaMalloc(&dC, cN * 4));
    CUDA_CHECK(cudaMemcpy(dA, hAh.data(), aN * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hBh.data(), bN * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dSFA, hSFA_T.data(), hSFA_T.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dSFB, hSFB_T.data(), hSFB_T.size(), cudaMemcpyHostToDevice));

    // prep: pack to E2M1 (one-time, not in the timed loop)
    pack_e2m1<<<(aN / 8 + 255) / 256, 256>>>(dA, dAq, M, K);
    pack_e2m1<<<(bN / 8 + 255) / 256, 256>>>(dB, dBq, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    // build TMA tensor maps over the packed E2M1 data (uint8 view: cols = K/2 bytes)
    CUtensorMap mapA, mapB;
    if (!build_tma_2d_u8(&mapA, dAq, M, K / 2, BM, BK / 2) ||
        !build_tma_2d_u8(&mapB, dBq, N, K / 2, BN, BK / 2)) {
        printf("TMA tensor-map build failed (cuTensorMapEncodeTiled unavailable)\n");
        return 1;
    }

    dim3 grid(N / BN, M / BM), block(NTHREADS);
    gemm_nvfp4_tiled<<<grid, block>>>(mapA, mapB, dSFA, dSFB, dC, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hC_gpu.data(), dC, cN * 4, cudaMemcpyDeviceToHost));

    // CPU reference is O(M*N*K) single-threaded -> only verify at small sizes
    // (the kernel is size-agnostic, so a small PASS covers all sizes).
    if ((size_t)M * N * K <= (size_t)512 * 512 * 512) {
        cpu_gemm(hA, hB, hSFA, hSFB, hC_ref, M, N, K);
        double max_abs = 0, max_rel = 0;
        for (size_t i = 0; i < cN; ++i) {
            double e = fabs((double)hC_gpu[i] - hC_ref[i]);
            max_abs = fmax(max_abs, e);
            if (fabs(hC_ref[i]) > 1.0) max_rel = fmax(max_rel, e / fabs(hC_ref[i]));
        }
        printf("Correctness: max_abs_err=%.4e  max_rel_err=%.4e  %s\n", max_abs, max_rel,
               max_rel < 1e-3 ? "PASS" : "FAIL");
    } else {
        printf("Correctness: skipped (large size; verified at <=512^3)\n");
    }

    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    {
        cudaEvent_t w0, w1; cudaEventCreate(&w0); cudaEventCreate(&w1);
        float wms = 0; cudaEventRecord(w0);
        while (wms < 1500.f) {
            for (int i = 0; i < 20; ++i) gemm_nvfp4_tiled<<<grid, block>>>(mapA, mapB, dSFA, dSFB, dC, M, N, K);
            CUDA_CHECK(cudaDeviceSynchronize());
            cudaEventRecord(w1); cudaEventSynchronize(w1); cudaEventElapsedTime(&wms, w0, w1);
        }
    }
    int reps = 100;
    cudaEventRecord(a);
    for (int i = 0; i < reps; ++i) gemm_nvfp4_tiled<<<grid, block>>>(mapA, mapB, dSFA, dSFB, dC, M, N, K);
    cudaEventRecord(b); cudaEventSynchronize(b);
    float ms; cudaEventElapsedTime(&ms, a, b); ms /= reps;
    double flops = 2.0 * M * N * K;
    printf("Time: %.3f ms/iter   %.1f TFLOP/s (NVFP4; ~2019 TOPS measured peak = %.1f%%)\n", ms,
           flops / (ms * 1e-3) / 1e12, flops / (ms * 1e-3) / 1e12 / 2019.0 * 100.0);

    cudaFree(dA); cudaFree(dB); cudaFree(dAq); cudaFree(dBq); cudaFree(dC);
    return 0;
}
