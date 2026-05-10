// src/compute/gemm_grouped_nvfp4_smallM.cu
#include "compute/gemm_grouped_nvfp4_smallM.h"
#include "core/logging.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <vector>
#include <cstdint>
#include <cstdio>

#include "cute/tensor.hpp"
#include "cute/atom/copy_atom.hpp"
#include "cute/atom/copy_traits_sm90_tma.hpp"  // SM90_TMA_LOAD / make_tma_copy

namespace imp {

namespace {

// cp.async helpers — retained for the small SFA/SFB tiles where TMA's 16-byte
// minimum innermost stride doesn't fit (a K=128 SFA row is only 8 bytes).
__device__ __forceinline__ void cp_async_cg_16_local(void* smem, const void* glob) {
    uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(glob));
}
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

// ---------------------------------------------------------------------------
// mbarrier + TMA PTX wrappers (mirrors tma_block_scale_bench.cu).
// ---------------------------------------------------------------------------
__device__ __forceinline__ void mbarrier_init(uint64_t* bar, uint32_t count) {
    uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(s), "r"(count));
}
__device__ __forceinline__ void mbarrier_invalidate(uint64_t* bar) {
    uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("mbarrier.inval.shared::cta.b64 [%0];\n" ::"r"(s));
}
__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint64_t* bar, uint32_t bytes) {
    uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile(
        "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
        :: "r"(s), "r"(bytes));
}
__device__ __forceinline__ void mbarrier_wait(uint64_t* bar, uint32_t phase) {
    uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "WAIT: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
        "@p bra DONE;\n"
        "bra WAIT;\n"
        "DONE:\n"
        "}\n"
        :: "r"(s), "r"(phase));
}
// 2-D bulk-tensor load. Emits UTMALDG on SM120.
__device__ __forceinline__ void cp_async_bulk_tensor_2d(
    void* smem_dst, const void* desc, int x, int y, uint64_t* mbar) {
    uint32_t s_dst = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
    uint32_t s_bar = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%2, %3}], [%4];\n"
        :: "r"(s_dst), "l"(desc), "r"(x), "r"(y), "r"(s_bar)
        : "memory");
}

// Inline-PTX wrapper for the block-scaled MMA on SM120.
__device__ __forceinline__ void mma_sync_mxf4nvf4_m16n8k64(
    float* d, const uint32_t* a, const uint32_t* b,
    uint32_t sfa, uint32_t sfb) {
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
// Retained as a ground-truth reference for cross-checking the production HW
// MMA kernel. Compiled only when SMALLM_SOFTWARE_REF is defined.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float e2m1_nibble_to_fp32(uint8_t nib) {
    static constexpr float kMag[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    float mag = kMag[nib & 0x7];
    return (nib & 0x8) ? -mag : mag;
}
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
// smallM kernel v2 — TMA loads + 3-stage producer/consumer pipeline.
//
// Grid:  (n_experts, N / TILE_N).  Each CTA owns one expert × one n-tile.
// Block: 256 threads (8 warps).
//
// Pipeline (3 stages):
//   * Stage SMEM holds A[3], B[3], SFA[3], SFB[3] sub-buffers + 3 mbarriers.
//   * Single producer thread (lane 0 of warp 0) issues TMA for A and B at
//     stage s (n-iter coord (k_packed, 0/n_base)) plus per-thread cp.async
//     for SFA/SFB (their gmem stride is K/16 bytes — too small for TMA on
//     low-K test cases; cheap enough that bulk-issue only saves us a warp
//     of cp.async ops per stage).
//   * mbarrier_arrive_expect_tx covers TMA + cp.async transactions per stage.
//   * Consumers (all 8 warps) wait on stage's mbarrier, then run MMAs over
//     SMEM[stage_idx % 3] accumulating into registers. This matches the
//     pre-existing fragment layout — same per-lane addressing inside each
//     stage's SMEM slice.
//
// SMEM budget @ TILE=128, 3 stages:
//   A:  3 × 8 KiB  = 24 KiB
//   B:  3 × 8 KiB  = 24 KiB
//   SFA: 3 × 1 KiB =  3 KiB
//   SFB: 3 × 1 KiB =  3 KiB
//   mbar: 3 × 16 B = ~64 B
//   total ≈ 54 KiB (well under 99 KiB cap on sm_120).
// ---------------------------------------------------------------------------

template <int TILE_M, int TILE_N, int TILE_K, int N_STAGES>
__global__ void smallM_kernel_v1(
    const void* const* __restrict__ d_A,
    const void* const* __restrict__ d_SFA,
    const void* const* __restrict__ d_B,
    const void* const* __restrict__ d_SFB,
    void* const* __restrict__ d_D,
    const float* __restrict__ d_alpha,
    const int* __restrict__ d_M_per_expert,
    const CUtensorMap* __restrict__ d_descs,    // 2 descriptors per expert: [A, B]
    int N, int K) {
    static_assert(TILE_M == 128 && TILE_N == 128,
                  "smallM_kernel_v1: TILE_M=TILE_N=128 fixed (Phase A)");
    static_assert(TILE_K == 128 || TILE_K == 256,
                  "smallM_kernel_v1: TILE_K must be 128 or 256");
    static_assert(N_STAGES >= 2 && N_STAGES <= 4, "N_STAGES in [2,4]");

    const int e      = blockIdx.x;
    const int n_tile = blockIdx.y;
    const int M_e = d_M_per_expert[e];
    if (M_e <= 0) return;

    const uint8_t* SFA_e = static_cast<const uint8_t*>(d_SFA[e]);
    const uint8_t* SFB_e = static_cast<const uint8_t*>(d_SFB[e]);
    half*          D_e   = static_cast<half*>(d_D[e]);
    const float    alpha = d_alpha[e];
    const CUtensorMap* desc_A = d_descs + 2 * e + 0;
    const CUtensorMap* desc_B = d_descs + 2 * e + 1;

    constexpr int A_BYTES_ROW   = TILE_K / 2;     // 64 bytes per A-row in tile
    constexpr int B_BYTES_ROW   = TILE_K / 2;     // 64 bytes per B-row in tile
    constexpr int SFA_BYTES_ROW = TILE_K / 16;    // 8 bytes per A-row in tile
    constexpr int SFB_BYTES_ROW = TILE_K / 16;    // 8 bytes per B-row in tile
    constexpr int A_TILE_BYTES   = TILE_M * A_BYTES_ROW;    // 8 KiB
    constexpr int B_TILE_BYTES   = TILE_N * B_BYTES_ROW;    // 8 KiB
    constexpr int SFA_TILE_BYTES = TILE_M * SFA_BYTES_ROW;  // 1 KiB
    constexpr int SFB_TILE_BYTES = TILE_N * SFB_BYTES_ROW;  // 1 KiB
    constexpr int TMA_BYTES_PER_STAGE = A_TILE_BYTES + B_TILE_BYTES;  // 16 KiB
    constexpr int CPASYNC_BYTES_PER_STAGE = SFA_TILE_BYTES + SFB_TILE_BYTES;  // 2 KiB
    (void)CPASYNC_BYTES_PER_STAGE;

    const int K_groups   = K / 16;
    const int n_base     = n_tile * TILE_N;
    const int M_eff      = min(M_e, TILE_M);

    // SMEM layout (aligned to 128 B for TMA):
    //   A[N_STAGES][TILE_M][A_BYTES_ROW]
    //   B[N_STAGES][TILE_N][B_BYTES_ROW]
    //   SFA[N_STAGES][TILE_M][SFA_BYTES_ROW]
    //   SFB[N_STAGES][TILE_N][SFB_BYTES_ROW]
    //   mbar[N_STAGES]   (8B each, padded to 16B)
    extern __shared__ __align__(128) uint8_t smem_raw[];
    uint8_t* smem_A   = smem_raw;
    uint8_t* smem_B   = smem_A   + N_STAGES * A_TILE_BYTES;
    uint8_t* smem_SFA = smem_B   + N_STAGES * B_TILE_BYTES;
    uint8_t* smem_SFB = smem_SFA + N_STAGES * SFA_TILE_BYTES;
    // 16-byte align mbarriers.
    uintptr_t mbar_base = reinterpret_cast<uintptr_t>(smem_SFB + N_STAGES * SFB_TILE_BYTES);
    mbar_base = (mbar_base + 15) & ~uintptr_t(15);
    uint64_t* smem_mbar = reinterpret_cast<uint64_t*>(mbar_base);

    auto stage_A   = [&](int s) -> uint8_t* { return smem_A   + s * A_TILE_BYTES;   };
    auto stage_B   = [&](int s) -> uint8_t* { return smem_B   + s * B_TILE_BYTES;   };
    auto stage_SFA = [&](int s) -> uint8_t* { return smem_SFA + s * SFA_TILE_BYTES; };
    auto stage_SFB = [&](int s) -> uint8_t* { return smem_SFB + s * SFB_TILE_BYTES; };

    const int tid = threadIdx.x;
    const int n_threads = blockDim.x;
    const int warp_id   = tid / 32;
    const int lane_id   = tid & 31;

    constexpr int M_SUBTILES       = TILE_M / 16;   // 8
    constexpr int N_SUBTILES       = TILE_N / 8;    // 16
    constexpr int K_STRIPES        = TILE_K / 64;   // 2 or 4
    constexpr int WARPS_PER_CTA    = 8;
    // Warp-specialized split: 4 producer warps + 4 consumer warps. Each
    // consumer warp owns N_ITERS_PER_CONSUMER N-iters (4 of 16 = 4 each).
    constexpr int N_PRODUCER_WARPS = 4;
    constexpr int N_CONSUMER_WARPS = 4;
    constexpr int PRODUCER_THREADS = N_PRODUCER_WARPS * 32;  // 128
    constexpr int CONSUMER_THREADS = N_CONSUMER_WARPS * 32;  // 128
    static_assert(N_PRODUCER_WARPS + N_CONSUMER_WARPS == WARPS_PER_CTA, "");
    constexpr int N_ITERS_PER_CONSUMER = N_SUBTILES / N_CONSUMER_WARPS;  // 4
    static_assert(N_ITERS_PER_CONSUMER * N_CONSUMER_WARPS == N_SUBTILES, "");

    // Two mbarrier arrays per stage: bar_full (data ready) + bar_empty (free).
    //   bar_full[s]: arrival_count = PRODUCER_THREADS, expected_tx = TMA_BYTES.
    //     • W0L0 contributes 1 arrival via mbarrier.arrive.expect_tx (also issues TMA).
    //     • The other 127 producer threads each cp.async + wait + plain arrive.
    //   bar_empty[s]: arrival_count = CONSUMER_THREADS.
    //     • Each consumer thread arrives once after MMA on stage S finishes.
    uint64_t* bar_full  = smem_mbar;                 // [N_STAGES]
    uint64_t* bar_empty = smem_mbar + N_STAGES;      // [N_STAGES]

    // ---- Init mbarriers (one elected thread).
    if (tid == 0) {
        #pragma unroll
        for (int s = 0; s < N_STAGES; ++s) {
            mbarrier_init(&bar_full[s],  PRODUCER_THREADS);
            mbarrier_init(&bar_empty[s], CONSUMER_THREADS);
        }
    }
    __syncthreads();

    const bool is_producer = (warp_id < N_PRODUCER_WARPS);

    // Total number of K-tiles.
    const int N_K_TILES = K / TILE_K;

    // ============================================================================
    // PRODUCER PATH (warps 0-3): TMA + cp.async, no MMA.
    // ============================================================================
    if (is_producer) {
        // Producer phase tracking: bar_empty[stage] starts unsignaled. We skip
        // waiting on it for the first N_STAGES iterations (initial fill).
        uint32_t pe_phase[N_STAGES];
        #pragma unroll
        for (int s = 0; s < N_STAGES; ++s) pe_phase[s] = 0u;

        for (int k_idx = 0; k_idx < N_K_TILES; ++k_idx) {
            const int stage = k_idx % N_STAGES;

            // After the initial fill, wait for consumers to free this stage.
            if (k_idx >= N_STAGES) {
                mbarrier_wait(&bar_empty[stage], pe_phase[stage]);
                pe_phase[stage] ^= 1u;
            }

            const int k_offset = k_idx * TILE_K;
            const int k_packed = k_offset / 2;
            const int k_sf     = k_offset / 16;

            uint64_t* bar = &bar_full[stage];

            // W0L0: arrive.expect_tx + issue both TMAs. Single arrival from W0L0.
            // W0L0 does NOT participate in cp.async (so its arrive.expect_tx
            // covers only its own writes — TMA bytes go via complete_tx).
            if (tid == 0) {
                mbarrier_arrive_expect_tx(bar, TMA_BYTES_PER_STAGE);
                cp_async_bulk_tensor_2d(stage_A(stage), desc_A, k_packed, 0,      bar);
                cp_async_bulk_tensor_2d(stage_B(stage), desc_B, k_packed, n_base, bar);
            } else {
                // Other producer threads: cp.async load SF + arrive.
                // cp.async work distribution: tid in [1, PRODUCER_THREADS).
                // SFA tile rows × SFA_BYTES_ROW. Ops per tile:
                //   TILE_K=128 → SFA: 128 rows × 8B = 128 ops total.
                //   TILE_K=256 → SFA: 128 rows × 16B = 256 ops total.
                // (PRODUCER_THREADS - 1) = 127 threads cover these ops.
                {
                    constexpr int N_OPS_PER_ROW = SFA_BYTES_ROW / 8;
                    constexpr int N_OPS         = SFA_TILE_BYTES / 8;
                    for (int op = tid - 1; op < N_OPS; op += (PRODUCER_THREADS - 1)) {
                        int row = op / N_OPS_PER_ROW;
                        int col = (op % N_OPS_PER_ROW) * 8;
                        uint8_t* dst = stage_SFA(stage) + (size_t)row * SFA_BYTES_ROW + col;
                        if (row < M_eff) {
                            const uint8_t* src = SFA_e + (size_t)row * K_groups + k_sf + col;
                            cp_async_ca_8_local(dst, src);
                        } else {
                            *reinterpret_cast<uint64_t*>(dst) = 0ULL;
                        }
                    }
                }
                {
                    constexpr int N_OPS_PER_ROW = SFB_BYTES_ROW / 8;
                    constexpr int N_OPS         = SFB_TILE_BYTES / 8;
                    for (int op = tid - 1; op < N_OPS; op += (PRODUCER_THREADS - 1)) {
                        int row = op / N_OPS_PER_ROW;
                        int col = (op % N_OPS_PER_ROW) * 8;
                        int n_global = n_base + row;
                        uint8_t* dst = stage_SFB(stage) + (size_t)row * SFB_BYTES_ROW + col;
                        if (n_global < N) {
                            const uint8_t* src = SFB_e + (size_t)n_global * K_groups + k_sf + col;
                            cp_async_ca_8_local(dst, src);
                        } else {
                            *reinterpret_cast<uint64_t*>(dst) = 0ULL;
                        }
                    }
                }
                cp_async_commit_local();
                cp_async_wait_group_local<0>();

                // Arrive on bar_full. Release semantics ensure cp.async writes
                // become visible to consumer warps once bar_full flips.
                uint32_t bs = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
                asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(bs));
            }
        }

        // Producer warps exit. Consumer warps continue with epilogue.
        // No mbarrier invalidate from here — we let the consumer-side handle it.
        return;
    }

    // ============================================================================
    // CONSUMER PATH (warps 4-7): wait + MMA + signal bar_empty.
    // ============================================================================

    // ---- Output accumulators (in registers).
    float acc[M_SUBTILES][N_ITERS_PER_CONSUMER][4];
    #pragma unroll
    for (int mi = 0; mi < M_SUBTILES; ++mi) {
        #pragma unroll
        for (int ni_local = 0; ni_local < N_ITERS_PER_CONSUMER; ++ni_local) {
            acc[mi][ni_local][0] = 0.f;
            acc[mi][ni_local][1] = 0.f;
            acc[mi][ni_local][2] = 0.f;
            acc[mi][ni_local][3] = 0.f;
        }
    }

    // Phase tracking per stage for bar_full (toggled each time stage is reused).
    uint32_t pf_phase[N_STAGES];
    #pragma unroll
    for (int s = 0; s < N_STAGES; ++s) pf_phase[s] = 0u;

    // Consumer warp index in [0, N_CONSUMER_WARPS). warps 4-7 → 0-3.
    const int consumer_warp = warp_id - N_PRODUCER_WARPS;

    for (int k_idx = 0; k_idx < N_K_TILES; ++k_idx) {
        const int stage = k_idx % N_STAGES;

        // Wait for producers to fill stage.
        mbarrier_wait(&bar_full[stage], pf_phase[stage]);
        pf_phase[stage] ^= 1u;

        // ---- MMA loop on stage SMEM.
        {
            const int T0 = lane_id & 3;
            const int T1 = lane_id >> 2;
            const int byte_offset = T0 * 4;
            const int m_sfa = T1 + (T0 & 1) * 8;
            const int n_sfb = T1;
            const int ni_base = consumer_warp * N_ITERS_PER_CONSUMER;

            uint8_t* sA   = stage_A(stage);
            uint8_t* sB   = stage_B(stage);
            uint8_t* sSFA = stage_SFA(stage);
            uint8_t* sSFB = stage_SFB(stage);

            #pragma unroll 1
            for (int mi = 0; mi < M_SUBTILES; ++mi) {
                const int m_lo = mi * 16 + T1;
                const int m_hi = m_lo + 8;

                #pragma unroll
                for (int ni_local = 0; ni_local < N_ITERS_PER_CONSUMER; ++ni_local) {
                    const int ni  = ni_base + ni_local;
                    const int n_b = ni * 8 + T1;

                    #pragma unroll
                    for (int ki = 0; ki < K_STRIPES; ++ki) {
                        const int stripe_byte = ki * 32;
                        const int kg_base = ki * 4;

                        uint32_t a0, a1, a2, a3;
                        a0 = *reinterpret_cast<const uint32_t*>(
                            sA + (size_t)m_lo * A_BYTES_ROW + stripe_byte +  0 + byte_offset);
                        a1 = *reinterpret_cast<const uint32_t*>(
                            sA + (size_t)m_hi * A_BYTES_ROW + stripe_byte +  0 + byte_offset);
                        a2 = *reinterpret_cast<const uint32_t*>(
                            sA + (size_t)m_lo * A_BYTES_ROW + stripe_byte + 16 + byte_offset);
                        a3 = *reinterpret_cast<const uint32_t*>(
                            sA + (size_t)m_hi * A_BYTES_ROW + stripe_byte + 16 + byte_offset);

                        uint32_t b0, b1;
                        b0 = *reinterpret_cast<const uint32_t*>(
                            sB + (size_t)n_b * B_BYTES_ROW + stripe_byte +  0 + byte_offset);
                        b1 = *reinterpret_cast<const uint32_t*>(
                            sB + (size_t)n_b * B_BYTES_ROW + stripe_byte + 16 + byte_offset);

                        uint32_t sfa = *reinterpret_cast<const uint32_t*>(
                            sSFA + (size_t)(mi * 16 + m_sfa) * SFA_BYTES_ROW + kg_base);
                        uint32_t sfb = *reinterpret_cast<const uint32_t*>(
                            sSFB + (size_t)(ni * 8  + n_sfb) * SFB_BYTES_ROW + kg_base);

                        uint32_t a_arr[4] = {a0, a1, a2, a3};
                        uint32_t b_arr[2] = {b0, b1};
                        mma_sync_mxf4nvf4_m16n8k64(acc[mi][ni_local], a_arr, b_arr, sfa, sfb);
                    }
                }
            }
        }

        // Signal producers that this stage is now free for refill.
        // Each consumer thread arrives once.
        {
            uint32_t bs = static_cast<uint32_t>(__cvta_generic_to_shared(&bar_empty[stage]));
            asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(bs));
        }
    }

    // ---- Epilogue: each consumer warp casts FP32 → FP16 and writes to global D.
    {
        const int T0 = lane_id & 3;
        const int T1 = lane_id >> 2;
        const int ni_base = consumer_warp * N_ITERS_PER_CONSUMER;

        #pragma unroll 1
        for (int mi = 0; mi < M_SUBTILES; ++mi) {
            #pragma unroll
            for (int ni_local = 0; ni_local < N_ITERS_PER_CONSUMER; ++ni_local) {
                const int ni = ni_base + ni_local;
                const int m0 = mi * 16 + T1;
                const int m1 = m0 + 8;
                const int n0_local = ni * 8 + T0 * 2;
                const int n1_local = n0_local + 1;
                const int n0_g = n_base + n0_local;
                const int n1_g = n_base + n1_local;

                const float a0_out = acc[mi][ni_local][0] * alpha;
                const float a1_out = acc[mi][ni_local][1] * alpha;
                const float a2_out = acc[mi][ni_local][2] * alpha;
                const float a3_out = acc[mi][ni_local][3] * alpha;

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
}

}  // anonymous namespace

#ifdef SMALLM_SOFTWARE_REF
// Debug entry point: dispatch the SOFTWARE reference kernel via the same
// public API. Used by the SmallMKernel.HwMatchesSoftwareReference test.
extern "C" bool gemm_grouped_nvfp4_smallM_software_ref(
    int n_experts, const int* host_M, int N, int K,
    const void* const* host_ptr_A,   const void* const* host_ptr_SFA,
    const void* const* host_ptr_B,   const void* const* host_ptr_SFB,
    void* const* host_ptr_D,         const float* dev_alpha,
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
    void** d_D = nullptr;
    int*   d_M = nullptr;
    cudaMallocAsync(&d_A,   sizeof(void*) * n_experts, stream);
    cudaMallocAsync(&d_SFA, sizeof(void*) * n_experts, stream);
    cudaMallocAsync(&d_B,   sizeof(void*) * n_experts, stream);
    cudaMallocAsync(&d_SFB, sizeof(void*) * n_experts, stream);
    cudaMallocAsync(&d_D,   sizeof(void*) * n_experts, stream);
    cudaMallocAsync(&d_M,   sizeof(int)   * n_experts, stream);
    cudaMemcpyAsync(d_A,   host_ptr_A,   sizeof(void*) * n_experts, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_SFA, host_ptr_SFA, sizeof(void*) * n_experts, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_B,   host_ptr_B,   sizeof(void*) * n_experts, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_SFB, host_ptr_SFB, sizeof(void*) * n_experts, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_D,   host_ptr_D,   sizeof(void*) * n_experts, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_M,   host_M,       sizeof(int)   * n_experts, cudaMemcpyHostToDevice, stream);

    dim3 grid(n_experts, N / TILE_N);
    dim3 block(256);
    smallM_kernel_v1_software_ref<TILE_M, TILE_N, TILE_K><<<grid, block, 0, stream>>>(
        (const void* const*)d_A, (const void* const*)d_SFA,
        (const void* const*)d_B, (const void* const*)d_SFB,
        d_D, dev_alpha, d_M, N, K);

    cudaFreeAsync(d_A, stream);   cudaFreeAsync(d_SFA, stream);
    cudaFreeAsync(d_B, stream);   cudaFreeAsync(d_SFB, stream);
    cudaFreeAsync(d_D, stream);
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

// Legacy CuTe TMA descriptor builders (referenced by the test file via
// indirect template instantiation; retained for source-level continuity with
// T1.6 spec). The production launcher below builds CUtensorMap descriptors
// via the CUDA driver API directly — same wire format, simpler to pass to
// the kernel.
template <int TILE_M, int TILE_K>
auto build_tma_a(const void* d_ptr, int M_e, int K) {
    auto tensor = make_tensor(
        make_gmem_ptr(static_cast<const uint8_t*>(d_ptr)),
        make_layout(make_shape(M_e, K / 2), make_stride(K / 2, _1{})));
    auto smem_layout = make_layout(Shape<Int<TILE_M>, Int<TILE_K / 2>>{});
    return make_tma_copy(SM90_TMA_LOAD{}, tensor, smem_layout);
}
template <int TILE_N, int TILE_K>
auto build_tma_b(const void* d_ptr, int N, int K) {
    auto tensor = make_tensor(
        make_gmem_ptr(static_cast<const uint8_t*>(d_ptr)),
        make_layout(make_shape(N, K / 2), make_stride(K / 2, _1{})));
    auto smem_layout = make_layout(Shape<Int<TILE_N>, Int<TILE_K / 2>>{});
    return make_tma_copy(SM90_TMA_LOAD{}, tensor, smem_layout);
}
template <int TILE_M, int TILE_K>
auto build_tma_sfa(const void* d_ptr, int M_e, int K) {
    auto tensor = make_tensor(
        make_gmem_ptr(static_cast<const uint8_t*>(d_ptr)),
        make_layout(make_shape(M_e, K / 16), make_stride(K / 16, _1{})));
    auto smem_layout = make_layout(Shape<Int<TILE_M>, Int<TILE_K / 16>>{});
    return make_tma_copy(SM90_TMA_LOAD{}, tensor, smem_layout);
}
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

// ---------------------------------------------------------------------------
// CUtensorMap building via runtime entry-point lookup (so we don't add a
// libcuda.so.1 hard dep). Mirrors the trick in tma_block_scale_bench.cu.
// ---------------------------------------------------------------------------
using PFN_cuTensorMapEncodeTiled_t = CUresult (*)(
    CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*,
    const cuuint64_t*, const cuuint32_t*, const cuuint32_t*,
    CUtensorMapInterleave, CUtensorMapSwizzle, CUtensorMapL2promotion,
    CUtensorMapFloatOOBfill);

static PFN_cuTensorMapEncodeTiled_t resolve_tensor_map_encode() {
    static PFN_cuTensorMapEncodeTiled_t pfn = nullptr;
    if (pfn) return pfn;
    cudaDriverEntryPointQueryResult q;
    void* p = nullptr;
    cudaError_t err = cudaGetDriverEntryPoint("cuTensorMapEncodeTiled",
                                               &p, cudaEnableDefault, &q);
    if (err != cudaSuccess || q != cudaDriverEntryPointSuccess || p == nullptr) {
        return nullptr;
    }
    pfn = reinterpret_cast<PFN_cuTensorMapEncodeTiled_t>(p);
    return pfn;
}

// Build a 2-D CUtensorMap over uint8 data with row-major layout.
//   gmem_rows × gmem_cols; row stride = gmem_cols bytes (contiguous).
//   Tile box = box_rows × box_cols.
// Returns true on success.
static bool build_tma_2d_u8(CUtensorMap* desc, void* gmem,
                             int gmem_rows, int gmem_cols,
                             int box_rows,  int box_cols) {
    PFN_cuTensorMapEncodeTiled_t pfn = resolve_tensor_map_encode();
    if (!pfn) return false;
    cuuint64_t shape[2]      = { (cuuint64_t)gmem_cols, (cuuint64_t)gmem_rows };
    cuuint64_t stride[1]     = { (cuuint64_t)gmem_cols };  // row stride bytes
    cuuint32_t box[2]        = { (cuuint32_t)box_cols,    (cuuint32_t)box_rows };
    cuuint32_t box_stride[2] = { 1u, 1u };
    CUresult r = pfn(
        desc, CU_TENSOR_MAP_DATA_TYPE_UINT8, /*rank=*/2,
        gmem, shape, stride, box, box_stride,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    return r == CUDA_SUCCESS;
}

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
    void* const* host_ptr_D,         const float* dev_alpha,
    cudaStream_t stream) {
    if (!gemm_grouped_nvfp4_smallM_available()) return false;
    if (n_experts <= 0 || N <= 0 || K <= 0) return false;
    if ((K % 128) != 0 || (N % 128) != 0) return false;

    // TMA constraint: innermost stride (= row bytes) must be ≥ 16 and
    // multiple of 16. A and B both have row = K/2 bytes — for K ≥ 32, ≥ 16.
    // With K%128==0 we always have K/2 ≥ 64, divisible by 16. OK.
    // (SFA/SFB stay on cp.async path so their tiny row doesn't matter.)

    // Phase A constraint: only support max_M ≤ 128 (single M-tile per expert).
    int max_M = 0;
    for (int e = 0; e < n_experts; ++e) max_M = std::max(max_M, host_M[e]);
    if (max_M > 128) return false;

    constexpr int TILE_M = 128, TILE_N = 128;
    // Pick TILE_K=256 when K is divisible by 256 (more bytes per TMA stage,
    // amortizes pipeline overhead). Fall back to TILE_K=128 for K=128, 384,
    // etc. STAGES selected to keep SMEM under sm_120 99 KiB cap.
    const bool use_tilek_256 = (K % 256) == 0;
    const int TILE_K_rt   = use_tilek_256 ? 256 : 128;

    // Build per-expert CUtensorMap descriptors on host.
    // 2 descriptors per active expert: [A, B]. Inactive experts get a dummy
    // descriptor pointing at a 1×16 dummy buffer — they won't be loaded
    // because the kernel early-exits on M_e ≤ 0.
    // Box geometry:
    //   A: gmem (M_e, K/2) bytes, box (TILE_M, TILE_K/2)
    //   B: gmem (N,   K/2) bytes, box (TILE_N, TILE_K/2)
    std::vector<CUtensorMap> h_descs(2 * n_experts);
    static uint8_t* s_dummy = nullptr;
    static int s_dummy_ready = 0;
    if (!s_dummy_ready) {
        cudaMalloc(&s_dummy, 256);
        cudaMemset(s_dummy, 0, 256);
        s_dummy_ready = 1;
    }
    for (int e = 0; e < n_experts; ++e) {
        const int M_e = host_M[e];
        if (M_e <= 0) {
            // Dummy descriptor — won't be used.
            build_tma_2d_u8(&h_descs[2 * e + 0], s_dummy, 16, 16, TILE_M, TILE_K_rt / 2);
            build_tma_2d_u8(&h_descs[2 * e + 1], s_dummy, 16, 16, TILE_N, TILE_K_rt / 2);
            continue;
        }
        // A: rows = max(M_e, TILE_M) (pad rows out-of-bounds, OOB filled by TMA)
        // We pass actual M_e — TMA's OOB handling fills with zeros for rows past M_e.
        // gmem_cols = K/2, box_cols = TILE_K/2.
        if (!build_tma_2d_u8(&h_descs[2 * e + 0],
                              const_cast<void*>(host_ptr_A[e]),
                              /*gmem_rows=*/M_e, /*gmem_cols=*/K / 2,
                              /*box_rows=*/TILE_M, /*box_cols=*/TILE_K_rt / 2)) {
            std::fprintf(stderr, "[smallM] cuTensorMapEncodeTiled(A) failed (e=%d M=%d K=%d)\n",
                         e, M_e, K);
            return false;
        }
        if (!build_tma_2d_u8(&h_descs[2 * e + 1],
                              const_cast<void*>(host_ptr_B[e]),
                              /*gmem_rows=*/N, /*gmem_cols=*/K / 2,
                              /*box_rows=*/TILE_N, /*box_cols=*/TILE_K_rt / 2)) {
            std::fprintf(stderr, "[smallM] cuTensorMapEncodeTiled(B) failed (e=%d N=%d K=%d)\n",
                         e, N, K);
            return false;
        }
    }

    // Upload pointer arrays + M + descriptors to device.
    void** d_A = nullptr;   void** d_SFA = nullptr;
    void** d_B = nullptr;   void** d_SFB = nullptr;
    void** d_D = nullptr;
    int*   d_M = nullptr;
    CUtensorMap* d_descs = nullptr;
    cudaMallocAsync(&d_A,   sizeof(void*) * n_experts, stream);
    cudaMallocAsync(&d_SFA, sizeof(void*) * n_experts, stream);
    cudaMallocAsync(&d_B,   sizeof(void*) * n_experts, stream);
    cudaMallocAsync(&d_SFB, sizeof(void*) * n_experts, stream);
    cudaMallocAsync(&d_D,   sizeof(void*) * n_experts, stream);
    cudaMallocAsync(&d_M,   sizeof(int)   * n_experts, stream);
    cudaMallocAsync(&d_descs, sizeof(CUtensorMap) * 2 * n_experts, stream);

    cudaMemcpyAsync(d_A,   host_ptr_A,   sizeof(void*) * n_experts, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_SFA, host_ptr_SFA, sizeof(void*) * n_experts, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_B,   host_ptr_B,   sizeof(void*) * n_experts, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_SFB, host_ptr_SFB, sizeof(void*) * n_experts, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_D,   host_ptr_D,   sizeof(void*) * n_experts, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_M,   host_M,       sizeof(int)   * n_experts, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_descs, h_descs.data(),
                    sizeof(CUtensorMap) * 2 * n_experts,
                    cudaMemcpyHostToDevice, stream);

    dim3 grid(n_experts, N / TILE_N);
    dim3 block(256);

    // SMEM budget per (TILE_K, N_STAGES):
    //   TILE_K=128, STAGES=3:  3 × (8 + 8 + 1 + 1) KiB + 128 B = ~54 KiB
    //   TILE_K=256, STAGES=2:  2 × (16 + 16 + 2 + 2) KiB + 128 B = ~72 KiB
    // Both exceed 48 KiB default static cap → opt-in via cudaFuncSetAttribute.
    auto smem_bytes_for = [](int TK, int NS) {
        const int A   = TILE_M * (TK / 2);
        const int B   = TILE_N * (TK / 2);
        const int SFA = TILE_M * (TK / 16);
        const int SFB = TILE_N * (TK / 16);
        return NS * (A + B + SFA + SFB) + 128;
    };

    static int s_smem_attr_set_128 = 0;
    static int s_smem_attr_set_256 = 0;

    if (use_tilek_256) {
        constexpr int TK_ = 256, NS_ = 2;
        const int SMEM_BYTES = smem_bytes_for(TK_, NS_);
        if (!s_smem_attr_set_256) {
            cudaFuncSetAttribute(
                (const void*)smallM_kernel_v1<TILE_M, TILE_N, TK_, NS_>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                SMEM_BYTES);
            s_smem_attr_set_256 = 1;
        }
        smallM_kernel_v1<TILE_M, TILE_N, TK_, NS_><<<grid, block, SMEM_BYTES, stream>>>(
            (const void* const*)d_A, (const void* const*)d_SFA,
            (const void* const*)d_B, (const void* const*)d_SFB,
            d_D, dev_alpha, d_M, d_descs, N, K);
    } else {
        constexpr int TK_ = 128, NS_ = 3;
        const int SMEM_BYTES = smem_bytes_for(TK_, NS_);
        if (!s_smem_attr_set_128) {
            cudaFuncSetAttribute(
                (const void*)smallM_kernel_v1<TILE_M, TILE_N, TK_, NS_>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                SMEM_BYTES);
            s_smem_attr_set_128 = 1;
        }
        smallM_kernel_v1<TILE_M, TILE_N, TK_, NS_><<<grid, block, SMEM_BYTES, stream>>>(
            (const void* const*)d_A, (const void* const*)d_SFA,
            (const void* const*)d_B, (const void* const*)d_SFB,
            d_D, dev_alpha, d_M, d_descs, N, K);
    }
    (void)TILE_K_rt;

    cudaFreeAsync(d_A, stream);   cudaFreeAsync(d_SFA, stream);
    cudaFreeAsync(d_B, stream);   cudaFreeAsync(d_SFB, stream);
    cudaFreeAsync(d_D, stream);
    cudaFreeAsync(d_M, stream);
    cudaFreeAsync(d_descs, stream);
    return true;
}

}  // namespace imp
