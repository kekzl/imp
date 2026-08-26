// nvfp4_gemm_smallm_v2.cu — small-M NVFP4 GEMM v2: y[m, n] = W[n, :] @ x[m, :]
// for M <= 32 activation rows, both sides packed NVFP4 in the PLAIN layout
// (nibbles + linear FP8-UE4M3 micro-scales), computed with the native
// block-scaled mma.sync.kind::mxf4nvf4 — no dequant anywhere, weights and
// scales stream through an asynchronous multi-stage smem pipeline with one
// producer warp (cp.async + mbarrier) and four consumer warps.
//
// Why v2 (docs/plans/2026-08-24-qwen38-port.md; gemm.h postmortem): the
// W4A16 v1 kernel wins isolated (23.9 vs CUTLASS 41.4 us on M=32 N=5120
// K=5120) and loses the real 32-stream step (45.8 us, -11% aggregate): its
// synchronous SIMT loads are exposed to the GDN scan's L2 pressure. What
// the shipping CUTLASS tile has that v1 lacks is an async pipeline; what
// CUTLASS cannot give us is a CTA_M < 128 block-scaled tile (SF atom
// static-asserts at 128 rows). v2 owns an M=32 tile natively:
//   - block tile M32 x N64 x K256 per stage, 4-6 stage ring, ~15 KiB/stage
//   - producer warp: cp.async.cg 16B chunks + cp.async.mbarrier.arrive;
//     consumers never issue a global load
//   - data stays 4-bit until the MMA; smem traffic is 4-bit + SF bytes
//   - Marlin-style striped K-partitioning, fixed per shape, deterministic
//     two-kernel split-K reduce (no atomics)
//
// Fragment/SF mappings are the CUTLASS SM120_16x8x64_TN_VS layouts,
// cross-checked in-tree against src/compute/mxf4nvf4_qkt_validate.cu:
//   A   (T32,V32)->(M16,K64): m = T1 + V1*8, k = T0*8 + V0 + V2*32
//   B   (T32,V16)->(N8, K64): n = T1,        k = T0*8 + V0 + V1*32
//   SFA lane t supplies the 4 K-group scales of row m = (t%2)*8 + t/4
//   SFB lane t supplies the 4 K-group scales of row n = t/4
// with T0 = t%4, T1 = t/4. The nibble order of the plain packed layout is
// exactly the fragment register order, so operands are plain u32 loads.

#include "quant/nvfp4_gemm.h"
#include "quant/nvfp4_gemm_internal.cuh"
#include "core/logging.h"

#include <cuda_fp16.h>

namespace imp {
namespace {

constexpr int kSmM = 32;       // M tile (activation rows, padded with zeros)
constexpr int kNR = 64;        // N rows per CTA
constexpr int kKT = 256;       // K elements per pipeline stage
constexpr int kThreads = 160;  // 4 consumer warps + 1 producer warp

// Row strides in smem. Nibble rows are padded 128 -> 144 bytes so the
// consumers' u32 fragment loads hit 32 distinct banks (bank = (36r + off/4)
// % 32 walks all banks across a warp); 144 = 9*16 keeps cp.async 16B
// alignment. SF rows stay at 16 bytes — the only conflict there is a 2-way
// on the SFA rows, one extra cycle on a 4-byte load.
constexpr int kNibStride = 144;
constexpr int kSfStride = 16;

constexpr int kWNibBytes = kNR * kNibStride;                                  // 9216
constexpr int kXNibBytes = kSmM * kNibStride;                                 // 4608
constexpr int kWSfBytes = kNR * kSfStride;                                    // 1024
constexpr int kXSfBytes = kSmM * kSfStride;                                   // 512
constexpr int kStageBytes = kWNibBytes + kXNibBytes + kWSfBytes + kXSfBytes;  // 15360
constexpr int kDefaultStages = 6;  // smem ring depth; 6 measured best at stripes=1 (sweep 2026-08-25)

// ---- mbarrier / cp.async primitives -----------------------------------------

__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
    const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" ::"r"(a), "r"(count));
}

__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
    const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    uint64_t st;
    asm volatile("mbarrier.arrive.shared.b64 %0, [%1];" : "=l"(st) : "r"(a) : "memory");
}

// Spin until the phase with the given parity completes.
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {
    const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile(
        "{\n\t"
        ".reg .pred P;\n"
        "SMALLM_V2_WAIT_%=:\n\t"
        "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n\t"
        "@!P bra SMALLM_V2_WAIT_%=;\n\t"
        "}" ::"r"(a),
        "r"(parity)
        : "memory");
}

// Async-arrive: one arrive on the mbarrier once all prior cp.async of this
// thread have completed. .noinc consumes one of the pre-initialized expected
// arrivals (the memcpy_async pattern: full-barrier expected count = 32
// producer lanes).
__device__ __forceinline__ void cp_async_mbar_arrive(uint64_t* bar) {
    const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];" ::"r"(a));
}

// 16-byte global->shared async copy, zero-filling when src_bytes == 0
// (activation rows >= M: nibbles AND scales land as 0, so the padded rows
// contribute exactly nothing).
__device__ __forceinline__ void cp_async16(void* dst, const void* src, int src_bytes) {
    const uint32_t d = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(d), "l"(src), "r"(src_bytes));
}

// ---- the MMA ----------------------------------------------------------------

__device__ __forceinline__ void mma_mxf4nvf4(float acc[4], const uint32_t a[4], uint32_t b0, uint32_t b1,
                                             uint32_t sfa, uint32_t sfb) {
#if __CUDA_ARCH__ >= 1200
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0, %1, %2, %3},"
        "{%4, %5, %6, %7},"
        "{%8, %9},"
        "{%0, %1, %2, %3},"
        "{%10},"
        "{%11, %12},"
        "{%13},"
        "{%14, %15};\n"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sfa),
          "h"(static_cast<uint16_t>(0)), "h"(static_cast<uint16_t>(0)), "r"(sfb),
          "h"(static_cast<uint16_t>(0)), "h"(static_cast<uint16_t>(0)));
#endif
}

// ---- kernel -----------------------------------------------------------------

// Shared CTA body. All tensor-dependent values arrive resolved (w/y/ts/N/
// n_base); the single-tensor and pair kernels differ only in how they resolve
// them from blockIdx. Everything from barrier init through the epilogue is
// identical to the shipped single-tensor kernel.
template <int kStages>
__device__ __forceinline__ void smallm_v2_cta_body(
    const uint8_t* __restrict__ w_packed, const uint8_t* __restrict__ w_scales,
    const uint8_t* __restrict__ xq_packed, const uint8_t* __restrict__ xq_scales,
    float* __restrict__ ws_partials, half* __restrict__ y, float ts, int acc_flag, int M, int N_out, int K,
    int stripes, int n_base, int stripe, uint8_t* smem, uint64_t* bar_full, uint64_t* bar_empty) {
    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid & 31;

    if (tid == 0) {
#pragma unroll
        for (int s = 0; s < kStages; ++s) {
            mbar_init(&bar_full[s], 32);    // producer lanes async-arrive
            mbar_init(&bar_empty[s], 128);  // consumer lanes arrive
        }
    }
    __syncthreads();

    // Stripe of K-tiles owned by this CTA.
    const int k_tiles = K / kKT;
    const int per_stripe = (k_tiles + stripes - 1) / stripes;
    const int kt0 = stripe * per_stripe;
    const int kt1 = min(k_tiles, kt0 + per_stripe);
    const int iters = kt1 - kt0;

    auto stage_base = [&](int s) { return smem + s * kStageBytes; };

    if (warp == 4) {
        // ---- producer warp: fill the ring, never compute ----
        // Per-lane chunk assignments are fixed; only the K offset advances.
        // w nibbles: 64 rows x 8 16B-chunks = 512 -> 16/lane
        // x nibbles: 32 rows x 8         = 256 -> 8/lane
        // w scales:  64 rows x 1 16B     = 64  -> 2/lane
        // x scales:  32 rows x 1         = 32  -> 1/lane
        const int64_t w_row_bytes = static_cast<int64_t>(K) / 2;
        const int64_t sf_row_bytes = static_cast<int64_t>(K) / kMicroBlockSize;
        for (int i = 0; i < iters; ++i) {
            const int s = i % kStages;
            const int use = i / kStages;
            if (i >= kStages)
                mbar_wait(&bar_empty[s], (use - 1) & 1);
            uint8_t* base = stage_base(s);
            uint8_t* s_wn = base;
            uint8_t* s_xn = base + kWNibBytes;
            uint8_t* s_wsf = base + kWNibBytes + kXNibBytes;
            uint8_t* s_xsf = s_wsf + kWSfBytes;
            const int kt = kt0 + i;
            const int64_t k_nib_off = static_cast<int64_t>(kt) * (kKT / 2);
            const int64_t k_sf_off = static_cast<int64_t>(kt) * (kKT / kMicroBlockSize);
#pragma unroll
            for (int v = 0; v < 16; ++v) {
                const int c = lane + v * 32;
                const int r = c / 8, j = c % 8;
                cp_async16(s_wn + r * kNibStride + j * 16,
                           w_packed + (n_base + r) * w_row_bytes + k_nib_off + j * 16, 16);
            }
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                const int c = lane + v * 32;
                const int r = c / 8, j = c % 8;
                cp_async16(s_xn + r * kNibStride + j * 16, xq_packed + r * w_row_bytes + k_nib_off + j * 16,
                           r < M ? 16 : 0);
            }
#pragma unroll
            for (int v = 0; v < 2; ++v) {
                const int r = lane + v * 32;
                cp_async16(s_wsf + r * kSfStride, w_scales + (n_base + r) * sf_row_bytes + k_sf_off, 16);
            }
            cp_async16(s_xsf + lane * kSfStride, xq_scales + lane * sf_row_bytes + k_sf_off,
                       lane < M ? 16 : 0);
            cp_async_mbar_arrive(&bar_full[s]);
        }
    } else {
        // ---- consumer warps: wait, MMA, release ----
        // Warp tile: M16 x N32. warp_m in {0,1}, warp_n in {0,1}.
        const int warp_m = warp & 1;
        const int warp_n = warp >> 1;
        const int T0 = lane & 3;
        const int T1 = lane >> 2;
        const int a_row = warp_m * 16 + T1;  // A fragment row (V1=0)
        const int sfa_row = warp_m * 16 + (lane & 1) * 8 + (lane >> 2);
        float acc[4][4];
#pragma unroll
        for (int nf = 0; nf < 4; ++nf)
#pragma unroll
            for (int r = 0; r < 4; ++r)
                acc[nf][r] = 0.0f;

        for (int i = 0; i < iters; ++i) {
            const int s = i % kStages;
            const int use = i / kStages;
            mbar_wait(&bar_full[s], use & 1);
            const uint8_t* base = stage_base(s);
            const uint8_t* s_wn = base;
            const uint8_t* s_xn = base + kWNibBytes;
            const uint8_t* s_wsf = base + kWNibBytes + kXNibBytes;
            const uint8_t* s_xsf = s_wsf + kWSfBytes;
#pragma unroll
            for (int c = 0; c < kKT / 64; ++c) {
                uint32_t a[4];
                const uint8_t* xr = s_xn + a_row * kNibStride + T0 * 4 + c * 32;
                a[0] = *reinterpret_cast<const uint32_t*>(xr);
                a[1] = *reinterpret_cast<const uint32_t*>(xr + 8 * kNibStride);
                a[2] = *reinterpret_cast<const uint32_t*>(xr + 16);
                a[3] = *reinterpret_cast<const uint32_t*>(xr + 8 * kNibStride + 16);
                const uint32_t sfa = *reinterpret_cast<const uint32_t*>(s_xsf + sfa_row * kSfStride + c * 4);
#pragma unroll
                for (int nf = 0; nf < 4; ++nf) {
                    const int n_row = warp_n * 32 + nf * 8 + T1;
                    const uint8_t* wr = s_wn + n_row * kNibStride + T0 * 4 + c * 32;
                    const uint32_t b0 = *reinterpret_cast<const uint32_t*>(wr);
                    const uint32_t b1 = *reinterpret_cast<const uint32_t*>(wr + 16);
                    const uint32_t sfb = *reinterpret_cast<const uint32_t*>(s_wsf + n_row * kSfStride +
                                                                            c * 4);
                    mma_mxf4nvf4(acc[nf], a, b0, b1, sfa, sfb);
                }
            }
            mbar_arrive(&bar_empty[s]);
        }

        // Stage the FP32 accumulators for coalesced plane stores. Ring smem
        // is dead after the loop; the __syncthreads below fences the reuse.
        __syncthreads();
        float* s_out = reinterpret_cast<float*>(smem);  // [kSmM][kNR]
#pragma unroll
        for (int nf = 0; nf < 4; ++nf) {
            const int n0 = warp_n * 32 + nf * 8 + T0 * 2;
            s_out[(warp_m * 16 + T1) * kNR + n0] = acc[nf][0];
            s_out[(warp_m * 16 + T1) * kNR + n0 + 1] = acc[nf][1];
            s_out[(warp_m * 16 + T1 + 8) * kNR + n0] = acc[nf][2];
            s_out[(warp_m * 16 + T1 + 8) * kNR + n0 + 1] = acc[nf][3];
        }
    }
    if (warp == 4)
        __syncthreads();  // producer joins the consumers' staging barrier
    __syncthreads();

    const float* s_out = reinterpret_cast<const float*>(smem);
    if (stripes == 1) {
        // Single stripe owns the full K range: write FP16 y directly, tensor
        // scale applied — no reduce launch, no partial-plane round-trip.
        for (int i = tid; i < kSmM * kNR; i += kThreads) {
            const int m = i / kNR;
            if (m >= M)
                continue;
            const int n = i % kNR;
            const int64_t o = static_cast<int64_t>(m) * N_out + n_base + n;
            y[o] = __float2half(ts * s_out[i] + (acc_flag ? __half2float(y[o]) : 0.0f));
        }
        return;
    }
    // Streaming stores: each partial is written once and read once by the
    // reduce kernel (v1 lesson — un-hinted partials knocked L2 sets out from
    // under the weight stream).
    float* plane = ws_partials + static_cast<size_t>(stripe) * kSmM * N_out;
    for (int i = tid; i < kSmM * kNR; i += kThreads) {
        const int m = i / kNR;
        const int n = i % kNR;
        __stcs(&plane[static_cast<int64_t>(m) * N_out + n_base + n], s_out[i]);
    }
}

// grid = (N/kNR, stripes). Each CTA walks a contiguous stripe of K-tiles for
// its n-tile and writes one FP32 partial plane per stripe (stripe-exclusive
// -> deterministic reduce). Dynamic smem: kStages * kStageBytes.
template <int kStages>
__global__ void gemm_nvfp4_smallm_v2_kernel(const uint8_t* __restrict__ w_packed,
                                            const uint8_t* __restrict__ w_scales,
                                            const uint8_t* __restrict__ xq_packed,
                                            const uint8_t* __restrict__ xq_scales,
                                            float* __restrict__ ws_partials, half* __restrict__ y, float ts,
                                            int acc_flag, int M, int N_out, int K, int stripes) {
    extern __shared__ uint8_t smem[];
    __shared__ uint64_t bar_full[kStages];
    __shared__ uint64_t bar_empty[kStages];
    smallm_v2_cta_body<kStages>(w_packed, w_scales, xq_packed, xq_scales, ws_partials, y, ts, acc_flag, M,
                                N_out, K, stripes, blockIdx.x * kNR, blockIdx.y, smem, bar_full, bar_empty);
}

// Pair variant: two weight tensors sharing ONE quantized activation, one
// launch. grid.x covers the n-tiles of W1 then W2; each CTA resolves which
// tensor it owns and runs the shared body unchanged. stripes == 1 only (every
// call-site N is >= 5120, where the stripe policy is 1), so there is no
// workspace and no reduce. Saves one launch's fixed cost + one tail wave per
// sibling pair (FFN gate|up, GDN in|z) per layer per batched-decode step.
template <int kStages>
__global__ void gemm_nvfp4_smallm_v2_pair_kernel(
    const uint8_t* __restrict__ w1, const uint8_t* __restrict__ s1, half* __restrict__ y1, float ts1,
    int n_tiles1, int N1, const uint8_t* __restrict__ w2, const uint8_t* __restrict__ s2,
    half* __restrict__ y2, float ts2, int N2, const uint8_t* __restrict__ xq_packed,
    const uint8_t* __restrict__ xq_scales, int M, int K) {
    extern __shared__ uint8_t smem[];
    __shared__ uint64_t bar_full[kStages];
    __shared__ uint64_t bar_empty[kStages];
    const int nt = blockIdx.x;
    const bool second = nt >= n_tiles1;
    smallm_v2_cta_body<kStages>(second ? w2 : w1, second ? s2 : s1, xq_packed, xq_scales,
                                /*ws_partials=*/nullptr, second ? y2 : y1, second ? ts2 : ts1,
                                /*acc_flag=*/0, M, second ? N2 : N1, K, /*stripes=*/1,
                                (second ? nt - n_tiles1 : nt) * kNR, /*stripe=*/0, smem, bar_full,
                                bar_empty);
}

// Reduce the stripe partial planes into FP16 y, applying the combined tensor
// scale. kAcc adds onto the existing y (o/down residual call sites, beta=1).
template <bool kAcc>
__global__ void smallm_v2_reduce_kernel(const float* __restrict__ ws_partials, half* __restrict__ y, int M,
                                        int N_out, int stripes, float ts) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M * N_out)
        return;
    const int m = i / N_out, n = i % N_out;
    float acc = 0.0f;
    for (int sp = 0; sp < stripes; ++sp)
        acc += __ldcs(
            &ws_partials[static_cast<size_t>(sp) * kSmM * N_out + static_cast<int64_t>(m) * N_out + n]);
    const int64_t o = static_cast<int64_t>(m) * N_out + n;
    y[o] = __float2half(ts * acc + (kAcc ? __half2float(y[o]) : 0.0f));
}

}  // namespace

// K-stripe count per shape: enough CTAs to cover the SMs about twice, capped
// by the K-tile count. Fixed per (N, K) — no per-launch scheduler state, so
// launches are graph-safe and the reduce is deterministic.
int gemm_nvfp4_smallm_v2_stripes(int N_out, int K) {
    const int n_tiles = N_out / kNR;
    const int k_tiles = K / kKT;
    // Measured (M=32 N=5120 K=5120 sweep, 2026-08-25): one stripe beats every
    // split-K config — the deltas track the extra reduce traffic, not the CTA
    // count. Split K only when the n-grid alone leaves most of the card idle.
    if (n_tiles >= 80)
        return 1;
    int s = (160 + n_tiles - 1) / n_tiles;
    if (s < 1)
        s = 1;
    if (s > k_tiles)
        s = k_tiles;
    return s;
}

size_t gemm_nvfp4_smallm_v2_workspace_bytes(int N_out, int K) {
    return static_cast<size_t>(gemm_nvfp4_smallm_v2_stripes(N_out, K)) * kSmM * N_out * sizeof(float);
}

// y[m, n] = W[n, :] @ x[m, :], both sides plain NVFP4. M <= 32; K % 256 == 0;
// N % 64 == 0. d_workspace: gemm_nvfp4_smallm_v2_workspace_bytes(N_out, K).
// Xq rows >= M are never read (zero-filled in the pipeline), so Xq buffers
// only need M rows.
namespace {

template <int kStages>
bool launch_smallm_v2(const NvFP4QuantResult& W, const NvFP4QuantResult& Xq, half* y, int M, int N_out, int K,
                      void* d_workspace, cudaStream_t stream, bool accumulate, int stripes) {
    static const bool smem_ok = [] {
        return cudaFuncSetAttribute(gemm_nvfp4_smallm_v2_kernel<kStages>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    kStages * kStageBytes) == cudaSuccess;
    }();
    if (!smem_ok)
        return false;
    const dim3 grid(N_out / kNR, stripes);
    const float ts = W.tensor_scale * Xq.tensor_scale;
    gemm_nvfp4_smallm_v2_kernel<kStages><<<grid, kThreads, kStages * kStageBytes, stream>>>(
        reinterpret_cast<const uint8_t*>(W.packed_data), reinterpret_cast<const uint8_t*>(W.micro_scales),
        reinterpret_cast<const uint8_t*>(Xq.packed_data), reinterpret_cast<const uint8_t*>(Xq.micro_scales),
        static_cast<float*>(d_workspace), y, ts, accumulate ? 1 : 0, M, N_out, K, stripes);
    IMP_CUDA_CHECK_LAUNCH();
    if (stripes == 1)
        return true;
    const int total = M * N_out;
    if (accumulate) {
        smallm_v2_reduce_kernel<true>
            <<<(total + 255) / 256, 256, 0, stream>>>(static_cast<const float*>(d_workspace), y, M, N_out,
                                                      stripes, ts);
        IMP_CUDA_CHECK_LAUNCH();
    } else {
        smallm_v2_reduce_kernel<false>
            <<<(total + 255) / 256, 256, 0, stream>>>(static_cast<const float*>(d_workspace), y, M, N_out,
                                                      stripes, ts);
        IMP_CUDA_CHECK_LAUNCH();
    }
    return true;
}

bool smallm_v2_args_ok(const NvFP4QuantResult& W, const NvFP4QuantResult& Xq, int M, int N_out, int K,
                       void* d_workspace, int stripes) {
    if (M <= 0 || M > kSmM || (K % kKT) != 0 || (N_out % kNR) != 0)
        return false;
    if (stripes > 1 && d_workspace == nullptr)
        return false;
    return W.packed_data != nullptr && W.micro_scales != nullptr && Xq.packed_data != nullptr &&
           Xq.micro_scales != nullptr;
}

}  // namespace

bool gemm_nvfp4_smallm_v2_a4(const NvFP4QuantResult& W, const NvFP4QuantResult& Xq, half* y, int M, int N_out,
                             int K, void* d_workspace, cudaStream_t stream, bool accumulate) {
    const int stripes = gemm_nvfp4_smallm_v2_stripes(N_out, K);
    if (!smallm_v2_args_ok(W, Xq, M, N_out, K, d_workspace, stripes))
        return false;
    return launch_smallm_v2<kDefaultStages>(W, Xq, y, M, N_out, K, d_workspace, stream, accumulate, stripes);
}

// Two sibling tensors (same K, same quantized activation), one launch. Only
// the stripes==1 regime (both Ns >= 5120) — a caller with a striped shape gets
// `false` and falls back to two single launches. No workspace, no accumulate:
// every pair call site writes fresh outputs (beta = 0).
bool gemm_nvfp4_smallm_v2_pair_a4(const NvFP4QuantResult& W1, const NvFP4QuantResult& W2,
                                  const NvFP4QuantResult& Xq, half* y1, half* y2, int M, int N1, int N2,
                                  int K, cudaStream_t stream) {
    if (gemm_nvfp4_smallm_v2_stripes(N1, K) != 1 || gemm_nvfp4_smallm_v2_stripes(N2, K) != 1)
        return false;
    if (!smallm_v2_args_ok(W1, Xq, M, N1, K, /*d_workspace=*/nullptr, /*stripes=*/1) ||
        !smallm_v2_args_ok(W2, Xq, M, N2, K, /*d_workspace=*/nullptr, /*stripes=*/1))
        return false;
    static const bool smem_ok = [] {
        return cudaFuncSetAttribute(gemm_nvfp4_smallm_v2_pair_kernel<kDefaultStages>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    kDefaultStages * kStageBytes) == cudaSuccess;
    }();
    if (!smem_ok)
        return false;
    const int n_tiles1 = N1 / kNR;
    const dim3 grid(n_tiles1 + N2 / kNR);
    gemm_nvfp4_smallm_v2_pair_kernel<kDefaultStages><<<grid, kThreads, kDefaultStages * kStageBytes, stream>>>(
        reinterpret_cast<const uint8_t*>(W1.packed_data), reinterpret_cast<const uint8_t*>(W1.micro_scales),
        y1, W1.tensor_scale * Xq.tensor_scale, n_tiles1, N1,
        reinterpret_cast<const uint8_t*>(W2.packed_data), reinterpret_cast<const uint8_t*>(W2.micro_scales),
        y2, W2.tensor_scale * Xq.tensor_scale, N2,
        reinterpret_cast<const uint8_t*>(Xq.packed_data), reinterpret_cast<const uint8_t*>(Xq.micro_scales),
        M, K);
    IMP_CUDA_CHECK_LAUNCH();
    return true;
}

// Tuning hook for the isolated sweep (tests only): explicit stage depth and
// stripe count. Workspace must hold `stripes` planes.
bool gemm_nvfp4_smallm_v2_a4_tuned(const NvFP4QuantResult& W, const NvFP4QuantResult& Xq, half* y, int M,
                                   int N_out, int K, void* d_workspace, cudaStream_t stream, bool accumulate,
                                   int stages, int stripes) {
    if (!smallm_v2_args_ok(W, Xq, M, N_out, K, d_workspace, stripes))
        return false;
    if (stripes < 1 || stripes > K / kKT)
        return false;
    switch (stages) {
        case 2:
            return launch_smallm_v2<2>(W, Xq, y, M, N_out, K, d_workspace, stream, accumulate, stripes);
        case 3:
            return launch_smallm_v2<3>(W, Xq, y, M, N_out, K, d_workspace, stream, accumulate, stripes);
        case 4:
            return launch_smallm_v2<4>(W, Xq, y, M, N_out, K, d_workspace, stream, accumulate, stripes);
        case 6:
            return launch_smallm_v2<6>(W, Xq, y, M, N_out, K, d_workspace, stream, accumulate, stripes);
        default:
            return false;
    }
}

}  // namespace imp
