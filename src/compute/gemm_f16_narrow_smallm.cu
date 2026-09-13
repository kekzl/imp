// Narrow FP16 GEMM, up to two (W,C) pairs per launch. Grid (n16 tiles over both pairs) x
// kSplit. A CTA owns one 16-column tile and one K range; its 8 warps take the k16 steps
// round-robin, each accumulating the whole 32x16 tile (2 m16 x 2 n8 mma.sync m16n8k16, FP32).
// Reduce: warps -> smem -> one FP32 partial per CTA in ws[ks][m][col]; the last CTA of a tile
// (atomic ticket) sums the kSplit partials in index order and writes C. Fixed order = bitwise
// deterministic. Operand fragments come straight from global memory (u32/lane, L2 hits).

#include "compute/gemm_f16_narrow_smallm.h"
#include "core/logging.h"
#include "core/pdl.h"
#include "core/pdl_device.cuh"

#include <cstdint>

namespace imp {
namespace {

constexpr int kMaxM = 32;
constexpr int kNT = 16;
constexpr int kWarps = 8;
constexpr int kThreads = kWarps * 32;
constexpr int kSplit = 8;
constexpr int kKAlign = kSplit * 16;  // K % 128 == 0

struct NarrowArgs {
    const half* A;
    int M;
    int K;
    const half* W[2];
    half* C[2];
    int N[2];
    int tiles0;  // tiles [0, tiles0) belong to pair 0
    float* ws;   // [kSplit][kMaxM][n_total]
    int* tickets;
    int n_total;
};

__device__ __forceinline__ uint32_t ld_u32(const half* p) {
    return __ldg(reinterpret_cast<const unsigned int*>(p));
}

__device__ __forceinline__ void mma_f16_16x8x16(float* c, const uint32_t* a, const uint32_t* b) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__global__ void __launch_bounds__(kThreads) gemm_f16_narrow_smallm_kernel(NarrowArgs args) {
    __shared__ float s_red[kWarps][kMaxM][kNT];
    __shared__ int s_last;

    pdl_wait();

    const int tile = blockIdx.x;
    const int ks = blockIdx.y;
    const int pair = (tile >= args.tiles0) ? 1 : 0;
    const int tile_in_pair = tile - (pair ? args.tiles0 : 0);
    const int K = args.K;
    const int M = args.M;
    const half* __restrict__ W = args.W[pair] + static_cast<size_t>(tile_in_pair) * kNT * K;

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int g = lane >> 2;
    const int tg = lane & 3;

    // A fragment rows g, g+8 (m-tile 0) and g+16, g+24 (m-tile 1); rows past M
    // read zero (pointer clamped, value masked).
    const half* Ar[4];
    bool vr[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int row = g + 8 * i;
        vr[i] = row < M;
        Ar[i] = args.A + static_cast<size_t>(vr[i] ? row : 0) * K + 2 * tg;
    }
    const half* Wg0 = W + static_cast<size_t>(g) * K + 2 * tg;
    const half* Wg1 = Wg0 + static_cast<size_t>(8) * K;
    const bool m1 = M > 16;

    const int k_cta = K / kSplit;
    const int k0 = ks * k_cta;
    const int nsteps = k_cta >> 4;

    float acc[2][2][4] = {};
    for (int s = warp; s < nsteps; s += kWarps) {
        const int k = k0 + s * 16;
        uint32_t a[2][4];
        a[0][0] = vr[0] ? ld_u32(Ar[0] + k) : 0u;
        a[0][1] = vr[1] ? ld_u32(Ar[1] + k) : 0u;
        a[0][2] = vr[0] ? ld_u32(Ar[0] + k + 8) : 0u;
        a[0][3] = vr[1] ? ld_u32(Ar[1] + k + 8) : 0u;
        if (m1) {
            a[1][0] = vr[2] ? ld_u32(Ar[2] + k) : 0u;
            a[1][1] = vr[3] ? ld_u32(Ar[3] + k) : 0u;
            a[1][2] = vr[2] ? ld_u32(Ar[2] + k + 8) : 0u;
            a[1][3] = vr[3] ? ld_u32(Ar[3] + k + 8) : 0u;
        }
        uint32_t b[2][2];
        b[0][0] = ld_u32(Wg0 + k);
        b[0][1] = ld_u32(Wg0 + k + 8);
        b[1][0] = ld_u32(Wg1 + k);
        b[1][1] = ld_u32(Wg1 + k + 8);
        mma_f16_16x8x16(acc[0][0], a[0], b[0]);
        mma_f16_16x8x16(acc[0][1], a[0], b[1]);
        if (m1) {
            mma_f16_16x8x16(acc[1][0], a[1], b[0]);
            mma_f16_16x8x16(acc[1][1], a[1], b[1]);
        }
    }
    pdl_trigger();

    // C fragment: c0,c1 at (row g, cols 2tg, 2tg+1), c2,c3 at (row g+8, same cols).
#pragma unroll
    for (int i = 0; i < 2; ++i) {
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int col = 8 * j + 2 * tg;
            s_red[warp][g + 16 * i][col] = acc[i][j][0];
            s_red[warp][g + 16 * i][col + 1] = acc[i][j][1];
            s_red[warp][g + 16 * i + 8][col] = acc[i][j][2];
            s_red[warp][g + 16 * i + 8][col + 1] = acc[i][j][3];
        }
    }
    __syncthreads();

    const int col0 = tile * kNT;
    float* ws = args.ws;
    for (int o = threadIdx.x; o < kMaxM * kNT; o += kThreads) {
        const int m = o / kNT;
        const int c = o - m * kNT;
        if (m >= M)
            continue;
        float sum = 0.0f;
#pragma unroll
        for (int w = 0; w < kWarps; ++w)
            sum += s_red[w][m][c];
        ws[(static_cast<size_t>(ks) * kMaxM + m) * args.n_total + col0 + c] = sum;
    }
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0)
        s_last = (atomicAdd(&args.tickets[tile], 1) == kSplit - 1) ? 1 : 0;
    __syncthreads();
    if (!s_last)
        return;
    __threadfence();

    half* __restrict__ C = args.C[pair];
    const int N = args.N[pair];
    for (int o = threadIdx.x; o < kMaxM * kNT; o += kThreads) {
        const int m = o / kNT;
        const int c = o - m * kNT;
        if (m >= M)
            continue;
        float sum = 0.0f;
#pragma unroll
        for (int k2 = 0; k2 < kSplit; ++k2)
            sum += __ldcg(&ws[(static_cast<size_t>(k2) * kMaxM + m) * args.n_total + col0 + c]);
        C[static_cast<size_t>(m) * N + tile_in_pair * kNT + c] = __float2half_rn(sum);
    }
    if (threadIdx.x == 0)
        args.tickets[tile] = 0;
}

size_t partials_bytes(int n_total) { return static_cast<size_t>(kSplit) * kMaxM * n_total * sizeof(float); }

}  // namespace

size_t gemm_f16_narrow_smallm_workspace_bytes(int n_total_max) {
    const size_t p = (partials_bytes(n_total_max) + 255) & ~size_t(255);
    return p + static_cast<size_t>(n_total_max / kNT + 1) * sizeof(int);
}

bool gemm_f16_narrow_smallm(const half* A, int M, int K, const half* W0, half* C0, int N0, const half* W1,
                            half* C1, int N1, void* ws, size_t ws_bytes, cudaStream_t stream) {
    if (!A || !W0 || !C0 || !ws || M < 1 || M > kMaxM || K < kKAlign || (K % kKAlign) != 0)
        return false;
    if (N0 <= 0 || (N0 % kNT) != 0 || N1 < 0 || (N1 % kNT) != 0 || (N1 > 0 && (!W1 || !C1)))
        return false;
    const int n_total = N0 + N1;
    if (ws_bytes < gemm_f16_narrow_smallm_workspace_bytes(n_total))
        return false;
    if ((reinterpret_cast<uintptr_t>(A) & 3) || (reinterpret_cast<uintptr_t>(W0) & 3) ||
        (W1 && (reinterpret_cast<uintptr_t>(W1) & 3)))
        return false;

    NarrowArgs args{};
    args.A = A;
    args.M = M;
    args.K = K;
    args.W[0] = W0;
    args.W[1] = W1;
    args.C[0] = C0;
    args.C[1] = C1;
    args.N[0] = N0;
    args.N[1] = N1;
    args.tiles0 = N0 / kNT;
    args.ws = static_cast<float*>(ws);
    args.tickets = reinterpret_cast<int*>(static_cast<char*>(ws) +
                                          ((partials_bytes(n_total) + 255) & ~size_t(255)));
    args.n_total = n_total;

    const dim3 grid(n_total / kNT, kSplit);
    pdl::enable_kernel(gemm_f16_narrow_smallm_kernel);
    pdl::launch(gemm_f16_narrow_smallm_kernel, grid, dim3(kThreads), size_t(0), stream, args);
    IMP_CUDA_CHECK_LAUNCH();
    return true;
}

}  // namespace imp
