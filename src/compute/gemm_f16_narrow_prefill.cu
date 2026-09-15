// Narrow FP16 GEMM for prefill rows, up to two (W,C) pairs per launch. Grid (64-row tiles) x
// split. Block tile 64 x NPAD x 32 (NPAD = N0+N1 rounded up to 16, <= 128), 4 warps each own
// 16 rows and every 16-column fragment (WMMA m16n16k16, FP32 accumulate); A and the
// concatenated [W0; W1] rows stream through a 3-4 stage cp.async pipeline in shared memory.
// Split > 1: one FP32 partial per split in ws[ks][m][col] and a second launch sums them in
// index order (a one-CTA-per-tile reduce behind a ticket ran at L2 latency: 12-24 us at
// M=512). Fixed order = bitwise deterministic.

#include "compute/gemm_f16_narrow_prefill.h"
#include "core/logging.h"
#include "core/pdl.h"
#include "core/pdl_device.cuh"

#include <cstdint>
#include <mma.h>

namespace imp {
namespace {

using namespace nvcuda;

constexpr int kBM = 64;
constexpr int kBK = 32;
constexpr int kWarps = 4;
constexpr int kThreads = kWarps * 32;
constexpr int kMaxN = 128;
constexpr int kMaxSplit = 32;
constexpr int kTargetCtas = 256;
constexpr int kMinKPerCta = 128;  // >= 4 BK iterations per CTA
constexpr int kChunkHalves = 8;   // one 16 B cp.async
constexpr int kChunksPerRow = kBK / kChunkHalves;
constexpr int kReduceThreads = 256;

struct PrefillArgs {
    const half* A;
    int M;
    int K;
    const half* W[2];
    half* C[2];
    int N[2];
    int n_total;
    int split;
    int k_cta;
    float* ws;  // [split][M][n_total]
};

__device__ __forceinline__ void cp_async_cg16_zero(void* smem_ptr, const void* gmem_ptr, bool valid) {
    unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));
    const int src_size = valid ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem_ptr), "r"(src_size));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n"); }
template <int N>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

// Pipeline depth under the 48 KB static smem limit: 4 stages up to NPAD=96, 3 at 128.
template <int NPAD>
constexpr int stages_for() {
    return NPAD >= 128 ? 3 : 4;
}

// NPAD rows of W and 64 rows of A per stage; the float epilogue tile aliases the stages.
template <int NPAD>
struct Smem {
    static constexpr int kStages = stages_for<NPAD>();
    static constexpr int kA = kBM * kBK;   // halves per stage
    static constexpr int kW = NPAD * kBK;  // halves per stage
    static constexpr int kStageHalves = kA + kW;
    static constexpr int kPipelineBytes = kStages * kStageHalves * static_cast<int>(sizeof(half));
    static constexpr int kEpilogueBytes = kBM * NPAD * static_cast<int>(sizeof(float));
    static constexpr int kBytes = kPipelineBytes > kEpilogueBytes ? kPipelineBytes : kEpilogueBytes;
};

// Stage (A rows of this row tile, W rows 0..n_total) for k-tile kt into stage buffer `s`.
template <int NPAD>
__device__ __forceinline__ void load_stage(half* stage, const PrefillArgs& args, int row0, int kt) {
    half* As = stage;
    half* Ws = stage + Smem<NPAD>::kA;
    const int k0 = kt * kBK;
    for (int c = threadIdx.x; c < kBM * kChunksPerRow; c += kThreads) {
        const int r = c / kChunksPerRow;
        const int kc = (c - r * kChunksPerRow) * kChunkHalves;
        const int gr = row0 + r;
        const bool valid = gr < args.M;
        const half* src = args.A + static_cast<size_t>(valid ? gr : 0) * args.K + k0 + kc;
        cp_async_cg16_zero(As + r * kBK + kc, src, valid);
    }
    for (int c = threadIdx.x; c < NPAD * kChunksPerRow; c += kThreads) {
        const int r = c / kChunksPerRow;
        const int kc = (c - r * kChunksPerRow) * kChunkHalves;
        const bool valid = r < args.n_total;
        const bool p1 = r >= args.N[0];
        const half* base = p1 ? args.W[1] : args.W[0];
        const int wr = valid ? (p1 ? r - args.N[0] : r) : 0;
        const half* src = (valid ? base : args.W[0]) + static_cast<size_t>(wr) * args.K + k0 + kc;
        cp_async_cg16_zero(Ws + r * kBK + kc, src, valid);
    }
}

template <int NPAD>
__global__ void __launch_bounds__(kThreads) gemm_f16_narrow_prefill_kernel(PrefillArgs args) {
    constexpr int NF = NPAD / 16;
    constexpr int kStages = Smem<NPAD>::kStages;
    __shared__ __align__(16) unsigned char smem_raw[Smem<NPAD>::kBytes];
    half* stages = reinterpret_cast<half*>(smem_raw);

    pdl_wait();

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int M = args.M;
    const int N0 = args.N[0];
    const int nt = args.n_total;
    const int ks = blockIdx.y;
    const int row0 = blockIdx.x * kBM;
    const int kt0 = (ks * args.k_cta) / kBK;
    const int n_tiles = args.k_cta / kBK;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[NF];
#pragma unroll
    for (int j = 0; j < NF; ++j)
        wmma::fill_fragment(acc[j], 0.0f);

#pragma unroll
    for (int s = 0; s < kStages - 1; ++s) {
        if (s < n_tiles)
            load_stage<NPAD>(stages + s * Smem<NPAD>::kStageHalves, args, row0, kt0 + s);
        cp_async_commit();
    }

    for (int kt = 0; kt < n_tiles; ++kt) {
        cp_async_wait_group<kStages - 2>();
        __syncthreads();
        const int nxt = kt + kStages - 1;
        if (nxt < n_tiles)
            load_stage<NPAD>(stages + (nxt % kStages) * Smem<NPAD>::kStageHalves, args, row0, kt0 + nxt);
        cp_async_commit();

        const half* As = stages + (kt % kStages) * Smem<NPAD>::kStageHalves;
        const half* Ws = As + Smem<NPAD>::kA;
#pragma unroll
        for (int kk = 0; kk < kBK / 16; ++kk) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
            wmma::load_matrix_sync(a_frag, As + warp * 16 * kBK + kk * 16, kBK);
#pragma unroll
            for (int j = 0; j < NF; ++j) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;
                wmma::load_matrix_sync(b_frag, Ws + j * 16 * kBK + kk * 16, kBK);
                wmma::mma_sync(acc[j], a_frag, b_frag, acc[j]);
            }
        }
    }
    cp_async_wait_group<0>();
    pdl_trigger();
    __syncthreads();

    // Epilogue through smem: [64][NPAD] floats, warp w owns rows 16w..16w+15.
    float* tile = reinterpret_cast<float*>(smem_raw);
#pragma unroll
    for (int j = 0; j < NF; ++j)
        wmma::store_matrix_sync(tile + warp * 16 * NPAD + j * 16, acc[j], NPAD, wmma::mem_row_major);
    __syncwarp();

    if (args.split == 1) {
        for (int o = lane; o < 16 * nt; o += 32) {
            const int r = o / nt;
            const int c = o - r * nt;
            const int m = row0 + warp * 16 + r;
            if (m >= M)
                continue;
            const bool p1 = c >= N0;
            half* C = args.C[p1 ? 1 : 0];
            const int N = args.N[p1 ? 1 : 0];
            C[static_cast<size_t>(m) * N + (p1 ? c - N0 : c)] = __float2half_rn(
                tile[(warp * 16 + r) * NPAD + c]);
        }
        return;
    }

    float* ws = args.ws;
    for (int o = lane; o < 16 * nt; o += 32) {
        const int r = o / nt;
        const int c = o - r * nt;
        const int m = row0 + warp * 16 + r;
        if (m >= M)
            continue;
        ws[(static_cast<size_t>(ks) * M + m) * nt + c] = tile[(warp * 16 + r) * NPAD + c];
    }
}

// Four columns per thread, every split's partial in flight before the first add. N0 % 8 == 0
// keeps a float4 inside one pair.
__global__ void __launch_bounds__(kReduceThreads) gemm_f16_narrow_prefill_reduce_kernel(PrefillArgs args) {
    pdl_wait();
    const int M = args.M;
    const int nt = args.n_total;
    const int nt4 = nt >> 2;
    const int o = blockIdx.x * kReduceThreads + threadIdx.x;
    if (o >= M * nt4)
        return;
    const int m = o / nt4;
    const int c = (o - m * nt4) * 4;
    const float* ws = args.ws;
    float4 p[kMaxSplit];
#pragma unroll
    for (int k2 = 0; k2 < kMaxSplit; ++k2)
        p[k2] = (k2 < args.split)
                    ? __ldcg(reinterpret_cast<const float4*>(&ws[(static_cast<size_t>(k2) * M + m) * nt + c]))
                    : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 sum = p[0];
#pragma unroll
    for (int k2 = 1; k2 < kMaxSplit; ++k2) {
        if (k2 < args.split) {
            sum.x += p[k2].x;
            sum.y += p[k2].y;
            sum.z += p[k2].z;
            sum.w += p[k2].w;
        }
    }
    pdl_trigger();
    const int N0 = args.N[0];
    const bool p1 = c >= N0;
    half* C = args.C[p1 ? 1 : 0];
    const int N = args.N[p1 ? 1 : 0];
    half2* dst = reinterpret_cast<half2*>(C + static_cast<size_t>(m) * N + (p1 ? c - N0 : c));
    dst[0] = __floats2half2_rn(sum.x, sum.y);
    dst[1] = __floats2half2_rn(sum.z, sum.w);
}

int row_tiles(int M) { return (M + kBM - 1) / kBM; }

int g_max_split = kMaxSplit;
int g_target_ctas = kTargetCtas;

template <int NPAD>
void launch(const PrefillArgs& args, const dim3& grid, cudaStream_t stream) {
    pdl::enable_kernel(gemm_f16_narrow_prefill_kernel<NPAD>);
    pdl::launch(gemm_f16_narrow_prefill_kernel<NPAD>, grid, dim3(kThreads), size_t(0), stream, args);
    IMP_CUDA_CHECK_LAUNCH();
}

}  // namespace

void gemm_f16_narrow_prefill_tune(int max_split, int target_ctas) {
    g_max_split = (max_split > 0 && max_split <= kMaxSplit) ? max_split : kMaxSplit;
    g_target_ctas = target_ctas > 0 ? target_ctas : kTargetCtas;
}

size_t gemm_f16_narrow_prefill_workspace_bytes(int M, int n_total, int split) {
    if (split <= 1)
        return 0;
    return static_cast<size_t>(split) * M * n_total * sizeof(float);
}

bool gemm_f16_narrow_prefill(const half* A, int M, int K, const half* W0, half* C0, int N0, const half* W1,
                             half* C1, int N1, void* ws, size_t ws_bytes, cudaStream_t stream) {
    if (!A || !W0 || !C0 || M < 1 || K < kBK || (K % kBK) != 0)
        return false;
    if (N0 <= 0 || (N0 % 8) != 0 || N1 < 0 || (N1 % 8) != 0 || (N1 > 0 && (!W1 || !C1)))
        return false;
    const int n_total = N0 + N1;
    if (n_total > kMaxN)
        return false;
    if ((reinterpret_cast<uintptr_t>(A) & 15) || (reinterpret_cast<uintptr_t>(W0) & 15) ||
        (W1 && (reinterpret_cast<uintptr_t>(W1) & 15)) || (reinterpret_cast<uintptr_t>(C0) & 3) ||
        (C1 && (reinterpret_cast<uintptr_t>(C1) & 3)) || (ws && (reinterpret_cast<uintptr_t>(ws) & 15)))
        return false;

    // Split-K until the grid reaches kTargetCtas CTAs, every CTA keeps >= kMinKPerCta of
    // BK-aligned K, and the partials fit the workspace (split 1 needs none).
    const int tiles = row_tiles(M);
    int split = 1;
    while (split < g_max_split && tiles * split < g_target_ctas && (K % (split * 2 * kBK)) == 0 &&
           K / (split * 2) >= kMinKPerCta)
        split *= 2;
    while (split > 1 && (!ws || ws_bytes < gemm_f16_narrow_prefill_workspace_bytes(M, n_total, split)))
        split /= 2;

    PrefillArgs args{};
    args.A = A;
    args.M = M;
    args.K = K;
    args.W[0] = W0;
    args.W[1] = W1;
    args.C[0] = C0;
    args.C[1] = C1;
    args.N[0] = N0;
    args.N[1] = N1;
    args.n_total = n_total;
    args.split = split;
    args.k_cta = K / split;
    args.ws = split > 1 ? static_cast<float*>(ws) : nullptr;

    const dim3 grid(tiles, split);
    if (n_total <= 32)
        launch<32>(args, grid, stream);
    else if (n_total <= 64)
        launch<64>(args, grid, stream);
    else if (n_total <= 96)
        launch<96>(args, grid, stream);
    else
        launch<128>(args, grid, stream);
    if (split > 1) {
        const int outputs = M * (n_total / 4);
        const dim3 rgrid((outputs + kReduceThreads - 1) / kReduceThreads);
        pdl::enable_kernel(gemm_f16_narrow_prefill_reduce_kernel);
        pdl::launch(gemm_f16_narrow_prefill_reduce_kernel, rgrid, dim3(kReduceThreads), size_t(0), stream,
                    args);
        IMP_CUDA_CHECK_LAUNCH();
    }
    return true;
}

}  // namespace imp
