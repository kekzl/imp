// Narrow FP16 GEMM for prefill rows, up to two (W,C) pairs per launch. Grid (row tiles) x split,
// the split CTAs of a row tile form one thread-block cluster. Block tile BM x NPAD x 32 (NPAD =
// N0+N1 rounded up to 16, <= 128; BM = 16 * WM), 4 warps as WM row x (4 / WM) column groups,
// mma.sync m16n8k16 FP32 accumulate from ldmatrix on XOR-swizzled 64 B smem rows (the WMMA form
// lost 72 % of its shared-load wavefronts to bank conflicts), cp.async pipeline for A and [W0; W1].
// Split > 1: every CTA parks its fragments in ws (L2), one cluster barrier, then CTA r sums its
// share over the split partials in rank order: deterministic, one launch (the separate reduce
// launch was 3.2 of 15 us at M=512; DSMEM reads lose to L2 on sm_120, gemm_f16_tile.cu).

#include "compute/gemm_f16_narrow_prefill.h"
#include "core/logging.h"
#include "core/pdl.h"
#include "core/pdl_device.cuh"

#include <cooperative_groups.h>
#include <cstdint>

namespace imp {
namespace {

constexpr int kBK = 32;
constexpr int kWarps = 4;
constexpr int kThreads = kWarps * 32;
constexpr int kMaxN = 128;
constexpr int kMaxSplit = 8;  // portable cluster size
constexpr int kTargetCtas = 256;
constexpr int kMinKPerCta = 128;  // >= 4 BK iterations per CTA
constexpr int kChunkHalves = 8;   // one 16 B cp.async
constexpr int kChunksPerRow = kBK / kChunkHalves;
constexpr int kWsRowPad = 64;  // ws rows per tile, the largest BM

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
    float4* ws;  // [tile][split][frag][thread], fragment-native
};

__device__ __forceinline__ void cp_async_cg16_zero(uint32_t s, const void* gmem_ptr, bool valid) {
    const int src_size = valid ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem_ptr), "r"(src_size));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n"); }
template <int N>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

// Byte offset of 16 B chunk c (0..3) of 64 B row r: chunk ^ ((r >> 1) & 3) puts the 8 rows of an
// ldmatrix 8x8 matrix (and 8 consecutive cp.async threads) on 8 distinct bank quads.
__device__ __forceinline__ uint32_t swz(int r, int c) {
    return static_cast<uint32_t>((r * kBK * 2) + ((c ^ ((r >> 1) & 3)) * 16));
}

template <int NPAD, int WM>
struct Cfg {
    static constexpr int kBM = 16 * WM;
    static constexpr int kWN = kWarps / WM;    // column groups
    static constexpr int kWCols = NPAD / kWN;  // columns per warp
    static constexpr int kNF = kWCols / 8;     // n8 fragments per warp
    static constexpr int kStageBytes = (kBM + NPAD) * kBK * 2;
    static constexpr int kStagesRaw = 49152 / kStageBytes;
    static constexpr int kStages = kStagesRaw > 6 ? 6 : kStagesRaw;
    static constexpr int kSmemBytes = kStages * kStageBytes;
    static_assert(kWCols % 16 == 0 && kStages >= 3, "ldmatrix.x4 B pairs, pipeline depth");
};

// One 16 B chunk per thread per 32 tile rows: A rows of this row tile, then W rows 0..NPAD.
// Source pointers and smem offsets are fixed per thread; a stage only adds its k offset.
template <int NPAD, int BM>
struct Loader {
    static constexpr int kN = (BM + NPAD) / (kThreads / kChunksPerRow);
    const half* src[kN];
    uint32_t dst[kN];
    unsigned ok = 0;

    __device__ __forceinline__ Loader(const PrefillArgs& args, int row0, int kt0) {
        const int kc = static_cast<int>(threadIdx.x) % kChunksPerRow;
        const int r0 = static_cast<int>(threadIdx.x) / kChunksPerRow;
        const size_t k_off = (static_cast<size_t>(kt0) * kBK) + static_cast<size_t>(kc * kChunkHalves);
#pragma unroll
        for (int i = 0; i < kN; ++i) {
            const int r = r0 + (i * (kThreads / kChunksPerRow));  // tile row: A below BM, then W
            bool valid;
            const half* row;
            if (r < BM) {
                valid = row0 + r < args.M;
                row = args.A + (static_cast<size_t>(valid ? row0 + r : 0) * args.K);
            } else {
                const int n = r - BM;
                valid = n < args.n_total;
                const bool p1 = valid && n >= args.N[0];
                row = (p1 ? args.W[1] : args.W[0]) +
                      (static_cast<size_t>(valid ? (p1 ? n - args.N[0] : n) : 0) * args.K);
            }
            src[i] = row + k_off;
            dst[i] = swz(r, kc);
            ok |= valid ? (1u << i) : 0u;
        }
    }

    // k-tile kt0 + step into the stage at smem address s.
    __device__ __forceinline__ void load(uint32_t s, int step) const {
#pragma unroll
        for (int i = 0; i < kN; ++i)
            cp_async_cg16_zero(s + dst[i], src[i] + (step * kBK), (ok >> i) & 1u);
    }
};

__device__ __forceinline__ void ldsm_x4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2,
                                        uint32_t& r3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(addr));
}

__device__ __forceinline__ void mma_f32(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Output column `col` (even, < n_total) of row m as a half2: N0 % 8 == 0 keeps an n8 fragment in one pair.
__device__ __forceinline__ void store_out(const PrefillArgs& args, int m, int col, float x, float y) {
    const bool p1 = col >= args.N[0];
    half* C = args.C[p1 ? 1 : 0];
    const int N = args.N[p1 ? 1 : 0];
    *reinterpret_cast<half2*>(C + (static_cast<size_t>(m) * N) +
                              (p1 ? col - args.N[0] : col)) = __floats2half2_rn(x, y);
}

template <int NPAD, int WM>
__global__ void __launch_bounds__(kThreads) gemm_f16_narrow_prefill_kernel(PrefillArgs args) {
    using C = Cfg<NPAD, WM>;
    constexpr int BM = C::kBM;
    constexpr int NF = C::kNF;
    constexpr int kStages = C::kStages;
    __shared__ __align__(128) unsigned char smem_raw[C::kSmemBytes];
    const uint32_t s0 = static_cast<uint32_t>(__cvta_generic_to_shared(smem_raw));

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int wr = warp % WM;                 // 16-row group
    const int wc0 = (warp / WM) * C::kWCols;  // first column
    const int ks = blockIdx.y;
    const int row0 = blockIdx.x * BM;
    const int kt0 = (ks * args.k_cta) / kBK;
    const int n_tiles = args.k_cta / kBK;
    const Loader<NPAD, BM> ld(args, row0, kt0);
    pdl_wait();  // A is the previous kernel's output

    float acc[NF][4];
#pragma unroll
    for (int j = 0; j < NF; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e)
            acc[j][e] = 0.0f;

#pragma unroll
    for (int s = 0; s < kStages - 1; ++s) {
        if (s < n_tiles)
            ld.load(s0 + (s * C::kStageBytes), s);
        cp_async_commit();
    }

    // ldmatrix lane roles: A x4 = rows (lane & 15), k half (lane >> 4); B x4 = n rows (lane & 7) +
    // 8 * (lane >> 4), k half (lane >> 3) & 1.
    const int a_row = (wr * 16) + (lane & 15);
    const int a_kh = lane >> 4;
    const int b_row = BM + wc0 + (lane & 7) + ((lane >> 4) << 3);
    const int b_kh = (lane >> 3) & 1;

    for (int kt = 0; kt < n_tiles; ++kt) {
        cp_async_wait_group<kStages - 2>();
        __syncthreads();
        const int nxt = kt + kStages - 1;
        if (nxt < n_tiles)
            ld.load(s0 + ((nxt % kStages) * C::kStageBytes), nxt);
        cp_async_commit();

        const uint32_t st = s0 + ((kt % kStages) * C::kStageBytes);
#pragma unroll
        for (int kk = 0; kk < kBK / 16; ++kk) {
            uint32_t a[4];
            ldsm_x4(st + swz(a_row, (kk * 2) + a_kh), a[0], a[1], a[2], a[3]);
#pragma unroll
            for (int jb = 0; jb < NF / 2; ++jb) {
                uint32_t b0, b1, b2, b3;
                ldsm_x4(st + swz(b_row + (jb * 16), (kk * 2) + b_kh), b0, b1, b2, b3);
                mma_f32(acc[2 * jb], a, b0, b1);
                mma_f32(acc[(2 * jb) + 1], a, b2, b3);
            }
        }
    }
    cp_async_wait_group<0>();
    pdl_trigger();

    // Fragment f of a warp: rows wr*16 + g and +8, columns wc0 + 8f + 2t (+1), g = lane / 4, t = lane % 4.
    if (args.split == 1) {
        const int m = row0 + (wr * 16) + (lane >> 2);
#pragma unroll
        for (int f = 0; f < NF; ++f) {
            const int col = wc0 + (f * 8) + ((lane & 3) * 2);
            if (col >= args.n_total)
                continue;
            if (m < args.M)
                store_out(args, m, col, acc[f][0], acc[f][1]);
            if (m + 8 < args.M)
                store_out(args, m + 8, col, acc[f][2], acc[f][3]);
        }
        return;
    }

    const int S = args.split;
    float4* tile_ws = args.ws + (static_cast<size_t>(blockIdx.x) * S * NF * kThreads);
    float4* own = tile_ws + (static_cast<size_t>(ks) * NF * kThreads);
#pragma unroll
    for (int f = 0; f < NF; ++f)
        __stcg(own + static_cast<ptrdiff_t>((f * kThreads) + threadIdx.x),
               make_float4(acc[f][0], acc[f][1], acc[f][2], acc[f][3]));

    cooperative_groups::this_cluster().sync();  // release/acquire: the peers' partials are visible

    constexpr int kItems = NF * kThreads;
    const int i0 = (ks * kItems) / S;
    const int i1 = ((ks + 1) * kItems) / S;
    for (int i = i0 + static_cast<int>(threadIdx.x); i < i1; i += kThreads) {
        float4 v[kMaxSplit];  // every partial in flight before the first add
#pragma unroll
        for (int q = 0; q < kMaxSplit; ++q)
            if (q < S)
                v[q] = __ldcg(tile_ws + (static_cast<size_t>(q) * kItems) + i);
        float4 sum = v[0];
#pragma unroll
        for (int q = 1; q < kMaxSplit; ++q)
            if (q < S) {
                sum.x += v[q].x;
                sum.y += v[q].y;
                sum.z += v[q].z;
                sum.w += v[q].w;
            }
        const int f = i / kThreads;
        const int t = i - (f * kThreads);
        const int tw = t >> 5;
        const int m = row0 + ((tw % WM) * 16) + ((t & 31) >> 2);
        const int col = ((tw / WM) * C::kWCols) + (f * 8) + ((t & 3) * 2);
        if (col >= args.n_total)
            continue;
        if (m < args.M)
            store_out(args, m, col, sum.x, sum.y);
        if (m + 8 < args.M)
            store_out(args, m + 8, col, sum.z, sum.w);
    }
}

int row_tiles(int M, int bm) { return (M + bm - 1) / bm; }

int g_max_split = kMaxSplit;
int g_target_ctas = kTargetCtas;
int g_bm = 0;

template <int NPAD, int WM>
bool launch(const PrefillArgs& args, int tiles, cudaStream_t stream) {
    auto* kern = gemm_f16_narrow_prefill_kernel<NPAD, WM>;
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3(tiles, args.split);
    cfg.blockDim = dim3(kThreads);
    cfg.dynamicSmemBytes = 0;
    cfg.stream = stream;
    cudaLaunchAttribute attrs[2]{};
    int n = 0;
    if (args.split > 1) {
        attrs[n].id = cudaLaunchAttributeClusterDimension;
        attrs[n].val.clusterDim.x = 1;
        attrs[n].val.clusterDim.y = static_cast<unsigned>(args.split);
        attrs[n].val.clusterDim.z = 1;
        ++n;
    }
    pdl::enable(reinterpret_cast<const void*>(kern));
    if (pdl::is_enabled(reinterpret_cast<const void*>(kern))) {
        attrs[n].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attrs[n].val.programmaticStreamSerializationAllowed = 1;
        ++n;
    }
    cfg.attrs = attrs;
    cfg.numAttrs = static_cast<unsigned>(n);
    if (cudaLaunchKernelEx(&cfg, kern, args) != cudaSuccess) {
        (void)cudaGetLastError();  // refused (cluster does not fit): the caller keeps its path
        return false;
    }
    return true;
}

template <int WM>
bool launch_npad(const PrefillArgs& args, int tiles, cudaStream_t stream) {
    if (args.n_total <= 32)
        return launch<32, WM>(args, tiles, stream);
    if (args.n_total <= 64)
        return launch<64, WM>(args, tiles, stream);
    if (args.n_total <= 96)
        return launch<96, WM>(args, tiles, stream);
    return launch<128, WM>(args, tiles, stream);
}

}  // namespace

void gemm_f16_narrow_prefill_tune(int max_split, int target_ctas, int bm) {
    g_max_split = (max_split > 0 && max_split <= kMaxSplit) ? max_split : kMaxSplit;
    g_target_ctas = target_ctas > 0 ? target_ctas : kTargetCtas;
    g_bm = (bm == 32 || bm == 64) ? bm : 0;
}

size_t gemm_f16_narrow_prefill_workspace_bytes(int M, int n_total, int split) {
    if (split <= 1)
        return 0;
    const int npad = (n_total + 15) & ~15;
    return static_cast<size_t>(row_tiles(M, kWsRowPad)) * kWsRowPad * split * npad * sizeof(float);
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

    // 32-row tiles while 64-row tiles at the largest split stay under the CTA target. Split-K
    // until the grid reaches it, every CTA keeps >= kMinKPerCta of BK-aligned K, and the
    // partials fit the workspace (split 1 needs none).
    const int bm = g_bm != 0 ? g_bm : (row_tiles(M, 64) * g_max_split < g_target_ctas ? 32 : 64);
    const int tiles = row_tiles(M, bm);
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
    args.ws = split > 1 ? static_cast<float4*>(ws) : nullptr;
    return bm == 32 ? launch_npad<2>(args, tiles, stream) : launch_npad<4>(args, tiles, stream);
}

}  // namespace imp
