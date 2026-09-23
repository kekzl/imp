// Memory peaks: DRAM read/write/copy, L2 working-set sweep, SMEM per SM, TMA bulk loads.
// DRAM rows use a 2 GiB buffer (21x the 96 MB L2); each rate is the best launch shape found.
#include "peaks_common.cuh"

#include <cstdint>

namespace peaks {
namespace {

constexpr size_t kDramBytes = size_t(2) << 30;

__device__ __forceinline__ uint32_t fold(uint4 v) { return v.x ^ v.y ^ v.z ^ v.w; }

// Reads `ws` uint4 `passes` times through L2 only (ld.global.cg: no L1 reuse across passes).
__global__ void read_kernel(const uint4* __restrict__ p, size_t ws, int passes, uint32_t* sink) {
    const size_t stride = size_t(gridDim.x) * blockDim.x;
    uint32_t acc = 0;
    for (int pass = 0; pass < passes; ++pass) {
        size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
        for (; i + 3 * stride < ws; i += 4 * stride) {
            uint4 a = __ldcg(p + i), b = __ldcg(p + i + stride);
            uint4 c = __ldcg(p + i + 2 * stride), d = __ldcg(p + i + 3 * stride);
            acc ^= fold(a) ^ fold(b) ^ fold(c) ^ fold(d);
        }
        for (; i < ws; i += stride)
            acc ^= fold(__ldcg(p + i));
    }
    if (acc == 0x9e3779b9u)
        sink[0] = acc;  // data-dependent: keeps the loads alive
}

__global__ void write_kernel(uint4* __restrict__ p, size_t n) {
    const size_t stride = size_t(gridDim.x) * blockDim.x;
    const uint4 v = make_uint4(threadIdx.x, blockIdx.x, 1u, 2u);
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride)
        p[i] = v;
}

__global__ void copy_kernel(const uint4* __restrict__ src, uint4* __restrict__ dst, size_t n) {
    const size_t stride = size_t(gridDim.x) * blockDim.x;
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    for (; i + stride < n; i += 2 * stride) {
        uint4 a = __ldcs(src + i), b = __ldcs(src + i + stride);
        dst[i] = a;
        dst[i + stride] = b;
    }
    for (; i < n; i += stride)
        dst[i] = __ldcs(src + i);
}

// Conflict-free 128-bit shared loads, `iters` per thread; one block of 1024 threads per SM.
__global__ void smem_kernel(int iters, uint32_t* sink) {
    __shared__ uint4 buf[4096];  // 64 KiB
    for (int i = threadIdx.x; i < 4096; i += blockDim.x)
        buf[i] = make_uint4(i, i + 1, i + 2, i + 3);
    __syncthreads();
    uint32_t acc = 0;
    uint32_t base = static_cast<uint32_t>(__cvta_generic_to_shared(buf));
    uint32_t idx = threadIdx.x;
#pragma unroll 8
    for (int it = 0; it < iters; ++it) {
        uint4 v;  // ld.volatile: ptxas CSEs plain ld.shared on the period-4 addresses (read 474 B/clk/SM)
        asm volatile("ld.volatile.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n"
                     : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                     : "r"(base + idx * 16u));
        acc ^= fold(v);
        idx = (idx + 1024u) & 4095u;
    }
    if (acc == 0x9e3779b9u)
        sink[0] = acc;
}

// TMA (cp.async.bulk) global->shared, one issuing thread per CTA, kStages x kChunk ring.
constexpr int kChunk = 16384;
constexpr int kStages = 3;

__device__ __forceinline__ void mbar_init(uint64_t* bar) {
    uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" ::"r"(a));
}

__device__ __forceinline__ void tma_issue(uint8_t* dst, const uint8_t* src, uint64_t* bar) {
    uint32_t d = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    uint32_t b = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(b), "r"(kChunk));
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" ::"r"(d),
        "l"(src), "r"(kChunk), "r"(b)
        : "memory");
}

__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {
    uint32_t b = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile(
        "{\n .reg .pred p;\n WAIT:\n"
        " mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
        " @!p bra WAIT;\n}\n" ::"r"(b),
        "r"(parity)
        : "memory");
}

// Loads `total` chunks; chunk c reads offset (c % ws_chunks) * kChunk (ws_chunks < total = L2 run).
__global__ void tma_kernel(const uint8_t* __restrict__ src, size_t total, size_t ws_chunks, uint32_t* sink) {
    extern __shared__ __align__(128) uint8_t smem[];
    uint64_t* bars = reinterpret_cast<uint64_t*>(smem + kStages * kChunk);
    if (threadIdx.x != 0)
        return;
    for (int s = 0; s < kStages; ++s)
        mbar_init(&bars[s]);
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    size_t c = blockIdx.x;
    int inflight = 0;
    for (int s = 0; s < kStages && c < total; ++s, c += gridDim.x, ++inflight)
        tma_issue(smem + s * kChunk, src + (c % ws_chunks) * kChunk, &bars[s]);
    uint32_t phase = 0, acc = 0;
    for (int s = 0; inflight > 0; s = (s + 1) % kStages) {
        mbar_wait(&bars[s], (phase >> s) & 1u);
        phase ^= 1u << s;
        acc ^= smem[s * kChunk];
        --inflight;
        if (c < total) {
            tma_issue(smem + s * kChunk, src + (c % ws_chunks) * kChunk, &bars[s]);
            c += gridDim.x;
            ++inflight;
        }
    }
    if (acc == 0x9eu)
        sink[0] = acc;
}

struct Shape {
    int blocks_per_sm, threads;
};
constexpr Shape kShapes[] = {{1, 1024}, {2, 512}, {4, 256}, {8, 256}, {16, 128}};

std::string shape_note(const Shape& s) {
    return "[" + std::to_string(s.blocks_per_sm) + "x" + std::to_string(s.threads) + "/SM]";
}

// Picks the fastest launch shape for `run(grid, threads)`; returns its per-launch times.
template <class F>
std::vector<double> best_shape(F&& run, int inner, std::string* note) {
    std::vector<double> best;
    double best_med = 1e30;
    for (const Shape& s : kShapes) {
        const int grid = s.blocks_per_sm * sm_count();
        auto ms = time_launches([&] { run(grid, s.threads); }, inner, 5, 0.5);
        std::vector<double> sorted = ms;
        std::sort(sorted.begin(), sorted.end());
        if (sorted[sorted.size() / 2] < best_med) {
            best_med = sorted[sorted.size() / 2];
            best = ms;
            *note = shape_note(s);
        }
    }
    return best;
}

}  // namespace

void run_mem() {
    uint8_t *a = nullptr, *b = nullptr;
    uint32_t* sink = nullptr;
    PK_CHECK(cudaMalloc(&a, kDramBytes));
    PK_CHECK(cudaMalloc(&b, kDramBytes));
    PK_CHECK(cudaMalloc(&sink, 64));
    PK_CHECK(cudaMemset(a, 1, kDramBytes));
    PK_CHECK(cudaMemset(b, 2, kDramBytes));
    const size_t nvec = kDramBytes / sizeof(uint4);
    auto* va = reinterpret_cast<uint4*>(a);
    auto* vb = reinterpret_cast<uint4*>(b);
    std::string note;

    auto rd = best_shape([&](int g, int t) { read_kernel<<<g, t>>>(va, nvec, 1, sink); }, 3, &note);
    record("dram", "read", "GB/s", kDramBytes / 1e9, rd, note);
    auto wr = best_shape([&](int g, int t) { write_kernel<<<g, t>>>(va, nvec); }, 3, &note);
    record("dram", "write", "GB/s", kDramBytes / 1e9, wr, note);
    auto cp = best_shape([&](int g, int t) { copy_kernel<<<g, t>>>(va, vb, nvec); }, 3, &note);
    record("dram", "copy (read+write bytes)", "GB/s", 2.0 * kDramBytes / 1e9, cp, note);

    // L2 size effect: same total traffic (~2 GiB) per launch over a shrinking working set.
    for (size_t mb : {8, 16, 32, 48, 64, 80, 96, 112, 128, 192, 256, 512}) {
        const size_t ws = (mb << 20) / sizeof(uint4);
        const int passes = static_cast<int>(std::max<size_t>(1, kDramBytes / (mb << 20)));
        auto ms = best_shape([&](int g, int t) { read_kernel<<<g, t>>>(va, ws, passes, sink); }, 2, &note);
        record("l2", "read ws=" + std::to_string(mb) + "MB", "GB/s", double(passes) * (mb << 20) / 1e9, ms,
               note);
    }

    // SMEM: bytes per SM per clock follow from the sampled SM clock (report divides later).
    const int iters = 1 << 16;
    auto sm = time_launches([&] { smem_kernel<<<sm_count(), 1024>>>(iters, sink); }, 5);
    const double smem_bytes = double(sm_count()) * 1024 * iters * sizeof(uint4);
    record("smem", "ld.shared.v4 all SMs", "GB/s", smem_bytes / 1e9, sm, "[1x1024/SM, 64 KiB]");
    record("smem", "ld.shared.v4 per SM", "GB/s", smem_bytes / sm_count() / 1e9, sm);

    // TMA bulk loads: DRAM stream and an 8 MiB L2-resident working set.
    const int tma_smem = kStages * kChunk + 64;
    PK_CHECK(cudaFuncSetAttribute(tma_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, tma_smem));
    const size_t total = kDramBytes / kChunk;
    for (int bps : {1, 2}) {
        const int grid = bps * sm_count();
        auto t = time_launches([&] { tma_kernel<<<grid, 32, tma_smem>>>(a, total, total, sink); }, 3);
        record("tma", "bulk 16KiB DRAM " + std::to_string(bps) + " CTA/SM", "GB/s", kDramBytes / 1e9, t,
               "[3 stages]");
        const size_t ws = (size_t(8) << 20) / kChunk;
        auto l = time_launches([&] { tma_kernel<<<grid, 32, tma_smem>>>(a, total, ws, sink); }, 3);
        record("tma", "bulk 16KiB L2 ws=8MB " + std::to_string(bps) + " CTA/SM", "GB/s", kDramBytes / 1e9, l,
               "[3 stages]");
    }

    PK_CHECK(cudaFree(a));
    PK_CHECK(cudaFree(b));
    PK_CHECK(cudaFree(sink));
}

}  // namespace peaks
