// tools/analysis/vmm_wsl2_probe.cu
// WSL2/WDDM viability spike for the CUDA virtual-memory-management APIs
// (docs/MEMORY_ARCHITECTURE.md A3.1 — the hard gate on A7 step 7, the growable
// VMM backend for the KV block pool).
//
// No imp dependencies. Build + run inside a CUDA container:
//   docker run --rm --gpus all -v $PWD:/w -w /w imp:builder \
//     bash -lc 'nvcc -O2 -arch=sm_120a -o /tmp/vmm_probe \
//                 tools/analysis/vmm_wsl2_probe.cu -lcuda && /tmp/vmm_probe'
//
// Establishes, each with printed evidence:
//   [1] cuMemAddressReserve of a large VA range costs no physical memory.
//   [2] cuMemCreate + cuMemMap + cuMemSetAccess in 256 MiB granules works and
//       cudaMemGetInfo reflects exactly the committed bytes.
//   [3] The base address is INVARIANT across a grow/shrink cycle (I3).
//   [4] Data in one committed region survives decommit+recommit of another.
//   [5] cuMemGetAllocationGranularity minimum + recommended on sm_120a.
//   [6] Decommitted memory is actually returned (free VRAM goes back up), and
//       which of cuMemUnmap / cuMemRelease is the call that returns it.
//   [7] Bonus, because A3.1 names it explicitly: a CUDA-graph-captured kernel
//       still reads correct data from a fixed VA after the pool grew underneath.
//   [8] Bonus: per-op latency of commit/decommit (WDDM has a history of making
//       driver-side mapping calls expensive).
//
// Exit code 0 = GO, non-zero = number of failed checks.

#include <cuda.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

// ---------------------------------------------------------------------------
// plumbing
// ---------------------------------------------------------------------------

static int g_fail = 0;
static int g_check = 0;

static const char* cu_err(CUresult r) {
    const char* n = nullptr;
    cuGetErrorName(r, &n);
    return n ? n : "UNKNOWN";
}
static const char* cu_msg(CUresult r) {
    const char* s = nullptr;
    cuGetErrorString(r, &s);
    return s ? s : "?";
}

// Records a driver-API failure but keeps going, so one broken call does not hide
// the rest of the picture.
#define CU_TRY(expr)                                                                   \
    ([&]() -> CUresult {                                                               \
        CUresult r_ = (expr);                                                          \
        if (r_ != CUDA_SUCCESS) {                                                      \
            printf("  !! %s failed: %s (%s)  [line %d]\n", #expr, cu_err(r_), cu_msg(r_), __LINE__); \
            ++g_fail;                                                                  \
        }                                                                              \
        return r_;                                                                     \
    })()

#define CUDA_TRY(expr)                                                                 \
    ([&]() -> cudaError_t {                                                            \
        cudaError_t e_ = (expr);                                                        \
        if (e_ != cudaSuccess) {                                                        \
            printf("  !! %s failed: %s  [line %d]\n", #expr, cudaGetErrorString(e_), __LINE__); \
            ++g_fail;                                                                   \
        }                                                                               \
        return e_;                                                                      \
    })()

static void check(bool ok, const char* what) {
    ++g_check;
    if (!ok) ++g_fail;
    printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
}

static const size_t kMiB = 1024ull * 1024ull;
static const size_t kGiB = 1024ull * kMiB;

static double mib(size_t b) { return double(b) / double(kMiB); }

struct MemInfo {
    size_t free_b = 0;
    size_t total_b = 0;
};

static MemInfo mem_info() {
    MemInfo m;
    cudaMemGetInfo(&m.free_b, &m.total_b);
    return m;
}

// signed MiB delta of *used* memory (free went down => used went up => positive)
static double used_delta_mib(size_t free_before, size_t free_after) {
    return (double(free_before) - double(free_after)) / double(kMiB);
}

static double now_ms() {
    using clock = std::chrono::steady_clock;
    static const clock::time_point t0 = clock::now();
    return std::chrono::duration<double, std::milli>(clock::now() - t0).count();
}

// ---------------------------------------------------------------------------
// kernels
// ---------------------------------------------------------------------------

// Deterministic pattern so a decommit/recommit elsewhere cannot silently pass.
__device__ __forceinline__ uint32_t pattern_at(size_t i, uint32_t seed) {
    return (uint32_t)(i * 2654435761u) ^ (seed * 0x9E3779B9u);
}

__global__ void fill_pattern(uint32_t* p, size_t n, uint32_t seed) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n; i += stride) p[i] = pattern_at(i, seed);
}

__global__ void check_pattern(const uint32_t* p, size_t n, uint32_t seed, unsigned long long* bad) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    unsigned long long local = 0;
    for (; i < n; i += stride)
        if (p[i] != pattern_at(i, seed)) ++local;
    if (local) atomicAdd(bad, local);
}

// Graph-captured body: reads a fixed VA, writes a reduction to a fixed sink.
__global__ void sum_stride(const uint32_t* p, size_t n, unsigned long long* out) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    unsigned long long local = 0;
    for (; i < n; i += stride) local += p[i];
    atomicAdd(out, local);
}

// ---------------------------------------------------------------------------
// VMM pool
// ---------------------------------------------------------------------------

struct VmmPool {
    CUdeviceptr va = 0;              // reserved base — must never change
    size_t reserved = 0;
    size_t chunk = 0;
    std::vector<CUmemGenericAllocationHandle> handles;  // 0 == not committed
    CUmemAllocationProp prop{};
    CUmemAccessDesc access{};

    size_t committed_bytes() const {
        size_t n = 0;
        for (auto h : handles)
            if (h) n += chunk;
        return n;
    }

    // Commit chunk index `i`. Returns per-call latency in ms (-1 on failure).
    double commit(size_t i) {
        if (i >= handles.size() || handles[i]) return 0.0;
        double t0 = now_ms();
        CUmemGenericAllocationHandle h = 0;
        if (CU_TRY(cuMemCreate(&h, chunk, &prop, 0)) != CUDA_SUCCESS) return -1.0;
        if (CU_TRY(cuMemMap(va + i * chunk, chunk, 0, h, 0)) != CUDA_SUCCESS) {
            cuMemRelease(h);
            return -1.0;
        }
        if (CU_TRY(cuMemSetAccess(va + i * chunk, chunk, &access, 1)) != CUDA_SUCCESS) return -1.0;
        handles[i] = h;
        return now_ms() - t0;
    }

    // Decommit chunk index `i` (unmap + release). Returns latency in ms.
    double decommit(size_t i) {
        if (i >= handles.size() || !handles[i]) return 0.0;
        double t0 = now_ms();
        CU_TRY(cuMemUnmap(va + i * chunk, chunk));
        CU_TRY(cuMemRelease(handles[i]));
        handles[i] = 0;
        return now_ms() - t0;
    }

    uint32_t* ptr(size_t i) const { return reinterpret_cast<uint32_t*>(va + i * chunk); }
};

// ---------------------------------------------------------------------------

int main() {
    printf("=== imp VMM WSL2/WDDM spike (docs/MEMORY_ARCHITECTURE.md A3.1) ===\n\n");

    CU_TRY(cuInit(0));
    CUDA_TRY(cudaSetDevice(0));
    CUDA_TRY(cudaFree(nullptr));  // force primary-context creation before any measurement

    CUdevice dev = 0;
    CU_TRY(cuDeviceGet(&dev, 0));
    char name[256] = {0};
    cuDeviceGetName(name, sizeof(name), dev);

    // The probe's own scratch is allocated BEFORE every baseline reading, so it
    // cannot show up later as a phantom residual. (It does: an 8-byte cudaMalloc
    // costs a full 2 MiB page, which is exactly the VMM granule — easy to
    // misread as a VMM leak.)
    unsigned long long* d_bad = nullptr;
    CUDA_TRY(cudaMalloc(&d_bad, sizeof(unsigned long long)));

    int vmm_supported = 0, rdma = 0, gdr = 0, cc_major = 0, cc_minor = 0;
    cuDeviceGetAttribute(&vmm_supported, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, dev);
    cuDeviceGetAttribute(&rdma, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED, dev);
    cuDeviceGetAttribute(&gdr, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED, dev);
    cuDeviceGetAttribute(&cc_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
    cuDeviceGetAttribute(&cc_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);

    int driver_ver = 0, runtime_ver = 0;
    cudaDriverGetVersion(&driver_ver);
    cudaRuntimeGetVersion(&runtime_ver);

    MemInfo base = mem_info();
    printf("device            : %s (sm_%d%d)\n", name, cc_major, cc_minor);
    printf("driver / runtime  : %d / %d\n", driver_ver, runtime_ver);
    printf("VRAM total        : %.1f MiB   free at start: %.1f MiB\n", mib(base.total_b), mib(base.free_b));
    printf("VMM supported attr: %d   posix-fd handle: %d   gdr: %d\n\n", vmm_supported, rdma, gdr);

    check(vmm_supported == 1, "CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED == 1");
    if (!vmm_supported) {
        printf("\nNO-GO: the device reports no VMM support. Stopping.\n");
        return 1;
    }

    // Free-VRAM measurement noise floor: WDDM shares the card with the desktop
    // compositor, so every delta below has to be read against this.
    printf("-- free-VRAM noise floor (5 back-to-back reads) --\n");
    size_t nmin = SIZE_MAX, nmax = 0;
    for (int i = 0; i < 5; ++i) {
        size_t f = mem_info().free_b;
        printf("   read %d: %.2f MiB\n", i, mib(f));
        if (f < nmin) nmin = f;
        if (f > nmax) nmax = f;
    }
    printf("   spread: %.3f MiB\n\n", mib(nmax - nmin));

    // -----------------------------------------------------------------------
    // [5] granularity (printed first — it constrains everything below)
    // -----------------------------------------------------------------------
    printf("== [5] cuMemGetAllocationGranularity ==\n");
    CUmemAllocationProp prop{};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = (int)dev;

    size_t gran_min = 0, gran_rec = 0;
    CU_TRY(cuMemGetAllocationGranularity(&gran_min, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    CU_TRY(cuMemGetAllocationGranularity(&gran_rec, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
    printf("   MINIMUM     = %zu bytes (%.2f MiB)\n", gran_min, mib(gran_min));
    printf("   RECOMMENDED = %zu bytes (%.2f MiB)\n", gran_rec, mib(gran_rec));
    check(gran_min > 0 && gran_rec > 0, "granularity query returns non-zero min and recommended");

    const size_t kChunk = 256 * kMiB;   // the commit granule the design proposes
    const size_t kReserve = 24 * kGiB;  // VA for the maximum KV the config could want
    check(kChunk % gran_min == 0, "256 MiB commit chunk is a multiple of MINIMUM granularity");
    check(kChunk % gran_rec == 0, "256 MiB commit chunk is a multiple of RECOMMENDED granularity");
    printf("\n");

    // -----------------------------------------------------------------------
    // [1] reserve a large VA range; it must cost no physical memory
    // -----------------------------------------------------------------------
    printf("== [1] cuMemAddressReserve(%.0f GiB) costs no physical memory ==\n", double(kReserve) / double(kGiB));
    size_t free_pre_reserve = mem_info().free_b;

    VmmPool pool;
    pool.chunk = kChunk;
    pool.reserved = kReserve;
    pool.prop = prop;
    pool.access.location = prop.location;
    pool.access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    double t_res0 = now_ms();
    CUresult rres = CU_TRY(cuMemAddressReserve(&pool.va, kReserve, 0 /*alignment: default*/, 0 /*addr hint*/, 0));
    double t_res = now_ms() - t_res0;
    if (rres != CUDA_SUCCESS) {
        printf("\nNO-GO: cuMemAddressReserve failed. Stopping.\n");
        return g_fail ? g_fail : 1;
    }
    size_t free_post_reserve = mem_info().free_b;

    printf("   base VA         = 0x%llx\n", (unsigned long long)pool.va);
    printf("   reserve latency = %.3f ms\n", t_res);
    printf("   free before     = %.2f MiB\n", mib(free_pre_reserve));
    printf("   free after      = %.2f MiB\n", mib(free_post_reserve));
    printf("   used delta      = %+.3f MiB   (expected ~0)\n", used_delta_mib(free_pre_reserve, free_post_reserve));
    check(pool.va != 0, "cuMemAddressReserve of 24 GiB VA succeeded");
    check(pool.va % gran_rec == 0, "reserved base is aligned to RECOMMENDED granularity");
    check(used_delta_mib(free_pre_reserve, free_post_reserve) < 1.0,
          "VA reservation consumed < 1 MiB of physical VRAM");
    printf("\n");

    const size_t kNumChunks = kReserve / kChunk;
    pool.handles.assign(kNumChunks, 0);

    // -----------------------------------------------------------------------
    // [2] commit in 256 MiB granules; free VRAM must track committed bytes 1:1
    // -----------------------------------------------------------------------
    printf("== [2] commit in %.0f MiB granules; cudaMemGetInfo tracks committed bytes ==\n", mib(kChunk));
    const int kInitialCommit = 4;
    bool commit_tracks = true;
    double commit_ms_sum = 0.0;
    for (int i = 0; i < kInitialCommit; ++i) {
        size_t f0 = mem_info().free_b;
        double dt = pool.commit((size_t)i);
        size_t f1 = mem_info().free_b;
        commit_ms_sum += dt;
        double d = used_delta_mib(f0, f1);
        printf("   commit chunk %d @ 0x%llx : %+8.2f MiB used  (%.3f ms)\n", i,
               (unsigned long long)(pool.va + (size_t)i * kChunk), d, dt);
        if (d < mib(kChunk) * 0.98 || d > mib(kChunk) * 1.05) commit_tracks = false;
    }
    size_t free_after_commit4 = mem_info().free_b;
    printf("   total used delta vs pre-commit: %+.2f MiB (committed %.0f MiB)\n",
           used_delta_mib(free_post_reserve, free_after_commit4), mib(pool.committed_bytes()));
    check(pool.committed_bytes() == (size_t)kInitialCommit * kChunk, "4 x 256 MiB committed (handles live)");
    check(commit_tracks, "each commit moved cudaMemGetInfo by exactly one chunk (+/-5%)");
    printf("\n");

    // Sanity: the committed region is actually usable from a kernel.
    const size_t kElemsPerChunk = kChunk / sizeof(uint32_t);

    auto verify_chunk = [&](size_t idx, uint32_t seed, const char* what) {
        unsigned long long zero = 0, bad = 0;
        CUDA_TRY(cudaMemcpy(d_bad, &zero, sizeof(zero), cudaMemcpyHostToDevice));
        check_pattern<<<1024, 256>>>(pool.ptr(idx), kElemsPerChunk, seed, d_bad);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess) {
            printf("  !! kernel on chunk %zu: %s\n", idx, cudaGetErrorString(e));
            ++g_fail;
            return false;
        }
        CUDA_TRY(cudaMemcpy(&bad, d_bad, sizeof(bad), cudaMemcpyDeviceToHost));
        printf("   %s: chunk %zu mismatching words = %llu / %zu\n", what, idx, bad, kElemsPerChunk);
        return bad == 0;
    };

    printf("== kernel R/W sanity on committed VA ==\n");
    fill_pattern<<<1024, 256>>>(pool.ptr(0), kElemsPerChunk, 0xA5A5u);
    CUDA_TRY(cudaDeviceSynchronize());
    check(verify_chunk(0, 0xA5A5u, "post-fill"), "kernel writes and reads a VMM-mapped chunk correctly");
    printf("\n");

    // -----------------------------------------------------------------------
    // [3] base address invariance across a grow/shrink cycle
    // -----------------------------------------------------------------------
    printf("== [3] base address INVARIANT across grow/shrink (I3) ==\n");
    const CUdeviceptr base_at_start = pool.va;
    uint32_t* chunk0_at_start = pool.ptr(0);
    bool base_stable = true;

    auto grow_to = [&](size_t n, const char* label) {
        for (size_t i = 0; i < n; ++i) pool.commit(i);
        printf("   %-22s committed=%5.0f MiB  base=0x%llx  chunk0=%p  free=%.1f MiB\n", label,
               mib(pool.committed_bytes()), (unsigned long long)pool.va, (void*)pool.ptr(0),
               mib(mem_info().free_b));
        if (pool.va != base_at_start || pool.ptr(0) != chunk0_at_start) base_stable = false;
    };
    auto shrink_to = [&](size_t n, const char* label) {
        for (size_t i = pool.handles.size(); i-- > n;) pool.decommit(i);
        printf("   %-22s committed=%5.0f MiB  base=0x%llx  chunk0=%p  free=%.1f MiB\n", label,
               mib(pool.committed_bytes()), (unsigned long long)pool.va, (void*)pool.ptr(0),
               mib(mem_info().free_b));
        if (pool.va != base_at_start || pool.ptr(0) != chunk0_at_start) base_stable = false;
    };

    grow_to(8, "grow  -> 8 chunks");
    shrink_to(2, "shrink-> 2 chunks");
    grow_to(6, "grow  -> 6 chunks");
    shrink_to(4, "shrink-> 4 chunks");
    grow_to(10, "grow  -> 10 chunks");
    check(base_stable, "base VA and chunk0 pointer identical across the whole grow/shrink cycle");
    // chunk0 was never decommitted; its data must have survived the whole cycle.
    check(verify_chunk(0, 0xA5A5u, "after grow/shrink"), "chunk 0 data survived the grow/shrink cycle");
    printf("\n");

    // -----------------------------------------------------------------------
    // [4] decommit+recommit of a DIFFERENT region leaves the first intact
    // -----------------------------------------------------------------------
    printf("== [4] decommit+recommit elsewhere does not disturb live data ==\n");
    // Two witnesses, one below and one above the region we churn.
    fill_pattern<<<1024, 256>>>(pool.ptr(1), kElemsPerChunk, 0x1234u);
    fill_pattern<<<1024, 256>>>(pool.ptr(9), kElemsPerChunk, 0xBEEFu);
    CUDA_TRY(cudaDeviceSynchronize());
    check(verify_chunk(1, 0x1234u, "witness pre"), "witness chunk 1 written");
    check(verify_chunk(9, 0xBEEFu, "witness pre"), "witness chunk 9 written");

    printf("   churning chunks 4..6: decommit, dirty-fill a fresh commit, decommit, recommit\n");
    for (int rep = 0; rep < 3; ++rep) {
        for (size_t i = 4; i <= 6; ++i) pool.decommit(i);
        for (size_t i = 4; i <= 6; ++i) pool.commit(i);
        // Write garbage into the recycled physical pages so a stale mapping shows up.
        for (size_t i = 4; i <= 6; ++i)
            fill_pattern<<<1024, 256>>>(pool.ptr(i), kElemsPerChunk, 0xDEAD0000u + rep);
        CUDA_TRY(cudaDeviceSynchronize());
    }
    check(verify_chunk(1, 0x1234u, "witness post"), "chunk 1 intact after 3 churn rounds on chunks 4-6");
    check(verify_chunk(9, 0xBEEFu, "witness post"), "chunk 9 intact after 3 churn rounds on chunks 4-6");
    check(verify_chunk(0, 0xA5A5u, "witness post"), "chunk 0 intact after 3 churn rounds on chunks 4-6");
    printf("\n");

    // -----------------------------------------------------------------------
    // [7] CUDA graph captured against a fixed VA, replayed after growth
    // -----------------------------------------------------------------------
    printf("== [7] graph-captured kernel still correct after the pool grows ==\n");
    {
        unsigned long long* d_sum = nullptr;
        CUDA_TRY(cudaMalloc(&d_sum, sizeof(unsigned long long)));
        cudaStream_t s;
        CUDA_TRY(cudaStreamCreate(&s));

        // Capture a kernel reading chunk 1 (the witness) at its fixed VA.
        const size_t kProbeElems = 1 << 20;  // 4 MiB slice, enough to be a real read
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t exec = nullptr;
        CUDA_TRY(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
        cudaMemsetAsync(d_sum, 0, sizeof(unsigned long long), s);
        sum_stride<<<256, 256, 0, s>>>(pool.ptr(1), kProbeElems, d_sum);
        CUDA_TRY(cudaStreamEndCapture(s, &graph));
        CUDA_TRY(cudaGraphInstantiate(&exec, graph, 0));

        unsigned long long ref_expect = 0;
        for (size_t i = 0; i < kProbeElems; ++i) ref_expect += (uint32_t)((i * 2654435761u) ^ (0x1234u * 0x9E3779B9u));

        unsigned long long got0 = 0;
        CUDA_TRY(cudaGraphLaunch(exec, s));
        CUDA_TRY(cudaStreamSynchronize(s));
        CUDA_TRY(cudaMemcpy(&got0, d_sum, sizeof(got0), cudaMemcpyDeviceToHost));
        printf("   graph replay before growth: sum=%llu expect=%llu\n", got0, ref_expect);
        check(got0 == ref_expect, "graph replay correct before growth");

        // Grow the pool by 6 more chunks (1.5 GiB) *while the graph exec is live*.
        for (size_t i = 10; i < 16; ++i) pool.commit(i);
        printf("   grew to %.0f MiB committed; base still 0x%llx\n", mib(pool.committed_bytes()),
               (unsigned long long)pool.va);

        unsigned long long got1 = 0;
        CUDA_TRY(cudaGraphLaunch(exec, s));
        CUDA_TRY(cudaStreamSynchronize(s));
        CUDA_TRY(cudaMemcpy(&got1, d_sum, sizeof(got1), cudaMemcpyDeviceToHost));
        printf("   graph replay after  growth: sum=%llu expect=%llu\n", got1, ref_expect);
        check(got1 == ref_expect, "graph replay correct after 1.5 GiB of growth (no re-instantiate)");

        // And after a shrink of an unrelated region.
        for (size_t i = 12; i < 16; ++i) pool.decommit(i);
        unsigned long long got2 = 0;
        CUDA_TRY(cudaGraphLaunch(exec, s));
        CUDA_TRY(cudaStreamSynchronize(s));
        CUDA_TRY(cudaMemcpy(&got2, d_sum, sizeof(got2), cudaMemcpyDeviceToHost));
        printf("   graph replay after  shrink: sum=%llu expect=%llu\n", got2, ref_expect);
        check(got2 == ref_expect, "graph replay correct after decommit of an unrelated region");

        cudaGraphExecDestroy(exec);
        cudaGraphDestroy(graph);
        cudaStreamDestroy(s);
        cudaFree(d_sum);
    }
    printf("\n");

    // -----------------------------------------------------------------------
    // [6] is decommitted memory actually returned to the OS?
    //     Split the two halves: unmap alone vs unmap + release.
    // -----------------------------------------------------------------------
    printf("== [6] decommit returns physical memory (unmap vs release) ==\n");
    {
        // Fresh single chunk at a free slot so this measurement is isolated.
        const size_t idx = 20;
        size_t f0 = mem_info().free_b;
        pool.commit(idx);
        size_t f1 = mem_info().free_b;
        printf("   commit                  : used %+7.2f MiB  (free %.1f -> %.1f MiB)\n",
               used_delta_mib(f0, f1), mib(f0), mib(f1));

        CUmemGenericAllocationHandle h = pool.handles[idx];
        CU_TRY(cuMemUnmap(pool.va + idx * kChunk, kChunk));
        size_t f2 = mem_info().free_b;
        printf("   cuMemUnmap only         : used %+7.2f MiB  (handle still alive)\n", used_delta_mib(f1, f2));

        CU_TRY(cuMemRelease(h));
        pool.handles[idx] = 0;
        size_t f3 = mem_info().free_b;
        printf("   cuMemRelease            : used %+7.2f MiB\n", used_delta_mib(f2, f3));
        printf("   net vs pre-commit       : used %+7.2f MiB  (expected ~0)\n", used_delta_mib(f0, f3));

        check(used_delta_mib(f1, f2) > -1.0 && used_delta_mib(f1, f2) < 1.0,
              "cuMemUnmap alone does NOT return memory (handle still holds it)");
        check(used_delta_mib(f2, f3) < -(mib(kChunk) * 0.95),
              "cuMemRelease returns the full chunk to the driver");
        check(used_delta_mib(f0, f3) > -1.0 && used_delta_mib(f0, f3) < 1.0,
              "commit -> decommit is byte-neutral against cudaMemGetInfo");
    }

    // Full teardown: everything back to the post-reserve baseline?
    for (size_t i = 0; i < pool.handles.size(); ++i) pool.decommit(i);
    size_t free_after_full_decommit = mem_info().free_b;
    printf("   after decommitting ALL chunks: free = %.2f MiB (post-reserve baseline %.2f MiB, delta %+.2f MiB)\n",
           mib(free_after_full_decommit), mib(free_post_reserve),
           used_delta_mib(free_post_reserve, free_after_full_decommit));
    check(used_delta_mib(free_post_reserve, free_after_full_decommit) < 2.0,
          "full decommit returns free VRAM to the post-reserve baseline (< 2 MiB residual)");
    printf("\n");

    // -----------------------------------------------------------------------
    // [8] latency of commit / decommit — WDDM tax check
    // -----------------------------------------------------------------------
    printf("== [8] per-op latency (WDDM tax check) ==\n");
    {
        const int kReps = 16;
        double c_sum = 0, c_max = 0, d_sum = 0, d_max = 0;
        for (int r = 0; r < kReps; ++r) {
            double c = pool.commit(0);
            if (c > c_max) c_max = c;
            c_sum += c;
            double d = pool.decommit(0);
            if (d > d_max) d_max = d;
            d_sum += d;
        }
        printf("   commit   %.0f MiB: mean %.3f ms  max %.3f ms  (%d reps)\n", mib(kChunk), c_sum / kReps, c_max, kReps);
        printf("   decommit %.0f MiB: mean %.3f ms  max %.3f ms  (%d reps)\n", mib(kChunk), d_sum / kReps, d_max, kReps);
        printf("   note: a KV pool grows a chunk at most once per few hundred decode steps;\n"
               "         budget against ~%.1f ms of stall per growth event.\n",
               c_sum / kReps);
    }
    printf("\n");

    // -----------------------------------------------------------------------
    // teardown
    // -----------------------------------------------------------------------
    CU_TRY(cuMemAddressFree(pool.va, kReserve));
    size_t free_end = mem_info().free_b;
    printf("== teardown ==\n");
    printf("   free at start %.2f MiB, free at end %.2f MiB, leaked %+.2f MiB\n\n", mib(base.free_b),
           mib(free_end), used_delta_mib(base.free_b, free_end));

    cudaFree(d_bad);

    printf("=== RESULT: %d checks, %d failures -> %s ===\n", g_check, g_fail, g_fail == 0 ? "GO" : "NO-GO");
    return g_fail;
}
