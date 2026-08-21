// Link-time interposition on the CUDA allocation symbols
// (docs/internals/MEMORY.md A6, AUDIT B8/B26).
//
// Why this exists. Acceptance criterion 3 — "an instrumented soak shows zero
// driver allocations after warmup" — needs a detector that sees EVERY device
// allocation, not just the ones that were polite enough to route through
// Backend. Three layers already exist and each has a blind spot:
//
//   * the allocation-phase guard sees only Backend traffic;
//   * the default mempool's UsedMemHigh sees every cudaMallocAsync but no
//     plain cudaMalloc;
//   * the graph mempool's UsedMemHigh sees capture-region allocations, and
//     measured zero (B26).
//
// What is left is plain cudaMalloc/cudaFree, which is exactly what the
// remaining per-request sites use (engine_graph_decode.cpp:308/316/341). This
// file closes that hole without touching a single call site: the linker
// redirects imp's references to __wrap_*, we record and forward to __real_*.
//
// Calls made INSIDE libcudart/libcublas/CUTLASS are not redirected — their
// references were resolved when those libraries were linked. That is the
// desired behaviour, not a limitation: the ~3.9 GiB library reserve (A1.5)
// stays out of the counter and is charged separately by the planner.
//
// Built only when IMP_ALLOC_INTERPOSE=ON, because --wrap has to be on the
// final executable link and we do not want it in shipping binaries.

#include "memory/backend.h"
#include "core/logging.h"

#include <cuda_runtime_api.h>

#include <dlfcn.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <mutex>
#include <vector>

extern "C" {
cudaError_t __real_cudaMalloc(void** devPtr, size_t size);
cudaError_t __real_cudaFree(void* devPtr);
cudaError_t __real_cudaMallocAsync(void** devPtr, size_t size, cudaStream_t stream);
cudaError_t __real_cudaFreeAsync(void* devPtr, cudaStream_t stream);
cudaError_t __real_cudaMallocHost(void** ptr, size_t size);
cudaError_t __real_cudaHostAlloc(void** ptr, size_t size, unsigned int flags);
}

namespace {

struct Counter {
    std::atomic<uint64_t> calls{0};
    std::atomic<uint64_t> bytes{0};
};

// Device-side only. Host-pinned allocations are counted separately because
// they do not consume VRAM and therefore are not what I2 is about — but a
// per-request cudaHostAlloc is still a latency defect worth seeing.
Counter g_dev_sync;    // cudaMalloc
Counter g_dev_async;   // cudaMallocAsync
Counter g_host_pinned; // cudaMallocHost / cudaHostAlloc

// Per-call-site tally. A bare count is not actionable — "444 allocations while
// serving" does not say which three lines to fix. dladdr() gives the module
// base, so the printed offset feeds straight into
//   addr2line -e <binary> <offset>
// Bounded table; the tail is aggregated rather than dropped silently.
struct Site {
    const void* ret = nullptr;
    uint64_t calls = 0;
    uint64_t bytes = 0;
};
constexpr size_t kMaxSites = 32;
std::mutex g_site_mu;
std::vector<Site> g_sites;
uint64_t g_sites_overflow_calls = 0;

void tally_site(const void* ret, size_t bytes) {
    std::lock_guard<std::mutex> lock(g_site_mu);
    for (auto& s : g_sites) {
        if (s.ret == ret) {
            s.calls++;
            s.bytes += bytes;
            return;
        }
    }
    if (g_sites.size() < kMaxSites) {
        g_sites.push_back(Site{ret, 1, bytes});
        return;
    }
    g_sites_overflow_calls++;
}

void print_sites() {
    std::lock_guard<std::mutex> lock(g_site_mu);
    if (g_sites.empty())
        return;
    std::sort(g_sites.begin(), g_sites.end(),
              [](const Site& a, const Site& b) { return a.calls > b.calls; });
    IMP_LOG_DEBUG("    by call site (addr2line -e <binary> <offset>):");
    for (const auto& s : g_sites) {
        Dl_info info{};
        const char* obj = "?";
        unsigned long long off = 0;
        if (dladdr(s.ret, &info) && info.dli_fbase) {
            obj = info.dli_fname ? info.dli_fname : "?";
            off = (unsigned long long)((const char*)s.ret - (const char*)info.dli_fbase);
        }
        IMP_LOG_DEBUG("      %8llu calls  %9.3f MiB   %s +0x%llx", (unsigned long long)s.calls,
                      s.bytes / (1024.0 * 1024.0), obj, off);
    }
    if (g_sites_overflow_calls)
        IMP_LOG_DEBUG("      %8llu calls  (further sites, table full)",
                      (unsigned long long)g_sites_overflow_calls);
}

void record(Counter& c, size_t bytes, imp::RegionTag tag, const void* site, bool device) {
    if (imp::alloc_phase() != imp::AllocPhase::Serving)
        return;
    c.calls.fetch_add(1, std::memory_order_relaxed);
    c.bytes.fetch_add(bytes, std::memory_order_relaxed);
    tally_site(site, bytes);
    if (device)
        imp::note_serving_allocation(tag, bytes, site);
}

// Report at process exit. A static destructor is enough: this target exists to
// be run once under a soak and read afterwards.
struct Reporter {
    ~Reporter() {
        const uint64_t ds = g_dev_sync.calls.load(), da = g_dev_async.calls.load(),
                       hp = g_host_pinned.calls.load();
        if ((ds | da | hp) == 0) {
            // The clean line is INFO for the same reason the violation is WARN:
            // the gate asserts that one of the two appears at all. Absent both,
            // the binary was built without -DIMP_ALLOC_INTERPOSE=ON and a grep
            // for violations passes for the wrong reason.
            IMP_LOG_INFO(
                "[alloc-interpose] steady state clean: 0 cudaMalloc, "
                "0 cudaMallocAsync, 0 pinned-host allocations while serving");
            return;
        }
        // WARN, not DEBUG. A detected invariant violation that only appears
        // when someone remembers to raise the log level is a finding nobody
        // finds: this is the line `make check-alloc-interpose` fails on.
        IMP_LOG_WARN(
            // The newline after the banner is load-bearing: without it the
            // first class is glued to the banner line, and any reader anchored
            // at the start of a line silently skips it. That is how
            // check_alloc_interpose.sh first reported 2 allocations when there
            // were 19.
            "[alloc-interpose] I2 VIOLATIONS while serving:\n"
            "    cudaMalloc       %8llu calls  %10.2f MiB\n"
            "    cudaMallocAsync  %8llu calls  %10.2f MiB\n"
            "    pinned host      %8llu calls  %10.2f MiB\n",
            (unsigned long long)ds, g_dev_sync.bytes.load() / (1024.0 * 1024.0), (unsigned long long)da,
            g_dev_async.bytes.load() / (1024.0 * 1024.0), (unsigned long long)hp,
            g_host_pinned.bytes.load() / (1024.0 * 1024.0));
        print_sites();
    }
};
Reporter g_reporter;

}  // namespace

extern "C" {

cudaError_t __wrap_cudaMalloc(void** devPtr, size_t size) {
    record(g_dev_sync, size, imp::RegionTag::Other, __builtin_return_address(0), true);
    return __real_cudaMalloc(devPtr, size);
}

cudaError_t __wrap_cudaFree(void* devPtr) { return __real_cudaFree(devPtr); }

cudaError_t __wrap_cudaMallocAsync(void** devPtr, size_t size, cudaStream_t stream) {
    record(g_dev_async, size, imp::RegionTag::Other, __builtin_return_address(0), true);
    return __real_cudaMallocAsync(devPtr, size, stream);
}

cudaError_t __wrap_cudaFreeAsync(void* devPtr, cudaStream_t stream) {
    return __real_cudaFreeAsync(devPtr, stream);
}

cudaError_t __wrap_cudaMallocHost(void** ptr, size_t size) {
    record(g_host_pinned, size, imp::RegionTag::HostStaging, __builtin_return_address(0), false);
    return __real_cudaMallocHost(ptr, size);
}

cudaError_t __wrap_cudaHostAlloc(void** ptr, size_t size, unsigned int flags) {
    record(g_host_pinned, size, imp::RegionTag::HostStaging, __builtin_return_address(0), false);
    return __real_cudaHostAlloc(ptr, size, flags);
}

}  // extern "C"
