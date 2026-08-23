// The growable backend: CUDA virtual memory management (A3.1, A7 step 7).
//
// Reserve address space for what the pool could ever need, commit physical
// pages into it as demand arrives, and release them when it goes away. The
// point is not fragmentation, it is that a pool built this way cannot be
// mis-sized, because it is no longer sized.
//
// Two properties make it usable here, both re-measured on this box against
// CUDA 13.3 with tools/analysis/vmm_wsl2_probe.cu before this file was written:
//
//   The base address is invariant across a grow/shrink cycle, so a CUDA graph
//   captured against a pointer into the region still reads correct data after
//   1.5 GiB was committed underneath it, with no re-instantiation. Graphs are
//   worth 2-3x of decode here, so a backend that invalidated them would be
//   unusable whatever else it offered.
//
//   cuMemRelease actually returns memory to the driver: after decommitting
//   everything, free VRAM is back at the post-reserve baseline to within 2 MiB.
//   That is the part cudaFree does NOT do on WSL2/WDDM, where a process keeps
//   its peak commitment for its lifetime. It is why shrinking is worth having
//   at all on this platform, and it is the only mechanism that lets two imp
//   processes share one card without one of them being restarted.
//
// Measured cost on the same run: 1.18 ms per 256 MiB commit, 2.58 ms per
// decommit. A KV pool grows at most once every few hundred decode steps, so the
// stall is budgeted against a growth event, not against a step.
//
// cuMemUnmap alone frees nothing — the allocation handle still holds the
// memory. Both calls are needed, in that order, which is why every mapping is
// tracked with its handle rather than just its address range.

#include "memory/backend.h"
#include "memory/vram_query.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <dlfcn.h>

#include <algorithm>
#include <map>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace imp {
namespace {

// The driver API is loaded by hand rather than linked.
//
// Linking it would put a hard DT_NEEDED on libcuda.so.1 into the imp library,
// and every CPU-only test binary would then fail to START in a container
// without a GPU — which is exactly what CI runs. The symbols were absent from
// the binary before this file existed only because nothing referenced them and
// --as-needed dropped the entry.
//
// A missing library is therefore the same answer as a device without virtual
// memory management: no growable backend, everything else unchanged.
struct DriverApi {
    CUresult (*MemAddressReserve)(CUdeviceptr*, size_t, size_t, CUdeviceptr, unsigned long long) = nullptr;
    CUresult (*MemAddressFree)(CUdeviceptr, size_t) = nullptr;
    CUresult (*MemCreate)(CUmemGenericAllocationHandle*, size_t, const CUmemAllocationProp*,
                          unsigned long long) = nullptr;
    CUresult (*MemMap)(CUdeviceptr, size_t, size_t, CUmemGenericAllocationHandle,
                       unsigned long long) = nullptr;
    CUresult (*MemSetAccess)(CUdeviceptr, size_t, const CUmemAccessDesc*, size_t) = nullptr;
    CUresult (*MemUnmap)(CUdeviceptr, size_t) = nullptr;
    CUresult (*MemRelease)(CUmemGenericAllocationHandle) = nullptr;
    CUresult (*MemGetAllocationGranularity)(size_t*, const CUmemAllocationProp*,
                                            CUmemAllocationGranularity_flags) = nullptr;
    CUresult (*DeviceGetAttribute)(int*, CUdevice_attribute, CUdevice) = nullptr;
    bool ok = false;
};

const DriverApi& driver() {
    static const DriverApi api = [] {
        DriverApi a;
        void* lib = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
        if (lib == nullptr)
            lib = dlopen("libcuda.so", RTLD_LAZY | RTLD_LOCAL);
        if (lib == nullptr)
            return a;
        auto sym = [lib](const char* name) { return dlsym(lib, name); };
        a.MemAddressReserve = reinterpret_cast<decltype(a.MemAddressReserve)>(sym("cuMemAddressReserve"));
        a.MemAddressFree = reinterpret_cast<decltype(a.MemAddressFree)>(sym("cuMemAddressFree"));
        a.MemCreate = reinterpret_cast<decltype(a.MemCreate)>(sym("cuMemCreate"));
        a.MemMap = reinterpret_cast<decltype(a.MemMap)>(sym("cuMemMap"));
        a.MemSetAccess = reinterpret_cast<decltype(a.MemSetAccess)>(sym("cuMemSetAccess"));
        a.MemUnmap = reinterpret_cast<decltype(a.MemUnmap)>(sym("cuMemUnmap"));
        a.MemRelease = reinterpret_cast<decltype(a.MemRelease)>(sym("cuMemRelease"));
        a.MemGetAllocationGranularity = reinterpret_cast<decltype(a.MemGetAllocationGranularity)>(
            sym("cuMemGetAllocationGranularity"));
        a.DeviceGetAttribute = reinterpret_cast<decltype(a.DeviceGetAttribute)>(sym("cuDeviceGetAttribute"));
        a.ok = a.MemAddressReserve && a.MemAddressFree && a.MemCreate && a.MemMap && a.MemSetAccess &&
               a.MemUnmap && a.MemRelease && a.MemGetAllocationGranularity && a.DeviceGetAttribute;
        return a;
    }();
    return api;
}

// One committed chunk: the driver hands back a handle, and the handle is what
// has to be released. Keeping the address alone would leak the memory while
// looking like it had been freed.
struct Chunk {
    CUmemGenericAllocationHandle handle = 0;
    size_t offset = 0;  // from the reservation base
    size_t size = 0;
};

struct Reservation {
    CUdeviceptr va = 0;
    size_t reserved = 0;
    size_t committed = 0;
    // Keyed by offset, because what gets committed is a set of interior ranges
    // rather than one growing prefix. Ordered, so a gap is found by looking at
    // one neighbour instead of scanning.
    std::map<size_t, Chunk> chunks;
};

size_t round_up(size_t v, size_t to) { return to == 0 ? v : ((v + to - 1) / to) * to; }

class VmmBackend final : public Backend {
public:
    // Whether this device can do any of it. Checked once: a device without VMM
    // support is a device where every call below would fail one at a time.
    static bool supported() {
        static const bool ok = probe_support_();
        return ok;
    }

    MemError do_commit_range(Region& region, size_t offset, size_t bytes) override {
        if (bytes == 0)
            return MemError::Ok;
        std::lock_guard<std::mutex> lock(mu_);
        auto it = regions_.find(region.base());
        if (it == regions_.end())
            return MemError::InvalidArgument;
        Reservation& r = it->second;
        const size_t begin = (offset / granularity_) * granularity_;
        const size_t end = std::min(round_up(offset + bytes, granularity_), r.reserved);
        if (begin >= end)
            return MemError::InvalidArgument;
        const MemError e = map_range_(r, begin, end);
        set_committed_(region, r.committed);
        return e;
    }

    MemError decommit_range(Region& region, size_t offset, size_t bytes) override {
        if (bytes == 0)
            return MemError::Ok;
        std::lock_guard<std::mutex> lock(mu_);
        auto it = regions_.find(region.base());
        if (it == regions_.end())
            return MemError::InvalidArgument;
        Reservation& r = it->second;
        // Round IN when releasing: a partially covered granule stays mapped,
        // because the rest of it may be somebody else's data. Rounding out here
        // would unmap memory the caller never asked to give back.
        const size_t begin = round_up(offset, granularity_);
        const size_t end = ((offset + bytes) / granularity_) * granularity_;
        unmap_range_(r, begin, end);
        set_committed_(region, r.committed);
        return MemError::Ok;
    }

    MemError do_commit(Region& region, size_t new_committed) override {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = regions_.find(region.base());
        if (it == regions_.end())
            return MemError::InvalidArgument;
        Reservation& r = it->second;
        const size_t target = round_up(std::min(new_committed, r.reserved), granularity_);
        if (target == r.committed) {
            set_committed_(region, r.committed);
            return MemError::Ok;
        }
        const MemError e = target > r.committed ? grow_(r, target) : shrink_(r, target);
        // Publish what was actually achieved even on a partial failure: the
        // caller has to know how much of the region it may touch, and a growth
        // that got halfway is still half a pool, not none.
        set_committed_(region, r.committed);
        return e;
    }

    BackendStats stats() const override {
        std::lock_guard<std::mutex> lock(mu_);
        BackendStats s = stats_;
        s.capacity = capacity();
        return s;
    }

    size_t capacity() const override { return vram_budget_bytes(); }

protected:
    // A fixed acquisition is a growable one that never grows. Same code path,
    // so the two cannot drift apart in how they map, align or account.
    MemError do_acquire(size_t bytes, size_t alignment, RegionTag tag, void** out_base,
                        size_t* out_reserved) override {
        const MemError e = do_acquire_growable(bytes, bytes, alignment, tag, out_base);
        if (e != MemError::Ok)
            return e;
        std::lock_guard<std::mutex> lock(mu_);
        *out_reserved = regions_[*out_base].reserved;
        return MemError::Ok;
    }

    MemError do_acquire_growable(size_t reserve_bytes, size_t initial_commit, size_t alignment,
                                 RegionTag /*tag*/, void** out_base) override {
        if (!supported())
            return MemError::NotGrowable;
        if (reserve_bytes == 0)
            return MemError::InvalidArgument;

        std::lock_guard<std::mutex> lock(mu_);
        const size_t align = std::max(alignment, granularity_);
        const size_t reserved = round_up(reserve_bytes, granularity_);

        // The budget caps COMMITTED bytes, not reserved ones. Reserving address
        // space costs no physical memory, and charging for it would make the
        // reservation-for-the-ceiling this backend exists for unaffordable.
        const size_t cap = capacity();
        const size_t want = round_up(std::min(initial_commit, reserved), granularity_);
        if (cap > 0 && stats_.live_bytes + want > cap)
            return MemError::BudgetExceeded;

        CUdeviceptr va = 0;
        if (driver().MemAddressReserve(&va, reserved, align, 0, 0) != CUDA_SUCCESS)
            return MemError::OutOfMemory;

        Reservation r;
        r.va = va;
        r.reserved = reserved;
        if (want > 0) {
            const MemError e = grow_(r, want);
            if (e != MemError::Ok) {
                shrink_(r, 0);
                driver().MemAddressFree(va, reserved);
                return e;
            }
        }
        *out_base = reinterpret_cast<void*>(va);
        stats_.reserved_bytes += reserved;
        stats_.acquire_count++;
        regions_.emplace(*out_base, std::move(r));
        return MemError::Ok;
    }

    void do_release(void* base, size_t /*committed*/, size_t /*reserved*/, RegionTag) override {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = regions_.find(base);
        if (it == regions_.end())
            return;
        Reservation& r = it->second;
        shrink_(r, 0);
        driver().MemAddressFree(r.va, r.reserved);
        stats_.reserved_bytes -= std::min(stats_.reserved_bytes, r.reserved);
        stats_.release_count++;
        regions_.erase(it);
    }

private:
    static bool probe_support_() {
        if (!driver().ok)
            return false;
        int dev = 0;
        if (cudaGetDevice(&dev) != cudaSuccess)
            return false;
        int supported = 0;
        // The driver call needs a context; the runtime has made one by the time
        // any allocation happens, and this is only ever reached from an
        // allocation path.
        if (driver().DeviceGetAttribute(&supported, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
                                        dev) != CUDA_SUCCESS)
            return false;
        return supported != 0;
    }

    CUmemAllocationProp prop_() const {
        CUmemAllocationProp p = {};
        p.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        p.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        p.location.id = device_;
        return p;
    }

    // Map every granule in [begin, end) that is not mapped yet, coalescing
    // each run of missing granules into ONE driver allocation. The measured
    // cost is per call and not per byte (1.18 ms for 256 MiB), so a run that
    // needs 128 granules must not become 128 calls.
    MemError map_range_(Reservation& r, size_t begin, size_t end) {
        const CUmemAllocationProp prop = prop_();
        CUmemAccessDesc access = {};
        access.location = prop.location;
        access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

        size_t off = begin;
        while (off < end) {
            auto next = r.chunks.upper_bound(off);
            if (next != r.chunks.begin()) {
                auto prev = std::prev(next);
                const size_t prev_end = prev->first + prev->second.size;
                if (prev_end > off) {  // already mapped here
                    off = prev_end;
                    continue;
                }
            }
            const size_t gap_end = next == r.chunks.end() ? end : std::min(end, next->first);
            if (gap_end <= off)
                break;

            const size_t size = std::min(gap_end - off, chunk_bytes_);
            const size_t cap = capacity();
            if (cap > 0 && stats_.live_bytes + size > cap)
                return MemError::BudgetExceeded;

            Chunk c;
            c.offset = off;
            c.size = size;
            if (driver().MemCreate(&c.handle, size, &prop, 0) != CUDA_SUCCESS)
                return MemError::OutOfMemory;
            if (driver().MemMap(r.va + c.offset, size, 0, c.handle, 0) != CUDA_SUCCESS) {
                driver().MemRelease(c.handle);
                return MemError::OutOfMemory;
            }
            if (driver().MemSetAccess(r.va + c.offset, size, &access, 1) != CUDA_SUCCESS) {
                driver().MemUnmap(r.va + c.offset, size);
                driver().MemRelease(c.handle);
                // Mapped but unusable is the same outcome as never mapped, and
                // the caller has exactly one recovery either way.
                return MemError::OutOfMemory;
            }
            r.chunks.emplace(c.offset, c);
            r.committed += size;
            stats_.live_bytes += size;
            stats_.peak_bytes = std::max(stats_.peak_bytes, stats_.live_bytes);
            off += size;
        }
        return MemError::Ok;
    }

    // Give back every chunk that lies entirely inside [begin, end). A chunk
    // that straddles the boundary is left mapped: splitting a driver
    // allocation is not possible, and unmapping it would take memory the
    // caller did not offer with it.
    //
    // Both calls are required and in this order: cuMemUnmap removes the
    // mapping, cuMemRelease is what returns the memory to the driver.
    // Unmapping alone measures as freeing nothing.
    void unmap_range_(Reservation& r, size_t begin, size_t end) {
        for (auto it = r.chunks.lower_bound(begin); it != r.chunks.end();) {
            const size_t c_end = it->first + it->second.size;
            if (it->first >= end)
                break;
            if (c_end > end) {
                ++it;
                continue;
            }
            driver().MemUnmap(r.va + it->second.offset, it->second.size);
            driver().MemRelease(it->second.handle);
            r.committed -= std::min(r.committed, it->second.size);
            stats_.live_bytes -= std::min(stats_.live_bytes, it->second.size);
            it = r.chunks.erase(it);
        }
    }

    MemError grow_(Reservation& r, size_t target) { return map_range_(r, 0, target); }

    MemError shrink_(Reservation& r, size_t target) {
        unmap_range_(r, target, r.reserved);
        return MemError::Ok;
    }

    mutable std::mutex mu_;
    BackendStats stats_;
    std::unordered_map<void*, Reservation> regions_;
    int device_ = init_device_();
    size_t granularity_ = init_granularity_();
    // Coarse enough that a growth event is a handful of driver calls rather
    // than hundreds: the measured cost is per call, not per byte.
    size_t chunk_bytes_ = std::max<size_t>(granularity_, size_t(64) << 20);

    static int init_device_() {
        int dev = 0;
        cudaGetDevice(&dev);
        return dev;
    }

    size_t init_granularity_() const {
        CUmemAllocationProp p = {};
        p.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        p.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        p.location.id = device_;
        size_t g = 0;
        if (driver().MemGetAllocationGranularity(&g, &p, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED) !=
                CUDA_SUCCESS ||
            g == 0)
            return size_t(2) << 20;  // the documented sm_120 minimum
        return g;
    }
};

}  // namespace

Backend* vmm_backend() {
    if (!VmmBackend::supported())
        return nullptr;
    static VmmBackend inst;
    return &inst;
}

}  // namespace imp
