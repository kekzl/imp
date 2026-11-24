#include "memory/device_alloc_all.h"

#include "core/logging.h"

#include <atomic>
#include <mutex>
#include <unordered_map>

namespace imp {

namespace {

// Pointer -> requested bytes of every live allocation of this file. Leaked on purpose: a static
// destructor in another TU may still free through here at exit.
struct LiveLedger {
    std::mutex mu;
    std::unordered_map<void*, size_t> bytes_of;
    std::atomic<size_t> live{0};
};

LiveLedger& ledger() {
    static LiveLedger* const l = new LiveLedger;
    return *l;
}

void ledger_add(void* p, size_t bytes) {
    if (p == nullptr)
        return;
    LiveLedger& l = ledger();
    std::lock_guard<std::mutex> lock(l.mu);
    const auto [it, inserted] = l.bytes_of.try_emplace(p, bytes);
    if (!inserted) {  // the driver reused an address whose free bypassed this file
        l.live.fetch_sub(it->second, std::memory_order_relaxed);
        it->second = bytes;
    }
    l.live.fetch_add(bytes, std::memory_order_relaxed);
}

// Called BEFORE the driver free: afterwards another thread may already own the address.
void ledger_remove(void* p) {
    LiveLedger& l = ledger();
    std::lock_guard<std::mutex> lock(l.mu);
    const auto it = l.bytes_of.find(p);
    if (it == l.bytes_of.end())
        return;
    l.live.fetch_sub(it->second, std::memory_order_relaxed);
    l.bytes_of.erase(it);
}

}  // namespace

size_t device_alloc_live_bytes() { return ledger().live.load(std::memory_order_relaxed); }

cudaError_t device_alloc_all_n(DeviceAllocReq* reqs, size_t n, bool async, cudaStream_t stream) {
    for (size_t i = 0; i < n; ++i)
        *reqs[i].ptr = nullptr;
    cudaError_t err = cudaSuccess;
    for (size_t i = 0; i < n && err == cudaSuccess; ++i) {
        if (reqs[i].bytes > 0) {
            err = async ? cudaMallocAsync(reqs[i].ptr, reqs[i].bytes, stream)
                        : cudaMalloc(reqs[i].ptr, reqs[i].bytes);
            if (err == cudaSuccess)
                ledger_add(*reqs[i].ptr, reqs[i].bytes);
        }
    }
    if (err == cudaSuccess)
        return err;
    (void)cudaGetLastError();
    for (size_t i = 0; i < n; ++i) {
        if (*reqs[i].ptr) {
            ledger_remove(*reqs[i].ptr);
            (void)(async ? cudaFreeAsync(*reqs[i].ptr, stream) : cudaFree(*reqs[i].ptr));
        }
        *reqs[i].ptr = nullptr;
    }
    return err;
}

cudaError_t device_alloc_pool_async(void** ptr, size_t bytes, cudaMemPool_t pool, cudaStream_t stream) {
    *ptr = nullptr;
    const cudaError_t err = pool ? cudaMallocFromPoolAsync(ptr, bytes, pool, stream)
                                 : cudaMallocAsync(ptr, bytes, stream);
    if (err != cudaSuccess)
        *ptr = nullptr;
    else
        ledger_add(*ptr, bytes);
    return err;
}

cudaError_t device_free_async_checked(void* ptr, cudaStream_t stream) {
    if (ptr == nullptr)
        return cudaSuccess;
    ledger_remove(ptr);
    return cudaFreeAsync(ptr, stream);
}

void device_free_all_n(void* const* ptrs, size_t n) {
    for (size_t i = 0; i < n; ++i)
        if (ptrs[i]) {
            ledger_remove(ptrs[i]);
            IMP_CUDA_CHECK_LOG(cudaFree(ptrs[i]));
        }
}

void device_free_all_async_n(void* const* ptrs, size_t n, cudaStream_t stream) {
    for (size_t i = 0; i < n; ++i)
        if (ptrs[i]) {
            ledger_remove(ptrs[i]);
            IMP_CUDA_CHECK_LOG(cudaFreeAsync(ptrs[i], stream));
        }
}

}  // namespace imp
