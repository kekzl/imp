// cudaMallocAsync failure injection (#2446): the device's current async mempool becomes a capped
// pool held full, so every plain cudaMallocAsync on the device returns cudaErrorMemoryAllocation.
// Pools set by cudaMallocFromPoolAsync callers are unaffected. Restore with cudaDeviceSetMemPool.
#ifndef IMP_TESTS_TEST_ALLOC_INJECT_H
#define IMP_TESTS_TEST_ALLOC_INJECT_H

#include <cuda_runtime.h>
#include <cstddef>

namespace imp_test {

// Returns the previous pool, or nullptr when the injection could not be armed (probe still succeeds).
inline cudaMemPool_t exhaust_async_pool() {
    constexpr size_t kCapBytes = 2u << 20;  // requested cap; the driver rounds up (32 MiB on sm_120)
    int dev = 0;
    cudaMemPool_t old_pool = nullptr, capped = nullptr;
    if (cudaGetDevice(&dev) != cudaSuccess || cudaDeviceGetMemPool(&old_pool, dev) != cudaSuccess)
        return nullptr;
    cudaMemPoolProps props{};
    props.allocType = cudaMemAllocationTypePinned;
    props.location.type = cudaMemLocationTypeDevice;
    props.location.id = dev;
    props.maxSize = kCapBytes;
    if (cudaMemPoolCreate(&capped, &props) != cudaSuccess)
        return nullptr;
    // Fill with halving sizes down to 1 B so even sub-allocations fail.
    for (size_t sz = kCapBytes; sz >= 1; sz /= 2) {
        void* p = nullptr;
        while (cudaMallocFromPoolAsync(&p, sz, capped, nullptr) == cudaSuccess) {}
        (void)cudaGetLastError();
    }
    if (cudaStreamSynchronize(nullptr) != cudaSuccess || cudaDeviceSetMemPool(dev, capped) != cudaSuccess)
        return nullptr;
    // Control arm: armed only if a 256 B async alloc now fails.
    void* probe = nullptr;
    const bool armed = cudaMallocAsync(&probe, 256, nullptr) != cudaSuccess;
    (void)cudaGetLastError();
    return armed ? old_pool : nullptr;
}

}  // namespace imp_test

#endif  // IMP_TESTS_TEST_ALLOC_INJECT_H
