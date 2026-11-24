#include "memory/device_alloc_all.h"

#include "core/logging.h"

namespace imp {

cudaError_t device_alloc_all_n(DeviceAllocReq* reqs, size_t n, bool async, cudaStream_t stream) {
    for (size_t i = 0; i < n; ++i)
        *reqs[i].ptr = nullptr;
    cudaError_t err = cudaSuccess;
    for (size_t i = 0; i < n && err == cudaSuccess; ++i) {
        if (reqs[i].bytes > 0)
            err = async ? cudaMallocAsync(reqs[i].ptr, reqs[i].bytes, stream)
                        : cudaMalloc(reqs[i].ptr, reqs[i].bytes);
    }
    if (err == cudaSuccess)
        return err;
    (void)cudaGetLastError();
    for (size_t i = 0; i < n; ++i) {
        if (*reqs[i].ptr)
            (void)(async ? cudaFreeAsync(*reqs[i].ptr, stream) : cudaFree(*reqs[i].ptr));
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
    return err;
}

cudaError_t device_free_async_checked(void* ptr, cudaStream_t stream) {
    return ptr ? cudaFreeAsync(ptr, stream) : cudaSuccess;
}

void device_free_all_n(void* const* ptrs, size_t n) {
    for (size_t i = 0; i < n; ++i)
        if (ptrs[i])
            IMP_CUDA_CHECK_LOG(cudaFree(ptrs[i]));
}

void device_free_all_async_n(void* const* ptrs, size_t n, cudaStream_t stream) {
    for (size_t i = 0; i < n; ++i)
        if (ptrs[i])
            IMP_CUDA_CHECK_LOG(cudaFreeAsync(ptrs[i], stream));
}

}  // namespace imp
