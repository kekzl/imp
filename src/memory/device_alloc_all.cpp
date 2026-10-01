#include "memory/device_alloc_all.h"

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

}  // namespace imp
