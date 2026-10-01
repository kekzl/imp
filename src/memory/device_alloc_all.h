#pragma once
// All-or-none device allocation for call sites that need several buffers before one launch (#2446).
// On a failure the buffers already taken are freed, every pointer is null, the (non-sticky) error is
// cleared and returned. A request of 0 bytes is skipped and leaves its pointer null.
#include <cuda_runtime_api.h>

#include <cstddef>

namespace imp {

struct DeviceAllocReq {
    void** ptr;
    size_t bytes;
};

template <typename T>
[[nodiscard]] inline DeviceAllocReq dev_req(T*& p, size_t bytes) {
    return {reinterpret_cast<void**>(&p), bytes};
}

// async: cudaMallocAsync/cudaFreeAsync on stream; otherwise cudaMalloc/cudaFree.
[[nodiscard]] cudaError_t device_alloc_all_n(DeviceAllocReq* reqs, size_t n, bool async, cudaStream_t stream);

template <typename... R>
[[nodiscard]] cudaError_t device_alloc_all(R... reqs) {
    DeviceAllocReq arr[] = {reqs...};
    return device_alloc_all_n(arr, sizeof...(R), /*async=*/false, nullptr);
}

template <typename... R>
[[nodiscard]] cudaError_t device_alloc_all_async(cudaStream_t stream, R... reqs) {
    DeviceAllocReq arr[] = {reqs...};
    return device_alloc_all_n(arr, sizeof...(R), /*async=*/true, stream);
}

}  // namespace imp
