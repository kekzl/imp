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

// One stream-ordered buffer from a memory pool (pool null: the default pool). The pointer is null
// on failure; the error is returned, not cleared (weights_alloc reports it).
[[nodiscard]] cudaError_t device_alloc_pool_async(void** ptr, size_t bytes, cudaMemPool_t pool,
                                                  cudaStream_t stream);
// Stream-ordered release that hands the driver's result back (teardown counts failures).
cudaError_t device_free_async_checked(void* ptr, cudaStream_t stream);

// Release of buffers taken with device_alloc_all; null pointers are skipped.
void device_free_all_n(void* const* ptrs, size_t n);

template <typename... P>
void device_free_all(P*... ptrs) {
    void* const arr[] = {static_cast<void*>(ptrs)...};
    device_free_all_n(arr, sizeof...(P));
}

// Stream-ordered release of buffers taken with device_alloc_all_async; null pointers are skipped.
void device_free_all_async_n(void* const* ptrs, size_t n, cudaStream_t stream);

template <typename... P>
void device_free_all_async(cudaStream_t stream, P*... ptrs) {
    void* const arr[] = {static_cast<void*>(ptrs)...};
    device_free_all_async_n(arr, sizeof...(P), stream);
}

}  // namespace imp
