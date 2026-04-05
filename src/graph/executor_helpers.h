#pragma once

#include "memory/vram_allocator.h"
#include "core/tensor.h"
#include "core/logging.h"
#include <cuda_runtime.h>
#include <vector>

namespace imp {

// Shared inline helpers used across executor TUs (workspace, pre-dequant, config).

static inline size_t align256(size_t x) { return (x + 255) & ~size_t(255); }

static inline Tensor make_workspace_tensor(char*& ptr, DType dtype,
                                           int64_t rows, int64_t cols, size_t aligned_sz) {
    int64_t shape[2] = {rows, cols};
    Tensor t(ptr, dtype, 2, shape, true);
    ptr += aligned_sz;
    return t;
}

static inline void* vram_alloc(VRAMAllocator* alloc, size_t bytes, const char* tag) {
    if (bytes == 0) return nullptr;
    if (alloc) return alloc->allocate(bytes, tag);
    void* ptr = nullptr;
    if (cudaMalloc(&ptr, bytes) != cudaSuccess) return nullptr;
    return ptr;
}

static inline void vram_free(VRAMAllocator* alloc, void* ptr) {
    if (!ptr) return;
    if (alloc) alloc->free(ptr);
    else IMP_CUDA_CHECK_LOG(cudaFree(ptr));
}

static inline int get_kv_layer(const std::vector<int>& kv_layer_map, int layer) {
    return kv_layer_map.empty() ? layer : kv_layer_map[layer];
}

} // namespace imp
