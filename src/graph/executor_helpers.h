#pragma once

#include "memory/vram_allocator.h"
#include "core/tensor.h"
#include "core/logging.h"
#include <cuda_runtime.h>
#include <vector>

namespace imp {

// Shared inline helpers used across executor TUs (workspace, pre-dequant, config).

static inline size_t align256(size_t x) { return (x + 255) & ~size_t(255); }

static inline Tensor make_workspace_tensor(char*& ptr, QType dtype,
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

static inline void* vram_alloc_force(VRAMAllocator* alloc, size_t bytes, const char* tag) {
    if (bytes == 0) return nullptr;
    if (alloc) return alloc->allocate(bytes, tag, /*bypass_headroom=*/true);
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

// Set L2 streaming hint for weight data: deprioritize in L2 eviction so weights
// don't pollute cache for KV data or activations reused across layers.
//
// The window size is clamped to the device's accessPolicyMaxWindowSize
// (on RTX 5090 / Blackwell consumer: 128 MiB; B200: 512 MiB). Without clamping,
// GDN models whose (w_gate,w_up,w_down) address span exceeds 128 MiB — e.g.
// Qwen3.5-9B Q8_0 spans ~200 MiB — cause cudaStreamSetAttribute to return
// cudaErrorInvalidValue, which poisons the stream's error state and garbles
// every subsequent kernel.
static inline void set_l2_streaming(cudaStream_t stream, const void* ptr, size_t bytes) {
    if (!ptr || bytes == 0 || !stream) return;
    static size_t s_max_bytes = 0;
    if (s_max_bytes == 0) {
        int max_w = 0;
        if (cudaDeviceGetAttribute(&max_w, cudaDevAttrMaxAccessPolicyWindowSize, 0)
            == cudaSuccess && max_w > 0) {
            s_max_bytes = static_cast<size_t>(max_w);
        } else {
            s_max_bytes = 128ULL * 1024 * 1024;  // conservative fallback
        }
    }
    if (bytes > s_max_bytes) bytes = s_max_bytes;
    cudaStreamAttrValue attr = {};
    attr.accessPolicyWindow.base_ptr = const_cast<void*>(ptr);
    attr.accessPolicyWindow.num_bytes = bytes;
    attr.accessPolicyWindow.hitRatio = 0.0f;
    attr.accessPolicyWindow.hitProp = cudaAccessPropertyStreaming;
    attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
    cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
}

static inline void clear_l2_policy(cudaStream_t stream) {
    if (!stream) return;
    cudaStreamAttrValue attr = {};
    attr.accessPolicyWindow.num_bytes = 0;
    cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
}

} // namespace imp
