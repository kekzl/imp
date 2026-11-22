#pragma once
// BF16 safetensors tensors -> device fp16 / fp32 through a VRAMAllocator: the loader of the Inkling2
// vision tower and draft model. Every miss sets ok = false and says which tensor.

#include "compute/inkling_ops.h"
#include "core/logging.h"
#include "core/tensor.h"
#include "memory/vram_allocator.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

namespace imp {

inline float bf16_to_float(uint16_t b) {
    const uint32_t u = static_cast<uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &u, sizeof f);
    return f;
}

struct Bf16Loader {
    std::unordered_map<std::string, Tensor> t;
    VRAMAllocator& va;
    std::vector<void*>& owned;
    const char* tag = "bf16_weights";
    bool ok = true;

    const uint16_t* find(const std::string& name, int64_t numel) {
        const auto it = t.find(name);
        if (it == t.end() || it->second.qtype != QType::BF16 || it->second.numel() != numel) {
            IMP_LOG_ERROR("%s: tensor %s missing or not BF16 with %lld elements", tag, name.c_str(),
                          static_cast<long long>(numel));
            ok = false;
            return nullptr;
        }
        return static_cast<const uint16_t*>(it->second.data);
    }
    void* dev(size_t bytes, const char* as = nullptr) {
        void* p = va.allocate(bytes, as ? as : tag);
        if (!p) {
            IMP_LOG_ERROR("%s: device allocation of %.1f MiB failed", tag, bytes / 1048576.0);
            ok = false;
            return nullptr;
        }
        owned.push_back(p);
        return p;
    }
    std::vector<float> host(const std::string& name, int64_t numel) {
        std::vector<float> h(static_cast<size_t>(numel), 0.0f);
        if (const uint16_t* b = find(name, numel))
            for (size_t i = 0; i < h.size(); ++i)
                h[i] = bf16_to_float(b[i]);
        return h;
    }
    float* upload(const std::vector<float>& h) {
        auto* p = static_cast<float*>(dev(h.size() * sizeof(float)));
        // Pageable source: the copy is staged before the call returns.
        if (p && cudaMemcpyAsync(p, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice, nullptr) !=
                     cudaSuccess)
            ok = false;
        return p;
    }
    half* upload_f16(const std::vector<float>& h) {
        std::vector<half> v(h.size());
        for (size_t i = 0; i < h.size(); ++i)
            v[i] = __float2half(h[i]);
        auto* p = static_cast<half*>(dev(v.size() * sizeof(half)));
        if (p && cudaMemcpyAsync(p, v.data(), v.size() * sizeof(half), cudaMemcpyHostToDevice, nullptr) !=
                     cudaSuccess)
            ok = false;
        return p;
    }
    float* f32(const std::string& name, int64_t numel) { return upload(host(name, numel)); }
    // BF16 tensor converted into dst (a slice of a larger allocation).
    void f16_into(const std::string& name, int64_t numel, half* dst) {
        const uint16_t* b = find(name, numel);
        if (!b || !dst)
            return;
        if (cudaMemcpyAsync(dst, b, static_cast<size_t>(numel) * 2, cudaMemcpyHostToDevice, nullptr) !=
            cudaSuccess)
            ok = false;
        inkling::bf16_to_f16(reinterpret_cast<const uint16_t*>(dst), dst, static_cast<size_t>(numel),
                             nullptr);
    }
    half* f16(const std::string& name, int64_t numel) {
        auto* p = static_cast<half*>(dev(static_cast<size_t>(numel) * sizeof(half)));
        f16_into(name, numel, p);
        return p;
    }
};

}  // namespace imp
