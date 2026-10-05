#include "memory/host_pinned.h"

#include "core/logging.h"

#include <cuda_runtime.h>

#include <cstdlib>

namespace imp {

namespace {

class CudaHostPinnedAllocator final : public HostPinnedAllocator {
public:
    bool alloc(size_t bytes, HostPinnedKind kind, void** out_host, void** out_device) override {
        if (out_device)
            *out_device = nullptr;
        if (!out_host || bytes == 0)
            return false;
        void* hp = nullptr;
        const unsigned flags = (kind == HostPinnedKind::Mapped) ? cudaHostAllocMapped : cudaHostAllocDefault;
        const cudaError_t herr = cudaHostAlloc(&hp, bytes, flags);
        if (herr != cudaSuccess) {
            IMP_LOG_WARN("pinned host alloc failed (%zu B, %s): %s", bytes,
                         kind == HostPinnedKind::Mapped ? "mapped" : "plain", cudaGetErrorString(herr));
            return false;
        }
        if (kind == HostPinnedKind::Mapped) {
            void* hdev = nullptr;
            const cudaError_t derr = cudaHostGetDevicePointer(&hdev, hp, 0);
            if (derr != cudaSuccess) {
                IMP_LOG_WARN("pinned host device pointer failed: %s", cudaGetErrorString(derr));
                IMP_CUDA_CHECK_LOG(cudaFreeHost(hp));
                return false;
            }
            if (out_device)
                *out_device = hdev;
        }
        *out_host = hp;
        return true;
    }

    void free(void* host) override {
        if (host)
            IMP_CUDA_CHECK_LOG(cudaFreeHost(host));
    }
};

// Owner of acquire_filled buffers: aligned_alloc'd pages pinned in place, so free = unregister + free.
class RegisteredHostAllocator final : public HostPinnedAllocator {
public:
    bool alloc(size_t, HostPinnedKind, void** out_host, void** out_device) override {
        if (out_host)
            *out_host = nullptr;
        if (out_device)
            *out_device = nullptr;
        return false;  // acquire_filled only: pages must be written before they are registered
    }
    void free(void* host) override {
        if (!host)
            return;
        IMP_CUDA_CHECK_LOG(cudaHostUnregister(host));
        std::free(host);
    }
};

RegisteredHostAllocator& registered_host_allocator() {
    static RegisteredHostAllocator a;
    return a;
}

}  // namespace

HostPinnedAllocator& cuda_host_pinned_allocator() {
    static CudaHostPinnedAllocator a;
    return a;
}

PinnedBuffer PinnedBuffer::acquire_filled(size_t bytes, HostPinnedKind kind,
                                          const std::function<void(void*)>& fill) {
    PinnedBuffer out;
    if (bytes == 0)
        return out;
    constexpr size_t kPage = 4096;
    void* host = std::aligned_alloc(kPage, (bytes + kPage - 1) / kPage * kPage);
    if (!host)
        return out;
    fill(host);
    const unsigned flags = kind == HostPinnedKind::Mapped ? cudaHostRegisterMapped : cudaHostRegisterDefault;
    if (const cudaError_t err = cudaHostRegister(host, bytes, flags); err != cudaSuccess) {
        IMP_LOG_WARN("cudaHostRegister of a filled %.1f MiB buffer failed: %s", bytes / (1024.0 * 1024.0),
                     cudaGetErrorString(err));
        std::free(host);
        return out;
    }
    void* device = nullptr;
    if (kind == HostPinnedKind::Mapped && cudaHostGetDevicePointer(&device, host, 0) != cudaSuccess) {
        IMP_CUDA_CHECK_LOG(cudaHostUnregister(host));
        std::free(host);
        return out;
    }
    out.owner_ = &registered_host_allocator();
    out.host_ = host;
    out.device_ = device;
    out.bytes_ = bytes;
    return out;
}

PinnedBuffer PinnedBuffer::acquire(HostPinnedAllocator& alloc, size_t bytes, HostPinnedKind kind) {
    PinnedBuffer out;
    if (bytes == 0)
        return out;  // a zero-size request is a caller bug, not an allocation
    void* host = nullptr;
    void* device = nullptr;
    if (!alloc.alloc(bytes, kind, &host, &device))
        return out;  // empty — the failure value, not an exception
    out.owner_ = &alloc;
    out.host_ = host;
    out.device_ = device;
    out.bytes_ = bytes;
    return out;
}

void PinnedBuffer::reset() {
    if (owner_ && host_)
        owner_->free(host_);
    owner_ = nullptr;
    host_ = nullptr;
    device_ = nullptr;
    bytes_ = 0;
}

namespace {

class CudaHostRegistrar final : public HostRegistrar {
public:
    bool register_read_only(void* ptr, size_t bytes) override {
        const cudaError_t err = cudaHostRegister(ptr, bytes, cudaHostRegisterReadOnly);
        if (err != cudaSuccess) {
            IMP_LOG_WARN("cudaHostRegister failed (%.2f MiB): %s — H2D will be slower",
                         bytes / (1024.0 * 1024.0), cudaGetErrorString(err));
            return false;
        }
        return true;
    }
    void unregister(void* ptr) override {
        if (ptr)
            (void)cudaHostUnregister(ptr);
    }
};

}  // namespace

HostRegistrar& cuda_host_registrar() {
    static CudaHostRegistrar r;
    return r;
}

HostRegistration HostRegistration::acquire_read_only(void* ptr, size_t bytes, HostRegistrar& reg) {
    HostRegistration out;
    if (!ptr || bytes == 0)
        return out;
    if (!reg.register_read_only(ptr, bytes))
        return out;
    out.reg_ = &reg;
    out.ptr_ = ptr;
    out.bytes_ = bytes;
    return out;
}

void HostRegistration::reset() {
    if (reg_ && ptr_)
        reg_->unregister(ptr_);
    reg_ = nullptr;
    ptr_ = nullptr;
    bytes_ = 0;
}

void PinnedBuffer::steal_(PinnedBuffer&& o) noexcept {
    owner_ = o.owner_;
    host_ = o.host_;
    device_ = o.device_;
    bytes_ = o.bytes_;
    o.owner_ = nullptr;
    o.host_ = nullptr;
    o.device_ = nullptr;
    o.bytes_ = 0;
}

}  // namespace imp
