#pragma once

// An owning handle for one device allocation whose family is decided at run time. VramOwned
// (memory/vram_owned.h) covers the sites that always hold a VRAMAllocator; the executor's
// buffers do not: its helper takes the VRAMAllocator when there is one and the driver when
// there is none, and other buffers come from device_alloc_all, released synchronously or
// stream-ordered. A raw pointer plus a free that repeats that choice elsewhere is the
// spelling that produced a release list missing six buffers and a cudaFree on a
// stream-ordered block (#834).
// This type stores the family with the pointer: the release cannot pick another one.
//   auto buf = DeviceOwned<void>::allocate(vram_alloc_, bytes, "moe_dequant");
//   if (!buf) return false;                 // allocation failed, nothing leaked
//   kernel(buf.get(), ...);                 // raw view at the hand-off
//   buf.reset();                            // or ~DeviceOwned: same family, exactly once
// Sizes are bytes (T may be void). Move-only: a copy would be a double free.

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

#include "memory/device_alloc_all.h"
#include "memory/vram_allocator.h"

namespace imp {

// Which release an allocation takes. Vram: VRAMAllocator::free on the stored allocator.
// Plain: device_free_all (cudaFree). Async: device_free_all_async on the stored stream.
enum class DeviceFamily : uint8_t { None, Vram, Plain, Async };

namespace detail {
// The one release of DeviceOwned. Null ptr and family None are no-ops.
void device_owned_release(void* ptr, DeviceFamily family, VRAMAllocator* alloc, cudaStream_t stream);
}  // namespace detail

// Test seam (CPU lane has no device): a non-null hook receives every release instead of the
// driver. Returns the previous hook. Never set in production.
using DeviceOwnedReleaseHook = void (*)(void* ptr, DeviceFamily family, VRAMAllocator* alloc,
                                        cudaStream_t stream);
DeviceOwnedReleaseHook set_device_owned_release_hook(DeviceOwnedReleaseHook hook);

template <typename T>
class DeviceOwned {
public:
    DeviceOwned() = default;
    ~DeviceOwned() { reset(); }

    DeviceOwned(const DeviceOwned&) = delete;
    DeviceOwned& operator=(const DeviceOwned&) = delete;

    DeviceOwned(DeviceOwned&& o) noexcept { steal_(o); }
    DeviceOwned& operator=(DeviceOwned&& o) noexcept {
        if (this != &o) {
            reset();
            steal_(o);
        }
        return *this;
    }

    // The executor helper's choice: alloc non-null takes VRAMAllocator (force: past the
    // headroom), alloc null takes device_alloc_all. Empty on failure or bytes == 0.
    [[nodiscard]] static DeviceOwned allocate(VRAMAllocator* alloc, size_t bytes, const char* tag,
                                              bool force = false) {
        DeviceOwned out;
        if (bytes == 0)
            return out;
        void* p = nullptr;
        if (alloc != nullptr)
            p = alloc->allocate(bytes, tag, force);
        else if (device_alloc_all(dev_req(p, bytes)) != cudaSuccess)
            p = nullptr;
        out.adopt(static_cast<T*>(p), bytes, alloc);
        return out;
    }

    // device_alloc_all with a synchronous release.
    [[nodiscard]] static DeviceOwned allocate_plain(size_t bytes) {
        return allocate(nullptr, bytes, nullptr);
    }

    // device_alloc_all_async on `stream`, released stream-ordered on the same stream.
    [[nodiscard]] static DeviceOwned allocate_async(cudaStream_t stream, size_t bytes) {
        DeviceOwned out;
        void* p = nullptr;
        if (bytes != 0 && device_alloc_all_async(stream, dev_req(p, bytes)) == cudaSuccess)
            out.adopt_async(static_cast<T*>(p), bytes, stream);
        return out;
    }

    // Takes over a buffer of a multi-buffer all-or-none call. alloc non-null: the buffer came
    // from that VRAMAllocator; null: from device_alloc_all. Releases what it held; null ptr = empty.
    void adopt(T* ptr, size_t bytes, VRAMAllocator* alloc) noexcept {
        set_(ptr, bytes, alloc != nullptr ? DeviceFamily::Vram : DeviceFamily::Plain, alloc, nullptr);
    }

    // Takes over a buffer of device_alloc_all_async; released on `stream`.
    void adopt_async(T* ptr, size_t bytes, cudaStream_t stream) noexcept {
        set_(ptr, bytes, DeviceFamily::Async, nullptr, stream);
    }

    // Async only: the stream the release is ordered on (teardown releases on the null stream).
    void set_release_stream(cudaStream_t stream) noexcept {
        if (family_ == DeviceFamily::Async)
            stream_ = stream;
    }

    // Frees early. Idempotent, and safe on a moved-from handle.
    void reset() noexcept {
        if (ptr_ != nullptr)
            detail::device_owned_release(static_cast<void*>(ptr_), family_, alloc_, stream_);
        clear_();
    }

    // Hands the buffer to an owner that is not a DeviceOwned (a weight-cache entry): returns
    // the pointer and empties the handle without a release. The receiver frees in the same family.
    [[nodiscard]] T* release() noexcept {
        T* p = ptr_;
        clear_();
        return p;
    }

    T* get() const noexcept { return ptr_; }
    explicit operator bool() const noexcept { return ptr_ != nullptr; }
    size_t bytes() const noexcept { return bytes_; }
    DeviceFamily family() const noexcept { return family_; }

private:
    void set_(T* ptr, size_t bytes, DeviceFamily family, VRAMAllocator* alloc, cudaStream_t stream) noexcept {
        reset();
        if (ptr == nullptr)
            return;
        ptr_ = ptr;
        bytes_ = bytes;
        family_ = family;
        alloc_ = alloc;
        stream_ = stream;
    }
    void steal_(DeviceOwned& o) noexcept {
        ptr_ = o.ptr_;
        bytes_ = o.bytes_;
        family_ = o.family_;
        alloc_ = o.alloc_;
        stream_ = o.stream_;
        o.clear_();
    }
    void clear_() noexcept {
        ptr_ = nullptr;
        bytes_ = 0;
        family_ = DeviceFamily::None;
        alloc_ = nullptr;
        stream_ = nullptr;
    }

    T* ptr_ = nullptr;
    size_t bytes_ = 0;
    VRAMAllocator* alloc_ = nullptr;
    cudaStream_t stream_ = nullptr;
    DeviceFamily family_ = DeviceFamily::None;
};

}  // namespace imp
