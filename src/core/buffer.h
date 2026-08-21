#pragma once

#include <cstddef>
#include <cstdint>

namespace imp {

// RAII wrapper for GPU/CPU/Pinned memory buffers.
// Move-only. Automatically frees memory on destruction.
class Buffer {
public:
    Buffer() = default;
    ~Buffer();

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    Buffer(Buffer&& other) noexcept;
    Buffer& operator=(Buffer&& other) noexcept;

    // Factory methods
    static Buffer device(size_t nbytes);
    static Buffer host(size_t nbytes);
    // There was a pinned() factory here with zero callers, and with it a
    // `pinned_` flag that reset() branched on. Pinned host memory has an owner
    // of its own now (memory/host_pinned.h, T5b); this class is the
    // device-or-pageable one. Removed rather than migrated — a second owner
    // wrapped around the first would have removed no free.

    // Accessors
    void* ptr() const { return data_; }
    size_t size() const { return size_; }
    explicit operator bool() const { return data_ != nullptr; }

    template <typename T>
    T* as() const {
        return static_cast<T*>(data_);
    }

    // Memset
    void zero();

    // Release ownership (caller takes responsibility for freeing)
    void* release();
    void reset();

private:
    void* data_ = nullptr;
    size_t size_ = 0;
    bool on_device_ = false;
};

}  // namespace imp
