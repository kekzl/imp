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
    static Buffer pinned(size_t nbytes);

    // Accessors
    void* ptr() const { return data_; }
    size_t size() const { return size_; }
    bool is_device() const { return on_device_; }
    bool is_pinned() const { return pinned_; }
    explicit operator bool() const { return data_ != nullptr; }

    template <typename T>
    T* as() const { return static_cast<T*>(data_); }

    // Copy operations (synchronous)
    void copy_from_host(const void* src, size_t nbytes);
    void copy_to_host(void* dst, size_t nbytes) const;
    void copy_from(const Buffer& src, size_t nbytes);

    // Async copy operations
    void copy_from_host_async(const void* src, size_t nbytes, void* stream);
    void copy_to_host_async(void* dst, size_t nbytes, void* stream) const;

    // Memset
    void zero();

    // Release ownership (caller takes responsibility for freeing)
    void* release();
    void reset();

private:
    void* data_    = nullptr;
    size_t size_   = 0;
    bool on_device_ = false;
    bool pinned_    = false;
};

} // namespace imp
