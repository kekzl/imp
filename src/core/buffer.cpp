#include "core/buffer.h"
#include "core/logging.h"
#include <cuda_runtime.h>
#include <cstring>
#include <stdexcept>

namespace imp {

static void check_cuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string(msg) + ": " + cudaGetErrorString(err));
    }
}

Buffer::~Buffer() {
    reset();
}

Buffer::Buffer(Buffer&& other) noexcept
    : data_(other.data_), size_(other.size_),
      on_device_(other.on_device_), pinned_(other.pinned_) {
    other.data_ = nullptr;
    other.size_ = 0;
}

Buffer& Buffer::operator=(Buffer&& other) noexcept {
    if (this != &other) {
        reset();
        data_ = other.data_;
        size_ = other.size_;
        on_device_ = other.on_device_;
        pinned_ = other.pinned_;
        other.data_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}

Buffer Buffer::device(size_t nbytes) {
    Buffer buf;
    buf.size_ = nbytes;
    buf.on_device_ = true;
    if (nbytes > 0) {
        check_cuda(cudaMalloc(&buf.data_, nbytes), "cudaMalloc");
    }
    return buf;
}

Buffer Buffer::host(size_t nbytes) {
    Buffer buf;
    buf.size_ = nbytes;
    if (nbytes > 0) {
        buf.data_ = std::malloc(nbytes);
        if (!buf.data_) throw std::bad_alloc();
    }
    return buf;
}

Buffer Buffer::pinned(size_t nbytes) {
    Buffer buf;
    buf.size_ = nbytes;
    buf.pinned_ = true;
    if (nbytes > 0) {
        check_cuda(cudaMallocHost(&buf.data_, nbytes), "cudaMallocHost");
    }
    return buf;
}

void Buffer::copy_from_host(const void* src, size_t nbytes) {
    if (nbytes == 0) return;
    if (on_device_) {
        check_cuda(
            cudaMemcpy(data_, src, nbytes, cudaMemcpyHostToDevice),
            "cudaMemcpy H2D");
    } else {
        std::memcpy(data_, src, nbytes);
    }
}

void Buffer::copy_to_host(void* dst, size_t nbytes) const {
    if (nbytes == 0) return;
    if (on_device_) {
        check_cuda(
            cudaMemcpy(dst, data_, nbytes, cudaMemcpyDeviceToHost),
            "cudaMemcpy D2H");
    } else {
        std::memcpy(dst, data_, nbytes);
    }
}

void Buffer::copy_from(const Buffer& src, size_t nbytes) {
    if (nbytes == 0) return;
    cudaMemcpyKind kind;
    if (on_device_ && src.on_device_) {
        kind = cudaMemcpyDeviceToDevice;
    } else if (on_device_ && !src.on_device_) {
        kind = cudaMemcpyHostToDevice;
    } else if (!on_device_ && src.on_device_) {
        kind = cudaMemcpyDeviceToHost;
    } else {
        std::memcpy(data_, src.data_, nbytes);
        return;
    }
    check_cuda(cudaMemcpy(data_, src.data_, nbytes, kind), "cudaMemcpy");
}

void Buffer::copy_from_host_async(const void* src, size_t nbytes, void* stream) {
    if (nbytes == 0) return;
    check_cuda(
        cudaMemcpyAsync(data_, src, nbytes, cudaMemcpyHostToDevice,
                        static_cast<cudaStream_t>(stream)),
        "cudaMemcpyAsync H2D");
}

void Buffer::copy_to_host_async(void* dst, size_t nbytes, void* stream) const {
    if (nbytes == 0) return;
    check_cuda(
        cudaMemcpyAsync(dst, data_, nbytes, cudaMemcpyDeviceToHost,
                        static_cast<cudaStream_t>(stream)),
        "cudaMemcpyAsync D2H");
}

void Buffer::zero() {
    if (!data_ || size_ == 0) return;
    if (on_device_) {
        check_cuda(cudaMemset(data_, 0, size_), "cudaMemset");
    } else {
        std::memset(data_, 0, size_);
    }
}

void* Buffer::release() {
    void* p = data_;
    data_ = nullptr;
    size_ = 0;
    return p;
}

void Buffer::reset() {
    if (!data_) return;
    if (on_device_) {
        cudaFree(data_);
    } else if (pinned_) {
        cudaFreeHost(data_);
    } else {
        std::free(data_);
    }
    data_ = nullptr;
    size_ = 0;
}

} // namespace imp
