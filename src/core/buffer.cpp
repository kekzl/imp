#include "core/buffer.h"
#include "core/logging.h"
#include <cuda_runtime.h>
#include <cstring>
#include <stdexcept>

namespace imp {

static void check_cuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
    }
}

Buffer::~Buffer() { reset(); }

Buffer::Buffer(Buffer&& other) noexcept
    : data_(other.data_), size_(other.size_), on_device_(other.on_device_) {
    other.data_ = nullptr;
    other.size_ = 0;
}

Buffer& Buffer::operator=(Buffer&& other) noexcept {
    if (this != &other) {
        reset();
        data_ = other.data_;
        size_ = other.size_;
        on_device_ = other.on_device_;
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
        if (!buf.data_)
            throw std::bad_alloc();
    }
    return buf;
}

void Buffer::zero() {
    if (!data_ || size_ == 0)
        return;
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
    if (!data_)
        return;
    if (on_device_) {
        IMP_CUDA_CHECK_LOG(cudaFree(data_));
    } else {
        std::free(data_);
    }
    data_ = nullptr;
    size_ = 0;
}

}  // namespace imp
