#pragma once

#include "core/tensor.h"

namespace imp {

// Non-owning view into a Tensor. Provides read-only access to tensor metadata
// without owning the underlying data. Lightweight, copyable.
class TensorView {
public:
    TensorView() = default;

    explicit TensorView(const Tensor& t)
        : data_(t.data), dtype_(t.dtype), ndim_(t.ndim), on_device_(t.on_device) {
        for (int i = 0; i < t.ndim; ++i) {
            shape_[i] = t.shape[i];
            stride_[i] = t.stride[i];
        }
    }

    TensorView(const void* data, DType dtype, int ndim,
               const int64_t* shape, const int64_t* stride, bool on_device)
        : data_(data), dtype_(dtype), ndim_(ndim), on_device_(on_device) {
        for (int i = 0; i < ndim; ++i) {
            shape_[i] = shape[i];
            stride_[i] = stride[i];
        }
    }

    const void* data() const { return data_; }
    DType dtype() const { return dtype_; }
    int ndim() const { return ndim_; }
    bool on_device() const { return on_device_; }
    int64_t shape(int i) const { return shape_[i]; }
    int64_t stride(int i) const { return stride_[i]; }

    int64_t numel() const {
        if (ndim_ == 0) return 0;
        int64_t n = 1;
        for (int i = 0; i < ndim_; ++i) n *= shape_[i];
        return n;
    }

    size_t nbytes() const {
        int64_t n = numel();
        if (dtype_ == DType::INT4) return static_cast<size_t>((n + 1) / 2);
        return static_cast<size_t>(n) * dtype_size(dtype_);
    }

    bool is_contiguous() const {
        if (ndim_ == 0) return true;
        int64_t expected = 1;
        for (int i = ndim_ - 1; i >= 0; --i) {
            if (stride_[i] != expected) return false;
            expected *= shape_[i];
        }
        return true;
    }

private:
    const void* data_ = nullptr;
    DType dtype_       = DType::FP32;
    int ndim_          = 0;
    int64_t shape_[kMaxDims]  = {};
    int64_t stride_[kMaxDims] = {};
    bool on_device_    = false;
};

} // namespace imp
