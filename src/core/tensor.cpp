#include "core/tensor.h"
#include <cstring>
#include <stdexcept>
#include <sstream>

namespace imp {

size_t dtype_size(DType dt) {
    switch (dt) {
        case DType::FP32:     return 4;
        case DType::FP16:     return 2;
        case DType::BF16:     return 2;
        case DType::FP8_E4M3: return 1;
        case DType::FP8_E5M2: return 1;
        case DType::INT8:     return 1;
        case DType::INT4:      return 1; // packed: 2 elements per byte
        case DType::INT32:     return 4;
        case DType::FP4_E2M1:  return 1; // packed: 2 elements per byte
    }
    return 0;
}

const char* dtype_name(DType dt) {
    switch (dt) {
        case DType::FP32:     return "FP32";
        case DType::FP16:     return "FP16";
        case DType::BF16:     return "BF16";
        case DType::FP8_E4M3: return "FP8_E4M3";
        case DType::FP8_E5M2: return "FP8_E5M2";
        case DType::INT8:     return "INT8";
        case DType::INT4:      return "INT4";
        case DType::INT32:     return "INT32";
        case DType::FP4_E2M1:  return "FP4_E2M1";
    }
    return "UNKNOWN";
}

Tensor::Tensor(void* data, DType dtype, int ndim, const int64_t* shape, bool on_device)
    : data(data), dtype(dtype), ndim(ndim), on_device(on_device) {
    assert(ndim >= 0 && ndim <= kMaxDims);
    for (int i = 0; i < ndim; ++i) {
        this->shape[i] = shape[i];
    }
    compute_strides();
}

Tensor::Tensor(void* data, DType dtype, int ndim, const int64_t* shape,
               const int64_t* stride, bool on_device)
    : data(data), dtype(dtype), ndim(ndim), on_device(on_device) {
    assert(ndim >= 0 && ndim <= kMaxDims);
    for (int i = 0; i < ndim; ++i) {
        this->shape[i] = shape[i];
        this->stride[i] = stride[i];
    }
}

int64_t Tensor::numel() const {
    if (ndim == 0) return 0;
    int64_t n = 1;
    for (int i = 0; i < ndim; ++i) {
        n *= shape[i];
    }
    return n;
}

size_t Tensor::nbytes() const {
    int64_t n = numel();
    if (dtype == DType::INT4 || dtype == DType::FP4_E2M1) {
        return static_cast<size_t>((n + 1) / 2); // 2 elements per byte
    }
    return static_cast<size_t>(n) * dtype_size(dtype);
}

bool Tensor::is_contiguous() const {
    if (ndim == 0) return true;
    int64_t expected = 1;
    for (int i = ndim - 1; i >= 0; --i) {
        if (stride[i] != expected) return false;
        expected *= shape[i];
    }
    return true;
}

void Tensor::compute_strides() {
    if (ndim == 0) return;
    stride[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; --i) {
        stride[i] = stride[i + 1] * shape[i + 1];
    }
}

Tensor Tensor::reshape(int new_ndim, const int64_t* new_shape) const {
    Tensor t;
    t.data = data;
    t.dtype = dtype;
    t.ndim = new_ndim;
    t.on_device = on_device;

    int64_t new_numel = 1;
    for (int i = 0; i < new_ndim; ++i) {
        t.shape[i] = new_shape[i];
        new_numel *= new_shape[i];
    }

    if (new_numel != numel()) {
        throw std::invalid_argument("reshape: numel mismatch");
    }

    t.compute_strides();
    return t;
}

Tensor Tensor::slice(int64_t start, int64_t end) const {
    assert(ndim > 0);
    assert(start >= 0 && end <= shape[0] && start < end);

    Tensor t = *this;
    t.shape[0] = end - start;
    t.data = static_cast<char*>(data) + start * stride[0] * static_cast<int64_t>(dtype_size(dtype));
    return t;
}

std::string Tensor::to_string() const {
    std::ostringstream ss;
    ss << "Tensor(shape=[";
    for (int i = 0; i < ndim; ++i) {
        if (i > 0) ss << ", ";
        ss << shape[i];
    }
    ss << "], dtype=" << dtype_name(dtype);
    ss << ", " << (on_device ? "cuda" : "cpu") << ")";
    return ss.str();
}

} // namespace imp
