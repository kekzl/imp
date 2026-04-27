#include "core/tensor.h"
#include <cstring>
#include <stdexcept>
#include <sstream>

namespace imp {

Tensor::Tensor(void* data, QType qtype, int ndim, const int64_t* shape, bool on_device)
    : data(data), qtype(qtype), ndim(ndim), on_device(on_device) {
    assert(ndim >= 0 && ndim <= kMaxDims);
    for (int i = 0; i < ndim; ++i) {
        this->shape[i] = shape[i];
    }
    compute_strides();
}

Tensor::Tensor(void* data, QType qtype, int ndim, const int64_t* shape,
               const int64_t* stride, bool on_device)
    : data(data), qtype(qtype), ndim(ndim), on_device(on_device) {
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
    if (qtype == QType::INT4 || qtype == QType::FP4_E2M1) {
        return static_cast<size_t>((n + 1) / 2); // 2 elements per byte
    }
    return static_cast<size_t>(n) * qtype_elem_bytes(qtype);
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
    t.qtype = qtype;
    t.scales = scales;
    t.tensor_scale = tensor_scale;  // copies the float by value
    t.ndim = new_ndim;
    t.on_device = on_device;
    t.kind = kind;

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
    t.data = static_cast<char*>(data) + start * stride[0] * static_cast<int64_t>(qtype_elem_bytes(qtype));
    return t;
}

std::string Tensor::to_string() const {
    std::ostringstream ss;
    ss << "Tensor(shape=[";
    for (int i = 0; i < ndim; ++i) {
        if (i > 0) ss << ", ";
        ss << shape[i];
    }
    ss << "], qtype=" << qtype_name(qtype);
    ss << ", " << (on_device ? "cuda" : "cpu") << ")";
    return ss.str();
}

} // namespace imp
