#pragma once

#include "core/qtype.h"
#include "imp/tensor_kind.h"
#include <cstdint>
#include <cstddef>
#include <cassert>
#include <string>

namespace imp {

// Legacy aliases. Prefer qtype_elem_bytes / qtype_name in new code.
inline size_t dtype_size(QType q) { return qtype_elem_bytes(q); }
inline const char* dtype_name(QType q) { return qtype_name(q); }

static constexpr int kMaxDims = 4;

struct Tensor {
    void*   data         = nullptr;
    QType   qtype        = QType::NONE;
    int     ndim         = 0;
    int64_t shape[kMaxDims]  = {};
    int64_t stride[kMaxDims] = {};
    bool    on_device    = false;
    TensorKind kind      = TensorKind::UNKNOWN;

    // Sidecar pointers for block-quantised tensors. Borrowed; lifetime
    // managed by the loader/WeightCaches that allocated them.
    //   scales       — per-block scales (FP8 micro-scales for NVFP4,
    //                  FP16 per-group scales for FP8 weights, etc.)
    //   tensor_scale — per-tensor scalar (FP32) for two-level schemes
    //                  like NVFP4 and absolute FP8.
    void*   scales       = nullptr;
    void*   tensor_scale = nullptr;

    Tensor() = default;

    // Create a tensor descriptor (does not allocate memory)
    Tensor(void* data, QType qtype, int ndim, const int64_t* shape, bool on_device);

    // Create with explicit strides
    Tensor(void* data, QType qtype, int ndim, const int64_t* shape,
           const int64_t* stride, bool on_device);

    int64_t numel() const;
    size_t  nbytes() const;
    bool    is_contiguous() const;
    void    compute_strides();

    Tensor reshape(int new_ndim, const int64_t* new_shape) const;
    Tensor slice(int64_t start, int64_t end) const;

    std::string to_string() const;
};

} // namespace imp
