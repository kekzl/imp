#pragma once

#include <cstdint>
#include <cstddef>
#include <cassert>
#include <string>

namespace imp {

enum class DType : uint8_t {
    FP32      = 0,
    FP16      = 1,
    BF16      = 2,
    FP8_E4M3  = 3,
    FP8_E5M2  = 4,
    INT8      = 5,
    INT4      = 6,
    INT32     = 7,
    FP4_E2M1  = 8,
};

// Bytes per element. INT4 returns 1 (two elements packed per byte).
size_t dtype_size(DType dt);
const char* dtype_name(DType dt);

static constexpr int kMaxDims = 4;

struct Tensor {
    void* data       = nullptr;
    DType dtype      = DType::FP32;
    int ndim         = 0;
    int64_t shape[kMaxDims]  = {};
    int64_t stride[kMaxDims] = {};
    bool on_device   = false;

    Tensor() = default;

    // Create a tensor descriptor (does not allocate memory)
    Tensor(void* data, DType dtype, int ndim, const int64_t* shape, bool on_device);

    // Create with explicit strides
    Tensor(void* data, DType dtype, int ndim, const int64_t* shape,
           const int64_t* stride, bool on_device);

    // Total number of elements
    int64_t numel() const;

    // Total size in bytes
    size_t nbytes() const;

    // Check if memory layout is contiguous (row-major)
    bool is_contiguous() const;

    // Compute row-major strides from shape
    void compute_strides();

    // Reshape (must have same numel). Returns new descriptor with same data ptr.
    Tensor reshape(int new_ndim, const int64_t* new_shape) const;

    // View a sub-range along dimension 0
    Tensor slice(int64_t start, int64_t end) const;

    // Debug string: "Tensor(shape=[...], dtype=FP16, device=cuda)"
    std::string to_string() const;
};

} // namespace imp
