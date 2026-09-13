#pragma once

#include "core/qtype.h"
#include "core/tensor_kind.h"
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
    void* data = nullptr;
    QType qtype = QType::NONE;
    int ndim = 0;
    int64_t shape[kMaxDims] = {};
    int64_t stride[kMaxDims] = {};
    bool on_device = false;
    TensorKind kind = TensorKind::UNKNOWN;

    // Sidecar metadata for block-quantized tensors. scales: borrowed per-block
    // scale pointer (FP8 E4M3 micro-scales for NVFP4 [N,K/16], FP16 per-group
    // for split Q4_0). tensor_scale: per-tensor FP32 scalar for two-level
    // schemes (NVFP4); default 1.0 = no-op. llm-compressor NVFP4 pre-applies the 1/x reciprocal.
    void* scales = nullptr;
    float tensor_scale = 1.0f;

    // GGUF MXFP4 has two on-disk block layouts: imp legacy (GGML type 31)
    // [data(16B)|scale(1B)], llama.cpp standard (type 39) [scale(1B)|data(16B)].
    // mxfp4_layout_v2=true for type-39; weight_upload swaps byte offsets so the GPU split layout matches.
    bool mxfp4_layout_v2 = false;

    // dropped_source=true: original GGUF source bytes were freed (Phase-4b);
    // `data` is a stale hash-key pointer. Any dispatch site dereferencing
    // `data` raw must skip and route via the overlay tier instead.
    bool dropped_source = false;

    Tensor() = default;

    // Create a tensor descriptor (does not allocate memory)
    Tensor(void* data, QType qtype, int ndim, const int64_t* shape, bool on_device);

    // Create with explicit strides
    Tensor(void* data, QType qtype, int ndim, const int64_t* shape, const int64_t* stride, bool on_device);

    int64_t numel() const;
    size_t nbytes() const;
    bool is_contiguous() const;
    void compute_strides();

    Tensor reshape(int new_ndim, const int64_t* new_shape) const;
    Tensor slice(int64_t start, int64_t end) const;

    std::string to_string() const;
};

}  // namespace imp
