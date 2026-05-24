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

    // Sidecar metadata for block-quantised tensors:
    //   scales       — borrowed device pointer to per-block scales
    //                  (FP8 E4M3 micro-scales for NVFP4 [N, K/16],
    //                  FP16 per-group scales for split Q4_0, etc.)
    //   tensor_scale — per-tensor scalar (FP32, by value) for two-level
    //                  schemes like NVFP4. Default 1.0 = no-op
    //                  (multiplicative identity). For llm-compressor
    //                  NVFP4 the loader pre-applies the 1/x reciprocal
    //                  so the runtime can always multiply.
    void* scales = nullptr;
    float tensor_scale = 1.0f;

    // GGUF MXFP4 has two on-disk block layouts:
    //   - imp legacy (GGML type 31): [data (16 bytes) | scale (1 byte)] per block
    //   - llama.cpp standard (GGML type 39): [scale (1 byte) | data (16 bytes)] per block
    // When loaded from a type-39 GGUF this flag is true; the weight_upload MXFP4
    // path swaps the byte offsets so the GPU-side split-layout is identical.
    bool mxfp4_layout_v2 = false;

    // Phase 5 PR #1 Commit 5.1.4.b: original GGUF source bytes have been
    // freed by Phase-4b. `data` is left as a stale hash-key pointer; any
    // dispatch site that dereferences `data` raw must skip when this is
    // true and route via overlay tier instead.
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
