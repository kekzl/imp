#pragma once

#include "imp/tensor_kind.h"
#include "imp/storage_tier.h"

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <vector>

namespace imp {

struct WeightHandle {
    TensorID    id             = kInvalidTensorID;
    TensorKind  kind           = TensorKind::UNKNOWN;
    StorageTier primary_tier   = StorageTier::Undefined;
    int64_t     shape[2]       = {0, 0};
    int64_t     owned_bytes    = 0;     // zero if storage is borrowed from legacy cache

    union {
        struct { float* data; }                                fp32;
        struct { half* data; }                                 fp16;
        struct { __nv_fp8_e4m3* data; float* d_scale; }        fp8;
        struct { uint8_t* data; uint8_t* block_scales;
                 float* tensor_scale; float* tensor_scale_2; } nvfp4;
        struct { void* weight; void* sf; float* global_scale; } cutlass_nvfp4;
        struct { void* weight; void* scales; void* linear_scales; } mxfp4;
    } payload;

    bool is_populated() const { return primary_tier != StorageTier::Undefined; }
};

class WeightRegistry {
public:
    TensorID reserve(TensorKind kind, int64_t rows, int64_t cols);
    WeightHandle& handle(TensorID id);
    const WeightHandle& handle(TensorID id) const;
    size_t size() const { return handles_.size(); }

    void clear();

private:
    std::vector<WeightHandle> handles_;
};

} // namespace imp
