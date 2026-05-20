#pragma once

// Internal helpers shared across pre_dequant_*.cu translation units.
// Not part of any public API; included only by src/exec/pre_dequant_*.cu.
//
// Phase 3 of docs/superpowers/specs/2026-05-20-architecture-refactor-roadmap-design.md

#include "core/storage_tier.h"
#include "core/tensor.h"
#include "exec/executor.h"
#include "exec/executor_helpers.h"
#include "exec/weight_handle.h"
#include "memory/vram_allocator.h"
#include "model/model.h"
#include "model/model_config.h"

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstddef>
#include <unordered_map>

namespace imp::pre_dequant_internal {

// Infer StorageTier from which wcache_ map the source pointer landed in.
inline StorageTier infer_tier_from_wcache(const WeightCaches& wc, const void* src_ptr) {
    if (wc.cutlass_nvfp4.count(src_ptr))
        return StorageTier::CUTLASS_NVFP4;
    if (wc.cutlass_mxfp4.count(src_ptr))
        return StorageTier::MXFP4;
    if (wc.nvfp4.count(src_ptr))
        return StorageTier::NVFP4;
    if (wc.fp8.count(src_ptr))
        return StorageTier::FP8;
    if (wc.fp16.count(src_ptr))
        return StorageTier::FP16;
    return StorageTier::Undefined;
}

// Fill a handle's payload by borrowing pointers from wcache_ entries.
inline void borrow_payload_from_wcache(WeightHandle& h, const WeightCaches& wc, const void* src_ptr) {
    switch (h.primary_tier) {
        case StorageTier::FP16: {
            auto it = wc.fp16.find(src_ptr);
            if (it != wc.fp16.end()) {
                h.payload.fp16.data = static_cast<half*>(it->second.data);
            }
            break;
        }
        case StorageTier::FP8: {
            auto it = wc.fp8.find(src_ptr);
            if (it != wc.fp8.end()) {
                h.payload.fp8.data = static_cast<__nv_fp8_e4m3*>(it->second.weight.data);
                h.payload.fp8.d_scale = it->second.d_scale;
            }
            break;
        }
        case StorageTier::NVFP4: {
            auto it = wc.nvfp4.find(src_ptr);
            if (it != wc.nvfp4.end()) {
                h.payload.nvfp4.data = static_cast<uint8_t*>(it->second.packed_data);
                h.payload.nvfp4.block_scales = static_cast<uint8_t*>(it->second.micro_scales);
                // Borrow a pointer to the host tensor_scale stored in the wcache entry.
                // The NvFP4QuantResult lives in wcache_.nvfp4 (stable address in unordered_map).
                // Callers that read tensor_scale must NOT pass this to cudaMemcpyDeviceToHost
                // (it's a host float, not a device pointer). They should read it as *tensor_scale.
                h.payload.nvfp4.tensor_scale = const_cast<float*>(&it->second.tensor_scale);
                h.payload.nvfp4.tensor_scale_2 = nullptr;
            }
            break;
        }
        case StorageTier::CUTLASS_NVFP4: {
            auto it = wc.cutlass_nvfp4.find(src_ptr);
            if (it != wc.cutlass_nvfp4.end()) {
                h.payload.cutlass_nvfp4.weight = const_cast<void*>(it->second.data);
                h.payload.cutlass_nvfp4.sf = it->second.scale_factors;
                h.payload.cutlass_nvfp4.global_scale = const_cast<float*>(&it->second.tensor_scale);
            }
            break;
        }
        case StorageTier::MXFP4: {
            auto it = wc.cutlass_mxfp4.find(src_ptr);
            if (it != wc.cutlass_mxfp4.end()) {
                h.payload.mxfp4.weight = const_cast<void*>(it->second.data);
                h.payload.mxfp4.scales = it->second.scale_factors;
                h.payload.mxfp4.linear_scales = it->second.linear_scales;
                h.payload.mxfp4.hadamard_bs = it->second.hadamard_bs;
            }
            break;
        }
        default:
            break;
    }
}

// Does this qtype benefit from NVFP4 conversion? (> 4.5 bits/elem)
// Used by Phase 1 (FFN-skip logic) and Phase 3 (NVFP4 decode cache).
inline bool nvfp4_beneficial(QType qt) {
    switch (qt) {
        case QType::Q8_0:
        case QType::Q8_K:
        case QType::Q6_K:
        case QType::Q5_K:
            return true;
        default:
            return false;
    }
}

inline void deduct_budget(size_t& budget, size_t amount) {
    budget = (budget > amount) ? (budget - amount) : 0;
}

inline bool create_fused_weight_pair(const Tensor& w_a, const Tensor& w_b,
                                     const std::unordered_map<const void*, Tensor>& fp16_cache,
                                     VRAMAllocator* allocator, size_t& total_cache_bytes,
                                     size_t remaining_budget, cudaStream_t stream,
                                     std::unordered_map<int, Tensor>& out_map, int layer_idx,
                                     bool& should_stop) {
    should_stop = false;
    if (!w_a.data || !w_b.data)
        return false;
    auto it_a = fp16_cache.find(w_a.data);
    auto it_b = fp16_cache.find(w_b.data);
    if (it_a == fp16_cache.end() || it_b == fp16_cache.end())
        return false;

    int a_rows = static_cast<int>(w_a.shape[0]);
    int K = static_cast<int>(w_a.shape[1]);
    size_t one_sz = static_cast<size_t>(a_rows) * K * sizeof(half);

    if (total_cache_bytes + 2 * one_sz > remaining_budget) {
        should_stop = true;
        return false;
    }

    void* fused_buf = vram_alloc(allocator, 2 * one_sz, "fp16_weight_cache");
    if (!fused_buf) {
        should_stop = true;
        return false;
    }

    IMP_CUDA_CHECK_LOG(
        cudaMemcpyAsync(fused_buf, it_a->second.data, one_sz, cudaMemcpyDeviceToDevice, stream));
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(static_cast<char*>(fused_buf) + one_sz, it_b->second.data, one_sz,
                                       cudaMemcpyDeviceToDevice, stream));

    int64_t shape[2] = {2 * a_rows, static_cast<int64_t>(K)};
    out_map[layer_idx] = Tensor(fused_buf, QType::F16, 2, shape, true);
    total_cache_bytes += 2 * one_sz;
    return true;
}

template <typename Fn>
void for_each_dense_weight(const Model& model, const ModelConfig& cfg, Fn&& fn) {
    for (int i = 0; i < cfg.n_layers; i++) {
        const auto& L = model.layer(i);
        fn(L.wq, L.wq.qtype);
        fn(L.wk, L.wk.qtype);
        fn(L.wv, L.wv.qtype);
        fn(L.wo, L.wo.qtype);
    }
    for (int i = 0; i < cfg.n_layers; i++) {
        const auto& L = model.layer(i);
        fn(L.ssm_in, L.ssm_in.qtype);
        fn(L.ssm_out, L.ssm_out.qtype);
        fn(L.w_gate_shared, L.w_gate_shared.qtype);
        fn(L.w_up_shared, L.w_up_shared.qtype);
        fn(L.w_down_shared, L.w_down_shared.qtype);
        fn(L.w_gate, L.w_gate.qtype);
        fn(L.w_up, L.w_up.qtype);
        fn(L.w_down, L.w_down.qtype);
    }
}

}  // namespace imp::pre_dequant_internal
