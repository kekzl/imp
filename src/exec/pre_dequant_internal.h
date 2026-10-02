#pragma once

// Internal helpers shared across pre_dequant_*.cu translation units. Not a public API;
// included only by src/exec/pre_dequant_*.cu.

#include "core/storage_tier.h"
#include "core/tensor.h"
#include "exec/executor.h"
#include "exec/executor_helpers.h"
#include "exec/weight_handle.h"
#include "memory/device_alloc_all.h"
#include "memory/vram_allocator.h"
#include "model/model.h"
#include "model/layer_host_keep.h"
#include "model/model_config.h"
#include "core/dispatch_policy.h"
#include "core/config/lm_head_mode.h"
#include "quant/dequant_gpu.h"

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstddef>
#include <unordered_map>

namespace imp::pre_dequant_internal {

// FP8 cache buffers: scale scratch (block maxes [max_grid], absmax, scales [n]) all or none, then the
// data bulk. Null bulk = pass skipped, never run on a null scratch (#2446); scratch freed by the caller.
struct Fp8CacheBuffers {
    float* block_maxes = nullptr;
    float* absmax = nullptr;
    float* scales = nullptr;
    uint8_t* bulk = nullptr;
};

[[nodiscard]] inline Fp8CacheBuffers alloc_fp8_cache_buffers(VRAMAllocator* va, size_t max_grid, size_t n,
                                                             size_t bulk_bytes, const char* tag) {
    Fp8CacheBuffers b;
    if (device_alloc_all(dev_req(b.block_maxes, max_grid * sizeof(float)), dev_req(b.absmax, sizeof(float)),
                         dev_req(b.scales, n * sizeof(float))) == cudaSuccess)
        b.bulk = static_cast<uint8_t*>(vram_alloc(va, bulk_bytes, tag));
    return b;
}

// The per-row FP8 head (pre_dequant_fp8_lm_head.cpp) can serve this head: F16 or GPU-dequantable
// source on device, F16 final norm (F16 hidden rows), d_model a multiple of 256.
[[nodiscard]] inline bool fp8_lm_head_eligible(const Model& m) {
    const Tensor& lm = m.output_proj();
    const bool src_ok = lm.qtype == QType::F16 || dequant_gpu_supported(lm.qtype);
    const bool f16_compute = !m.output_norm().data || m.output_norm().qtype == QType::F16;
    return lm.data && lm.on_device && lm.ndim == 2 && src_ok && f16_compute && (lm.shape[1] % 256) == 0;
}

// The checkpoint ships the per-row FP8 head (#2479): E4M3 codes + F32 row scales on device. Shape,
// F16 norm and d_model % 256 are refused at upload (upload_lm_head_row_scales).
[[nodiscard]] inline bool lm_head_checkpoint_fp8_rows(const Model& m) {
    const Tensor& lm = m.output_proj();
    const Tensor& sc = m.output_proj_row_scales();
    return lm.data && lm.on_device && lm.qtype == QType::FP8_E4M3 && sc.data && sc.on_device;
}

// This load builds the FP8 head: fp8, or auto with a 16-bit source (#2224), and the head is eligible.
[[nodiscard]] inline bool fp8_lm_head_wanted(const DispatchPolicy& rc, const Model& m) {
    return lm_head_mode_fp8(lm_head_mode(rc.gemm.nvfp4_lm_head), m.output_proj().qtype) &&
           fp8_lm_head_eligible(m);
}

// #982 net rule for the NVFP4 LM-head decode cache (gemm.nvfp4_lm_head: auto/on/off/fp8,
// legacy bool accepted). auto with an FP8-eligible head (fp8_head) -> no NVFP4 head (#2166).
// 8-bit quantized heads (Q8_0, FP8): auto -> OFF, checkpoint precision (#2224).
// Narrower QUANTIZED (GGUF) heads: auto -> ON iff dense && d_model <= 4096
// (net-positive only on small dense models; net-negative at larger sizes).
// EXCEPTION: GDN/SSM hybrids are owned by gemm.nvfp4_lm_head_gdn (GOAL-listed, default
// ON) instead; the dense/MoE net rule's is_dense=false arm silently voided that flag on
// quantized hybrids, so auto defers to nvfp4_lm_head_gdn via is_gdn_hybrid.
// NATIVE (F16/BF16) heads: auto -> ON unconditionally (cuBLAS GEMV alternative, NVFP4
// cache wins on bytes; GOAL-listed).
[[nodiscard]] inline bool nvfp4_lm_head_enabled(const DispatchPolicy& rc, bool quantized_source, QType head_qtype,
                                  bool is_dense, int d_model, bool is_gdn_hybrid, bool fp8_head) {
    const LmHeadMode mode = lm_head_mode(rc.gemm.nvfp4_lm_head);
    if (mode == LmHeadMode::Nvfp4)
        return true;
    if (mode == LmHeadMode::Source || mode == LmHeadMode::Fp8 || fp8_head ||
        lm_head_auto_keeps_source(mode, head_qtype))
        return false;
    if (!quantized_source)
        return true;
    if (is_gdn_hybrid)
        return true;  // decided by the callers' gemm.nvfp4_lm_head_gdn gate
    return is_dense && d_model <= 4096;
}

// Any MoE layer whose packed experts stay on the host (the expert cache then gets freed VRAM).
inline bool has_host_resident_experts(const Model& m) {
    for (int i = 0; i < m.config().n_layers; ++i) {
        const auto& L = m.layer(i);
        if (L.expert_up_packed.data && !L.expert_up_packed.on_device)
            return true;
        // NVFP4-prequant checkpoints hold per-expert 2-D tensors instead of a packed 3-D one.
        if (!L.expert_w_up.empty() && L.expert_w_up[0].data && !L.expert_w_up[0].on_device)
            return true;
    }
    return false;
}

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
                // Borrow a pointer to the host tensor_scale stored in the wcache entry (stable address in
                // unordered_map). Callers must NOT pass this to cudaMemcpyDeviceToHost: it's a host
                // float, not a device pointer. Read as *tensor_scale.
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

// nvfp4_beneficial() moved to core/qtype.h — the VRAM-budget heuristic
// (vram_budget.cpp) shares the same policy and the two must not drift.
using imp::nvfp4_beneficial;

inline void deduct_budget(size_t& budget, size_t amount) {
    budget = (budget > amount) ? (budget - amount) : 0;
}

[[nodiscard]] inline bool create_fused_weight_pair(const Tensor& w_a, const Tensor& w_b,
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

    int64_t shape[2] = {2 * static_cast<int64_t>(a_rows), static_cast<int64_t>(K)};
    out_map[layer_idx] = Tensor(fused_buf, QType::F16, 2, shape, true);
    total_cache_bytes += 2 * one_sz;
    return true;
}

// Skips host-streamed weights (--gpu-layers, model/layer_host_keep.h).
template <typename Fn>
void for_each_dense_weight(const Model& model, const ModelConfig& cfg, Fn&& fn) {
    imp::for_each_device_dense_weight(model, cfg.n_layers, std::forward<Fn>(fn));
}

}  // namespace imp::pre_dequant_internal
