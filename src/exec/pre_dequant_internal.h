#pragma once

// Internal helpers shared across pre_dequant_*.cu translation units.
// Not part of any public API; included only by src/exec/pre_dequant_*.cu.
//
// Phase 3 of the architecture-refactor roadmap (archived: docs/archive/README.md)

#include "core/storage_tier.h"
#include "core/tensor.h"
#include "exec/executor.h"
#include "exec/executor_helpers.h"
#include "exec/weight_handle.h"
#include "memory/vram_allocator.h"
#include "model/model.h"
#include "model/model_config.h"
#include "core/dispatch_policy.h"

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstddef>
#include <unordered_map>

namespace imp::pre_dequant_internal {

// #982 net rule for the NVFP4 LM-head decode cache (gemm.nvfp4_lm_head,
// "auto"/"on"/"off", legacy true/false accepted). The 2026-07-12 parity sweep
// measured the cache PER SOURCE TYPE:
//  - QUANTIZED (GGUF) heads: dp4a already decodes them fast, so the cache buys
//    little but stacks NVFP4 on the quant lattice — net-positive ONLY on small
//    dense models (4B +6.6% decode/+3.8% PPL, 8B +5.8%/+2.6%); net-NEGATIVE at
//    14B Q6_K (+1.9%/+2.1%) and 30B-A3B Q4_K_M (+3.7%/+5.0%), 35B a wash.
//    auto → ON iff dense && d_model <= 4096 (the measured net-positive set).
//    EXCEPTION — GDN/SSM hybrids: their head trade is owned by
//    gemm.nvfp4_lm_head_gdn (GOAL-listed, default ON: +5.3% decode / +1.4% PPL
//    on the Qwen3.6-35B UD-Q4_K_M hero, re-measured 2026-07-15). The dense/MoE
//    net rule's is_dense=false arm silently voided that flag on quantized
//    hybrids and cost the hero −5% decode — pass is_gdn_hybrid so auto defers
//    to the callers' nvfp4_lm_head_gdn gate instead.
//  - NATIVE (F16/BF16) heads: 2 B/elem cuBLAS GEMV is the alternative, the
//    cache is a 4x byte win (+8-16% decode, +2.2% PPL, owner-accepted trade,
//    GOAL-listed). auto → ON (unchanged).
// GDN/SSM hybrids remain additionally gated by gemm.nvfp4_lm_head_gdn.
inline bool nvfp4_lm_head_enabled(const DispatchPolicy& rc, bool quantized_source, bool is_dense, int d_model,
                                  bool is_gdn_hybrid = false) {
    const std::string& v = rc.gemm.nvfp4_lm_head;
    if (v == "on" || v == "true" || v == "1")
        return true;
    if (v == "off" || v == "false" || v == "0")
        return false;
    if (!quantized_source)
        return true;
    if (is_gdn_hybrid)
        return true;  // decided by the callers' gemm.nvfp4_lm_head_gdn gate
    return is_dense && d_model <= 4096;
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
    // Marlin W4A16 sidecar attaches independent of the primary tier.
    if (auto it = wc.marlin.find(src_ptr); it != wc.marlin.end())
        h.marlin_sidecar = &it->second;
}

// nvfp4_beneficial() moved to core/qtype.h — the VRAM-budget heuristic
// (vram_budget.cpp) shares the same policy and the two must not drift.
using imp::nvfp4_beneficial;

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
