#include "graph/weight_cache_manager.h"
#include "quant/nvfp4_quant.h"
#include "compute/gemm_cutlass_sm120.h"
#include "compute/gemm_cutlass_mxfp4_sm120.h"
#include "core/logging.h"
#include <cuda_runtime.h>

namespace imp {

static void vram_free_wc(VRAMAllocator* alloc, void* ptr) {
    if (!ptr) return;
    if (alloc) alloc->free(ptr);
    else IMP_CUDA_CHECK_LOG(cudaFree(ptr));
}

void WeightCacheManager::free(VRAMAllocator* alloc) {
    // Free fused KV weight cache
    for (auto& [idx, tensor] : fused_kv) {
        if (tensor.data) vram_free_wc(alloc, tensor.data);
    }
    fused_kv.clear();

    // Free fused gate+up weight cache
    for (auto& [idx, tensor] : fused_gate_up) {
        if (tensor.data) vram_free_wc(alloc, tensor.data);
    }
    fused_gate_up.clear();

    // Free FP16 weight cache
    for (auto& [ptr, tensor] : fp16) {
        vram_free_wc(alloc, tensor.data);
    }
    fp16.clear();
    fp16_bytes = 0;

    // Free NVFP4 decode weight cache
    for (auto& [ptr, result] : nvfp4) {
        free_nvfp4_result(result);
    }
    nvfp4.clear();
    nvfp4_bytes = 0;

    // Free NVFP4 MoE expert weight cache
    for (auto& [ptr, result] : nvfp4_moe) {
        free_nvfp4_moe_result(result);
    }
    nvfp4_moe.clear();
    nvfp4_moe_bytes = 0;

    // Free CUTLASS sm_120 NVFP4 weight cache
    for (auto& [ptr, cw] : cutlass_nvfp4) {
        free_cutlass_nvfp4_weight(cw);
    }
    cutlass_nvfp4.clear();
    cutlass_nvfp4_bytes = 0;

    // Free CUTLASS MXFP4 weight cache
    for (auto& [ptr, mw] : cutlass_mxfp4) {
        free_cutlass_mxfp4_weight(mw);
    }
    cutlass_mxfp4.clear();
    cutlass_mxfp4_bytes = 0;

    // Free FP8 weight cache (complex: entries may point into bulk buffers)
    for (auto& [ptr, entry] : fp8) {
        if (entry.weight.data) {
            bool in_migrated_data = fp8_migrated_data &&
                reinterpret_cast<uintptr_t>(entry.weight.data) >= reinterpret_cast<uintptr_t>(fp8_migrated_data) &&
                reinterpret_cast<uintptr_t>(entry.weight.data) < reinterpret_cast<uintptr_t>(fp8_migrated_data) + fp8_migrated_data_size;
            bool in_overflow_data = fp8_overflow_data &&
                reinterpret_cast<uintptr_t>(entry.weight.data) >= reinterpret_cast<uintptr_t>(fp8_overflow_data) &&
                reinterpret_cast<uintptr_t>(entry.weight.data) < reinterpret_cast<uintptr_t>(fp8_overflow_data) + fp8_overflow_data_size;
            if (!in_migrated_data && !in_overflow_data) cudaFree(entry.weight.data);
        }
        if (entry.d_scale) {
            bool in_migrated = fp8_migrated_scales &&
                               entry.d_scale >= fp8_migrated_scales &&
                               entry.d_scale < fp8_migrated_scales + fp8_migrated_count;
            bool in_overflow = fp8_overflow_scales &&
                               entry.d_scale >= fp8_overflow_scales &&
                               entry.d_scale < fp8_overflow_scales + fp8_overflow_count;
            if (!in_migrated && !in_overflow) cudaFree(entry.d_scale);
        }
    }
    fp8.clear();
    fp8_bytes = 0;

    if (fp8_migrated_scales) {
        IMP_CUDA_CHECK_LOG(cudaFree(fp8_migrated_scales));
        fp8_migrated_scales = nullptr;
        fp8_migrated_count = 0;
    }
    if (fp8_migrated_data) {
        vram_free_wc(alloc, fp8_migrated_data);
        fp8_migrated_data = nullptr;
        fp8_migrated_data_size = 0;
    }
    if (fp8_overflow_scales) {
        IMP_CUDA_CHECK_LOG(cudaFree(fp8_overflow_scales));
        fp8_overflow_scales = nullptr;
        fp8_overflow_count = 0;
    }
    if (fp8_overflow_data) {
        vram_free_wc(alloc, fp8_overflow_data);
        fp8_overflow_data = nullptr;
        fp8_overflow_data_size = 0;
    }
}

} // namespace imp
