#include "model/model.h"
#include "model/model_arch.h"
#include <cuda_runtime.h>
#include <cmath>

#ifdef __linux__
#include <sys/mman.h>
#endif

namespace imp {

Model::~Model() {
    // Free all GPU-side weight buffers.
    for (void* ptr : gpu_allocations_) {
        if (ptr) {
            cudaFree(ptr);
        }
    }
    gpu_allocations_.clear();

    // Unpin host-registered expert weight regions before munmap.
    for (void* ptr : host_pinned_) {
        if (ptr) {
            cudaHostUnregister(ptr);
        }
    }
    host_pinned_.clear();

    // Free cudaHostAlloc'd expert buffers (WSL2 DMA path).
    for (void* ptr : host_pinned_allocs_) {
        if (ptr) {
            cudaFreeHost(ptr);
        }
    }
    host_pinned_allocs_.clear();

    gpu_weights_ready_ = false;

#ifdef __linux__
    if (mmap_base_ && mmap_size_ > 0) {
        munmap(mmap_base_, mmap_size_);
        mmap_base_ = nullptr;
        mmap_size_ = 0;
    }
#endif
}

const char* model_arch_name(ModelArch arch) {
    switch (arch) {
        case ModelArch::LLAMA:    return "llama";
        case ModelArch::MISTRAL:  return "mistral";
        case ModelArch::MIXTRAL:  return "mixtral";
        case ModelArch::DEEPSEEK:       return "deepseek";
        case ModelArch::NEMOTRON_H_MOE: return "nemotron_h_moe";
        case ModelArch::QWEN3:          return "qwen3";
        case ModelArch::QWEN3_MOE:      return "qwen3moe";
        case ModelArch::GEMMA3:         return "gemma3";
        case ModelArch::GENERIC:        return "generic";
    }
    return "unknown";
}

ModelArch parse_model_arch(const std::string& s) {
    if (s == "llama")                        return ModelArch::LLAMA;
    if (s == "mistral")                      return ModelArch::MISTRAL;
    if (s == "mixtral")                      return ModelArch::MIXTRAL;
    if (s == "deepseek" || s == "deepseek2") return ModelArch::DEEPSEEK;
    if (s == "nemotron_h_moe")               return ModelArch::NEMOTRON_H_MOE;
    if (s == "qwen3")                        return ModelArch::QWEN3;
    if (s == "qwen3moe")                     return ModelArch::QWEN3_MOE;
    if (s == "gemma3")                       return ModelArch::GEMMA3;
    if (s == "gemma" || s == "gemma2")       return ModelArch::GEMMA3;  // treat all gemma as gemma3
    return ModelArch::GENERIC;
}

void apply_arch_defaults(ModelConfig& cfg) {
    switch (cfg.arch) {
        case ModelArch::GEMMA3:
            cfg.embed_scale = std::sqrt(static_cast<float>(cfg.d_model));
            cfg.ffn_activation = FFNActivation::GEGLU;
            cfg.norm_placement = NormPlacement::POST_NORM;
            break;
        case ModelArch::NEMOTRON_H_MOE:
            cfg.moe_sigmoid_gating = true;
            cfg.ffn_activation = FFNActivation::RELU_SQR;
            break;
        case ModelArch::QWEN3_MOE:
            cfg.expert_weights_norm = true;
            break;
        default:
            break;
    }
}

} // namespace imp
