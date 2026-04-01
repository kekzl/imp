#include "model/model.h"
#include "model/model_arch.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <unordered_map>

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
    // Unmap split shard files first
    for (auto& [ptr, sz] : split_mmaps_) {
        if (ptr && sz > 0) munmap(ptr, sz);
    }
    split_mmaps_.clear();

    if (mmap_base_ && mmap_size_ > 0) {
        munmap(mmap_base_, mmap_size_);
        mmap_base_ = nullptr;
        mmap_size_ = 0;
    }
#endif
}

void Model::release_gpu_allocation(void* ptr) {
    if (!ptr) return;
    auto it = std::find(gpu_allocations_.begin(), gpu_allocations_.end(), ptr);
    if (it != gpu_allocations_.end()) {
        gpu_allocations_.erase(it);
    }
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
        case ModelArch::QWEN35:         return "qwen35";
        case ModelArch::QWEN35_MOE:     return "qwen35moe";
        case ModelArch::GEMMA3:         return "gemma3";
        case ModelArch::LLAMA4:         return "llama4";
        case ModelArch::GENERIC:        return "generic";
    }
    return "unknown";
}

ModelArch parse_model_arch(const std::string& s) {
    static const std::unordered_map<std::string, ModelArch> registry = {
        // GGUF architecture strings
        {"llama", ModelArch::LLAMA},
        {"mistral", ModelArch::MISTRAL},
        {"mixtral", ModelArch::MIXTRAL},
        {"deepseek", ModelArch::DEEPSEEK},
        {"deepseek2", ModelArch::DEEPSEEK},
        {"nemotron_h_moe", ModelArch::NEMOTRON_H_MOE},
        {"qwen3", ModelArch::QWEN3},
        {"qwen3moe", ModelArch::QWEN3_MOE},
        {"qwen35", ModelArch::QWEN35},
        {"qwen35moe", ModelArch::QWEN35_MOE},
        {"gemma3", ModelArch::GEMMA3},
        {"gemma", ModelArch::GEMMA3},
        {"gemma2", ModelArch::GEMMA3},
        {"llama4", ModelArch::LLAMA4},
        {"qwen2", ModelArch::LLAMA},
        {"phi3", ModelArch::LLAMA},
        // HuggingFace architecture class names (from config.json "architectures")
        {"LlamaForCausalLM", ModelArch::LLAMA},
        {"MistralForCausalLM", ModelArch::MISTRAL},
        {"MixtralForCausalLM", ModelArch::MIXTRAL},
        {"Qwen2ForCausalLM", ModelArch::QWEN3},
        {"Qwen2MoeForCausalLM", ModelArch::QWEN3_MOE},
        {"Gemma2ForCausalLM", ModelArch::GEMMA3},
        {"GemmaForCausalLM", ModelArch::GEMMA3},
        {"Gemma3ForCausalLM", ModelArch::GEMMA3},
        {"DeepseekV2ForCausalLM", ModelArch::DEEPSEEK},
        {"DeepseekV3ForCausalLM", ModelArch::DEEPSEEK},
        {"PhiForCausalLM", ModelArch::LLAMA},
        {"Phi3ForCausalLM", ModelArch::LLAMA},
        {"Phi3SmallForCausalLM", ModelArch::LLAMA},
        {"InternLM2ForCausalLM", ModelArch::LLAMA},
        {"Starcoder2ForCausalLM", ModelArch::LLAMA},
        {"CohereForCausalLM", ModelArch::LLAMA},
    };
    auto it = registry.find(s);
    return (it != registry.end()) ? it->second : ModelArch::GENERIC;
}

void apply_arch_defaults(ModelConfig& cfg) {
    switch (cfg.arch) {
        case ModelArch::LLAMA:
        case ModelArch::LLAMA4:
        case ModelArch::MISTRAL:
        case ModelArch::MIXTRAL:
            // LLaMA/Mistral use interleaved RoPE (2i, 2i+1), not NeoX split (i, i+d/2)
            cfg.rope_neox = false;
            break;
        case ModelArch::GEMMA3:
            // Gemma-3 uses NeoX/split RoPE (default rope_neox=true is correct)
            cfg.embed_scale = std::sqrt(static_cast<float>(cfg.d_model));
            cfg.ffn_activation = FFNActivation::GEGLU;
            // Gemma-3 uses sandwich norm: pre-norm (attn_norm/ffn_norm) AND
            // post-norm (post_attention_norm/post_ffw_norm). The POST_NORM
            // flag activates the FP32 residual accumulator for stability.
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
