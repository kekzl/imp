#include "model/model.h"
#include "model/model_arch.h"
#include "core/logging.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <unordered_map>

#ifdef __linux__
#include <sys/mman.h>
#endif

namespace imp {

Model::~Model() {
    // Free all GPU-side weight buffers (allocated via cudaMallocAsync).
    // At program exit the CUDA runtime may tear down the default mempool
    // before this destructor runs, making cudaFree return cudaErrorInvalidValue.
    // Silently ignore — the driver reclaims all device memory on context destroy.
    for (void* ptr : gpu_allocations_) {
        if (ptr)
            (void)cudaFree(ptr);
    }
    gpu_allocations_.clear();

    for (void* ptr : host_pinned_) {
        if (ptr)
            (void)cudaHostUnregister(ptr);
    }
    host_pinned_.clear();

    for (void* ptr : host_pinned_allocs_) {
        if (ptr)
            (void)cudaFreeHost(ptr);
    }
    host_pinned_allocs_.clear();

    // Free heap-allocated permuted weight buffers (Qwen3.5/3.6 GDN reorder).
    for (void* ptr : host_owned_buffers_) {
        if (ptr)
            std::free(ptr);
    }
    host_owned_buffers_.clear();

    gpu_weights_ready_ = false;

#ifdef __linux__
    // Unmap split shard files first
    for (auto& [ptr, sz] : split_mmaps_) {
        if (ptr && sz > 0)
            munmap(ptr, sz);
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
    if (!ptr)
        return;
    auto it = std::find(gpu_allocations_.begin(), gpu_allocations_.end(), ptr);
    if (it != gpu_allocations_.end()) {
        gpu_allocations_.erase(it);
    }
}

bool Model::is_base_gpu_allocation(void* ptr) const {
    if (!ptr)
        return false;
    return std::find(gpu_allocations_.begin(), gpu_allocations_.end(), ptr) != gpu_allocations_.end();
}

// ---------------------------------------------------------------------------
// Architecture registry — single source of truth for all per-arch metadata.
// Replaces scattered switch statements in model.cpp, chat_template.cpp,
// presets.cpp, and imp_api.cpp (RF-003).
// ---------------------------------------------------------------------------
struct ArchEntry {
    ModelArch arch;
    const char* name;
    int c_api_id;  // IMP_ARCH_* enum value

    // Config defaults
    int rope_neox;       // -1 = don't override, 0 = false, 1 = true
    float embed_scale;   // 0 = don't override
    int ffn_activation;  // -1 = don't override, else FFNActivation cast
    int norm_placement;  // -1 = don't override, else NormPlacement cast
    bool moe_sigmoid_gating;
    bool expert_weights_norm;

    // Sampling defaults
    float temperature;
    float top_p;
    int top_k;
};

// IMP_ARCH_* values from include/imp/types.h (avoid header dependency)
enum {
    kApiLlama = 0,
    kApiMistral = 1,
    kApiMixtral = 2,
    kApiDeepseek = 3,
    kApiNemotronHMoe = 4,
    kApiQwen3 = 5,
    kApiQwen3Moe = 6,
    kApiGemma3 = 7,
    kApiLlama4 = 8,
    kApiGeneric = 9,
    kApiQwen35 = 10,
    kApiQwen35Moe = 11,
    kApiGemma4 = 12,
    kApiQwen36Moe = 13,
};

static constexpr ArchEntry kArchRegistry[] = {
    // arch                      name              c_api    rope  embed  ffn  norm  sigm  ewnorm  temp  top_p
    // top_k
    {ModelArch::LLAMA, "llama", kApiLlama, 0, 0, -1, -1, false, false, 0.6f, 0.95f, 0},
    {ModelArch::MISTRAL, "mistral", kApiMistral, 0, 0, -1, -1, false, false, 0.6f, 0.95f, 0},
    {ModelArch::MIXTRAL, "mixtral", kApiMixtral, 0, 0, -1, -1, false, false, 0.6f, 0.95f, 0},
    {ModelArch::DEEPSEEK, "deepseek", kApiDeepseek, -1, 0, -1, -1, false, false, 0.6f, 0.95f, 0},
    {ModelArch::NEMOTRON_H_MOE, "nemotron_h_moe", kApiNemotronHMoe, -1, 0, 2, -1, true, false, 0.6f, 0.95f,
     0},
    {ModelArch::QWEN3, "qwen3", kApiQwen3, -1, 0, -1, -1, false, false, 0.6f, 0.95f, 20},
    {ModelArch::QWEN3_MOE, "qwen3moe", kApiQwen3Moe, -1, 0, -1, -1, false, true, 0.6f, 0.95f, 20},
    {ModelArch::QWEN35, "qwen35", kApiQwen35, -1, 0, -1, -1, false, false, 0.6f, 0.95f, 20},
    {ModelArch::QWEN35_MOE, "qwen35moe", kApiQwen35Moe, -1, 0, -1, -1, false, true, 0.6f, 0.95f, 20},
    {ModelArch::QWEN36_MOE, "qwen36moe", kApiQwen36Moe, -1, 0, -1, -1, false, true, 0.6f, 0.95f, 20},
    {ModelArch::GEMMA3, "gemma3", kApiGemma3, -1, 0, 1, 1, false, false, 0.6f, 0.95f, 0},
    {ModelArch::GEMMA4, "gemma4", kApiGemma4, -1, 0, 1, 1, false, true, 0.6f, 0.9f, 20},
    {ModelArch::LLAMA4, "llama4", kApiLlama4, 0, 0, -1, -1, false, false, 0.6f, 0.95f, 0},
    {ModelArch::GENERIC, "generic", kApiGeneric, -1, 0, -1, -1, false, false, 0.6f, 0.95f, 0},
};

static const ArchEntry& lookup_arch(ModelArch arch) {
    for (const auto& e : kArchRegistry)
        if (e.arch == arch)
            return e;
    return kArchRegistry[sizeof(kArchRegistry) / sizeof(kArchRegistry[0]) - 1];  // GENERIC
}

const char* model_arch_name(ModelArch arch) { return lookup_arch(arch).name; }

int model_arch_c_api_id(ModelArch arch) { return lookup_arch(arch).c_api_id; }

void model_arch_sampling_defaults(ModelArch arch, float& temperature, float& top_p, int& top_k) {
    const auto& e = lookup_arch(arch);
    temperature = e.temperature;
    top_p = e.top_p;
    top_k = e.top_k;
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
        {"qwen36moe", ModelArch::QWEN36_MOE},
        {"qwen3.6_moe", ModelArch::QWEN36_MOE},
        {"qwen3.6moe", ModelArch::QWEN36_MOE},
        {"gemma3", ModelArch::GEMMA3},
        {"gemma", ModelArch::GEMMA3},
        {"gemma2", ModelArch::GEMMA3},
        {"gemma4", ModelArch::GEMMA4},
        {"llama4", ModelArch::LLAMA4},
        {"qwen2", ModelArch::LLAMA},
        {"phi3", ModelArch::LLAMA},
        // HuggingFace architecture class names (from config.json "architectures")
        {"LlamaForCausalLM", ModelArch::LLAMA},
        {"MistralForCausalLM", ModelArch::MISTRAL},
        {"MixtralForCausalLM", ModelArch::MIXTRAL},
        {"Qwen2ForCausalLM", ModelArch::QWEN3},
        {"Qwen2MoeForCausalLM", ModelArch::QWEN3_MOE},
        {"Qwen3ForCausalLM", ModelArch::QWEN3},
        {"Qwen3MoeForCausalLM", ModelArch::QWEN3_MOE},
        {"Qwen3_5ForCausalLM", ModelArch::QWEN35},
        {"Qwen3_5ForConditionalGeneration", ModelArch::QWEN35},
        {"Qwen3_5MoeForCausalLM", ModelArch::QWEN36_MOE},
        {"Qwen3_5MoeForConditionalGeneration", ModelArch::QWEN36_MOE},
        {"NemotronHForCausalLM", ModelArch::NEMOTRON_H_MOE},
        {"Gemma2ForCausalLM", ModelArch::GEMMA3},
        {"GemmaForCausalLM", ModelArch::GEMMA3},
        {"Gemma3ForCausalLM", ModelArch::GEMMA3},
        {"Gemma3ForConditionalGeneration", ModelArch::GEMMA3},
        {"Gemma4ForCausalLM", ModelArch::GEMMA4},
        {"Gemma4ForConditionalGeneration", ModelArch::GEMMA4},
        {"DeepseekV2ForCausalLM", ModelArch::DEEPSEEK},
        {"DeepseekV3ForCausalLM", ModelArch::DEEPSEEK},
        {"Llama4ForCausalLM", ModelArch::LLAMA4},
        {"Llama4ForConditionalGeneration", ModelArch::LLAMA4},
        {"MistralForCausalLM", ModelArch::MISTRAL},
        {"Mistral3ForConditionalGeneration", ModelArch::MISTRAL},
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
    const auto& e = lookup_arch(cfg.arch);
    if (e.rope_neox >= 0)
        cfg.rope_neox = (e.rope_neox != 0);
    if (e.embed_scale > 0)
        cfg.embed_scale = e.embed_scale;
    // Gemma-3/4 computes embed_scale from d_model
    if (cfg.arch == ModelArch::GEMMA3 || cfg.arch == ModelArch::GEMMA4)
        cfg.embed_scale = std::sqrt(static_cast<float>(cfg.d_model));
    // Gemma norm weights store (1 + learned) directly: the GGUF converter bakes
    // the +1 into every *norm.weight (see gguf_loader.cpp "already baked"), so the
    // runtime must use the weight as-is (offset 0), same as Gemma-4 and Qwen3.5.
    // Verified on gemma-3-12b-it-Q4_K_M: attn_norm min≈0.99 (== learned≈0 + 1), not
    // centered at 0. Applying a +1 offset here double-counts → garbage output.
    if (e.ffn_activation >= 0)
        cfg.ffn_activation = static_cast<FFNActivation>(e.ffn_activation);
    if (e.norm_placement >= 0)
        cfg.norm_placement = static_cast<NormPlacement>(e.norm_placement);
    if (e.moe_sigmoid_gating)
        cfg.moe_sigmoid_gating = true;
    if (e.expert_weights_norm)
        cfg.expert_weights_norm = true;

    // Nemotron-H family attention is NoPE: the Mamba layers carry position,
    // the attention layers are trained WITHOUT rotary embeddings. The HF
    // config still ships rope_theta=10000 / partial_rotary_factor=1.0 (class
    // defaults), and applying RoPE scrambles positional binding — the model
    // reads prompts as a bag of words ("17 + 25" → "25 + 17"/"15 + 7";
    // "ALPHA BRAVO CHARLIE" → hallucinated different prompt). Verified by
    // disabling rotation (rope_theta=1e12 config patch): prompt reading
    // becomes exact.
    if (cfg.arch == ModelArch::NEMOTRON_H_MOE)
        cfg.rope_attn_disabled = true;
}

}  // namespace imp
