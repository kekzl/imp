#include "model/model.h"
#include "model/model_arch.h"
#include "core/logging.h"
#include "memory/mem_account.h"  // trim_device_mempool
#include "memory/weight_snapshot.h"
// Complete type needed here: ~Model destroys the unique_ptr<VisionModel>.
#include "vision/vision_model.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <unordered_map>

#ifdef __linux__
#include <sys/mman.h>
#endif

namespace imp {

Model::Model() = default;

Model::~Model() {
    // #834: free with cudaFreeAsync, not cudaFree - on this stack (CUDA 13.3/WSL2) cudaFree on
    // a stream-ordered alloc returns success without returning the block to the pool, leaking
    // weights-sized VRAM per unload. At exit the runtime may tear down the pool before this
    // destructor runs; ignore errors then, the driver reclaims all device memory on context destroy.
    const bool had_gpu_weights = !gpu_allocations_.empty();
    int free_fail = 0;
    cudaError_t first_err = cudaSuccess;
    for (void* ptr : gpu_allocations_) {
        if (ptr) {
            cudaError_t e = cudaFreeAsync(ptr, nullptr);
            if (e != cudaSuccess) {
                if (free_fail == 0)
                    first_err = e;
                free_fail++;
            }
        }
    }
    if (had_gpu_weights)
        (void)cudaStreamSynchronize(nullptr);  // retire the frees for the pool
    if (free_fail > 0) {
        IMP_LOG_WARN("Model teardown: %d/%zu weight frees failed (first: %s)", free_fail,
                     gpu_allocations_.size(), cudaGetErrorString(first_err));
        (void)cudaGetLastError();  // clear sticky error (e.g. process-exit teardown)
    }
    gpu_allocations_.clear();
    // Returns freed weights to the driver: the default cudaMallocAsync pool has an unbounded
    // release threshold, so cudaFreeAsync only parks memory - the next plain cudaMalloc can't
    // see it and OOMs. Trim here so ANY model destruction returns it, not just C-API teardown.
    if (had_gpu_weights)
        trim_device_mempool();

    host_pinned_.clear();  // HostRegistration un-registers each entry

    host_pinned_allocs_.clear();  // PinnedBuffer owns each entry

    // Free heap-allocated permuted weight buffers (Qwen3.5/3.6 GDN reorder).
    for (void* ptr : host_owned_buffers_) {
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

size_t Model::release_weight_pages() {
#ifdef __linux__
    size_t advised = 0;
    auto drop = [&advised](void* p, size_t n) {
        if (!p || n == 0)
            return;
        // Page-aligned by construction (mmap). MADV_DONTNEED on a PROT_READ MAP_PRIVATE mapping
        // discards resident pages only: a later read refaults from the page cache, so every
        // pointer into the mapping stays valid and correct.
        if (madvise(p, n, MADV_DONTNEED) == 0)
            advised += n;
    };
    drop(mmap_base_, mmap_size_);
    for (auto& [ptr, sz] : split_mmaps_)
        drop(ptr, sz);
    return advised;
#else
    return 0;
#endif
}

void Model::release_gpu_allocation(void* ptr) {
    if (!ptr)
        return;
    auto it = std::find(gpu_allocations_.begin(), gpu_allocations_.end(), ptr);
    if (it != gpu_allocations_.end()) {
        gpu_allocations_.erase(it);
        // Suspend-to-RAM: this source is about to be freed — its snapshot
        // record is no longer capturable (the tensor re-uploads cold at resume).
        if (upload_log_)
            upload_log_->evict_ptr(ptr);
        // Every caller releases a source weight tensor about to cudaFree (Phase-3 MoE
        // expert-source drop, Phase-4b redundant-source drop). Marks so Engine::init rejects a
        // second engine on this handle instead of reading freed memory and poisoning CUDA (#830).
        sources_consumed_ = true;
    }
}

bool Model::is_base_gpu_allocation(void* ptr) const {
    if (!ptr)
        return false;
    return std::find(gpu_allocations_.begin(), gpu_allocations_.end(), ptr) != gpu_allocations_.end();
}

// Architecture registry: single source of truth for per-arch metadata, replacing scattered
// switch statements in model.cpp/chat_template.cpp/presets.cpp/imp_api.cpp (RF-003).
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
    kApiGptOss = 14,
    kApiNomicBert = 15,
};

static constexpr ArchEntry kArchRegistry[] = {
    // arch                      name              c_api    rope  embed  ffn  norm  sigm  ewnorm  temp  top_p
    // top_k
    {ModelArch::LLAMA, "llama", kApiLlama, 0, 0, -1, -1, false, false, 0.6f, 0.95f, 0},
    {ModelArch::MISTRAL, "mistral", kApiMistral, 0, 0, -1, -1, false, false, 0.6f, 0.95f, 0},
    {ModelArch::MIXTRAL, "mixtral", kApiMixtral, 0, 0, -1, -1, false, false, 0.6f, 0.95f, 0},
    // rope=0 (interleaved): DeepSeek-V2/V3 apply_rotary_pos_emb deinterleaves
    // q_pe/k_pe (view[d/2,2].transpose) before NeoX rotate_half, so the effective
    // decoupled-RoPE rotation is on interleaved pairs (2i,2i+1), NOT NeoX (i,i+d/2).
    {ModelArch::DEEPSEEK, "deepseek", kApiDeepseek, 0, 0, -1, -1, false, false, 0.6f, 0.95f, 0},
    {ModelArch::NEMOTRON_H_MOE, "nemotron_h_moe", kApiNemotronHMoe, -1, 0, 2, -1, true, false, 0.6f, 0.95f,
     0},
    {ModelArch::QWEN3, "qwen3", kApiQwen3, -1, 0, -1, -1, false, false, 0.6f, 0.95f, 20},
    {ModelArch::QWEN3_MOE, "qwen3moe", kApiQwen3Moe, -1, 0, -1, -1, false, true, 0.6f, 0.95f, 20},
    {ModelArch::QWEN35, "qwen35", kApiQwen35, -1, 0, -1, -1, false, false, 0.6f, 0.95f, 20},
    {ModelArch::QWEN35_MOE, "qwen35moe", kApiQwen35Moe, -1, 0, -1, -1, false, true, 0.6f, 0.95f, 20},
    {ModelArch::QWEN36_MOE, "qwen36moe", kApiQwen36Moe, -1, 0, -1, -1, false, true, 0.6f, 0.95f, 20},
    // ewnorm=true: gpt-oss routes softmax-after-topk - selection on biased logits +
    // renormalized top-k softmax equals imp's norm_weights path once router bias is added to
    // logits. embed_scale=2^-4: residual rescale for gpt-oss's BF16 activations overflowing
    // FP16 hidden (inf by L23); all h-contributors scaled to match, RMSNorm keeps output exact.
    {ModelArch::GPT_OSS, "gpt_oss", kApiGptOss, 1, 0.0625f, 3 /*GPT_OSS_GLU*/, -1, false, true, 1.0f, 1.0f,
     0},
    {ModelArch::GEMMA3, "gemma3", kApiGemma3, -1, 0, 1, 1, false, false, 0.6f, 0.95f, 0},
    {ModelArch::GEMMA4, "gemma4", kApiGemma4, -1, 0, 1, 1, false, true, 0.6f, 0.9f, 20},
    {ModelArch::LLAMA4, "llama4", kApiLlama4, 0, 0, -1, -1, false, false, 0.6f, 0.95f, 0},
    // Encoder-only embedder (#836): rotary (neox pairs), SwiGLU FFN; sampling
    // defaults unused (no LM head). Post-LN + pooling handled by the encoder
    // forward, not the decoder loop.
    {ModelArch::NOMIC_BERT, "nomic-bert", kApiNomicBert, 1, 0, -1, -1, false, false, 1.0f, 1.0f, 0},
    {ModelArch::GENERIC, "generic", kApiGeneric, -1, 0, -1, -1, false, false, 0.6f, 0.95f, 0},
};

static const ArchEntry& lookup_arch(ModelArch arch) {
    for (const auto& e : kArchRegistry)
        if (e.arch == arch)
            return e;
    return kArchRegistry[sizeof(kArchRegistry) / sizeof(kArchRegistry[0]) - 1];  // GENERIC
}

const char* model_arch_name(ModelArch arch) { return lookup_arch(arch).name; }

// Arch families verified safe to honor the checkpoint's kv_cache_quant_algo=FP8 hint by
// default (kv_cache.dtype=auto): QWEN3, QWEN3_MOE, LLAMA (Phi-4-style hint-declaring
// checkpoints only), NEMOTRON_H_MOE (small KV surface, mostly SSM layers). This list IS
// the long-context quality gate: keep conservative, extend only with fresh evidence.
bool kv_fp8_hint_default_safe(ModelArch arch) {
    switch (arch) {
        case ModelArch::QWEN3:
        case ModelArch::QWEN3_MOE:
        case ModelArch::LLAMA:
        case ModelArch::NEMOTRON_H_MOE:
            return true;
        default:
            return false;
    }
}

// Arch families verified safe for default FP8 KV even with no checkpoint hint (GGUF never
// declares the hint; FP16 KV at 16k context costs ~-40% decode). Stricter bar than the
// hint list (author didn't opt in): QWEN3, QWEN3_MOE only. QWEN36_MOE and LLAMA measured
// but excluded: NVFP4 compounds with FP8 KV / baseline PPL is broken-high.
bool kv_fp8_no_hint_default_safe(ModelArch arch) {
    switch (arch) {
        case ModelArch::QWEN3:
        case ModelArch::QWEN3_MOE:
            return true;
        default:
            return false;
    }
}

// Arch families measured safe for default NVFP4 KV: a capacity gate (shrinks the KV cache
// that bounds max_seq_len), not a speed one - distinct bar from the FP8 lists above.
// QWEN35 only: ~+0.3% PPL cost buys ~2.7x max_model_len on a GDN hybrid's small
// attention-only KV surface. kv_cache.dtype=fp16 opts out; QWEN36_MOE/QWEN35_MOE stay off
// (FP8 KV already costs them +1.47% PPL there; NVFP4 KV unmeasured for them).
bool kv_nvfp4_default_safe(ModelArch arch) {
    switch (arch) {
        case ModelArch::QWEN35:
            return true;
        default:
            return false;
    }
}

int kv_pin_context_cost_factor(ModelArch arch, QType pinned) {
    // Only the NVFP4-default families have anything to lose: on every other
    // arch `auto` lands on FP8 or FP16 already.
    if (!kv_nvfp4_default_safe(arch))
        return 0;
    switch (pinned) {
        case QType::F16:
        case QType::BF16:
            return 4;  // 16 bit per element against NVFP4's 4
        case QType::FP8_E4M3:
        case QType::FP8_E5M2:
        case QType::INT8:
            return 2;
        default:
            return 0;  // NVFP4 / MXFP4_KV / INT4 - same width or narrower
    }
}

int max_seq_len_operator_value(int preset, int file_key) {
    if (preset > 0)
        return preset;
    return file_key > 0 ? file_key : 0;
}

bool kv_dtype_is_explicit_pin(QType cli_dtype, const std::string& conf_dtype) {
    if (cli_dtype != QType::F16)
        return true;  // the resolver only ever runs while this is still F16
    // Every value the resolver accepts as a choice. "auto" is not one, and an
    // unrecognised string is not either - that one already warns on its own.
    return conf_dtype == "fp16" || conf_dtype == "fp8" || conf_dtype == "int8" || conf_dtype == "int4" ||
           conf_dtype == "nvfp4" || conf_dtype == "mxfp4";
}

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
        {"gpt_oss", ModelArch::GPT_OSS},
        {"gpt-oss", ModelArch::GPT_OSS},
        {"gemma3", ModelArch::GEMMA3},
        {"gemma", ModelArch::GEMMA3},
        {"gemma2", ModelArch::GEMMA3},
        {"gemma4", ModelArch::GEMMA4},
        {"llama4", ModelArch::LLAMA4},
        {"nomic-bert", ModelArch::NOMIC_BERT},
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
    if (it != registry.end())
        return it->second;

    // Unrecognised architecture falls back to GENERIC (plausible output, not an error, #1206):
    // the registry maps known strings and a Llama-shaped checkpoint usually works through it.
    // is_encoder_only_arch() guards the one family where silent failure was loud (BERT: IMA,
    // #818); everything else just runs. Logs the unrecognised string once, at parse time.
    if (!s.empty())
        IMP_LOG_WARN(
            "Unrecognised architecture '%s' — loading as GENERIC (Llama-shaped decoder). "
            "Output may be wrong if the checkpoint needs arch-specific handling.",
            s.c_str());
    return ModelArch::GENERIC;
}

bool is_encoder_only_arch(const std::string& s) {
    // Matches both llama.cpp GGUF arch ids (lowercase, e.g. nomic-bert) and HF architectures
    // CamelCase class names (e.g. NomicBertModel). Case-sensitive find("bert") missed
    // "NomicBertModel", falling through to the generic decoder -> CUDA IMA (#818). Lowercase first.
    std::string l;
    l.reserve(s.size());
    for (char c : s)
        l += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    // No causal-decoder arch name (GGUF id or HF class) contains any of these.
    if (l.find("bert") != std::string::npos || l.find("roberta") != std::string::npos ||
        l.find("mpnet") != std::string::npos)
        return true;
    // Encoder-only T5 + embedding-family aliases (exact ids).
    return l == "t5encoder" || l == "e5" || l == "bge";
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
    // Gemma norm weights store (1+learned) directly (GGUF converter bakes +1 into every
    // *norm.weight); the runtime must use the weight as-is (offset 0), same as Gemma-4/Qwen3.5.
    // Applying a +1 offset here double-counts, producing garbage output.
    if (e.ffn_activation >= 0)
        cfg.ffn_activation = static_cast<FFNActivation>(e.ffn_activation);
    if (e.norm_placement >= 0)
        cfg.norm_placement = static_cast<NormPlacement>(e.norm_placement);
    if (e.moe_sigmoid_gating)
        cfg.moe_sigmoid_gating = true;
    if (e.expert_weights_norm)
        cfg.expert_weights_norm = true;

    // Nemotron-H attention is NoPE: the Mamba layers carry position, attention layers are
    // trained WITHOUT rotary embeddings. HF config still ships rope_theta/partial_rotary_factor
    // class defaults; applying RoPE scrambles positional binding into a bag-of-words reading.
    if (cfg.arch == ModelArch::NEMOTRON_H_MOE)
        cfg.rope_attn_disabled = true;
}

}  // namespace imp
