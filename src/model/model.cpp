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
    // Free all GPU-side weight buffers (allocated via cudaMallocAsync).
    //
    // #834: these MUST be freed with cudaFreeAsync, not cudaFree. On this
    // stack (CUDA 13.3 / WSL2) cudaFree on a stream-ordered allocation
    // returns cudaSuccess but does NOT return the block to the async mempool
    // — the pool's used counter keeps counting it, cudaMemPoolTrimTo can
    // reclaim nothing, and every model unload leaks weights-sized VRAM
    // (~8.3 GiB per Qwen3-8B-Q8_0 cycle; the reload test's cycle-2 audit
    // showed pool used=16600 MiB > reserved=8320 MiB). cudaFreeAsync on the
    // legacy stream + a sync retires the frees so the trim in
    // imp_model_free/imp_context_free actually hands the memory back.
    //
    // At program exit the CUDA runtime may tear down the default mempool
    // before this destructor runs, making the frees return an error.
    // Silently ignore — the driver reclaims all device memory on context
    // destroy.
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
    // Return the freed weights to the driver. The default cudaMallocAsync pool
    // runs with an unbounded release threshold (engine init), so cudaFreeAsync
    // above only parks weights-sized memory in the pool — the next plain
    // cudaMalloc (e.g. a subsequent model's token-embedding upload) can't see
    // it and OOMs. Previously this trim only ran at the C-API teardown boundary
    // (imp_model_free), so direct-C++ model teardown (tests, embedders) leaked
    // the reservation. Trim here so ANY model destruction returns it.
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
        // Every caller releases a source weight tensor it is about to cudaFree
        // (Phase-3 MoE expert-source drop, Phase-4b redundant-source drop). Once
        // a source is freed the model can no longer rebuild caches for a SECOND
        // engine on this handle — mark it so Engine::init rejects that cleanly
        // instead of reading freed memory and poisoning the CUDA context (#830).
        sources_consumed_ = true;
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
    // ewnorm=true: gpt-oss routes softmax-after-topk — selection on biased
    // logits + renormalized top-k softmax is exactly imp's norm_weights path
    // once the router bias is added to the logits (compute_moe_routing).
    // embed_scale=2^-4: residual-stream rescale — gpt-oss's massive BF16
    // activations overflow the FP16 hidden state (±inf by L23). All other
    // h-contributors (Wo, o_bias, expert down + bias) are scaled to match
    // at load; RMSNorm scale-invariance makes the model output exact.
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

// Arch families empirically verified safe to honor a model author's
// kv_cache_quant_algo=FP8 hint BY DEFAULT (kv_cache.dtype=auto, no explicit
// --kv-fp8). Measured 2026-06-13 on a 3.9k-token context, FP8 vs FP16 KV, same
// corpus, RTX 5090:
//   QWEN3     (Qwen3-14B-NVFP4, dense):  PPL 13.95 -> 14.10  (+1.07%), coherent
//   QWEN3_MOE (Qwen3-30B-A3B-NVFP4):     PPL ~16.20 -> ~15.99 (neutral), coherent
//   LLAMA     (Phi-4-reasoning-plus-NVFP4, llama arch): PPL 7.135 -> 7.153
//             (+0.25%), measured 2026-06-23 on a 3.8k-token context. Vanilla
//             Llama checkpoints do not declare the FP8 hint, so honoring it for
//             the LLAMA arch only affects Phi-4-style models whose checkpoint
//             opts in via kv_cache_quant_algo=FP8.
//   NEMOTRON_H_MOE (Nemotron-3-Nano-30B): FP8-KV mean PPL 4.2774 vs FP16 4.2774
//             = ~0.00% over a 26.5k-token corpus (5 runs each), measured 2026-06-29.
//             The earlier 2026-06-23 single A/B read +0.47%..+1.83%, but that was
//             run-to-run noise: this MoE+Mamba2 hybrid is nondeterministic even in
//             deterministic mode (cross-context GDN), and the ~0.9% spread per dtype
//             swamps the dtype effect — the multi-run means coincide. Only 6/52
//             layers carry a KV cache (the rest are SSM), so the FP8 surface is small.
// Measured 2026-06-23 but NOT added (same gate):
//   GEMMA4 (Gemma-4-26B-A4B-NVFP4): baseline PPL on this corpus is broken (~243),
//     can't gate. QWEN36MOE (Qwen3.6-35B-A3B): quality fine (-0.97%) but the
//     checkpoint does not declare the hint, so allowlisting is moot.
// Other families stay FP16 until measured; force FP8 explicitly with --kv-fp8.
// This list IS the long-context quality gate — keep it conservative, extend only
// with evidence.
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

// Arch families verified safe for default FP8 KV WITHOUT a checkpoint hint.
// GGUF exports never declare kv_cache_quant_algo, so the hint gate alone left
// every GGUF model on FP16 KV — measured −39%..−41% decode at 16k context
// (Qwen3-8B Q8_0: 150→211 tok/s with FP8 KV, 2026-07-12). Evidence bar is
// stricter than the hint list (the author did not opt in): teacher-forced PPL
// over a ~13.5k-token corpus at max_seq_len 16384, FP8 vs FP16 KV, fresh
// process per run, measured 2026-07-12:
//   QWEN3     Qwen3-8B Q8_0    11.0906 → 11.0962 (+0.05%, bit-stable ×3 runs)
//             Qwen3-14B Q6_K    9.4706 →  9.4797 (+0.10%, bit-stable ×2)
//   QWEN3_MOE Qwen3-30B Q4_K_M  mean 11.506 → 11.489 (−0.15%, 3 runs/arm —
//             inside the ±0.17% MoE run-to-run spread; neutral)
// Measured but NOT added (same session):
//   QWEN36_MOE: GGUF 35B UD-Q4KM is neutral (−0.15%, 3 runs/arm), but the
//     no-hint list is arch-wide and would also flip Qwen3.6-35B-NVFP4, where
//     FP8 KV costs a REAL +1.47% PPL (13.746→13.948 mean, arms non-overlapping
//     across 2 runs each — NVFP4 attention weights compound with FP8 KV).
//     Hybrids also have a small KV surface (most layers are GDN), so the
//     long-context decode upside is small; not worth the NVFP4 quality cost.
//   LLAMA: Llama-3.2-3B Q8_0 reads +0.87%, but its baseline PPL on the gate
//     corpus is broken-high (252 — same class as the GEMMA4 ~243 exclusion),
//     so the measurement doesn't clear the evidence bar. Phi-4-style LLAMA
//     checkpoints that DO declare the hint still upgrade via the hint list.
bool kv_fp8_no_hint_default_safe(ModelArch arch) {
    switch (arch) {
        case ModelArch::QWEN3:
        case ModelArch::QWEN3_MOE:
            return true;
        default:
            return false;
    }
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

    // Unrecognised architecture → GENERIC, which loads and produces
    // plausible-looking output rather than an error (#1206). That fallback is
    // deliberate — the registry above maps 30 strings onto known archs and a
    // Llama-shaped checkpoint usually works through it — but it used to be
    // completely silent, so a genuinely unsupported model looked supported.
    // is_encoder_only_arch() guards the one family where the failure was loud
    // (BERT encoders: IMA on the first request, #818); everything else just
    // ran. Say which string was not recognised, once, at parse time.
    if (!s.empty())
        IMP_LOG_WARN(
            "Unrecognised architecture '%s' — loading as GENERIC (Llama-shaped decoder). "
            "Output may be wrong if the checkpoint needs arch-specific handling.",
            s.c_str());
    return ModelArch::GENERIC;
}

bool is_encoder_only_arch(const std::string& s) {
    // Matches BOTH naming conventions: llama.cpp GGUF arch ids (lowercase,
    // e.g. "nomic-bert", "jina-bert-v2", "roberta") AND HuggingFace config.json
    // `architectures` class names (CamelCase, e.g. "NomicBertModel",
    // "BertModel", "XLMRobertaModel", "MPNetModel"). The GGUF path fed lowercase
    // ids; the SafeTensors/HF path feeds CamelCase, so a case-sensitive
    // find("bert") missed "NomicBertModel" and let it fall through to the
    // generic decoder → CUDA IMA on the first request (#818). Lowercase first.
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
