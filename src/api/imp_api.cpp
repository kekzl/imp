#include "api/imp_internal.h"
#include "exec/executor.h"
#include "runtime/engine.h"
#include "model/gguf_loader.h"
#include "model/safetensors_loader.h"
#include "model/tokenizer.h"
#include "model/chat_template.h"
#include "memory/kv_cache.h"

#include "core/logging.h"

#include <cstring>
#include <memory>
#include <vector>
#include <new>
#include <exception>

#include <cuda_runtime.h>

namespace {

// Return retained default-mempool blocks to the driver after a teardown.
//
// Engine init raises the default cudaMallocAsync pool's release threshold to
// UINT64_MAX so freed blocks are kept for re-use. Model weights are allocated
// from that pool (weight_upload.cu checked_cuda_malloc), so freeing a model
// parks ~weights-sized memory in the pool instead of returning it. The next
// model load can't see it: cudaMemGetInfo-based sizing (engine_init_resolver,
// the upload oversubscription gate) and plain-cudaMalloc paths only observe
// driver-free memory. Symptom: server auto-swap found ~1.5 GB free on a 32 GB
// card and failed with "Failed to upload token embedding".
void trim_default_mempool() {
    // Retire pending async frees before trimming, otherwise their blocks are
    // still referenced and survive the trim.
    cudaDeviceSynchronize();
    int dev = 0;
    cudaMemPool_t pool = nullptr;
    if (cudaGetDevice(&dev) == cudaSuccess &&
        cudaDeviceGetDefaultMemPool(&pool, dev) == cudaSuccess && pool != nullptr) {
        cudaMemPoolTrimTo(pool, 0);
    }
}

}  // namespace

// --- Error string ---

const char* imp_error_string(ImpError err) {
    switch (err) {
        case IMP_SUCCESS:
            return "success";
        case IMP_ERROR_INVALID_ARG:
            return "invalid argument";
        case IMP_ERROR_OUT_OF_MEMORY:
            return "out of memory";
        case IMP_ERROR_CUDA:
            return "CUDA error";
        case IMP_ERROR_FILE_NOT_FOUND:
            return "file not found";
        case IMP_ERROR_INVALID_MODEL:
            return "invalid model";
        case IMP_ERROR_UNSUPPORTED:
            return "unsupported operation";
        case IMP_ERROR_INTERNAL:
            return "internal error";
        case IMP_ERROR_CANCELLED:
            return "cancelled";
        default:
            return "unknown error";
    }
}

// --- Config defaults ---

ImpConfig imp_config_default(void) {
    ImpConfig config;
    config.device_id = 0;
    config.gpu_memory_pool_size = 0;  // auto
    config.kv_cache_max_blocks = 0;   // auto
    config.max_batch_size = 0;        // auto (engine detects from model size)
    config.max_seq_len = 0;           // auto (engine detects from model metadata + VRAM)
    config.compute_dtype = IMP_DTYPE_FP16;
    config.temperature = 0.6f;
    config.top_p = 0.95f;
    config.top_k = 0;
    config.max_tokens = 256;
    config.enable_green_contexts = 0;
    config.green_ctx_prefill_ratio = 0.8f;
    config.enable_pdl = 1;
    config.enable_cuda_graphs = 1;
    config.gpu_layers = -1;  // all on GPU
    config.kv_cache_dtype = IMP_DTYPE_FP16;
    config.ssm_state_dtype = IMP_DTYPE_FP32;
    config.vram_budget_mb = 0;                // use all available
    config.prefill_chunk_size = -1;           // per-arch default (512 if supported, 0 otherwise)
    config.use_fp8_prefill = 0;               // FP16 weight cache by default
    config.use_nvfp4_decode = -1;             // auto (sm_120→mode2, sm_90→mode1)
    config.use_mxfp4_prefill = 0;             // off by default
    config.dual_path_quant = 0;               // off by default
    config.min_kv_tokens = 0;                 // auto (pick reasonable minimum based on model)
    config.use_prefix_caching = 0;            // off by default
    config.prefix_cache_path[0] = '\0';       // no persistence by default
    config.prefix_pin_budget_pct = 25;        // cache_control pins capped at 25% of the KV pool
    config.num_cpu_threads = 0;               // auto
    config.mmproj_path = NULL;                // no vision model
    config.turboquant_sketch_multiplier = 2;  // sketch_dim = 2 * head_dim
    config.streaming_kv_enabled = 0;          // off by default (opt-in)
    config.streaming_kv_auto = 1;             // auto-enable when KV cache >90% full
    config.streaming_kv_n_sinks = 4;          // StreamingLLM paper default
    config.streaming_kv_window = 0;           // 0 = derive from ModelConfig::sliding_window
    config.streaming_kv_threshold = 0;        // 0 = auto
    return config;
}

// --- Generate params defaults ---

ImpGenerateParams imp_generate_params_default(void) {
    ImpGenerateParams params;
    params.temperature = 1.0f;
    params.top_p = 1.0f;
    params.top_k = 0;
    params.max_tokens = 256;
    params.seed = -1;
    params.min_p = 0.0f;
    params.typical_p = 1.0f;
    params.repetition_penalty = 1.0f;
    params.frequency_penalty = 0.0f;
    params.presence_penalty = 0.0f;
    params.repeat_last_n = 0;
    params.dry_multiplier = 0.0f;
    params.dry_base = 1.75f;
    params.dry_allowed_length = 2;
    params.dry_penalty_last_n = 0;
    params.mirostat = 0;
    params.mirostat_tau = 5.0f;
    params.mirostat_eta = 0.1f;
    params.apply_chat_template = 1;
    params.ignore_eos = 0;
    params.logprobs = 0;
    params.top_logprobs = 0;
    params.json_mode = 0;
    return params;
}

// --- Version ---

const char* imp_version(void) { return "0.10.0"; }

// --- Helper: map ImpDType to imp::QType ---

static imp::QType map_dtype(ImpDType dt) {
    switch (dt) {
        case IMP_DTYPE_FP32:
            return imp::QType::F32;
        case IMP_DTYPE_FP16:
            return imp::QType::F16;
        case IMP_DTYPE_BF16:
            return imp::QType::BF16;
        case IMP_DTYPE_FP8_E4M3:
            return imp::QType::FP8_E4M3;
        case IMP_DTYPE_FP8_E5M2:
            return imp::QType::FP8_E5M2;
        case IMP_DTYPE_INT8:
            return imp::QType::INT8;
        case IMP_DTYPE_INT4:
            return imp::QType::INT4;
        case IMP_DTYPE_INT32:
            return imp::QType::INT32;
        case IMP_DTYPE_FP4_E2M1:
            return imp::QType::FP4_E2M1;
        case IMP_DTYPE_TURBOQUANT:
            // DEPRECATED: TurboQuant retired. IMP_DTYPE_TURBOQUANT is kept for ABI
            // compatibility and silently maps to MXFP4_KV. Callers should switch to
            // IMP_DTYPE_MXFP4_KV. A one-shot warning is printed the first time.
            {
                static bool warned = false;
                if (!warned) {
                    warned = true;
                    IMP_LOG_WARN("IMP_DTYPE_TURBOQUANT is deprecated (TurboQuant retired). "
                                 "Using IMP_DTYPE_MXFP4_KV instead.");
                }
            }
            return imp::QType::MXFP4_KV;
        case IMP_DTYPE_TURBOQUANT_LITE:
            // DEPRECATED: TurboQuant Lite retired. Maps to MXFP4_KV.
            {
                static bool warned = false;
                if (!warned) {
                    warned = true;
                    IMP_LOG_WARN("IMP_DTYPE_TURBOQUANT_LITE is deprecated (TurboQuant retired). "
                                 "Using IMP_DTYPE_MXFP4_KV instead.");
                }
            }
            return imp::QType::MXFP4_KV;
        case IMP_DTYPE_NVFP4:
            return imp::QType::NVFP4;
        case IMP_DTYPE_MXFP4_KV:
            return imp::QType::MXFP4_KV;
        default:
            return imp::QType::F16;
    }
}

// --- Model Loading ---

ImpError imp_model_load_ex(const char* path, ImpModelFormat format, int load_mtp_head,
                           ImpModel* out_model) {
    if (!path || !out_model) {
        return IMP_ERROR_INVALID_ARG;
    }
    *out_model = nullptr;

    try {
        std::unique_ptr<imp::Model> model;

        switch (format) {
            case IMP_FORMAT_GGUF:
                model = imp::load_gguf(path);
                break;
            case IMP_FORMAT_SAFETENSORS:
                model = imp::load_safetensors(path, load_mtp_head != 0);
                break;
            default:
                return IMP_ERROR_INVALID_ARG;
        }

        if (!model) {
            return IMP_ERROR_FILE_NOT_FOUND;
        }

        auto handle = new (std::nothrow) ImpModel_T();
        if (!handle) {
            return IMP_ERROR_OUT_OF_MEMORY;
        }

        handle->model = std::move(model);
        *out_model = handle;
        return IMP_SUCCESS;
    } catch (const std::bad_alloc&) {
        return IMP_ERROR_OUT_OF_MEMORY;
    } catch (const std::exception& e) {
        IMP_LOG_ERROR("imp_model_load: %s", e.what());
        return IMP_ERROR_INTERNAL;
    } catch (...) {
        return IMP_ERROR_INTERNAL;
    }
}

ImpError imp_model_load(const char* path, ImpModelFormat format, ImpModel* out_model) {
    // Default load path: do NOT load the MTP head (saves ~1.57 GiB VRAM on
    // Qwen3.6). Callers that want spec-decode use imp_model_load_ex(.., 1, ..).
    return imp_model_load_ex(path, format, /*load_mtp_head=*/0, out_model);
}

void imp_model_free(ImpModel model) {
    if (!model)
        return;
    delete model;
    trim_default_mempool();
}

ImpModelArch imp_model_arch(ImpModel model) {
    if (!model || !model->model) {
        return IMP_ARCH_GENERIC;
    }
    return static_cast<ImpModelArch>(imp::model_arch_c_api_id(model->model->config().arch));
}

int imp_model_n_layers(ImpModel model) {
    if (!model || !model->model) {
        return 0;
    }
    return model->model->config().n_layers;
}

int imp_model_d_model(ImpModel model) {
    if (!model || !model->model) {
        return 0;
    }
    return model->model->config().d_model;
}

int imp_model_vocab_size(ImpModel model) {
    if (!model || !model->model) {
        return 0;
    }
    return model->model->config().vocab_size;
}

int32_t imp_model_bos_token(ImpModel model) {
    if (!model || !model->model || !model->model->tokenizer()) {
        return -1;
    }
    const auto* tok = model->model->tokenizer();
    return tok->add_bos() ? tok->bos_id() : -1;
}

ImpError imp_lora_load(ImpContext ctx, const char* path, int32_t* out_id) {
    if (!ctx || !ctx->engine || !path || !out_id)
        return IMP_ERROR_INVALID_ARG;
    try {
        int id = ctx->engine->lora_load(path);
        if (id <= 0)
            return IMP_ERROR_INVALID_MODEL;
        *out_id = id;
        return IMP_SUCCESS;
    } catch (const std::exception& e) {
        IMP_LOG_ERROR("imp_lora_load: %s", e.what());
        return IMP_ERROR_INTERNAL;
    }
}

ImpError imp_lora_set(ImpContext ctx, int32_t adapter_id) {
    if (!ctx || !ctx->engine)
        return IMP_ERROR_INVALID_ARG;
    try {
        return ctx->engine->lora_set(adapter_id) ? IMP_SUCCESS : IMP_ERROR_INVALID_ARG;
    } catch (const std::exception& e) {
        IMP_LOG_ERROR("imp_lora_set: %s", e.what());
        return IMP_ERROR_INTERNAL;
    }
}

int imp_model_max_seq_len(ImpModel model) {
    if (!model || !model->model) {
        return 0;
    }
    return model->model->config().max_seq_len;
}

// --- Context / Runtime ---

ImpError imp_context_create(ImpModel model, const ImpConfig* config, ImpContext* out_ctx) {
    if (!model || !config || !out_ctx) {
        return IMP_ERROR_INVALID_ARG;
    }

    *out_ctx = nullptr;

    if (!model->model) {
        return IMP_ERROR_INVALID_MODEL;
    }

    try {
        // Build EngineConfig from ImpConfig
        imp::EngineConfig ecfg;
        ecfg.max_batch_size = config->max_batch_size;
        ecfg.max_seq_len = config->max_seq_len;
        ecfg.kv_cache_max_blocks = static_cast<int>(config->kv_cache_max_blocks);
        ecfg.compute_dtype = map_dtype(config->compute_dtype);
        ecfg.use_green_contexts = (config->enable_green_contexts != 0);
        ecfg.green_ctx_prefill_ratio = config->green_ctx_prefill_ratio;
        ecfg.use_cuda_graphs = (config->enable_cuda_graphs != 0);
        ecfg.use_pdl = (config->enable_pdl != 0);
        ecfg.gpu_layers = config->gpu_layers;
        ecfg.kv_cache_dtype = map_dtype(config->kv_cache_dtype);
        ecfg.ssm_state_dtype = map_dtype(config->ssm_state_dtype);
        ecfg.vram_budget_mb = config->vram_budget_mb;
        ecfg.temperature = config->temperature;
        ecfg.top_p = config->top_p;
        ecfg.top_k = config->top_k;
        ecfg.prefill_chunk_size = config->prefill_chunk_size;
        ecfg.use_fp8_prefill = (config->use_fp8_prefill != 0);
        ecfg.use_nvfp4_decode = config->use_nvfp4_decode;
        ecfg.min_kv_tokens = config->min_kv_tokens;
        ecfg.use_mxfp4_prefill = (config->use_mxfp4_prefill != 0);
        ecfg.dual_path_quant = (config->dual_path_quant != 0);
        ecfg.use_prefix_caching = (config->use_prefix_caching != 0);
        if (config->prefix_cache_path[0] != '\0')
            ecfg.prefix_cache_path = config->prefix_cache_path;
        ecfg.prefix_pin_budget_pct = config->prefix_pin_budget_pct;
        ecfg.turboquant_sketch_multiplier = config->turboquant_sketch_multiplier;
        if (config->mmproj_path)
            ecfg.mmproj_path = config->mmproj_path;
        ecfg.streaming_kv_enabled = (config->streaming_kv_enabled != 0);
        ecfg.streaming_kv_auto = (config->streaming_kv_auto != 0);
        ecfg.streaming_kv_n_sinks = config->streaming_kv_n_sinks;
        ecfg.streaming_kv_window = config->streaming_kv_window;
        ecfg.streaming_kv_threshold = config->streaming_kv_threshold;

        // Create and initialize the engine
        auto engine = std::make_unique<imp::Engine>();
        if (!engine->init(model->model, ecfg)) {
            return IMP_ERROR_INTERNAL;
        }

        // Create the context handle
        auto ctx = new (std::nothrow) ImpContext_T();
        if (!ctx) {
            return IMP_ERROR_OUT_OF_MEMORY;
        }

        ctx->model_handle = model;
        ctx->engine = std::move(engine);
        ctx->active_request = nullptr;

        *out_ctx = ctx;
        return IMP_SUCCESS;
    } catch (const std::bad_alloc&) {
        return IMP_ERROR_OUT_OF_MEMORY;
    } catch (const std::exception& e) {
        IMP_LOG_ERROR("imp_context_create: %s", e.what());
        return IMP_ERROR_INTERNAL;
    } catch (...) {
        return IMP_ERROR_INTERNAL;
    }
}

void imp_context_free(ImpContext ctx) {
    if (!ctx)
        return;
    delete ctx;
    trim_default_mempool();
}

// --- Helper: tokenize a prompt using chat template or raw encoding ---

namespace {

static std::vector<int32_t> tokenize_prompt(ImpContext ctx, const char* prompt,
                                            const ImpGenerateParams* params) {
    auto* tok = ctx->model_handle->model->tokenizer();
    const auto& tmpl = ctx->engine->chat_template();
    bool has_img = ctx->engine->has_vision() && ctx->engine->has_vision_input();

    // Tokenize the prompt, injecting image tokens if a vision image is set.
    if (params->apply_chat_template && !tmpl.is_raw()) {
        std::vector<imp::ChatMessage> messages = {{"user", prompt}};
        if (has_img) {
            return tmpl.apply_with_image(*tok, messages, 256);
        } else {
            return tmpl.apply(*tok, messages);
        }
    }

    auto tokens = tok->encode(prompt);
    if (tok->add_bos() && (tokens.empty() || tokens[0] != tok->bos_id())) {
        tokens.insert(tokens.begin(), static_cast<int32_t>(tok->bos_id()));
    }
    return tokens;
}

// --- Helper: apply sampling params from ImpGenerateParams to a Request ---

static void apply_sampling_params(imp::Request& req, const ImpGenerateParams* params) {
    req.max_tokens = params->max_tokens;
    req.temperature = params->temperature;
    req.top_p = params->top_p;
    req.top_k = params->top_k;
    req.seed = params->seed;
    req.min_p = params->min_p;
    req.typical_p = params->typical_p;
    req.repetition_penalty = params->repetition_penalty;
    req.frequency_penalty = params->frequency_penalty;
    req.presence_penalty = params->presence_penalty;
    req.repeat_last_n = params->repeat_last_n;
    req.dry_multiplier = params->dry_multiplier;
    req.dry_base = params->dry_base;
    req.dry_allowed_length = params->dry_allowed_length;
    req.dry_penalty_last_n = params->dry_penalty_last_n;
    req.mirostat = params->mirostat;
    req.mirostat_tau = params->mirostat_tau;
    req.mirostat_eta = params->mirostat_eta;
    if (params->mirostat == 2 && req.mirostat_mu == 0.0f)
        req.mirostat_mu = 2.0f * params->mirostat_tau;
}

}  // anonymous namespace

// --- Generation ---

// Thin wrapper helper: tokenise the prompt, prefill, then loop decode_step.
// `on_token` is invoked for every newly decoded token; return true to keep
// going, false to stop early (the caller's request gets cancelled).
//
// This is the shared body of imp_generate / imp_generate_streaming. Both
// public entry points stay ABI-stable; only their bodies collapse into this
// helper + imp_prefill_with_params + imp_decode_step.
namespace {

template <typename OnToken>
static ImpError generate_via_prefill_decode_loop(ImpContext ctx, const char* prompt,
                                                 const ImpGenerateParams* params,
                                                 OnToken&& on_token) {
    auto* tok = ctx->model_handle->model->tokenizer();
    if (!tok)
        return IMP_ERROR_INVALID_MODEL;

    auto tokens = tokenize_prompt(ctx, prompt, params);

    // Empty token stream would walk into executor_forward.cu's "n_tokens must
    // be positive" guard and then trip a FATAL via the slice() of an
    // uninitialised logits tensor. Bail out cleanly here instead of crashing.
    if (tokens.empty()) {
        IMP_LOG_WARN(
            "imp_generate: prompt tokenised to 0 tokens (prompt='%.80s%s', "
            "model may lack vocab coverage or chat-template guard rejected it)",
            prompt, std::strlen(prompt) > 80 ? "…" : "");
        return IMP_ERROR_INVALID_ARG;
    }

    // Prefill: imp_prefill_with_params handles request lifecycle, sampling
    // params for the first-token sample, and the engine->step() loop until
    // PREFILLING completes.
    ImpError err = imp_prefill_with_params(ctx, tokens.data(),
                                           static_cast<int>(tokens.size()), params);
    if (err != IMP_SUCCESS)
        return err;

    // The prefill last-chunk sampler emits the FIRST generation token into
    // req->output_tokens. imp_prefill_with_params marks those as "already
    // consumed" so a token-level (prefill+decode_step) caller doesn't get
    // them — but the high-level imp_generate / imp_generate_streaming
    // contract says every generated token reaches the caller. Reset the
    // cursor so imp_decode_step drains the prefill-sampled token(s) on its
    // first call(s) before stepping the engine again.
    ctx->consumed_output = 0;

    // Decode loop: imp_decode_step handles per-step sampling params,
    // multi-token (self-spec) consumption, and clears active_request when the
    // engine marks FINISHED (EOS or its own max_tokens guard fires).
    const int max_tokens = params->max_tokens > 0 ? params->max_tokens : 1;
    for (int i = 0; i < max_tokens; ++i) {
        int32_t token = 0;
        ImpError step_err = imp_decode_step(ctx, params, &token);
        if (step_err != IMP_SUCCESS) {
            // INTERNAL after natural finish (active_request was cleared) is
            // the normal stop signal — anything else propagates.
            if (step_err == IMP_ERROR_INTERNAL && !ctx->active_request)
                break;
            return step_err;
        }

        if (!on_token(token))
            return IMP_ERROR_CANCELLED;

        // Engine signalled FINISHED (EOS / its own max_tokens) — decode_step
        // already cleaned up and set active_request = nullptr.
        if (!ctx->active_request)
            break;
    }

    return IMP_SUCCESS;
}

}  // namespace

ImpError imp_generate_streaming(ImpContext ctx, const char* prompt, const ImpGenerateParams* params,
                                ImpTokenCallback cb, void* user_data) {
    if (!ctx || !prompt || !params || !cb) {
        return IMP_ERROR_INVALID_ARG;
    }
    if (!ctx->engine) {
        return IMP_ERROR_INTERNAL;
    }

    try {
        auto* tok = ctx->model_handle->model->tokenizer();
        if (!tok)
            return IMP_ERROR_INVALID_MODEL;

        return generate_via_prefill_decode_loop(
            ctx, prompt, params, [&](int32_t token) -> bool {
                std::string text = tok->decode({token});
                int stop = cb(text.c_str(), text.size(), user_data);
                return stop == 0;
            });
    } catch (const std::bad_alloc&) {
        return IMP_ERROR_OUT_OF_MEMORY;
    } catch (const std::exception& e) {
        IMP_LOG_ERROR("imp_generate_streaming: %s", e.what());
        return IMP_ERROR_INTERNAL;
    } catch (...) {
        return IMP_ERROR_INTERNAL;
    }
}

ImpError imp_generate(ImpContext ctx, const char* prompt, const ImpGenerateParams* params, char* output_buf,
                      size_t output_buf_size, size_t* output_len) {
    if (!ctx || !prompt || !params || !output_buf || output_buf_size == 0) {
        return IMP_ERROR_INVALID_ARG;
    }

    if (!ctx->engine) {
        return IMP_ERROR_INTERNAL;
    }

    try {
        auto* tok = ctx->model_handle->model->tokenizer();
        if (!tok)
            return IMP_ERROR_INVALID_MODEL;

        std::vector<int32_t> output_tokens;
        output_tokens.reserve(params->max_tokens > 0 ? params->max_tokens : 256);

        ImpError err = generate_via_prefill_decode_loop(
            ctx, prompt, params, [&](int32_t token) -> bool {
                output_tokens.push_back(token);
                return true;
            });
        if (err != IMP_SUCCESS) {
            if (output_len) *output_len = 0;
            if (output_buf_size > 0) output_buf[0] = '\0';
            return err;
        }

        // Detokenise the accumulated tokens.
        std::string result = tok->decode(output_tokens);

        // Copy result to output buffer.
        size_t copy_len = result.size();
        if (copy_len >= output_buf_size) {
            copy_len = output_buf_size - 1;
        }
        std::memcpy(output_buf, result.data(), copy_len);
        output_buf[copy_len] = '\0';

        if (output_len) {
            *output_len = copy_len;
        }

        return IMP_SUCCESS;
    } catch (const std::bad_alloc&) {
        return IMP_ERROR_OUT_OF_MEMORY;
    } catch (const std::exception& e) {
        IMP_LOG_ERROR("imp_generate: %s", e.what());
        return IMP_ERROR_INTERNAL;
    } catch (...) {
        return IMP_ERROR_INTERNAL;
    }
}

ImpError imp_tokenize(ImpModel model, const char* text, int32_t* tokens, int* n_tokens, int max_tokens) {
    if (!model || !text || !tokens || !n_tokens || max_tokens <= 0) {
        return IMP_ERROR_INVALID_ARG;
    }

    auto* tok = model->model ? model->model->tokenizer() : nullptr;
    if (!tok || tok->vocab_size() == 0) {
        *n_tokens = 0;
        return IMP_ERROR_INVALID_MODEL;
    }

    try {
        auto ids = tok->encode(text);
        int count = static_cast<int>(ids.size());
        if (count > max_tokens)
            count = max_tokens;

        for (int i = 0; i < count; i++) {
            tokens[i] = ids[i];
        }
        *n_tokens = count;
        return IMP_SUCCESS;
    } catch (const std::bad_alloc&) {
        return IMP_ERROR_OUT_OF_MEMORY;
    } catch (const std::exception& e) {
        IMP_LOG_ERROR("imp_tokenize: %s", e.what());
        return IMP_ERROR_INTERNAL;
    } catch (...) {
        return IMP_ERROR_INTERNAL;
    }
}

ImpError imp_detokenize(ImpModel model, const int32_t* tokens, int n_tokens, char* output_buf,
                        size_t output_buf_size) {
    if (!model || !tokens || !output_buf || output_buf_size == 0 || n_tokens < 0) {
        return IMP_ERROR_INVALID_ARG;
    }

    auto* tok = model->model ? model->model->tokenizer() : nullptr;
    if (!tok || tok->vocab_size() == 0) {
        output_buf[0] = '\0';
        return IMP_ERROR_INVALID_MODEL;
    }

    try {
        std::vector<int32_t> ids(tokens, tokens + n_tokens);
        std::string text = tok->decode(ids);

        size_t copy_len = text.size();
        if (copy_len >= output_buf_size)
            copy_len = output_buf_size - 1;
        std::memcpy(output_buf, text.data(), copy_len);
        output_buf[copy_len] = '\0';
        return IMP_SUCCESS;
    } catch (const std::bad_alloc&) {
        return IMP_ERROR_OUT_OF_MEMORY;
    } catch (const std::exception& e) {
        IMP_LOG_ERROR("imp_detokenize: %s", e.what());
        return IMP_ERROR_INTERNAL;
    } catch (...) {
        return IMP_ERROR_INTERNAL;
    }
}

ImpError imp_prefill_with_params(ImpContext ctx, const int32_t* tokens, int n_tokens,
                                 const ImpGenerateParams* params) {
    if (!ctx || !tokens || n_tokens <= 0) {
        return IMP_ERROR_INVALID_ARG;
    }

    if (!ctx->engine) {
        return IMP_ERROR_INTERNAL;
    }

    try {
        // If there is an existing active request, free its KV cache and mark
        // cancelled so the scheduler removes it from active_ on next schedule().
        if (ctx->active_request) {
            ctx->engine->kv_manager()->free_sequence(ctx->active_request->id);
            ctx->engine->reset_ssm_state(ctx->active_request->id);
            ctx->active_request->status = imp::RequestStatus::CANCELLED;
            ctx->active_request = nullptr;
        }

        // Create a request with the input tokens
        auto req = std::make_shared<imp::Request>();
        req->input_tokens.assign(tokens, tokens + n_tokens);
        req->max_tokens = 4096;  // Large default; decode_step controls actual stopping
        // Apply caller-supplied sampling params so the prefill last-chunk
        // sampler honours top_p / top_k / temperature for the FIRST token.
        // Without this Gemma-4-NVFP4 (and other noisy-logit-tail quants)
        // can sample garbage like <|end_of_text|> on token #0 and never
        // recover, even with temperature == 0.7 + properly-loaded
        // generation_config.json defaults.
        if (params) {
            req->temperature = params->temperature;
            req->top_p = params->top_p;
            req->top_k = params->top_k;
            req->seed = params->seed;
            req->min_p = params->min_p;
            req->typical_p = params->typical_p;
            req->repetition_penalty = params->repetition_penalty;
            req->frequency_penalty = params->frequency_penalty;
            req->presence_penalty = params->presence_penalty;
            req->repeat_last_n = params->repeat_last_n;
            req->dry_multiplier = params->dry_multiplier;
            req->dry_base = params->dry_base;
            req->dry_allowed_length = params->dry_allowed_length;
            req->dry_penalty_last_n = params->dry_penalty_last_n;
            req->mirostat = params->mirostat;
            req->mirostat_tau = params->mirostat_tau;
            req->mirostat_eta = params->mirostat_eta;
            if (params->mirostat == 2 && req->mirostat_mu == 0.0f)
                req->mirostat_mu = 2.0f * params->mirostat_tau;
        }
        req->ignore_eos = true;  // Don't stop during prefill — decode_step controls stopping
        req->status = imp::RequestStatus::PENDING;

        // Add to engine (assigns request id)
        ctx->engine->add_request(req);

        // Store as the active request for subsequent decode_step calls
        ctx->active_request = req;
        ctx->consumed_output = 0;

        // Run steps until prefill completes (may take multiple steps with chunked prefill)
        do {
            (void)ctx->engine->step();
        } while (req->status == imp::RequestStatus::PREFILLING);

        // Verify the request was prefilled
        if (req->status == imp::RequestStatus::CANCELLED) {
            ctx->active_request = nullptr;
            return IMP_ERROR_OUT_OF_MEMORY;
        }

        // After prefill, any tokens already in output_tokens are "consumed"
        // by the prefill path (the first decode token).
        ctx->consumed_output = req->output_tokens.size();

        return IMP_SUCCESS;
    } catch (const std::bad_alloc&) {
        return IMP_ERROR_OUT_OF_MEMORY;
    } catch (const std::exception& e) {
        IMP_LOG_ERROR("imp_prefill: %s", e.what());
        return IMP_ERROR_INTERNAL;
    } catch (...) {
        return IMP_ERROR_INTERNAL;
    }
}

// Legacy entry point — defaults to no caller-supplied sampling, leaves the
// first-token sample at end of prefill on Request struct defaults
// (top_p=1, top_k=0). Kept for ABI; new callers should use
// imp_prefill_with_params and pass the same params they'll use in
// imp_decode_step.
ImpError imp_prefill(ImpContext ctx, const int32_t* tokens, int n_tokens) {
    return imp_prefill_with_params(ctx, tokens, n_tokens, nullptr);
}

ImpError imp_perplexity(ImpContext ctx, const int32_t* tokens, int n_tokens, double* out_ppl) {
    if (!ctx || !tokens || n_tokens < 2 || !out_ppl) {
        return IMP_ERROR_INVALID_ARG;
    }
    if (!ctx->engine) {
        return IMP_ERROR_INTERNAL;
    }
    *out_ppl = -1.0;
    try {
        // Fresh context so the prefill covers exactly this corpus.
        imp_context_reset(ctx);
        // Chunked-prefill-aware: the engine accumulates per-position NLL
        // after every chunk it forwards. (The executor's hidden_ only
        // retains the most recent chunk, so the historical post-hoc
        // executor()->perplexity_nll() silently scored stale positions
        // whenever the resolved prefill chunk size was smaller than the
        // corpus — which is the C-API DEFAULT: prefill_chunk_size=-1
        // resolves to 512 on dense archs.)
        if (!ctx->engine->begin_perplexity_capture(tokens, n_tokens))
            return IMP_ERROR_INTERNAL;
        ImpError e = imp_prefill(ctx, tokens, n_tokens);
        double ppl = -1.0;
        const bool reduced = ctx->engine->end_perplexity_capture(&ppl);  // always frees buffers
        // Release the prefill request's KV + recurrent slot. NOTE: do NOT
        // null active_request first — imp_context_reset only cleans up
        // (free_sequence / reset_ssm_state / slot release) when it still
        // sees the request; nulling early leaked the KV sequence AND the
        // SSM/GDN slot on every imp_perplexity call, so repeated scoring
        // on hybrid models drifted (stale recurrent state, slot pool decay).
        imp_context_reset(ctx);
        if (e != IMP_SUCCESS)
            return e;
        if (!reduced || ppl < 0.0)
            return IMP_ERROR_INTERNAL;
        *out_ppl = ppl;
        return IMP_SUCCESS;
    } catch (const std::exception& ex) {
        IMP_LOG_ERROR("imp_perplexity: %s", ex.what());
        return IMP_ERROR_INTERNAL;
    }
}

ImpError imp_decode_step(ImpContext ctx, const ImpGenerateParams* params, int32_t* out_token) {
    if (!ctx || !params || !out_token) {
        return IMP_ERROR_INVALID_ARG;
    }

    *out_token = 0;

    if (!ctx->engine || !ctx->active_request) {
        return IMP_ERROR_INTERNAL;
    }

    try {
        auto& req = ctx->active_request;

        // Apply ignore_eos BEFORE the finished check — when benchmarking with
        // synthetic tokens, prefill may produce EOS as the first output token
        // (e.g. Gemma-3), marking the request FINISHED.  If the caller wants
        // to ignore EOS, we must reset the request back to GENERATING.
        bool caller_ignore_eos = (params->ignore_eos != 0);
        if (caller_ignore_eos && req->status == imp::RequestStatus::FINISHED) {
            req->status = imp::RequestStatus::DECODING;
        }

        // Check if already finished
        if (req->status == imp::RequestStatus::FINISHED || req->status == imp::RequestStatus::CANCELLED) {
            return IMP_ERROR_INTERNAL;
        }

        // Update sampling params on the request for this step
        apply_sampling_params(*req, params);
        req->ignore_eos = caller_ignore_eos;
        req->logprobs = (params->logprobs != 0);
        req->top_logprobs = std::max(0, std::min(20, params->top_logprobs));
        req->json_mode = (params->json_mode != 0);

        // Self-speculative (and future multi-token) steps may produce
        // multiple tokens per engine->step().  Track how many have been
        // consumed so we only call step() when all previous tokens are
        // returned to the caller.
        if (ctx->consumed_output < req->output_tokens.size()) {
            // Still have unconsumed tokens from a previous multi-token step
            *out_token = req->output_tokens[ctx->consumed_output++];
        } else {
            // Need a new engine step. A step may legitimately yield ZERO new
            // tokens when it only launches an async graph-loop burst (n-gram
            // speculation miss path) — the burst's tokens arrive on the next
            // step's drain. Retry a bounded number of times; a persistent
            // zero-token stream is still an internal error.
            size_t prev_output_size = req->output_tokens.size();
            for (int attempts = 0;
                 attempts < 8 && req->output_tokens.size() == prev_output_size &&
                 req->status == imp::RequestStatus::DECODING;
                 ++attempts) {
                (void)ctx->engine->step();
            }

            if (req->output_tokens.size() > prev_output_size) {
                ctx->consumed_output = prev_output_size;
                *out_token = req->output_tokens[ctx->consumed_output++];
            } else {
                return IMP_ERROR_INTERNAL;
            }
        }

        // If the request finished (eos or max_tokens), clean up
        if (req->status == imp::RequestStatus::FINISHED) {
            // KV cache is already freed by engine step() on FINISHED
            ctx->active_request = nullptr;
        }

        return IMP_SUCCESS;
    } catch (const std::bad_alloc&) {
        return IMP_ERROR_OUT_OF_MEMORY;
    } catch (const std::exception& e) {
        IMP_LOG_ERROR("imp_decode_step: %s", e.what());
        return IMP_ERROR_INTERNAL;
    } catch (...) {
        return IMP_ERROR_INTERNAL;
    }
}

ImpError imp_context_reset(ImpContext ctx) {
    if (!ctx) {
        return IMP_ERROR_INVALID_ARG;
    }

    if (!ctx->engine) {
        return IMP_ERROR_INTERNAL;
    }

    // Free the active request's KV cache and mark it cancelled so the
    // scheduler removes it from active_ on the next schedule() call.
    if (ctx->active_request) {
        ctx->engine->kv_manager()->free_sequence(ctx->active_request->id);
        // Evict all cached blocks to prevent prefix cache hits with stale data
        while (ctx->engine->kv_manager()->evict_cached_block()) {}
        // Reset SSM state for hybrid models (Mamba2)
        ctx->engine->reset_ssm_state(ctx->active_request->id);
        ctx->active_request->status = imp::RequestStatus::CANCELLED;
        ctx->active_request = nullptr;
        ctx->consumed_output = 0;
    }

    // Sync GPU to ensure all async operations from the previous request complete
    // before resetting state. Without this, stale async graph loops or pending
    // kernel launches can corrupt the next request's data.
    cudaDeviceSynchronize();

    // Invalidate cached CUDA graphs — stale graph captures from the previous
    // request can produce non-deterministic output if replayed for a new request.
    ctx->engine->invalidate_graphs();

    // Reset batch pool upload cache — the next request may reuse the same
    // physical KV cache blocks, and stale cached block table pointers would
    // cause the GPU to read from old KV data.
    ctx->engine->reset_batch_pool_cache();

    // Reset MTP-side KV cache + accuracy telemetry for a clean new session.
    ctx->engine->mtp_accuracy_reset();

    return IMP_SUCCESS;
}

// --- MTP spec-decode (Phase 4) ---

ImpError imp_enable_mtp_spec_decode(ImpContext ctx, int k) {
    if (!ctx) return IMP_ERROR_INVALID_ARG;
    if (!ctx->engine) return IMP_ERROR_INTERNAL;
    if (k <= 0) return IMP_ERROR_INVALID_ARG;
    return ctx->engine->enable_mtp_spec_decode(k) ? IMP_SUCCESS
                                                  : IMP_ERROR_INVALID_ARG;
}

// --- Vision (Multimodal) ---

ImpError imp_set_image(ImpContext ctx, const char* image_path) {
    if (!ctx)
        return IMP_ERROR_INVALID_ARG;
    if (!ctx->engine)
        return IMP_ERROR_INTERNAL;

    if (!image_path) {
        ctx->engine->clear_image();
        return IMP_SUCCESS;
    }

    if (!ctx->engine->has_vision()) {
        IMP_LOG_ERROR("imp_set_image: no vision model loaded (mmproj_path not set)");
        return IMP_ERROR_UNSUPPORTED;
    }

    try {
        if (!ctx->engine->set_image(image_path)) {
            return IMP_ERROR_INTERNAL;
        }
        return IMP_SUCCESS;
    } catch (const std::exception& e) {
        IMP_LOG_ERROR("imp_set_image: %s", e.what());
        return IMP_ERROR_INTERNAL;
    }
}

ImpError imp_set_image_from_memory(ImpContext ctx, const uint8_t* data, size_t len) {
    if (!ctx)
        return IMP_ERROR_INVALID_ARG;
    if (!ctx->engine)
        return IMP_ERROR_INTERNAL;

    if (!data || len == 0) {
        ctx->engine->clear_image();
        return IMP_SUCCESS;
    }

    if (!ctx->engine->has_vision()) {
        IMP_LOG_ERROR("imp_set_image_from_memory: no vision model loaded");
        return IMP_ERROR_UNSUPPORTED;
    }

    try {
        if (!ctx->engine->set_image_from_memory(data, len)) {
            return IMP_ERROR_INTERNAL;
        }
        return IMP_SUCCESS;
    } catch (const std::exception& e) {
        IMP_LOG_ERROR("imp_set_image_from_memory: %s", e.what());
        return IMP_ERROR_INTERNAL;
    }
}
