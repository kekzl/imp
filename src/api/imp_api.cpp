#include "imp/imp.h"
#include "model/model.h"
#include "model/gguf_loader.h"
#include "model/safetensors_loader.h"
#include "model/tokenizer.h"
#include "model/chat_template.h"
#include "runtime/engine.h"
#include "runtime/request.h"
#include "memory/kv_cache.h"

#include "core/logging.h"

#include <cstring>
#include <memory>
#include <vector>
#include <new>
#include <exception>

// --- Internal handle types ---

struct ImpModel_T {
    std::shared_ptr<imp::Model> model;
};

struct ImpContext_T {
    ImpModel model_handle = nullptr;
    std::unique_ptr<imp::Engine> engine;

    // State for token-level prefill/decode API
    std::shared_ptr<imp::Request> active_request;
};

// --- Error string ---

const char* imp_error_string(ImpError err) {
    switch (err) {
        case IMP_SUCCESS:              return "success";
        case IMP_ERROR_INVALID_ARG:    return "invalid argument";
        case IMP_ERROR_OUT_OF_MEMORY:  return "out of memory";
        case IMP_ERROR_CUDA:           return "CUDA error";
        case IMP_ERROR_FILE_NOT_FOUND: return "file not found";
        case IMP_ERROR_INVALID_MODEL:  return "invalid model";
        case IMP_ERROR_UNSUPPORTED:    return "unsupported operation";
        case IMP_ERROR_INTERNAL:       return "internal error";
        case IMP_ERROR_CANCELLED:      return "cancelled";
        default:                       return "unknown error";
    }
}

// --- Config defaults ---

ImpConfig imp_config_default(void) {
    ImpConfig config;
    config.device_id = 0;
    config.gpu_memory_pool_size = 0;    // auto
    config.kv_cache_max_blocks = 0;     // auto
    config.max_batch_size = 32;
    config.max_seq_len = 4096;
    config.compute_dtype = IMP_DTYPE_FP16;
    config.temperature = 1.0f;
    config.top_p = 1.0f;
    config.top_k = 0;
    config.max_tokens = 256;
    config.enable_green_contexts = 0;
    config.green_ctx_prefill_ratio = 0.8f;
    config.enable_pdl = 1;
    config.enable_cuda_graphs = 1;
    config.gpu_layers = -1;             // all on GPU
    config.kv_cache_dtype = IMP_DTYPE_FP16;
    config.ssm_state_dtype = IMP_DTYPE_FP32;
    config.vram_budget_mb = 0;          // use all available
    config.prefill_chunk_size = 0;      // no chunking
    config.use_fp8_prefill = 0;         // FP16 weight cache by default
    config.use_nvfp4_decode = -1;       // auto (sm_120→mode2, sm_90→mode1)
    config.num_cpu_threads = 0;         // auto
    config.mmproj_path = NULL;          // no vision model
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

const char* imp_version(void) {
    return "0.1.0";
}

// --- Helper: map ImpDType to imp::DType ---

static imp::DType map_dtype(ImpDType dt) {
    switch (dt) {
        case IMP_DTYPE_FP32:     return imp::DType::FP32;
        case IMP_DTYPE_FP16:     return imp::DType::FP16;
        case IMP_DTYPE_BF16:     return imp::DType::BF16;
        case IMP_DTYPE_FP8_E4M3: return imp::DType::FP8_E4M3;
        case IMP_DTYPE_FP8_E5M2: return imp::DType::FP8_E5M2;
        case IMP_DTYPE_INT8:     return imp::DType::INT8;
        case IMP_DTYPE_INT4:     return imp::DType::INT4;
        case IMP_DTYPE_INT32:    return imp::DType::INT32;
        default:                 return imp::DType::FP16;
    }
}

// --- Model Loading ---

ImpError imp_model_load(const char* path, ImpModelFormat format,
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
                model = imp::load_safetensors(path);
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

void imp_model_free(ImpModel model) {
    delete model;
}

ImpModelArch imp_model_arch(ImpModel model) {
    if (!model || !model->model) {
        return IMP_ARCH_GENERIC;
    }
    switch (model->model->config().arch) {
        case imp::ModelArch::LLAMA:          return IMP_ARCH_LLAMA;
        case imp::ModelArch::MISTRAL:        return IMP_ARCH_MISTRAL;
        case imp::ModelArch::MIXTRAL:        return IMP_ARCH_MIXTRAL;
        case imp::ModelArch::DEEPSEEK:       return IMP_ARCH_DEEPSEEK;
        case imp::ModelArch::NEMOTRON_H_MOE: return IMP_ARCH_NEMOTRON_H_MOE;
        case imp::ModelArch::QWEN3:          return IMP_ARCH_QWEN3;
        case imp::ModelArch::QWEN3_MOE:      return IMP_ARCH_QWEN3_MOE;
        case imp::ModelArch::GEMMA3:         return IMP_ARCH_GEMMA3;
        case imp::ModelArch::GENERIC:        return IMP_ARCH_GENERIC;
    }
    return IMP_ARCH_GENERIC;
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

int imp_model_max_seq_len(ImpModel model) {
    if (!model || !model->model) {
        return 0;
    }
    return model->model->config().max_seq_len;
}

// --- Context / Runtime ---

ImpError imp_context_create(ImpModel model, const ImpConfig* config,
                            ImpContext* out_ctx) {
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
        if (config->mmproj_path)
            ecfg.mmproj_path = config->mmproj_path;

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
    delete ctx;
}

// --- Generation ---

ImpError imp_generate_streaming(ImpContext ctx, const char* prompt,
                                const ImpGenerateParams* params,
                                ImpTokenCallback cb, void* user_data) {
    if (!ctx || !prompt || !params || !cb) {
        return IMP_ERROR_INVALID_ARG;
    }
    if (!ctx->engine) {
        return IMP_ERROR_INTERNAL;
    }

    try {
        auto* tok = ctx->model_handle->model->tokenizer();
        if (!tok) return IMP_ERROR_INVALID_MODEL;

        // Tokenize the prompt, injecting image tokens if a vision image is set.
        std::vector<int32_t> tokens;
        const auto& tmpl = ctx->engine->chat_template();
        bool has_img = ctx->engine->has_vision() &&
                       ctx->engine->has_vision_input();
        if (params->apply_chat_template && !tmpl.is_raw()) {
            std::vector<imp::ChatMessage> messages = {{"user", prompt}};
            if (has_img) {
                tokens = tmpl.apply_with_image(*tok, messages, 256);
            } else {
                tokens = tmpl.apply(*tok, messages);
            }
        } else {
            tokens = tok->encode(prompt);
            if (tok->add_bos() && (tokens.empty() || tokens[0] != tok->bos_id())) {
                tokens.insert(tokens.begin(), static_cast<int32_t>(tok->bos_id()));
            }
        }

        // Create request
        auto req = std::make_shared<imp::Request>();
        req->input_tokens = std::move(tokens);
        req->max_tokens = params->max_tokens;
        req->temperature = params->temperature;
        req->top_p = params->top_p;
        req->top_k = params->top_k;
        req->seed = params->seed;
        req->min_p = params->min_p;
        req->typical_p = params->typical_p;
        req->repetition_penalty = params->repetition_penalty;
        req->frequency_penalty = params->frequency_penalty;
        req->presence_penalty = params->presence_penalty;
        req->dry_multiplier = params->dry_multiplier;
        req->dry_base = params->dry_base;
        req->dry_allowed_length = params->dry_allowed_length;
        req->dry_penalty_last_n = params->dry_penalty_last_n;
        req->mirostat = params->mirostat;
        req->mirostat_tau = params->mirostat_tau;
        req->mirostat_eta = params->mirostat_eta;
        if (params->mirostat == 2) req->mirostat_mu = 2.0f * params->mirostat_tau;
        req->status = imp::RequestStatus::PENDING;

        ctx->engine->add_request(req);

        // Prefill
        while (req->status == imp::RequestStatus::PENDING ||
               req->status == imp::RequestStatus::PREFILLING) {
            bool has_work = ctx->engine->step();
            if (!has_work) break;
        }

        // Decode with streaming callback
        size_t prev_output_size = req->output_tokens.size();
        while (req->status != imp::RequestStatus::FINISHED &&
               req->status != imp::RequestStatus::CANCELLED) {
            bool has_work = ctx->engine->step();
            if (!has_work && req->status != imp::RequestStatus::FINISHED) break;

            // Deliver new tokens via callback
            while (prev_output_size < req->output_tokens.size()) {
                int32_t token = req->output_tokens[prev_output_size];
                std::string text = tok->decode({token});
                int stop = cb(text.c_str(), text.size(), user_data);
                prev_output_size++;
                if (stop != 0) {
                    // User requested stop
                    ctx->engine->kv_manager()->free_sequence(req->id);
                    req->status = imp::RequestStatus::CANCELLED;
                    return IMP_ERROR_CANCELLED;
                }
            }
        }

        return IMP_SUCCESS;
    } catch (const std::bad_alloc&) {
        return IMP_ERROR_OUT_OF_MEMORY;
    } catch (const std::exception& e) {
        IMP_LOG_ERROR("imp_generate_streaming: %s", e.what());
        return IMP_ERROR_INTERNAL;
    } catch (...) {
        return IMP_ERROR_INTERNAL;
    }
}

ImpError imp_generate(ImpContext ctx, const char* prompt,
                      const ImpGenerateParams* params,
                      char* output_buf, size_t output_buf_size,
                      size_t* output_len) {
    if (!ctx || !prompt || !params || !output_buf || output_buf_size == 0) {
        return IMP_ERROR_INVALID_ARG;
    }

    if (!ctx->engine) {
        return IMP_ERROR_INTERNAL;
    }

    try {
        // Call engine generate with sampling parameters
        std::string result = ctx->engine->generate(
            prompt,
            params->max_tokens,
            params->temperature,
            params->top_p,
            params->top_k,
            params->seed,
            params->apply_chat_template != 0,
            params->min_p,
            params->repetition_penalty,
            params->frequency_penalty,
            params->presence_penalty
        );

        // Copy result to output buffer
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

ImpError imp_tokenize(ImpModel model, const char* text,
                      int32_t* tokens, int* n_tokens, int max_tokens) {
    if (!model || !text || !tokens || !n_tokens) {
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
        if (count > max_tokens) count = max_tokens;

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

ImpError imp_detokenize(ImpModel model, const int32_t* tokens,
                        int n_tokens, char* output_buf,
                        size_t output_buf_size) {
    if (!model || !tokens || !output_buf || output_buf_size == 0) {
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
        if (copy_len >= output_buf_size) copy_len = output_buf_size - 1;
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

ImpError imp_prefill(ImpContext ctx, const int32_t* tokens, int n_tokens) {
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
        req->status = imp::RequestStatus::PENDING;

        // Add to engine (assigns request id)
        ctx->engine->add_request(req);

        // Store as the active request for subsequent decode_step calls
        ctx->active_request = req;

        // Run steps until prefill completes (may take multiple steps with chunked prefill)
        do {
            ctx->engine->step();
        } while (req->status == imp::RequestStatus::PREFILLING);

        // Verify the request was prefilled
        if (req->status == imp::RequestStatus::CANCELLED) {
            ctx->active_request = nullptr;
            return IMP_ERROR_OUT_OF_MEMORY;
        }

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

ImpError imp_decode_step(ImpContext ctx, const ImpGenerateParams* params,
                         int32_t* out_token) {
    if (!ctx || !params || !out_token) {
        return IMP_ERROR_INVALID_ARG;
    }

    *out_token = 0;

    if (!ctx->engine || !ctx->active_request) {
        return IMP_ERROR_INTERNAL;
    }

    try {
        auto& req = ctx->active_request;

        // Check if already finished
        if (req->status == imp::RequestStatus::FINISHED ||
            req->status == imp::RequestStatus::CANCELLED) {
            return IMP_ERROR_INTERNAL;
        }

        // Update sampling params on the request for this step
        req->temperature = params->temperature;
        req->top_p = params->top_p;
        req->top_k = params->top_k;
        req->seed = params->seed;
        req->max_tokens = params->max_tokens;
        req->ignore_eos = (params->ignore_eos != 0);
        req->min_p = params->min_p;
        req->typical_p = params->typical_p;
        req->repetition_penalty = params->repetition_penalty;
        req->frequency_penalty = params->frequency_penalty;
        req->presence_penalty = params->presence_penalty;
        req->dry_multiplier = params->dry_multiplier;
        req->dry_base = params->dry_base;
        req->dry_allowed_length = params->dry_allowed_length;
        req->dry_penalty_last_n = params->dry_penalty_last_n;
        req->mirostat = params->mirostat;
        req->mirostat_tau = params->mirostat_tau;
        req->mirostat_eta = params->mirostat_eta;
        req->logprobs = (params->logprobs != 0);
        req->top_logprobs = std::max(0, std::min(20, params->top_logprobs));
        req->json_mode = (params->json_mode != 0);
        // Initialize mu on first decode step with mirostat enabled
        if (params->mirostat == 2 && req->mirostat_mu == 0.0f)
            req->mirostat_mu = 2.0f * params->mirostat_tau;

        // Record output size before the step
        size_t prev_output_size = req->output_tokens.size();

        // Run one decode step
        ctx->engine->step();

        // Return the newly generated token
        if (req->output_tokens.size() > prev_output_size) {
            *out_token = req->output_tokens.back();
        } else {
            // No token was generated (should not happen in normal operation)
            return IMP_ERROR_INTERNAL;
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
        // Reset SSM state for hybrid models (Mamba2)
        ctx->engine->reset_ssm_state(ctx->active_request->id);
        ctx->active_request->status = imp::RequestStatus::CANCELLED;
        ctx->active_request = nullptr;
    }

    // Note: We do not reset the entire scheduler here because other sequences
    // may exist in batch scenarios. For the single-sequence C API, clearing
    // the active request is sufficient. A full reset would drain all pending
    // and active requests from the scheduler, which is beyond the scope of
    // this function.

    return IMP_SUCCESS;
}

// --- Speculative Decoding ---

ImpError imp_set_draft_model(ImpContext ctx, const char* draft_model_path,
                              ImpModelFormat format) {
    if (!ctx || !draft_model_path) {
        return IMP_ERROR_INVALID_ARG;
    }
    if (!ctx->engine) {
        return IMP_ERROR_INTERNAL;
    }

    // Currently only GGUF is supported for draft models (same as init_speculative)
    if (format != IMP_FORMAT_GGUF) {
        return IMP_ERROR_UNSUPPORTED;
    }

    try {
        if (!ctx->engine->set_draft_model(draft_model_path)) {
            return IMP_ERROR_INTERNAL;
        }

        return IMP_SUCCESS;
    } catch (const std::bad_alloc&) {
        return IMP_ERROR_OUT_OF_MEMORY;
    } catch (const std::exception& e) {
        IMP_LOG_ERROR("imp_set_draft_model: %s", e.what());
        return IMP_ERROR_INTERNAL;
    } catch (...) {
        return IMP_ERROR_INTERNAL;
    }
}

// --- Vision (Multimodal) ---

ImpError imp_set_image(ImpContext ctx, const char* image_path) {
    if (!ctx) return IMP_ERROR_INVALID_ARG;
    if (!ctx->engine) return IMP_ERROR_INTERNAL;

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
    if (!ctx) return IMP_ERROR_INVALID_ARG;
    if (!ctx->engine) return IMP_ERROR_INTERNAL;

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
