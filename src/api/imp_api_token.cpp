// Token-level entry points of the C API (tokenize, detokenize, the prefill-sampled token), split out
// of imp_api.cpp (file-size gate).

#include "api/imp_internal.h"
#include "model/tokenizer.h"

#include <cstring>
#include <string>
#include <vector>

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

// The token the last prefill sampled; set by imp_prefill_with_params (#2251).
ImpError imp_prefill_token(ImpContext ctx, int32_t* out_token) {
    return imp::api_guard("imp_prefill_token", [&] {
        if (!ctx || !out_token || ctx->prefill_token < 0)
            return IMP_ERROR_INVALID_ARG;
        *out_token = ctx->prefill_token;
        return IMP_SUCCESS;
    });
}
