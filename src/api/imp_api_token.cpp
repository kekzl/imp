// Token-level entry points of the C API (tokenize, detokenize, the prefill-sampled token, forced decode),
// split out of imp_api.cpp (file-size gate).

#include "api/imp_internal.h"
#include "model/tokenizer.h"
#include "runtime/request.h"

#include <algorithm>
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
        count = std::min(count, max_tokens);

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

ImpError imp_set_forced_decode(ImpContext ctx, const int32_t* tokens, int n_tokens) {
    if (!ctx || !tokens || n_tokens <= 0)
        return IMP_ERROR_INVALID_ARG;
    return imp::api_guard("imp_set_forced_decode", [&]() -> ImpError {
        auto& req = ctx->active_request;
        // Only between imp_prefill and the first imp_decode_step (prefill emitted exactly one token).
        if (!req || req->output_tokens.size() != 1 || ctx->consumed_output != 1)
            return IMP_ERROR_INVALID_ARG;
        req->forced_decode.assign(tokens, tokens + n_tokens);
        req->output_tokens.back() = req->forced_decode[0];
        ctx->prefill_token = req->forced_decode[0];
        req->forced_pos = 1;
        return IMP_SUCCESS;
    });
}

int imp_forced_decode_count(ImpContext ctx) {
    if (!ctx || !ctx->active_request)
        return -1;
    return static_cast<int>(ctx->active_request->forced_pos);
}

ImpError imp_forced_decode_nll(ImpContext ctx, double* out_nll_sum, int* out_n) {
    if (!ctx || !out_nll_sum || !out_n)
        return IMP_ERROR_INVALID_ARG;
    return imp::api_guard("imp_forced_decode_nll", [&]() -> ImpError {
        if (!ctx->active_request)
            return IMP_ERROR_INVALID_ARG;
        *out_nll_sum = ctx->active_request->forced_nll_sum;
        *out_n = ctx->active_request->forced_nll_n;
        return IMP_SUCCESS;
    });
}
