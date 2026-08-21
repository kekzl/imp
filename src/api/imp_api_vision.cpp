// The C API's image surface.
//
// Split out of imp_api.cpp because that file sits exactly on the 800-code-LOC
// hard threshold and this is the part with room to grow: a tower that takes
// several images needs more entry points than one that takes a single fixed
// one, and each is a thin guard over Engine.

#include "api/imp_internal.h"
#include "runtime/engine.h"

#include "core/logging.h"
#include <span>

#include <exception>

namespace {

// Every entry point here answers the same two questions first, and the second
// one is the interesting one: `has_vision()` covers BOTH towers (Qwen3-VL ships
// its own in the checkpoint, Gemma-3/4 need mmproj_path), so a message naming
// only mmproj would be wrong for half the models that reach it.
ImpError check_vision_ready(ImpContext ctx, const char* fn) {
    if (!ctx)
        return IMP_ERROR_INVALID_ARG;
    if (!ctx->engine)
        return IMP_ERROR_INTERNAL;
    if (!ctx->engine->has_vision()) {
        IMP_LOG_ERROR("%s: model has no vision tower (Gemma-3/4 need mmproj_path)", fn);
        return IMP_ERROR_UNSUPPORTED;
    }
    return IMP_SUCCESS;
}

}  // namespace

int imp_pending_image_tokens(ImpContext ctx) {
    return (ctx && ctx->engine) ? ctx->engine->pending_image_tokens() : 0;
}

ImpError imp_set_image(ImpContext ctx, const char* image_path) {
    if (!ctx)
        return IMP_ERROR_INVALID_ARG;
    if (!ctx->engine)
        return IMP_ERROR_INTERNAL;
    if (!image_path) {
        ctx->engine->clear_image();
        return IMP_SUCCESS;
    }
    if (ImpError e = check_vision_ready(ctx, "imp_set_image"); e != IMP_SUCCESS)
        return e;
    try {
        return ctx->engine->set_image(image_path) ? IMP_SUCCESS : IMP_ERROR_INTERNAL;
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
    if (ImpError e = check_vision_ready(ctx, "imp_set_image_from_memory"); e != IMP_SUCCESS)
        return e;
    try {
        return ctx->engine->set_image_from_memory(std::span(data, len)) ? IMP_SUCCESS : IMP_ERROR_INTERNAL;
    } catch (const std::exception& e) {
        IMP_LOG_ERROR("imp_set_image_from_memory: %s", e.what());
        return IMP_ERROR_INTERNAL;
    }
}

// Appending is Qwen3-VL-only: the mmproj tower encodes one image into a fixed
// token count, so a second one has nowhere to go. Reporting that beats keeping
// whichever image happened to be last.
ImpError imp_add_image(ImpContext ctx, const char* image_path) {
    if (!image_path)
        return IMP_ERROR_INVALID_ARG;
    if (ImpError e = check_vision_ready(ctx, "imp_add_image"); e != IMP_SUCCESS)
        return e;
    if (!ctx->engine->has_qwen_vision()) {
        IMP_LOG_ERROR("imp_add_image: this vision tower takes one image per request");
        return IMP_ERROR_UNSUPPORTED;
    }
    try {
        return ctx->engine->add_image(image_path) ? IMP_SUCCESS : IMP_ERROR_INTERNAL;
    } catch (const std::exception& e) {
        IMP_LOG_ERROR("imp_add_image: %s", e.what());
        return IMP_ERROR_INTERNAL;
    }
}

ImpError imp_add_image_from_memory(ImpContext ctx, const uint8_t* data, size_t len) {
    if (!data || len == 0)
        return IMP_ERROR_INVALID_ARG;
    if (ImpError e = check_vision_ready(ctx, "imp_add_image_from_memory"); e != IMP_SUCCESS)
        return e;
    if (!ctx->engine->has_qwen_vision()) {
        IMP_LOG_ERROR("imp_add_image_from_memory: this vision tower takes one image per request");
        return IMP_ERROR_UNSUPPORTED;
    }
    try {
        return ctx->engine->add_image_from_memory(std::span(data, len)) ? IMP_SUCCESS : IMP_ERROR_INTERNAL;
    } catch (const std::exception& e) {
        IMP_LOG_ERROR("imp_add_image_from_memory: %s", e.what());
        return IMP_ERROR_INTERNAL;
    }
}
