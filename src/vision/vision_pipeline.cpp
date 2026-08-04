#include "vision/vision_pipeline.h"
#include "memory/engine_arena.h"
#include "vision/vision_loader.h"
#include "vision/image_processor.h"
#include "model/model.h"

#include <cmath>
#include <vector>

namespace imp {

// Arena slices: the arena releases wholesale on close, so this only drops
// pointers (F-12).
VisionPipeline::~VisionPipeline() {
    d_embeddings_ = nullptr;
    d_pixels_ = nullptr;
}

// Everything the T2 arena owes the mmproj (Gemma-3 / Gemma-4v) path: the SigLIP
// tower plus the pipeline and encoder workspace. Engine::init asks this one
// question before the arena opens; the probe runs the real loader in counting
// mode, so the answer cannot drift from what the warmup then takes.
size_t vision_mmproj_arena_bytes(const std::string& mmproj_path, int lm_d_model) {
    VisionConfig cfg;
    int probed_lm_d = 0;
    const size_t tower = vision_gguf_probe(mmproj_path, &cfg, &probed_lm_d);
    if (tower == 0)
        return 0;
    const int lm_d = probed_lm_d > 0 ? probed_lm_d : lm_d_model;
    return tower + VisionPipeline::demand_bytes(cfg, lm_d);
}

size_t VisionPipeline::taken_bytes() const {
    return taken_bytes_ + (encoder_ ? encoder_->taken_bytes() : 0);
}

size_t VisionPipeline::demand_bytes(const VisionConfig& cfg, int lm_d_model) {
    size_t total = static_cast<size_t>(cfg.num_image_tokens) * lm_d_model * sizeof(half);  // embeddings
    total += static_cast<size_t>(3) * cfg.image_size * cfg.image_size * sizeof(half);  // pixels
    return total + VisionEncoder::demand_bytes(cfg);
}

bool VisionPipeline::init(const std::string& mmproj_path, int lm_d_model, Model* text_model,
                          cudaStream_t stream) {
    taken_bytes_ = 0;

    model_ = load_vision_gguf(mmproj_path);
    if (!model_) {
        IMP_LOG_ERROR("Failed to load vision model: %s", mmproj_path.c_str());
        return false;
    }

    int lm_d = model_->lm_d_model > 0 ? model_->lm_d_model : lm_d_model;
    lm_d_ = lm_d;
    encoder_ = std::make_unique<VisionEncoder>();
    if (!encoder_->init(*model_, lm_d, stream)) {
        IMP_LOG_ERROR("Failed to init vision encoder");
        encoder_.reset();
        model_.reset();
        return false;
    }

    // Allocate device buffer for vision embeddings
    int n_img_tokens = model_->config.num_image_tokens;
    size_t emb_bytes = static_cast<size_t>(n_img_tokens) * lm_d * sizeof(half);
    {
        auto slab = engine_arena().take_bytes(emb_bytes);
        d_embeddings_ = slab.empty() ? nullptr : reinterpret_cast<half*>(slab.data());
        if (d_embeddings_)
            taken_bytes_ += emb_bytes;
    }
    if (!d_embeddings_) {
        IMP_LOG_ERROR("Failed to allocate vision embedding buffer (%zu bytes)", emb_bytes);
        encoder_.reset();
        model_.reset();
        return false;
    }

    // Pre-allocate pixel buffer to avoid per-image cudaMalloc/Free
    int img_sz = model_->config.image_size;
    d_pixels_size_ = static_cast<size_t>(3) * img_sz * img_sz * sizeof(half);
    {
        auto slab = engine_arena().take_bytes(d_pixels_size_);
        d_pixels_ = slab.empty() ? nullptr : reinterpret_cast<half*>(slab.data());
        if (d_pixels_)
            taken_bytes_ += d_pixels_size_;
    }
    if (!d_pixels_) {
        IMP_LOG_WARN("Failed to pre-allocate vision pixel buffer (%.1f MiB), will alloc per-image",
                     d_pixels_size_ / (1024.0 * 1024.0));
        d_pixels_size_ = 0;
    }

    // Resolve vision special token IDs
    Tokenizer* tok = text_model->tokenizer();
    if (tok) {
        const auto& mcfg = text_model->config();
        // Gemma-3: <image_soft_token> / <start_of_image> / <end_of_image>.
        // Gemma-4: <|image|> (repeated soft) / <|image> (begin) / <image|> (end).
        soft_token_id_ = tok->find_token("<image_soft_token>");
        if (soft_token_id_ < 0)
            soft_token_id_ = tok->find_token("<|image|>");
        if (soft_token_id_ < 0 && mcfg.vocab_size > 262144) {
            soft_token_id_ = 262144;
        }
        boi_id_ = tok->find_token("<start_of_image>");
        if (boi_id_ < 0)
            boi_id_ = tok->find_token("<|image>");
        if (boi_id_ < 0 && mcfg.vocab_size > 255999)
            boi_id_ = 255999;
        eoi_id_ = tok->find_token("<end_of_image>");
        if (eoi_id_ < 0)
            eoi_id_ = tok->find_token("<image|>");
        if (eoi_id_ < 0 && mcfg.vocab_size > 256000)
            eoi_id_ = 256000;
        IMP_LOG_INFO("Vision tokens: soft=%d, boi=%d, eoi=%d", soft_token_id_, boi_id_, eoi_id_);
    }

    IMP_LOG_INFO("Vision encoder ready: %d image tokens -> %d-dim embeddings", n_img_tokens, lm_d);
    return true;
}

bool VisionPipeline::encode_image(const half* h_pixels, int n_pixels, cudaStream_t stream) {
    size_t pixel_bytes = static_cast<size_t>(n_pixels) * sizeof(half);
    half* d_px = d_pixels_;
    bool need_free = false;
    if (!d_px || pixel_bytes > d_pixels_size_) {
        if (cudaMalloc(&d_px, pixel_bytes) != cudaSuccess) {
            IMP_LOG_ERROR("Vision: cudaMalloc failed for %d pixels", n_pixels);
            return false;
        }
        need_free = true;
    }
    cudaMemcpyAttributes attrs{};
    attrs.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
    attrs.srcLocHint     = { cudaMemLocationTypeHostNumaCurrent, {0} };
    attrs.dstLocHint     = { cudaMemLocationTypeDevice, {0} };
    attrs.flags          = 0;
    cudaMemcpyWithAttributesAsync(d_px, h_pixels, pixel_bytes, &attrs, stream);

    bool ok = encoder_->encode(d_px, d_embeddings_, stream);
    cudaStreamSynchronize(stream);
    if (need_free)
        cudaFree(d_px);

    if (ok) {
        has_input_ = true;
        IMP_LOG_INFO("Vision: encoded image -> %d tokens", model_->config.num_image_tokens);
    }
    return ok;
}

bool VisionPipeline::preprocess(const uint8_t* data, size_t len, ImageData& out) const {
    if (!model_)
        return false;
    return load_and_preprocess_image_from_memory(data, len, model_->config.image_size,
                                                  model_->config.image_mean, model_->config.image_std, out);
}

bool VisionPipeline::encode_to(const ImageData& img, half* out, cudaStream_t stream) {
    if (!encoder_ || !d_embeddings_ || !out || img.pixels.empty())
        return false;
    // Encode into the stable scratch d_embeddings_ (the encoder CUDA graph is
    // keyed on the output pointer — encoding straight into a per-request buffer
    // would recapture ~200 kernels every image), then copy to the caller buffer.
    if (!encode_image(img.pixels.data(), static_cast<int>(img.pixels.size()), stream))
        return false;
    cudaMemcpyAsync(out, d_embeddings_, embeddings_bytes(), cudaMemcpyDeviceToDevice, stream);
    cudaStreamSynchronize(stream);
    // encode_image sets the legacy global has_input_ flag as a side effect; the
    // per-request path doesn't use it, so clear it to avoid leaking global state.
    has_input_ = false;
    return true;
}

bool VisionPipeline::set_image(const std::string& path, cudaStream_t stream) {
    if (!encoder_) {
        IMP_LOG_ERROR("set_image: no vision model loaded (missing --mmproj)");
        return false;
    }

    ImageData img;
    if (!load_and_preprocess_image(path, model_->config.image_size, model_->config.image_mean,
                                   model_->config.image_std, img)) {
        return false;
    }

    int n_pixels = 3 * img.width * img.height;
    return encode_image(img.pixels.data(), n_pixels, stream);
}

bool VisionPipeline::set_image_from_memory(const uint8_t* data, size_t len, cudaStream_t stream) {
    if (!encoder_) {
        IMP_LOG_ERROR("set_image_from_memory: no vision model loaded");
        return false;
    }

    ImageData img;
    if (!load_and_preprocess_image_from_memory(data, len, model_->config.image_size,
                                               model_->config.image_mean, model_->config.image_std, img)) {
        return false;
    }

    int n_pixels = 3 * img.width * img.height;
    return encode_image(img.pixels.data(), n_pixels, stream);
}

}  // namespace imp
