#include "runtime/vision_pipeline.h"
#include "vision/vision_loader.h"
#include "vision/image_processor.h"
#include "model/model.h"

namespace imp {

VisionPipeline::~VisionPipeline() {
    if (d_embeddings_ && alloc_) {
        alloc_->free(d_embeddings_);
        d_embeddings_ = nullptr;
    }
    if (d_pixels_ && alloc_) {
        alloc_->free(d_pixels_);
        d_pixels_ = nullptr;
    }
}

bool VisionPipeline::init(const std::string& mmproj_path, int lm_d_model,
                           Model* text_model, VRAMAllocator& alloc,
                           cudaStream_t stream) {
    alloc_ = &alloc;

    model_ = load_vision_gguf(mmproj_path);
    if (!model_) {
        IMP_LOG_ERROR("Failed to load vision model: %s", mmproj_path.c_str());
        return false;
    }

    int lm_d = model_->lm_d_model > 0 ? model_->lm_d_model : lm_d_model;
    encoder_ = std::make_unique<VisionEncoder>();
    if (!encoder_->init(*model_, lm_d, stream, &alloc)) {
        IMP_LOG_ERROR("Failed to init vision encoder");
        encoder_.reset();
        model_.reset();
        return false;
    }

    // Allocate device buffer for vision embeddings
    int n_img_tokens = model_->config.num_image_tokens;
    size_t emb_bytes = static_cast<size_t>(n_img_tokens) * lm_d * sizeof(half);
    d_embeddings_ = static_cast<half*>(alloc.allocate(emb_bytes, "vision_embeddings"));
    if (!d_embeddings_) {
        IMP_LOG_ERROR("Failed to allocate vision embedding buffer (%zu bytes)", emb_bytes);
        encoder_.reset();
        model_.reset();
        return false;
    }

    // Pre-allocate pixel buffer to avoid per-image cudaMalloc/Free
    int img_sz = model_->config.image_size;
    d_pixels_size_ = static_cast<size_t>(3) * img_sz * img_sz * sizeof(half);
    d_pixels_ = static_cast<half*>(alloc.allocate(d_pixels_size_, "vision_pixels"));
    if (!d_pixels_) {
        IMP_LOG_WARN("Failed to pre-allocate vision pixel buffer (%.1f MiB), will alloc per-image",
                     d_pixels_size_ / (1024.0 * 1024.0));
        d_pixels_size_ = 0;
    }

    // Resolve vision special token IDs
    Tokenizer* tok = text_model->tokenizer();
    if (tok) {
        const auto& mcfg = text_model->config();
        soft_token_id_ = tok->find_token("<image_soft_token>");
        if (soft_token_id_ < 0) {
            if (mcfg.vocab_size > 262144) {
                soft_token_id_ = 262144;
            }
        }
        boi_id_ = tok->find_token("<start_of_image>");
        if (boi_id_ < 0 && mcfg.vocab_size > 255999)
            boi_id_ = 255999;
        eoi_id_ = tok->find_token("<end_of_image>");
        if (eoi_id_ < 0 && mcfg.vocab_size > 256000)
            eoi_id_ = 256000;
        IMP_LOG_INFO("Vision tokens: soft=%d, boi=%d, eoi=%d",
                     soft_token_id_, boi_id_, eoi_id_);
    }

    IMP_LOG_INFO("Vision encoder ready: %d image tokens -> %d-dim embeddings",
                 n_img_tokens, lm_d);
    return true;
}

bool VisionPipeline::encode_image(const half* h_pixels, int n_pixels,
                                   cudaStream_t stream) {
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
    cudaMemcpyAsync(d_px, h_pixels, pixel_bytes,
                    cudaMemcpyHostToDevice, stream);

    bool ok = encoder_->encode(d_px, d_embeddings_, stream);
    cudaStreamSynchronize(stream);
    if (need_free) cudaFree(d_px);

    if (ok) {
        has_input_ = true;
        IMP_LOG_INFO("Vision: encoded image -> %d tokens",
                     model_->config.num_image_tokens);
    }
    return ok;
}

bool VisionPipeline::set_image(const std::string& path, cudaStream_t stream) {
    if (!encoder_) {
        IMP_LOG_ERROR("set_image: no vision model loaded (missing --mmproj)");
        return false;
    }

    ImageData img;
    if (!load_and_preprocess_image(path, model_->config.image_size,
                                    model_->config.image_mean,
                                    model_->config.image_std, img)) {
        return false;
    }

    int n_pixels = 3 * img.width * img.height;
    return encode_image(img.pixels.data(), n_pixels, stream);
}

bool VisionPipeline::set_image_from_memory(const uint8_t* data, size_t len,
                                            cudaStream_t stream) {
    if (!encoder_) {
        IMP_LOG_ERROR("set_image_from_memory: no vision model loaded");
        return false;
    }

    ImageData img;
    if (!load_and_preprocess_image_from_memory(data, len,
                                                model_->config.image_size,
                                                model_->config.image_mean,
                                                model_->config.image_std, img)) {
        return false;
    }

    int n_pixels = 3 * img.width * img.height;
    return encode_image(img.pixels.data(), n_pixels, stream);
}

} // namespace imp
