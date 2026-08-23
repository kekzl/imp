#pragma once

#include "memory/vram_allocator.h"
#include "vision/vision_model.h"
#include "vision/vision_encoder.h"
#include "core/logging.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <span>
#include <cstdint>

namespace imp {

class Model;
struct ImageData;  // src/vision/image_processor.h

// Encapsulates the vision (multimodal) pipeline: model loading, image
// preprocessing, SigLIP encoding, and embedding buffer management.
class VisionPipeline {
public:
    VisionPipeline() = default;
    ~VisionPipeline();

    // Initialize vision encoder from mmproj GGUF. Returns false on failure.
    [[nodiscard]] bool init(const std::string& mmproj_path, int lm_d_model, Model* model,
                            cudaStream_t stream);

    // Device bytes init() takes from the T2 arena, encoder included. Answerable
    // from the probed config before the mmproj is loaded, which is what lets
    // Engine::init size the arena for it.
    static size_t demand_bytes(const VisionConfig& cfg, int lm_d_model);
    size_t taken_bytes() const;

    // Encode an image from file path. Blocks on stream sync.
    [[nodiscard]] bool set_image(const std::string& path, cudaStream_t stream);

    // Encode an image from memory buffer. Blocks on stream sync.
    [[nodiscard]] bool set_image_from_memory(std::span<const uint8_t> data, cudaStream_t stream);

    void clear_image() { has_input_ = false; }

    // --- Per-request binding API (server batched path) -------------------
    // CPU-only preprocess (decode/resize/normalize) — safe to call off the
    // batch worker (e.g. an HTTP handler thread). Fills `out` with FP16 pixels.
    [[nodiscard]] bool preprocess(std::span<const uint8_t> data, ImageData& out) const;
    // Encode a preprocessed image into the caller's `out` device buffer (sized
    // embeddings_bytes()). Encodes into the STABLE scratch d_embeddings_ first
    // (the encoder CUDA graph is keyed on the output pointer, so a per-request
    // output would force a full recapture every image) then copies to `out`.
    // Serialized: the caller MUST be the sole GPU driver (the batch worker).
    [[nodiscard]] bool encode_to(const ImageData& img, half* out, cudaStream_t stream);
    size_t embeddings_bytes() const noexcept {
        return static_cast<size_t>(num_image_tokens()) * static_cast<size_t>(lm_d_) * sizeof(half);
    }
    int lm_d() const noexcept { return lm_d_; }

    // Accessors
    bool is_available() const noexcept { return encoder_ != nullptr; }
    bool has_input() const noexcept { return has_input_; }
    half* embeddings() const noexcept { return d_embeddings_; }
    int num_image_tokens() const noexcept { return model_ ? model_->config.num_image_tokens : 0; }
    int32_t soft_token_id() const noexcept { return soft_token_id_; }
    const VisionModel* vision_model() const noexcept { return model_.get(); }

private:
    std::unique_ptr<VisionModel> model_;
    std::unique_ptr<VisionEncoder> encoder_;
    size_t taken_bytes_ = 0;

    half* d_embeddings_ = nullptr;
    half* d_pixels_ = nullptr;
    size_t d_pixels_size_ = 0;

    bool has_input_ = false;
    int lm_d_ = 0;  // LLM embedding dim (vision embeddings are [num_image_tokens, lm_d_])
    int32_t soft_token_id_ = -1;
    int32_t boi_id_ = -1;
    int32_t eoi_id_ = -1;

    // Internal: upload pixels and encode
    bool encode_image(const half* h_pixels, int n_pixels, cudaStream_t stream);
};

// Arena demand for the mmproj vision path, answerable before the file is loaded.
size_t vision_mmproj_arena_bytes(const std::string& mmproj_path, int lm_d_model);

}  // namespace imp
