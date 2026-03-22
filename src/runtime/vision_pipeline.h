#pragma once

#include "memory/vram_allocator.h"
#include "vision/vision_model.h"
#include "vision/vision_encoder.h"
#include "core/logging.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <cstdint>

namespace imp {

class Model;

// Encapsulates the vision (multimodal) pipeline: model loading, image
// preprocessing, SigLIP encoding, and embedding buffer management.
class VisionPipeline {
public:
    VisionPipeline() = default;
    ~VisionPipeline();

    // Initialize vision encoder from mmproj GGUF. Returns false on failure.
    [[nodiscard]] bool init(const std::string& mmproj_path, int lm_d_model,
                            Model* model, VRAMAllocator& alloc,
                            cudaStream_t stream);

    // Encode an image from file path. Blocks on stream sync.
    [[nodiscard]] bool set_image(const std::string& path, cudaStream_t stream);

    // Encode an image from memory buffer. Blocks on stream sync.
    [[nodiscard]] bool set_image_from_memory(const uint8_t* data, size_t len,
                                              cudaStream_t stream);

    void clear_image() { has_input_ = false; }

    // Accessors
    bool is_available() const noexcept { return encoder_ != nullptr; }
    bool has_input() const noexcept { return has_input_; }
    half* embeddings() const noexcept { return d_embeddings_; }
    int num_image_tokens() const noexcept {
        return model_ ? model_->config.num_image_tokens : 0;
    }
    int32_t soft_token_id() const noexcept { return soft_token_id_; }
    int32_t boi_id() const noexcept { return boi_id_; }
    int32_t eoi_id() const noexcept { return eoi_id_; }
    const VisionModel* vision_model() const noexcept { return model_.get(); }

private:
    std::unique_ptr<VisionModel> model_;
    std::unique_ptr<VisionEncoder> encoder_;
    VRAMAllocator* alloc_ = nullptr;

    half* d_embeddings_ = nullptr;
    half* d_pixels_ = nullptr;
    size_t d_pixels_size_ = 0;

    bool has_input_ = false;
    int32_t soft_token_id_ = -1;
    int32_t boi_id_ = -1;
    int32_t eoi_id_ = -1;

    // Internal: upload pixels and encode
    bool encode_image(const half* h_pixels, int n_pixels, cudaStream_t stream);
};

} // namespace imp
