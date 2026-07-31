#pragma once

// Image bytes to LM-ready embeddings, for Qwen3-VL.
//
// Separate from `VisionPipeline` (the GGUF mmproj path) because the two differ
// in the thing that shapes everything else: that one has a fixed
// `num_image_tokens` from config, this one has none. The token count comes from
// the image, so buffers are sized to a patch budget and each image uses a
// prefix of them.
//
// The tower's weights live in the Model — they came from the same checkpoint —
// so this holds a reference, not ownership, and uploads them to the device on
// first init.

#include "vision/qwen3vl_encoder.h"
#include "vision/vision_model.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace imp {

class VRAMAllocator;

// One encoded image. The buffers belong to the pipeline and stay valid until
// the next encode call.
struct Qwen3VLImage {
    int grid_rows = 0;  // merged grid — LM tokens, not patches
    int grid_cols = 0;
    int tokens = 0;  // grid_rows * grid_cols
    const half* d_embeddings = nullptr;
    // One per `config.deepstack_indexes`, same shape as `d_embeddings`. These
    // are added into the LM's hidden state at its FIRST layers, not at the
    // vision blocks they were tapped from.
    std::vector<const half*> d_deepstack;
};

class Qwen3VLPipeline {
public:
    Qwen3VLPipeline() = default;
    ~Qwen3VLPipeline();
    Qwen3VLPipeline(const Qwen3VLPipeline&) = delete;
    Qwen3VLPipeline& operator=(const Qwen3VLPipeline&) = delete;

    // `tower` must be a loaded Qwen3-VL vision tower; it is uploaded to the
    // device here if it is still host-resident. `max_patches` bounds the image
    // size this pipeline will accept — it sizes every workspace, so it is also
    // what an oversized image is rejected against.
    bool init(VisionModel& tower, VRAMAllocator& alloc, int max_patches);

    bool is_ready() const { return encoder_ != nullptr; }
    int max_patches() const { return max_patches_; }
    // Largest image, in pixels, this pipeline's patch budget allows.
    int64_t max_pixels() const;

    bool encode_file(const std::string& path, Qwen3VLImage& out, cudaStream_t stream);
    bool encode_memory(const uint8_t* data, size_t len, Qwen3VLImage& out, cudaStream_t stream);

private:
    bool encode_rgb(const uint8_t* rgb, int width, int height, Qwen3VLImage& out, cudaStream_t stream);
    void free_buffers();

    VisionModel* tower_ = nullptr;
    VRAMAllocator* alloc_ = nullptr;
    std::unique_ptr<Qwen3VLEncoder> encoder_;
    int max_patches_ = 0;
    // Whether THIS pipeline uploaded the tower, and so has to invalidate it.
    bool uploaded_tower_ = false;

    half* d_patches_ = nullptr;
    half* d_out_ = nullptr;
    std::vector<half*> d_deepstack_;
    std::vector<void*> allocs_;
};

}  // namespace imp
