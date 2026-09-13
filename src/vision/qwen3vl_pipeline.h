#pragma once

// Image bytes to LM-ready embeddings, Qwen3-VL. Separate from VisionPipeline (GGUF mmproj
// path) because that has a fixed num_image_tokens from config while this has none (token
// count comes from the image, buffers sized to a patch budget). Tower weights live in the
// Model (same checkpoint); this holds a reference, not ownership, uploading on first init.

#include "vision/image_processor.h"  // QwenPatches
#include "vision/qwen3vl_encoder.h"
#include "vision/vision_model.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <span>

namespace imp {


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

    // `tower` must be a loaded Qwen3-VL vision tower, uploaded here if still host-resident.
    // max_patches bounds accepted image size, sizing every workspace and rejecting oversized
    // images. lazy: validates+remembers the tower, defers upload and arena takes to first
    // encode (vram.lazy_commit); is_ready() is true from init either way, false only if a
    // deferred build later fails.
    [[nodiscard]] bool init(VisionModel& tower, int max_patches, bool lazy = false);

    // Device bytes init() takes from the T2 arena, tower excluded (that's
    // qwen3vl_vision_tower_device_bytes()). Answerable before the arena opens (reads
    // config/shapes, not weights); taken_bytes() reports what was actually taken, asserted
    // equal in the pipeline tests.
    static size_t demand_bytes(const VisionModel& tower, int max_patches);

    // The patch budget the engine will actually use, from the configured value (0=default).
    // Engine::init sizes the arena with this and vision warmup initialises with it; routing
    // through one function keeps reservation and allocation from drifting apart.
    static int patch_budget(const VisionModel& tower, int configured);
    // Includes the encoder's slices: demand_bytes() covers both, so the pair the
    // drift test compares has to cover both as well.
    size_t taken_bytes() const;

    bool is_ready() const { return configured_; }
    int max_patches() const { return max_patches_; }
    // Largest image, in pixels, this pipeline's patch budget allows.
    int64_t max_pixels() const;

    bool encode_file(const std::string& path, Qwen3VLImage& out, cudaStream_t stream);

    // Per-request path (the server): CPU only, decode+resize+patchify. Safe off the batch
    // worker (an HTTP handler thread); tells the caller how many image tokens the prompt must
    // reserve, BEFORE any GPU work happens.
    bool preprocess(std::span<const uint8_t> data, QwenPatches& out) const;
    // Image tokens a patchified image becomes.
    int merged_tokens_of(const QwenPatches& p) const;
    // Bytes one request's embedding buffer needs (same for each DeepStack tap).
    size_t embedding_bytes(int tokens) const;
    // Elements per image token — the stride a caller concatenating several
    // images into one buffer has to advance by.
    int embedding_dim() const;
    int deepstack_taps() const;

    // Encodes into the CALLER's buffers via the pipeline's own stable scratch first, then
    // copies out: the encoder's workspaces are sized once and shared, a per-request output
    // would force re-sizing per image. Serialized: caller must be the sole GPU driver (the
    // batch worker).
    bool encode_patches_to(const QwenPatches& patches, half* d_out, const std::vector<half*>& d_deepstack,
                           Qwen3VLImage& shape_out, cudaStream_t stream);

private:
    bool encode_rgb(const uint8_t* rgb, int width, int height, Qwen3VLImage& out, cudaStream_t stream);
    bool encode_patches(const QwenPatches& patches, Qwen3VLImage& out, cudaStream_t stream);
    QwenPatchifyConfig patchify_config() const;
    void free_buffers();
    bool build_();
    bool ensure_ready_();

    VisionModel* tower_ = nullptr;
    size_t taken_bytes_ = 0;
    std::unique_ptr<Qwen3VLEncoder> encoder_;
    int max_patches_ = 0;
    // Whether THIS pipeline uploaded the tower, and so has to invalidate it.
    bool uploaded_tower_ = false;
    bool configured_ = false;
    bool lazy_ = false;
    std::mutex ready_mu_;

    half* d_patches_ = nullptr;
    half* d_out_ = nullptr;
    std::vector<half*> d_deepstack_;
    std::vector<void*> allocs_;
};

}  // namespace imp
