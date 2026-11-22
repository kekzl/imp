#pragma once
// Inkling2 vision tower (model.visual.* of the checkpoint): ViT with a class token, learned position
// table resampled to the patch grid, axial 2D RoPE, layer scale, 2x2 patch merge into one LM token.
// Reference: transformers models/inkling2/modeling_inkling2.py (Inkling2VisionModel).

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

namespace imp {

class VRAMAllocator;

struct InklingVisionConfig {
    int layers = 23, hidden = 1024, heads = 16, patch = 20, merge = 2, grid = 32;
    int proj_hidden = 4096, out = 2304;
    float eps = 1e-5f, rope_theta = 10000.0f;
    int token_px() const { return patch * merge; }  // one LM token covers token_px x token_px pixels
};

class InklingVision {
public:
    // Largest image in LM tokens (2560 x 1440 = 64 x 36); the caller scales a bigger one down.
    static constexpr int kMaxTokens = 2304;

    InklingVision() = default;
    InklingVision(const InklingVision&) = delete;
    InklingVision& operator=(const InklingVision&) = delete;
    ~InklingVision();

    // Uploads model.visual.* from <dir>/model-*.safetensors and allocates the workspace for
    // kMaxTokens (tags inkling_vision, inkling_vision_ws). False = a tensor is missing or mis-shaped.
    [[nodiscard]] bool load(const std::string& dir, VRAMAllocator& va);
    const InklingVisionConfig& config() const { return cfg_; }
    // LM tokens of an h x w image: rows x cols cells of token_px pixels, the rest padded.
    void grid(int h, int w, int& rows, int& cols) const;

    // rgb: host [h, w, 3] bytes. out: device fp16 [rows * cols, out], raster order.
    [[nodiscard]] bool encode(const uint8_t* rgb, int h, int w, half* out, cudaStream_t s);

private:
    struct Layer {
        half *qkv = nullptr, *wo = nullptr, *fc1 = nullptr, *fc2 = nullptr;  // qkv: [3H, H] rows q, k, v
        float *qkv_b = nullptr, *wo_b = nullptr, *fc1_b = nullptr, *fc2_b = nullptr;
        float *ln1_w = nullptr, *ln1_b = nullptr, *ln2_w = nullptr, *ln2_b = nullptr;
        float *ls1 = nullptr, *ls2 = nullptr;
    };
    void lin_(const half* x, int T, int K, const half* w, int N, half* y, cudaStream_t s);
    [[nodiscard]] bool alloc_workspace_();

    InklingVisionConfig cfg_;
    VRAMAllocator* va_ = nullptr;
    std::vector<void*> owned_;
    std::vector<Layer> layers_;
    half *patch_w_ = nullptr, *proj_in_ = nullptr, *proj_out_ = nullptr;
    float *cls_ = nullptr, *pre_w_ = nullptr, *pre_b_ = nullptr;
    std::vector<float> pos_table_;  // host [grid * grid, hidden]
    // Workspace for kMaxTokens.
    float *x_ = nullptr, *pos_ = nullptr, *cs_ = nullptr;
    half *h16_ = nullptr, *qkv_ = nullptr, *q_ = nullptr, *k_ = nullptr, *vt_ = nullptr, *s_ = nullptr;
    half *o_ = nullptr, *y_ = nullptr, *fc_ = nullptr, *pe_ = nullptr, *px_ = nullptr, *m1_ = nullptr;
    std::vector<float> h_px_, h_pos_, h_cs_;  // host staging
};

}  // namespace imp
