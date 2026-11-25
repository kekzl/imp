#pragma once

// InternViT + pixel shuffle + projector (HF InternVL3.5): one fixed-size image -> num_image_tokens
// LM embeddings. Weights from internvl_vision.h, uploaded through the shared tower upload.

#include "memory/span.h"
#include "vision/vision_model.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <vector>

namespace imp {

// Optional device copies of intermediate stages (tests): embeddings and each layer output are
// [1 + patches, hidden], shuffled is [tokens, 4*hidden]. A null pointer skips that stage.
struct InternVLStageTaps {
    half* embeddings = nullptr;
    std::vector<half*> layers;
    half* shuffled = nullptr;
};

class InternVLEncoder {
public:
    InternVLEncoder() = default;
    ~InternVLEncoder();
    InternVLEncoder(const InternVLEncoder&) = delete;
    InternVLEncoder& operator=(const InternVLEncoder&) = delete;

    // `model` uploaded and outliving this encoder. Buffers come from the engine arena.
    [[nodiscard]] bool init(const VisionModel& model);
    void free_buffers();
    static size_t demand_bytes(const VisionConfig& c);
    size_t taken_bytes() const { return taken_bytes_; }

    // d_patches: [patches, 3*P*P] FP16, raster patch order, (channel, row, col) inside a patch
    // (the Conv2d weight's flatten order). d_out: [num_image_tokens, out_hidden_size] FP16.
    [[nodiscard]] bool encode(const half* d_patches, half* d_out, cudaStream_t stream,
                              const InternVLStageTaps* taps = nullptr);

private:
    [[nodiscard]] bool attention(int tokens, cudaStream_t stream);

    const VisionModel* model_ = nullptr;
    size_t taken_bytes_ = 0;
    StableSpan<half> d_hidden_, d_normed_, d_proj_, d_qkv_;  // engine arena slices
    StableSpan<half> d_q_, d_k_, d_v_, d_attn_, d_scores_;
    StableSpan<half> d_ffn_, d_shuffled_, d_merge_norm_, d_merge_fc_;
};

}  // namespace imp
