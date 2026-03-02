#pragma once

#include "core/tensor.h"
#include <vector>

namespace imp {

struct VisionConfig {
    int image_size = 896;
    int patch_size = 14;
    int hidden_size = 1152;
    int intermediate_size = 4304;
    int num_layers = 27;
    int num_heads = 16;
    int head_dim = 72;           // hidden_size / num_heads
    int num_patches = 4096;      // (image_size / patch_size)^2
    int num_image_tokens = 256;  // after avg pooling
    float image_mean[3] = {0.5f, 0.5f, 0.5f};
    float image_std[3]  = {0.5f, 0.5f, 0.5f};
};

struct VisionLayerWeights {
    Tensor ln1_w, ln1_b;           // pre-attention LayerNorm
    Tensor wq, wk, wv;            // [hidden, hidden]
    Tensor bq, bk, bv;            // [hidden] biases
    Tensor wo, bo;                 // attention output projection + bias
    Tensor ln2_w, ln2_b;           // pre-FFN LayerNorm
    Tensor ffn_up_w, ffn_up_b;     // [intermediate, hidden]
    Tensor ffn_down_w, ffn_down_b; // [hidden, intermediate]
};

struct VisionModel {
    VisionConfig config;

    // Patch embedding
    Tensor patch_embd_w;    // [hidden_size, patch_size*patch_size*3]
    Tensor patch_embd_b;    // [hidden_size]

    // Positional embedding
    Tensor position_embd;   // [num_patches, hidden_size]

    // Post-encoder LayerNorm
    Tensor post_norm_w, post_norm_b;

    // Multimodal projector
    Tensor mm_pre_norm_w;   // RMSNorm before linear projection
    Tensor mm_proj_w;       // [d_model, hidden_size]
    Tensor mm_proj_b;       // [d_model]
    Tensor mm_post_norm_w;  // RMSNorm after projection

    // Transformer layers
    std::vector<VisionLayerWeights> layers;

    // GPU allocations for cleanup
    std::vector<void*> gpu_allocs;

    int lm_d_model = 0;    // LLM hidden dimension (from mm_proj output)

    ~VisionModel();
    void free_gpu();
};

} // namespace imp
