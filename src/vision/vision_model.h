#pragma once

#include "core/tensor.h"
#include <vector>

namespace imp {

class VRAMAllocator;

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
    int n_merge = 4;             // spatial avg-pool kernel (gemma3=4, gemma4v=3)
    float image_mean[3] = {0.5f, 0.5f, 0.5f};
    float image_std[3] = {0.5f, 0.5f, 0.5f};
    // gemma4v: a structurally different encoder (RMSNorm blocks, per-head q/k/v
    // norm, 2D axial NEOX RoPE, sandwich post-norms, GeGLU FFN, scale-1 attn).
    bool is_gemma4v = false;
    float rope_theta = 0.0f;  // gemma4v vision RoPE base (100.0)

    // Qwen3-VL: dynamic resolution (no fixed image_size — num_patches varies per
    // image), fused QKV, plain (non-gated) MLP, and a two-layer patch merger
    // instead of a single projection. See docs/plans/2026-07-31-qwen3-vl-vision.md.
    bool is_qwen3vl = false;
    int merge_size = 1;           // spatial merge factor (2 => 2x2 patches per token)
    int temporal_patch_size = 1;  // still images repeat along this axis
    int out_hidden_size = 0;      // merger output width (the LM's d_model)
    // Side of the LEARNED position-embedding grid (48 for Qwen3-VL, from
    // num_position_embeddings = 2304). A real image rarely has this grid, so the
    // table is resampled per image — this is the SOURCE resolution, not the
    // image's.
    int pos_embed_grid = 0;
    // Vision blocks whose hidden state is tapped for DeepStack. NOTE these index
    // VISION blocks; the LM-side injection happens at LM layers 0..n-1, a
    // different index space entirely.
    std::vector<int> deepstack_indexes;
};

// Qwen3-VL patch merger: norm -> fc1 -> GELU -> fc2. The main merger normalises
// BEFORE the 2x2 concat (norm width = hidden_size) and each DeepStack merger
// normalises AFTER it (norm width = hidden_size * merge_size^2). Upstream calls
// that flag `use_postshuffle_norm`; here the norm tensor's own width says which
// it is, so nothing needs to be remembered.
struct VisionMergerWeights {
    Tensor norm_w, norm_b;
    Tensor fc1_w, fc1_b;
    Tensor fc2_w, fc2_b;
};

struct VisionLayerWeights {
    Tensor ln1_w, ln1_b;            // pre-attention LayerNorm / RMSNorm
    Tensor wq, wk, wv;              // [hidden, hidden]
    Tensor bq, bk, bv;              // [hidden] biases
    Tensor wo, bo;                  // attention output projection + bias
    Tensor ln2_w, ln2_b;            // pre-FFN LayerNorm / RMSNorm
    Tensor ffn_up_w, ffn_up_b;      // [intermediate, hidden]
    Tensor ffn_down_w, ffn_down_b;  // [hidden, intermediate]
    // gemma4v-only
    Tensor q_norm, k_norm;                 // per-head RMSNorm weights [head_dim]
    Tensor attn_post_norm, ffn_post_norm;  // sandwich post-norms [hidden]
    Tensor ffn_gate_w;                     // GeGLU gate [intermediate, hidden]
};

struct VisionModel {
    VisionConfig config;

    // Patch embedding
    Tensor patch_embd_w;  // [hidden_size, patch_size*patch_size*3]
    Tensor patch_embd_b;  // [hidden_size]

    // Positional embedding
    Tensor position_embd;  // [num_patches, hidden_size]

    // Post-encoder LayerNorm
    Tensor post_norm_w, post_norm_b;

    // Qwen3-VL mergers: `merger` produces the image tokens the LM consumes;
    // `deepstack_mergers` are the extra taps, one per config.deepstack_indexes.
    VisionMergerWeights merger;
    std::vector<VisionMergerWeights> deepstack_mergers;

    // Multimodal projector
    Tensor mm_pre_norm_w;   // RMSNorm before linear projection
    Tensor mm_proj_w;       // [d_model, hidden_size]
    Tensor mm_proj_b;       // [d_model]
    Tensor mm_post_norm_w;  // RMSNorm after projection

    // gemma4v projector tail: per-channel affine before the pre-projection RMSNorm
    Tensor std_scale, std_bias;  // [hidden_size]

    // Transformer layers
    std::vector<VisionLayerWeights> layers;

    // GPU allocations for cleanup. Released through `allocator` when one is set
    // (the Qwen3-VL path, so the VRAM ledger stays accurate) and with a plain
    // cudaFree otherwise (the legacy GGUF mmproj path).
    std::vector<void*> gpu_allocs;
    VRAMAllocator* allocator = nullptr;

    int lm_d_model = 0;  // LLM hidden dimension (from mm_proj output)

    ~VisionModel();
    void free_gpu();
};

}  // namespace imp
