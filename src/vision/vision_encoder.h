#pragma once

#include "memory/span.h"
#include "runtime/cuda_graph.h"
#include "vision/vision_model.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace imp {

class VisionEncoder {
public:
    VisionEncoder() = default;
    ~VisionEncoder();

    // Initialize workspace buffers. lm_d_model = LLM hidden dim.
    [[nodiscard]] bool init(const VisionModel& model, int lm_d_model, cudaStream_t stream);

    // Device bytes init() takes from the T2 arena, answerable from config alone so Engine::init
    // can size the arena before the mmproj loads. taken_bytes() reports what was actually taken;
    // a test asserts they agree, since nothing else stops them drifting.
    static size_t demand_bytes(const VisionConfig& cfg);
    size_t taken_bytes() const { return taken_bytes_; }

    // Encode a preprocessed image.
    // d_pixels: [3, image_size, image_size] FP16 on device
    // d_output: [num_image_tokens, lm_d_model] FP16 on device (caller-allocated)
    [[nodiscard]] bool encode(const half* d_pixels, half* d_output, cudaStream_t stream);

private:
    size_t taken_bytes_ = 0;
    const VisionModel* model_ = nullptr;
    int lm_d_model_ = 0;

    // Workspace buffers (pre-allocated, reused per encode)
    StableSpan<half> d_patches_;      // [num_patches, patch_dim]
    StableSpan<half> d_hidden_;       // [num_patches, hidden_size]
    StableSpan<half> d_residual_;     // [num_patches, hidden_size]
    StableSpan<half> d_q_;            // [num_patches, hidden_size]
    StableSpan<half> d_k_;            // [num_patches, hidden_size]
    StableSpan<half> d_v_;            // [num_patches, hidden_size]
    StableSpan<half> d_attn_out_;     // [num_patches, hidden_size]
    StableSpan<half> d_attn_scores_;  // [num_heads, num_patches, num_patches]
    StableSpan<half> d_ffn_;          // [num_patches, intermediate_size]
    StableSpan<half> d_pooled_;       // [num_image_tokens, hidden_size]
    StableSpan<half> d_gate_;         // [num_patches, intermediate_size] (gemma4v GeGLU gate)
    StableSpan<int> d_pos_x_;         // [num_patches] axial column index (gemma4v)
    StableSpan<int> d_pos_y_;         // [num_patches] axial row index (gemma4v)

    // CUDA graph for the full encoder forward: topology fixed once model/workspace sizes are
    // known, only input/output pointers change across calls. Invalidated when those pointers
    // differ from the ones baked in at capture time.
    CudaGraphRunner encode_graph_;
    const half* graph_d_pixels_ = nullptr;
    half* graph_d_output_ = nullptr;

    [[nodiscard]] bool encode_impl(const half* d_pixels, half* d_output, cudaStream_t stream);
    [[nodiscard]] bool encode_impl_gemma4v(const half* d_pixels, half* d_output, cudaStream_t stream);

    void free_buffers();
};

}  // namespace imp
