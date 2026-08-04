#pragma once

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

    // Device bytes init() takes from the T2 arena, answerable from config alone so
    // Engine::init can size the arena before the mmproj is loaded. taken_bytes()
    // reports what was actually taken; a test asserts they agree, because these are
    // two expressions of one buffer list and nothing else stops them drifting.
    static size_t demand_bytes(const VisionConfig& cfg);
    size_t taken_bytes() const { return taken_bytes_; }

    // Encode a preprocessed image.
    // d_pixels: [3, image_size, image_size] FP16 on device
    // d_output: [num_image_tokens, lm_d_model] FP16 on device (caller-allocated)
    bool encode(const half* d_pixels, half* d_output, cudaStream_t stream);

private:
    size_t taken_bytes_ = 0;
    const VisionModel* model_ = nullptr;
    int lm_d_model_ = 0;

    // Workspace buffers (pre-allocated, reused per encode)
    half* d_patches_ = nullptr;      // [num_patches, patch_dim]
    half* d_hidden_ = nullptr;       // [num_patches, hidden_size]
    half* d_residual_ = nullptr;     // [num_patches, hidden_size]
    half* d_q_ = nullptr;            // [num_patches, hidden_size]
    half* d_k_ = nullptr;            // [num_patches, hidden_size]
    half* d_v_ = nullptr;            // [num_patches, hidden_size]
    half* d_attn_out_ = nullptr;     // [num_patches, hidden_size]
    half* d_attn_scores_ = nullptr;  // [num_heads, num_patches, num_patches]
    half* d_ffn_ = nullptr;          // [num_patches, intermediate_size]
    half* d_pooled_ = nullptr;       // [num_image_tokens, hidden_size]
    half* d_gate_ = nullptr;         // [num_patches, intermediate_size] (gemma4v GeGLU gate)
    int* d_pos_x_ = nullptr;         // [num_patches] axial column index (gemma4v)
    int* d_pos_y_ = nullptr;         // [num_patches] axial row index (gemma4v)

    // CUDA graph for the full encoder forward. Topology is fixed once model
    // and workspace sizes are known; only the input/output pointers change
    // across calls. The graph is invalidated when those pointers differ from
    // the ones baked in at capture time.
    CudaGraphRunner encode_graph_;
    const half* graph_d_pixels_ = nullptr;
    half* graph_d_output_ = nullptr;

    bool encode_impl(const half* d_pixels, half* d_output, cudaStream_t stream);
    bool encode_impl_gemma4v(const half* d_pixels, half* d_output, cudaStream_t stream);

    void free_buffers();
};

}  // namespace imp
