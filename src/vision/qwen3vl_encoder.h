#pragma once

// The Qwen3-VL vision encoder forward.
//
// Unlike the fixed-resolution encoder next door, this one has no `image_size`:
// the token count comes from the image. Buffers are therefore sized once to a
// configured maximum and the forward runs over whatever prefix an image needs,
// which is also why attention is chunked over query rows — a full
// [heads, tokens, tokens] score matrix is the one buffer that grows
// quadratically and would dominate everything else.

#include "vision/qwen3vl_vision_grid.h"
#include "vision/vision_model.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <vector>

namespace imp {

class VRAMAllocator;

class Qwen3VLEncoder {
public:
    Qwen3VLEncoder() = default;
    ~Qwen3VLEncoder();
    Qwen3VLEncoder(const Qwen3VLEncoder&) = delete;
    Qwen3VLEncoder& operator=(const Qwen3VLEncoder&) = delete;

    // `model` must already be uploaded (qwen3vl_upload_vision_tower). Its
    // lifetime must outlast this encoder — only the pointer is kept.
    [[nodiscard]] bool init(const VisionModel& model, VRAMAllocator* alloc, int max_tokens);
    void free_buffers();

    int max_tokens() const { return max_tokens_; }
    // Merged image tokens produced for a `tokens`-patch image.
    int merged_tokens(int tokens) const;

    // d_patches: [grid.tokens, features] FP16 device, in the patchifier's
    //   merge-block token order.
    // d_out: [merged_tokens, out_hidden_size] FP16 device, caller-allocated.
    // d_deepstack_out: one buffer per config.deepstack_indexes, same shape as
    //   d_out. Empty is allowed and simply skips the taps.
    bool encode(const half* d_patches, const QwenVisionGrid& grid, half* d_out,
                const std::vector<half*>& d_deepstack_out, cudaStream_t stream);

private:
    bool run_merger(const VisionMergerWeights& m, const half* d_hidden, int tokens, half* d_out,
                    cudaStream_t stream);
    bool attention(int tokens, cudaStream_t stream);

    const VisionModel* model_ = nullptr;
    VRAMAllocator* alloc_ = nullptr;
    int max_tokens_ = 0;

    half* d_hidden_ = nullptr;  // [max_tokens, hidden]
    half* d_normed_ = nullptr;  // [max_tokens, hidden]
    half* d_proj_ = nullptr;    // [max_tokens, hidden]
    half* d_qkv_ = nullptr;     // [max_tokens, 3*hidden]
    half* d_q_ = nullptr;       // [heads, max_tokens, head_dim]
    half* d_k_ = nullptr;
    half* d_v_ = nullptr;
    half* d_attn_ = nullptr;        // [heads, max_tokens, head_dim]
    half* d_scores_ = nullptr;      // [heads, chunk, max_tokens]
    half* d_ffn_ = nullptr;         // [max_tokens, intermediate]
    half* d_merge_norm_ = nullptr;  // [max_tokens, hidden] (== [merged, merge^2*hidden])
    half* d_merge_fc_ = nullptr;    // [merged, merge^2*hidden]

    int32_t* d_row_ = nullptr;
    int32_t* d_col_ = nullptr;
    int32_t* d_taps_ = nullptr;
    float* d_weights_ = nullptr;

    std::vector<void*> allocs_;
};

}  // namespace imp
