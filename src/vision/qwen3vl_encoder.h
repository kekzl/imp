#pragma once

// Qwen3-VL vision encoder forward: unlike the fixed-resolution encoder, this has no
// image_size, token count comes from the image. Buffers sized once to a configured maximum;
// forward runs over whatever prefix an image needs, and attention is chunked over query
// rows since a full [heads,tokens,tokens] score matrix is the one buffer that grows
// quadratically.

#include "memory/span.h"
#include "vision/qwen3vl_vision_grid.h"
#include "vision/vision_model.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <vector>

namespace imp {


class Qwen3VLEncoder {
public:
    Qwen3VLEncoder() = default;
    ~Qwen3VLEncoder();
    Qwen3VLEncoder(const Qwen3VLEncoder&) = delete;
    Qwen3VLEncoder& operator=(const Qwen3VLEncoder&) = delete;

    // `model` must already be uploaded (qwen3vl_upload_vision_tower). Its
    // lifetime must outlast this encoder — only the pointer is kept.
    [[nodiscard]] bool init(const VisionModel& model, int max_tokens);
    void free_buffers();

    // Device bytes init() takes from the T2 arena, answerable before the arena opens (reads
    // config, not weights). init() records what it actually took in taken_bytes(); the two are
    // asserted equal in the encoder tests, so a buffer added to init() without updating this
    // shows as a test failure rather than arena exhaustion elsewhere.
    static size_t demand_bytes(const VisionConfig& c, int max_tokens);
    size_t taken_bytes() const { return taken_bytes_; }

    int max_tokens() const { return max_tokens_; }
    // Merged image tokens produced for a `tokens`-patch image.
    int merged_tokens(int tokens) const;

    // d_patches: [grid.tokens,features] FP16 device, patchifier's merge-block token order.
    // d_out: [merged_tokens,out_hidden_size] FP16 device, caller-allocated. d_deepstack_out:
    // one buffer per config.deepstack_indexes, same shape as d_out; empty skips the taps.
    [[nodiscard]] bool encode(const half* d_patches, const QwenVisionGrid& grid, half* d_out,
                const std::vector<half*>& d_deepstack_out, cudaStream_t stream);

private:
    [[nodiscard]] bool run_merger(const VisionMergerWeights& m, const half* d_hidden, int tokens, half* d_out,
                    cudaStream_t stream);
    [[nodiscard]] bool attention(int tokens, cudaStream_t stream);

    const VisionModel* model_ = nullptr;
    int max_tokens_ = 0;
    size_t taken_bytes_ = 0;

    StableSpan<half> d_hidden_;  // [max_tokens, hidden]
    StableSpan<half> d_normed_;  // [max_tokens, hidden]
    StableSpan<half> d_proj_;    // [max_tokens, hidden]
    StableSpan<half> d_qkv_;     // [max_tokens, 3*hidden]
    StableSpan<half> d_q_;       // [heads, max_tokens, head_dim]
    StableSpan<half> d_k_;
    StableSpan<half> d_v_;
    StableSpan<half> d_attn_;        // [heads, max_tokens, head_dim]
    StableSpan<half> d_scores_;      // [heads, chunk, max_tokens]
    StableSpan<half> d_ffn_;         // [max_tokens, intermediate]
    StableSpan<half> d_merge_norm_;  // [max_tokens, hidden] (== [merged, merge^2*hidden])
    StableSpan<half> d_merge_fc_;    // [merged, merge^2*hidden]

    StableSpan<int32_t> d_row_;
    StableSpan<int32_t> d_col_;
    StableSpan<int32_t> d_taps_;
    StableSpan<float> d_weights_;
};

}  // namespace imp
