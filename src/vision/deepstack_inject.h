#pragma once

// DeepStack: extra visual features added into the LM's hidden state after its
// FIRST few layers, at image-token positions only.
//
// Two things about this are easy to get backwards, and neither fails loudly:
//
//   - The taps come from vision blocks 5/11/17, but they are injected at LM
//     layers 0/1/2. Two different index spaces; upstream only ever writes
//     `layer_idx in range(len(embeds))`.
//   - It is an ADD onto the existing hidden state, not a replace. The initial
//     image embedding was already written at those positions by the embedding
//     replacement; this stacks on top of it.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace imp {

// hidden[p, :] += embeddings[k, :] for the k-th token whose id is
// `vision_token_id`. `n_vision_tokens` bounds k, so a prompt with more
// placeholders than the encoder produced leaves the surplus untouched rather
// than reading past the buffer.
void launch_add_vision_embeddings(half* hidden, const int32_t* token_ids, const half* embeddings,
                                  int vision_token_id, int n_tokens, int d_model, int n_vision_tokens,
                                  cudaStream_t stream);

}  // namespace imp
