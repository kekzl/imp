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

// hidden[p, :] += embeddings[emb_offset + k, :] for the k-th token in THIS call
// whose id is `vision_token_id`. `n_vision_tokens` bounds `emb_offset + k`, so a
// prompt with more placeholders than the encoder produced leaves the surplus
// untouched rather than reading past the buffer.
//
// `emb_offset` is how many image tokens the caller already consumed. Under
// chunked prefill `token_ids` is one CHUNK, so the k-th placeholder in a later
// chunk is NOT the k-th of the image — without the offset a run of image tokens
// that straddles a chunk boundary gets the FIRST embeddings again in the second
// chunk, which is silently the wrong picture rather than a crash.
void launch_add_vision_embeddings(half* hidden, const int32_t* token_ids, const half* embeddings,
                                  int vision_token_id, int n_tokens, int d_model, int n_vision_tokens,
                                  int emb_offset, cudaStream_t stream);

}  // namespace imp
