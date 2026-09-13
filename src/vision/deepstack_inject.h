#pragma once

// DeepStack: extra visual features added into the LM hidden state after its first few
// layers, at image-token positions only. Taps come from vision blocks 5/11/17 but inject at
// LM layers 0/1/2 (two different index spaces). It is an ADD onto the existing hidden state
// (already holding the initial image embedding), not a replace.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace imp {

// hidden[p,:] += embeddings[emb_offset+k,:] for the k-th token in THIS call with id
// vision_token_id. n_vision_tokens bounds emb_offset+k, so extra placeholders beyond what
// the encoder produced are left untouched. emb_offset = image tokens already consumed:
// under chunked prefill a later chunk's k-th placeholder is not the image's k-th, so without
// the offset a straddling run gets the FIRST embeddings again in the second chunk (wrong
// picture, not a crash).
void launch_add_vision_embeddings(half* hidden, const int32_t* token_ids, const half* embeddings,
                                  int vision_token_id, int n_tokens, int d_model, int n_vision_tokens,
                                  int emb_offset, cudaStream_t stream);

}  // namespace imp
