#include "vision/deepstack_inject.h"

#include "core/logging.h"

namespace imp {

namespace {

// One block per image token. The scan for the k-th placeholder mirrors the
// embedding-replacement kernel deliberately: both have to agree on which
// position is the k-th image token, and reading the same way is the cheapest
// way to keep them from drifting apart.
__global__ void add_vision_embeddings_kernel(half* __restrict__ hidden, const int32_t* __restrict__ token_ids,
                                             const half* __restrict__ embeddings, int vision_token_id,
                                             int n_tokens, int d_model, int n_vision_tokens, int emb_offset) {
    // Placeholder index within this call's token span; the embedding it wants
    // sits `emb_offset` further into the buffer (see the header).
    const int vision_idx = blockIdx.x;
    const int emb_idx = vision_idx + emb_offset;
    if (emb_idx >= n_vision_tokens)
        return;

    int count = 0;
    int token_pos = -1;
    for (int i = 0; i < n_tokens; ++i) {
        if (token_ids[i] == vision_token_id) {
            if (count == vision_idx) {
                token_pos = i;
                break;
            }
            ++count;
        }
    }
    if (token_pos < 0)
        return;

    // FP32 accumulation: the hidden state after a few layers and the merger's
    // output are both O(1..10), and adding them in FP16 would round twice.
    for (int d = threadIdx.x; d < d_model; d += blockDim.x) {
        const int64_t at = static_cast<int64_t>(token_pos) * d_model + d;
        hidden[at] = __float2half(__half2float(hidden[at]) +
                                  __half2float(embeddings[static_cast<int64_t>(emb_idx) * d_model + d]));
    }
}

}  // namespace

void launch_add_vision_embeddings(half* hidden, const int32_t* token_ids, const half* embeddings,
                                  int vision_token_id, int n_tokens, int d_model, int n_vision_tokens,
                                  int emb_offset, cudaStream_t stream) {
    if (n_vision_tokens <= 0 || !embeddings || !token_ids)
        return;
    // Everything from emb_offset on is still unwritten; blocks past that would
    // only read out of bounds.
    const int remaining = n_vision_tokens - emb_offset;
    if (emb_offset < 0 || remaining <= 0)
        return;
    add_vision_embeddings_kernel<<<remaining, 256, 0, stream>>>(hidden, token_ids, embeddings,
                                                                vision_token_id, n_tokens, d_model,
                                                                n_vision_tokens, emb_offset);
    IMP_CUDA_CHECK_LAUNCH();
}

}  // namespace imp
