// Image tokens under CHUNKED prefill.
//
// Both vision kernels are handed one chunk at a time and locate "the k-th
// placeholder" by scanning the token ids they were given. That span is the
// chunk, not the prompt, so on its own the k-th placeholder of a second chunk
// looks like the k-th placeholder of the image. It is not — and taking the
// image's first embeddings again is the wrong region of the picture, produced
// silently: no error, no crash, a model that still answers.
//
// Chunked prefill is reachable with defaults here (chunk size 2048, and
// `supports_chunked_prefill_` admits Qwen3 with FP8 KV, which is the default KV
// dtype for that family), so a prompt with enough text before its image puts an
// image run across a boundary.
//
// The guard is `emb_offset`: how many image tokens earlier chunks consumed.
// Both kernels have to agree on it, which is why they are tested together.

#include "vision/deepstack_inject.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <vector>

namespace imp {

// Declared where it is used (src/exec/executor_forward.cu); no header.
void launch_replace_vision_embeddings(half* hidden, const int32_t* token_ids, const half* vision_emb,
                                      int vision_token_id, int n_tokens, int d_model, int n_vision_tokens,
                                      int emb_offset, cudaStream_t stream);

namespace {

constexpr int kD = 4;
constexpr int kImg = 42;

bool gpu_available() {
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

// One embedding per image token, each filled with a distinct value, so a wrong
// index is visible as a wrong number rather than as noise.
std::vector<float> distinct_embeddings(int n_vision_tokens) {
    std::vector<float> e(static_cast<size_t>(n_vision_tokens) * kD);
    for (int k = 0; k < n_vision_tokens; ++k)
        for (int d = 0; d < kD; ++d)
            e[static_cast<size_t>(k) * kD + d] = 1.0f + static_cast<float>(k);
    return e;
}

enum class Op { Replace, Add };

// Run one chunk through one of the two kernels and return the hidden state.
std::vector<float> run_chunk(Op op, const std::vector<int32_t>& chunk_tokens,
                             const std::vector<float>& hidden_in, const std::vector<float>& emb,
                             int n_vision_tokens, int emb_offset) {
    const int n = static_cast<int>(chunk_tokens.size());
    std::vector<half> h(hidden_in.size()), e(emb.size());
    for (size_t i = 0; i < hidden_in.size(); ++i)
        h[i] = __float2half(hidden_in[i]);
    for (size_t i = 0; i < emb.size(); ++i)
        e[i] = __float2half(emb[i]);

    half *d_h = nullptr, *d_e = nullptr;
    int32_t* d_t = nullptr;
    EXPECT_EQ(cudaMalloc(&d_h, h.size() * sizeof(half)), cudaSuccess);
    EXPECT_EQ(cudaMalloc(&d_e, e.size() * sizeof(half)), cudaSuccess);
    EXPECT_EQ(cudaMalloc(&d_t, chunk_tokens.size() * sizeof(int32_t)), cudaSuccess);
    cudaMemcpy(d_h, h.data(), h.size() * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_e, e.data(), e.size() * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_t, chunk_tokens.data(), chunk_tokens.size() * sizeof(int32_t), cudaMemcpyHostToDevice);

    if (op == Op::Replace)
        launch_replace_vision_embeddings(d_h, d_t, d_e, kImg, n, kD, n_vision_tokens, emb_offset, nullptr);
    else
        launch_add_vision_embeddings(d_h, d_t, d_e, kImg, n, kD, n_vision_tokens, emb_offset, nullptr);
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    cudaMemcpy(h.data(), d_h, h.size() * sizeof(half), cudaMemcpyDeviceToHost);
    cudaFree(d_h);
    cudaFree(d_e);
    cudaFree(d_t);

    std::vector<float> out(h.size());
    for (size_t i = 0; i < h.size(); ++i)
        out[i] = __half2float(h[i]);
    return out;
}

// The prompt is [text, IMG, IMG, IMG, IMG, text] split after position 2, so the
// image run straddles the boundary: 2 of its 4 tokens land in each chunk.
struct SplitPrompt {
    std::vector<int32_t> chunk0{7, kImg, kImg};
    std::vector<int32_t> chunk1{kImg, kImg, 8};
    int n_vision_tokens = 4;
    int consumed_by_chunk0 = 2;
};

TEST(VisionChunkOffset, SecondChunkTakesTheEmbeddingsItsPositionsEarned) {
    if (!gpu_available())
        GTEST_SKIP() << "no CUDA device";

    const SplitPrompt p;
    const auto emb = distinct_embeddings(p.n_vision_tokens);
    const std::vector<float> hidden(p.chunk1.size() * kD, 0.0f);

    const auto out = run_chunk(Op::Replace, p.chunk1, hidden, emb, p.n_vision_tokens, p.consumed_by_chunk0);

    // Positions 0 and 1 of this chunk are the image's tokens 2 and 3.
    for (int d = 0; d < kD; ++d) {
        EXPECT_NEAR(out[0 * kD + d], 3.0f, 1e-3) << "took embedding 0 (the first chunk's) instead of 2";
        EXPECT_NEAR(out[1 * kD + d], 4.0f, 1e-3) << "took embedding 1 (the first chunk's) instead of 3";
        EXPECT_NEAR(out[2 * kD + d], 0.0f, 1e-3) << "text token was overwritten";
    }
}

TEST(VisionChunkOffset, DeepStackFollowsTheSameOffset) {
    if (!gpu_available())
        GTEST_SKIP() << "no CUDA device";

    const SplitPrompt p;
    const auto emb = distinct_embeddings(p.n_vision_tokens);
    const std::vector<float> hidden(p.chunk1.size() * kD, 10.0f);

    const auto out = run_chunk(Op::Add, p.chunk1, hidden, emb, p.n_vision_tokens, p.consumed_by_chunk0);

    // An ADD, so 10 + the right embedding. The two kernels must not disagree
    // about which position gets which feature, or DeepStack stacks a tap from
    // one part of the image onto the embedding of another.
    for (int d = 0; d < kD; ++d) {
        EXPECT_NEAR(out[0 * kD + d], 13.0f, 2e-2) << "added embedding 0 instead of 2";
        EXPECT_NEAR(out[1 * kD + d], 14.0f, 2e-2) << "added embedding 1 instead of 3";
        EXPECT_NEAR(out[2 * kD + d], 10.0f, 2e-2) << "text token was touched";
    }
}

// The first chunk is the case that already worked; it must keep working, since
// offset 0 is what every unchunked prompt passes.
TEST(VisionChunkOffset, FirstChunkIsUnchangedAtOffsetZero) {
    if (!gpu_available())
        GTEST_SKIP() << "no CUDA device";

    const SplitPrompt p;
    const auto emb = distinct_embeddings(p.n_vision_tokens);
    const std::vector<float> hidden(p.chunk0.size() * kD, 0.0f);

    const auto out = run_chunk(Op::Replace, p.chunk0, hidden, emb, p.n_vision_tokens, 0);

    for (int d = 0; d < kD; ++d) {
        EXPECT_NEAR(out[0 * kD + d], 0.0f, 1e-3) << "text token was overwritten";
        EXPECT_NEAR(out[1 * kD + d], 1.0f, 1e-3);
        EXPECT_NEAR(out[2 * kD + d], 2.0f, 1e-3);
    }
}

// Reassembling both chunks must reproduce the unchunked result exactly —
// the property the offset exists to preserve.
TEST(VisionChunkOffset, ChunkedMatchesUnchunked) {
    if (!gpu_available())
        GTEST_SKIP() << "no CUDA device";

    const SplitPrompt p;
    const auto emb = distinct_embeddings(p.n_vision_tokens);

    std::vector<int32_t> whole = p.chunk0;
    whole.insert(whole.end(), p.chunk1.begin(), p.chunk1.end());
    const std::vector<float> zero_whole(whole.size() * kD, 0.0f);
    const auto unchunked = run_chunk(Op::Replace, whole, zero_whole, emb, p.n_vision_tokens, 0);

    const std::vector<float> zero0(p.chunk0.size() * kD, 0.0f);
    const std::vector<float> zero1(p.chunk1.size() * kD, 0.0f);
    auto a = run_chunk(Op::Replace, p.chunk0, zero0, emb, p.n_vision_tokens, 0);
    const auto b = run_chunk(Op::Replace, p.chunk1, zero1, emb, p.n_vision_tokens, p.consumed_by_chunk0);
    a.insert(a.end(), b.begin(), b.end());

    ASSERT_EQ(a.size(), unchunked.size());
    for (size_t i = 0; i < a.size(); ++i)
        EXPECT_NEAR(a[i], unchunked[i], 1e-3) << "chunked and unchunked disagree at " << i;
}

// A chunk whose image tokens are entirely behind us must not write at all,
// rather than read past the embedding buffer.
TEST(VisionChunkOffset, OffsetPastTheBufferWritesNothing) {
    if (!gpu_available())
        GTEST_SKIP() << "no CUDA device";

    const std::vector<int32_t> tokens{kImg, kImg};
    const auto emb = distinct_embeddings(2);
    const std::vector<float> hidden(tokens.size() * kD, 5.0f);

    const auto out = run_chunk(Op::Replace, tokens, hidden, emb, /*n_vision_tokens=*/2, /*emb_offset=*/2);

    for (float v : out)
        EXPECT_NEAR(v, 5.0f, 1e-3) << "wrote with nothing left to write";
}

}  // namespace
}  // namespace imp
