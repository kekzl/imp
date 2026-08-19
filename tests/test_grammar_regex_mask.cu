#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "compute/grammar_constrain.h"
#include "compute/regex_constrain.h"
#include "model/tokenizer.h"

#include <cfloat>
#include <string>
#include <vector>

#include "test_cuda_skip.h"

namespace imp {
namespace {

// The allow list is sized from the TOKENIZER vocabulary at init, but apply_mask
// is handed the width of the logits, which is the MODEL vocabulary. On every
// checkpoint whose lm_head is padded the second number is larger. Qwen3-8B
// ships 151936 rows against 151669 tokenizer entries, and the upload of the
// allow list then wrote past its own buffer.
//
// The failure was not visible as a failure: cudaMemcpyAsync returned
// `invalid argument`, the device allow list kept whatever it held, and the mask
// kernel masked everything. Greedy argmax over an all -FLT_MAX row picks id 0,
// so every regex- and grammar-constrained request answered "!!!!..." until it
// hit max_tokens. JsonConstrainer and SchemaConstrainer size both sides from
// the tokenizer and were never affected; they are also the only two of the four
// that had a GPU test.
//
// Mirrors JsonConstrainTest.ModelVocabLargerThanTokenizerMasksPadding.

// Builds a tokenizer whose vocabulary is deliberately narrower than the logits
// row the constrainer will be asked to mask.
Tokenizer make_tokenizer(std::vector<std::string>& toks) {
    std::vector<float> scores(toks.size(), 0.0f);
    Tokenizer tok;
    tok.load_vocab(toks, scores, /*bos_id=*/1, /*eos_id=*/2);
    return tok;
}

TEST(RegexConstrainMaskTest, ModelVocabLargerThanTokenizerMasksPadding) {
    SKIP_IF_NO_CUDA();

    //                                  0        1      2       3        4       5
    std::vector<std::string> toks = {"<unk>", "<s>", "</s>", "Paris", "Lyon", "xyz"};
    Tokenizer tok = make_tokenizer(toks);

    const int tok_vocab = static_cast<int>(toks.size());
    const int model_vocab = tok_vocab + 9;  // simulated lm_head padding rows

    RegexConstrainer rc;
    ASSERT_TRUE(rc.init("Paris", &tok));

    std::vector<float> h(model_vocab, 1.0f);
    float* d = nullptr;
    ASSERT_EQ(cudaMalloc(&d, model_vocab * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d, h.data(), model_vocab * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);

    ASSERT_EQ(cudaGetLastError(), cudaSuccess) << "test setup left a CUDA error behind";
    rc.apply_mask(d, model_vocab, 0);
    EXPECT_EQ(cudaGetLastError(), cudaSuccess)
        << "apply_mask raised a CUDA error: the allow-list upload is sized from the model "
           "vocabulary while the buffer is sized from the tokenizer";

    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(h.data(), d, model_vocab * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    cudaFree(d);

    EXPECT_GT(h[3], -1e30f) << "'Paris' matches the pattern and must survive the mask";
    EXPECT_FLOAT_EQ(h[4], -FLT_MAX) << "'Lyon' does not match the pattern and must be masked";
    for (int i = tok_vocab; i < model_vocab; i++)
        EXPECT_FLOAT_EQ(h[i], -FLT_MAX) << "padding id " << i << " leaked through the regex mask";
}

TEST(GrammarConstrainMaskTest, ModelVocabLargerThanTokenizerMasksPadding) {
    SKIP_IF_NO_CUDA();

    //                                  0        1      2       3        4       5
    std::vector<std::string> toks = {"<unk>", "<s>", "</s>", "Paris", "Lyon", "xyz"};
    Tokenizer tok = make_tokenizer(toks);

    const int tok_vocab = static_cast<int>(toks.size());
    const int model_vocab = tok_vocab + 9;

    GrammarConstrainer gc;
    ASSERT_TRUE(gc.init("root ::= \"Paris\"", &tok));

    std::vector<float> h(model_vocab, 1.0f);
    float* d = nullptr;
    ASSERT_EQ(cudaMalloc(&d, model_vocab * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d, h.data(), model_vocab * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);

    ASSERT_EQ(cudaGetLastError(), cudaSuccess) << "test setup left a CUDA error behind";
    gc.apply_mask(d, model_vocab, 0);
    EXPECT_EQ(cudaGetLastError(), cudaSuccess)
        << "apply_mask raised a CUDA error: the allow-list upload is sized from the model "
           "vocabulary while the buffer is sized from the tokenizer";

    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(h.data(), d, model_vocab * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    cudaFree(d);

    EXPECT_GT(h[3], -1e30f) << "'Paris' is the only production and must survive the mask";
    EXPECT_FLOAT_EQ(h[4], -FLT_MAX) << "'Lyon' is not in the grammar and must be masked";
    for (int i = tok_vocab; i < model_vocab; i++)
        EXPECT_FLOAT_EQ(h[i], -FLT_MAX) << "padding id " << i << " leaked through the grammar mask";
}

}  // namespace
}  // namespace imp
