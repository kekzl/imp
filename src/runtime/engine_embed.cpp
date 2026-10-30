// /v1/embeddings on the decoder path (#1005): request preparation, per-chunk pooling, finish.
// Mean pooling sums every chunk; last-token embedders (ModelConfig::embed_last_token_pool,
// Qwen3-Embedding) pool the final-norm state of the appended end token on the last chunk only.

#include "runtime/engine.h"
#include "runtime/request.h"
#include "exec/executor.h"
#include "model/model.h"
#include "core/cuda_errors.h"
#include "core/logging.h"

#include <vector>

namespace imp {

void Engine::embed_prepare_(Request& req) const {
    const int32_t eos = model_->config().embed_append_token;
    if (req.embedding_request && eos >= 0 && (req.input_tokens.empty() || req.input_tokens.back() != eos))
        req.input_tokens.push_back(eos);  // HF TemplateProcessing appends it; encode() does not
}

void Engine::embed_accumulate_chunk_(Request& req, int chunk_len, cudaStream_t stream, bool last_chunk) {
    const bool last_token = model_->config().embed_last_token_pool;
    if (last_token && !last_chunk)
        return;
    const int d_model = static_cast<int>(model_->config().d_model);
    if (!d_embed_pool_scratch_) {
        d_embed_pool_scratch_ = static_cast<float*>(
            vram_alloc_.allocate(static_cast<size_t>(d_model) * sizeof(float), "embed_pool_scratch"));
        if (!d_embed_pool_scratch_) {
            IMP_LOG_ERROR("VRAMAllocator failed for embed pool scratch (%d floats)", d_model);
            return;
        }
    }
    if (last_token)
        executor_->pool_hidden_last(chunk_len, d_embed_pool_scratch_, stream);
    else
        executor_->pool_hidden_sum(chunk_len, d_embed_pool_scratch_, stream);
    std::vector<float> partial(d_model);
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(partial.data(), d_embed_pool_scratch_,
                                       static_cast<size_t>(d_model) * sizeof(float), cudaMemcpyDeviceToHost,
                                       stream));
    IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream));
    if (req.embedding_out.empty())
        req.embedding_out.assign(d_model, 0.0f);
    for (int d = 0; d < d_model; d++)
        req.embedding_out[d] += partial[d];
}

void Engine::embed_finish_(Request& req) const {
    const size_t total = req.input_tokens.size();
    if (model_->config().embed_last_token_pool || total == 0 || req.embedding_out.empty())
        return;
    const float inv = 1.0f / static_cast<float>(total);
    for (float& v : req.embedding_out)
        v *= inv;
}

}  // namespace imp
