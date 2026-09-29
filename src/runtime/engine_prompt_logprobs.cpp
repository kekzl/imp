// Prompt logprobs (#2207): per-chunk device gather into Request::prompt_lp.

#include "runtime/engine.h"
#include "runtime/prompt_lp_chunk.h"
#include "runtime/request.h"
#include "exec/executor.h"
#include "compute/prompt_logprobs_rows.h"
#include "core/logging.h"

#include <algorithm>

#include <cuda_runtime.h>

namespace imp {

static_assert(kMaxPromptLogprobs <= kPlpMaxTopN, "prompt_logprobs_rows caps top-N at kPlpMaxTopN");

void Engine::prompt_logprobs_chunk_(Request& req, int offset, int chunk_len, cudaStream_t stream) {
    if (req.prompt_logprobs < 0 || !executor_)
        return;
    const int n_prompt = static_cast<int>(req.input_tokens.size());
    // The last prompt row has no prompt target (it predicts the first output token).
    const int rows = std::min(chunk_len, n_prompt - 1 - offset);
    if (rows <= 0)
        return;
    PromptLogprobs& out = req.prompt_lp;
    const int top_n = std::clamp(req.prompt_logprobs, 0, kMaxPromptLogprobs);
    // Rows land at their prompt position; a re-prefill from 0 starts over, a skipped chunk leaves
    // out.rows short (the server answers 500).
    if (offset == 0)
        out = PromptLogprobs{};
    if (out.rows != offset)
        return;
    if (out.token_lp.empty()) {
        const auto total = static_cast<size_t>(n_prompt - 1);
        out.top_n = top_n;
        out.token_lp.assign(total, 0.0f);
        out.rank.assign(total, 0);
        out.top_ids.assign(total * static_cast<size_t>(top_n), 0);
        out.top_lp.assign(total * static_cast<size_t>(top_n), 0.0f);
    }

    PromptLpScratch& s = prompt_lp_scratch_;
    if (rows > s.rows || top_n > s.top_n) {
        const int r = std::max(rows, s.rows);
        const int t = std::max(top_n, s.top_n);
        s = PromptLpScratch{};
        s.targets = VramOwned<int32_t>(vram_alloc_, static_cast<size_t>(r), "prompt_lp_targets");
        s.rank = VramOwned<int32_t>(vram_alloc_, static_cast<size_t>(r), "prompt_lp_rank");
        s.lp = VramOwned<float>(vram_alloc_, static_cast<size_t>(r), "prompt_lp_lp");
        if (t > 0) {
            s.top_ids = VramOwned<int32_t>(vram_alloc_, static_cast<size_t>(r) * t, "prompt_lp_top_ids");
            s.top_lp = VramOwned<float>(vram_alloc_, static_cast<size_t>(r) * t, "prompt_lp_top_lp");
        }
        if (!s.targets || !s.rank || !s.lp || (t > 0 && (!s.top_ids || !s.top_lp))) {
            IMP_LOG_ERROR("prompt_logprobs: scratch allocation failed (%d rows, top %d)", r, t);
            s = PromptLpScratch{};
            return;
        }
        s.rows = r;
        s.top_n = t;
    }

    // Logits chunk for one LM-head GEMM per chunk (#2257); a failed grow keeps the per-batch driver.
    const int vocab = model_->config().vocab_size;
    const int want = prompt_lp_chunk_rows(rows, vocab, vram_alloc_.available());
    if (want > s.logit_rows) {
        s.logits.reset();
        s.logits = VramOwned<float>(vram_alloc_, static_cast<size_t>(want) * static_cast<size_t>(vocab),
                                    "prompt_lp_logits");
        s.logit_rows = s.logits ? want : 0;
    }

    const size_t nrows = static_cast<size_t>(rows);
    const size_t ntop = nrows * static_cast<size_t>(top_n);
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(s.targets.get(), req.input_tokens.data() + offset + 1,
                                       nrows * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    executor_->prompt_logprobs_partial(s.targets.get(), rows, top_n, s.lp.get(), s.rank.get(),
                                       top_n > 0 ? s.top_ids.get() : nullptr,
                                       top_n > 0 ? s.top_lp.get() : nullptr, s.logits.get(),
                                       std::min(rows, s.logit_rows), stream);
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(out.token_lp.data() + offset, s.lp.get(), nrows * sizeof(float),
                                       cudaMemcpyDeviceToHost, stream));
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(out.rank.data() + offset, s.rank.get(), nrows * sizeof(int32_t),
                                       cudaMemcpyDeviceToHost, stream));
    if (top_n > 0) {
        const size_t base = static_cast<size_t>(offset) * static_cast<size_t>(top_n);
        IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(out.top_ids.data() + base, s.top_ids.get(), ntop * sizeof(int32_t),
                                           cudaMemcpyDeviceToHost, stream));
        IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(out.top_lp.data() + base, s.top_lp.get(), ntop * sizeof(float),
                                           cudaMemcpyDeviceToHost, stream));
    }
    if (cudaStreamSynchronize(stream) != cudaSuccess) {
        IMP_LOG_ERROR("prompt_logprobs: stream sync failed at offset %d", offset);
        return;
    }
    out.rows = offset + rows;
}

}  // namespace imp
