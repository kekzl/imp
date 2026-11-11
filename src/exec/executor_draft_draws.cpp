// Coupled MTP draft draws (speculative.mtp_draft_coupled): scratch and host staging of the rows
// lm_head_rows_argmax draws inside the head pass. No kernel launch in this TU.

#include "compute/sampling.h"
#include "exec/executor.h"
#include "exec/executor_sampling_internal.h"
#include "quant/nvfp4_quant.h"

#include <cuda_runtime.h>

#include <algorithm>

namespace imp {

bool GraphExecutor::ensure_draft_draw_scratch_() {
    if (draft_draw_scratch_ && !h_draft_draw_args_.empty())
        return true;
    if (vram_alloc_ == nullptr || draft_draw_scratch_)
        return false;
    draft_draw_scratch_ = VramOwned<char>(*vram_alloc_,
                                          verify_sample_args_bytes(kDraftDrawRows) +
                                              (static_cast<size_t>(kDraftDrawRows) * SAMPLE_SCRATCH_BYTES),
                                          "mtp_draft_draw_rows");
    h_draft_draw_args_ = PinnedBuffer::acquire(cuda_host_pinned_allocator(),
                                               static_cast<size_t>(kDraftDrawRows) * sizeof(TopkRowArgs));
    return draft_draw_scratch_ && !h_draft_draw_args_.empty();
}

int GraphExecutor::draft_rows_vocab_(const NvFP4QuantResult* draft_nvfp4, int draft_vocab) const {
    const void* w8 = nullptr;
    const float* s8 = nullptr;
    const bool w4a16 = draft_nvfp4 != nullptr && draft_nvfp4->packed_data != nullptr;
    const bool fp8 = !w4a16 && !lm_head_cutlass_ready_ && lm_head_fp8_view(w8, s8);
    const int vocab = model_->config().vocab_size;
    return fp8 && draft_vocab > 0 ? std::min(draft_vocab, vocab) : vocab;
}

int GraphExecutor::stage_draft_draws(const VerifySampling* draws, int n, int m,
                                     const NvFP4QuantResult* draft_nvfp4, int draft_vocab,
                                     cudaStream_t stream) {
    if (!initialized_ || draws == nullptr || n <= 0 || m <= 0 || n > kDraftDrawRows ||
        !ensure_draft_draw_scratch_())
        return 0;
    // Row lv * m + i reads chunk row i % mb of its call's logits (lm_head_rows_argmax packs V to a row).
    const size_t V = static_cast<size_t>(draft_rows_vocab_(draft_nvfp4, draft_vocab));
    const int mb = max_logit_tokens_ > 0 ? max_logit_tokens_ : 1;
    auto* h_args = h_draft_draw_args_.as<TopkRowArgs>();
    char* slots = draft_draw_scratch_.get() + verify_sample_args_bytes(kDraftDrawRows);
    int max_top_k = 0;
    for (int r = 0; r < n; ++r) {
        if (draws[r].top_k < 1 || draws[r].top_k > SAMPLE_MAX_TOP_K)
            return 0;
        max_top_k = std::max(max_top_k, draws[r].top_k);
        stage_draft_row(static_cast<const float*>(logits_.data) + (static_cast<size_t>((r % m) % mb) * V),
                        draws[r], slots + (static_cast<size_t>(r) * SAMPLE_SCRATCH_BYTES), h_args[r]);
    }
    if (cudaMemcpyAsync(draft_draw_scratch_.get(), h_args, static_cast<size_t>(n) * sizeof(TopkRowArgs),
                        cudaMemcpyHostToDevice, stream) != cudaSuccess)
        return 0;
    return max_top_k;
}

}  // namespace imp
