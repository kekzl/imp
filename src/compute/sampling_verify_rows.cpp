// Sampled speculative verify rows (engine_spec_ngram.cpp, engine_spec_batch_verify.cpp): each chunk row draws
// from the request's distribution with the eager sampler's seed for that output index. Accepting a
// one-token-per-row draft while the draw equals it is exact rejection sampling: accept d with p(d), else a
// draw from p without d.

#include "compute/sampling.h"

#include <cuda_runtime.h>

#include <algorithm>

namespace imp {

namespace {

// Filters row lp in place (min_p, typical_p) and stages its top_k/top_p/temperature draw.
void stage_verify_row(float* lp, int vocab_size, const VerifySampling& s, unsigned seed, void* slot,
                      TopkRowArgs& a, cudaStream_t stream) {
    apply_min_p(lp, vocab_size, s.min_p, stream);          // no-op at 0
    apply_typical_p(lp, vocab_size, s.typical_p, stream);  // no-op outside (0, 1)
    VerifySampling d = s;
    d.seed0 = static_cast<int>(seed & 0x7fffffffu);
    stage_draft_row(lp, d, slot, a);
}

void* row_slot(void* d_slots, int r) {
    return static_cast<char*>(d_slots) + (static_cast<size_t>(r) * SAMPLE_SCRATCH_BYTES);
}

}  // namespace

void stage_draft_row(const float* lp, const VerifySampling& s, void* slot, TopkRowArgs& a) {
    a.logits = lp;
    a.d_result = static_cast<int32_t*>(slot);
    a.inv_temperature = 1.0f / (s.temperature > 0.0f ? s.temperature : 1.0f);
    a.top_p = s.top_p > 0.0f ? s.top_p : 1.0f;
    a.seed = static_cast<unsigned>(s.seed0) & 0x7fffffffu;
    a.top_k = s.top_k;
}

cudaError_t draw_uploaded_rows(const TopkRowArgs* d_args, int n_rows, int max_top_k, int vocab_size,
                               const void* d_slots, int32_t* d_out, cudaStream_t stream) {
    if (const cudaError_t e = launch_topk_topp_rows(d_args, n_rows, max_top_k, vocab_size, stream);
        e != cudaSuccess)
        return e;
    // Token = first int32 of each row's scratch slot.
    return cudaMemcpy2DAsync(d_out, sizeof(int32_t), d_slots, SAMPLE_SCRATCH_BYTES, sizeof(int32_t),
                             static_cast<size_t>(n_rows), cudaMemcpyDeviceToDevice, stream);
}

namespace {

// Uploads n_rows staged rows (h_args) and draws them into d_out[n_rows].
cudaError_t draw_staged_rows(int n_rows, int max_top_k, int vocab_size, const TopkRowArgs* h_args,
                             TopkRowArgs* d_args, void* d_slots, int32_t* d_out, cudaStream_t stream) {
    if (const cudaError_t e = cudaMemcpyAsync(d_args, h_args, sizeof(TopkRowArgs) * n_rows,
                                              cudaMemcpyHostToDevice, stream);
        e != cudaSuccess)
        return e;
    return draw_uploaded_rows(d_args, n_rows, max_top_k, vocab_size, d_slots, d_out, stream);
}

}  // namespace

cudaError_t sample_verify_rows(float* d_logits, int n_rows, int vocab_size, const VerifySampling& s, int row0,
                               TopkRowArgs* h_args, TopkRowArgs* d_args, void* d_slots, int32_t* d_out,
                               cudaStream_t stream) {
    if (n_rows <= 0)
        return cudaSuccess;
    if (s.top_k < 1 || s.top_k > SAMPLE_MAX_TOP_K || !h_args || !d_args || !d_slots || !d_out)
        return cudaErrorInvalidValue;
    for (int r = 0; r < n_rows; ++r)
        stage_verify_row(d_logits + (static_cast<size_t>(r) * vocab_size), vocab_size, s,
                         static_cast<unsigned>(s.seed0) + static_cast<unsigned>(row0 + r),
                         row_slot(d_slots, r), h_args[r], stream);
    return draw_staged_rows(n_rows, s.top_k, vocab_size, h_args, d_args, d_slots, d_out, stream);
}

cudaError_t sample_verify_rows_each(float* d_logits, int n_rows, int vocab_size,
                                    const VerifySampling* per_row, TopkRowArgs* h_args, TopkRowArgs* d_args,
                                    void* d_slots, int32_t* d_out, cudaStream_t stream) {
    if (n_rows <= 0)
        return cudaSuccess;
    if (!per_row || !h_args || !d_args || !d_slots || !d_out)
        return cudaErrorInvalidValue;
    int max_top_k = 0;
    for (int r = 0; r < n_rows; ++r) {
        if (per_row[r].top_k < 1 || per_row[r].top_k > SAMPLE_MAX_TOP_K)
            return cudaErrorInvalidValue;
        max_top_k = std::max(max_top_k, per_row[r].top_k);
    }
    for (int r = 0; r < n_rows; ++r)
        stage_verify_row(d_logits + (static_cast<size_t>(r) * vocab_size), vocab_size, per_row[r],
                         static_cast<unsigned>(per_row[r].seed0), row_slot(d_slots, r), h_args[r], stream);
    return draw_staged_rows(n_rows, max_top_k, vocab_size, h_args, d_args, d_slots, d_out, stream);
}

}  // namespace imp
