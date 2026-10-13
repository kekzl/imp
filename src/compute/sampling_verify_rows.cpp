// Sampled speculative verify rows (engine_spec_ngram.cpp): each chunk row draws from the request's
// distribution with the eager sampler's seed for that output index. Accepting a one-token-per-row draft
// while the draw equals it is exact rejection sampling: accept d with p(d), else a draw from p without d.

#include "compute/sampling.h"

#include <cuda_runtime.h>

namespace imp {

cudaError_t sample_verify_rows(float* d_logits, int n_rows, int vocab_size, const VerifySampling& s, int row0,
                               TopkRowArgs* h_args, TopkRowArgs* d_args, void* d_slots, int32_t* d_out,
                               cudaStream_t stream) {
    if (n_rows <= 0)
        return cudaSuccess;
    if (s.top_k < 1 || s.top_k > SAMPLE_MAX_TOP_K || !h_args || !d_args || !d_slots || !d_out)
        return cudaErrorInvalidValue;
    const float temperature = s.temperature > 0.0f ? s.temperature : 1.0f;
    for (int r = 0; r < n_rows; ++r) {
        float* lp = d_logits + static_cast<size_t>(r) * vocab_size;
        apply_min_p(lp, vocab_size, s.min_p, stream);          // no-op at 0
        apply_typical_p(lp, vocab_size, s.typical_p, stream);  // no-op outside (0, 1)
        TopkRowArgs& a = h_args[r];
        a.logits = lp;
        a.d_result = reinterpret_cast<int32_t*>(static_cast<char*>(d_slots) +
                                                static_cast<size_t>(r) * SAMPLE_SCRATCH_BYTES);
        a.inv_temperature = 1.0f / temperature;
        a.top_p = s.top_p > 0.0f ? s.top_p : 1.0f;
        a.seed = (static_cast<unsigned>(s.seed0) + static_cast<unsigned>(row0 + r)) & 0x7fffffffu;
        a.top_k = s.top_k;
    }
    if (const cudaError_t e = cudaMemcpyAsync(d_args, h_args, sizeof(TopkRowArgs) * n_rows,
                                              cudaMemcpyHostToDevice, stream);
        e != cudaSuccess)
        return e;
    if (const cudaError_t e = launch_topk_topp_rows(d_args, n_rows, s.top_k, vocab_size, stream);
        e != cudaSuccess)
        return e;
    // Token = first int32 of each row's scratch slot.
    return cudaMemcpy2DAsync(d_out, sizeof(int32_t), d_slots, SAMPLE_SCRATCH_BYTES, sizeof(int32_t),
                             static_cast<size_t>(n_rows), cudaMemcpyDeviceToDevice, stream);
}

}  // namespace imp
