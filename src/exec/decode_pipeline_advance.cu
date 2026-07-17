// Pipelined batched-decode chain advance (runtime.decode_pipeline).
//
// One tiny single-block launch that makes step N+1 runnable BEFORE the host
// has read step N's tokens: feed step N's sampled slot tokens (the per-row
// SAMPLE_SCRATCH_BYTES slots) as step N+1's input token ids, bump positions
// and context lens, append each token to the per-row device output history
// (rep/freq/presence penalty rows sample step N+1 against a history that
// includes the token the host has not seen yet), and scatter freshly
// appended KV block-table entries. The patch/pos arrays live in mapped
// pinned memory (host-written just before launch, parity-alternated by the
// engine so a set is never overwritten while a kernel may still read it).
//
// Own TU (split from executor_elementwise.cu): distinct logical unit on the
// serving hot path — an edit here must not re-ptxas the elementwise grab-bag.

#include "core/logging.h"

#include <cuda_runtime.h>
#include <cstdint>

namespace imp {

__global__ void decode_pipeline_advance_kernel(int n_rows, const char* __restrict__ slot_base,
                                               size_t slot_stride_bytes,
                                               int32_t* __restrict__ token_ids,
                                               int* __restrict__ positions,
                                               int* __restrict__ context_lens,
                                               int* __restrict__ block_tables, int n_patches,
                                               const int* __restrict__ patch_offsets,
                                               const int* __restrict__ patch_values,
                                               int32_t* __restrict__ hist_base, int hist_stride,
                                               const int* __restrict__ hist_pos) {
    int i = threadIdx.x;
    if (i < n_rows) {
        const int32_t tok = *reinterpret_cast<const int32_t*>(slot_base + i * slot_stride_bytes);
        token_ids[i] = tok;
        positions[i] += 1;
        context_lens[i] += 1;
        // Per-row output-token history append: penalty rows sample the NEXT
        // step against a history that includes this (host-unseen) token.
        if (hist_base != nullptr)
            hist_base[static_cast<size_t>(i) * hist_stride + hist_pos[i]] = tok;
    } else if (i < n_rows + n_patches) {
        block_tables[patch_offsets[i - n_rows]] = patch_values[i - n_rows];
    }
}

void decode_pipeline_advance(int n_rows, const int32_t* slot_tokens, size_t slot_stride_bytes,
                             int32_t* d_token_ids, int* d_positions, int* d_context_lens,
                             int* d_block_tables, int n_patches, const int* d_patch_offsets,
                             const int* d_patch_values, int32_t* d_hist_base, int hist_stride,
                             const int* d_hist_pos, cudaStream_t stream) {
    int threads = n_rows + n_patches;
    if (threads <= 0)
        return;
    decode_pipeline_advance_kernel<<<1, threads, 0, stream>>>(
        n_rows, reinterpret_cast<const char*>(slot_tokens), slot_stride_bytes, d_token_ids,
        d_positions, d_context_lens, d_block_tables, n_patches, d_patch_offsets, d_patch_values,
        d_hist_base, hist_stride, d_hist_pos);
    IMP_CUDA_CHECK_LAUNCH();
}

}  // namespace imp
