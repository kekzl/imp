#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace imp {

// Pipelined batched-decode chain advance (decode_pipeline_advance.cu): token_ids[i] = slot i's sampled
// token, positions[i]++, context_lens[i]++, per-row penalty history append, plus n_patches block-table
// scatter writes. Patch/pos arrays must be device-readable (mapped pinned memory). Host-safe, so host-only
// TUs (engine_decode_pipeline.cpp) can launch it. Arguments in the old parameter order (audit D-11).
struct DecodePipelineAdvanceArgs {
    int n_rows = 0;
    const int32_t* slot_tokens = nullptr;
    size_t slot_stride_bytes = 0;
    int32_t* d_token_ids = nullptr;
    int* d_positions = nullptr;
    int* d_context_lens = nullptr;
    int* d_block_tables = nullptr;
    int n_patches = 0;
    const int* d_patch_offsets = nullptr;
    const int* d_patch_values = nullptr;
    int32_t* d_hist_base = nullptr;
    int hist_stride = 0;
    const int* d_hist_pos = nullptr;
    cudaStream_t stream = nullptr;
};
void decode_pipeline_advance(const DecodePipelineAdvanceArgs& a);

}  // namespace imp
