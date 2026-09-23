#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace imp {

// Operands of one Mamba2 scan launch (layouts: ssm.cu ssm_scan_kernel).
struct SsmScanArgs {
    const half* x;
    const half* B;
    const half* C;
    const half* dt;
    const float* A_log;
    const float* D;
    const float* dt_bias;
    void* h_state;
    half* y;
    const half* z;
    int n_tokens, n_heads, head_dim_ssm, state_size, n_groups;
    const int* d_real_n;
    void* h_snap;
    const int* d_snap_n;
    cudaStream_t stream;
};

// Register-resident scan, bit-identical to the legacy kernel at the same s_tiles.
// Returns false (nothing launched) for shapes it does not cover; the caller falls back.
bool ssm_scan_reg_launch(const SsmScanArgs& a, int s_tiles, bool fp16);

// Legacy per-token global-state kernel only (tests compare both paths bitwise).
void ssm_scan_legacy_launch(const SsmScanArgs& a, bool fp16);

}  // namespace imp
