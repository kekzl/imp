// src/compute/gemm_grouped_nvfp4_smallM.h
#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

namespace imp {

// Hand-rolled persistent NVFP4 grouped GEMM for SM120 (RTX 5090).
// Drop-in alternative to gemm_grouped_cutlass_3x_nvfp4 with M-aware
// tile selection (16/32/64/128). Reads native row-major UE4M3 scales
// directly from cache_moe_native_nvfp4's nvfp4_moe_ms_native buffer.
//
//   A_i  : [M_i, K]      packed NVFP4, K-contiguous, K/2 bytes per row
//   SFA_i: [M_i, K/16]   UE4M3 native row-major (1 byte per scale)
//   B_i  : [N,   K]      packed NVFP4 (per-expert weight)
//   SFB_i: [N,   K/16]   UE4M3 native row-major
//   D_i  : [M_i, N]      FP16 output, RowMajor
//   alpha_i: per-expert tensor_scale (applied as GEMM alpha)
//
// K and N must be identical across all experts. M_i varies.
// Returns false if SM120 unavailable or any precondition fails.
bool gemm_grouped_nvfp4_smallM(
    int n_experts,
    const int* host_M,                // [n_experts] M_i per expert
    int N, int K,
    const void* const* host_ptr_A,    // [n_experts] device packed A
    const void* const* host_ptr_SFA,  // [n_experts] device SFA (native row-major)
    const void* const* host_ptr_B,    // [n_experts] device packed B
    const void* const* host_ptr_SFB,  // [n_experts] device SFB (native row-major)
    void* const* host_ptr_D,          // [n_experts] device FP16 outputs
    const float* host_alpha,          // [n_experts] per-expert tensor_scale
    cudaStream_t stream);

bool gemm_grouped_nvfp4_smallM_available();
void gemm_grouped_nvfp4_smallM_cleanup();

namespace detail {

struct WorkItem {
    int expert_id;
    int m_tile_idx;       // tile index along M (per expert)
    int n_tile_idx;       // tile index along N
    uint8_t m_tile_size;  // 16, 32, 64, or 128
};

// Pick the smallest viable M-tile for an expert with M_e tokens.
int pick_m_tile(int M_e);

// Build the work queue, sorted by descending tile size for shorter tail latency.
// Inactive experts (M_e <= 0) are skipped.
std::vector<WorkItem> build_work_queue(int n_experts, const int* M_per, int N);

}  // namespace detail

}  // namespace imp
