// src/compute/gemm_grouped_nvfp4_smallM.h
#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

namespace imp {

// Hand-rolled persistent NVFP4 grouped GEMM for SM120. Drop-in alternative to
// gemm_grouped_cutlass_3x_nvfp4 with M-aware tile selection (16/32/64/128). Reads native
// row-major UE4M3 scales directly from cache_moe_native_nvfp4's nvfp4_moe_ms_native buffer.
//   A_i [M_i,K] packed NVFP4, K/2 bytes/row; SFA_i [M_i,K/16] UE4M3 native row-major
//   B_i [N,K] packed NVFP4; SFB_i [N,K/16] UE4M3 native row-major
//   D_i [M_i,N] FP16 RowMajor; alpha_i per-expert tensor_scale as GEMM alpha
// dev_alpha is a DEVICE pointer to [n_experts] floats; caller keeps it live until the stream
// consumes the kernel. K and N must be identical across experts; M_i varies.
// dev_tables: caller-owned device buffer, >= smallM_table_bytes(n_experts), 128 B aligned, reused
// stream-ordered across calls; the kernel's pointer/M/descriptor tables are copied into it (#2451).
[[nodiscard]] bool gemm_grouped_nvfp4_smallM(
    int n_experts,
    const int* host_M,                // [n_experts] M_i per expert
    int N, int K,
    const void* const* host_ptr_A,    // [n_experts] device packed A
    const void* const* host_ptr_SFA,  // [n_experts] device SFA (native row-major)
    const void* const* host_ptr_B,    // [n_experts] device packed B
    const void* const* host_ptr_SFB,  // [n_experts] device SFB (native row-major)
    void* const* host_ptr_D,          // [n_experts] device FP16 outputs
    const float* dev_alpha,           // [n_experts] per-expert tensor_scale (DEVICE ptr)
    void* dev_tables, size_t dev_tables_bytes,
    cudaStream_t stream);

// Tables per call, each at a 128 B boundary: 5 pointer arrays, M, 2 CUtensorMap (128 B) per expert.
inline constexpr size_t kSmallMTensorMapBytes = 128;
inline constexpr size_t smallM_table_bytes(int n_experts) {
    const size_t ne = n_experts > 0 ? static_cast<size_t>(n_experts) : 0;
    auto r128 = [](size_t b) { return (b + 127) / 128 * 128; };
    return 5 * r128(ne * sizeof(void*)) + r128(ne * sizeof(int)) + 2 * ne * kSmallMTensorMapBytes;
}

[[nodiscard]] bool gemm_grouped_nvfp4_smallM_available();

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
