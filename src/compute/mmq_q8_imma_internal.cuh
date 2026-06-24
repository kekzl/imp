#ifndef IMP_COMPUTE_MMQ_Q8_IMMA_INTERNAL_CUH
#define IMP_COMPUTE_MMQ_Q8_IMMA_INTERNAL_CUH

// =============================================================================
// mmq_q8_imma_internal.cuh — shared internals for the INT8 IMMA prefill GEMM
// family (sm_120a). Split out of mmq_q8_imma.cu (recompile-blast-radius gate).
//
// Holds the tile constants, the cp.async primitives, and the cross-TU template
// kernel declarations. Definitions live in the per-format .cu files
// (mmq_q8_imma_q4k.cu / _q6k.cu / _q51.cu); the dispatch in mmq_q8_imma.cu
// launches them. Kept BYTE-IDENTICAL to the original inline code.
// =============================================================================

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace imp {

constexpr int kBN = 128;
constexpr int kBK = 64;   // 2 sub-blocks per K-step
constexpr int kPad = 16;  // smem row pad (bytes): 80-B stride = conflict-free rl-lane access
constexpr int kRow = kBK + kPad;
constexpr int kStages = 2;
constexpr int kThreads = 256;

__device__ __forceinline__ void cp_async_cg_16(void* smem, const void* glob, bool valid) {
    uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    int src_size = valid ? 16 : 0;  // src-size 0 → zero-fill (OOB M-tail rows)
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(glob),
                 "r"(src_size));
}
__device__ __forceinline__ void cp_async_ca_8(void* smem, const void* glob, bool valid) {
    uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    int src_size = valid ? 8 : 0;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;\n" ::"r"(s), "l"(glob),
                 "r"(src_size));
}
__device__ __forceinline__ void cp_async_ca_4(void* smem, const void* glob, bool valid) {
    uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    int src_size = valid ? 4 : 0;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(s), "l"(glob),
                 "r"(src_size));
}
__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n");
}
template <int N>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

constexpr int kQ6Stride = 224;  // repacked super-block stride (210 padded, 16-B aligned)
constexpr int kQlRow = 64 + 16;
constexpr int kQhRow = 32 + 16;

// Cross-TU template kernel declarations (definitions + explicit instantiations
// live in the per-format .cu files). External linkage so the dispatch in
// mmq_q8_imma.cu can launch / cudaFuncSetAttribute them.
template <int BM, bool BETA1>
__global__ void mmq_imma_q4k_raw_kernel(const int8_t* __restrict__ X_s8,
                                        const __half* __restrict__ x_scale,
                                        const float* __restrict__ x_rowsum,
                                        const uint8_t* __restrict__ Wq4k, __half* __restrict__ out,
                                        int M, int N, int K,
                                        const int32_t* __restrict__ expert_offsets,
                                        size_t w_stride_blocks);

template <int BM, bool BETA1>
__global__ void mmq_imma_q6k_raw_kernel(const int8_t* __restrict__ X_s8,
                                        const __half* __restrict__ x_scale,
                                        const uint8_t* __restrict__ Wq6k, __half* __restrict__ out,
                                        int M, int N, int K,
                                        const int32_t* __restrict__ expert_offsets,
                                        size_t w_stride_blocks);

template <int BM, bool BETA1>
__global__ void mmq_imma_q51_raw_kernel(const int8_t* __restrict__ X_s8,
                                        const __half* __restrict__ x_scale,
                                        const float* __restrict__ x_rowsum,
                                        const uint8_t* __restrict__ Wq51, __half* __restrict__ out,
                                        int M, int N, int K,
                                        const int32_t* __restrict__ expert_offsets,
                                        size_t w_stride_blocks);

constexpr size_t q6k_smem_bytes(int BM) {
    return static_cast<size_t>(kStages) *
           (static_cast<size_t>(BM) * kRow + static_cast<size_t>(kBN) * kQlRow +
            static_cast<size_t>(kBN) * kQhRow + static_cast<size_t>(kBN) * 8 +
            static_cast<size_t>(BM) * 4);
}

}  // namespace imp

#endif  // IMP_COMPUTE_MMQ_Q8_IMMA_INTERNAL_CUH
