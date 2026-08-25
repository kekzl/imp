#pragma once
// Marlin W4A16 GEMM for NVFP4 weights on sm_120a — imp-native entry points
// around the vendored vLLM Marlin kernel (Apache-2.0; see marlin_template.h).
//
// Why: batched decode at n_seq<=32 pays 613 us/token in the GEMM class on the
// CUTLASS 128x128 block-scaled tile while vLLM's Marlin does the same class in
// 468 on the same card (BENCHMARKS.md, "The 1.58x concurrency gap"). Seven
// no-K-split approaches were measured to a local optimum below that
// (docs/plans/2026-08-24-qwen38-port.md); Marlin's striped split-K with
// global reduce is the design that wins, so imp runs the real thing.
//
// Weight path: plain NVFP4 (packed nibbles [N,K/2] + FP8-E4M3 micro-scales
// [N,K/16] + tensor scale) is repacked at init into Marlin tile layout with
// scales in the shifted "S0E5M3" format vLLM prepares for FP16 activations
// (marlin_utils_fp4.py recipe, scale_factor=1 for FP16).

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace imp {
namespace marlin_w4a16 {

// Prepared weight in Marlin layout. Buffers are device memory owned by the
// holder (see prepare()).
struct MarlinWeight {
    void* qweight = nullptr;          // [K/16, N*16/8] int32 Marlin tiles (K*N/2 bytes)
    void* scales = nullptr;           // [K/16, N] uint8 processed scales
    float* d_global_scale = nullptr;  // 1 float on device: tensor_scale * 2^7
    int N = 0;
    int K = 0;
};

// True when (N, K) is servable: K % 64 == 0, N % 64 == 0, K % 16 == 0 groups.
bool shape_supported(int N, int K);

// Repack a plain-NVFP4 weight into Marlin layout. Allocates the three device
// buffers into `out` (cudaMalloc; caller owns / frees via release()).
// d_packed:       [N, K/2] packed FP4 nibbles (low nibble = even k)
// d_micro_scales: [N, K/16] FP8 E4M3
// Synchronizes the stream (host-side scale processing). Init-time only.
// Returns false (and leaves `out` empty) on unsupported shape or OOM.
bool prepare(const void* d_packed, const void* d_micro_scales, float tensor_scale, int N, int K,
             MarlinWeight& out, cudaStream_t stream);

void release(MarlinWeight& w);

// Device scratch the GEMM needs, allocated once by the caller:
//  - locks:  workspace_bytes() of int32, ZEROED once after allocation
//    (the kernel restores them to zero on completion).
//  - c_tmp:  c_tmp_bytes(max_m) of float for the FP32 global reduce.
size_t workspace_bytes();
size_t c_tmp_bytes(int max_m);

// C[M, N] = A[M, K] @ dequant(W)^T. A row-major FP16 with row stride lda
// (elements), C row-major FP16, overwritten. Graph-capture-safe: no
// allocations, no host sync. Returns false if no kernel instantiation covers
// the shape (caller falls back).
bool gemm(const MarlinWeight& W, const half* A, half* C, int M, int lda, int* locks, float* c_tmp,
          cudaStream_t stream);

}  // namespace marlin_w4a16
}  // namespace imp
