#pragma once

#include "core/tensor.h"
#include "model/model_config.h"  // QType (Q6_K, Q4_0, etc.)
#include "compute/gemm.h"        // block_q8_1
#include "exec/decode_pipeline_advance.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <unordered_map>

namespace imp {

// Forward declarations for legacy gemm_dispatch parameters
struct FP8CacheEntry;
struct NvFP4QuantResult;
struct NvFP4SfAtomWeight;
struct MxFP4SfAtomWeight;

// Host launcher: dispatches the right T based on dtype size (2 = half/bf16).
void attn_gate_split_interleaved(const void* src, void* q_dst, void* gate_dst, int n_tokens, int nh, int hd,
                                 int q_out_dim, int element_bytes, cudaStream_t stream);

// Host launchers for executor_attention.cpp (kernels in executor_kernels.cuh).
// rmsnorm_fp32_accum_to_fp16_kernel<<<n, 256>>>.
void rmsnorm_fp32_accum_to_fp16(const half* input, const half* norm_w, float* fp32_accum, half* output, int n,
                                int d_model, float eps, float weight_offset, cudaStream_t stream);
// Arguments of write_kv_cache_rope_fused (audit D-11): designated initializers keep the eight
// adjacent ints (block_stride .. head_dim) from swapping silently.
struct KvWriteRopeArgs {
    dim3 grid = dim3();  // 1 x 1 x 1
    int threads = 0;
    cudaStream_t stream = nullptr;
    const half* k_in = nullptr;
    const half* v_in = nullptr;
    const int* positions = nullptr;
    const int* block_tables = nullptr;
    half* k_cache_base = nullptr;
    half* v_cache_base = nullptr;
    int block_stride = 0;
    int row_elems = 0;
    int block_size = 0;
    int n_tokens = 0;
    int max_blocks_per_seq = 0;
    int n_sequences = 0;
    int n_kv_heads = 0;
    int head_dim = 0;
    float theta = 0.0f;
    float inv_scaling = 1.0f;
    int rope_pairs = 0;
    bool neox = false;
    const float* longrope_inv_freqs = nullptr;
};
// write_kv_cache_rope_fused_kernel<<<grid, threads>>>.
void write_kv_cache_rope_fused(const KvWriteRopeArgs& a);
// rope_q_only_fp16_kernel<<<dim3(1, n_heads), rope_pairs>>>.
void rope_q_only_fp16(half* Q, const int* positions, int n_heads, int head_dim, float theta, float inv_scaling,
                      int rope_pairs, bool neox, const float* longrope_inv_freqs, cudaStream_t stream);

// ---------------------------------------------------------------------------
// dp4a GEMV helpers (shared by executor_forward.cu and executor_kernels.cu)
// ---------------------------------------------------------------------------

// Returns true if the quant type supports dp4a (Q8_1-input) GEMV kernels.
inline bool is_dp4a_qtype(QType qt) {
    return qt == QType::Q6_K || qt == QType::Q8_0 || qt == QType::Q4_0 || qt == QType::Q4_K ||
           qt == QType::Q5_K || qt == QType::Q2_K || qt == QType::Q3_K || qt == QType::IQ4_NL ||
           qt == QType::IQ4_XS || qt == QType::IQ3_S;
}

// Dispatch dp4a GEMV by quant type: y = W @ q8_1 (FP16 output).
void dispatch_dp4a_gemv(QType qtype, const void* W, const block_q8_1* q8_1, const float* d8, half* y, int M,
                        int K, cudaStream_t stream);

// ---------------------------------------------------------------------------
// Host-side helper functions
// ---------------------------------------------------------------------------

void elementwise_add(Tensor& a, const Tensor& b, cudaStream_t stream);

// PDL registration for elementwise_add_fp16_kernel; defined next to the kernel.
void elementwise_add_pdl_register();

// D2D copy as a kernel launch (stream-async, ~10 us host cost) instead of
// cudaMemcpyAsync's WDDM DMA submission (~165 us blocked host time/call on
// this WSL2 host). Use for per-layer copies on the decode hot path; falls back to cudaMemcpyAsync for
// unaligned buffers.
void device_copy_async(void* dst, const void* src, size_t bytes, cudaStream_t stream);

void elementwise_add_store(const Tensor& a, const Tensor& b, Tensor& out, cudaStream_t stream);

// Cohere2 parallel block: h[i] += a[i] - b[i], FP16 storage, FP32 math.
void parallel_residual_merge(half* h, const half* a, const half* b, int64_t n, cudaStream_t stream);

void add_bias(Tensor& out, const Tensor& bias, cudaStream_t stream);

// Fused 3-way bias add: out_a += bias_a, out_b += bias_b, out_c += bias_c in one launch.
// Skips any output where bias.data == nullptr.
void add_bias_3way(Tensor& out_a, const Tensor& bias_a, Tensor& out_b, const Tensor& bias_b, Tensor& out_c,
                   const Tensor& bias_c, cudaStream_t stream);

// Fused residual add + RMSNorm: hidden += residual; output = rmsnorm(hidden, weight).
// Saves 1 kernel launch + 1 DRAM round-trip vs separate add + norm.
void residual_add_rmsnorm(Tensor& hidden, const Tensor& residual, const Tensor& weight, Tensor& output,
                          float eps, cudaStream_t stream, float weight_offset = 0.0f);

// Fused add-store + RMSNorm: hidden = a + b; hidden = rmsnorm(hidden, weight).
// Replaces: elementwise_add_store(a, b, h) + rmsnorm(h, w, no) + memcpy(h, no).
// 3 ops → 1 kernel. Used by sandwich-norm post-attention and post-FFN paths.
void add_rmsnorm_inplace(const Tensor& a, const Tensor& b, Tensor& hidden, const Tensor& weight, float eps,
                         cudaStream_t stream, float weight_offset = 0.0f);

// Fused RMSNorm + residual add: output = rmsnorm(input, weight) + residual.
// Replaces: rmsnorm(in, w, out) + elementwise_add(out, r).
// 2 ops → 1 kernel. Used by sandwich-norm post-FFN path.
void rmsnorm_add_residual(const Tensor& input, const Tensor& weight, const Tensor& residual, Tensor& output,
                          float eps, cudaStream_t stream, float weight_offset = 0.0f);

Tensor slice_rows(const Tensor& buf, int n_tokens);

// GemmContext forward decl — defined in gemm_context.h.
// The legacy gemm_dispatch free function is now a file-local uncached
// fallback (gemm_dispatch_uncached_fallback in executor_kernels.cu).
struct GemmContext;

// MMVQ scratch buffer prewarm + hot-path getter live in gemm_scratch.h since
// R5 Slice 8.6 (TU hoist).

}  // namespace imp
