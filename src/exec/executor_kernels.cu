#include "exec/executor_kernels.h"
#include "exec/gemm_context.h"
#include "exec/gemm_kernel_registry.h"
#include "exec/executor.h"
#include "compute/weight_dispatch.h"
#include "core/tensor_kind.h"
#include "core/logging.h"
#include "runtime/config.h"
#include "compute/gemm.h"
#include "compute/gemm_q4k.h"
#include "compute/gemm_q6k.h"
#include "compute/gemm_cutlass_sm120.h"
#include "compute/gemm_cutlass_mxfp4_sm120.h"
#include "compute/hadamard.h"
#include "quant/quant_gemm.h"
#include "quant/dequant_gpu.h"
#include "quant/fp8_quant.h"
#include "quant/nvfp4_gemm.h"
#include "quant/mxfp4_gemm.h"
#include "compute/ggml_mmvq.h"
#include "compute/mmq_q8_imma.h"
#include "exec/gemm_kernel_q4k_hmma.h"
#include "compute/hadamard.h"
#include "runtime/pdl.h"
#include "compute/ptx92_utils.cuh"
#include "compute/warp_reduce.cuh"  // kWarpSize

#include <cuda_bf16.h>

namespace imp {

// ---------------------------------------------------------------------------
// dp4a GEMV dispatch helper (file-local)
// ---------------------------------------------------------------------------

// Dispatch dp4a GEMV by quant type: y = W @ q8_1 (FP16 output).
// Defined here, declared in executor_kernels.h for use by executor_forward.cu.
void dispatch_dp4a_gemv(QType qtype, const void* W, const block_q8_1* q8_1, const float* d8, half* y, int M,
                        int K, cudaStream_t stream) {
    switch (qtype) {
        case QType::Q6_K:
            gemv_q6k_q8_1(W, q8_1, d8, y, M, K, stream);
            break;
        case QType::Q4_0:
            gemv_q4_0_q8_1(W, q8_1, d8, y, M, K, stream);
            break;
        case QType::Q4_K:
            gemv_q4_k_q8_1(W, q8_1, d8, y, M, K, stream);
            break;
        case QType::Q5_K:
            gemv_q5_k_q8_1(W, q8_1, d8, y, M, K, stream);
            break;
        case QType::Q2_K:
            gemv_q2_k_q8_1(W, q8_1, d8, y, M, K, stream);
            break;
        case QType::Q3_K:
            gemv_q3_k_q8_1(W, q8_1, d8, y, M, K, stream);
            break;
        default:
            gemv_q8_0_q8_1(W, q8_1, d8, y, M, K, stream);
            break;
    }
}

// ---------------------------------------------------------------------------
// GDN attention output-gate split: replaces nh × 2 cudaMemcpy2DAsync loop
// with one launch. Source row layout per token is interleaved
// [Q_h0 | Gate_h0 | Q_h1 | Gate_h1 | ...] each chunk of size hd; both
// destinations are contiguous [n, nh*hd]. Grid: (n × nh) blocks of hd
// threads — each block copies one (token, head) pair's Q + gate vectors.
// ---------------------------------------------------------------------------
template <typename T>
__global__ __launch_bounds__(256) void attn_gate_split_interleaved_kernel(
    const T* __restrict__ src, T* __restrict__ q_dst, T* __restrict__ gate_dst, int n_tokens, int nh,
    int hd, int q_out_dim) {
    int t = blockIdx.x;
    int h = blockIdx.y;
    int tid = threadIdx.x;
    if (t >= n_tokens || h >= nh || tid >= hd)
        return;
    const T* src_row = src + static_cast<int64_t>(t) * q_out_dim;
    int64_t dst_off = static_cast<int64_t>(t) * (nh * hd) + static_cast<int64_t>(h) * hd + tid;
    int q_src = h * 2 * hd + tid;
    int g_src = h * 2 * hd + hd + tid;
    q_dst[dst_off] = src_row[q_src];
    gate_dst[dst_off] = src_row[g_src];
}

// Explicit template instantiations (FP16 + BF16 paths used by attention).
template __global__ void attn_gate_split_interleaved_kernel<half>(
    const half*, half*, half*, int, int, int, int);
template __global__ void attn_gate_split_interleaved_kernel<__nv_bfloat16>(
    const __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, int, int, int, int);

void attn_gate_split_interleaved(const void* src, void* q_dst, void* gate_dst, int n_tokens, int nh, int hd,
                                 int q_out_dim, int element_bytes, cudaStream_t stream) {
    if (n_tokens <= 0 || nh <= 0 || hd <= 0)
        return;
    int threads = (hd <= 256) ? hd : 256;
    dim3 grid(n_tokens, nh);
    if (element_bytes == 2) {
        // FP16 path (also used for BF16 since same byte width — caller
        // controls reinterpret).
        attn_gate_split_interleaved_kernel<half><<<grid, threads, 0, stream>>>(
            static_cast<const half*>(src), static_cast<half*>(q_dst), static_cast<half*>(gate_dst), n_tokens,
            nh, hd, q_out_dim);
        IMP_CUDA_CHECK_LAUNCH();
    } else {
        // FP32 fallback — uses uint32_t reinterpret since templated half→FP32
        // dispatch requires another instantiation. For now log + fall through
        // to caller's loop on unsupported dtype.
        // (No FP32 path expected in attention compute; keep guard for safety.)
    }
}

}  // namespace imp
