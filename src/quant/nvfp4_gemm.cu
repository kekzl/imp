#include "quant/nvfp4_gemm.h"
#include "quant/nvfp4_quant.h"
#include "compute/gemm.h"
#include "core/tensor.h"
#include "core/logging.h"
#include "runtime/pdl.h"
#include "runtime/process_diag.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublasLt.h>
#include <cstdint>
#include <cassert>
#include <mutex>
#include <stdexcept>

namespace imp {

// ---------------------------------------------------------------------------
// The NVFP4 GEMV kernels themselves live in the per-family TUs (kept under the
// kernel .cu size gate): nvfp4_gemv_dense.cu, nvfp4_gemv_fused.cu and
// nvfp4_gemv_moe.cu. This TU keeps the tensor-based GEMV wrapper, the
// dequant-to-FP16 GEMM fallback, and the PDL registration. Forward-declare the
// kernel symbols that nvfp4_gemv_pdl_register() needs to take the address of;
// the definitions stay byte-identical in their respective TUs.
// ---------------------------------------------------------------------------
__global__ void gemv_nvfp4_kpar_kernel(const uint8_t* packed_data, const uint8_t* micro_scales,
                                       float tensor_scale, const half* x, half* y, int M, int K);
__global__ void gemv_nvfp4_kpar_fp32_kernel(const uint8_t* packed_data, const uint8_t* micro_scales,
                                            float tensor_scale, const half* x, float* y, int M, int K);
template <int NR>
__global__ void gemv_nvfp4_multirow_kernel(const uint8_t* packed_data, const uint8_t* micro_scales,
                                           float tensor_scale, const half* x, half* y, int M, int K);
template <int NR>
__global__ void gemv_nvfp4_multirow_fp32_kernel(const uint8_t* packed_data, const uint8_t* micro_scales,
                                                float tensor_scale, const half* x, float* y, int M, int K);
__global__ void gemv_nvfp4_residual_kernel(const uint8_t* packed_data, const uint8_t* micro_scales,
                                           float tensor_scale, const half* x, half* y, const half* residual,
                                           int M, int K);
template <int NR>
__global__ void gemv_nvfp4_residual_mr_kernel(const uint8_t* packed_data, const uint8_t* micro_scales,
                                              float tensor_scale, const half* x, half* y,
                                              const half* residual, int M, int K);
__global__ void gemv_nvfp4_qkv_fused_kernel(const uint8_t* packed_q, const uint8_t* ms_q, float ts_q,
                                            const uint8_t* packed_k, const uint8_t* ms_k, float ts_k,
                                            const uint8_t* packed_v, const uint8_t* ms_v, float ts_v,
                                            const half* x, half* yq, half* yk, half* yv, int q_rows,
                                            int k_rows, int v_rows, int K);
template <int NR>
__global__ void gemv_nvfp4_qkv_fused_mr_kernel(const uint8_t* packed_q, const uint8_t* ms_q, float ts_q,
                                               const uint8_t* packed_k, const uint8_t* ms_k, float ts_k,
                                               const uint8_t* packed_v, const uint8_t* ms_v, float ts_v,
                                               const half* x, half* yq, half* yk, half* yv, int q_rows,
                                               int k_rows, int v_rows, int K);
__global__ void gemv_nvfp4_gate_up_fused_kernel(const uint8_t* packed_g, const uint8_t* ms_g, float ts_g,
                                                const uint8_t* packed_u, const uint8_t* ms_u, float ts_u,
                                                const half* x, half* yg, half* yu, int rows, int K);
template <int NR>
__global__ void gemv_nvfp4_gate_up_fused_mr_kernel(const uint8_t* packed_g, const uint8_t* ms_g, float ts_g,
                                                   const uint8_t* packed_u, const uint8_t* ms_u, float ts_u,
                                                   const half* x, half* yg, half* yu, int rows, int K);
__global__ void gemv_nvfp4_swiglu_residual_kernel(const uint8_t* packed_data, const uint8_t* micro_scales,
                                                  float tensor_scale, const half* gate, const half* up,
                                                  half* y, const half* residual, int M, int K);
template <int NR>
__global__ void gemv_nvfp4_swiglu_residual_mr_kernel(const uint8_t* packed_data, const uint8_t* micro_scales,
                                                     float tensor_scale, const half* gate, const half* up,
                                                     half* y, const half* residual, int M, int K);
__global__ void gemv_nvfp4_geglu_residual_kernel(const uint8_t* packed_data, const uint8_t* micro_scales,
                                                 float tensor_scale, const half* gate, const half* up,
                                                 half* y, const half* residual, int M, int K);
template <int NR>
__global__ void gemv_nvfp4_geglu_residual_mr_kernel(const uint8_t* packed_data, const uint8_t* micro_scales,
                                                    float tensor_scale, const half* gate, const half* up,
                                                    half* y, const half* residual, int M, int K);
__global__ void gemv_nvfp4_moe_decode_kernel(const uint8_t* packed_data, const uint8_t* micro_scales,
                                             const float* tensor_scales, const int32_t* expert_indices,
                                             const half* x, half* y, int rows, int K,
                                             size_t expert_stride_packed, size_t expert_stride_ms,
                                             int x_stride, int blocks_per_expert);
__global__ void gemv_nvfp4_moe_gate_up_fused_kernel(
    const uint8_t* gate_packed, const uint8_t* gate_ms, const float* gate_ts, const uint8_t* up_packed,
    const uint8_t* up_ms, const float* up_ts, const int32_t* expert_indices, const half* x, half* y_gate,
    half* y_up, int rows, int K, size_t expert_stride_packed, size_t expert_stride_ms, int blocks_per_expert);

// ---------------------------------------------------------------------------
// Tensor-based launcher (existing API, delegates to K-parallel kernel)
// ---------------------------------------------------------------------------
void gemv_nvfp4(const NvFP4QuantResult& A, const Tensor& x, Tensor& y, cudaStream_t stream) {
    IMP_CHECK(A.packed_data != nullptr, "gemv_nvfp4: A.packed_data is null");
    IMP_CHECK(x.on_device, "gemv_nvfp4: x must be on device");
    IMP_CHECK(y.on_device, "gemv_nvfp4: y must be on device");
    IMP_CHECK(x.qtype == QType::F16, "gemv_nvfp4: x must be FP16, got qtype=%d", static_cast<int>(x.qtype));
    IMP_CHECK(y.qtype == QType::F16, "gemv_nvfp4: y must be FP16, got qtype=%d", static_cast<int>(y.qtype));

    int M = static_cast<int>(A.N);
    int K = static_cast<int>(A.K);

    gemv_nvfp4_kpar(A, reinterpret_cast<const half*>(x.data), reinterpret_cast<half*>(y.data), M, K, stream);
}

// ---------------------------------------------------------------------------
// GEMM for NVFP4 weights:  C = input @ A^T
//
//   A (NvFP4QuantResult): weight matrix [N, K] in NVFP4 packed format
//   input (Tensor):       activation     [M, K] in FP16
//   C (Tensor):           output         [M, N] in FP16
//
// Strategy: dequantize A to FP16, then call cuBLAS gemm.
// ---------------------------------------------------------------------------

static void* s_nvfp4_dequant_buf = nullptr;
static size_t s_nvfp4_dequant_buf_size = 0;
static std::mutex s_nvfp4_dequant_mtx;

// FP32 -> FP16 copy for the small-M batched-GEMV path (the batched kernel
// accumulates and writes FP32; gemm_nvfp4's contract is an FP16 C).
__global__ void nvfp4_fp32_to_fp16_kernel(const float* __restrict__ in, __half* __restrict__ out,
                                          int64_t n) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = __float2half(in[i]);
}

// Pre-allocated workspace from the executor. When set, ensure_dequant_buffer
// uses this instead of the lazy cudaMalloc path — which would fail inside
// CUDA stream capture. Lifetime owned by caller of set_nvfp4_dequant_workspace.
static void* s_nvfp4_dequant_ws_buf = nullptr;
static size_t s_nvfp4_dequant_ws_size = 0;

void set_nvfp4_dequant_workspace(void* buf, size_t size_bytes) {
    std::lock_guard<std::mutex> lock(s_nvfp4_dequant_mtx);
    s_nvfp4_dequant_ws_buf = buf;
    s_nvfp4_dequant_ws_size = size_bytes;
}

size_t nvfp4_lazy_dequant_buf_size_for_testing() {
    std::lock_guard<std::mutex> lock(s_nvfp4_dequant_mtx);
    return s_nvfp4_dequant_buf_size;
}

// Must be called with s_nvfp4_dequant_mtx held.
static void* ensure_dequant_buffer(size_t needed, cudaStream_t stream) {
    // Prefer the pre-allocated workspace (graph-safe path).
    if (s_nvfp4_dequant_ws_buf && needed <= s_nvfp4_dequant_ws_size)
        return s_nvfp4_dequant_ws_buf;

    // If the stream is in capture mode, cudaMalloc would fail with "operation
    // not permitted when stream is capturing". Refuse cleanly with a clear
    // error rather than letting the runtime crash the capture state.
    cudaStreamCaptureStatus cap_status = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &cap_status) == cudaSuccess &&
        cap_status == cudaStreamCaptureStatusActive) {
        IMP_LOG_ERROR(
            "gemm_nvfp4: dequant fallback called inside CUDA stream capture but no "
            "pre-allocated workspace covers %zu bytes (have %zu). cudaMalloc inside "
            "capture would fail. Pre-allocate via set_nvfp4_dequant_workspace() "
            "before capture begins.",
            needed, s_nvfp4_dequant_ws_size);
        return nullptr;
    }

    // Non-capture path: legacy lazy cudaMalloc (re-grows as needed).
    if (needed <= s_nvfp4_dequant_buf_size)
        return s_nvfp4_dequant_buf;
    if (s_nvfp4_dequant_buf)
        cudaFree(s_nvfp4_dequant_buf);
    s_nvfp4_dequant_buf = nullptr;
    s_nvfp4_dequant_buf_size = 0;
    cudaError_t err = cudaMalloc(&s_nvfp4_dequant_buf, needed);
    if (err != cudaSuccess) {
        IMP_LOG_ERROR("gemm_nvfp4: failed to allocate %zu bytes for dequant buffer: %s", needed,
                      cudaGetErrorString(err));
        return nullptr;
    }
    s_nvfp4_dequant_buf_size = needed;
    return s_nvfp4_dequant_buf;
}

void gemm_nvfp4(const NvFP4QuantResult& A, const Tensor& B, Tensor& C, cudaStream_t stream, float beta) {
    IMP_CHECK(A.packed_data != nullptr, "gemm_nvfp4: A.packed_data is null");
    IMP_CHECK(B.on_device, "gemm_nvfp4: B (input) must be on device");
    IMP_CHECK(C.on_device, "gemm_nvfp4: C (output) must be on device");
    IMP_CHECK(B.ndim == 2, "gemm_nvfp4: B must be 2D [M, K], got ndim=%d", B.ndim);
    IMP_CHECK(C.ndim == 2, "gemm_nvfp4: C must be 2D [M, N], got ndim=%d", C.ndim);

    const int64_t N = A.N;
    const int64_t K = A.K;
    const int64_t M = B.shape[0];

    IMP_CHECK(B.shape[1] == K, "gemm_nvfp4: B.shape[1]=%lld must equal weight K=%lld",
              static_cast<long long>(B.shape[1]), static_cast<long long>(K));
    IMP_CHECK(C.shape[0] == M && C.shape[1] == N,
              "gemm_nvfp4: C shape [%lld, %lld] must equal [M=%lld, N=%lld]",
              static_cast<long long>(C.shape[0]), static_cast<long long>(C.shape[1]),
              static_cast<long long>(M), static_cast<long long>(N));

    if (M == 1) {
        gemv_nvfp4(A, B, C, stream);
        return;
    }

    // Small-M chunks (spec-verify: M = drafts+1, short/boundary prefills): the
    // dequant fallback below rewrites the whole FP16 weight EVERY call — on
    // Qwen3.6-27B MTP-only verify that was 49% of all GPU time (nsys, ~52
    // dequants x ~600 us per verify). The batched-M GEMV reads the NVFP4
    // weight once per MR=4 tile instead: at M<=16 that is <=4 passes at 0.25x
    // FP16 bytes vs the fallback's ~2.25x (dequant read+write + GEMM read).
    // beta!=0, non-F16 output, and the nvfp4-force-dequant bisect flag keep
    // the fallback. FP32 accumulate + convert — same numerics class as the
    // M=1 decode GEMV.
    constexpr int64_t kSmallMBatchedGemv = 16;
    if (M <= kSmallMBatchedGemv && beta == 0.0f && B.qtype == QType::F16 &&
        C.qtype == QType::F16 && !process_diag_nvfp4_force_dequant()) {
        std::lock_guard<std::mutex> lock(s_nvfp4_dequant_mtx);
        const size_t y32_bytes = static_cast<size_t>(M * N) * sizeof(float);
        void* y32 = ensure_dequant_buffer(y32_bytes, stream);
        if (y32 != nullptr) {
            gemv_nvfp4_kpar_batched_fp32(A, reinterpret_cast<const half*>(B.data),
                                         static_cast<float*>(y32), static_cast<int>(N),
                                         static_cast<int>(K), static_cast<int>(M), stream);
            const int64_t total = M * N;
            const int block = 256;
            const int grid = static_cast<int>((total + block - 1) / block);
            nvfp4_fp32_to_fp16_kernel<<<grid, block, 0, stream>>>(
                static_cast<const float*>(y32), reinterpret_cast<half*>(C.data), total);
            return;
        }
        // Scratch unavailable (capture-active without workspace): fall through —
        // the dequant path below throws the loud capture error.
    }

    // Fallback path: dequantize full weight matrix to FP16 and use cuBLAS GEMM.
    // This is slow and memory-heavy — warn on first use.
    static bool s_warned = false;
    if (!s_warned) {
        IMP_LOG_WARN(
            "gemm_nvfp4: using slow dequant-to-FP16 fallback for M=%lld "
            "(CUTLASS/cuBLASLt NVFP4 unavailable). "
            "Allocating %.1f MiB dequant buffer for [%lld, %lld] weight matrix",
            (long long)M, (double)((size_t)(N * K) * sizeof(half)) / (1024.0 * 1024.0), (long long)N,
            (long long)K);
        s_warned = true;
    }

    std::lock_guard<std::mutex> lock(s_nvfp4_dequant_mtx);

    size_t A_fp16_bytes = (size_t)(N * K) * sizeof(half);
    void* dequant_buf = ensure_dequant_buffer(A_fp16_bytes, stream);
    if (!dequant_buf) {
        // A silent return here corrupts whatever runs next: the output tensor
        // keeps garbage, and under stream capture the recorded graph simply
        // LACKS this GEMM — the #855 census "hybrid crash" was this exact
        // hole (Nemotron: pre-alloc skipped for a >cap weight, fallback
        // refused mid-capture, graph launched with an uninitialized
        // activation buffer -> misaligned address). Throw instead; the
        // verify capturer fails the capture cleanly and falls back eager.
        throw std::runtime_error("gemm_nvfp4: no dequant workspace for M>1 fallback (capture-active "
                                 "or allocation failure) — cannot run this GEMM");
    }

    dequantize_nvfp4_to_fp16(A, dequant_buf, stream);

    int64_t A_shape[2] = {N, K};
    Tensor A_fp16(dequant_buf, QType::F16, 2, A_shape, /*on_device=*/true);

    gemm(B, A_fp16, C, 1.0f, beta, stream);
}

// ---------------------------------------------------------------------------
// PDL registration for all NVFP4 GEMV kernels.
// Called from GraphExecutor::init() when PDL is enabled.
// ---------------------------------------------------------------------------
void nvfp4_gemv_pdl_register() {
    constexpr int NR = 8;

// GEMV kernels are bandwidth-bound with minimal SMEM usage.
// Maximize L1 cache to improve weight data locality.
#define NVFP4_REGISTER(kern)                                                       \
    do {                                                                           \
        pdl::enable_kernel(kern);                                                  \
        cudaFuncSetAttribute(kern, cudaFuncAttributePreferredSharedMemoryCarveout, \
                             cudaSharedmemCarveoutMaxL1);                          \
    } while (0)

    // Dense GEMV kernels
    NVFP4_REGISTER(gemv_nvfp4_kpar_kernel);
    NVFP4_REGISTER(gemv_nvfp4_kpar_fp32_kernel);
    NVFP4_REGISTER(gemv_nvfp4_multirow_kernel<NR>);
    NVFP4_REGISTER(gemv_nvfp4_multirow_fp32_kernel<NR>);
    NVFP4_REGISTER(gemv_nvfp4_residual_kernel);
    NVFP4_REGISTER(gemv_nvfp4_residual_mr_kernel<NR>);
    NVFP4_REGISTER(gemv_nvfp4_qkv_fused_kernel);
    NVFP4_REGISTER(gemv_nvfp4_qkv_fused_mr_kernel<NR>);
    NVFP4_REGISTER(gemv_nvfp4_gate_up_fused_kernel);
    NVFP4_REGISTER(gemv_nvfp4_gate_up_fused_mr_kernel<NR>);
    NVFP4_REGISTER(gemv_nvfp4_swiglu_residual_kernel);
    NVFP4_REGISTER(gemv_nvfp4_swiglu_residual_mr_kernel<NR>);
    NVFP4_REGISTER(gemv_nvfp4_geglu_residual_kernel);
    NVFP4_REGISTER(gemv_nvfp4_geglu_residual_mr_kernel<NR>);
    // MoE GEMV kernels
    NVFP4_REGISTER(gemv_nvfp4_moe_decode_kernel);
    NVFP4_REGISTER(gemv_nvfp4_moe_gate_up_fused_kernel);

#undef NVFP4_REGISTER
}

}  // namespace imp
