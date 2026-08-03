// =============================================================================
// attention_mxfp4_prefill.cu -- MXFP4 tensor core prefill attention (sm_120)
// =============================================================================
//
// Uses CUTLASS block-scaled MXFP4×MXFP4 GEMM for Q·K^T, providing ~2x
// compute throughput over FP16 tensor cores on Blackwell's 5th-gen TCs.
// P·V uses cuBLAS FP16 GEMM (compute-light, memory-heavy).
//
// Pipeline per (batch, head):
//   1. Quantize K [seq_kv, hd] → MXFP4  (once per KV head, reused for GQA)
//   2. Quantize Q [seq_q, hd]  → MXFP4  (per Q head)
//   3. CUTLASS MXFP4 GEMM: S [seq_q, seq_kv] = Q_mxfp4 @ K_mxfp4^T
//   4. Fused scale + softcap + causal mask + softmax (in-place on S)
//   5. cuBLAS FP16 GEMM: O [seq_q, hd] = P @ V
//
// Decode path is unaffected — GEMV is memory-bound, scalar dequant suffices.
// =============================================================================

#include "compute/attention_mxfp4_prefill.h"
#include "compute/gemm_cutlass_mxfp4_sm120.h"
#include "core/cuda_static_reset.h"
#include "core/logging.h"
#include "runtime/process_diag.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cassert>
#include <cfloat>
#include <cstdlib>
#include <mutex>

namespace imp {

// =============================================================================
// Constants
// =============================================================================

static constexpr int kMxGroupSize = 32;  // MXFP4 UE8M0 scale group size
static constexpr int kAtomRows = 128;
static constexpr int kAtomKGroups = 4;
static constexpr int kAtomKElems = kMxGroupSize * kAtomKGroups;  // 128
static constexpr int kAtomSize = kAtomRows * kAtomKGroups;       // 512

// Maximum S matrix size per head. Beyond this, fall back to flash attention.
static constexpr size_t kMaxSBytesPerHead = 256ULL * 1024 * 1024;  // 256 MiB

static constexpr auto kGemmAlgo = CUBLAS_GEMM_AUTOTUNE;

// =============================================================================
// Device helpers (self-contained — no cross-TU __device__ linkage)
// =============================================================================

__device__ __forceinline__ uint8_t mxfp4_float_to_ue8m0(float val) {
    if (val <= 0.0f)
        return 0;
    uint32_t fbits;
    memcpy(&fbits, &val, sizeof(float));
    int f_exp = (int)((fbits >> 23) & 0xFF);
    if (fbits & 0x7FFFFF)
        f_exp++;
    if (f_exp < 0)
        return 0;
    if (f_exp > 254)
        return 254;
    return (uint8_t)f_exp;
}

__device__ __forceinline__ float mxfp4_ue8m0_to_float(uint8_t bits) {
    if (bits == 0)
        return 5.877472e-39f;  // 2^-127
    uint32_t fp32 = ((uint32_t)bits) << 23;
    return __uint_as_float(fp32);
}

__device__ __forceinline__ uint8_t mxfp4_quantize_abs(float abs_val) {
    // Branchless: count of midpoint thresholds exceeded gives the E2M1 code.
    uint8_t code = (abs_val >= 0.25f) + (abs_val >= 0.75f) + (abs_val >= 1.25f) + (abs_val >= 1.75f) +
                   (abs_val >= 2.5f) + (abs_val >= 3.5f) + (abs_val >= 5.0f);
    return code;  // 0..7
}

// HW FP4 conversion: two scaled FP32 values → one byte of packed E2M1
// nibbles (low = v0, high = v1). Replaces the SW cascade above for
// sm_120+. Rounding is IEEE RNE (differs from SW midpoint on boundary).
__device__ __forceinline__ uint8_t mxfp4_pack_pair_hw(float v0, float v1) {
#if __CUDA_ARCH__ >= 1200
    uint32_t out;
    asm volatile(
        "{ .reg .b8 b;\n"
        "  cvt.rn.satfinite.e2m1x2.f32 b, %2, %1;\n"
        "  cvt.u32.u8 %0, b; }\n"
        : "=r"(out)
        : "f"(v0), "f"(v1));
    return static_cast<uint8_t>(out);
#else
    uint8_t sign0 = (v0 < 0.0f) ? 1u : 0u;
    uint8_t code0 = (sign0 << 3) | mxfp4_quantize_abs(fabsf(v0));
    uint8_t sign1 = (v1 < 0.0f) ? 1u : 0u;
    uint8_t code1 = (sign1 << 3) | mxfp4_quantize_abs(fabsf(v1));
    return (code1 << 4) | code0;
#endif
}

__device__ __forceinline__ int mxfp4_sfatom_offset(int row, int k_group, int n_k_tiles) {
    int tile_row = row / kAtomRows;
    int tile_k = k_group / kAtomKGroups;
    int row_local = row % kAtomRows;
    int k_local = k_group % kAtomKGroups;
    int n0 = row_local % 32;
    int n1 = row_local / 32;
    int atom_offset = n0 * 16 + n1 * 4 + k_local;
    int tile_base = (tile_row * n_k_tiles + tile_k) * kAtomSize;
    return tile_base + atom_offset;
}

// =============================================================================
// Strided MXFP4 quantization kernel
// =============================================================================
//
// Reads FP16 data with arbitrary row stride (for per-head access in
// [seq, n_heads*hd] layout), outputs contiguous MXFP4 packed + SfAtom.
// Each thread processes one group of 32 elements.

__global__ void quantize_fp16_mxfp4_strided_kernel(const half* __restrict__ input,
                                                   int input_row_stride,  // in half elements, not bytes
                                                   uint8_t* __restrict__ packed_out,  // [M, K/2] contiguous
                                                   uint8_t* __restrict__ sf_out,      // SfAtom layout UE8M0
                                                   int M, int K, int n_k_tiles) {
    int mb_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int K_groups = K / kMxGroupSize;
    int total_mb = M * K_groups;
    if (mb_idx >= total_mb)
        return;

    int row = mb_idx / K_groups;
    int k_group = mb_idx % K_groups;
    int base = row * input_row_stride + k_group * kMxGroupSize;

    // Load 32 FP16 values via vectorized half2 loads, track absmax
    float vals[32];
    float local_absmax = 0.0f;
    const half2* src_h2 = reinterpret_cast<const half2*>(input + base);
#pragma unroll
    for (int i = 0; i < 16; i++) {
        half2 h2 = src_h2[i];
        vals[i * 2] = __half2float(h2.x);
        vals[i * 2 + 1] = __half2float(h2.y);
        local_absmax = fmaxf(local_absmax, fmaxf(fabsf(vals[i * 2]), fabsf(vals[i * 2 + 1])));
    }

    // UE8M0 scale = ceil_pow2(absmax / 6.0)
    uint8_t ue8m0 = mxfp4_float_to_ue8m0(local_absmax / 6.0f);
    float actual_scale = mxfp4_ue8m0_to_float(ue8m0);
    if (actual_scale == 0.0f)
        actual_scale = 5.877472e-39f;
    float inv_scale = 1.0f / actual_scale;

    // Write scale to SfAtom position
    sf_out[mxfp4_sfatom_offset(row, k_group, n_k_tiles)] = ue8m0;

    // Pack E2M1 nibbles (output is contiguous [M, K/2])
    int packed_base = row * (K / 2) + k_group * (kMxGroupSize / 2);
#pragma unroll
    for (int i = 0; i < 32; i += 2) {
        float s0 = vals[i] * inv_scale;
        float s1 = vals[i + 1] * inv_scale;
        packed_out[packed_base + i / 2] = mxfp4_pack_pair_hw(s0, s1);
    }
}

// =============================================================================
// Fused scale + softcap + causal mask + softmax (in-place on FP16 S matrix)
// =============================================================================
//
// One block per query row. 3-pass online softmax (max, exp+sum, normalize).

__global__ void mxfp4_attn_softmax_kernel(half* __restrict__ S, int seq_q, int seq_kv,
                                          int q_offset,  // global Q position offset (for causal masking)
                                          float scale, float softcap, bool causal) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int n_warps = blockDim.x / 32;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    half* row_ptr = S + (int64_t)row * seq_kv;
    int gq = q_offset + row;

    // --- Pass 1: row max ---
    float max_val = -FLT_MAX;
    for (int j = tid; j < seq_kv; j += blockDim.x) {
        float val;
        if (causal && j > gq) {
            val = -FLT_MAX;
        } else {
            val = __half2float(row_ptr[j]) * scale;
            if (softcap > 0.0f)
                val = softcap * tanhf(val / softcap);
        }
        max_val = fmaxf(max_val, val);
    }

    // Warp-shuffle + cross-warp reduction
    for (int mask = 16; mask > 0; mask >>= 1)
        max_val = fmaxf(max_val, __shfl_xor_sync(0xffffffff, max_val, mask));
    __shared__ float s_reduce[32];
    if (lane_id == 0)
        s_reduce[warp_id] = max_val;
    __syncthreads();
    if (tid < 32) {
        float v = (tid < n_warps) ? s_reduce[tid] : -FLT_MAX;
        for (int mask = 16; mask > 0; mask >>= 1)
            v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, mask));
        s_reduce[0] = v;
    }
    __syncthreads();
    max_val = s_reduce[0];

    // --- Pass 2: exp + sum ---
    float sum_val = 0.0f;
    for (int j = tid; j < seq_kv; j += blockDim.x) {
        float val;
        if (causal && j > gq) {
            val = 0.0f;
        } else {
            val = __half2float(row_ptr[j]) * scale;
            if (softcap > 0.0f)
                val = softcap * tanhf(val / softcap);
            val = expf(val - max_val);
        }
        sum_val += val;
    }

    for (int mask = 16; mask > 0; mask >>= 1)
        sum_val += __shfl_xor_sync(0xffffffff, sum_val, mask);
    if (lane_id == 0)
        s_reduce[warp_id] = sum_val;
    __syncthreads();
    if (tid < 32) {
        float v = (tid < n_warps) ? s_reduce[tid] : 0.0f;
        for (int mask = 16; mask > 0; mask >>= 1)
            v += __shfl_xor_sync(0xffffffff, v, mask);
        s_reduce[0] = v;
    }
    __syncthreads();
    float inv_sum = (s_reduce[0] > 0.0f) ? (1.0f / s_reduce[0]) : 0.0f;

    // --- Pass 3: normalize and write FP16 ---
    for (int j = tid; j < seq_kv; j += blockDim.x) {
        float val;
        if (causal && j > gq) {
            val = 0.0f;
        } else {
            val = __half2float(row_ptr[j]) * scale;
            if (softcap > 0.0f)
                val = softcap * tanhf(val / softcap);
            val = expf(val - max_val) * inv_sum;
        }
        row_ptr[j] = __float2half(val);
    }
}

// =============================================================================
// Static workspace
// =============================================================================

struct MxFP4AttnWorkspace {
    void* q_packed = nullptr;  // [max_seq_q, hd/2]
    void* q_sf = nullptr;      // SfAtom for Q
    void* k_packed = nullptr;  // [max_seq_kv, hd/2]
    void* k_sf = nullptr;      // SfAtom for K
    void* s_matrix = nullptr;  // [max_seq_q, max_seq_kv] FP16
    void* gemm_ws = nullptr;
    size_t gemm_ws_size = 0;
    int alloc_seq_q = 0;
    int alloc_seq_kv = 0;
    int alloc_hd = 0;
};

static MxFP4AttnWorkspace s_ws;
static std::mutex s_ws_mutex;

static void ensure_workspace(int seq_q, int seq_kv, int hd) {
    std::lock_guard<std::mutex> lock(s_ws_mutex);
    if (seq_q <= s_ws.alloc_seq_q && seq_kv <= s_ws.alloc_seq_kv && hd <= s_ws.alloc_hd)
        return;

    // Free existing
    auto safe_free = [](void*& p) {
        if (p) {
            IMP_CUDA_CHECK_LOG(cudaFree(p));
            p = nullptr;
        }
    };
    safe_free(s_ws.q_packed);
    safe_free(s_ws.q_sf);
    safe_free(s_ws.k_packed);
    safe_free(s_ws.k_sf);
    safe_free(s_ws.s_matrix);
    safe_free(s_ws.gemm_ws);

    size_t q_packed_bytes = (size_t)seq_q * (hd / 2);
    size_t k_packed_bytes = (size_t)seq_kv * (hd / 2);
    size_t q_sf_bytes = cutlass_mxfp4_sf_size(seq_q, hd);
    size_t k_sf_bytes = cutlass_mxfp4_sf_size(seq_kv, hd);
    size_t s_bytes = (size_t)seq_q * seq_kv * sizeof(half);

    IMP_CUDA_CHECK_LOG(cudaMalloc(&s_ws.q_packed, q_packed_bytes));
    IMP_CUDA_CHECK_LOG(cudaMalloc(&s_ws.q_sf, q_sf_bytes));
    IMP_CUDA_CHECK_LOG(cudaMalloc(&s_ws.k_packed, k_packed_bytes));
    IMP_CUDA_CHECK_LOG(cudaMalloc(&s_ws.k_sf, k_sf_bytes));
    IMP_CUDA_CHECK_LOG(cudaMalloc(&s_ws.s_matrix, s_bytes));

    size_t gws = gemm_mxfp4_cutlass_sm120_workspace(seq_q, seq_kv, hd);
    if (gws > 0) {
        IMP_CUDA_CHECK_LOG(cudaMalloc(&s_ws.gemm_ws, gws));
        s_ws.gemm_ws_size = gws;
    }

    s_ws.alloc_seq_q = seq_q;
    s_ws.alloc_seq_kv = seq_kv;
    s_ws.alloc_hd = hd;

    IMP_LOG_DEBUG("MXFP4 attn workspace: sq=%d skv=%d hd=%d S=%.1f MiB", seq_q, seq_kv, hd,
                  s_bytes / (1024.0 * 1024.0));
}

// =============================================================================
// cuBLAS handle (separate from gemm.cu to avoid TU coupling)
// =============================================================================

static cublasHandle_t s_cublas_handle = nullptr;  // file-scope so the reset hook can reach it

static cublasHandle_t get_cublas() {
    if (!s_cublas_handle) {
        cublasCreate(&s_cublas_handle);
        cublasSetMathMode(s_cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH);
    }
    return s_cublas_handle;
}

// Pre-cudaDeviceReset hook (see core/cuda_static_reset.h).
void attention_mxfp4_prefill_reset_static_cuda_state() {
    {
        std::lock_guard<std::mutex> lock(s_ws_mutex);
        auto safe_free = [](void*& p) {
            if (p) {
                (void)cudaFree(p);
                p = nullptr;
            }
        };
        safe_free(s_ws.q_packed);
        safe_free(s_ws.q_sf);
        safe_free(s_ws.k_packed);
        safe_free(s_ws.k_sf);
        safe_free(s_ws.s_matrix);
        safe_free(s_ws.gemm_ws);
        s_ws.gemm_ws_size = 0;
        s_ws.alloc_seq_q = 0;
        s_ws.alloc_seq_kv = 0;
        s_ws.alloc_hd = 0;
    }
    if (s_cublas_handle) {
        (void)cublasDestroy(s_cublas_handle);
        s_cublas_handle = nullptr;
    }
}

// Registered as a pre-cudaDeviceReset hook (#1207); see core/cuda_static_reset.h.
namespace {
IMP_REGISTER_CUDA_STATIC_RESET(attention_mxfp4_prefill_reset_static_cuda_state);
}  // namespace

// =============================================================================
// Main entry point
// =============================================================================

bool attention_mxfp4_prefill(const Tensor& Q, const Tensor& K, const Tensor& V, Tensor& O, float scale,
                             bool causal, float softcap, cudaStream_t stream) {
    // --- Extract dimensions ---
    const int batch = static_cast<int>(Q.shape[0]);
    const int seq_q = static_cast<int>(Q.shape[1]);
    const int n_heads = static_cast<int>(Q.shape[2]);
    const int hd = static_cast<int>(Q.shape[3]);
    const int seq_kv = static_cast<int>(K.shape[1]);
    const int n_kv = static_cast<int>(K.shape[2]);

    // --- Validate ---
    if (hd % kMxGroupSize != 0)
        return false;
    if (seq_q == 0 || seq_kv == 0)
        return false;
    if (!cutlass_sm120_mxfp4_available())
        return false;

    size_t s_bytes = (size_t)seq_q * seq_kv * sizeof(half);
    if (s_bytes > kMaxSBytesPerHead) {
        IMP_LOG_DEBUG("MXFP4 attn: S too large (%zu MiB), falling back", s_bytes >> 20);
        return false;
    }

    ensure_workspace(seq_q, seq_kv, hd);

    const int gqa_ratio = n_heads / n_kv;
    const int q_row_stride = n_heads * hd;
    const int kv_row_stride = n_kv * hd;

    // Quantization grid
    const int K_groups = hd / kMxGroupSize;
    const int n_k_tiles = (hd + kAtomKElems - 1) / kAtomKElems;
    const int q_total_mb = seq_q * K_groups;
    const int k_total_mb = seq_kv * K_groups;
    const int q_blocks = (q_total_mb + 255) / 256;
    const int k_blocks = (k_total_mb + 255) / 256;

    size_t q_sf_bytes = cutlass_mxfp4_sf_size(seq_q, hd);
    size_t k_sf_bytes = cutlass_mxfp4_sf_size(seq_kv, hd);

    // cuBLAS for P·V
    cublasHandle_t cublas = get_cublas();
    cublasSetStream(cublas, stream);

    // Reusable K weight descriptor for CUTLASS GEMM
    CutlassMxFP4Weight k_weight;
    k_weight.data = s_ws.k_packed;
    k_weight.scale_factors = s_ws.k_sf;
    k_weight.tensor_scale = 1.0f;  // scales absorbed in UE8M0
    k_weight.N = seq_kv;
    k_weight.K = hd;
    k_weight.sf_bytes = k_sf_bytes;

    half* S = static_cast<half*>(s_ws.s_matrix);

    const half* Q_base = static_cast<const half*>(Q.data);
    const half* K_base = static_cast<const half*>(K.data);
    const half* V_base = static_cast<const half*>(V.data);
    half* O_base = static_cast<half*>(O.data);

    // Softmax thread count heuristic
    int sm_threads = (seq_kv <= 128) ? 128 : (seq_kv <= 256) ? 256 : (seq_kv <= 512) ? 512 : 1024;

    float one = 1.0f, zero = 0.0f;

    for (int b = 0; b < batch; b++) {
        const half* Q_b = Q_base + (int64_t)b * seq_q * q_row_stride;
        const half* K_b = K_base + (int64_t)b * seq_kv * kv_row_stride;
        const half* V_b = V_base + (int64_t)b * seq_kv * kv_row_stride;
        half* O_b = O_base + (int64_t)b * seq_q * q_row_stride;

        for (int g = 0; g < n_kv; g++) {
            // ---- Quantize K for this KV head ----
            const half* K_head = K_b + g * hd;

            IMP_CUDA_CHECK_LOG(cudaMemsetAsync(s_ws.k_sf, 0, k_sf_bytes, stream));
            quantize_fp16_mxfp4_strided_kernel<<<k_blocks, 256, 0, stream>>>(
                K_head, kv_row_stride, static_cast<uint8_t*>(s_ws.k_packed), static_cast<uint8_t*>(s_ws.k_sf),
                seq_kv, hd, n_k_tiles);
            IMP_CUDA_CHECK_LAUNCH();

            // ---- Process each Q head in this GQA group ----
            for (int h_local = 0; h_local < gqa_ratio; h_local++) {
                int h = g * gqa_ratio + h_local;

                // Quantize Q for this head
                const half* Q_head = Q_b + h * hd;
                IMP_CUDA_CHECK_LOG(cudaMemsetAsync(s_ws.q_sf, 0, q_sf_bytes, stream));
                quantize_fp16_mxfp4_strided_kernel<<<q_blocks, 256, 0, stream>>>(
                    Q_head, q_row_stride, static_cast<uint8_t*>(s_ws.q_packed),
                    static_cast<uint8_t*>(s_ws.q_sf), seq_q, hd, n_k_tiles);
                IMP_CUDA_CHECK_LAUNCH();

                // MXFP4 GEMM: S[seq_q, seq_kv] = Q_mxfp4 @ K_mxfp4^T
                bool ok = gemm_mxfp4_cutlass_sm120(s_ws.q_packed, s_ws.q_sf, k_weight, S, seq_q, seq_kv, hd,
                                                   s_ws.gemm_ws, s_ws.gemm_ws_size, stream);
                if (!ok) {
                    IMP_LOG_WARN("MXFP4 attn GEMM failed (b=%d h=%d)", b, h);
                    return false;
                }

                // Scale + softcap + causal mask + softmax
                mxfp4_attn_softmax_kernel<<<seq_q, sm_threads, 0, stream>>>(S, seq_q, seq_kv, /*q_offset=*/0,
                                                                            scale, softcap, causal);
                IMP_CUDA_CHECK_LAUNCH();

                // P·V via cuBLAS: O[seq_q, hd] = S[seq_q, seq_kv] @ V[seq_kv, hd]
                //
                // cuBLAS column-major view (D = A @ B):
                //   A = V^T : [hd, seq_kv],  ld = kv_row_stride
                //   B = S^T : [seq_kv, seq_q], ld = seq_kv
                //   D = O^T : [hd, seq_q],  ld = q_row_stride
                const half* V_head = V_b + g * hd;
                half* O_head = O_b + h * hd;

                cublasGemmEx(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                             hd,      // M
                             seq_q,   // N
                             seq_kv,  // K
                             &one, V_head, CUDA_R_16F, kv_row_stride, S, CUDA_R_16F, seq_kv, &zero, O_head,
                             CUDA_R_16F, q_row_stride, CUBLAS_COMPUTE_32F, kGemmAlgo);
            }
        }
    }

    return true;
}

bool attention_mxfp4_available() {
    static int result = -1;
    if (result >= 0)
        return result != 0;

    // [attention] mxfp4 = "auto" enables when supported; default "auto" is OFF
    // for FMHA (the legacy IMP_MXFP4_ATTENTION env was opt-in). Set to "always"
    // to force the MXFP4 prefill path on.
    const std::string& mode = process_diag_attention_mxfp4_mode();
    if (mode != "always") {
        result = 0;
        return false;
    }
    if (!cutlass_sm120_mxfp4_available()) {
        result = 0;
        return false;
    }

    result = 1;
    IMP_LOG_INFO("MXFP4 tensor core attention enabled for prefill");
    return true;
}

}  // namespace imp
