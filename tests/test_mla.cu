// tests/test_mla.cu — GPU numeric test for MLA materialized KV projection (Task 2.3)
//
// Tests that the MLA two-step projection:
//   kv_a  = norm_out @ kv_a_proj^T          [n, 576]
//   latent = kv_a[:, :kv_lora_rank]         [n, 512]
//   k_rope = kv_a[:, kv_lora_rank:]         [n, 64]  shared across all heads
//   latent = rmsnorm(latent, kv_a_layernorm) [n, 512]
//   kv_b   = latent @ kv_b_proj^T            [n, n_heads*(k_nope + v_dim)]
//   K[h]   = [pe(64) | nope(128)]  with pe = k_rope (replicated), nope = kv_b first 128 dims
//   V[h]   = kv_b second 128 dims
//
// RoPE layout choice: (b) — pe FIRST in each head so existing rope kernel
// (which rotates first rope_dim=64 dims) works unchanged.
// Q also reordered from [nope(128)|pe(64)] to [pe(64)|nope(128)] for consistency.
//
// Registered in test-compute module (CMakeLists.txt).

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "compute/gemm.h"
#include "compute/layernorm.h"
#include "compute/mla_kv_assemble.h"  // production mla_assemble_kv / mla_reorder_q
#include "core/tensor.h"
#include "test_cuda_skip.h"

#include <vector>
#include <cmath>
#include <cstdlib>
#include <random>

namespace imp {
namespace {

// ---------------------------------------------------------------------------
// Test dimensions — use the real DeepSeek-V2-Lite MLA geometry
// ---------------------------------------------------------------------------
static constexpr int kNTokens    = 3;    // small batch
static constexpr int kDModel     = 32;   // shrunk hidden dim (real=2048; use small for fast test)
static constexpr int kNHeads     = 2;    // real=16; use 2 for fast test
static constexpr int kKvLoraRank = 16;   // real=512
static constexpr int kRopeDim    = 8;    // real=64 (qk_rope_head_dim)
static constexpr int kNopeDim    = 16;   // real=128 (qk_nope_head_dim)
static constexpr int kVHeadDim   = 16;   // real=128
static constexpr int kKvAOut     = kKvLoraRank + kRopeDim;      // 24
static constexpr int kKvBOut     = kNHeads * (kNopeDim + kVHeadDim); // 64
static constexpr int kHeadDim    = kNopeDim + kRopeDim;         // 24 (full K head dim)
static constexpr float kEps      = 1e-5f;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static Tensor make_gpu_fp16(const std::vector<float>& h, std::initializer_list<int64_t> shape) {
    Tensor t;
    t.qtype = QType::F16;
    t.ndim = static_cast<int>(shape.size());
    int i = 0;
    for (auto s : shape) t.shape[i++] = s;
    t.compute_strides();
    t.on_device = true;
    cudaMalloc(&t.data, t.nbytes());
    std::vector<half> hh(h.size());
    for (size_t j = 0; j < h.size(); j++) hh[j] = __float2half(h[j]);
    cudaMemcpy(t.data, hh.data(), t.nbytes(), cudaMemcpyHostToDevice);
    return t;
}

static Tensor alloc_gpu_fp16(std::initializer_list<int64_t> shape) {
    Tensor t;
    t.qtype = QType::F16;
    t.ndim = static_cast<int>(shape.size());
    int i = 0;
    for (auto s : shape) t.shape[i++] = s;
    t.compute_strides();
    t.on_device = true;
    cudaMalloc(&t.data, t.nbytes());
    cudaMemset(t.data, 0, t.nbytes());
    return t;
}

static std::vector<float> read_gpu_fp16(const Tensor& t) {
    std::vector<half> hh(t.numel());
    cudaMemcpy(hh.data(), t.data, t.nbytes(), cudaMemcpyDeviceToHost);
    std::vector<float> out(t.numel());
    for (size_t j = 0; j < out.size(); j++) out[j] = __half2float(hh[j]);
    return out;
}

static void free_gpu(Tensor& t) {
    if (t.data) { cudaFree(t.data); t.data = nullptr; }
}

// ---------------------------------------------------------------------------
// CPU reference: RMSNorm (single-precision)
// ---------------------------------------------------------------------------
static void ref_rmsnorm(const float* x, const float* w, float* out, int rows, int cols, float eps) {
    for (int r = 0; r < rows; r++) {
        float ss = 0.f;
        for (int c = 0; c < cols; c++) { float v = x[r*cols+c]; ss += v*v; }
        float inv = 1.f / std::sqrt(ss/cols + eps);
        for (int c = 0; c < cols; c++)
            out[r*cols+c] = x[r*cols+c] * inv * w[c];
    }
}

// ---------------------------------------------------------------------------
// CPU reference: FP16-precision MatMul C = A @ B^T  (all [rows_A,K] @ [rows_B,K])
// Uses float accumulator (matches half-precision GEMM).
// ---------------------------------------------------------------------------
static void ref_gemm_fp16(const std::vector<float>& A, int rowsA,
                           const std::vector<float>& B, int rowsB,
                           int K, std::vector<float>& C) {
    C.assign(rowsA * rowsB, 0.f);
    for (int m = 0; m < rowsA; m++)
        for (int n = 0; n < rowsB; n++) {
            float acc = 0.f;
            for (int k = 0; k < K; k++)
                // Convert to fp16 and back to match gemv rounding
                acc += __half2float(__float2half(A[m*K+k])) *
                       __half2float(__float2half(B[n*K+k]));
            C[m*rowsB+n] = acc;
        }
}

// ---------------------------------------------------------------------------
// CPU reference for the full MLA KV-projection pipeline
//   Input:  norm_out [n, d_model]
//           kv_a_w   [kv_a_out, d_model]  (kv_a_proj weight)
//           kv_a_norm_w [kv_lora_rank]   (kv_a_layernorm weight)
//           kv_b_w   [kv_b_out, kv_lora_rank] (kv_b_proj weight)
//   Output: K [n, n_heads, head_dim]  layout [pe|nope]
//           V [n, n_heads, v_head_dim]
// ---------------------------------------------------------------------------
static void ref_mla_kv(
        const std::vector<float>& norm_out,  // [n, D]
        const std::vector<float>& kv_a_w,    // [kv_a_out, D]
        const std::vector<float>& kv_a_norm_w,
        const std::vector<float>& kv_b_w,    // [kv_b_out, kv_lora_rank]
        int n, int D,
        std::vector<float>& K_ref,           // [n, n_heads, head_dim]
        std::vector<float>& V_ref)           // [n, n_heads, v_head_dim]
{
    const int kva_out = kKvAOut;
    const int kva_out_proj = kKvLoraRank;
    const int rope_dim = kRopeDim;
    const int nope_dim = kNopeDim;
    const int v_head_dim = kVHeadDim;
    const int n_heads = kNHeads;
    const int head_dim = kHeadDim;

    // Step 1: kv_a = norm_out @ kv_a_w^T   [n, kva_out]
    std::vector<float> kv_a(n * kva_out);
    ref_gemm_fp16(norm_out, n, kv_a_w, kva_out, D, kv_a);

    // Step 2: split latent / k_rope
    std::vector<float> latent(n * kva_out_proj);
    std::vector<float> k_rope(n * rope_dim);
    for (int t = 0; t < n; t++) {
        for (int j = 0; j < kva_out_proj; j++)
            latent[t*kva_out_proj+j] = kv_a[t*kva_out+j];
        for (int j = 0; j < rope_dim; j++)
            k_rope[t*rope_dim+j] = kv_a[t*kva_out+kva_out_proj+j];
    }

    // Step 3: latent = rmsnorm(latent, kv_a_norm_w)
    std::vector<float> latent_n(n * kva_out_proj);
    ref_rmsnorm(latent.data(), kv_a_norm_w.data(), latent_n.data(), n, kva_out_proj, kEps);

    // Step 4: kv_b = latent_n @ kv_b_w^T   [n, kv_b_out]
    const int kv_b_out = kKvBOut;
    std::vector<float> kv_b(n * kv_b_out);
    ref_gemm_fp16(latent_n, n, kv_b_w, kv_b_out, kva_out_proj, kv_b);

    // Step 5: assemble K and V
    // kv_b layout: [n_heads * (nope_dim + v_head_dim)] per token
    // K layout: [pe(rope_dim) | nope(nope_dim)] — approach (b), pe first
    K_ref.assign(n * n_heads * head_dim, 0.f);
    V_ref.assign(n * n_heads * v_head_dim, 0.f);
    for (int t = 0; t < n; t++) {
        for (int h = 0; h < n_heads; h++) {
            const float* kv_b_h = kv_b.data() + t*kv_b_out + h*(nope_dim+v_head_dim);
            const float* nope_src = kv_b_h;
            const float* v_src    = kv_b_h + nope_dim;
            const float* rope_src = k_rope.data() + t*rope_dim;

            float* k_dst = K_ref.data() + t*n_heads*head_dim + h*head_dim;
            float* v_dst = V_ref.data() + t*n_heads*v_head_dim + h*v_head_dim;

            // K = [pe | nope]
            for (int j = 0; j < rope_dim; j++)  k_dst[j]          = rope_src[j];  // pe first
            for (int j = 0; j < nope_dim; j++)  k_dst[rope_dim+j] = nope_src[j];
            // V
            for (int j = 0; j < v_head_dim; j++) v_dst[j] = v_src[j];
        }
    }
}

// NOTE: the GPU scatter kernels (mla_assemble_kv / mla_reorder_q) are the
// PRODUCTION implementations from src/compute/mla_kv_assemble.cu, included via
// compute/mla_kv_assemble.h. This test exercises the real kernels so a
// regression in production code is caught — it does NOT define shadow copies.

// ---------------------------------------------------------------------------
// Full MLA projection on GPU using gemm() + rmsnorm() + scatter kernel
// ---------------------------------------------------------------------------
static void run_mla_kv_gpu(
        const Tensor& norm_out,        // [n, D]  FP16
        const Tensor& kv_a_w,          // [kva_out, D]  FP16
        const Tensor& kv_a_norm_w_t,   // [kv_lora_rank]  FP16
        const Tensor& kv_b_w,          // [kv_b_out, kv_lora_rank]  FP16
        int n_tokens,
        Tensor& K_out,                 // [n, n_heads, head_dim]
        Tensor& V_out,                 // [n, n_heads, v_head_dim]
        cudaStream_t stream = nullptr)
{
    const int kva_out      = kKvAOut;
    const int kv_lora_rank = kKvLoraRank;
    const int rope_dim     = kRopeDim;
    const int n_heads      = kNHeads;
    const int nope_dim     = kNopeDim;
    const int v_head_dim   = kVHeadDim;

    // Step 1: kv_a = norm_out @ kv_a_w^T   [n, kva_out]
    int64_t kva_shape[2] = {n_tokens, kva_out};
    Tensor kv_a;
    kv_a.qtype = QType::F16; kv_a.ndim = 2;
    kv_a.shape[0] = n_tokens; kv_a.shape[1] = kva_out;
    kv_a.compute_strides(); kv_a.on_device = true;
    cudaMalloc(&kv_a.data, kv_a.nbytes());

    gemm(norm_out, kv_a_w, kv_a, 1.f, 0.f, stream);

    // Step 2: split latent [n, kv_lora_rank] and k_rope [n, rope_dim]
    //   latent = kv_a[:, :kv_lora_rank]
    //   k_rope = kv_a[:, kv_lora_rank:]
    // We'll point into kv_a buffer directly (contiguous layout).
    // kv_a is [n, kva_out] row-major: latent is first kv_lora_rank cols.
    // Create a strided view of latent: [n, kv_lora_rank] with stride kva_out.
    // Since rmsnorm() reads row-major, we need to extract latent into a compact buffer.
    Tensor latent;
    latent.qtype = QType::F16; latent.ndim = 2;
    latent.shape[0] = n_tokens; latent.shape[1] = kv_lora_rank;
    latent.compute_strides(); latent.on_device = true;
    cudaMalloc(&latent.data, latent.nbytes());

    // Extract latent: 2D memcpy (strided)
    cudaMemcpy2DAsync(latent.data,
                      kv_lora_rank * sizeof(half),           // dst pitch
                      kv_a.data,
                      kva_out * sizeof(half),                 // src pitch
                      kv_lora_rank * sizeof(half),            // width in bytes
                      n_tokens,
                      cudaMemcpyDeviceToDevice, stream);

    // k_rope: point into kv_a at offset kv_lora_rank per row — extract to compact
    Tensor k_rope;
    k_rope.qtype = QType::F16; k_rope.ndim = 2;
    k_rope.shape[0] = n_tokens; k_rope.shape[1] = rope_dim;
    k_rope.compute_strides(); k_rope.on_device = true;
    cudaMalloc(&k_rope.data, k_rope.nbytes());
    cudaMemcpy2DAsync(k_rope.data,
                      rope_dim * sizeof(half),
                      static_cast<const char*>(kv_a.data) + kv_lora_rank * sizeof(half),
                      kva_out * sizeof(half),
                      rope_dim * sizeof(half),
                      n_tokens,
                      cudaMemcpyDeviceToDevice, stream);

    // Step 3: latent = rmsnorm(latent, kv_a_norm_w)
    rmsnorm(latent, kv_a_norm_w_t, latent, kEps, stream);

    // Step 4: kv_b = latent @ kv_b_w^T   [n, kv_b_out]
    const int kv_b_out = kKvBOut;
    Tensor kv_b;
    kv_b.qtype = QType::F16; kv_b.ndim = 2;
    kv_b.shape[0] = n_tokens; kv_b.shape[1] = kv_b_out;
    kv_b.compute_strides(); kv_b.on_device = true;
    cudaMalloc(&kv_b.data, kv_b.nbytes());
    gemm(latent, kv_b_w, kv_b, 1.f, 0.f, stream);

    // Step 5: scatter into K and V
    mla_assemble_kv(
        static_cast<const half*>(kv_b.data),
        static_cast<const half*>(k_rope.data),
        static_cast<half*>(K_out.data),
        static_cast<half*>(V_out.data),
        n_tokens, n_heads, nope_dim, v_head_dim, rope_dim, stream);

    cudaStreamSynchronize(stream);
    cudaFree(kv_a.data);
    cudaFree(latent.data);
    cudaFree(k_rope.data);
    cudaFree(kv_b.data);
}

// ---------------------------------------------------------------------------
// Test: MLAProjection.TwoStepKVMatchesCPUReference
// ---------------------------------------------------------------------------
TEST(MLAProjection, TwoStepKVMatchesCPUReference) {
    SKIP_IF_NO_CUDA();

    const int n = kNTokens;
    const int D = kDModel;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    auto rand_vec = [&](int sz) {
        std::vector<float> v(sz);
        for (auto& x : v) x = dist(rng);
        return v;
    };

    // Random weight matrices
    auto h_norm_out   = rand_vec(n * D);
    auto h_kv_a_w     = rand_vec(kKvAOut * D);
    auto h_kv_a_norm  = rand_vec(kKvLoraRank);
    auto h_kv_b_w     = rand_vec(kKvBOut * kKvLoraRank);

    // CPU reference
    std::vector<float> K_ref, V_ref;
    ref_mla_kv(h_norm_out, h_kv_a_w, h_kv_a_norm, h_kv_b_w, n, D, K_ref, V_ref);

    // Upload to GPU
    Tensor g_norm_out = make_gpu_fp16(h_norm_out, {n, D});
    Tensor g_kv_a_w   = make_gpu_fp16(h_kv_a_w,   {kKvAOut, D});
    // kv_a_norm_w: [kv_lora_rank] — 1D tensor used by rmsnorm
    Tensor g_kv_a_norm;
    {
        g_kv_a_norm.qtype = QType::F16; g_kv_a_norm.ndim = 1;
        g_kv_a_norm.shape[0] = kKvLoraRank;
        g_kv_a_norm.compute_strides(); g_kv_a_norm.on_device = true;
        cudaMalloc(&g_kv_a_norm.data, kKvLoraRank * sizeof(half));
        std::vector<half> hh(kKvLoraRank);
        for (int i = 0; i < kKvLoraRank; i++) hh[i] = __float2half(h_kv_a_norm[i]);
        cudaMemcpy(g_kv_a_norm.data, hh.data(), kKvLoraRank * sizeof(half), cudaMemcpyHostToDevice);
    }
    Tensor g_kv_b_w   = make_gpu_fp16(h_kv_b_w,   {kKvBOut, kKvLoraRank});

    // Output buffers
    Tensor g_K = alloc_gpu_fp16({n, kNHeads, kHeadDim});
    Tensor g_V = alloc_gpu_fp16({n, kNHeads, kVHeadDim});

    run_mla_kv_gpu(g_norm_out, g_kv_a_w, g_kv_a_norm, g_kv_b_w, n, g_K, g_V);

    auto gpu_K = read_gpu_fp16(g_K);
    auto gpu_V = read_gpu_fp16(g_V);

    // Compare K
    ASSERT_EQ((int)gpu_K.size(), n * kNHeads * kHeadDim);
    for (int i = 0; i < (int)gpu_K.size(); i++) {
        EXPECT_NEAR(gpu_K[i], K_ref[i], 1e-2f)
            << "K mismatch at flat index " << i
            << " (token=" << (i/(kNHeads*kHeadDim))
            << " head=" << ((i/kHeadDim) % kNHeads)
            << " dim=" << (i%kHeadDim) << ")";
    }

    // Compare V
    ASSERT_EQ((int)gpu_V.size(), n * kNHeads * kVHeadDim);
    for (int i = 0; i < (int)gpu_V.size(); i++) {
        EXPECT_NEAR(gpu_V[i], V_ref[i], 1e-2f)
            << "V mismatch at flat index " << i;
    }

    // Verify pe (k_rope) is correctly replicated across all heads for token 0
    // K[t=0, h=0, 0..rope_dim) and K[t=0, h=1, 0..rope_dim) must be identical
    if (kNHeads >= 2) {
        for (int j = 0; j < kRopeDim; j++) {
            float k_h0 = gpu_K[0*kNHeads*kHeadDim + 0*kHeadDim + j];
            float k_h1 = gpu_K[0*kNHeads*kHeadDim + 1*kHeadDim + j];
            EXPECT_NEAR(k_h0, k_h1, 1e-4f)
                << "k_rope should be identical across heads at pe dim " << j;
        }
    }

    free_gpu(g_norm_out); free_gpu(g_kv_a_w); free_gpu(g_kv_a_norm);
    free_gpu(g_kv_b_w);  free_gpu(g_K);       free_gpu(g_V);
}

// ---------------------------------------------------------------------------
// Test: MLAProjection.RMSNormAppliedToLatent
//   Isolate: if kv_a_norm_w is all-ones, latent after norm should be
//   unit-length (l2-norm per row = sqrt(kv_lora_rank)).
// ---------------------------------------------------------------------------
TEST(MLAProjection, RMSNormAppliedToLatent) {
    SKIP_IF_NO_CUDA();

    const int n = 2;
    const int rank = kKvLoraRank;

    // Input: constant vector (all 1.0f)
    std::vector<float> h_input(n * rank, 1.f);
    std::vector<float> h_weight(rank, 1.f);  // identity norm weight

    Tensor g_in  = make_gpu_fp16(h_input, {n, rank});
    Tensor g_out = alloc_gpu_fp16({n, rank});

    Tensor g_w;
    g_w.qtype = QType::F16; g_w.ndim = 1; g_w.shape[0] = rank;
    g_w.compute_strides(); g_w.on_device = true;
    cudaMalloc(&g_w.data, rank * sizeof(half));
    std::vector<half> hh(rank, __float2half(1.f));
    cudaMemcpy(g_w.data, hh.data(), rank * sizeof(half), cudaMemcpyHostToDevice);

    rmsnorm(g_in, g_w, g_out, kEps, nullptr);
    cudaDeviceSynchronize();

    auto result = read_gpu_fp16(g_out);

    // For all-ones input, rmsnorm with weight=1 => out[i] = 1/rms = sqrt(rank/(rank*1+eps)) ≈ 1
    const float expected = 1.f / std::sqrt(1.f + kEps);  // rms(1,1,...) * 1 = 1*inv_rms*1
    for (int i = 0; i < n * rank; i++) {
        EXPECT_NEAR(result[i], expected, 2e-2f) << "at index " << i;
    }

    free_gpu(g_in); free_gpu(g_out); free_gpu(g_w);
}

}  // namespace
}  // namespace imp
