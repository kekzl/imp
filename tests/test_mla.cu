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
#include "compute/attention_paged.h"  // paged_attention_decode with v_head_dim
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

// ---------------------------------------------------------------------------
// Test: MLAAttnOutput.VHeadDimWidth
//
// Verifies that paged_attention_decode with v_head_dim != head_dim (MLA asymmetric
// QK/V dims) produces an output of width n_heads * v_head_dim, not n_heads * head_dim,
// and that the V values are correctly accumulated.
//
// Geometry: qk_hd=12, v_hd=8, n_heads=2, n_kv_heads=2, block_size=4, n_ctx=2
// V layout in cache: over-allocated at qk_hd=12 per slot; only first v_hd=8 are valid.
// ---------------------------------------------------------------------------
TEST(MLAAttnOutput, VHeadDimWidth) {
    SKIP_IF_NO_CUDA();

    // Small test geometry
    static constexpr int qk_hd   = 12;   // QK head dim (head_dim in K cache slots)
    static constexpr int v_hd    = 8;    // V head dim (output head dim)
    static constexpr int n_heads = 2;
    static constexpr int n_kv_heads = 2;
    static constexpr int block_size = 4;
    static constexpr int n_ctx   = 2;    // 2 context tokens (1 KV block)
    static constexpr int batch   = 1;
    static constexpr int n_blocks = 1;   // one KV block covers n_ctx tokens

    // K cache: [n_blocks, block_size, n_kv_heads, qk_hd] — slots are qk_hd-wide
    // V cache: [n_blocks, block_size, n_kv_heads, qk_hd] — over-allocated; only v_hd valid
    // This mirrors the over-allocation approach: V slots are qk_hd-sized in the pool.
    const int kv_slot_elems = n_kv_heads * qk_hd;            // elements per KV token slot
    const int kv_block_elems = block_size * kv_slot_elems;   // elements per KV block
    const int cache_elems = n_blocks * kv_block_elems;

    // Allocate K/V caches on GPU
    half* d_k_cache = nullptr;
    half* d_v_cache = nullptr;
    cudaMalloc(&d_k_cache, cache_elems * sizeof(half));
    cudaMalloc(&d_v_cache, cache_elems * sizeof(half));
    cudaMemset(d_k_cache, 0, cache_elems * sizeof(half));
    cudaMemset(d_v_cache, 0, cache_elems * sizeof(half));

    // Fill K: all-ones (12 elements per head per slot)
    // Fill V: pattern — slot 0 head 0: [1,2,...,8, junk, junk, junk]
    //                    slot 0 head 1: [10,20,...,80, junk, junk, junk]
    //                    slot 1 head 0: [2,4,...,16, ...]
    //                    slot 1 head 1: [20,40,...,160, ...]
    // "junk" elements at positions v_hd..qk_hd-1 should NOT appear in output.
    std::vector<half> h_k(cache_elems), h_v(cache_elems, __float2half(0.f));
    for (int i = 0; i < cache_elems; i++)
        h_k[i] = __float2half(1.0f);  // K = 1 everywhere (uniform attention weights)

    // Layout: [block][slot][kv_head][qk_hd]
    // For block=0, slot t, kv_head h, dim d:
    //   index = t * kv_slot_elems + h * qk_hd + d
    for (int t = 0; t < n_ctx; t++) {
        for (int kh = 0; kh < n_kv_heads; kh++) {
            float head_scale = (kh == 0) ? 1.0f : 10.0f;  // head 1 values are 10× bigger
            for (int d = 0; d < v_hd; d++) {
                // Valid V values: distinct per (t, kh, d)
                float val = head_scale * (float)(d + 1) * (float)(t + 1);
                h_v[t * kv_slot_elems + kh * qk_hd + d] = __float2half(val);
            }
            // Junk values at d = v_hd..qk_hd-1 (should be ignored by kernel)
            for (int d = v_hd; d < qk_hd; d++) {
                h_v[t * kv_slot_elems + kh * qk_hd + d] = __float2half(9999.0f);
            }
        }
    }
    cudaMemcpy(d_k_cache, h_k.data(), cache_elems * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v_cache, h_v.data(), cache_elems * sizeof(half), cudaMemcpyHostToDevice);

    // Q: [batch=1, 1, n_heads, qk_hd] — uniform query (all 1/qk_hd for unit dot)
    const int q_elems = batch * n_heads * qk_hd;
    std::vector<half> h_q(q_elems);
    for (int i = 0; i < q_elems; i++)
        h_q[i] = __float2half(1.0f);  // Q all-ones, scale=1/sqrt(qk_hd) applied below
    half* d_q = nullptr;
    cudaMalloc(&d_q, q_elems * sizeof(half));
    cudaMemcpy(d_q, h_q.data(), q_elems * sizeof(half), cudaMemcpyHostToDevice);

    // O: [batch=1, 1, n_heads, v_hd] — the key assertion: width is v_hd, not qk_hd
    const int o_elems = batch * n_heads * v_hd;
    half* d_o = nullptr;
    cudaMalloc(&d_o, o_elems * sizeof(half));
    cudaMemset(d_o, 0, o_elems * sizeof(half));

    // Block tables: block 0 for sequence 0
    int h_bt[1] = {0};
    int* d_bt = nullptr;
    cudaMalloc(&d_bt, sizeof(int));
    cudaMemcpy(d_bt, h_bt, sizeof(int), cudaMemcpyHostToDevice);

    // Context lens: n_ctx = 2
    int h_cl[1] = {n_ctx};
    int* d_cl = nullptr;
    cudaMalloc(&d_cl, sizeof(int));
    cudaMemcpy(d_cl, h_cl, sizeof(int), cudaMemcpyHostToDevice);

    // Build Tensor wrappers
    // Q: [1, 1, n_heads, qk_hd]
    int64_t q_shape[4] = {batch, 1, n_heads, qk_hd};
    Tensor Q_t(d_q, QType::F16, 4, q_shape, true);

    // K_cache / V_cache: [n_blocks, block_size, n_kv_heads, qk_hd]
    int64_t kv_cache_shape[4] = {n_blocks, block_size, n_kv_heads, qk_hd};
    Tensor K_c(d_k_cache, QType::F16, 4, kv_cache_shape, true);
    Tensor V_c(d_v_cache, QType::F16, 4, kv_cache_shape, true);

    // O: [1, 1, n_heads, v_hd]
    int64_t o_shape[4] = {batch, 1, n_heads, v_hd};
    Tensor O_t(d_o, QType::F16, 4, o_shape, true);

    // Scale: 1/sqrt(qk_hd) makes each Q.K dot = qk_hd * 1 * (1/sqrt(qk_hd)) = sqrt(qk_hd)
    float scale = 1.0f / sqrtf((float)qk_hd);

    // Call paged_attention_decode with v_head_dim=v_hd
    paged_attention_decode(Q_t, K_c, V_c, O_t,
                           d_bt, d_cl,
                           block_size, scale,
                           /*max_context_len=*/n_ctx,
                           /*sliding_window=*/0, /*softcap=*/0.0f,
                           /*stream=*/nullptr,
                           /*max_blocks_per_seq=*/1,
                           /*n_sinks=*/0,
                           /*attn_sinks=*/nullptr,
                           /*v_head_dim=*/v_hd);
    cudaDeviceSynchronize();

    // Read output — should be [batch=1, n_heads=2, v_hd=8] = 16 floats
    std::vector<half> h_o(o_elems);
    cudaMemcpy(h_o.data(), d_o, o_elems * sizeof(half), cudaMemcpyDeviceToHost);

    // Assert output size: n_heads * v_hd = 16 (not 24 = n_heads * qk_hd)
    ASSERT_EQ(o_elems, n_heads * v_hd)
        << "Output width must be n_heads * v_hd (" << n_heads * v_hd
        << "), got " << o_elems << " — expected MLA asymmetric dim";

    // Compute expected output.
    // With K=1, Q=1, scale=1/sqrt(qk_hd), all tokens get uniform softmax weight 1/n_ctx.
    // output[head h, dim d] = (1/n_ctx) * sum_t(V[t, h, d])
    //   head 0: val = (t+1)*(d+1), sum over t=0..1 = (d+1)*3, /2 = (d+1)*1.5
    //   head 1: val = 10*(t+1)*(d+1), sum = 10*(d+1)*3, /2 = (d+1)*15
    for (int h = 0; h < n_heads; h++) {
        float head_scale = (h == 0) ? 1.0f : 10.0f;
        for (int d = 0; d < v_hd; d++) {
            float sum_v = 0.0f;
            for (int t = 0; t < n_ctx; t++)
                sum_v += head_scale * (float)(d + 1) * (float)(t + 1);
            float expected = sum_v / (float)n_ctx;
            float got = __half2float(h_o[h * v_hd + d]);
            EXPECT_NEAR(got, expected, 0.1f)
                << "head=" << h << " dim=" << d
                << ": expected " << expected << " got " << got;
        }
    }

    // Also assert that no junk values (9999) leaked into the output
    for (int i = 0; i < o_elems; i++) {
        float val = __half2float(h_o[i]);
        EXPECT_LT(fabsf(val), 200.0f)
            << "junk value leaked at output index " << i << ": " << val;
    }

    cudaFree(d_k_cache);
    cudaFree(d_v_cache);
    cudaFree(d_q);
    cudaFree(d_o);
    cudaFree(d_bt);
    cudaFree(d_cl);
}

// ---------------------------------------------------------------------------
// Test: MLAAttnOutput.PrefillCompaction
//
// Verifies the prefill V-output compaction path: the prefill attention kernels
// accumulate V at head_dim (V is zero-padded to head_dim), producing an output
// of [n, n_heads, head_dim] with real values in the first v_head_dim dims and
// zeros in the tail. mla_compact_attn_output must compact this to
// [n, n_heads, v_head_dim] correctly (per-head, not a naive contiguous slice).
// ---------------------------------------------------------------------------
TEST(MLAAttnOutput, PrefillCompaction) {
    SKIP_IF_NO_CUDA();

    static constexpr int n_tokens = 3;
    static constexpr int n_heads  = 4;
    static constexpr int hd       = 12;  // QK head dim (padded V width)
    static constexpr int v_hd     = 8;   // real V head dim

    // Build hd-strided source: head h, dim d -> value (h*100 + d) for d<v_hd,
    // 7777 (junk that must be dropped) for d in [v_hd, hd).
    const int src_elems = n_tokens * n_heads * hd;
    const int dst_elems = n_tokens * n_heads * v_hd;
    // Values kept < 2048 so FP16 represents each integer exactly (the kernel is
    // a pure copy — any rounding here would be a test artifact, not a kernel bug).
    auto src_val = [](int t, int h, int d) { return (float)(t * 200 + h * 40 + d); };
    std::vector<float> h_src(src_elems);
    for (int t = 0; t < n_tokens; t++)
        for (int h = 0; h < n_heads; h++)
            for (int d = 0; d < hd; d++) {
                int idx = (t * n_heads + h) * hd + d;
                h_src[idx] = (d < v_hd) ? src_val(t, h, d) : 7777.0f;
            }

    Tensor g_src = make_gpu_fp16(h_src, {n_tokens, n_heads, hd});
    Tensor g_dst = alloc_gpu_fp16({n_tokens, n_heads, v_hd});

    mla_compact_attn_output(static_cast<const half*>(g_src.data),
                            static_cast<half*>(g_dst.data),
                            n_tokens, n_heads, hd, v_hd, nullptr);
    cudaDeviceSynchronize();

    auto out = read_gpu_fp16(g_dst);
    ASSERT_EQ((int)out.size(), dst_elems);

    // Every compacted element must equal its source first-v_hd value; no 7777.
    for (int t = 0; t < n_tokens; t++)
        for (int h = 0; h < n_heads; h++)
            for (int d = 0; d < v_hd; d++) {
                int idx = (t * n_heads + h) * v_hd + d;
                float expected = src_val(t, h, d);
                EXPECT_NEAR(out[idx], expected, 0.5f)
                    << "compaction mismatch at t=" << t << " h=" << h << " d=" << d;
            }
    for (int i = 0; i < dst_elems; i++)
        EXPECT_LT(out[i], 7000.0f) << "junk leaked into compacted output at " << i;

    free_gpu(g_src);
    free_gpu(g_dst);
}

// ---------------------------------------------------------------------------
// Test: MLAAttnOutput.PaddedVAssembleZeroesTail
//
// Verifies mla_assemble_kv with v_dst_head_dim > v_head_dim writes the real V
// values into the first v_head_dim dims of each hd-wide head slot and zeroes the
// tail. This is the over-allocation that lets prefill kernels accumulate V at hd.
// ---------------------------------------------------------------------------
TEST(MLAAttnOutput, PaddedVAssembleZeroesTail) {
    SKIP_IF_NO_CUDA();

    static constexpr int n        = 2;
    static constexpr int n_heads  = 2;
    static constexpr int nope     = 4;
    static constexpr int v_hd     = 3;
    static constexpr int rope     = 2;
    static constexpr int hd       = nope + rope;  // 6 — padded V width
    static constexpr int kvb_out  = n_heads * (nope + v_hd);

    std::vector<float> h_kvb(n * kvb_out), h_rope(n * rope);
    for (int i = 0; i < n * kvb_out; i++) h_kvb[i]  = (float)(i + 1);
    for (int i = 0; i < n * rope; i++)    h_rope[i] = (float)(i + 1) * 0.5f;

    Tensor g_kvb  = make_gpu_fp16(h_kvb,  {n, kvb_out});
    Tensor g_rope = make_gpu_fp16(h_rope, {n, rope});
    Tensor g_K    = alloc_gpu_fp16({n, n_heads, hd});
    Tensor g_V    = alloc_gpu_fp16({n, n_heads, hd});  // padded to hd

    mla_assemble_kv(static_cast<const half*>(g_kvb.data),
                    static_cast<const half*>(g_rope.data),
                    static_cast<half*>(g_K.data),
                    static_cast<half*>(g_V.data),
                    n, n_heads, nope, v_hd, rope, nullptr, /*v_dst_head_dim=*/hd);
    cudaDeviceSynchronize();

    auto V = read_gpu_fp16(g_V);
    // Per head: first v_hd dims = kv_b[nope..nope+v_hd), tail [v_hd, hd) = 0.
    for (int t = 0; t < n; t++)
        for (int h = 0; h < n_heads; h++) {
            const float* kv_b_h = h_kvb.data() + t * kvb_out + h * (nope + v_hd);
            for (int d = 0; d < hd; d++) {
                float got = V[(t * n_heads + h) * hd + d];
                float expected = (d < v_hd) ? kv_b_h[nope + d] : 0.0f;
                EXPECT_NEAR(got, expected, 1e-2f)
                    << "padded V mismatch t=" << t << " h=" << h << " d=" << d;
            }
        }

    free_gpu(g_kvb); free_gpu(g_rope); free_gpu(g_K); free_gpu(g_V);
}

}  // namespace
}  // namespace imp
