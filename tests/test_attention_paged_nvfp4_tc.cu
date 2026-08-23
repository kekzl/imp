#include <gtest/gtest.h>
#include "compute/attention_paged.h"
#include "core/tensor.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <vector>

namespace imp {
namespace {

// Phase 1 of the BitDecoding port: launch-success test for the TC variant.
//
// A full numerical-equivalence test against the scalar reference is deferred
// because synthetic NVFP4 input (random-byte K/V + uniform UE4M3 scales) drives
// the existing scalar `paged_attention_decode_nvfp4` to NaN output even before
// our TC variant runs — the test would compare NaN-to-NaN. Verified via the
// `paged_attention_decode_nvfp4` baseline alone (DIAG O: 4096/4096 NaN). The
// scalar kernel works on real model KV (calibrated FP4 magnitudes + scaled
// attention) but synthetic input is not in its working set.
//
// Equivalence on REAL input is covered by:
// 1. Phase-0 microbench (`tools/analysis/bench_nvfp4_qk_tc_vs_scalar.sh`):
//    isolated Q.K dot, max_abs_err 9.15e-05 rel 1.10e-04 vs scalar reference.
// 2. End-to-end smoke (Qwen3-8B Q8_0 + --kv-nvfp4 + kv_cache.bitdecoding_qk=true):
//    both paths produce "The capital of France is Paris" coherent.
// 3. SASS audit: TC kernel emits 24 HMMA per template instantiation; scalar
//    kernel remains 0 HMMA / 346 scalar (default path unchanged).
//
// This test verifies the TC kernel launches without CUDA errors at the typical
// production decode shape — guards against silent build-time/dispatch-time
// regressions during Phase 2+ refactors.

class PagedAttentionNvfp4TCTest : public ::testing::Test {
protected:
    void SetUp() override { cudaStreamCreate(&stream_); }
    void TearDown() override { cudaStreamDestroy(stream_); }
    cudaStream_t stream_ = nullptr;
};

TEST_F(PagedAttentionNvfp4TCTest, LaunchSucceeds_HD128) {
    constexpr int batch = 1;
    constexpr int n_heads = 32;
    constexpr int n_kv_heads = 32;
    constexpr int HEAD_DIM = 128;
    constexpr int seqlen_kv = 64;
    constexpr int block_size = 16;
    constexpr int n_blocks = (seqlen_kv + block_size - 1) / block_size;

    size_t q_bytes = static_cast<size_t>(batch) * n_heads * HEAD_DIM * sizeof(half);
    size_t kv_bytes = static_cast<size_t>(n_blocks) * block_size * n_kv_heads * (HEAD_DIM / 2);
    size_t sc_bytes = static_cast<size_t>(n_blocks) * block_size * n_kv_heads * (HEAD_DIM / 16);

    std::vector<half> h_Q(batch * n_heads * HEAD_DIM, __float2half(0.0f));
    std::vector<uint8_t> h_K(kv_bytes, 0), h_V(kv_bytes, 0);
    std::vector<uint8_t> h_Ks(sc_bytes, 0x20), h_Vs(sc_bytes, 0x20);

    void* d_Q = nullptr;
    void* d_K = nullptr;
    void* d_V = nullptr;
    void* d_Ks = nullptr;
    void* d_Vs = nullptr;
    void* d_O = nullptr;
    int* d_bt = nullptr;
    int* d_cl = nullptr;
    cudaMalloc(&d_Q, q_bytes);
    cudaMalloc(&d_K, kv_bytes);
    cudaMalloc(&d_V, kv_bytes);
    cudaMalloc(&d_Ks, sc_bytes);
    cudaMalloc(&d_Vs, sc_bytes);
    cudaMalloc(&d_O, q_bytes);
    cudaMalloc(&d_bt, n_blocks * sizeof(int));
    cudaMalloc(&d_cl, sizeof(int));

    cudaMemcpy(d_Q, h_Q.data(), q_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K.data(), kv_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V.data(), kv_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Ks, h_Ks.data(), sc_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Vs, h_Vs.data(), sc_bytes, cudaMemcpyHostToDevice);
    std::vector<int> bt(n_blocks);
    for (int i = 0; i < n_blocks; i++) bt[i] = i;
    cudaMemcpy(d_bt, bt.data(), n_blocks * sizeof(int), cudaMemcpyHostToDevice);
    int ctx_len = seqlen_kv;
    cudaMemcpy(d_cl, &ctx_len, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_O, 0, q_bytes);

    int64_t Q_shape[]  = {batch, 1, n_heads, HEAD_DIM};
    int64_t KV_shape[] = {n_blocks, block_size, n_kv_heads, HEAD_DIM / 2};
    Tensor Q_t(d_Q, QType::F16, 4, Q_shape, true);
    Tensor K_t(d_K, QType::FP4_E2M1, 4, KV_shape, true);
    Tensor V_t(d_V, QType::FP4_E2M1, 4, KV_shape, true);
    Tensor O_t(d_O, QType::F16, 4, Q_shape, true);

    float scale = 1.0f / std::sqrt(static_cast<float>(HEAD_DIM));

    paged_attention_decode_nvfp4_tc(Q_t, K_t, V_t, O_t,
                                    static_cast<const uint8_t*>(d_Ks),
                                    static_cast<const uint8_t*>(d_Vs),
                                    d_bt, d_cl, block_size, scale, ctx_len,
                                    /*sliding_window=*/0, /*softcap=*/0.0f, stream_);
    cudaStreamSynchronize(stream_);

    EXPECT_EQ(cudaGetLastError(), cudaSuccess) << "TC kernel launch failed";

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_Ks);
    cudaFree(d_Vs);
    cudaFree(d_O);
    cudaFree(d_bt);
    cudaFree(d_cl);
}

// A -1 in the block table is what StreamingLLM eviction leaves behind, and a
// negative physical block turns into a read BEFORE the KV pool. The FP16 twin
// has skipped those since #963; the quantised kernels dereferenced them
// unguarded (#1678).
//
// The failure this pins is not a wrong number - it is an illegal access, which
// is sticky: one fault takes every later test in the process down with it
// (#1699 was 73 failures from one). So the assertion is "no CUDA error and no
// NaN", and it has to run in a process that has not already faulted.
TEST_F(PagedAttentionNvfp4TCTest, EvictedBlockSentinelIsSkipped) {
    constexpr int batch = 1;
    constexpr int n_heads = 32;
    constexpr int n_kv_heads = 32;
    constexpr int HEAD_DIM = 128;
    constexpr int seqlen_kv = 64;
    constexpr int block_size = 16;
    constexpr int n_blocks = (seqlen_kv + block_size - 1) / block_size;

    size_t q_bytes = static_cast<size_t>(batch) * n_heads * HEAD_DIM * sizeof(half);
    size_t kv_bytes = static_cast<size_t>(n_blocks) * block_size * n_kv_heads * (HEAD_DIM / 2);
    size_t sc_bytes = static_cast<size_t>(n_blocks) * block_size * n_kv_heads * (HEAD_DIM / 16);

    std::vector<half> h_Q(batch * n_heads * HEAD_DIM, __float2half(0.05f));
    std::vector<uint8_t> h_K(kv_bytes, 0x42), h_V(kv_bytes, 0x24);
    std::vector<uint8_t> h_Ks(sc_bytes, 0x20), h_Vs(sc_bytes, 0x20);

    void* d_Q = nullptr;
    void* d_K = nullptr;
    void* d_V = nullptr;
    void* d_Ks = nullptr;
    void* d_Vs = nullptr;
    void* d_O = nullptr;
    int* d_bt = nullptr;
    int* d_cl = nullptr;
    cudaMalloc(&d_Q, q_bytes);
    cudaMalloc(&d_K, kv_bytes);
    cudaMalloc(&d_V, kv_bytes);
    cudaMalloc(&d_Ks, sc_bytes);
    cudaMalloc(&d_Vs, sc_bytes);
    cudaMalloc(&d_O, q_bytes);
    cudaMalloc(&d_bt, n_blocks * sizeof(int));
    cudaMalloc(&d_cl, sizeof(int));

    cudaMemcpy(d_Q, h_Q.data(), q_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K.data(), kv_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V.data(), kv_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Ks, h_Ks.data(), sc_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Vs, h_Vs.data(), sc_bytes, cudaMemcpyHostToDevice);

    // Two evicted blocks, both still inside ctx_len - which is the point: a
    // block past the context would never be read at all.
    //
    // -1 is the sentinel the eviction path actually writes. It alone does not
    // make a good test: the read it produces lands one block BEFORE the pool,
    // which is still mapped, so the kernel returns quiet garbage rather than
    // faulting and the assertions below pass either way (measured: this test
    // was green against the unguarded kernel when -1 was the only entry). The
    // second entry is far enough out to leave the mapping, so "no negative
    // block is dereferenced" becomes observable. One guard covers both.
    std::vector<int> bt(n_blocks);
    for (int i = 0; i < n_blocks; i++)
        bt[i] = i;
    ASSERT_GE(n_blocks, 3);
    bt[1] = -1;
    bt[2] = -(1 << 20);
    cudaMemcpy(d_bt, bt.data(), n_blocks * sizeof(int), cudaMemcpyHostToDevice);
    int ctx_len = seqlen_kv;
    cudaMemcpy(d_cl, &ctx_len, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_O, 0, q_bytes);

    int64_t Q_shape[] = {batch, 1, n_heads, HEAD_DIM};
    int64_t KV_shape[] = {n_blocks, block_size, n_kv_heads, HEAD_DIM / 2};
    Tensor Q_t(d_Q, QType::F16, 4, Q_shape, true);
    Tensor K_t(d_K, QType::FP4_E2M1, 4, KV_shape, true);
    Tensor V_t(d_V, QType::FP4_E2M1, 4, KV_shape, true);
    Tensor O_t(d_O, QType::F16, 4, Q_shape, true);

    float scale = 1.0f / std::sqrt(static_cast<float>(HEAD_DIM));

    paged_attention_decode_nvfp4_tc(Q_t, K_t, V_t, O_t, static_cast<const uint8_t*>(d_Ks),
                                    static_cast<const uint8_t*>(d_Vs), d_bt, d_cl, block_size, scale, ctx_len,
                                    /*sliding_window=*/0, /*softcap=*/0.0f, stream_);
    cudaStreamSynchronize(stream_);
    EXPECT_EQ(cudaGetLastError(), cudaSuccess) << "a -1 block table entry faulted the kernel";

    std::vector<half> h_O(batch * n_heads * HEAD_DIM);
    cudaMemcpy(h_O.data(), d_O, q_bytes, cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < h_O.size(); i++) {
        const float v = __half2float(h_O[i]);
        ASSERT_TRUE(std::isfinite(v)) << "output element " << i << " is not finite";
    }

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_Ks);
    cudaFree(d_Vs);
    cudaFree(d_O);
    cudaFree(d_bt);
    cudaFree(d_cl);
}

}  // namespace
}  // namespace imp
