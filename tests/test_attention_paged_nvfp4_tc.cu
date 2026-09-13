#include <gtest/gtest.h>
#include "compute/attention_paged.h"
#include "core/tensor.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <vector>

namespace imp {
namespace {

// Phase 1 BitDecoding port: launch-success test only. Synthetic random-byte NVFP4 input
// drives the existing scalar paged_attention_decode_nvfp4 to NaN even before the TC variant
// runs, so a synthetic numeric-equivalence test would compare NaN-to-NaN.
// Real-input equivalence: bench_nvfp4_qk_tc_vs_scalar.sh (rel err 1.10e-04), a Qwen3-8B e2e
// smoke, and a SASS audit (TC: 24 HMMA/instantiation; scalar: 0 HMMA/346 scalar, unchanged).

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

// -1 block-table entry (StreamingLLM eviction sentinel) turns into a read before the KV
// pool; the FP16 kernel skips it since #963, quantised kernels dereferenced it unguarded
// (#1678). Illegal access is sticky (#1699: one fault took 73 later tests down), so assert
// no CUDA error/NaN in a process that has not already faulted.
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

    // -1 alone reads one block before the pool (still mapped -> quiet garbage, passes either
    // way); a second entry far enough out leaves the mapping so an unguarded negative-block
    // read becomes observable. Both entries stay inside ctx_len.
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
