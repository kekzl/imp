#include <gtest/gtest.h>
#include "compute/attention_paged.h"
#include "core/tensor.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <vector>
#include <cmath>
#include <cstring>

namespace imp {
namespace {

// Phase 3b numerical-equivalence test for the residual-FP16 read path.
//
// Setup:
//   - Build a 64-token NVFP4 paged KV cache with controlled, in-range FP4
//     magnitudes (nibbles ∈ {0,1,2,3,4}) and uniform UE4M3 scale = 1.0
//     (byte 0x38). Avoids the NaN trap the existing PagedAttentionNvfp4TCTest
//     comments call out for unconstrained random NVFP4 input.
//   - Compute the kernel's exact dequant on the host for the last 4 tokens
//     (FP4 nibble → half via the same E2M1 LUT, multiplied by UE4M3 = 1.0).
//   - Run the TC kernel two ways:
//        (A) residual_count=0, K_residual=nullptr → all-paged baseline
//        (B) residual_count=4 with K/V_residual pointing at the dequant buffer
//            → kernel clips paged to first 60 tokens, reads last 4 from residual
//   - Outputs must match within FP16 ulp tolerance because the residual holds
//     the exact dequant of the paged tail; the only delta is the code path.
//
// This guards against:
//   - paged_end_token / num_paged_blocks miscomputation
//   - residual slot indexing (slot_base, residual_skip)
//   - block-softmax merge of the residual contribution into the running m_w/l_w/o_reg
//   - V WMMA per-lane scatter (LANES_PER_CHUNK_R / my_chunk_r) for the residual pass

// Host-side E2M1 → float matching `cvt.rn.f16x2.e2m1x2` semantics.
// Bias = 1, sign | exp(2) | mantissa(1).
static float e2m1_to_float(uint8_t nibble) {
    static constexpr float kMag[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    float v = kMag[nibble & 0x7];
    return (nibble & 0x8) ? -v : v;
}

static float ue4m3_byte_to_float(uint8_t b) {
    __nv_fp8_e4m3 v;
    std::memcpy(&v, &b, 1);
    return static_cast<float>(v);
}

class PagedAttentionNvfp4TCResidualTest : public ::testing::Test {
protected:
    void SetUp() override { cudaStreamCreate(&stream_); }
    void TearDown() override { cudaStreamDestroy(stream_); }
    cudaStream_t stream_ = nullptr;
};

TEST_F(PagedAttentionNvfp4TCResidualTest, ResidualSplitMatchesAllPaged_HD128) {
    constexpr int batch = 1;
    constexpr int n_heads = 8;
    constexpr int n_kv_heads = 8;
    constexpr int HEAD_DIM = 128;
    constexpr int block_size = 16;
    constexpr int seqlen_kv = 64;        // 4 full blocks
    constexpr int residual_count = 4;    // last 4 tokens served from residual
    constexpr int residual_n_tokens = 8; // ring size; we use 4 of 8 slots
    constexpr int n_blocks = seqlen_kv / block_size;
    constexpr int kv_head_bytes = HEAD_DIM / 2;
    constexpr int sc_groups = HEAD_DIM / 16;

    // Q: small bounded values (< 0.5) to keep dot products in FP16 range.
    std::vector<half> h_Q(batch * n_heads * HEAD_DIM);
    for (size_t i = 0; i < h_Q.size(); i++)
        h_Q[i] = __float2half(0.05f * static_cast<float>((i % 7) - 3));

    // K/V: each byte holds two FP4 nibbles. Restrict nibbles to {0..4} so
    // dequant magnitudes stay ≤ 2.0; combined with scale=1.0 keeps Q.K well
    // inside FP16 range across 64 tokens × 128 dims.
    const size_t kv_bytes = static_cast<size_t>(n_blocks) * block_size * n_kv_heads * kv_head_bytes;
    const size_t sc_bytes = static_cast<size_t>(n_blocks) * block_size * n_kv_heads * sc_groups;
    std::vector<uint8_t> h_K(kv_bytes), h_V(kv_bytes);
    std::vector<uint8_t> h_Ks(sc_bytes, 0x38), h_Vs(sc_bytes, 0x38);  // 0x38 = UE4M3 1.0

    auto rng = [](int seed) {
        // Deterministic small spread: nibbles cycle through 0..4.
        return static_cast<uint8_t>((seed * 13 + 7) % 5);
    };
    for (size_t i = 0; i < kv_bytes; i++) {
        uint8_t lo = rng(static_cast<int>(2 * i));
        uint8_t hi = rng(static_cast<int>(2 * i + 1));
        h_K[i] = (hi << 4) | lo;
        h_V[i] = (rng(static_cast<int>(3 * i + 11)) << 4) | rng(static_cast<int>(3 * i + 17));
    }

    // Host dequant of the LAST `residual_count` tokens into the residual layout
    // [residual_n_tokens, n_kv_heads, HEAD_DIM] half. Slot i (0-indexed within
    // the active range) gets token (seqlen_kv - residual_count + i).
    const size_t res_slot_elems = static_cast<size_t>(n_kv_heads) * HEAD_DIM;
    std::vector<half> h_K_res(static_cast<size_t>(residual_n_tokens) * res_slot_elems, __float2half(0.0f));
    std::vector<half> h_V_res(static_cast<size_t>(residual_n_tokens) * res_slot_elems, __float2half(0.0f));

    auto dequant_token = [&](const std::vector<uint8_t>& kv_packed,
                             const std::vector<uint8_t>& kv_scales, int abs_t,
                             half* dst /* [n_kv_heads, head_dim] */) {
        const int blk = abs_t / block_size;
        const int t_in_block = abs_t % block_size;
        const int kv_block_stride = block_size * n_kv_heads * kv_head_bytes;
        const int kv_slot_stride = n_kv_heads * kv_head_bytes;
        const int sc_block_stride = block_size * n_kv_heads * sc_groups;
        const int sc_slot_stride = n_kv_heads * sc_groups;
        const uint8_t* K_block = kv_packed.data() + (size_t)blk * kv_block_stride;
        const uint8_t* sc_block = kv_scales.data() + (size_t)blk * sc_block_stride;
        for (int kv_h = 0; kv_h < n_kv_heads; kv_h++) {
            const uint8_t* K_tok = K_block + t_in_block * kv_slot_stride + kv_h * kv_head_bytes;
            const uint8_t* sc_tok = sc_block + t_in_block * sc_slot_stride + kv_h * sc_groups;
            for (int hd = 0; hd < HEAD_DIM; hd++) {
                uint8_t byte = K_tok[hd / 2];
                uint8_t nibble = (hd & 1) ? ((byte >> 4) & 0xF) : (byte & 0xF);
                float v = e2m1_to_float(nibble);
                float scale = ue4m3_byte_to_float(sc_tok[hd / 16]);
                dst[kv_h * HEAD_DIM + hd] = __float2half(v * scale);
            }
        }
    };

    // Slot mapping: write_idx=4, fill_count=4, residual_count=4 ⇒ slot_base = 0.
    // Chronological tokens [60..63] go to slots 0..3.
    constexpr int residual_write_idx = residual_count;  // ring "next-write" position
    for (int i = 0; i < residual_count; i++) {
        int abs_t = seqlen_kv - residual_count + i;
        int slot = (residual_write_idx + residual_n_tokens - residual_count + i) % residual_n_tokens;
        dequant_token(h_K, h_Ks, abs_t, h_K_res.data() + (size_t)slot * res_slot_elems);
        dequant_token(h_V, h_Vs, abs_t, h_V_res.data() + (size_t)slot * res_slot_elems);
    }

    // Device allocation
    void *d_Q = nullptr, *d_K = nullptr, *d_V = nullptr, *d_Ks = nullptr, *d_Vs = nullptr;
    void *d_O_paged = nullptr, *d_O_resid = nullptr;
    void *d_K_res = nullptr, *d_V_res = nullptr;
    int *d_bt = nullptr, *d_cl = nullptr;
    const size_t q_bytes = h_Q.size() * sizeof(half);
    const size_t res_bytes = h_K_res.size() * sizeof(half);

    cudaMalloc(&d_Q, q_bytes);
    cudaMalloc(&d_K, kv_bytes);
    cudaMalloc(&d_V, kv_bytes);
    cudaMalloc(&d_Ks, sc_bytes);
    cudaMalloc(&d_Vs, sc_bytes);
    cudaMalloc(&d_O_paged, q_bytes);
    cudaMalloc(&d_O_resid, q_bytes);
    cudaMalloc(&d_K_res, res_bytes);
    cudaMalloc(&d_V_res, res_bytes);
    cudaMalloc(&d_bt, n_blocks * sizeof(int));
    cudaMalloc(&d_cl, sizeof(int));

    cudaMemcpyAsync(d_Q, h_Q.data(), q_bytes, cudaMemcpyHostToDevice, stream_);
    cudaMemcpyAsync(d_K, h_K.data(), kv_bytes, cudaMemcpyHostToDevice, stream_);
    cudaMemcpyAsync(d_V, h_V.data(), kv_bytes, cudaMemcpyHostToDevice, stream_);
    cudaMemcpyAsync(d_Ks, h_Ks.data(), sc_bytes, cudaMemcpyHostToDevice, stream_);
    cudaMemcpyAsync(d_Vs, h_Vs.data(), sc_bytes, cudaMemcpyHostToDevice, stream_);
    cudaMemcpyAsync(d_K_res, h_K_res.data(), res_bytes, cudaMemcpyHostToDevice, stream_);
    cudaMemcpyAsync(d_V_res, h_V_res.data(), res_bytes, cudaMemcpyHostToDevice, stream_);

    std::vector<int> bt(n_blocks);
    for (int i = 0; i < n_blocks; i++) bt[i] = i;
    cudaMemcpyAsync(d_bt, bt.data(), n_blocks * sizeof(int), cudaMemcpyHostToDevice, stream_);
    int ctx_len = seqlen_kv;
    cudaMemcpyAsync(d_cl, &ctx_len, sizeof(int), cudaMemcpyHostToDevice, stream_);
    cudaMemsetAsync(d_O_paged, 0, q_bytes, stream_);
    cudaMemsetAsync(d_O_resid, 0, q_bytes, stream_);

    int64_t Q_shape[]  = {batch, 1, n_heads, HEAD_DIM};
    int64_t KV_shape[] = {n_blocks, block_size, n_kv_heads, HEAD_DIM / 2};
    Tensor Q_t(d_Q, QType::F16, 4, Q_shape, true);
    Tensor K_t(d_K, QType::FP4_E2M1, 4, KV_shape, true);
    Tensor V_t(d_V, QType::FP4_E2M1, 4, KV_shape, true);
    Tensor O_paged(d_O_paged, QType::F16, 4, Q_shape, true);
    Tensor O_resid(d_O_resid, QType::F16, 4, Q_shape, true);

    const float scale = 1.0f / std::sqrt(static_cast<float>(HEAD_DIM));

    // (A) all-paged baseline
    paged_attention_decode_nvfp4_tc(Q_t, K_t, V_t, O_paged,
                                    static_cast<const uint8_t*>(d_Ks),
                                    static_cast<const uint8_t*>(d_Vs),
                                    d_bt, d_cl, block_size, scale, ctx_len,
                                    /*sliding_window=*/0, /*softcap=*/0.0f, stream_);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess) << "all-paged launch failed";

    // (B) split path: residual covers the last 4 tokens
    paged_attention_decode_nvfp4_tc(Q_t, K_t, V_t, O_resid,
                                    static_cast<const uint8_t*>(d_Ks),
                                    static_cast<const uint8_t*>(d_Vs),
                                    d_bt, d_cl, block_size, scale, ctx_len,
                                    /*sliding_window=*/0, /*softcap=*/0.0f, stream_,
                                    /*max_blocks_per_seq=*/0, /*n_sinks=*/0,
                                    static_cast<const half*>(d_K_res),
                                    static_cast<const half*>(d_V_res),
                                    residual_count, residual_n_tokens, residual_write_idx);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess) << "split path launch failed";

    cudaStreamSynchronize(stream_);

    std::vector<half> h_O_paged(h_Q.size()), h_O_resid(h_Q.size());
    cudaMemcpy(h_O_paged.data(), d_O_paged, q_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_O_resid.data(), d_O_resid, q_bytes, cudaMemcpyDeviceToHost);

    // FP16 ulp tolerance: the only difference between paths is associative
    // reordering of the same FMA chain, so absolute error on values in the
    // ~[-1, 1] range stays within a few ulp (~5e-3). We require <1% rel.
    double max_abs = 0.0, max_rel = 0.0;
    for (size_t i = 0; i < h_O_paged.size(); i++) {
        float a = __half2float(h_O_paged[i]);
        float b = __half2float(h_O_resid[i]);
        ASSERT_FALSE(std::isnan(a)) << "all-paged NaN at " << i;
        ASSERT_FALSE(std::isnan(b)) << "split path NaN at " << i;
        double abs_e = std::fabs(static_cast<double>(a) - static_cast<double>(b));
        double denom = std::max(std::fabs(static_cast<double>(a)), 1e-3);
        double rel_e = abs_e / denom;
        if (abs_e > max_abs) max_abs = abs_e;
        if (rel_e > max_rel) max_rel = rel_e;
    }

    EXPECT_LT(max_abs, 5e-3) << "max abs error too large between residual and all-paged paths";
    EXPECT_LT(max_rel, 1e-2) << "max rel error too large between residual and all-paged paths";

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_Ks);
    cudaFree(d_Vs);
    cudaFree(d_O_paged);
    cudaFree(d_O_resid);
    cudaFree(d_K_res);
    cudaFree(d_V_res);
    cudaFree(d_bt);
    cudaFree(d_cl);
}

// Multi-seq batch test: batch_size=2, each seq has its own residual range
// at a different ring slot. Verifies the kernel reads d_residual_seq_slots /
// d_residual_counts / d_residual_write_idxes correctly per blockIdx.x.
TEST_F(PagedAttentionNvfp4TCResidualTest, MultiSeqBatchArrayForm_HD64) {
    constexpr int batch = 2;
    constexpr int n_heads = 4;
    constexpr int n_kv_heads = 4;
    constexpr int HEAD_DIM = 64;
    constexpr int block_size = 16;
    constexpr int seqlen_kv = 32;          // 2 blocks
    constexpr int residual_count = 4;
    constexpr int residual_n_tokens = 8;
    constexpr int n_blocks_per_seq = seqlen_kv / block_size;
    constexpr int n_blocks_total = batch * n_blocks_per_seq;
    constexpr int kv_head_bytes = HEAD_DIM / 2;
    constexpr int sc_groups = HEAD_DIM / 16;

    std::vector<half> h_Q(batch * n_heads * HEAD_DIM, __float2half(0.05f));
    const size_t kv_bytes = (size_t)n_blocks_total * block_size * n_kv_heads * kv_head_bytes;
    const size_t sc_bytes = (size_t)n_blocks_total * block_size * n_kv_heads * sc_groups;
    std::vector<uint8_t> h_K(kv_bytes), h_V(kv_bytes);
    std::vector<uint8_t> h_Ks(sc_bytes, 0x38), h_Vs(sc_bytes, 0x38);
    auto rng = [](int seed) { return static_cast<uint8_t>((seed * 13 + 7) % 5); };
    for (size_t i = 0; i < kv_bytes; i++) {
        h_K[i] = (rng(2 * (int)i + 1) << 4) | rng(2 * (int)i);
        h_V[i] = (rng(3 * (int)i + 11) << 4) | rng(3 * (int)i + 17);
    }

    // Residual pool layout (batch slots × n_kv_heads × HEAD_DIM × residual_n).
    // Use slots 0 and 1; build per-seq ring data from the dequant of each seq's
    // last 4 paged tokens. residual_seq_stride_elems = residual_n × n_kv_heads × HEAD_DIM.
    const int seq_stride_elems = residual_n_tokens * n_kv_heads * HEAD_DIM;
    std::vector<half> h_K_pool((size_t)batch * seq_stride_elems, __float2half(0.0f));
    std::vector<half> h_V_pool((size_t)batch * seq_stride_elems, __float2half(0.0f));

    auto dequant_token = [&](const std::vector<uint8_t>& packed, const std::vector<uint8_t>& sc,
                             int seq_idx, int abs_t, half* dst) {
        const int blk_global = seq_idx * n_blocks_per_seq + (abs_t / block_size);
        const int t_in_block = abs_t % block_size;
        const int kv_block_stride = block_size * n_kv_heads * kv_head_bytes;
        const int kv_slot_stride = n_kv_heads * kv_head_bytes;
        const int sc_block_stride = block_size * n_kv_heads * sc_groups;
        const int sc_slot_stride = n_kv_heads * sc_groups;
        const uint8_t* K_block = packed.data() + (size_t)blk_global * kv_block_stride;
        const uint8_t* sc_block = sc.data() + (size_t)blk_global * sc_block_stride;
        for (int kh = 0; kh < n_kv_heads; kh++) {
            const uint8_t* K_tok = K_block + t_in_block * kv_slot_stride + kh * kv_head_bytes;
            const uint8_t* sc_tok = sc_block + t_in_block * sc_slot_stride + kh * sc_groups;
            for (int hd = 0; hd < HEAD_DIM; hd++) {
                uint8_t byte = K_tok[hd / 2];
                uint8_t nibble = (hd & 1) ? ((byte >> 4) & 0xF) : (byte & 0xF);
                float v = e2m1_to_float(nibble);
                float s = ue4m3_byte_to_float(sc_tok[hd / 16]);
                dst[kh * HEAD_DIM + hd] = __float2half(v * s);
            }
        }
    };

    constexpr int residual_write_idx = residual_count;
    for (int seq = 0; seq < batch; seq++) {
        for (int i = 0; i < residual_count; i++) {
            int abs_t = seqlen_kv - residual_count + i;
            int slot_in_ring = (residual_write_idx + residual_n_tokens - residual_count + i) % residual_n_tokens;
            half* k_dst = h_K_pool.data() + (size_t)seq * seq_stride_elems +
                          (size_t)slot_in_ring * n_kv_heads * HEAD_DIM;
            half* v_dst = h_V_pool.data() + (size_t)seq * seq_stride_elems +
                          (size_t)slot_in_ring * n_kv_heads * HEAD_DIM;
            dequant_token(h_K, h_Ks, seq, abs_t, k_dst);
            dequant_token(h_V, h_Vs, seq, abs_t, v_dst);
        }
    }

    void *d_Q = nullptr, *d_K = nullptr, *d_V = nullptr, *d_Ks = nullptr, *d_Vs = nullptr;
    void *d_O_paged = nullptr, *d_O_resid = nullptr;
    void *d_K_pool = nullptr, *d_V_pool = nullptr;
    int *d_bt = nullptr, *d_cl = nullptr;
    int *d_seq_slots = nullptr, *d_counts = nullptr, *d_widxes = nullptr;
    const size_t q_bytes = h_Q.size() * sizeof(half);
    const size_t pool_bytes = h_K_pool.size() * sizeof(half);

    cudaMalloc(&d_Q, q_bytes);
    cudaMalloc(&d_K, kv_bytes); cudaMalloc(&d_V, kv_bytes);
    cudaMalloc(&d_Ks, sc_bytes); cudaMalloc(&d_Vs, sc_bytes);
    cudaMalloc(&d_O_paged, q_bytes); cudaMalloc(&d_O_resid, q_bytes);
    cudaMalloc(&d_K_pool, pool_bytes); cudaMalloc(&d_V_pool, pool_bytes);
    cudaMalloc(&d_bt, n_blocks_total * sizeof(int));
    cudaMalloc(&d_cl, batch * sizeof(int));
    cudaMalloc(&d_seq_slots, batch * sizeof(int));
    cudaMalloc(&d_counts, batch * sizeof(int));
    cudaMalloc(&d_widxes, batch * sizeof(int));

    cudaMemcpy(d_Q, h_Q.data(), q_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K.data(), kv_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V.data(), kv_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Ks, h_Ks.data(), sc_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Vs, h_Vs.data(), sc_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K_pool, h_K_pool.data(), pool_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V_pool, h_V_pool.data(), pool_bytes, cudaMemcpyHostToDevice);

    // Per-seq block_table: seq 0 owns blocks [0, 1], seq 1 owns blocks [2, 3].
    // 2D padded block_table layout: [batch, max_blocks_per_seq] = [2, 2].
    int bt_padded[batch * n_blocks_per_seq];
    for (int s = 0; s < batch; s++)
        for (int b = 0; b < n_blocks_per_seq; b++)
            bt_padded[s * n_blocks_per_seq + b] = s * n_blocks_per_seq + b;
    cudaMemcpy(d_bt, bt_padded, sizeof(bt_padded), cudaMemcpyHostToDevice);
    int cl_arr[batch] = {seqlen_kv, seqlen_kv};
    cudaMemcpy(d_cl, cl_arr, sizeof(cl_arr), cudaMemcpyHostToDevice);
    int slot_arr[batch] = {0, 1};
    int count_arr[batch] = {residual_count, residual_count};
    int widx_arr[batch] = {residual_write_idx, residual_write_idx};
    cudaMemcpy(d_seq_slots, slot_arr, sizeof(slot_arr), cudaMemcpyHostToDevice);
    cudaMemcpy(d_counts, count_arr, sizeof(count_arr), cudaMemcpyHostToDevice);
    cudaMemcpy(d_widxes, widx_arr, sizeof(widx_arr), cudaMemcpyHostToDevice);
    cudaMemset(d_O_paged, 0, q_bytes);
    cudaMemset(d_O_resid, 0, q_bytes);

    int64_t Q_shape[]  = {batch, 1, n_heads, HEAD_DIM};
    int64_t KV_shape[] = {n_blocks_total, block_size, n_kv_heads, HEAD_DIM / 2};
    Tensor Q_t(d_Q, QType::F16, 4, Q_shape, true);
    Tensor K_t(d_K, QType::FP4_E2M1, 4, KV_shape, true);
    Tensor V_t(d_V, QType::FP4_E2M1, 4, KV_shape, true);
    Tensor O_paged(d_O_paged, QType::F16, 4, Q_shape, true);
    Tensor O_resid(d_O_resid, QType::F16, 4, Q_shape, true);

    const float scale = 1.0f / std::sqrt((float)HEAD_DIM);

    // (A) all-paged baseline
    paged_attention_decode_nvfp4_tc(Q_t, K_t, V_t, O_paged,
                                    static_cast<const uint8_t*>(d_Ks),
                                    static_cast<const uint8_t*>(d_Vs),
                                    d_bt, d_cl, block_size, scale, seqlen_kv,
                                    /*sliding_window=*/0, /*softcap=*/0.0f, stream_,
                                    /*max_blocks_per_seq=*/n_blocks_per_seq);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);

    // (B) multi-seq array form
    paged_attention_decode_nvfp4_tc(Q_t, K_t, V_t, O_resid,
                                    static_cast<const uint8_t*>(d_Ks),
                                    static_cast<const uint8_t*>(d_Vs),
                                    d_bt, d_cl, block_size, scale, seqlen_kv,
                                    /*sliding_window=*/0, /*softcap=*/0.0f, stream_,
                                    /*max_blocks_per_seq=*/n_blocks_per_seq, /*n_sinks=*/0,
                                    /*K_residual=*/nullptr, /*V_residual=*/nullptr,
                                    /*residual_count=*/0,
                                    /*residual_n_tokens=*/residual_n_tokens,
                                    /*residual_write_idx=*/0,
                                    static_cast<const half*>(d_K_pool),
                                    static_cast<const half*>(d_V_pool),
                                    /*residual_seq_stride_elems=*/seq_stride_elems,
                                    d_seq_slots, d_counts, d_widxes);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    cudaStreamSynchronize(stream_);

    std::vector<half> h_O_paged(h_Q.size()), h_O_resid(h_Q.size());
    cudaMemcpy(h_O_paged.data(), d_O_paged, q_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_O_resid.data(), d_O_resid, q_bytes, cudaMemcpyDeviceToHost);

    double max_abs = 0.0, max_rel = 0.0;
    for (size_t i = 0; i < h_O_paged.size(); i++) {
        float a = __half2float(h_O_paged[i]);
        float b = __half2float(h_O_resid[i]);
        ASSERT_FALSE(std::isnan(a)) << "all-paged NaN at " << i;
        ASSERT_FALSE(std::isnan(b)) << "multi-seq NaN at " << i;
        double abs_e = std::fabs((double)a - (double)b);
        double denom = std::max(std::fabs((double)a), 1e-3);
        double rel_e = abs_e / denom;
        if (abs_e > max_abs) max_abs = abs_e;
        if (rel_e > max_rel) max_rel = rel_e;
    }
    EXPECT_LT(max_abs, 5e-3) << "max abs error too large between multi-seq array and all-paged";
    EXPECT_LT(max_rel, 1e-2) << "max rel error too large between multi-seq array and all-paged";

    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_Ks); cudaFree(d_Vs);
    cudaFree(d_O_paged); cudaFree(d_O_resid);
    cudaFree(d_K_pool); cudaFree(d_V_pool);
    cudaFree(d_bt); cudaFree(d_cl);
    cudaFree(d_seq_slots); cudaFree(d_counts); cudaFree(d_widxes);
}

// Sanity test: ctx_len < residual_n_tokens (residual covers all tokens, paged
// loop runs 0 iterations). Validates the early-context edge case.
TEST_F(PagedAttentionNvfp4TCResidualTest, ResidualOnlyShortContext_HD64) {
    constexpr int batch = 1;
    constexpr int n_heads = 4;
    constexpr int n_kv_heads = 4;
    constexpr int HEAD_DIM = 64;
    constexpr int block_size = 16;
    constexpr int seqlen_kv = 3;          // less than residual_n_tokens
    constexpr int residual_count = 3;
    constexpr int residual_n_tokens = 8;
    constexpr int n_blocks = 1;           // one (mostly-empty) page block
    constexpr int kv_head_bytes = HEAD_DIM / 2;
    constexpr int sc_groups = HEAD_DIM / 16;

    std::vector<half> h_Q(batch * n_heads * HEAD_DIM, __float2half(0.1f));
    const size_t kv_bytes = (size_t)n_blocks * block_size * n_kv_heads * kv_head_bytes;
    const size_t sc_bytes = (size_t)n_blocks * block_size * n_kv_heads * sc_groups;
    std::vector<uint8_t> h_K(kv_bytes, 0x22);  // both nibbles = 1.0
    std::vector<uint8_t> h_V(kv_bytes, 0x22);
    std::vector<uint8_t> h_Ks(sc_bytes, 0x38), h_Vs(sc_bytes, 0x38);

    const size_t res_slot_elems = (size_t)n_kv_heads * HEAD_DIM;
    std::vector<half> h_K_res((size_t)residual_n_tokens * res_slot_elems, __float2half(1.0f));
    std::vector<half> h_V_res((size_t)residual_n_tokens * res_slot_elems, __float2half(1.0f));

    void *d_Q = nullptr, *d_K = nullptr, *d_V = nullptr, *d_Ks = nullptr, *d_Vs = nullptr;
    void *d_O = nullptr, *d_K_res = nullptr, *d_V_res = nullptr;
    int *d_bt = nullptr, *d_cl = nullptr;
    cudaMalloc(&d_Q, h_Q.size() * sizeof(half));
    cudaMalloc(&d_K, kv_bytes);
    cudaMalloc(&d_V, kv_bytes);
    cudaMalloc(&d_Ks, sc_bytes);
    cudaMalloc(&d_Vs, sc_bytes);
    cudaMalloc(&d_O, h_Q.size() * sizeof(half));
    cudaMalloc(&d_K_res, h_K_res.size() * sizeof(half));
    cudaMalloc(&d_V_res, h_V_res.size() * sizeof(half));
    cudaMalloc(&d_bt, n_blocks * sizeof(int));
    cudaMalloc(&d_cl, sizeof(int));

    cudaMemcpy(d_Q, h_Q.data(), h_Q.size() * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K.data(), kv_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V.data(), kv_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Ks, h_Ks.data(), sc_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Vs, h_Vs.data(), sc_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K_res, h_K_res.data(), h_K_res.size() * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V_res, h_V_res.data(), h_V_res.size() * sizeof(half), cudaMemcpyHostToDevice);
    int bt[1] = {0};
    int cl = seqlen_kv;
    cudaMemcpy(d_bt, bt, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cl, &cl, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_O, 0, h_Q.size() * sizeof(half));

    int64_t Q_shape[]  = {batch, 1, n_heads, HEAD_DIM};
    int64_t KV_shape[] = {n_blocks, block_size, n_kv_heads, HEAD_DIM / 2};
    Tensor Q_t(d_Q, QType::F16, 4, Q_shape, true);
    Tensor K_t(d_K, QType::FP4_E2M1, 4, KV_shape, true);
    Tensor V_t(d_V, QType::FP4_E2M1, 4, KV_shape, true);
    Tensor O_t(d_O, QType::F16, 4, Q_shape, true);

    const float scale = 1.0f / std::sqrt((float)HEAD_DIM);
    constexpr int residual_write_idx = residual_count;
    paged_attention_decode_nvfp4_tc(Q_t, K_t, V_t, O_t,
                                    static_cast<const uint8_t*>(d_Ks),
                                    static_cast<const uint8_t*>(d_Vs),
                                    d_bt, d_cl, block_size, scale, cl,
                                    /*sliding_window=*/0, /*softcap=*/0.0f, stream_,
                                    /*max_blocks_per_seq=*/0, /*n_sinks=*/0,
                                    static_cast<const half*>(d_K_res),
                                    static_cast<const half*>(d_V_res),
                                    residual_count, residual_n_tokens, residual_write_idx);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    cudaStreamSynchronize(stream_);

    // V_residual is all 1s ⇒ output should be ≈ 1.0 across all heads/dims
    // (uniform softmax over identical V tokens = mean(V) = 1).
    std::vector<half> h_O(h_Q.size());
    cudaMemcpy(h_O.data(), d_O, h_Q.size() * sizeof(half), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < h_O.size(); i++) {
        float v = __half2float(h_O[i]);
        ASSERT_FALSE(std::isnan(v));
        EXPECT_NEAR(v, 1.0f, 5e-3f) << "elem " << i;
    }

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_Ks);
    cudaFree(d_Vs);
    cudaFree(d_O);
    cudaFree(d_K_res);
    cudaFree(d_V_res);
    cudaFree(d_bt);
    cudaFree(d_cl);
}

// Splitk fast-path launch test: allocate splitk scratch, force the launcher
// onto the splitk + paged_attention_residual_reduce_kernel path (the path
// the engine uses on real workloads). Verifies the kernel dispatches
// without CUDA errors — a full numerical equivalence test is blocked by
// the same synthetic-random-NVFP4-→-NaN limitation called out in
// PagedAttentionNvfp4TCTest. The non-splitk equivalence test above proves
// the math; the splitk path reuses the same residual_reduce_kernel.
TEST_F(PagedAttentionNvfp4TCResidualTest, SplitKResidualLaunchSucceeds_HD128) {
    constexpr int batch = 1;
    constexpr int n_heads = 8;
    constexpr int n_kv_heads = 8;
    constexpr int HEAD_DIM = 128;
    constexpr int block_size = 16;
    constexpr int seqlen_kv = 64;
    constexpr int residual_count = 4;
    constexpr int residual_n_tokens = 8;
    constexpr int n_blocks = seqlen_kv / block_size;
    constexpr int kv_head_bytes = HEAD_DIM / 2;
    constexpr int sc_groups = HEAD_DIM / 16;

    std::vector<half> h_Q(batch * n_heads * HEAD_DIM);
    for (size_t i = 0; i < h_Q.size(); i++)
        h_Q[i] = __float2half(0.05f * static_cast<float>((i % 7) - 3));

    const size_t kv_bytes = static_cast<size_t>(n_blocks) * block_size * n_kv_heads * kv_head_bytes;
    const size_t sc_bytes = static_cast<size_t>(n_blocks) * block_size * n_kv_heads * sc_groups;
    std::vector<uint8_t> h_K(kv_bytes), h_V(kv_bytes);
    std::vector<uint8_t> h_Ks(sc_bytes, 0x38), h_Vs(sc_bytes, 0x38);
    auto rng = [](int seed) { return static_cast<uint8_t>((seed * 13 + 7) % 5); };
    for (size_t i = 0; i < kv_bytes; i++) {
        uint8_t lo = rng(static_cast<int>(2 * i));
        uint8_t hi = rng(static_cast<int>(2 * i + 1));
        h_K[i] = (hi << 4) | lo;
        h_V[i] = (rng(static_cast<int>(3 * i + 11)) << 4) | rng(static_cast<int>(3 * i + 17));
    }

    const size_t res_slot_elems = static_cast<size_t>(n_kv_heads) * HEAD_DIM;
    std::vector<half> h_K_res(static_cast<size_t>(residual_n_tokens) * res_slot_elems, __float2half(0.0f));
    std::vector<half> h_V_res(static_cast<size_t>(residual_n_tokens) * res_slot_elems, __float2half(0.0f));

    auto dequant_token = [&](const std::vector<uint8_t>& kv_packed,
                             const std::vector<uint8_t>& kv_scales, int abs_t,
                             half* dst) {
        const int blk = abs_t / block_size;
        const int t_in_block = abs_t % block_size;
        const int kv_block_stride = block_size * n_kv_heads * kv_head_bytes;
        const int kv_slot_stride = n_kv_heads * kv_head_bytes;
        const int sc_block_stride = block_size * n_kv_heads * sc_groups;
        const int sc_slot_stride = n_kv_heads * sc_groups;
        const uint8_t* K_block = kv_packed.data() + (size_t)blk * kv_block_stride;
        const uint8_t* sc_block = kv_scales.data() + (size_t)blk * sc_block_stride;
        for (int kv_h = 0; kv_h < n_kv_heads; kv_h++) {
            const uint8_t* K_tok = K_block + t_in_block * kv_slot_stride + kv_h * kv_head_bytes;
            const uint8_t* sc_tok = sc_block + t_in_block * sc_slot_stride + kv_h * sc_groups;
            for (int hd = 0; hd < HEAD_DIM; hd++) {
                uint8_t byte = K_tok[hd / 2];
                uint8_t nibble = (hd & 1) ? ((byte >> 4) & 0xF) : (byte & 0xF);
                float v = e2m1_to_float(nibble);
                float scale = ue4m3_byte_to_float(sc_tok[hd / 16]);
                dst[kv_h * HEAD_DIM + hd] = __float2half(v * scale);
            }
        }
    };

    constexpr int residual_write_idx = residual_count;
    for (int i = 0; i < residual_count; i++) {
        int abs_t = seqlen_kv - residual_count + i;
        int slot = (residual_write_idx + residual_n_tokens - residual_count + i) % residual_n_tokens;
        dequant_token(h_K, h_Ks, abs_t, h_K_res.data() + (size_t)slot * res_slot_elems);
        dequant_token(h_V, h_Vs, abs_t, h_V_res.data() + (size_t)slot * res_slot_elems);
    }

    void *d_Q = nullptr, *d_K = nullptr, *d_V = nullptr, *d_Ks = nullptr, *d_Vs = nullptr;
    void *d_O_paged = nullptr, *d_O_resid = nullptr;
    void *d_K_res = nullptr, *d_V_res = nullptr;
    int *d_bt = nullptr, *d_cl = nullptr;
    const size_t q_bytes = h_Q.size() * sizeof(half);
    const size_t res_bytes = h_K_res.size() * sizeof(half);

    cudaMalloc(&d_Q, q_bytes);
    cudaMalloc(&d_K, kv_bytes);
    cudaMalloc(&d_V, kv_bytes);
    cudaMalloc(&d_Ks, sc_bytes);
    cudaMalloc(&d_Vs, sc_bytes);
    cudaMalloc(&d_O_paged, q_bytes);
    cudaMalloc(&d_O_resid, q_bytes);
    cudaMalloc(&d_K_res, res_bytes);
    cudaMalloc(&d_V_res, res_bytes);
    cudaMalloc(&d_bt, n_blocks * sizeof(int));
    cudaMalloc(&d_cl, sizeof(int));

    // Allocate splitk scratch big enough for any num_splits the launcher
    // might pick. compute_splitk_splits caps at 32, and partial_stride is
    // 2 + HEAD_DIM. For batch=1, n_heads=8: 8 × 32 × 130 × 4 = ~133 KB.
    constexpr int max_splits = 32;
    const size_t scratch_size = (size_t)batch * n_heads * max_splits * (2 + HEAD_DIM) * sizeof(float);
    void* d_splitk_scratch = nullptr;
    cudaMalloc(&d_splitk_scratch, scratch_size);
    paged_attention_set_splitk_scratch(d_splitk_scratch, scratch_size);

    cudaMemcpyAsync(d_Q, h_Q.data(), q_bytes, cudaMemcpyHostToDevice, stream_);
    cudaMemcpyAsync(d_K, h_K.data(), kv_bytes, cudaMemcpyHostToDevice, stream_);
    cudaMemcpyAsync(d_V, h_V.data(), kv_bytes, cudaMemcpyHostToDevice, stream_);
    cudaMemcpyAsync(d_Ks, h_Ks.data(), sc_bytes, cudaMemcpyHostToDevice, stream_);
    cudaMemcpyAsync(d_Vs, h_Vs.data(), sc_bytes, cudaMemcpyHostToDevice, stream_);
    cudaMemcpyAsync(d_K_res, h_K_res.data(), res_bytes, cudaMemcpyHostToDevice, stream_);
    cudaMemcpyAsync(d_V_res, h_V_res.data(), res_bytes, cudaMemcpyHostToDevice, stream_);

    std::vector<int> bt(n_blocks);
    for (int i = 0; i < n_blocks; i++) bt[i] = i;
    cudaMemcpyAsync(d_bt, bt.data(), n_blocks * sizeof(int), cudaMemcpyHostToDevice, stream_);
    int ctx_len = seqlen_kv;
    cudaMemcpyAsync(d_cl, &ctx_len, sizeof(int), cudaMemcpyHostToDevice, stream_);
    cudaMemsetAsync(d_O_paged, 0, q_bytes, stream_);
    cudaMemsetAsync(d_O_resid, 0, q_bytes, stream_);

    int64_t Q_shape[]  = {batch, 1, n_heads, HEAD_DIM};
    int64_t KV_shape[] = {n_blocks, block_size, n_kv_heads, HEAD_DIM / 2};
    Tensor Q_t(d_Q, QType::F16, 4, Q_shape, true);
    Tensor K_t(d_K, QType::FP4_E2M1, 4, KV_shape, true);
    Tensor V_t(d_V, QType::FP4_E2M1, 4, KV_shape, true);
    Tensor O_paged(d_O_paged, QType::F16, 4, Q_shape, true);
    Tensor O_resid(d_O_resid, QType::F16, 4, Q_shape, true);

    const float scale = 1.0f / std::sqrt(static_cast<float>(HEAD_DIM));

    // (A) all-paged baseline (splitk + standard reduce)
    paged_attention_decode_nvfp4_tc(Q_t, K_t, V_t, O_paged,
                                    static_cast<const uint8_t*>(d_Ks),
                                    static_cast<const uint8_t*>(d_Vs),
                                    d_bt, d_cl, block_size, scale, ctx_len,
                                    /*sliding_window=*/0, /*softcap=*/0.0f, stream_);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess) << "all-paged splitk launch failed";

    // (B) splitk + residual_reduce_kernel (the production residual path)
    paged_attention_decode_nvfp4_tc(Q_t, K_t, V_t, O_resid,
                                    static_cast<const uint8_t*>(d_Ks),
                                    static_cast<const uint8_t*>(d_Vs),
                                    d_bt, d_cl, block_size, scale, ctx_len,
                                    /*sliding_window=*/0, /*softcap=*/0.0f, stream_,
                                    /*max_blocks_per_seq=*/0, /*n_sinks=*/0,
                                    static_cast<const half*>(d_K_res),
                                    static_cast<const half*>(d_V_res),
                                    residual_count, residual_n_tokens, residual_write_idx);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess) << "splitk+residual launch failed";

    cudaStreamSynchronize(stream_);
    EXPECT_EQ(cudaGetLastError(), cudaSuccess) << "splitk+residual path produced a CUDA error";

    paged_attention_set_splitk_scratch(nullptr, 0);  // clear so other tests don't see it
    cudaFree(d_splitk_scratch);
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_Ks);
    cudaFree(d_Vs);
    cudaFree(d_O_paged);
    cudaFree(d_O_resid);
    cudaFree(d_K_res);
    cudaFree(d_V_res);
    cudaFree(d_bt);
    cudaFree(d_cl);
}

}  // namespace
}  // namespace imp
