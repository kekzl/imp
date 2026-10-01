// Sparse prefill (attention.sparse_prefill_topk_tokens): one page selection per continuation
// chunk, compacted past gather, FA2 with the selected past as its causal offset. The same calls
// the chunked prefill path makes (executor_attention_prefill.cpp). Kernels: sparse_attn_select.cu.

#include <gtest/gtest.h>
#include "compute/attention_fmha_sm120.h"
#include "compute/kv_gather.h"
#include "core/tensor.h"
#include "exec/sparse_attn_select.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace imp {
namespace {

constexpr int kBS = 16;
constexpr int kNh = 8, kNkv = 2, kHd = 128;
constexpr int kRow = kNkv * kHd;
constexpr int kPast = 21 * kBS - 11;  // 21 blocks, partial tail of 5
constexpr int kNBlocks = (kPast + kBS - 1) / kBS;
constexpr int kN = 64;  // chunk rows
constexpr int kCap = 64;
constexpr int kNeedleBlock = 10;

float val(int a, int b) { return static_cast<float>(((a * 131 + b * 17) % 255) - 127) / 1024.0f; }

template <typename T>
T* dalloc(size_t n) {
    T* p = nullptr;
    EXPECT_EQ(cudaMalloc(&p, n * sizeof(T)), cudaSuccess);
    EXPECT_EQ(cudaMemset(p, 0, n * sizeof(T)), cudaSuccess);
    return p;
}

class SparsePrefillTest : public ::testing::Test {
protected:
    void SetUp() override {
        int dev = 0;
        if (cudaGetDeviceCount(&dev) != cudaSuccess || dev == 0)
            GTEST_SKIP() << "no CUDA device";
        // Physical ids reversed: the table, not the logical order, addresses the pool.
        for (int b = 0; b < kNBlocks; b++)
            bt_h_.push_back(kNBlocks - 1 - b);
        std::vector<half> k(kCap * kBS * kRow, __float2half(0.f)), v(k.size(), __float2half(0.f));
        std::vector<half> qh(static_cast<size_t>(kN) * kNh * kHd);
        for (size_t i = 0; i < qh.size(); i++)
            qh[i] = __float2half(val(static_cast<int>((i / kHd) % kNh), static_cast<int>(i % kHd)));
        for (int p = 0; p < kPast; p++) {
            const size_t off = (static_cast<size_t>(bt_h_[p / kBS]) * kBS + p % kBS) * kRow;
            for (int e = 0; e < kRow; e++) {
                float kv = val(p + 3, e);
                // Needle page: keys aligned with every query head of its kv group, so it ranks first.
                if (p / kBS == kNeedleBlock)
                    kv = 4.0f * __half2float(qh[(e / kHd) * (kNh / kNkv) * kHd + e % kHd]);
                k[off + e] = __float2half(kv);
                v[off + e] = __float2half(val(p + 11, e));
            }
        }
        std::vector<half> kc(static_cast<size_t>(kN) * kRow), vc(kc.size());
        for (size_t i = 0; i < kc.size(); i++) {
            kc[i] = __float2half(val(static_cast<int>(i / kRow) + 5000, static_cast<int>(i % kRow)));
            vc[i] = __float2half(val(static_cast<int>(i / kRow) + 7000, static_cast<int>(i % kRow)));
        }
        k_ = put(k);
        v_ = put(v);
        q_ = put(qh);
        kc_ = put(kc);
        vc_ = put(vc);
        bt_ = put(bt_h_);
        std::vector<int> pos(kPast);
        for (int p = 0; p < kPast; p++)
            pos[p] = p;
        int* d_pos = put(pos);
        mm_ = dalloc<__half2>(static_cast<size_t>(kCap) * kRow);
        sparse_update_key_minmax_all_layers(QType::F16, k_, 0, nullptr, 0, mm_, 0, d_pos, bt_, nullptr, 1,
                                            kNkv, kHd, kBS, kPast, kNBlocks, 1, /*meanstd=*/true, nullptr);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        cudaFree(d_pos);
        scores_ = dalloc<float>(64 * kCap);
        agg_ = dalloc<float>(kCap);
        tbl_ = dalloc<int>(kCap);
        ctx_ = dalloc<int>(2);
        kf_ = dalloc<half>(static_cast<size_t>(kPast + kN) * kRow);
        vf_ = dalloc<half>(static_cast<size_t>(kPast + kN) * kRow);
    }
    void TearDown() override {
        for (void* p : {(void*)k_, (void*)v_, (void*)q_, (void*)kc_, (void*)vc_, (void*)bt_, (void*)mm_,
                        (void*)scores_, (void*)agg_, (void*)tbl_, (void*)ctx_, (void*)kf_, (void*)vf_})
            cudaFree(p);
    }
    template <typename T>
    T* put(const std::vector<T>& h) {
        T* d = dalloc<T>(h.size());
        EXPECT_EQ(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), cudaSuccess);
        return d;
    }
    int select(int budget_blocks, int sink_blocks, int recent_blocks) {
        return sparse_prefill_select_past(q_, kN, 16, mm_, bt_, kPast, kNh, kNkv, kHd, kBS, kCap,
                                          budget_blocks, sink_blocks, recent_blocks, /*meanstd=*/true, 1.0f,
                                          scores_, agg_, tbl_, ctx_, nullptr);
    }
    // Gather `past` tokens through `table`, append the chunk, FA2 with q_offset = past.
    std::vector<uint16_t> attend(const int* table, int past) {
        paged_kv_gather_fp16(kf_, k_, table, past, kBS, kNkv, kHd, nullptr);
        paged_kv_gather_fp16(vf_, v_, table, past, kBS, kNkv, kHd, nullptr);
        EXPECT_EQ(cudaMemcpy(kf_ + static_cast<size_t>(past) * kRow, kc_, sizeof(half) * kN * kRow,
                             cudaMemcpyDeviceToDevice),
                  cudaSuccess);
        EXPECT_EQ(cudaMemcpy(vf_ + static_cast<size_t>(past) * kRow, vc_, sizeof(half) * kN * kRow,
                             cudaMemcpyDeviceToDevice),
                  cudaSuccess);
        half* o = dalloc<half>(static_cast<size_t>(kN) * kNh * kHd);
        int64_t qs[4] = {1, kN, kNh, kHd}, ks[4] = {1, past + kN, kNkv, kHd};
        Tensor Q(q_, QType::F16, 4, qs, true), K(kf_, QType::F16, 4, ks, true),
            V(vf_, QType::F16, 4, ks, true);
        Tensor O(o, QType::F16, 4, qs, true);
        EXPECT_TRUE(fmha_sm120_fa2_prefill(Q, K, V, O, 1.0f / std::sqrt(static_cast<float>(kHd)), true, 0,
                                           0.0f, nullptr, past, /*fp16_qk=*/true));
        EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        std::vector<uint16_t> out(static_cast<size_t>(kN) * kNh * kHd);
        EXPECT_EQ(cudaMemcpy(out.data(), o, out.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost),
                  cudaSuccess);
        cudaFree(o);
        return out;
    }
    std::vector<int> table_host(int n) {
        std::vector<int> t(n);
        EXPECT_EQ(cudaMemcpy(t.data(), tbl_, n * sizeof(int), cudaMemcpyDeviceToHost), cudaSuccess);
        return t;
    }

    std::vector<int> bt_h_;
    half *k_ = nullptr, *v_ = nullptr, *q_ = nullptr, *kc_ = nullptr, *vc_ = nullptr, *kf_ = nullptr,
         *vf_ = nullptr;
    int *bt_ = nullptr, *tbl_ = nullptr, *ctx_ = nullptr;
    __half2* mm_ = nullptr;
    float *scores_ = nullptr, *agg_ = nullptr;
};

// Budget >= past blocks: the compacted table is the dense table, output bit-identical; dropping
// one selected page (the mutation) must change it.
TEST_F(SparsePrefillTest, BudgetCoveringThePastIsBitIdenticalAndADroppedPageIsNot) {
    const auto dense = attend(bt_, kPast);
    const int past = select(kNBlocks, 1, 1);
    ASSERT_EQ(past, kPast);
    ASSERT_EQ(table_host(kNBlocks), bt_h_);
    const auto sparse = attend(tbl_, past);
    for (size_t i = 0; i < dense.size(); i++)
        ASSERT_EQ(dense[i], sparse[i]) << "bit mismatch at " << i;

    std::vector<int> dropped = table_host(kNBlocks);
    dropped.erase(dropped.begin() + kNeedleBlock);
    ASSERT_EQ(cudaMemcpy(tbl_, dropped.data(), dropped.size() * sizeof(int), cudaMemcpyHostToDevice),
              cudaSuccess);
    const auto mutated = attend(tbl_, kPast - kBS);
    size_t diff = 0;
    for (size_t i = 0; i < dense.size(); i++)
        diff += dense[i] != mutated[i];
    EXPECT_GT(diff, dense.size() / 2) << "dropping a page must change the chunk's attention";
}

// Budget 8 of 21: sink + 2 recent forced, the needle page selected, ascending logical order, the
// count is (budget - 1) full blocks plus the partial tail.
TEST_F(SparsePrefillTest, BudgetSelectsSinkRecentAndTheNeedleInOrder) {
    const int past = select(8, 1, 2);
    EXPECT_EQ(past, sparse_prefill_past_tokens(kPast, kBS, 8));
    EXPECT_EQ(past, 7 * kBS + (kPast - (kNBlocks - 1) * kBS));
    const auto t = table_host(8);
    std::vector<int> logical;
    for (int phys : t)
        logical.push_back(kNBlocks - 1 - phys);
    EXPECT_TRUE(std::is_sorted(logical.begin(), logical.end()));
    EXPECT_EQ(logical.front(), 0) << "sink";
    EXPECT_EQ(logical[6], kNBlocks - 2) << "recent";
    EXPECT_EQ(logical[7], kNBlocks - 1) << "recent tail";
    EXPECT_NE(std::find(logical.begin(), logical.end(), kNeedleBlock), logical.end()) << "needle page";
    const auto out = attend(tbl_, past);
    for (uint16_t h : out)
        ASSERT_TRUE(std::isfinite(__half2float(__ushort_as_half(h))));
}

TEST(SparsePrefillGeometry, PastTokensKeepsTheTail) {
    EXPECT_EQ(sparse_prefill_past_tokens(100, 16, 7), 100);
    EXPECT_EQ(sparse_prefill_past_tokens(100, 16, 4), 3 * 16 + 4);
    EXPECT_EQ(sparse_prefill_past_tokens(128, 16, 4), 4 * 16);
}

}  // namespace
}  // namespace imp
