// KVCache::copy_blocks_device: whole-block D2D copy across all layers (+scale regions),
// used by multi-candidate spec verify (speculative.token_recycling route a) - each candidate
// gets a private copy of the committed partial block, the winner's block copied back.

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "memory/kv_cache.h"

#include <vector>

namespace imp {
namespace {

void fill_block(KVCache& c, int block, uint8_t seed) {
    for (int l = 0; l < c.n_layers(); ++l) {
        std::vector<uint8_t> pat(c.block_bytes());
        for (size_t i = 0; i < pat.size(); ++i)
            pat[i] = static_cast<uint8_t>(seed + l + i);
        cudaMemcpy(c.k_ptr(l, block), pat.data(), pat.size(), cudaMemcpyHostToDevice);
        for (auto& b : pat) b ^= 0xA5;
        cudaMemcpy(c.v_ptr(l, block), pat.data(), pat.size(), cudaMemcpyHostToDevice);
        if (c.k_scale_ptr(l, block)) {
            std::vector<uint8_t> sp(c.scale_block_bytes(l), static_cast<uint8_t>(seed ^ l));
            cudaMemcpy(c.k_scale_ptr(l, block), sp.data(), sp.size(), cudaMemcpyHostToDevice);
            for (auto& b : sp) b ^= 0x5A;
            cudaMemcpy(c.v_scale_ptr(l, block), sp.data(), sp.size(), cudaMemcpyHostToDevice);
        }
    }
}

bool blocks_equal(KVCache& c, int a, int b) {
    for (int l = 0; l < c.n_layers(); ++l) {
        std::vector<uint8_t> pa(c.block_bytes()), pb(c.block_bytes());
        cudaMemcpy(pa.data(), c.k_ptr(l, a), pa.size(), cudaMemcpyDeviceToHost);
        cudaMemcpy(pb.data(), c.k_ptr(l, b), pb.size(), cudaMemcpyDeviceToHost);
        if (pa != pb) return false;
        cudaMemcpy(pa.data(), c.v_ptr(l, a), pa.size(), cudaMemcpyDeviceToHost);
        cudaMemcpy(pb.data(), c.v_ptr(l, b), pb.size(), cudaMemcpyDeviceToHost);
        if (pa != pb) return false;
        if (c.k_scale_ptr(l, a)) {
            std::vector<uint8_t> sa(c.scale_block_bytes(l)), sb(c.scale_block_bytes(l));
            cudaMemcpy(sa.data(), c.k_scale_ptr(l, a), sa.size(), cudaMemcpyDeviceToHost);
            cudaMemcpy(sb.data(), c.k_scale_ptr(l, b), sb.size(), cudaMemcpyDeviceToHost);
            if (sa != sb) return false;
            cudaMemcpy(sa.data(), c.v_scale_ptr(l, a), sa.size(), cudaMemcpyDeviceToHost);
            cudaMemcpy(sb.data(), c.v_scale_ptr(l, b), sb.size(), cudaMemcpyDeviceToHost);
            if (sa != sb) return false;
        }
    }
    return true;
}

TEST(KVBlockCopy, FP16SingleCopy) {
    KVCache c(4, /*n_kv_heads=*/2, /*head_dim=*/64, QType::F16, /*max_blocks=*/8);
    fill_block(c, 1, 17);
    fill_block(c, 3, 99);
    const int src = 1, dst = 3;
    c.copy_blocks_device(&src, &dst, 1, /*stream=*/nullptr);
    cudaDeviceSynchronize();
    EXPECT_TRUE(blocks_equal(c, 1, 3));
}

TEST(KVBlockCopy, FP8FanOutFromOneSource) {
    KVCache c(3, 2, 64, QType::FP8_E4M3, 8);
    fill_block(c, 0, 42);
    fill_block(c, 2, 1);
    fill_block(c, 4, 2);
    const int srcs[2] = {0, 0};
    const int dsts[2] = {2, 4};
    c.copy_blocks_device(srcs, dsts, 2, nullptr);
    cudaDeviceSynchronize();
    EXPECT_TRUE(blocks_equal(c, 0, 2));
    EXPECT_TRUE(blocks_equal(c, 0, 4));
}

// Key min/max metadata travels with the block: a transcript-snapshot clone must score like
// its source under sparse attention (default on, #2405).
TEST(KVBlockCopy, KeyMinmaxMetadataCopied) {
    KVCache c(3, 2, 64, QType::FP8_E4M3, 8);
    ASSERT_TRUE(c.enable_key_minmax());
    const size_t mb = KVCache::minmax_block_bytes(2, 64);
    std::vector<uint8_t> pat(mb), got(mb), other(mb, 0x33);
    for (int l = 0; l < c.n_layers(); ++l) {
        for (size_t i = 0; i < mb; ++i)
            pat[i] = static_cast<uint8_t>(7 * l + i);
        cudaMemcpy(c.key_minmax_ptr(l, 1), pat.data(), mb, cudaMemcpyHostToDevice);
        cudaMemcpy(c.key_minmax_ptr(l, 5), other.data(), mb, cudaMemcpyHostToDevice);
        cudaMemcpy(c.key_minmax_ptr(l, 6), other.data(), mb, cudaMemcpyHostToDevice);
    }
    const int src = 1, dst = 5;
    c.copy_blocks_device(&src, &dst, 1, nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    for (int l = 0; l < c.n_layers(); ++l) {
        for (size_t i = 0; i < mb; ++i)
            pat[i] = static_cast<uint8_t>(7 * l + i);
        cudaMemcpy(got.data(), c.key_minmax_ptr(l, 5), mb, cudaMemcpyDeviceToHost);
        EXPECT_EQ(got, pat) << "layer " << l;
        cudaMemcpy(got.data(), c.key_minmax_ptr(l, 6), mb, cudaMemcpyDeviceToHost);
        EXPECT_EQ(got, other) << "untouched block 6, layer " << l;
    }
}

TEST(KVBlockCopy, UntouchedBlockStaysIntact) {
    KVCache c(2, 2, 64, QType::F16, 8);
    fill_block(c, 0, 5);
    fill_block(c, 1, 66);
    fill_block(c, 2, 77);
    const int src = 0, dst = 2;
    c.copy_blocks_device(&src, &dst, 1, nullptr);
    cudaDeviceSynchronize();
    // Block 1 must not be modified by copying 0 -> 2.
    std::vector<uint8_t> p(c.block_bytes()), q(c.block_bytes());
    for (size_t i = 0; i < p.size(); ++i)
        q[i] = static_cast<uint8_t>(66 + 0 + i);  // layer 0 pattern from fill_block
    cudaMemcpy(p.data(), c.k_ptr(0, 1), p.size(), cudaMemcpyDeviceToHost);
    EXPECT_EQ(p, q);
}

}  // namespace
}  // namespace imp
