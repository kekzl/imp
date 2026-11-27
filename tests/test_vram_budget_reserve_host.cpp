// Host-only cases of tests/test_vram_budget_reserve.cpp, in test-core (CPU lane): no CUDA call.
#include <gtest/gtest.h>

#include "compute/gemm_cutlass_sm120.h"
#include "exec/moe_nvfp4_cache_rule.h"
#include "memory/vram_query.h"
#include "model/model.h"
#include "runtime/vram_budget.h"
#include "vram_budget_fixture.h"

#include <algorithm>
#include <cstdint>
#include <cuda_fp16.h>

using namespace imp;
using namespace imp::vram_fixture;

// compute_native_cache_demand must mirror phase 3b's SfAtom slab sizing exactly
// (nvfp4_sfatom_size + 256-byte alignment, contiguous expert groups as ONE entry) and
// include the GDN/SSM projections phase 0b registers; the old elems/16 heuristic omitted
// them and was not an upper bound under SfAtom padding.

namespace {

size_t sf_entry(int n, int k_packed) {
    constexpr size_t kSfAlign = 256;
    size_t sz = imp::nvfp4_sfatom_size(n, k_packed * 2);
    return (sz + kSfAlign - 1) / kSfAlign * kSfAlign;
}

size_t sf_group(int ne, int n, int k_packed) {
    constexpr size_t kSfAlign = 256;
    size_t sz = static_cast<size_t>(ne) * imp::nvfp4_sfatom_size(n, k_packed * 2);
    return (sz + kSfAlign - 1) / kSfAlign * kSfAlign;
}

size_t moe_slab(int ne, int n, int k_packed) {
    int64_t k = static_cast<int64_t>(k_packed) * 2;
    return static_cast<size_t>(ne) * n * k_packed + static_cast<size_t>(ne) * n * (k / 16) +
           static_cast<size_t>(ne) * sizeof(float);
}

}  // namespace

TEST(VramBudgetReserve, NativeCacheDemandMatchesPhase3Sizing) {
    Model m;
    fill_prequant_model(m);

    NativeCacheDemand d = compute_native_cache_demand(m);

    const size_t expected_sf = 2 * (sf_entry(1024, 512)      // wq
                                    + sf_entry(2048, 1024)   // ssm_in (GDN hybrid!)
                                    + sf_entry(512, 1024)    // gdn_gate
                                    + sf_group(4, 768, 512)  // expert up group
                                    + sf_group(4, 256, 768)  // expert down group
                                   );
    EXPECT_EQ(d.sf_bytes, expected_sf);

    const size_t expected_slab = std::max(moe_slab(4, 768, 512), moe_slab(4, 256, 768));
    EXPECT_EQ(d.moe_slab_bytes, expected_slab);
    EXPECT_EQ(d.total(), expected_sf + expected_slab);
}

TEST(VramBudgetReserve, NativeCacheDemandZeroForNonPrequant) {
    Model m;
    fill_model(m, QType::Q6_K, QType::Q6_K);

    NativeCacheDemand d = compute_native_cache_demand(m);
    EXPECT_EQ(d.sf_bytes, 0u);
    EXPECT_EQ(d.moe_slab_bytes, 0u);
}

// #942: kv_block_bytes_per_layer is the single source for KV-size estimates. The pre-upload
// expert-offload reserve used to multiply by raw dtype_size(), which returns 0 for
// NVFP4/MXFP4_KV (zeroing KV headroom) and 1 byte/elem for INT4 (2x the packed size), and
// never counted scale overhead.
TEST(VramBudgetReserve, KvBlockBytesPerLayerIsPackingAndScaleAware) {
    constexpr int bs = 16, nkv = 8, hd = 128;
    const size_t elems = static_cast<size_t>(bs) * nkv * hd;

    // Plain dtypes: elems * elem_size * 2 (K+V), no scale overhead.
    EXPECT_EQ(kv_block_bytes_per_layer(QType::F16, bs, nkv, hd), elems * 2 * 2);
    EXPECT_EQ(kv_block_bytes_per_layer(QType::FP8_E4M3, bs, nkv, hd), elems * 1 * 2);

    // INT8: 1 byte/elem + per-token half scale.
    EXPECT_EQ(kv_block_bytes_per_layer(QType::INT8, bs, nkv, hd),
              (elems + static_cast<size_t>(bs) * nkv * sizeof(half)) * 2);

    // INT4: packed 2 elems/byte + per-token half scale — NOT dtype_size()'s
    // 1 byte/elem.
    EXPECT_EQ(kv_block_bytes_per_layer(QType::INT4, bs, nkv, hd),
              (elems / 2 + static_cast<size_t>(bs) * nkv * sizeof(half)) * 2);

    // NVFP4 / MXFP4_KV: packed + per-16-element-group scales; raw dtype_size()
    // returns 0 here, which is exactly the #942 failure mode.
    const size_t packed4 = (elems / 2 + static_cast<size_t>(bs) * nkv * (hd / 16)) * 2;
    EXPECT_EQ(kv_block_bytes_per_layer(QType::NVFP4, bs, nkv, hd), packed4);
    EXPECT_EQ(kv_block_bytes_per_layer(QType::MXFP4_KV, bs, nkv, hd), packed4);
    EXPECT_GT(kv_block_bytes_per_layer(QType::NVFP4, bs, nkv, hd), 0u);
}

// planned_post_cache_reserve: what a live-free-sized mode-2 cache leaves free once a plan applied.
TEST(VramBudgetReserve, PlannedPostCacheReserveSumsLaterClaims) {
    const size_t MiB = 1024ull * 1024;
    VRAMBudget b;
    EXPECT_EQ(planned_post_cache_reserve(b, 32579 * MiB), 0u) << "no plan, no planned reserve";
    b.kv_plan_bytes = 2178 * MiB;
    b.library_reserve_bytes = 543 * MiB;
    b.imma_plane_bytes = 100 * MiB;
    b.ssm_footprint_bytes = 34 * MiB;
    EXPECT_EQ(planned_post_cache_reserve(b, 32579 * MiB),
              (2178 + 256 + 543 + 100 + 34) * MiB + vram_allocator_headroom(32579 * MiB));
}

// GGUF MoE NVFP4 demand (gguf_moe_nvfp4_cache_bytes): whole eligible layers only, so the plan's
// cache grant covers exactly what phase 3-moe may build.
TEST(VramBudgetReserve, GgufMoeNvfp4DemandCountsWholeLayersOnly) {
    Model m;
    m.config_.n_layers = 2;
    auto experts = [](QType qt, int64_t rows, int64_t cols, uintptr_t s) {
        Tensor t = make_weight(TensorKind::W_GATE, rows, cols, qt, s);
        t.ndim = 3;
        t.shape[0] = 8;  // experts
        t.shape[1] = rows;
        t.shape[2] = cols;
        t.on_device = true;
        return t;
    };
    TransformerLayer whole;  // all Q6_K: eligible
    whole.expert_gate_packed = experts(QType::Q6_K, 768, 2048, 11);
    whole.expert_up_packed = experts(QType::Q6_K, 768, 2048, 12);
    whole.expert_down_packed = experts(QType::Q6_K, 2048, 768, 13);
    TransformerLayer partial;  // UD-Q4_K_M shape: Q4_K gate/up, Q6_K down
    partial.expert_gate_packed = experts(QType::Q4_K, 768, 2048, 21);
    partial.expert_up_packed = experts(QType::Q4_K, 768, 2048, 22);
    partial.expert_down_packed = experts(QType::Q6_K, 2048, 768, 23);
    m.layers_.push_back(std::move(whole));
    m.layers_.push_back(std::move(partial));

    const size_t per_tensor = moe_nvfp4_packed_bytes(8, 768, 2048);
    EXPECT_EQ(gguf_moe_nvfp4_cache_bytes(m, /*decode_all=*/false, 2), 3 * per_tensor);
    EXPECT_EQ(gguf_moe_nvfp4_cache_bytes(m, /*decode_all=*/true, 2), 6 * per_tensor);
    EXPECT_EQ(gguf_moe_nvfp4_cache_bytes(m, /*decode_all=*/true, 0), 0u) << "no NVFP4 decode, no cache";

    m.layers_[0].expert_up_packed.on_device = false;  // host-resident experts are not cached
    EXPECT_EQ(gguf_moe_nvfp4_cache_bytes(m, /*decode_all=*/false, 2), 0u);
}
