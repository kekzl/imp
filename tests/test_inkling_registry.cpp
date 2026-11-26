// Inkling2-Loop in the engine's WeightRegistry and its LoRA contract (B-111 S5, exec/inkling_registry.h):
// which handles its projections get, which adapter tensors map to q / k / v / o, which refuse the load.
// Host only: the weights' device pointers are never read.

#include <gtest/gtest.h>

#include <cstdint>
#include <string>

#include "exec/inkling_registry.h"
#include "lora/lora_adapter.h"
#include "model/inkling_weights.h"

using namespace imp;

namespace {

// Inkling-2-Mini's shape: 24 layers, full attention on 5, 11, 17 and 23.
InklingConfig mini() {
    InklingConfig c;
    c.global.assign(static_cast<size_t>(c.n_layers), false);
    for (int l : {5, 11, 17, 23})
        c.global[static_cast<size_t>(l)] = true;
    return c;
}

// Fake device addresses laid out as load_inkling_weights lays out the row block; never dereferenced.
half* at(uintptr_t a) { return reinterpret_cast<half*>(a); }

void fake_layers(InklingWeights& w) {
    const InklingConfig& c = w.cfg;
    const uintptr_t H = static_cast<uintptr_t>(c.hidden);
    const uintptr_t KD = static_cast<uintptr_t>(c.n_kv_heads) * c.head_dim;
    w.layers.resize(static_cast<size_t>(c.n_layers));
    for (size_t i = 0; i < w.layers.size(); ++i) {
        InklingLayer& ly = w.layers[i];
        const uintptr_t TD = c.global[i] ? static_cast<uintptr_t>(c.n_kv_heads) * 4 : 0;
        const uintptr_t q_rows = static_cast<uintptr_t>(inkling_q_rows(c));
        const uintptr_t base = (i + 1) << 32;
        ly.proj_rows = static_cast<int>(q_rows + 2 * (KD + TD));
        ly.wq = at(base);
        ly.wk = at(base + q_rows * H * sizeof(half));
        ly.wv = at(base + (q_rows + KD + TD) * H * sizeof(half));
        ly.wo = at(base + (1ull << 31));
    }
}

}  // namespace

TEST(InklingRegistry, FiveBorrowedFp16HandlesPerLayerAppendedAfterTheModels) {
    InklingWeights w;
    w.cfg = mini();
    fake_layers(w);
    WeightRegistry reg;
    // Phase 4's handles (LM head, embedding) stay where they are.
    (void)reg.reserve(TensorKind::LM_HEAD, 200058, 2304);
    (void)reg.reserve(TensorKind::TOK_EMBED, 201024, 2304);
    const std::vector<InklingProjIds> ids = register_inkling_weights(w, reg);
    ASSERT_EQ(ids.size(), 24u);
    EXPECT_EQ(reg.size(), 2u + 24 * 5);
    EXPECT_EQ(ids[0].qkv, 2);
    const int64_t H = 2304, QD = 3072, KD = 1024, q_rows = 2 * 3072 + 24 * 16 + 24;
    EXPECT_EQ(inkling_q_rows(w.cfg), q_rows);
    for (size_t i = 0; i < ids.size(); ++i) {
        const InklingLayer& ly = w.layers[i];
        const int64_t kv = KD + (w.cfg.global[i] ? 32 : 0);
        struct Row {
            TensorID id;
            TensorKind kind;
            const half* p;
            int64_t rows, cols;
        };
        const Row rows[] = {{ids[i].qkv, TensorKind::QKV_FUSED, ly.wq, ly.proj_rows, H},
                            {ids[i].q, TensorKind::WQ, ly.wq, q_rows, H},
                            {ids[i].k, TensorKind::WK, ly.wk, kv, H},
                            {ids[i].v, TensorKind::WV, ly.wv, kv, H},
                            {ids[i].o, TensorKind::WO, ly.wo, H, QD}};
        for (const Row& r : rows) {
            const WeightHandle& h = reg.handle(r.id);
            EXPECT_EQ(h.kind, r.kind) << i;
            EXPECT_EQ(h.shape[0], r.rows) << i << " " << static_cast<int>(r.kind);
            EXPECT_EQ(h.shape[1], r.cols) << i;
            EXPECT_EQ(h.primary_tier, StorageTier::FP16);
            EXPECT_EQ(h.prefill_tier, StorageTier::FP16);
            EXPECT_EQ(h.decode_tier, StorageTier::FP16);
            EXPECT_EQ(h.payload.fp16.data, r.p);
            EXPECT_EQ(h.source_data, r.p);
            EXPECT_EQ(h.source_qtype, QType::F16);
            EXPECT_EQ(h.owned_bytes, 0) << "borrowed: InklingWeights owns the bytes";
            EXPECT_FALSE(h.can_drop_source());
        }
        // The q, k and v handles tile the row block the sliding first pass multiplies at once.
        EXPECT_EQ(q_rows + 2 * kv, ly.proj_rows) << i;
    }
    // A layer without weights (never loaded) gets no handle.
    w.layers[3].wo = nullptr;
    WeightRegistry reg2;
    EXPECT_EQ(register_inkling_weights(w, reg2)[3].o, kInvalidTensorID);
    EXPECT_EQ(reg2.size(), 24u * 5 - 1);
}

TEST(InklingLoraTargets, InklingModuleNamesMapToQkvo) {
    struct Row {
        const char* module;
        LoraProj p;
    };
    const Row rows[] = {{"attn.wq_du", LoraProj::Q},
                        {"attn.wk_dv", LoraProj::K},
                        {"attn.wv_dv", LoraProj::V},
                        {"attn.wo_ud", LoraProj::O}};
    for (const Row& r : rows)
        for (const char* ab : {"lora_A", "lora_B"}) {
            const std::string key = std::string("base_model.model.model.llm.layers.17.") + r.module + "." +
                                    ab + ".weight";
            int layer = -1;
            LoraProj p{};
            bool is_A = false;
            ASSERT_TRUE(lora_parse_key(key, &layer, &p, &is_A)) << key;
            EXPECT_EQ(layer, 17);
            EXPECT_EQ(p, r.p) << key;
            EXPECT_EQ(is_A, std::string(ab) == "lora_A");
            EXPECT_EQ(lora_key_refusal(key, kInklingLoraTargets), "") << key;
        }
}

TEST(InklingLoraTargets, TensorsOutsideQkvoRefuseTheLoad) {
    const std::string p = "base_model.model.model.llm.layers.6.";
    for (const char* refused : {"mlp.gate_proj.lora_A.weight", "mlp.down_proj.lora_B.weight",
                                "mlp.experts.w13_weight.lora_A.weight", "attn.wr_du.lora_A.weight",
                                "controllers.k.down_weight.lora_A.weight"}) {
        const std::string why = lora_key_refusal(p + refused, kInklingLoraTargets);
        EXPECT_NE(why.find(p + refused), std::string::npos) << why;
        EXPECT_NE(why.find("q/k/v/o only"), std::string::npos) << why;
        // Every other forward skips what it does not map (#522 v1), or loads the FFN pair.
        EXPECT_EQ(lora_key_refusal(p + refused, LoraTargets{}), "") << refused;
    }
    EXPECT_NE(lora_key_refusal("base_model.model.model.llm.embed.lora_embedding_A", kInklingLoraTargets), "");
    // Not an adapter tensor: never refused.
    EXPECT_EQ(lora_key_refusal(p + "attn.wq_du.weight", kInklingLoraTargets), "");
}

TEST(InklingLoraTargets, DimsAndScratchFollowTheCheckpoint) {
    const LoraDims d = inkling_lora_dims(mini());
    int K = 0, N = 0;
    ASSERT_TRUE(lora_proj_expected(LoraProj::Q, d, &K, &N));
    EXPECT_EQ(K, 2304);
    EXPECT_EQ(N, 3072);
    ASSERT_TRUE(lora_proj_expected(LoraProj::V, d, &K, &N));
    EXPECT_EQ(N, 1024);
    ASSERT_TRUE(lora_proj_expected(LoraProj::O, d, &K, &N));
    EXPECT_EQ(K, 3072);
    EXPECT_EQ(N, 2304);
    EXPECT_FALSE(lora_proj_expected(LoraProj::UP, d, &K, &N));
    // q, k and v run on three streams at once: three rank planes of rows x rank fp32.
    EXPECT_EQ(inkling_lora_scratch_bytes(4096, 16), 3u * 4096 * 16 * 4);
}
