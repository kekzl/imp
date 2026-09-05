// AUDIT_arch_2026 F1-6: an adapter's declared widths are held against the
// base model at load. The rule table is a pure function so it runs in the
// unit lane; the loader-level refusals live in test_lora.cpp (model-backed).
#include <gtest/gtest.h>
#include "lora/lora_adapter.h"

namespace imp {
namespace {

// Llama-3.2-3B: d_model 3072, 24 heads x 128, 8 KV heads x 128, d_ff 8192.
const LoraDims kLlama3B{3072, 3072, 1024, 8192};

TEST(LoraDimsTest, ProjectionWidthsFollowTheBaseModel) {
    struct Row {
        LoraProj p;
        int K, N;
    };
    const Row rows[] = {{LoraProj::Q, 3072, 3072},    {LoraProj::K, 3072, 1024}, {LoraProj::V, 3072, 1024},
                        {LoraProj::O, 3072, 3072},    {LoraProj::GATE, 3072, 8192},
                        {LoraProj::UP, 3072, 8192},   {LoraProj::DOWN, 8192, 3072}};
    for (const Row& r : rows) {
        int K = 0, N = 0;
        ASSERT_TRUE(lora_proj_expected(r.p, kLlama3B, &K, &N)) << static_cast<int>(r.p);
        EXPECT_EQ(K, r.K) << static_cast<int>(r.p);
        EXPECT_EQ(N, r.N) << static_cast<int>(r.p);
    }
}

TEST(LoraDimsTest, GqaKeepsKvNarrowerThanQ) {
    int K = 0, N = 0;
    ASSERT_TRUE(lora_proj_expected(LoraProj::V, kLlama3B, &K, &N));
    EXPECT_LT(N, kLlama3B.q_out);
    EXPECT_EQ(K, kLlama3B.d_model);
}

TEST(LoraDimsTest, FfnTargetsRefusedWithoutDenseFfn) {
    LoraDims moe = kLlama3B;
    moe.d_ff = 0;
    int K = 0, N = 0;
    EXPECT_FALSE(lora_proj_expected(LoraProj::GATE, moe, &K, &N));
    EXPECT_FALSE(lora_proj_expected(LoraProj::UP, moe, &K, &N));
    EXPECT_FALSE(lora_proj_expected(LoraProj::DOWN, moe, &K, &N));
    EXPECT_TRUE(lora_proj_expected(LoraProj::Q, moe, &K, &N));
}

}  // namespace
}  // namespace imp
