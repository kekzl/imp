// The 3-D expert stack -> per-expert 2-D split imp-quantize runs on gpt-oss and Gemma-4
// sources. Every wrong answer here is silent: a transposed or mis-paired matrix quantizes to a
// checkpoint that loads and generates. Shapes are the real ones (gpt-oss-20b BF16,
// Gemma-4-26B-A4B-it BF16), values are small integers so BF16 -> FP16 is exact.

#include "../tools/imp-quantize/expert_destack.h"

#include "core/fp_bits.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

namespace imp::quantize {
namespace {

RawTensor tensor(const std::string& name, std::vector<int64_t> shape, const std::string& dtype = "BF16") {
    RawTensor t;
    t.name = name;
    t.dtype = dtype;
    t.shape = std::move(shape);
    return t;
}

// A stack whose element at [e][a][b] is e*64 + a*stride_a + b: unique per position and exact
// in BF16 while it stays below 256 (stride_a = d2 rounded up, d1 * stride_a <= 64).
struct Stack {
    RawTensor t;
    std::vector<uint16_t> bits;
    Stack(const std::string& name, int64_t ne, int64_t d1, int64_t d2, int64_t stride_a,
          const std::string& dtype = "BF16")
        : t(tensor(name, {ne, d1, d2}, dtype)) {
        bits.resize(static_cast<size_t>(ne * d1 * d2));
        for (int64_t e = 0; e < ne; ++e)
            for (int64_t a = 0; a < d1; ++a)
                for (int64_t b = 0; b < d2; ++b) {
                    const float v = static_cast<float>(e * 64 + a * stride_a + b);
                    EXPECT_LT(v, 256.0f) << "encoding must stay exact in BF16";
                    bits[static_cast<size_t>((e * d1 + a) * d2 + b)] = dtype == "F16" ? float_to_half(v)
                                                                                      : float_to_bf16(v);
                }
        t.data = bits.data();
        t.nbytes = bits.size() * 2;
    }
};

const StackedExpertLayout kGptOss{/*transposed=*/true, GateUpOrder::Interleaved};
const StackedExpertLayout kGemma4{/*transposed=*/false, GateUpOrder::Concatenated};

TEST(ExpertDestack, LayoutIsLookedUpByModelType) {
    const auto oss = stacked_expert_layout("gpt_oss");
    ASSERT_TRUE(oss);
    EXPECT_TRUE(oss->transposed);
    EXPECT_EQ(oss->gate_up, GateUpOrder::Interleaved);
    for (const char* mt : {"gemma4", "gemma4_text"}) {
        const auto g = stacked_expert_layout(mt);
        ASSERT_TRUE(g) << mt;
        EXPECT_FALSE(g->transposed) << mt;
        EXPECT_EQ(g->gate_up, GateUpOrder::Concatenated) << mt;
    }
    EXPECT_FALSE(stacked_expert_layout("qwen3_moe")) << "per-expert 2-D models have no stack";
    EXPECT_FALSE(stacked_expert_layout(""));
}

TEST(ExpertDestack, PlanNamesGptOssMatricesLikeTheHfPerExpertLayout) {
    const auto gu = destack_plan(tensor("model.layers.3.mlp.experts.gate_up_proj", {32, 2880, 5760}),
                                 kGptOss);
    ASSERT_TRUE(gu) << gu.error();
    ASSERT_EQ(gu->size(), 64u);
    EXPECT_EQ((*gu)[0].name, "model.layers.3.mlp.experts.0.gate_proj.weight");
    EXPECT_EQ((*gu)[0].part, ExpertPart::Gate);
    EXPECT_EQ((*gu)[1].name, "model.layers.3.mlp.experts.0.up_proj.weight");
    EXPECT_EQ((*gu)[1].part, ExpertPart::Up);
    EXPECT_EQ((*gu)[63].name, "model.layers.3.mlp.experts.31.up_proj.weight");
    EXPECT_EQ((*gu)[63].expert, 31);
    for (const auto& m : *gu) {
        EXPECT_EQ(m.N, 2880) << m.name;
        EXPECT_EQ(m.K, 2880) << m.name;
    }
    const auto dn = destack_plan(tensor("model.layers.3.mlp.experts.down_proj", {32, 2880, 2880}), kGptOss);
    ASSERT_TRUE(dn) << dn.error();
    ASSERT_EQ(dn->size(), 32u);
    EXPECT_EQ((*dn)[5].name, "model.layers.3.mlp.experts.5.down_proj.weight");
    EXPECT_EQ((*dn)[5].part, ExpertPart::Down);
    EXPECT_EQ((*dn)[5].N, 2880);
    EXPECT_EQ((*dn)[5].K, 2880);
}

// Gemma-4 omits the `mlp.` segment; the split keeps the parent so the loader's
// `experts.{e}.{proj}.weight` branch (weight_map.cpp) sees the same spelling its NVFP4 export has.
TEST(ExpertDestack, PlanReadsGemma4ShapesInLinearOrientation) {
    const auto gu = destack_plan(tensor("model.language_model.layers.0.experts.gate_up_proj",
                                        {128, 1408, 2816}),
                                 kGemma4);
    ASSERT_TRUE(gu) << gu.error();
    ASSERT_EQ(gu->size(), 256u);
    EXPECT_EQ((*gu)[10].name, "model.language_model.layers.0.experts.5.gate_proj.weight");
    EXPECT_EQ((*gu)[10].N, 704);
    EXPECT_EQ((*gu)[10].K, 2816);
    const auto dn = destack_plan(tensor("model.language_model.layers.0.experts.down_proj", {128, 2816, 704}),
                                 kGemma4);
    ASSERT_TRUE(dn) << dn.error();
    ASSERT_EQ(dn->size(), 128u);
    EXPECT_EQ((*dn)[127].name, "model.language_model.layers.0.experts.127.down_proj.weight");
    EXPECT_EQ((*dn)[127].N, 2816);
    EXPECT_EQ((*dn)[127].K, 704);
}

TEST(ExpertDestack, PlanRefusesWhatItCannotSplit) {
    EXPECT_FALSE(
        destack_plan(tensor("model.layers.0.mlp.experts.gate_up_proj_blocks", {32, 5760, 90}, "U8"), kGptOss))
        << "somebody else's packed export";
    EXPECT_FALSE(destack_plan(tensor("model.layers.0.mlp.experts.gate_up_proj", {32, 2880}), kGptOss))
        << "2-D";
    EXPECT_FALSE(destack_plan(tensor("model.layers.0.mlp.experts.gate_up_proj", {32, 2880, 5761}), kGptOss))
        << "odd fused rows cannot pair";
    EXPECT_FALSE(destack_plan(tensor("model.layers.0.experts.gate_up_proj", {128, 1408, 2820}), kGemma4))
        << "K not a multiple of 16";
    EXPECT_FALSE(destack_plan(tensor("model.layers.0.mlp.experts.router_proj", {32, 64, 16}), kGptOss))
        << "not a stack this tool knows";
}

// Gemma-4 [ne, 2I, K]: gate = rows [0, I), up = rows [I, 2I), element (n, k) at row*K + k.
TEST(ExpertDestack, ReadsConcatenatedGateUp) {
    const Stack s("m.layers.0.experts.gate_up_proj", /*ne=*/2, /*rows=*/4, /*K=*/16, /*stride_a=*/16);
    const auto plan = destack_plan(s.t, kGemma4);
    ASSERT_TRUE(plan) << plan.error();
    ASSERT_EQ(plan->size(), 4u);
    const auto gate1 = destack_read(s.t, kGemma4, (*plan)[2]);  // expert 1, gate
    const auto up1 = destack_read(s.t, kGemma4, (*plan)[3]);    // expert 1, up
    ASSERT_EQ(gate1.size(), 32u);
    for (int64_t n = 0; n < 2; ++n)
        for (int64_t k = 0; k < 16; ++k) {
            EXPECT_EQ(gate1[n * 16 + k], float_to_half(static_cast<float>(64 + n * 16 + k))) << n << "," << k;
            EXPECT_EQ(up1[n * 16 + k], float_to_half(static_cast<float>(64 + (n + 2) * 16 + k)))
                << n << "," << k;
        }
}

// gpt-oss [ne, K, 2I]: element (n, k) of gate at k*R + 2n, of up at k*R + 2n + 1.
TEST(ExpertDestack, ReadsInterleavedTransposedGateUp) {
    const Stack s("m.layers.0.mlp.experts.gate_up_proj", /*ne=*/2, /*K=*/16, /*fused=*/4, /*stride_a=*/4);
    const auto plan = destack_plan(s.t, kGptOss);
    ASSERT_TRUE(plan) << plan.error();
    ASSERT_EQ(plan->size(), 4u);
    EXPECT_EQ((*plan)[0].N, 2);
    EXPECT_EQ((*plan)[0].K, 16);
    const auto gate1 = destack_read(s.t, kGptOss, (*plan)[2]);
    const auto up1 = destack_read(s.t, kGptOss, (*plan)[3]);
    for (int64_t n = 0; n < 2; ++n)
        for (int64_t k = 0; k < 16; ++k) {
            // stack value = e*64 + k*4 + r, with r the fused column
            EXPECT_EQ(gate1[n * 16 + k], float_to_half(static_cast<float>(64 + k * 4 + 2 * n)))
                << n << "," << k;
            EXPECT_EQ(up1[n * 16 + k], float_to_half(static_cast<float>(64 + k * 4 + 2 * n + 1)))
                << n << "," << k;
        }
}

// gpt-oss down [ne, I, hidden] = [ne, K, N]: out(n, k) = stack[e][k][n].
TEST(ExpertDestack, ReadsTransposedDown) {
    const Stack s("m.layers.0.mlp.experts.down_proj", /*ne=*/2, /*K=*/16, /*N=*/3, /*stride_a=*/4);
    const auto plan = destack_plan(s.t, kGptOss);
    ASSERT_TRUE(plan) << plan.error();
    ASSERT_EQ(plan->size(), 2u);
    EXPECT_EQ((*plan)[1].N, 3);
    EXPECT_EQ((*plan)[1].K, 16);
    const auto d1 = destack_read(s.t, kGptOss, (*plan)[1]);
    ASSERT_EQ(d1.size(), 48u);
    for (int64_t n = 0; n < 3; ++n)
        for (int64_t k = 0; k < 16; ++k)
            EXPECT_EQ(d1[n * 16 + k], float_to_half(static_cast<float>(64 + k * 4 + n))) << n << "," << k;
}

// Gemma-4 down [ne, hidden, I] = [ne, N, K] is already a Linear weight: a straight row copy.
TEST(ExpertDestack, ReadsLinearDownAsRows) {
    const Stack s("m.layers.0.experts.down_proj", /*ne=*/1, /*N=*/3, /*K=*/16, /*stride_a=*/16);
    const auto plan = destack_plan(s.t, kGemma4);
    ASSERT_TRUE(plan) << plan.error();
    const auto d0 = destack_read(s.t, kGemma4, (*plan)[0]);
    for (int64_t n = 0; n < 3; ++n)
        for (int64_t k = 0; k < 16; ++k)
            EXPECT_EQ(d0[n * 16 + k], float_to_half(static_cast<float>(n * 16 + k))) << n << "," << k;
}

TEST(ExpertDestack, F16SourceBitsPassThrough) {
    const Stack s("m.layers.0.experts.down_proj", 1, 2, 16, /*stride_a=*/16, "F16");
    const auto plan = destack_plan(s.t, kGemma4);
    ASSERT_TRUE(plan) << plan.error();
    const auto d0 = destack_read(s.t, kGemma4, (*plan)[0]);
    ASSERT_EQ(d0.size(), 32u);
    for (size_t i = 0; i < d0.size(); ++i)
        EXPECT_EQ(d0[i], s.bits[i]) << i;
}

}  // namespace
}  // namespace imp::quantize
