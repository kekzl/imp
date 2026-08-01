// Which tensors imp-quantize touches, and which checkpoints it refuses.
//
// This rule has already been wrong in the direction that ships a working-looking
// model: #1159 quantized MLA latent projections and MoE routers because they are
// 2-D and K-aligned, producing a checkpoint that loaded and then emitted
// garbage. Nothing here is a shape assertion for its own sake — each case is a
// tensor role that must or must not survive.

#include "../tools/imp-quantize/tensor_policy.h"

#include <gtest/gtest.h>

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

bool quantizes(const RawTensor& t, bool lm_head = false) {
    std::string why;
    return should_quantize(t, lm_head, why);
}

std::string reason(const RawTensor& t, bool lm_head = false) {
    std::string why;
    should_quantize(t, lm_head, why);
    return why;
}

TEST(QuantizePolicy, TakesOrdinaryLinearWeights) {
    EXPECT_TRUE(quantizes(tensor("model.layers.0.self_attn.q_proj.weight", {4096, 4096})));
    EXPECT_TRUE(quantizes(tensor("model.layers.0.mlp.down_proj.weight", {4096, 11008})));
    EXPECT_TRUE(quantizes(tensor("model.layers.0.mlp.experts.7.gate_proj.weight", {1408, 2048})))
        << "per-expert 2-D is the supported MoE layout";
}

// The two roles from #1159. Both are 2-D and K-aligned, so only the name
// distinguishes them from a weight that must be quantized.
TEST(QuantizePolicy, RefusesMlaLatentProjectionsAndTheRouter) {
    EXPECT_FALSE(quantizes(tensor("model.layers.0.self_attn.kv_a_proj_with_mqa.weight", {576, 2048})));
    EXPECT_FALSE(quantizes(tensor("model.layers.0.self_attn.kv_b_proj.weight", {4096, 512})));
    EXPECT_FALSE(quantizes(tensor("model.layers.0.mlp.gate.weight", {64, 2048})));
    EXPECT_FALSE(quantizes(tensor("model.layers.0.mlp.router.weight", {64, 2048})));
}

// `gate_proj` is an expert projection, `gate` is the router. A substring test
// instead of a suffix test would silently stop quantizing every expert.
TEST(QuantizePolicy, GateProjIsNotTheRouter) {
    EXPECT_TRUE(quantizes(tensor("model.layers.0.mlp.experts.0.gate_proj.weight", {1408, 2048})));
    EXPECT_TRUE(quantizes(tensor("model.layers.0.mlp.gate_proj.weight", {11008, 4096})));
}

TEST(QuantizePolicy, LeavesEmbeddingsNormsAndBiasesAlone) {
    EXPECT_FALSE(quantizes(tensor("model.embed_tokens.weight", {151936, 4096})));
    EXPECT_FALSE(quantizes(tensor("model.layers.0.input_layernorm.weight", {4096})));
    EXPECT_FALSE(quantizes(tensor("model.layers.0.self_attn.q_proj.bias", {4096})));
}

TEST(QuantizePolicy, LmHeadNeedsTheFlag) {
    const auto t = tensor("lm_head.weight", {151936, 4096});
    EXPECT_FALSE(quantizes(t, /*lm_head=*/false));
    EXPECT_TRUE(quantizes(t, /*lm_head=*/true));
}

TEST(QuantizePolicy, RefusesMisalignedKAndNonFloatDtypes) {
    EXPECT_FALSE(quantizes(tensor("model.layers.0.mlp.up_proj.weight", {4096, 100})))
        << "K must be a multiple of the 16-value micro-block";
    EXPECT_FALSE(quantizes(tensor("model.layers.0.mlp.up_proj.weight", {4096, 4096}, "U8")));
}

// The rank check has to come BEFORE the name check. With the other order a
// stacked tensor was diagnosed as "not a .weight tensor" — which produced no
// SKIP line, no counter and no exclusion entry, so it was copied through in
// silence. Real stacked checkpoints never use the `.weight` suffix.
TEST(QuantizePolicy, DiagnosesAStackWhateverItIsNamed) {
    EXPECT_NE(reason(tensor("model.layers.0.mlp.experts.gate_up_proj", {128, 1408, 2048})).find("3-D"),
              std::string::npos);
    EXPECT_NE(reason(tensor("model.layers.0.mlp.experts.gate_up_proj_blocks", {32, 5760, 90})).find("3-D"),
              std::string::npos);
    EXPECT_NE(reason(tensor("model.layers.0.mlp.experts.down_proj.weight", {128, 2048, 1408})).find("3-D"),
              std::string::npos)
        << "and still a stack when it does end in .weight";
}

TEST(StackedExperts, FindsThemUnderEveryRealNamingConvention) {
    const std::vector<RawTensor> tensors = {
        tensor("model.layers.0.mlp.experts.gate_up_proj", {128, 1408, 2048}),   // Gemma-4 style
        tensor("model.layers.0.mlp.experts.down_proj_blocks", {32, 5760, 90}),  // gpt-oss style
        tensor("model.layers.0.self_attn.q_proj.weight", {4096, 4096}),         // ordinary 2-D
        tensor("model.embed_tokens.weight", {151936, 4096}),
    };
    const auto found = find_stacked_expert_tensors(tensors);
    ASSERT_EQ(found.size(), 2u);
    EXPECT_EQ(found[0]->name, "model.layers.0.mlp.experts.gate_up_proj");
    EXPECT_EQ(found[1]->name, "model.layers.0.mlp.experts.down_proj_blocks");
}

// Rank 3 on its own is not a reason to refuse a checkpoint — a vision tower has
// plenty of 3-D tensors that have nothing to do with MoE.
TEST(StackedExperts, IgnoresThreeDTensorsThatAreNotExperts) {
    const std::vector<RawTensor> tensors = {
        tensor("model.vision.patch_embed.proj.weight", {1024, 3, 16}),
        tensor("model.layers.0.conv1d.weight", {4096, 1, 4}),
        tensor("model.vision.pos_embed", {1, 2304, 1024}),
    };
    EXPECT_TRUE(find_stacked_expert_tensors(tensors).empty());
}

// Already-quantized stacks are somebody else's export, not an input this tool
// would have mangled — it refuses them on dtype long before rank matters.
TEST(StackedExperts, IgnoresNonFloatStacks) {
    const std::vector<RawTensor> tensors = {
        tensor("model.layers.0.mlp.experts.gate_up_proj_blocks", {32, 5760, 90}, "U8"),
    };
    EXPECT_TRUE(find_stacked_expert_tensors(tensors).empty());
}

}  // namespace
}  // namespace imp::quantize
