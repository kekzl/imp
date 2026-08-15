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

// ── fused Q + gate projections ───────────────────────────────────────
//
// #1273's root cause. A Qwen3.5 / Qwen3-Next `attn_output_gate` layer emits Q
// and the sigmoid gate from ONE q_proj, so the tensor carries two roles with
// very different sensitivity to NVFP4. Detection is by shape rather than by a
// config flag: a gated q_proj emits twice what the layer's o_proj consumes.

// A gated hybrid attention layer: 16 heads x 256, so o_proj consumes 4096 and
// q_proj emits 8192 (Q + gate). Matches Qwen3.6-35B-A3B.
std::vector<RawTensor> gated_layer(const std::string& prefix) {
    return {tensor(prefix + ".self_attn.q_proj.weight", {8192, 2048}),
            tensor(prefix + ".self_attn.k_proj.weight", {512, 2048}),
            tensor(prefix + ".self_attn.v_proj.weight", {512, 2048}),
            tensor(prefix + ".self_attn.o_proj.weight", {2048, 4096})};
}

// A dense attention layer: q_proj emits exactly what o_proj consumes.
std::vector<RawTensor> plain_layer(const std::string& prefix) {
    return {tensor(prefix + ".self_attn.q_proj.weight", {4096, 4096}),
            tensor(prefix + ".self_attn.o_proj.weight", {4096, 4096})};
}

TEST(FusedGateQProj, FindsTheGatedProjectionAndLeavesADenseOneAlone) {
    std::vector<RawTensor> ts = gated_layer("model.layers.3");
    const auto found = find_fused_gate_q_projections(ts);
    ASSERT_EQ(found.size(), 1u) << "a q_proj emitting 2x the o_proj input carries a gate";
    EXPECT_EQ(found[0]->name, "model.layers.3.self_attn.q_proj.weight");

    std::vector<RawTensor> dense = plain_layer("model.layers.3");
    EXPECT_TRUE(find_fused_gate_q_projections(dense).empty()) << "a dense q_proj must not match";
}

TEST(FusedGateQProj, MatchesPerLayerRatherThanAcrossTheCheckpoint) {
    // Two gated layers with DIFFERENT head counts: 16x256 (o_proj takes 4096)
    // and 24x256 (o_proj takes 6144). Pairing a q_proj against some other
    // layer's o_proj finds only one of them.
    //
    // Written this way after a mutation run: the first version used a gated
    // layer plus a DENSE one, and replacing the per-layer lookup with "any
    // o_proj" left all four tests green — the dense layer failed to match under
    // both the right rule and the wrong one, so it could not tell them apart.
    std::vector<RawTensor> ts = gated_layer("model.layers.3");
    ts.push_back(tensor("model.layers.7.self_attn.q_proj.weight", {12288, 2048}));
    ts.push_back(tensor("model.layers.7.self_attn.o_proj.weight", {2048, 6144}));

    const auto found = find_fused_gate_q_projections(ts);
    ASSERT_EQ(found.size(), 2u) << "each q_proj must be compared against its OWN layer's o_proj";
    EXPECT_EQ(found[0]->name, "model.layers.3.self_attn.q_proj.weight");
    EXPECT_EQ(found[1]->name, "model.layers.7.self_attn.q_proj.weight");
}

TEST(FusedGateQProj, FindsThemUnderTheNestedLanguageModelPrefix) {
    // The staged NVFP4 hybrids name them model.language_model.layers.N.*
    std::vector<RawTensor> ts = gated_layer("model.language_model.layers.31");
    const auto found = find_fused_gate_q_projections(ts);
    ASSERT_EQ(found.size(), 1u);
    EXPECT_EQ(found[0]->name, "model.language_model.layers.31.self_attn.q_proj.weight");
}

// The MTP draft head. Its loader takes `mtp.*.weight` by name and never looks
// for the scale companions, so quantizing it yields a head that loads, drafts,
// and has every draft rejected. Measured: 81% acceptance to 0 of 24.
TEST(ShouldQuantize, LeavesTheMtpDraftHeadAlone) {
    EXPECT_FALSE(quantizes(tensor("mtp.fc.weight", {5120, 10240})));
    EXPECT_FALSE(quantizes(tensor("mtp.layers.0.mlp.down_proj.weight", {5120, 17408})));
    EXPECT_FALSE(quantizes(tensor("mtp.layers.0.self_attn.q_proj.weight", {12288, 5120})));

    // Anchored at the start, so a main-model tensor whose name merely contains
    // the letters is unaffected. Nothing should widen this by accident.
    EXPECT_TRUE(quantizes(tensor("model.layers.0.mtp_like.weight", {5120, 5120})));
}

TEST(FusedGateQProj, DoesNotGuessWhenTheLayerHasNoOProj) {
    // Without the o_proj there is nothing to compare against, and a bare
    // "shape[0] is even" heuristic would flag every ordinary projection.
    std::vector<RawTensor> ts = {tensor("model.layers.3.self_attn.q_proj.weight", {8192, 2048})};
    EXPECT_TRUE(find_fused_gate_q_projections(ts).empty()) << "no reference — must not guess";
}

// The --dry-run size forecast is this arithmetic, and the writing path asserts
// its real buffers against it. Pinning the three components separately means a
// layout change breaks the test that explains the layout, not just a total.
TEST(Nvfp4OutputBytes, CountsPackedNibblesMicroScalesAndTensorScale) {
    // A real Qwen3.8 projection: [17408, 5120].
    const int64_t N = 17408, K = 5120;
    const size_t packed = size_t(N) * K / 2;   // two 4-bit values per byte
    const size_t micro = size_t(N) * K / 16;   // one FP8 scale per 16 values
    EXPECT_EQ(nvfp4_output_bytes(N, K), packed + micro + sizeof(float));

    // The point of quantizing: 2 bytes per value become 0.5 + 0.0625.
    const double ratio = double(size_t(N) * K * 2) / double(nvfp4_output_bytes(N, K));
    EXPECT_NEAR(ratio, 32.0 / 9.0, 0.01) << "BF16 against packed+micro is 2 / (0.5 + 0.0625)";
}

TEST(Nvfp4OutputBytes, RefusesToInventBytesForAnEmptyMatrix) {
    // Guards the forecast against a degenerate shape adding a phantom 4 bytes
    // per tensor, which on a 1000-tensor checkpoint is a visible drift.
    EXPECT_EQ(nvfp4_output_bytes(0, 5120), 0u);
    EXPECT_EQ(nvfp4_output_bytes(17408, 0), 0u);
    EXPECT_EQ(nvfp4_output_bytes(-1, 5120), 0u);
}

}  // namespace
}  // namespace imp::quantize
