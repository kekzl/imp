// GGUF loader's half of "say what you dropped" (src/model/CLAUDE.md: a silently skipped
// tensor is the failure mode; IMP_LOG_DEBUG-only reporting is not reporting). SafeTensors got
// its per-class breakdown in #1929; these are the two remaining GGUF-path spots.
// Both LATENT, not live on any local checkpoint (a header parse over all 15 local models
// finds zero unassigned tensors except the reranker's cls.output.weight, which imp is right
// to ignore). Pinned anyway: the cost when they fire is a clobbered weight and a wrong
// answer at exit code 0.

#include "model/gguf_loader_internal.h"
#include "model/model.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace imp {
namespace {

Tensor fake_weight(void* backing, int64_t n, int64_t k) {
    Tensor t;
    t.data = backing;
    t.qtype = QType::F16;
    t.ndim = 2;
    t.shape[0] = n;
    t.shape[1] = k;
    return t;
}

// A dense FFN bias would be assign_quant'd straight into the weight slot (assign_quant is a
// plain assignment, loader_assign.h), silently losing the weight. The neighbouring attn_*
// and _exps arms test the suffix; these three didn't. imp carries no dense FFN bias, so the
// right answer is to leave it unassigned and let the loader report it.
TEST(GgufSilentDrops, DenseFfnBiasDoesNotClobberTheWeight) {
    std::vector<uint16_t> backing(1024, 0);
    Model model;
    model.config_.arch = ModelArch::LLAMA;
    model.config_.n_layers = 1;
    model.layers_.resize(1);

    const Tensor w = fake_weight(backing.data(), 8, 8);
    ASSERT_TRUE(assign_tensor(model, "blk.0.ffn_up.weight", w, GgufWireType::F16));
    const void* after_weight = model.layers_[0].w_up.data;
    ASSERT_NE(after_weight, nullptr) << "the weight itself must still land";

    std::vector<uint16_t> other(64, 1);
    const Tensor bias = fake_weight(other.data(), 8, 1);
    EXPECT_FALSE(assign_tensor(model, "blk.0.ffn_up.bias", bias, GgufWireType::F16))
        << "an unassignable name must be reported, not absorbed";
    EXPECT_EQ(model.layers_[0].w_up.data, after_weight) << "the bias overwrote the weight";
}

TEST(GgufSilentDrops, DenseFfnGateAndDownBiasesAreRefusedToo) {
    std::vector<uint16_t> backing(1024, 0);
    Model model;
    model.config_.arch = ModelArch::LLAMA;
    model.config_.n_layers = 1;
    model.layers_.resize(1);
    const Tensor t = fake_weight(backing.data(), 8, 8);

    EXPECT_FALSE(assign_tensor(model, "blk.0.ffn_gate.bias", t, GgufWireType::F16));
    EXPECT_FALSE(assign_tensor(model, "blk.0.ffn_down.bias", t, GgufWireType::F16));
    // And the weights themselves are unaffected by the guard.
    EXPECT_TRUE(assign_tensor(model, "blk.0.ffn_gate.weight", t, GgufWireType::F16));
    EXPECT_TRUE(assign_tensor(model, "blk.0.ffn_down.weight", t, GgufWireType::F16));
}

// The per-expert arms already tested the suffix and route the bias to its own
// slot. Pinned so the fix above cannot be "simplified" into them.
TEST(GgufSilentDrops, ExpertFfnBiasStillReachesItsOwnSlot) {
    std::vector<uint16_t> backing(1024, 0);
    Model model;
    model.config_.arch = ModelArch::LLAMA;
    model.config_.n_layers = 1;
    model.layers_.resize(1);
    const Tensor t = fake_weight(backing.data(), 8, 8);

    EXPECT_TRUE(assign_tensor(model, "blk.0.ffn_up_exps.bias", t, GgufWireType::F16));
    EXPECT_NE(model.layers_[0].expert_up_bias.data, nullptr);
}

// The grouping that turns a dropped subtree into one log line. A 355-tensor
// vision tower must collapse to one family, or the report is as unreadable as
// the DEBUG lines it replaces.
TEST(GgufSilentDrops, NameFamilyCollapsesASubtree) {
    EXPECT_EQ(gguf_name_family("v.blk.12.attn_q.weight"), "v");
    EXPECT_EQ(gguf_name_family("v.blk.13.attn_q.weight"), "v");
    EXPECT_EQ(gguf_name_family("mm.0.weight"), "mm");
    EXPECT_EQ(gguf_name_family("cls.output.weight"), "cls");
    // Digit runs inside the head segment collapse, so an indexed head is still
    // one family rather than one per index.
    EXPECT_EQ(gguf_name_family("enc12.blk.0.weight"), "encN");
    EXPECT_EQ(gguf_name_family("enc7.blk.0.weight"), "encN");
    // A name with no dot is its own family, and an empty name is named.
    EXPECT_EQ(gguf_name_family("output_norm"), "output_norm");
    EXPECT_EQ(gguf_name_family(""), "(empty)");
}

// Qwen3.8-27B UD-Q4_K_M: block_count 65, nextn_predict_layers 1 -> 64 decoder layers; the
// MTP block ran as layer 65 before (PPL 4.94 vs 4.32 in llama.cpp).
TEST(GgufSilentDrops, NextnBlocksAreNotDecoderLayers) {
    EXPECT_EQ(gguf_decoder_layers(65, 1), 64u);
    EXPECT_EQ(gguf_decoder_layers(64, 0), 64u);
    EXPECT_EQ(gguf_decoder_layers(2, 2), 2u);  // nothing left: claim ignored
    EXPECT_EQ(gguf_decoder_layers(0, 1), 0u);
}

// The UD nextn block (blk.64 after 64 decoder layers) maps onto the Qwen MtpHead keys; other
// layers and unknown fields map to nothing.
TEST(GgufSilentDrops, NextnBlockMapsOntoMtpHeadKeys) {
    EXPECT_EQ(gguf_nextn_mtp_name("blk.64.nextn.eh_proj.weight", 64), "mtp.fc.weight");
    EXPECT_EQ(gguf_nextn_mtp_name("blk.64.nextn.enorm.weight", 64), "mtp.pre_fc_norm_embedding.weight");
    EXPECT_EQ(gguf_nextn_mtp_name("blk.64.nextn.hnorm.weight", 64), "mtp.pre_fc_norm_hidden.weight");
    EXPECT_EQ(gguf_nextn_mtp_name("blk.64.nextn.shared_head_norm.weight", 64), "mtp.norm.weight");
    EXPECT_EQ(gguf_nextn_mtp_name("blk.64.attn_q.weight", 64), "mtp.layers.0.self_attn.q_proj.weight");
    EXPECT_EQ(gguf_nextn_mtp_name("blk.64.ffn_down.weight", 64), "mtp.layers.0.mlp.down_proj.weight");
    EXPECT_EQ(gguf_nextn_mtp_name("blk.63.attn_q.weight", 64), "");
    EXPECT_EQ(gguf_nextn_mtp_name("blk.640.attn_q.weight", 64), "");
    EXPECT_EQ(gguf_nextn_mtp_name("blk.64.nextn.embed_tokens.weight", 64), "");
}

// H-012: "blk.3x" is a malformed index, not layer 3. Dropped with a WARN naming the tensor.
TEST(GgufSilentDrops, MalformedLayerIndexIsSkippedNotMapped) {
    std::vector<uint16_t> backing(1024, 0);
    Model model;
    model.config_.arch = ModelArch::LLAMA;
    model.config_.n_layers = 4;
    model.layers_.resize(4);
    const Tensor t = fake_weight(backing.data(), 8, 8);

    for (const char* name : {"blk.3x.attn_q.weight", "blk.x.attn_q.weight", "blk.+3.attn_q.weight",
                             "blk.4294967299.attn_q.weight"}) {
        testing::internal::CaptureStderr();
        EXPECT_FALSE(assign_tensor(model, name, t, GgufWireType::F16)) << name;
        const std::string log = testing::internal::GetCapturedStderr();
        EXPECT_NE(log.find(name), std::string::npos) << "the skip must name the tensor: " << log;
    }
    for (const auto& layer : model.layers_)
        EXPECT_EQ(layer.wq.data, nullptr) << "a malformed index landed in a layer";

    EXPECT_TRUE(assign_tensor(model, "blk.3.attn_q.weight", t, GgufWireType::F16));
    EXPECT_NE(model.layers_[3].wq.data, nullptr);
}

TEST(GgufSilentDrops, MalformedExpertIndexIsSkippedNotMapped) {
    std::vector<uint16_t> backing(1024, 0);
    Model model;
    model.config_.arch = ModelArch::LLAMA;
    model.config_.n_layers = 1;
    model.config_.n_experts = 4;
    model.layers_.resize(1);
    model.layers_[0].expert_w_gate.resize(4);
    const Tensor t = fake_weight(backing.data(), 8, 8);

    testing::internal::CaptureStderr();
    EXPECT_FALSE(assign_tensor(model, "blk.0.ffn_gate.3x.weight", t, GgufWireType::F16));
    const std::string log = testing::internal::GetCapturedStderr();
    EXPECT_NE(log.find("blk.0.ffn_gate.3x.weight"), std::string::npos) << log;
    for (const auto& e : model.layers_[0].expert_w_gate)
        EXPECT_EQ(e.data, nullptr) << "a malformed expert index landed in a slot";

    EXPECT_TRUE(assign_tensor(model, "blk.0.ffn_gate.3.weight", t, GgufWireType::F16));
    EXPECT_NE(model.layers_[0].expert_w_gate[3].data, nullptr);
}

}  // namespace
}  // namespace imp
