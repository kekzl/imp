// The GGUF loader's half of "say what you dropped".
//
// `src/model/CLAUDE.md` states the failure this file guards: "A silently
// skipped tensor is the failure mode here", and "IMP_LOG_DEBUG is invisible at
// the default log level. A skip reported only there is not reported." The
// SafeTensors side got its per-class breakdown in #1929; these are the two
// places on the GGUF path that still did it the old way.
//
// Both are LATENT, not live on any checkpoint present. A GGUF header parse over
// all 15 local models finds zero unassigned tensors in every text model and
// exactly one in the reranker (`cls.output.weight`), which imp is right to
// ignore: /v1/rerank scores from the yes/no token logits and never reads a
// classification head. And no local GGUF ships a dense `ffn_*.bias`. They are
// pinned anyway, because the cost when they do fire is a clobbered weight and a
// wrong answer at exit code 0.

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

// A dense FFN bias would have been assign_quant'd straight into the weight
// slot, and assign_quant is a plain assignment (`loader_assign.h`), so the
// weight was gone. The neighbouring `attn_*` arms and the `_exps` arms below
// both test the suffix; these three did not. imp carries no dense FFN bias, so
// the right answer is to leave it unassigned and let the loader report it.
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

}  // namespace
}  // namespace imp
