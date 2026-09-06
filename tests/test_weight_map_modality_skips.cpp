// A dropped modality must be nameable, not folded into a total.
//
// `WeightMap::apply_weights` counted every unassigned tensor into one `skipped`
// integer. On Gemma-4-12B-NVFP4 that number was 11: one
// `model.embed_audio.embedding_projection.weight` (4.69 MiB) and ten
// `model.embed_vision.*` (95.22 MiB). The audio half is the one nothing in imp
// owns - no encoder, no input type, no tokenizer route - so the checkpoint
// loaded as a text model and said so nowhere (roadmap Open 8).
//
// These tests pin the breakdown, and pin that a text-only load still counts
// nothing. Tensor names are the checkpoint's own spelling, read from
// Gemma-4-12B-NVFP4's shard index.

#include "model/model.h"
#include "model/weight_map.h"

#include <gtest/gtest.h>

#include <string>
#include <unordered_map>
#include <vector>

namespace imp {
namespace {

// Nothing here reads the storage: the skip decision is made on the name alone,
// and the assignment path only needs a non-null `data`.
Tensor fake_weight(void* backing, int64_t n, int64_t k) {
    Tensor t;
    t.data = backing;
    t.qtype = QType::F16;
    t.ndim = 2;
    t.shape[0] = n;
    t.shape[1] = k;
    return t;
}

struct Fixture {
    std::vector<uint16_t> backing{1024, 0};
    Model model;
    std::unordered_map<std::string, Tensor> tensors;

    Fixture() {
        model.config_.arch = ModelArch::GEMMA4;
        model.config_.n_layers = 1;
        model.config_.d_model = 8;
        model.config_.n_experts = 0;
        model.layers_.resize(1);
    }

    void* mem() { return backing.data(); }

    void add(const std::string& name) { tensors[name] = fake_weight(mem(), 8, 8); }

    // Enough of a text tower that `assigned > 0` and the load returns true.
    void add_language_model() {
        add("model.language_model.embed_tokens.weight");
        add("model.language_model.norm.weight");
        add("model.language_model.layers.0.input_layernorm.weight");
        add("model.language_model.layers.0.self_attn.q_proj.weight");
    }

    // The ten `model.embed_vision.*` tensors of Gemma-4-12B-NVFP4.
    void add_vision_embedder() {
        add("model.embed_vision.multimodal_embedder.embedding_projection.weight");
        add("model.embed_vision.patch_dense.bias");
        add("model.embed_vision.patch_dense.weight");
        add("model.embed_vision.patch_ln1.bias");
        add("model.embed_vision.patch_ln1.weight");
        add("model.embed_vision.patch_ln2.bias");
        add("model.embed_vision.patch_ln2.weight");
        add("model.embed_vision.pos_embedding");
        add("model.embed_vision.pos_norm.bias");
        add("model.embed_vision.pos_norm.weight");
    }
};

// The Gemma-4-12B-NVFP4 shape: 1 audio + 10 vision embedder tensors. Before the
// split this was "skipped 11" and the audio drop was unrecoverable from the log.
TEST(WeightMapModalitySkips, AudioIsCountedApartFromVision) {
    Fixture f;
    f.add_language_model();
    f.add_vision_embedder();
    f.add("model.embed_audio.embedding_projection.weight");

    WeightMap wm(ModelArch::GEMMA4);
    ASSERT_TRUE(wm.apply_weights(f.model, f.tensors));

    const auto& s = wm.skip_stats();
    EXPECT_EQ(s.audio, 1);
    EXPECT_EQ(s.vision, 10);
    EXPECT_EQ(s.mtp, 0);
    EXPECT_EQ(s.unrecognised, 0);
    EXPECT_EQ(s.total, 11);
}

// Gemma-4-26B-A4B-it-NVFP4 has a vision tower and no audio at all. Vision alone
// must not read as an audio drop, or the warning cries wolf on every Gemma-4.
TEST(WeightMapModalitySkips, VisionOnlyReportsNoAudio) {
    Fixture f;
    f.add_language_model();
    f.add("model.embed_vision.embedding_projection.weight");
    f.add("model.vision_tower.encoder.layers.0.mlp.fc1.weight");
    f.add("model.vision_tower.encoder.layers.0.mlp.fc2.weight");

    WeightMap wm(ModelArch::GEMMA4);
    ASSERT_TRUE(wm.apply_weights(f.model, f.tensors));

    const auto& s = wm.skip_stats();
    EXPECT_EQ(s.audio, 0);
    EXPECT_EQ(s.vision, 3);
    EXPECT_EQ(s.total, 3);
}

// A text-only checkpoint stays silent: every counter zero, so the new INFO and
// WARN lines never print on the common load.
TEST(WeightMapModalitySkips, TextOnlyLoadSkipsNothing) {
    Fixture f;
    f.add_language_model();

    WeightMap wm(ModelArch::GEMMA4);
    ASSERT_TRUE(wm.apply_weights(f.model, f.tensors));

    const auto& s = wm.skip_stats();
    EXPECT_EQ(s.total, 0);
    EXPECT_EQ(s.audio, 0);
    EXPECT_EQ(s.vision, 0);
    EXPECT_EQ(s.unrecognised, 0);
}

// The other half of the split: an unreadable name is a different event from a
// dropped modality. Folded together, neither could be grepped for.
TEST(WeightMapModalitySkips, UnrecognisedNameIsNotAModalityDrop) {
    Fixture f;
    f.add_language_model();
    f.add("some.tensor.this.map.does.not.read");

    WeightMap wm(ModelArch::GEMMA4);
    ASSERT_TRUE(wm.apply_weights(f.model, f.tensors));

    const auto& s = wm.skip_stats();
    EXPECT_EQ(s.unrecognised, 1);
    EXPECT_EQ(s.audio, 0);
    EXPECT_EQ(s.vision, 0);
    EXPECT_EQ(s.total, 1);
}

// The MTP head has its own loader on the `mtp.` spelling; on the multimodal
// path it is stripped here. It is not a modality drop either.
TEST(WeightMapModalitySkips, MtpPrefixCountedApart) {
    Fixture f;
    f.add_language_model();
    f.add("model.mtp.norm.weight");

    WeightMap wm(ModelArch::GEMMA4);
    ASSERT_TRUE(wm.apply_weights(f.model, f.tensors));

    const auto& s = wm.skip_stats();
    EXPECT_EQ(s.mtp, 1);
    EXPECT_EQ(s.audio, 0);
    EXPECT_EQ(s.unrecognised, 0);
}

// Counters start clean, so a caller reading them before the load cannot mistake
// leftover state for a finding.
TEST(WeightMapModalitySkips, StatsAreZeroBeforeApply) {
    WeightMap wm(ModelArch::GEMMA4);
    const auto& s = wm.skip_stats();
    EXPECT_EQ(s.total, 0);
    EXPECT_EQ(s.audio, 0);
    EXPECT_EQ(s.vision, 0);
    EXPECT_EQ(s.mtp, 0);
    EXPECT_EQ(s.unrecognised, 0);
}

}  // namespace
}  // namespace imp
