// The MTP name rule, and the presence probe that reads it.
//
// Why this exists: the rule lives in three call sites (the divert in
// load_shard, the modelopt harvest, and the probe that tells the operator
// about a head it is not loading). They used to carry three literal copies of
// the prefixes. A checkpoint whose head one copy sees and another misses is
// exactly the "two places, same question, different answer" defect that
// produced #1384 and #1443, so the rule is one function and this pins it.
#include "model/mtp_head.h"
#include <filesystem>
#include <cstdlib>
#include <fstream>
#include <gtest/gtest.h>

namespace fs = std::filesystem;

TEST(MtpNameRule, AcceptsBothCheckpointSpellings) {
    // Sidecar and llm-compressor checkpoints write the bare prefix; modelopt
    // keeps the outer `model.` on. Both are the same head.
    EXPECT_TRUE(imp::name_is_mtp_tensor("mtp.layers.0.eh_proj.weight"));
    EXPECT_TRUE(imp::name_is_mtp_tensor("model.mtp.layers.0.eh_proj.weight"));
}

TEST(MtpNameRule, RejectsMainModelTensors) {
    // Must anchor at position 0: a main-model tensor that merely CONTAINS the
    // substring would otherwise be diverted out of the model and silently lost.
    EXPECT_FALSE(imp::name_is_mtp_tensor("model.layers.0.mtp.weight"));
    EXPECT_FALSE(imp::name_is_mtp_tensor("model.embed_tokens.weight"));
    EXPECT_FALSE(imp::name_is_mtp_tensor("lm_head.weight"));
    EXPECT_FALSE(imp::name_is_mtp_tensor(""));
}

TEST(MtpNameRule, HeadKeyAcceptsBothCheckpointShapes) {
    // dispatch_mtp() branches on the same two constants, so probe and dispatch
    // cannot drift apart. These cases pin the literal spellings a checkpoint
    // actually uses, which the shared constants alone do not.
    EXPECT_TRUE(imp::name_is_mtp_head_key("mtp.fc.weight"));
    EXPECT_TRUE(imp::name_is_mtp_head_key("model.mtp.fc.weight"));
    EXPECT_TRUE(imp::name_is_mtp_head_key("mtp.layers.0.eh_proj.weight"));
    EXPECT_TRUE(imp::name_is_mtp_head_key("model.mtp.layers.0.eh_proj.weight"));
}

TEST(MtpNameRule, HeadKeyRejectsAnMtpGroupWithoutAFusionProjection) {
    // `mtp.*` tensors alone are not a usable head: without the fusion
    // projection dispatch_mtp() warns and disables spec-decode, so the probe
    // must not advertise one.
    EXPECT_FALSE(imp::name_is_mtp_head_key("mtp.norm.weight"));
    EXPECT_FALSE(imp::name_is_mtp_head_key("mtp.layers.0.self_attn.q_proj.weight"));
    EXPECT_FALSE(imp::name_is_mtp_head_key("lm_head.weight"));
}

namespace {
// A checkpoint dir is one of three layouts, and a probe that knows one of them
// reports "no head" on the other two. Each layout gets its own case.
struct TempDir {
    fs::path p;
    explicit TempDir(const std::string& tag) {
        p = fs::temp_directory_path() / ("imp_mtp_probe_" + tag);
        fs::remove_all(p);
        fs::create_directories(p);
    }
    ~TempDir() { fs::remove_all(p); }
};

void write_single_file(const fs::path& file, const std::string& header_json) {
    std::ofstream f(file, std::ios::binary);
    uint64_t n = header_json.size();
    f.write(reinterpret_cast<const char*>(&n), sizeof(n));
    f.write(header_json.data(), static_cast<std::streamsize>(n));
    const char pad[8] = {};  // a body, so header_size validation sees a real file
    f.write(pad, sizeof(pad));
}
}  // namespace

TEST(MtpProbe, FindsSidecarLayout) {
    TempDir d("sidecar");
    write_single_file(d.p / "model_mtp.safetensors", R"({"mtp.fc.weight":{"dtype":"BF16"}})");
    EXPECT_TRUE(imp::probe_mtp_head(d.p.string()));
}

TEST(MtpProbe, RejectsASidecarThatIsNotAHead) {
    // A file by that name is not evidence. Before it was read, an empty or
    // unrelated model_mtp.safetensors advertised a head that enabling would
    // then fail to dispatch.
    TempDir d("sidecar_empty");
    std::ofstream(d.p / "model_mtp.safetensors") << "not empty, not a head";
    EXPECT_FALSE(imp::probe_mtp_head(d.p.string()));
}

TEST(MtpProbe, FindsShardedLayoutViaWeightMap) {
    TempDir d("sharded");
    std::ofstream(d.p / "model.safetensors.index.json")
        << R"({"weight_map":{"model.embed_tokens.weight":"a.safetensors",)"
           R"("mtp.fc.weight":"b.safetensors"}})";
    EXPECT_TRUE(imp::probe_mtp_head(d.p.string()));
}

TEST(MtpProbe, FindsSingleFileLayoutViaHeader) {
    TempDir d("single");
    write_single_file(d.p / "model.safetensors", R"({"model.mtp.layers.0.eh_proj.weight":{"dtype":"BF16"}})");
    EXPECT_TRUE(imp::probe_mtp_head(d.p.string()));
}

TEST(MtpProbe, StaysQuietWithoutAHead) {
    // The negative case is the one that matters for the hint: a checkpoint
    // without a head must not advertise one, in any of the three layouts.
    TempDir d("nohead");
    std::ofstream(d.p / "model.safetensors.index.json")
        << R"({"weight_map":{"model.embed_tokens.weight":"a.safetensors"}})";
    EXPECT_FALSE(imp::probe_mtp_head(d.p.string()));

    TempDir e("nohead_single");
    write_single_file(e.p / "model.safetensors", R"({"lm_head.weight":{"dtype":"BF16"}})");
    EXPECT_FALSE(imp::probe_mtp_head(e.p.string()));

    TempDir g("mtp_without_fusion");
    std::ofstream(g.p / "model.safetensors.index.json")
        << R"({"weight_map":{"mtp.norm.weight":"a.safetensors",)"
           R"("mtp.layers.0.self_attn.q_proj.weight":"a.safetensors"}})";
    EXPECT_FALSE(imp::probe_mtp_head(g.p.string()));

    TempDir f("empty");
    EXPECT_FALSE(imp::probe_mtp_head(f.p.string()));
    EXPECT_FALSE(imp::probe_mtp_head(""));
}

// The synthetic cases above pin the rule; this one pins it against a real
// checkpoint, because a hand-written weight_map is a statement about what I
// think checkpoints look like. Reads the index only, no GPU, no weights.
// Skipped where the model is absent, which includes CI.
TEST(MtpProbe, FindsTheHeadInARealCheckpoint) {
    const char* env = std::getenv("IMP_MTP_MODEL");
    std::string path = env ? env : "/models/Qwen3.8-27B-NVFP4";
    if (!fs::exists(path))
        GTEST_SKIP() << "checkpoint not present: " << path;
    EXPECT_TRUE(imp::probe_mtp_head(path)) << "known to carry mtp.fc.weight (15 tensors)";
}
