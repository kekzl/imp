// MTP name rule lives in three call sites (load_shard divert, modelopt harvest, presence
// probe); they used to carry three literal prefix copies. A checkpoint one copy sees and
// another misses is the #1384/#1443 defect class; now one function, pinned here.
#include "model/model.h"
#include "model/mtp_head.h"
#include "model/safetensors_loader.h"
#include <filesystem>
#include <cstdlib>
#include <fstream>
#include <string>
#include <utility>
#include <vector>
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

// speculative.mtp_head: a valid Qwen-layout mtp.* file (d_model 8, F16 zeros) attaches to a model of the
// same d_model and is refused for another d_model or for a file without a head.
static void write_qwen_mtp_file(const fs::path& file, int d) {
    const int hd = 4, heads = 2, ff = 16;
    const std::vector<std::pair<std::string, std::vector<int>>> t = {
        {"mtp.pre_fc_norm_embedding.weight", {d}},
        {"mtp.pre_fc_norm_hidden.weight", {d}},
        {"mtp.fc.weight", {d, 2 * d}},
        {"mtp.layers.0.input_layernorm.weight", {d}},
        {"mtp.layers.0.post_attention_layernorm.weight", {d}},
        {"mtp.layers.0.self_attn.q_proj.weight", {2 * heads * hd, d}},
        {"mtp.layers.0.self_attn.k_proj.weight", {hd, d}},
        {"mtp.layers.0.self_attn.v_proj.weight", {hd, d}},
        {"mtp.layers.0.self_attn.o_proj.weight", {d, heads * hd}},
        {"mtp.layers.0.self_attn.q_norm.weight", {hd}},
        {"mtp.layers.0.self_attn.k_norm.weight", {hd}},
        {"mtp.norm.weight", {d}},
        {"mtp.layers.0.mlp.gate_proj.weight", {ff, d}},
        {"mtp.layers.0.mlp.up_proj.weight", {ff, d}},
        {"mtp.layers.0.mlp.down_proj.weight", {d, ff}},
    };
    std::string h = "{";
    size_t off = 0;
    for (const auto& [name, shape] : t) {
        size_t n = 1;
        std::string s;
        for (int x : shape) {
            n *= static_cast<size_t>(x);
            s += (s.empty() ? "" : ",") + std::to_string(x);
        }
        h += (h.size() > 1 ? "," : "") + std::string("\"") + name + "\":{\"dtype\":\"F16\",\"shape\":[" + s +
             "],\"data_offsets\":[" + std::to_string(off) + "," + std::to_string(off + (2 * n)) + "]}";
        off += 2 * n;
    }
    h += "}";
    std::ofstream f(file, std::ios::binary);
    const uint64_t hn = h.size();
    f.write(reinterpret_cast<const char*>(&hn), sizeof(hn));
    f.write(h.data(), static_cast<std::streamsize>(hn));
    const std::vector<char> body(off, 0);
    f.write(body.data(), static_cast<std::streamsize>(body.size()));
}

TEST(MtpExternalHead, AttachesAMatchingQwenHeadOnly) {
    TempDir d("attach");
    write_qwen_mtp_file(d.p / "mtp.safetensors", 8);
    imp::Model match;
    match.config_.d_model = 8;
    EXPECT_TRUE(imp::attach_mtp_head_file(match, (d.p / "mtp.safetensors").string()));
    ASSERT_NE(match.mtp_, nullptr);
    EXPECT_TRUE(match.mtp_->loaded);
    EXPECT_EQ(match.mtp_->fc.shape[1], 16);
    imp::Model other;
    other.config_.d_model = 16;
    EXPECT_FALSE(imp::attach_mtp_head_file(other, (d.p / "mtp.safetensors").string()));
    EXPECT_EQ(other.mtp_, nullptr);
    write_single_file(d.p / "x.safetensors", R"({"model.embed_tokens.weight":{"dtype":"BF16"}})");
    EXPECT_FALSE(imp::attach_mtp_head_file(match, (d.p / "missing.safetensors").string()));
    EXPECT_FALSE(imp::attach_mtp_head_file(other, (d.p / "x.safetensors").string()));
}
