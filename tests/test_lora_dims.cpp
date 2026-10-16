// AUDIT_ARCH_2026 F1-6: an adapter's declared widths are held against the
// base model at load. The rule table is a pure function so it runs in the
// unit lane; the loader-level refusals live in test_lora.cpp (model-backed).
#include <gtest/gtest.h>
#include "lora/lora_adapter.h"

#include <unistd.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>

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

// H-012: "layers.x" is a malformed index, not layer 0 (atoi), and "3x" is not layer 3.
TEST(LoraDimsTest, MalformedLayerIndexIsNotMapped) {
    for (const char* key : {"base_model.model.model.layers.x.self_attn.q_proj.lora_A.weight",
                            "base_model.model.model.layers.3x.self_attn.q_proj.lora_A.weight",
                            "base_model.model.model.layers.4294967296.self_attn.q_proj.lora_A.weight"}) {
        int layer = 99;
        LoraProj proj{};
        bool is_A = false;
        ASSERT_TRUE(lora_parse_key(key, &layer, &proj, &is_A)) << key;
        EXPECT_EQ(layer, -1) << key;
    }
    int layer = -1;
    LoraProj proj{};
    bool is_A = false;
    ASSERT_TRUE(lora_parse_key("base_model.model.model.layers.3.self_attn.q_proj.lora_A.weight", &layer,
                               &proj, &is_A));
    EXPECT_EQ(layer, 3);
    EXPECT_EQ(proj, LoraProj::Q);
    EXPECT_TRUE(is_A);
}

// The load refuses the adapter and names the tensor, before any device upload.
TEST(LoraDimsTest, MalformedLayerIndexRefusesTheLoad) {
    const std::string key = "base_model.model.model.layers.x.self_attn.q_proj.lora_A.weight";
    const std::string hdr = "{\"" + key + "\":{\"dtype\":\"F32\",\"shape\":[2,4],\"data_offsets\":[0,32]}}";
    const auto path = std::filesystem::temp_directory_path() /
                      ("imp_lora_malformed_" + std::to_string(::getpid()) + ".safetensors");
    {
        std::ofstream f(path, std::ios::binary);
        const uint64_t n = hdr.size();
        f.write(reinterpret_cast<const char*>(&n), sizeof(n));
        f.write(hdr.data(), static_cast<std::streamsize>(hdr.size()));
        const std::string payload(32, '\0');
        f.write(payload.data(), static_cast<std::streamsize>(payload.size()));
    }
    LoraAdapter a;
    testing::internal::CaptureStderr();
    const bool ok = a.load(path.string(), 4);
    const std::string log = testing::internal::GetCapturedStderr();
    std::filesystem::remove(path);
    EXPECT_FALSE(ok);
    EXPECT_NE(log.find("malformed layer index"), std::string::npos) << log;
    EXPECT_NE(log.find(key), std::string::npos) << log;
}

}  // namespace
}  // namespace imp
