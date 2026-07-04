#include <gtest/gtest.h>
#include "model/gguf_loader.h"
#include "model/model_arch.h"

#include <stdexcept>
#include <string>

#include <cstdio>
#include <cstring>
#include <vector>
#include <unistd.h>

namespace imp {
namespace {

// Helper: write raw bytes to a temp file and return its path.
static std::string write_temp_gguf(const std::vector<uint8_t>& data) {
    char path[] = "/tmp/imp_test_XXXXXX.gguf";
    int fd = mkstemps(path, 5);
    if (fd < 0)
        return "";
    ssize_t written = write(fd, data.data(), data.size());
    (void)written;
    close(fd);
    return std::string(path);
}

// Helper: build a minimal GGUF v3 header (24 bytes).
static std::vector<uint8_t> make_gguf_header(uint32_t magic, uint32_t version, uint64_t n_tensors,
                                             uint64_t n_kv) {
    std::vector<uint8_t> buf(24);
    memcpy(buf.data() + 0, &magic, 4);
    memcpy(buf.data() + 4, &version, 4);
    memcpy(buf.data() + 8, &n_tensors, 8);
    memcpy(buf.data() + 16, &n_kv, 8);
    return buf;
}

TEST(GgufLoaderTest, LoadNonexistentFile) {
    auto model = load_gguf("/nonexistent/path/model.gguf");
    EXPECT_EQ(model, nullptr);
}

TEST(GgufLoaderTest, GGMLTypeHelpers) {
    // Block sizes
    EXPECT_EQ(gguf_blck_size(GgufWireType::F32), 1);
    EXPECT_EQ(gguf_blck_size(GgufWireType::F16), 1);
    EXPECT_EQ(gguf_blck_size(GgufWireType::Q4_0), 32);
    EXPECT_EQ(gguf_blck_size(GgufWireType::Q8_0), 32);
    EXPECT_EQ(gguf_blck_size(GgufWireType::Q4_K), 256);

    // Type sizes
    EXPECT_EQ(gguf_type_size(GgufWireType::F32), 4u);
    EXPECT_EQ(gguf_type_size(GgufWireType::F16), 2u);
    EXPECT_EQ(gguf_type_size(GgufWireType::BF16), 2u);
    EXPECT_EQ(gguf_type_size(GgufWireType::Q4_0), 18u);
    EXPECT_EQ(gguf_type_size(GgufWireType::Q8_0), 34u);

    // Row size
    EXPECT_EQ(gguf_row_size(GgufWireType::F32, 4096), 4096u * 4);
    EXPECT_EQ(gguf_row_size(GgufWireType::F16, 4096), 4096u * 2);
    // Q4_0: 4096 elements / 32 elements_per_block * 18 bytes_per_block
    EXPECT_EQ(gguf_row_size(GgufWireType::Q4_0, 4096), (4096u / 32) * 18);

    // Type names
    EXPECT_STREQ(gguf_type_name(GgufWireType::F32), "F32");
    EXPECT_STREQ(gguf_type_name(GgufWireType::Q4_K), "Q4_K");

    // QType conversion: wire-stable values map exactly to QType.
    EXPECT_EQ(gguf_type_to_qtype(GgufWireType::F32), QType::F32);
    EXPECT_EQ(gguf_type_to_qtype(GgufWireType::F16), QType::F16);
    EXPECT_EQ(gguf_type_to_qtype(GgufWireType::BF16), QType::BF16);
    EXPECT_EQ(gguf_type_to_qtype(GgufWireType::Q4_0), QType::Q4_0);
    EXPECT_EQ(gguf_type_to_qtype(GgufWireType::Q4_K), QType::Q4_K);
    EXPECT_EQ(gguf_type_to_qtype(GgufWireType::Q8_0), QType::Q8_0);
    EXPECT_EQ(gguf_type_to_qtype(GgufWireType::Q6_K), QType::Q6_K);
    EXPECT_EQ(gguf_type_to_qtype(GgufWireType::MXFP4), QType::MXFP4);
}

TEST(GgufLoaderTest, InvalidMagic) {
    // Create a small file with wrong magic to test error handling
    auto model = load_gguf("/dev/null");
    EXPECT_EQ(model, nullptr);
}

// ---- New tests for GGUF parser robustness ----

TEST(GgufLoaderTest, TruncatedFile) {
    // Valid magic but truncated after 8 bytes (missing n_tensors and n_kv).
    // The parser should return nullptr without crashing.
    std::vector<uint8_t> data(8);
    uint32_t magic = GGUF_MAGIC;
    uint32_t version = 3;
    memcpy(data.data(), &magic, 4);
    memcpy(data.data() + 4, &version, 4);

    std::string path = write_temp_gguf(data);
    ASSERT_FALSE(path.empty());

    auto model = load_gguf(path);
    EXPECT_EQ(model, nullptr);

    unlink(path.c_str());
}

TEST(GgufLoaderTest, UnsupportedVersion) {
    // Valid magic but version=99 (only v2 and v3 are supported).
    auto data = make_gguf_header(GGUF_MAGIC, 99, 0, 0);

    std::string path = write_temp_gguf(data);
    ASSERT_FALSE(path.empty());

    auto model = load_gguf(path);
    EXPECT_EQ(model, nullptr);

    unlink(path.c_str());
}

TEST(GgufLoaderTest, ZeroTensors) {
    // Valid v3 header with n_tensors=0, n_kv=0. The parser should succeed
    // and return a model (or at minimum not crash). An empty model with no
    // tensors and no metadata is structurally valid GGUF.
    auto data = make_gguf_header(GGUF_MAGIC, 3, 0, 0);

    std::string path = write_temp_gguf(data);
    ASSERT_FALSE(path.empty());

    // load_gguf may return nullptr if it requires metadata (e.g. architecture
    // key) to build a Model, but it must not crash or abort.
    auto model = load_gguf(path);
    // Either a valid model or a graceful nullptr is acceptable.
    // The key assertion is that we reached this line without crashing.
    (void)model;

    unlink(path.c_str());
}

TEST(GgufLoaderTest, LargeMetadataCount) {
    // Valid header but n_kv=100000 with only 24 bytes total.
    // The parser should detect truncation when trying to read
    // the first KV pair and return nullptr (not OOM or crash).
    uint64_t n_kv = 100000;
    auto data = make_gguf_header(GGUF_MAGIC, 3, 0, n_kv);

    std::string path = write_temp_gguf(data);
    ASSERT_FALSE(path.empty());

    auto model = load_gguf(path);
    EXPECT_EQ(model, nullptr);

    unlink(path.c_str());
}


// ---------------------------------------------------------------------------
// #818: encoder-only architectures must be rejected at load, not fall through
// to the generic-decoder path (which "loads", reports healthy, and IMAs on the
// first /v1/embeddings request, poisoning the CUDA context).
// ---------------------------------------------------------------------------

TEST(GgufLoaderTest, EncoderOnlyArchDetection) {
    // llama.cpp GGUF arch ids for the BERT-family encoders.
    EXPECT_TRUE(is_encoder_only_arch("bert"));
    EXPECT_TRUE(is_encoder_only_arch("nomic-bert"));
    EXPECT_TRUE(is_encoder_only_arch("nomic-bert-moe"));
    EXPECT_TRUE(is_encoder_only_arch("jina-bert-v2"));
    EXPECT_TRUE(is_encoder_only_arch("jina-bert-v3"));
    EXPECT_TRUE(is_encoder_only_arch("modern-bert"));
    EXPECT_TRUE(is_encoder_only_arch("roberta"));
    EXPECT_TRUE(is_encoder_only_arch("t5encoder"));
    EXPECT_TRUE(is_encoder_only_arch("e5"));
    EXPECT_TRUE(is_encoder_only_arch("bge"));
    // HuggingFace config.json `architectures` class names (CamelCase) — the
    // SafeTensors/HF path feeds these, and a case-sensitive find("bert") used
    // to miss them, letting NomicBertModel fall through to the decoder (#818).
    EXPECT_TRUE(is_encoder_only_arch("BertModel"));
    EXPECT_TRUE(is_encoder_only_arch("NomicBertModel"));
    EXPECT_TRUE(is_encoder_only_arch("BertForMaskedLM"));
    EXPECT_TRUE(is_encoder_only_arch("RobertaModel"));
    EXPECT_TRUE(is_encoder_only_arch("XLMRobertaModel"));
    EXPECT_TRUE(is_encoder_only_arch("MPNetModel"));
    EXPECT_TRUE(is_encoder_only_arch("nomic_bert"));  // model_type form
    // Decoder archs must never be rejected (neither GGUF id nor HF class name).
    EXPECT_FALSE(is_encoder_only_arch("llama"));
    EXPECT_FALSE(is_encoder_only_arch("LlamaForCausalLM"));
    EXPECT_FALSE(is_encoder_only_arch("Qwen3ForCausalLM"));
    EXPECT_FALSE(is_encoder_only_arch("Gemma4ForCausalLM"));
    EXPECT_FALSE(is_encoder_only_arch("qwen3"));
    EXPECT_FALSE(is_encoder_only_arch("qwen3moe"));
    EXPECT_FALSE(is_encoder_only_arch("gemma4"));
    EXPECT_FALSE(is_encoder_only_arch("gpt_oss"));
    EXPECT_FALSE(is_encoder_only_arch("nemotron_h_moe"));
    EXPECT_FALSE(is_encoder_only_arch("generic-unknown"));
}

TEST(GgufLoaderTest, EncoderArchGgufRejectedAtLoad) {
    // Minimal valid GGUF v3 with general.architecture = "bert" and no
    // tensors. Encoder archs WITHOUT a dedicated path (classic BERT/bge/e5:
    // learned positions, CLS pooling) must still throw a clear error instead
    // of mapping to the generic decoder. nomic-bert is admitted since #836
    // (dedicated encoder forward — e2e covered by EncoderEmbedTest).
    auto append_str = [](std::vector<uint8_t>& buf, const std::string& s) {
        uint64_t len = s.size();
        const uint8_t* lp = reinterpret_cast<const uint8_t*>(&len);
        buf.insert(buf.end(), lp, lp + 8);
        buf.insert(buf.end(), s.begin(), s.end());
    };
    auto data = make_gguf_header(GGUF_MAGIC, 3, /*n_tensors=*/0, /*n_kv=*/1);
    append_str(data, "general.architecture");
    uint32_t t_string = 8;  // GGUFValueType::STRING
    const uint8_t* tp = reinterpret_cast<const uint8_t*>(&t_string);
    data.insert(data.end(), tp, tp + 4);
    append_str(data, "bert");

    std::string path = write_temp_gguf(data);
    ASSERT_FALSE(path.empty());

    try {
        auto model = load_gguf(path);
        FAIL() << "expected load_gguf to throw for encoder-only arch, got "
               << (model ? "a model" : "nullptr");
    } catch (const std::exception& e) {
        EXPECT_NE(std::string(e.what()).find("encoder-only"), std::string::npos) << e.what();
        EXPECT_NE(std::string(e.what()).find("bert"), std::string::npos) << e.what();
    }

    unlink(path.c_str());
}

}  // namespace
}  // namespace imp
