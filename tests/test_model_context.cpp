// /v1/models context ceiling for models not loaded (tools/imp-server/model_context.h).
#include "model_context.h"

#include "runtime/engine_config.h"

#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>

namespace fs = std::filesystem;
using imp_server::advertised_context_tokens;
using imp_server::declared_context_tokens;

namespace {

struct TempModelDir {
    fs::path dir;
    TempModelDir() {
        dir = fs::temp_directory_path() /
              ("imp_model_context_" +
               std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        fs::create_directories(dir);
    }
    ~TempModelDir() {
        std::error_code ec;
        fs::remove_all(dir, ec);
    }
    void write_config(const std::string& text) const { std::ofstream(dir / "config.json") << text; }
};

}  // namespace

TEST(ModelContext, ConfiguredWinsElseDeclaredCappedAtTheAutoCeiling) {
    EXPECT_EQ(advertised_context_tokens(262144, 0), imp::kAutoMaxSeqLenCap);
    EXPECT_EQ(advertised_context_tokens(32768, 0), 32768);
    EXPECT_EQ(advertised_context_tokens(262144, 65536), 65536);
    EXPECT_EQ(advertised_context_tokens(0, 0), 0) << "unknown stays unknown, the field is omitted";
}

TEST(ModelContext, ReadsTextConfigOfAMultimodalWrapper) {
    TempModelDir m;
    m.write_config(
        R"({"max_position_embeddings": 4096, "text_config": {"max_position_embeddings": 262144}})");
    EXPECT_EQ(declared_context_tokens(m.dir.string()), 262144);
}

TEST(ModelContext, ReadsTheRootOfAPlainConfig) {
    TempModelDir m;
    m.write_config(R"({"max_position_embeddings": 40960})");
    EXPECT_EQ(declared_context_tokens(m.dir.string()), 40960);
}

TEST(ModelContext, UnknownWithoutAConfigOrTheKey) {
    TempModelDir m;
    EXPECT_EQ(declared_context_tokens(m.dir.string()), 0) << "no config.json (a GGUF file path too)";
    m.write_config(R"({"hidden_size": 4096})");
    EXPECT_EQ(declared_context_tokens(m.dir.string()), 0);
    m.write_config("not json");
    fs::last_write_time(m.dir / "config.json", fs::file_time_type::clock::now() + std::chrono::seconds(5));
    EXPECT_EQ(declared_context_tokens(m.dir.string()), 0);
}

TEST(ModelContext, RereadsAConfigThatChanged) {
    TempModelDir m;
    m.write_config(R"({"max_position_embeddings": 8192})");
    ASSERT_EQ(declared_context_tokens(m.dir.string()), 8192);
    m.write_config(R"({"max_position_embeddings": 16384})");
    fs::last_write_time(m.dir / "config.json", fs::file_time_type::clock::now() + std::chrono::seconds(10));
    EXPECT_EQ(declared_context_tokens(m.dir.string()), 16384);
}
