// Real huggingface.co fetch (#2200), opt-in: IMP_TEST_NETWORK=1 build-dev/test-hf-network.
// Not registered with ctest; CI has no outbound-network lane. Downloads 1.2 MB.

#include <gtest/gtest.h>

#include "model/hf_fetch.h"

#include <cstdlib>
#include <filesystem>

#include <unistd.h>

namespace {

namespace fs = std::filesystem;
using namespace imp::hf;

bool network_enabled() {
    const char* v = std::getenv("IMP_TEST_NETWORK");
    return v && std::string(v) == "1";
}

class HfFetchNetwork : public ::testing::Test {
protected:
    void SetUp() override {
        if (!network_enabled())
            GTEST_SKIP() << "set IMP_TEST_NETWORK=1 to run against huggingface.co";
        cache_ = fs::temp_directory_path() / ("imp_hf_net_" + std::to_string(::getpid()));
        fs::remove_all(cache_);
    }
    void TearDown() override {
        if (!cache_.empty())
            fs::remove_all(cache_);
    }
    FetchOptions opts() const {
        FetchOptions o;
        o.cache_dir = cache_;
        return o;
    }
    std::string cache_;
};

// ggml-org/tiny-llamas has 4 GGUFs; stories260K.gguf is 1185376 bytes, LFS sha256 below.
TEST_F(HfFetchNetwork, TinyGgufFetchVerifiesShaThenHitsCache) {
    FetchResult r = fetch(HfUri{"ggml-org/tiny-llamas", "stories260K.gguf"}, opts());
    ASSERT_TRUE(r.ok) << r.error;
    EXPECT_FALSE(r.cache_hit);
    EXPECT_EQ(r.bytes_downloaded, 1185376u);
    EXPECT_EQ(fs::file_size(r.path), 1185376u);
    EXPECT_EQ(sha256_file(r.path), "047bf46455a544931cff6fef14d7910154c56afbc23ab1c5e56a72e69912c04b");

    FetchResult again = fetch(HfUri{"ggml-org/tiny-llamas", "stories260K.gguf"}, opts());
    ASSERT_TRUE(again.ok) << again.error;
    EXPECT_TRUE(again.cache_hit);
    EXPECT_EQ(again.bytes_downloaded, 0u);
    EXPECT_EQ(again.path, r.path);
}

TEST_F(HfFetchNetwork, SeveralGgufsWithoutSelectorListThem) {
    FetchResult r = fetch(HfUri{"ggml-org/tiny-llamas", ""}, opts());
    ASSERT_FALSE(r.ok);
    EXPECT_NE(r.error.find("stories260K.gguf"), std::string::npos) << r.error;
}

// meta-llama/Llama-3.2-1B is gated ("manual"): without HF_TOKEN nothing is written.
TEST_F(HfFetchNetwork, GatedRepoWithoutTokenFailsClean) {
    FetchResult r = fetch(HfUri{"meta-llama/Llama-3.2-1B", ""}, opts());
    ASSERT_FALSE(r.ok);
    EXPECT_NE(r.error.find("HF_TOKEN"), std::string::npos) << r.error;
    EXPECT_FALSE(fs::exists(repo_cache_dir(cache_, "meta-llama/Llama-3.2-1B") + "/snapshots"));
}

}  // namespace
