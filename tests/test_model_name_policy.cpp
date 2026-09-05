// The request `model` field is a name, never a path (AUDIT_arch_2026 F2-1).
//
// Until 2026-09-05 find_model_path() fell through to resolve_model_auto() for
// any name containing '/', whose first step is fs::exists: a request body
// naming any readable .gguf (or a directory holding model.safetensors) loaded
// it and evicted the resident model. The policy that stops it is two pure
// functions in tools/imp-server/model_name_policy.h; this pins them in the CPU
// lane. Mutation: drop the leading-'/' rule and AbsolutePathIsRejected fails;
// drop the trailing-slash normalisation and TrailingSlashBase fails.

#include "model_name_policy.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>

namespace {

using imp_server::classify_model_name;
using imp_server::ModelNameKind;
using imp_server::path_within;

TEST(ModelNamePolicy, BasenameIsLookedUpInModelsDir) {
    EXPECT_EQ(classify_model_name("Qwen3-8B-Q8_0.gguf"), ModelNameKind::Basename);
    EXPECT_EQ(classify_model_name("Qwen3-8B-NVFP4-cortecs"), ModelNameKind::Basename);
    EXPECT_EQ(classify_model_name("gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"), ModelNameKind::Basename);
}

TEST(ModelNamePolicy, HfRepoIdShape) {
    EXPECT_EQ(classify_model_name("Qwen/Qwen3-8B"), ModelNameKind::HfRepoId);
    EXPECT_EQ(classify_model_name("kekzle/Qwen3.8-27B-NVFP4-vllm"), ModelNameKind::HfRepoId);
}

TEST(ModelNamePolicy, AbsolutePathIsRejected) {
    EXPECT_EQ(classify_model_name("/tmp/x.gguf"), ModelNameKind::Rejected);
    EXPECT_EQ(classify_model_name("/etc/passwd"), ModelNameKind::Rejected);
    EXPECT_EQ(classify_model_name("/models/Qwen3-8B-Q8_0.gguf"), ModelNameKind::Rejected);
}

TEST(ModelNamePolicy, RelativeAndTraversalPathsAreRejected) {
    EXPECT_EQ(classify_model_name("./x.gguf"), ModelNameKind::Rejected);
    EXPECT_EQ(classify_model_name("../x.gguf"), ModelNameKind::Rejected);
    EXPECT_EQ(classify_model_name("org/../../x"), ModelNameKind::Rejected);
    EXPECT_EQ(classify_model_name("~/models/x.gguf"), ModelNameKind::Rejected);
    EXPECT_EQ(classify_model_name("a/b/c"), ModelNameKind::Rejected);
    EXPECT_EQ(classify_model_name("org\\repo"), ModelNameKind::Rejected);
}

TEST(ModelNamePolicy, RepoIdWithModelFileExtensionIsAPathNotAnId) {
    // "org/x.gguf" is what a relative path to a file looks like; hf_hub.cpp
    // itself refuses to treat it as a repo id.
    EXPECT_EQ(classify_model_name("org/x.gguf"), ModelNameKind::Rejected);
    EXPECT_EQ(classify_model_name("org/model.safetensors"), ModelNameKind::Rejected);
    EXPECT_EQ(classify_model_name("org/"), ModelNameKind::Rejected);
    EXPECT_EQ(classify_model_name(""), ModelNameKind::Rejected);
}

class PathWithin : public ::testing::Test {
protected:
    void SetUp() override {
        base_ = std::filesystem::temp_directory_path() / "imp_policy_test";
        std::filesystem::create_directories(base_ / "sub");
        std::ofstream(base_ / "sub" / "m.gguf") << "x";
        std::filesystem::create_directories(base_.string() + "2");
    }
    void TearDown() override {
        std::error_code ec;
        std::filesystem::remove_all(base_, ec);
        std::filesystem::remove_all(base_.string() + "2", ec);
    }
    std::filesystem::path base_;
};

TEST_F(PathWithin, InsideAndOutside) {
    EXPECT_TRUE(path_within(base_, base_ / "sub" / "m.gguf"));
    EXPECT_TRUE(path_within(base_, base_));
    EXPECT_FALSE(path_within(base_, base_.parent_path()));
    EXPECT_FALSE(path_within(base_, "/etc/passwd"));
}

TEST_F(PathWithin, SiblingWithCommonPrefixIsOutside) {
    // "/x/imp_policy_test" must not contain "/x/imp_policy_test2/..."
    EXPECT_FALSE(path_within(base_, std::filesystem::path(base_.string() + "2") / "m.gguf"));
}

TEST_F(PathWithin, TraversalIsResolvedBeforeTheCompare) {
    EXPECT_FALSE(path_within(base_, base_ / "sub" / ".." / ".." / "m.gguf"));
    EXPECT_TRUE(path_within(base_, base_ / "sub" / ".." / "sub" / "m.gguf"));
}

TEST_F(PathWithin, TrailingSlashBase) {
    EXPECT_TRUE(path_within(base_.string() + "/", base_ / "sub" / "m.gguf"));
}

}  // namespace
